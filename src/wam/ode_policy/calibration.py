"""Parent-state encoding and frozen Phoenix action-slot calibration."""
import json
from pathlib import Path

import numpy as np
import torch

from .kitchen import ACTIVE, ARMS, GRIPPERS, JointCoordinates, observation


def load_source(path, group, identity, *, include_action=True):
    profile = json.loads(Path(path).read_text())
    allowed = [profile["source_checkpoint_sha256"], *profile.get("release_sha256", [])]
    if profile["slots"] != ACTIVE or identity["phoenix_weights"] not in allowed:
        raise ValueError("ARX scale profile belongs to a different parent or slot order")
    if group not in profile["groups"]:
        raise ValueError(f"Select an explicit source group from {list(profile['groups'])}")
    row = profile["groups"][group]
    result = {"group": group, "norm_sha256": profile["norm_sha256"]}
    for kind in (("state", "action") if include_action else ("state",)):
        bounds = torch.tensor(row[kind], dtype=torch.float32)
        if bounds.shape != (2, 14) or not bounds.isfinite().all() or (bounds[1] < bounds[0]).any():
            raise ValueError(f"Invalid {kind} bounds")
        result[kind] = bounds
    return result


def indices(data, per_episode, episodes_per_task=None):
    selected = None
    if episodes_per_task:
        tasks = {}
        for i, episode in enumerate(data.episodes):
            tasks.setdefault(episode[2], []).append(i)
        selected = set()
        for episodes in tasks.values():
            count = min(episodes_per_task, len(episodes))
            selected.update(episodes[(2*j+1)*len(episodes)//(2*count)] for j in range(count))
    for i, (_, name, task, episode, _, _) in enumerate(data.episodes):
        if selected is not None and i not in selected:
            continue
        start = data.ends[i-1] if i else 0
        end = data.ends[i]
        count = min(per_episode, end-start)
        # Centers of equal time intervals avoid forcing every anchor to an episode boundary.
        for j in range(count):
            yield name, task, episode, start + ((2*j+1)*(end-start)-1)//(2*count)


def fit_affine(pairs):
    x = torch.cat([raw.reshape(-1, 14) for raw, _ in pairs]).double()
    y = torch.cat([target.reshape(-1, 14) for _, target in pairs]).double()
    if not (x.isfinite().all() and y.isfinite().all()):
        raise ValueError("Nonfinite calibration window")
    xc, yc = x-x.mean(0), y-y.mean(0)
    variance = xc.square().mean(0)
    slope = torch.where(variance > 1e-12, (xc*yc).mean(0)/variance.clamp_min(1e-12), 0.)
    return slope.float(), (y.mean(0)-slope*x.mean(0)).float()


class PriorCoordinates:
    def __init__(self, kitchen, source=None, affine=None):
        self.kitchen = kitchen
        self.source = source
        device = kitchen.scale.device
        self.state = JointCoordinates(source["state"], device) if source else kitchen
        self.action = JointCoordinates(source["action"], device) if source and "action" in source else None
        self.affine = tuple(v.to(device) for v in affine) if affine is not None else None

    def slots(self, raw):
        if self.affine is not None:
            scale, offset = self.affine
            return raw * scale + offset
        if self.action is not None:
            return self.kitchen.normalized(self.action.physical(raw))
        return raw


def trajectory_metrics(pred, target, current, bounds):
    """Physical errors; compare net arm displacement at each horizon to the same anchor."""
    err = (pred-target).square()
    span = bounds[1]-bounds[0]
    moved = (target-current).abs()[:, ARMS] > (0.01*span[ARMS]).clamp_min(1e-6)
    good = ((pred-current)[:, ARMS] * (target-current)[:, ARMS]) > 0
    # Compare the first crossing in each direction, separately for each gripper.
    middle = bounds.mean(0)[GRIPPERS]
    def crossings(path):
        side = torch.cat((current[None], path))[:, GRIPPERS] > middle
        return side[1:].int()-side[:-1].int()
    expected, actual = crossings(target), crossings(pred)
    timing, missed, extra = [], 0, 0
    for side in range(2):
        for direction in (-1, 1):
            gt = torch.where(expected[:, side] == direction)[0]
            pr = torch.where(actual[:, side] == direction)[0]
            if len(gt) and len(pr):
                timing.append(int(abs(gt[0]-pr[0])))
            elif len(gt):
                missed += 1
            elif len(pr):
                extra += 1
    return dict(arm_rmse_by_tick=err[:, ARMS].mean(1).sqrt().tolist(),
                gripper_rmse=err[:, GRIPPERS].mean().sqrt().item(),
                arm_direction_matches=int((good & moved).sum()), arm_moving_targets=int(moved.sum()),
                gripper_timing_abs_ticks=timing, missed_gripper_crossings=missed, extra_gripper_crossings=extra)


@torch.no_grad()
def audit_prior(args, cfg, model, text, identity):
    from .kitchen import KitchenWindows
    from .phoenix import read_prior
    from types import SimpleNamespace

    horizon = cfg["policy"]["horizon"]
    train = KitchenWindows(args.data_root, cfg["sources"], "train", horizon)
    val = KitchenWindows(args.data_root, cfg["sources"], "val", horizon)
    bounds = train.fit_train_bounds()
    kitchen = JointCoordinates(bounds, model.device)
    source = load_source(args.source_profile, args.source_group, identity) if args.source_profile else None
    coords = PriorCoordinates(kitchen, source)
    request = SimpleNamespace(family="ab", horizon=horizon)
    def predict(sample, idx):
        task, rgb, states, _, targets = sample
        obs = observation(rgb, coords.state.normalized(states[0])[None], text[task])
        raw = read_prior(model, request, obs, active=ACTIVE,
                         denoise_steps=cfg["policy"]["denoise_steps"], seed=cfg["train"]["seed"]+idx)
        return raw[0].cpu(), targets
    anchors = list(indices(train, args.fit_windows, getattr(args, "fit_episodes_per_task", None)))
    pairs = [predict(train[idx], idx) for _, _, _, idx in anchors]
    affine = fit_affine([(raw, kitchen.normalized(y).cpu()) for raw,y in pairs])
    report = []
    for name, task, episode, idx in indices(val, args.windows):
        sample = val[idx]
        raw, y = predict(sample, idx)
        raw = raw.to(model.device)
        q = sample[2][0]
        paths = {"hold": q.expand_as(y), "direct_slots": kitchen.physical(raw).cpu(),
                 "fitted": kitchen.physical(raw*affine[0].to(model.device)+affine[1].to(model.device)).cpu()}
        if coords.action is not None:
            paths["source_physical"] = coords.action.physical(raw).cpu()
        report.append(dict(source=name, task=task, episode=episode, window=idx,
                           metrics={k:trajectory_metrics(v,y,q,bounds) for k,v in paths.items()}))
    payload = dict(identity=identity, sources=cfg["sources"], instructions=cfg["instructions"],
                   policy=cfg["policy"], bounds=bounds, source=source, affine=affine,
                   fit_windows=len(pairs), fit_anchors=anchors, report=report, seed=cfg["train"]["seed"])
    from .runner import save_new
    save_new(payload, args.output)
    for task in sorted({r["task"] for r in report}):
        rows = [r["metrics"] for r in report if r["task"] == task]
        for mode in rows[0]:
            metrics = [r[mode] for r in rows]
            timing = [v for m in metrics for v in m["gripper_timing_abs_ticks"]]
            n = sum(m["arm_moving_targets"] for m in metrics)
            print(json.dumps(dict(task=task, mode=mode, windows=len(rows),
                arm_rmse=float(np.sqrt(np.mean([np.square(m["arm_rmse_by_tick"]) for m in metrics]))),
                gripper_rmse=float(np.sqrt(np.mean([m["gripper_rmse"]**2 for m in metrics]))),
                direction_accuracy=sum(m["arm_direction_matches"] for m in metrics)/n if n else None,
                matched_crossings=len(timing), missed=sum(m["missed_gripper_crossings"] for m in metrics),
                extra=sum(m["extra_gripper_crossings"] for m in metrics),
                gripper_timing_mae_ticks=float(np.mean(timing)) if timing else None)))
