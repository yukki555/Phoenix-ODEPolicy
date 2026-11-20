"""Train or replay the shared A/B, C1 and C2 heads on the four kitchen tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
import yaml

from .head import ActionBlock, ODEHead
from .kitchen import ACTIVE, ARMS, GRIPPERS, JointCoordinates, KitchenWindows, observation


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_assets(args, cfg):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from wam.training.checkpoint import _load_joint96_checkpoint
    from wam.utils.config_resolvers import register_default_resolvers
    from wam.datasets.robotwin_joint96_io import official_text_padding

    register_default_resolvers()
    config = OmegaConf.load(args.model_config)
    model_cfg = config.get("model", config)
    if args.text_cache is None:
        model_cfg.load_text_encoder = True
    model = instantiate(model_cfg, device=args.device, model_dtype=torch.bfloat16)
    payload = torch.load(args.phoenix_weights, mmap=True, weights_only=True, map_location="cpu")
    _load_joint96_checkpoint(model, payload)
    model.eval().requires_grad_(False)
    if not model.joint96_three_view or model.action_expert.action_dim != 80:
        raise ValueError("A canonical80 Joint96 parent is required")
    if args.text_cache is None:
        from wam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        text = {}
        for task, instruction in cfg["instructions"].items():
            context, mask = model.encode_prompt(DEFAULT_PROMPT.format(task=instruction))
            text[task] = context[0], mask[0].bool()
    else:
        embeddings = torch.load(args.text_cache, map_location="cpu", weights_only=True)["embeddings"]
        text = {task: official_text_padding(row["context"], row["context_mask"])
                for task, row in embeddings.items()}
    if not set(cfg["instructions"]).issubset(text):
        raise ValueError("Missing task text embeddings")
    return model, text


def make_head(cfg, family, model):
    p = cfg["policy"]
    if p["solver"] != "euler":
        raise ValueError("The four-way comparison uses fixed-step Euler")
    return ODEHead(family, model.action_expert.hidden_dim, horizon=p["horizon"],
                   hz=p["hz"], hidden_dim=p["hidden_dim"], context_dim=p["context_dim"]).to(model.device)


def sample_inputs(sample, coordinates, texts):
    task, rgb, states, reference, targets = sample
    states = coordinates.normalized(states)[None]
    return (observation(rgb, states[:, 0], texts[task]), states,
            coordinates.normalized(reference)[None], coordinates.normalized(targets)[None])


def train(args, cfg, model, text, identity):
    from .phoenix import read_prior

    cfg_train = cfg["train"]
    head = make_head(cfg, args.family, model)
    data = KitchenWindows(args.data_root, cfg["sources"], "train", head.horizon)
    bounds = data.fit_train_bounds()
    coords = JointCoordinates(bounds, model.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg_train["lr"])
    seed = cfg_train["seed"]
    sampler, boundaries = random.Random(seed), random.Random(seed + 1)
    batch = args.batch_size or cfg_train["batch_size"]
    steps = args.steps or cfg_train["steps"]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(batch):
            index = sampler.randrange(len(data))
            obs, states, reference, targets = sample_inputs(data[index], coords, text)
            prior = read_prior(model, head, obs, active=ACTIVE,
                               denoise_steps=cfg["policy"]["denoise_steps"], seed=seed + index)
            k = 0
            if head.family == "ab" and step >= cfg_train["suffix_warmup_steps"] and boundaries.random() < cfg_train["suffix_probability"]:
                k = boundaries.randrange(1, head.horizon)
            loss = head.path_loss(prior, states, reference, targets, boundary=k,
                                  model_boundary=boundaries.random() < cfg_train["model_boundary_probability"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ODE path loss")
            (loss / batch).backward()
            total += loss.item() / batch
        optimizer.step()
        print(json.dumps({"step": step+1, "loss": total}), flush=True)
    temp = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(dict(family=head.family, config=cfg, identity=identity, bounds=bounds,
                        head=head.state_dict(), steps=steps, batch_size=batch), temp)
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def replay(mode, head, prior, reference, states, tick_times=None):
    if mode == "phoenix":
        return prior
    block = ActionBlock(head, prior, reference)
    if mode not in {"b", "b_stale"}:
        return block.targets(states[:, 0], reference, start=0, count=head.horizon)
    outputs = []
    for k in range(head.horizon):
        if tick_times is not None:
            sync(reference.device)
            tick_start = time.perf_counter()
        observed = states[:, k] if mode == "b" else states[:, 0]
        reference = block.targets(observed, reference, start=k, count=1)[:, 0]
        if tick_times is not None:
            sync(reference.device)
            tick_times.append((time.perf_counter() - tick_start) * 1000)
        outputs.append(reference)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def evaluate(args, model, text, identity):
    from .phoenix import read_prior

    saved = torch.load(args.head_weights, weights_only=True, map_location="cpu")
    if saved["identity"] != identity:
        raise ValueError("Evaluation assets differ from head training")
    cfg = saved["config"]
    family = saved["family"]
    if args.mode not in ({"a", "b", "b_stale", "phoenix"} if family == "ab" else {family}):
        raise ValueError("Evaluation mode is incompatible with this head family")
    head = make_head(cfg, family, model)
    head.load_state_dict(saved["head"])
    head.eval()
    coords = JointCoordinates(saved["bounds"], model.device)
    data = KitchenWindows(args.data_root, cfg["sources"], "val", head.horizon)
    # Fixed offsets per episode for paired comparisons across model families.
    totals, latencies, tick_times = {}, [], []
    for ep_index, (_, source, task, ep, _, _) in enumerate(data.episodes):
        begin = data.ends[ep_index-1] if ep_index else 0
        end = data.ends[ep_index]
        offsets = np.unique(np.linspace(begin, end-1, min(end-begin, args.windows), dtype=int))
        errors = []
        for idx in offsets:
            sample = data[int(idx)]
            _, rgb, physical_states, initial, _ = sample
            states = coords.normalized(physical_states)[None]
            sync(model.device)
            start = time.perf_counter()
            reference = coords.normalized(initial)[None]
            obs = observation(rgb, coords.normalized(physical_states[0])[None], text[task])
            prior = read_prior(model, head, obs, active=ACTIVE,
                               denoise_steps=cfg["policy"]["denoise_steps"], seed=cfg["train"]["seed"] + int(idx))
            sync(model.device)
            read = time.perf_counter()
            pred = replay(args.mode, head, prior, reference, states, tick_times=tick_times)
            physical = coords.physical(pred).cpu()
            sync(model.device)
            end_time = time.perf_counter()
            latencies.append(((read-start)*1000, (end_time-read)*1000, (end_time-start)*1000))
            errors.append((physical - sample[-1][None]).square().mean(dim=(0, 1)))
        mse = torch.stack(errors).mean(0)
        totals.setdefault(task, []).append(mse)
        print(json.dumps(dict(source=source, episode=ep, task=task, windows=len(offsets),
                              arm_rmse=mse[ARMS].mean().sqrt().item(),
                              gripper_rmse=mse[GRIPPERS].mean().sqrt().item())))
    for task, rows in totals.items():
        mse = torch.stack(rows).mean(0)
        print(json.dumps(dict(task=task, episodes=len(rows), arm_rmse=mse[ARMS].mean().sqrt().item(),
                              gripper_rmse=mse[GRIPPERS].mean().sqrt().item())))
    if tick_times:
        print(json.dumps(dict(mode=args.mode, feedback_ticks=len(tick_times),
                              head_tick_p50_ms=float(np.percentile(tick_times, 50)),
                              head_tick_p95_ms=float(np.percentile(tick_times, 95)))))
    print(json.dumps(dict(mode=args.mode, windows=len(latencies),
                          latency_ms_columns=["observation_and_prior", "head_and_output", "total"],
                          p50_ms=np.percentile(latencies, 50, axis=0).tolist(),
                          p95_ms=np.percentile(latencies, 95, axis=0).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["train", "eval"])
    parser.add_argument("--config", type=Path, default=Path("configs/ode_policy/kitchen.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True, help="Resolved parent model configuration")
    parser.add_argument("--phoenix-weights", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, help="Existing kitchen T5 cache; otherwise encode four task prompts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--family", choices=["ab", "c1", "c2"])
    parser.add_argument("--mode", choices=["phoenix", "a", "b", "b_stale", "c1", "c2"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--head-weights", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--windows", type=int, default=3, help="Held-out windows per episode")
    args = parser.parse_args()
    if args.operation == "train" and not (args.output and args.family):
        parser.error("train requires --family and --output")
    if args.operation == "eval" and not (args.head_weights and args.mode):
        parser.error("eval requires --head-weights and --mode")
    if any(x is not None and x < 1 for x in (args.steps, args.batch_size, args.windows)):
        parser.error("counts must be positive")
    if args.operation == "train":
        cfg = yaml.safe_load(args.config.read_text())
    else:
        cfg = torch.load(args.head_weights, weights_only=True, map_location="cpu")["config"]
    torch.manual_seed(cfg["train"]["seed"])
    identity = {key: digest(getattr(args, key)) for key in ("model_config", "phoenix_weights")}
    identity["text_cache"] = digest(args.text_cache) if args.text_cache else None
    model, text = load_assets(args, cfg)
    if args.operation == "train":
        train(args, cfg, model, text, identity)
    else:
        evaluate(args, model, text, identity)


if __name__ == "__main__":
    main()
