"""Train and evaluate kitchen action ODE heads from an experiment recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
from collections import Counter

import numpy as np
import torch
import yaml

from .calibration import PriorCoordinates, audit_prior, load_source
from .factory import build_architecture, make_head, restore_architecture
from .architectures import ARCHITECTURES, ActionPlanResidual, ActionTokens, architecture_type
from .recipe import (EXECUTIONS, HISTORY_CONTROLS, LATENT_CONTROLS, MODE_ALIASES, checkpoint_recipe, evaluation_modes, legacy_config,
                     legacy_recipe, resolve_recipe)
from .metrics import motion_metrics, summarize
from .replay import feedback_action_probe, replay, sync
from .kitchen import ACTIVE, JointCoordinates, KitchenWindows, observation


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


def sample_inputs(sample, coordinates, texts, prior_coords):
    task, rgb, states, reference, targets = sample
    phoenix_state = prior_coords.state.normalized(states[0])[None]
    states = coordinates.normalized(states)[None]
    return (observation(rgb, phoenix_state, texts[task]), states,
            coordinates.normalized(reference)[None], coordinates.normalized(targets)[None])


def cached_action_tokens(cache_dir, model, head, obs, *, denoise_steps, seed):
    """Content-address frozen action tokens; the directory identifies the parent and reader settings."""
    from .phoenix import read_prior

    key = hashlib.sha256(str(seed).encode())
    for name in ("input_image", "proprio", "context", "context_mask"):
        tensor = obs[name].detach().cpu().contiguous()
        key.update(str((name, tensor.dtype, tuple(tensor.shape))).encode())
        key.update(tensor.view(torch.uint8).numpy().tobytes())
    path = cache_dir / key.hexdigest()[:2] / (key.hexdigest() + ".pt")
    if path.is_file():
        return torch.load(path, map_location=model.device, weights_only=True)
    tokens = read_prior(model, head, obs, active=ACTIVE, denoise_steps=denoise_steps, seed=seed)
    try:
        save_new(tokens.detach().cpu(), path)
    except FileExistsError:
        pass  # Concurrent readers computed the identical frozen condition.
    return tokens


# Historical API used by existing experiment scripts.
c1_training_tokens = cached_action_tokens


def initialize_from_direct_action_tokens(head, path, spec, identity, source, bounds):
    """Initialize a frozen-head adapter experiment from the matched direct action-token head."""
    saved = torch.load(path, weights_only=True, map_location="cpu")
    base = checkpoint_recipe(saved)
    conditioning = base.get("conditioning", {})
    if (base["architecture"] != "action_token_ode" or conditioning.get("feedback_route", "direct") != "direct"
            or conditioning.get("token_reduction", "none") != spec["conditioning"]["token_reduction"]):
        raise ValueError("Expected a direct action-token checkpoint with matching token reduction")
    if (saved["identity"] != identity or any(base[key] != spec[key] for key in ("sources", "instructions"))
            or any(base["policy"][key] != value for key, value in spec["policy"].items())):
        raise ValueError("Base action-token parent, dataset or policy differs from the latent feedback recipe")
    original = saved.get("source")
    if (not torch.equal(saved["bounds"], bounds) or original is None or original["group"] != source["group"]
            or not torch.equal(original["state"], source["state"])):
        raise ValueError("Base action-token coordinates differ from adapter training")
    adapter_keys = {"feedback_adapter." + k for k in head.feedback_adapter.state_dict()}
    if saved["head"].keys() != head.state_dict().keys() - adapter_keys:
        raise ValueError("Base checkpoint must contain exactly the direct action-token parameters")
    head.load_state_dict({**head.state_dict(), **saved["head"]})
    return digest(path)


initialize_from_direct_c1 = initialize_from_direct_action_tokens


def train(args, cfg, model, text, identity, *, recipe=None):
    data = KitchenWindows(args.data_root, cfg["sources"], "train", cfg["policy"]["horizon"])
    bounds = data.fit_train_bounds()
    coords = JointCoordinates(bounds, model.device)
    source, affine = None, None
    name = recipe["architecture"] if recipe is not None else getattr(args, "architecture", None)
    if name is None:
        name = args.family  # Programmatic callers of the legacy trainer.
    architecture = architecture_type(name)
    needs_action_scale = isinstance(architecture, ActionPlanResidual)
    if recipe is not None:
        parent = recipe["parent"]
        source = load_source(parent["state_stats_profile"], parent["state_stats_group"], identity,
                             include_action=needs_action_scale)
    if getattr(args, "prior_audit", None):
        audit = torch.load(args.prior_audit, map_location="cpu", weights_only=True)
        if (audit["identity"] != identity or audit["sources"] != cfg["sources"]
                or audit["policy"] != cfg["policy"] or audit["instructions"] != cfg["instructions"]
                or not torch.equal(audit["bounds"], bounds)):
            raise ValueError("Prior audit differs from current assets, data, or coordinate system")
        audit_source = audit["source"]
        if source is not None and (audit_source is None or audit_source["group"] != source["group"]
                                   or not all(torch.equal(audit_source[k], source[k]) for k in source if k in {"state", "action"})):
            raise ValueError("Prior audit coordinates differ from recipe")
        source = audit_source
        if args.prior_kind == "fitted":
            affine = audit["affine"]
        elif source is None:
            raise ValueError("Native ARX decoding requires source statistics")
    elif recipe is None and getattr(args, "source_profile", None):
        source = load_source(args.source_profile, args.source_group, identity, include_action=needs_action_scale)
    spec = recipe if recipe is not None else legacy_recipe(cfg, architecture.name, source, affine)
    if not needs_action_scale:
        source = {k: v for k, v in source.items() if k != "action"} if source else None
        affine = None
    prior_coords = PriorCoordinates(coords, source, affine)
    arch = build_architecture(spec, model, prior_coords)
    head = arch.head
    cfg_train = spec["training"]
    if head.feedback_adapter is not None:
        base_weights = getattr(args, "base_action_token_weights", None) or getattr(args, "base_c1_weights", None)
        base_sha = initialize_from_direct_action_tokens(head, base_weights, spec, identity, source, bounds)
        cfg_train = {**cfg_train, "base_c1_sha256": base_sha}
    optimizer = torch.optim.AdamW((p for p in head.parameters() if p.requires_grad), lr=cfg_train["lr"])
    seed = cfg_train["seed"]
    sampler, boundaries = random.Random(seed), random.Random(seed + 1)
    feedback_rng = random.Random(seed + 2)
    schedule = cfg_train["observation_schedule"]
    probability = cfg_train.get("per_tick_probability", 0.5)
    if not 0 <= probability <= 1:
        raise ValueError("feedback_probability must be between 0 and 1")
    mode_counts = Counter()
    batch = args.batch_size or cfg_train["batch_size"]
    steps = args.steps or cfg_train["steps"]
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    cache_dir = getattr(args, "action_token_cache_dir", None) or getattr(args, "c1_cache_dir", None)
    if cache_dir is not None:
        if not isinstance(architecture, ActionTokens):
            raise ValueError("Only frozen action tokens can use the training token cache")
        settings = ("c1-raw-tokens-v1", identity, head.horizon, cfg["policy"]["denoise_steps"], ACTIVE)
        cache_dir = cache_dir / hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(batch):
            index = sampler.randrange(len(data))
            obs, states, reference, targets = sample_inputs(data[index], coords, text, prior_coords)
            if cache_dir is None:
                prior = arch.read_condition(obs, seed=seed + index)
            else:
                prior = cached_action_tokens(cache_dir, model, head, obs, denoise_steps=arch.denoise_steps, seed=seed + index)
            k = 0
            if schedule == "sample_suffix_boundary" and step >= cfg_train["suffix_warmup_steps"] and boundaries.random() < cfg_train["suffix_probability"]:
                k = boundaries.randrange(1, head.horizon)
            feedback = (schedule == "latest_joint_each_tick" or
                        (schedule == "mix_block_start_and_per_tick" and feedback_rng.random() < probability))
            if schedule != "sample_suffix_boundary":
                mode_counts["feedback" if feedback else "fixed"] += 1
            loss = head.path_loss(prior, states, reference, targets, boundary=k, feedback=feedback,
                                  model_boundary=boundaries.random() < cfg_train.get("model_boundary_probability", 0.5))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite ODE path loss")
            (loss / batch).backward()
            total += loss.item() / batch
        optimizer.step()
        print(json.dumps({"step": step+1, "loss": total}), flush=True)
    spec = {**spec, "training": {**cfg_train, "steps": steps, "batch_size": batch}}
    save_new(dict(schema_version=2, architecture=architecture.name, family=architecture.legacy, config=cfg, recipe=spec, identity=identity, bounds=bounds,
                  source=source, prior_affine=affine, head=head.state_dict(),
                  steps=steps, batch_size=batch, training_windows=dict(mode_counts)), output)


def save_new(payload, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(payload, temp)
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)


@torch.no_grad()
def evaluate(args, model, text, identity):
    saved = torch.load(args.head_weights, weights_only=True, map_location="cpu")
    spec = checkpoint_recipe(saved)
    requested = getattr(args, "evaluation_recipe", None)
    if requested:
        for key in ("architecture", "conditioning", "sources", "instructions"):
            if requested.get(key) != spec.get(key):
                raise ValueError(f"Evaluation recipe differs from checkpoint: {key}")
        policy_keys = ["horizon", "hz", "solver", "hidden_dim", "context_dim"]
        if spec["phoenix_read"] != "single_current_query":
            policy_keys.append("denoise_steps")
        if any(requested["policy"][key] != spec["policy"][key] for key in policy_keys):
            raise ValueError("Evaluation integration or parent sampling settings differ from checkpoint")
        if requested["parent"]["state_stats_group"] != spec["parent"]["state_stats_group"]:
            raise ValueError("Evaluation state statistics differ from checkpoint")
        if requested.get("action_prior", {}).get("mapping") != spec.get("action_prior", {}).get("mapping"):
            raise ValueError("Evaluation action mapping differs from checkpoint")
        spec = {**spec, "evaluation": requested["evaluation"]}
    arch = restore_architecture(saved, model, identity)
    prior_coords, head = arch.coordinates, arch.head
    coords = prior_coords.kitchen
    data = KitchenWindows(args.data_root, spec["sources"], "val", head.horizon)
    modes = evaluation_modes(spec, args.mode)
    paired_probe = args.mode in (None, "paired") and any(m in modes for m in ("adapter_off", "feedback_off"))
    reference_head = None
    reference_path = getattr(args, "reference_head_weights", None)
    reference_origin = "checkpoint_nominal"
    token_seed = spec["training"]["seed"]
    if reference_path:
        if not paired_probe:
            raise ValueError("A common reference head requires paired feedback diagnostics")
        baseline = torch.load(reference_path, weights_only=True, map_location="cpu")
        baseline_spec = checkpoint_recipe(baseline)
        baseline_source, source = baseline.get("source"), saved.get("source")
        same_state_scale = (baseline_source is None and source is None) or (
            baseline_source is not None and source is not None and baseline_source["group"] == source["group"]
            and torch.equal(baseline_source["state"], source["state"]))
        if (baseline_spec["architecture"] != ActionTokens.name or not same_state_scale
                or baseline_spec.get("conditioning", {}).get("feedback_route", "direct") != "direct"
                or any(spec[key] != baseline_spec[key] for key in ("sources", "instructions"))
                or any(spec["policy"][key] != baseline_spec["policy"][key] for key in ("denoise_steps", "hz", "horizon"))
                or not torch.equal(baseline["bounds"], saved["bounds"])):
            raise ValueError("Expected a matched direct action-token checkpoint for the common reference")
        reference_head = restore_architecture(baseline, model, identity).head
        reference_origin = digest(reference_path)
        token_seed = baseline_spec["training"]["seed"]
    totals = {mode: {} for mode in modes}
    latencies = {mode: [] for mode in modes}
    tick_times = {mode: [] for mode in modes}
    for ep_index, (_, source, task, ep, _, _) in enumerate(data.episodes):
        begin = data.ends[ep_index-1] if ep_index else 0
        end = data.ends[ep_index]
        offsets = np.unique(np.linspace(begin, end-1, min(end-begin, args.windows), dtype=int))
        rows = {mode: [] for mode in modes}
        for idx in offsets:
            _, rgb, physical_states, initial, targets = data[int(idx)]
            if modes != ["hold_latest"]:
                states = coords.normalized(physical_states)[None]
                sync(model.device)
                start = time.perf_counter()
                reference = coords.normalized(initial)[None]
                obs = observation(rgb, prior_coords.state.normalized(physical_states[0])[None], text[task])
                prior = arch.read_condition(obs, seed=token_seed + int(idx))
                sync(model.device)
                prior_ms = (time.perf_counter()-start)*1000
            for mode in modes:
                if mode == "hold_latest":
                    physical = physical_states
                else:
                    tick = tick_times[mode]
                    start = time.perf_counter()
                    ready = []
                    pred = replay(mode, head, prior, reference, states, tick_times=tick,
                                  first_ready=ready, coordinates=coords)
                    physical = pred[0]
                    head_ms = (time.perf_counter()-start)*1000
                    latencies[mode].append((prior_ms, head_ms, prior_ms+head_ms, prior_ms+ready[0]))
                rows[mode].append(motion_metrics(physical, targets, physical_states, saved["bounds"]))
            if paired_probe:
                previous = None
                if reference_head is not None:
                    fixed_path = replay("block_start", reference_head, prior, reference, states)
                    previous = torch.cat((reference[:, None], fixed_path[:, :-1]), dim=1)
                probe = feedback_action_probe(head, prior, reference, states, coords, targets, previous=previous)
                print(json.dumps(dict(diagnostic="shared_reference_feedback", task=task, source=source,
                                      episode=ep, window=int(idx), reference_origin=reference_origin, **probe)))
        for mode in modes:
            totals[mode].setdefault(task, []).append(rows[mode])
            print(json.dumps(dict(mode=mode, source=source, episode=ep, task=task,
                                  window_indices=offsets.tolist(), **summarize([rows[mode]]))))
    for mode in modes:
        for task, episodes in totals[mode].items():
            print(json.dumps(dict(mode=mode, task=task, **summarize(episodes))))
        if tick_times[mode]:
            samples = np.asarray(tick_times[mode])
            print(json.dumps(dict(mode=mode, feedback_ticks=len(samples),
                                  head_tick_p50_ms=float(np.percentile(samples, 50)),
                                  head_tick_p95_ms=float(np.percentile(samples, 95)),
                                  head_tick_deadline_miss_fraction=float(np.mean(samples > 1000/head.hz)))))
        if latencies[mode]:
            print(json.dumps(dict(mode=mode, windows=len(latencies[mode]),
                                  latency_ms_columns=["observation_and_prior", "head_and_output", "total", "first_target"],
                                  p50_ms=np.percentile(latencies[mode], 50, axis=0).tolist(),
                                  p95_ms=np.percentile(latencies[mode], 95, axis=0).tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["train", "eval", "audit-prior", "resolve"])
    parser.add_argument("--config", type=Path, help="Legacy dataset/training YAML")
    parser.add_argument("--recipe", type=Path, help="Experiment recipe")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-config", type=Path, help="Resolved parent model configuration")
    parser.add_argument("--phoenix-weights", type=Path)
    parser.add_argument("--text-cache", type=Path, help="Existing kitchen T5 cache; otherwise encode four task prompts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--architecture", "--family", choices=list(ARCHITECTURES),
                        type=lambda value: architecture_type(value).name, help="Architecture for legacy dataset configs")
    parser.add_argument("--mode", choices=[*EXECUTIONS, *LATENT_CONTROLS, *HISTORY_CONTROLS, *MODE_ALIASES, "paired", "hold_latest", "b_stale", "mapped_action_plan"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--head-weights", type=Path)
    parser.add_argument("--reference-head-weights", type=Path,
                        help="Direct action-token checkpoint supplying identical references across paired evaluations")
    parser.add_argument("--base-action-token-weights", "--base-c1-weights", type=Path,
                        help="Direct action-token checkpoint for the frozen-head adapter experiment")
    parser.add_argument("--action-token-cache-dir", "--c1-cache-dir", type=Path,
                        help="Per-window .pt token cache for the single-process RGB trainer")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--windows", type=int, default=3, help="Held-out windows per episode")
    parser.add_argument("--source-profile", type=Path, help="Parent-bound ARX scale JSON")
    parser.add_argument("--source-group", help="Exact stats_group for the candidate ARX domain")
    parser.add_argument("--prior-audit", type=Path, help="Reuse train-only map and input convention")
    parser.add_argument("--prior-kind", choices=["fitted", "native"], default="fitted")
    parser.add_argument("--raw-slot-control", action="store_true", help="Explicit legacy kitchen-scale assumption")
    parser.add_argument("--fit-windows", type=int, default=1, help="Fit anchors per training episode")
    parser.add_argument("--fit-episodes-per-task", type=int, help="Use evenly spaced training episodes for a pilot")
    args = parser.parse_args()
    if args.reference_head_weights and args.operation != "eval":
        parser.error("--reference-head-weights applies to paired evaluation")
    recipe = None
    if args.recipe:
        if args.operation == "audit-prior" or any((args.config, args.architecture, args.source_profile,
                                                   args.source_group, args.raw_slot_control)):
            parser.error("--recipe supplies the architecture, dataset and parent coordinates")
        recipe = resolve_recipe(args.recipe)
        args.architecture = recipe["architecture"]
        if args.operation == "train":
            prior = recipe.get("action_prior", {})
            if prior.get("mapping") == "fitted_affine":
                args.prior_audit = args.prior_audit or prior.get("audit")
                if not args.prior_audit:
                    parser.error("fitted_affine requires action_prior.audit or --prior-audit")
                args.prior_kind = "fitted"
            elif args.prior_audit:
                parser.error("A recipe accepts --prior-audit only for fitted_affine")
        elif args.operation == "eval":
            args.evaluation_recipe = recipe
    if args.operation == "resolve":
        if recipe is None:
            parser.error("resolve requires --recipe")
        print(yaml.safe_dump(recipe, sort_keys=False), end="")
        return
    if not (args.model_config and args.phoenix_weights and args.data_root):
        parser.error("Supply --model-config, --phoenix-weights and --data-root")
    needs_base = (args.operation == "train" and recipe is not None
                  and recipe["training"].get("trainable") == "feedback_adapter")
    if needs_base != bool(args.base_action_token_weights):
        parser.error("--base-action-token-weights is required exclusively for latent feedback adapter training")
    if args.action_token_cache_dir and (args.operation != "train" or args.architecture != ActionTokens.name):
        parser.error("--action-token-cache-dir requires training the action_token_ode architecture")
    if bool(args.source_profile) != bool(args.source_group):
        parser.error("Supply both --source-profile and --source-group")
    if sum(map(bool, (args.prior_audit, args.source_profile, args.raw_slot_control))) > 1:
        parser.error("Choose a prior audit, a source profile, or the raw control")
    if args.operation == "audit-prior" and (not args.output or args.prior_audit):
        parser.error("audit-prior requires a new --output and fits its own map")
    if args.operation == "train" and recipe is None and not any((args.prior_audit, args.source_profile, args.raw_slot_control)):
        parser.error("Choose the parent state scale (--source-profile/--source-group or --prior-audit); --raw-slot-control reproduces the kitchen-scale assumption")
    if args.operation == "eval" and any((args.prior_audit, args.source_profile, args.raw_slot_control)):
        parser.error("eval uses the head checkpoint's saved coordinates")
    if args.operation == "train" and not (args.output and args.architecture):
        parser.error("train requires an architecture and --output")
    if args.operation == "eval" and not args.head_weights:
        parser.error("eval requires --head-weights")
    if any(x is not None and x < 1 for x in (args.steps, args.batch_size, args.windows, args.fit_windows, args.fit_episodes_per_task)):
        parser.error("counts must be positive")
    if args.operation == "eval":
        saved = torch.load(args.head_weights, weights_only=True, map_location="cpu")
        cfg = legacy_config(checkpoint_recipe(saved))
    elif recipe is not None:
        cfg = legacy_config(recipe)
    else:
        cfg = yaml.safe_load((args.config or Path("configs/ode_policy/kitchen.yaml")).read_text())
    torch.manual_seed(cfg["train"]["seed"])
    identity = {key: digest(getattr(args, key)) for key in ("model_config", "phoenix_weights")}
    identity["text_cache"] = digest(args.text_cache) if args.text_cache else None
    model, text = load_assets(args, cfg)
    if args.operation == "train":
        train(args, cfg, model, text, identity, recipe=recipe)
    elif args.operation == "audit-prior":
        audit_prior(args, cfg, model, text, identity)
    else:
        evaluate(args, model, text, identity)


if __name__ == "__main__":
    main()
