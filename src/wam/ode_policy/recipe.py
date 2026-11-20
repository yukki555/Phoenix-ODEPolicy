"""Compose human-readable experiments and adapt the original kitchen CLI/checkpoints."""
from copy import deepcopy
from pathlib import Path

import yaml

from .architectures import architecture_type

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs/ode_policy"
EXECUTIONS = ("block_start", "latest_joint_each_tick")
MODE_ALIASES = {"a": EXECUTIONS[0], "fixed": EXECUTIONS[0], "c1": EXECUTIONS[0], "c2": EXECUTIONS[0],
                "b": EXECUTIONS[1], "feedback": EXECUTIONS[1], "phoenix": "mapped_action_plan"}


def legacy_recipe(cfg, family, source=None, affine=None):
    arch = architecture_type(family)
    training = dict(cfg["train"])
    probability = training.pop("feedback_probability", 0.5)
    spec = dict(architecture=arch.name, policy=dict(cfg["policy"]), training=training,
                parent=dict(frozen=True, state_stats_group=source.get("group") if source else None),
                evaluation=dict(executions=list(EXECUTIONS), baselines=["hold_latest"], paired_windows=True))
    for key in ("sources", "instructions"):
        if key in cfg:
            spec[key] = cfg[key]
    training["objective"] = "action_path_mse"
    if arch.legacy == "ab":
        training["observation_schedule"] = "sample_suffix_boundary"
        mapping = "fitted_affine" if affine is not None else ("source_physical" if source else "raw_slots")
        spec["action_prior"] = {"mapping": mapping}
    else:
        training["observation_schedule"] = "mix_block_start_and_per_tick"
        training["per_tick_probability"] = probability
        for key in ("suffix_warmup_steps", "suffix_probability", "model_boundary_probability"):
            training.pop(key, None)
    if arch.legacy == "c1":
        spec["conditioning"] = {"token_reduction": "none"}
    spec["phoenix_read"], spec["integrates"] = arch.reads, arch.integrates
    return spec


def checkpoint_recipe(saved):
    if "recipe" in saved:
        return saved["recipe"]
    cfg = saved["config"]
    spec = legacy_recipe(cfg, saved["family"], saved.get("source"), saved.get("prior_affine"))
    training = spec["training"]
    if (saved["family"] in {"c1", "c2"} and "training_windows" not in saved
            and "feedback_probability" not in cfg["train"]):
        # Pre-feedback C heads trained whole chunks using block-start measurements.
        training["observation_schedule"] = "block_start"
        training.pop("per_tick_probability", None)
    for key in ("steps", "batch_size"):
        if key in saved:
            training[key] = saved[key]
    return spec


def validate_recipe(spec):
    arch = architecture_type(spec["architecture"])
    if spec["parent"]["frozen"] is not True or spec["parent"]["checkpoint"] != "phoenix_joint96_256013":
        raise ValueError("Kitchen recipes require frozen Phoenix Joint96 step 256013")
    if (spec["phoenix_read"], spec["integrates"]) != (arch.reads, arch.integrates):
        raise ValueError("Condition or integration semantics conflict with the architecture")
    if arch.legacy == "ab":
        if spec["action_prior"]["mapping"] not in {"source_physical", "fitted_affine"}:
            raise ValueError("Choose source_physical or fitted_affine for the action plan")
    elif "action_prior" in spec:
        raise ValueError("Action mapping applies only to the residual plan architecture")
    conditioning = spec.get("conditioning", {})
    if conditioning and (arch.legacy != "c1" or set(conditioning) != {"token_reduction"}
                         or conditioning["token_reduction"] not in {"none", "temporal_mean_repeat"}):
        raise ValueError("Invalid token reduction for this architecture")
    t = spec["training"]
    if t["objective"] != "action_path_mse" or t["observation_schedule"] not in {
            "sample_suffix_boundary", "mix_block_start_and_per_tick", *EXECUTIONS}:
        raise ValueError("Unsupported training objective or observation schedule")
    if t["steps"] < 1 or t["batch_size"] < 1 or t["lr"] <= 0:
        raise ValueError("Training steps, batch size, and learning rate must be positive")
    if t["observation_schedule"] == "sample_suffix_boundary" and (
        t["suffix_warmup_steps"] < 0 or not 0 <= t["suffix_probability"] <= 1
        or not 0 <= t["model_boundary_probability"] <= 1
    ):
        raise ValueError("Invalid suffix sampling settings")
    if t["observation_schedule"] == "mix_block_start_and_per_tick" and not 0 <= t["per_tick_probability"] <= 1:
        raise ValueError("per_tick_probability must lie in [0,1]")
    e = spec["evaluation"]
    baselines = {"hold_latest", "mapped_action_plan"} if arch.legacy == "ab" else {"hold_latest"}
    modes = e["executions"] + e["baselines"]
    if (not modes or len(modes) != len(set(modes)) or not set(e["executions"]) <= set(EXECUTIONS)
            or not set(e["baselines"]) <= baselines or (len(modes) > 1 and e["paired_windows"] is not True)):
        raise ValueError("Specify distinct supported executions/baselines with paired_windows for comparisons")


def resolve_recipe(path):
    path = Path(path)
    root = path.parent.parent if (path.parent.parent / "architectures").is_dir() else CONFIG_ROOT
    spec = yaml.safe_load(path.read_text())
    arch = architecture_type(spec["architecture"])
    dataset = yaml.safe_load((root / (spec["dataset"] + ".yaml")).read_text())
    preset = yaml.safe_load((root / "architectures" / (arch.name + ".yaml")).read_text())
    spec["sources"], spec["instructions"] = dataset["sources"], dataset["instructions"]
    spec["policy"] = {**{k: dataset["policy"][k] for k in ("horizon", "hz")},
                      **preset["policy"], **spec.get("policy", {})}
    if "conditioning" in preset:
        spec["conditioning"] = {**preset["conditioning"], **spec.get("conditioning", {})}
    spec["phoenix_read"], spec["integrates"] = arch.reads, arch.integrates
    parent = spec["parent"]
    parent["state_stats_profile"] = str((root / parent["state_stats_profile"]).resolve())
    if spec.get("action_prior", {}).get("audit"):
        prior = spec["action_prior"]
        prior["audit"] = str((path.parent / prior["audit"]).resolve())
    validate_recipe(spec)
    return spec


def legacy_config(spec):
    """Config view for original readers, audits, and external analysis scripts."""
    training = deepcopy(spec["training"])
    training["feedback_probability"] = training.get("per_tick_probability", 0.5)
    return dict(policy=spec["policy"], train=training, sources=spec["sources"], instructions=spec["instructions"])


def evaluation_modes(spec, mode=None):
    if mode is None:
        return spec["evaluation"]["executions"] + spec["evaluation"]["baselines"]
    if mode == "paired":
        return ["fixed", "feedback", "hold_latest"]
    family = architecture_type(spec["architecture"]).legacy
    if mode in {"a", "b", "b_stale", "phoenix", "mapped_action_plan"} and family != "ab":
        raise ValueError("Plan comparison requires an action-plan checkpoint")
    if mode in {"c1", "c2"} and mode != family:
        raise ValueError("Legacy mode and checkpoint family disagree")
    if mode not in {*EXECUTIONS, *MODE_ALIASES, "b_stale", "hold_latest", "mapped_action_plan"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    return [mode]
