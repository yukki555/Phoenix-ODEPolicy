"""Construct a frozen parent plus a trainable ODE head without importing the training CLI."""
from dataclasses import dataclass

from torch import nn

from .architectures import architecture_type
from .calibration import PriorCoordinates
from .head import ActionBlock, ODEHead
from .kitchen import ACTIVE, JointCoordinates
from .recipe import checkpoint_recipe


@dataclass
class Architecture:
    phoenix: nn.Module
    head: ODEHead
    coordinates: PriorCoordinates
    denoise_steps: int

    def read_condition(self, obs, *, seed):
        strategy = self.head.architecture
        raw = strategy.read(self.phoenix, self.head, obs, ACTIVE, self.denoise_steps, seed)
        return strategy.map_condition(raw, self.coordinates)

    def prepare_block(self, obs, reference, *, seed):
        return ActionBlock(self.head, self.read_condition(obs, seed=seed), reference)


def _make_head(spec, phoenix):
    """Shared implementation for the factory and the legacy head constructor."""
    strategy = architecture_type(spec["architecture"])
    policy = spec["policy"]
    if policy["solver"] != "euler":
        raise ValueError("Kitchen heads require a fixed-step Euler solver")
    reduction = spec.get("conditioning", {}).get("token_reduction", "none")
    if reduction != "none" and (strategy.legacy != "c1" or reduction != "temporal_mean_repeat"):
        raise ValueError("Unsupported reduction for this architecture")
    head = ODEHead(strategy.name, phoenix.action_expert.hidden_dim, horizon=policy["horizon"],
                   hz=policy["hz"], hidden_dim=policy["hidden_dim"], context_dim=policy["context_dim"]).to(phoenix.device)
    head.token_reduction = reduction
    return head


def make_head(cfg, family, model):
    """Legacy unbound head constructor for existing analysis scripts."""
    return _make_head({"architecture": family, "policy": cfg["policy"]}, model)


def build_architecture(spec, phoenix, coordinates):
    """Bind a resolved architecture and its verified physical coordinate transforms."""
    strategy = architecture_type(spec["architecture"])
    if strategy.legacy == "ab":
        mapping = spec["action_prior"]["mapping"]
        actual = "fitted_affine" if coordinates.affine is not None else (
            "source_physical" if coordinates.action is not None else "raw_slots")
        if actual != mapping:
            raise ValueError(f"Configured action mapping {mapping} differs from available mapping {actual}")
    phoenix.eval().requires_grad_(False)
    head = _make_head(spec, phoenix)
    return Architecture(phoenix, head, coordinates, spec["policy"].get("denoise_steps", 1))


def restore_architecture(saved, phoenix, identity):
    if saved["identity"] != identity:
        raise ValueError("Parent assets differ from head training")
    spec = checkpoint_recipe(saved)
    kitchen = JointCoordinates(saved["bounds"], phoenix.device)
    coordinates = PriorCoordinates(kitchen, saved.get("source"), saved.get("prior_affine"))
    arch = build_architecture(spec, phoenix, coordinates)
    arch.head.load_state_dict(saved["head"], strict=True)
    arch.head.eval()
    return arch
