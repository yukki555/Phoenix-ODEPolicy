"""Construct a frozen parent plus a trainable ODE head without importing the training CLI."""
from dataclasses import dataclass

from torch import nn

from .architectures import ActionPlanResidual, architecture_type
from .calibration import PriorCoordinates
from .head import ActionBlock, ODEHead
from .kitchen import ACTIVE, JointCoordinates
from .recipe import checkpoint_recipe, validate_conditioning


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

    def prepare_block(self, obs, reference, *, seed, initial_q=None):
        return ActionBlock(self.head, self.read_condition(obs, seed=seed), reference, initial_q=initial_q)


def build_head(spec, *, token_width, device):
    """Build just the head for cached training or bind it to Phoenix via build_architecture."""
    strategy = architecture_type(spec["architecture"])
    policy = spec["policy"]
    if policy["solver"] != "euler":
        raise ValueError("Kitchen heads require a fixed-step Euler solver")
    conditioning = spec.get("conditioning", {})
    validate_conditioning(strategy, conditioning)
    head = ODEHead(strategy.name, token_width, horizon=policy["horizon"],
                   hz=policy["hz"], hidden_dim=policy["hidden_dim"], context_dim=policy["context_dim"],
                   latent_feedback=conditioning.get("feedback_route") == "latent_adapter",
                   history_feedback=conditioning.get("history_feedback")).to(device)
    head.token_reduction = conditioning.get("token_reduction", "none")
    return head


def make_head(cfg, family, model):
    """Legacy unbound head constructor for existing analysis scripts."""
    return build_head({"architecture": family, "policy": cfg["policy"]},
                      token_width=model.action_expert.hidden_dim, device=model.device)


def build_architecture(spec, phoenix, coordinates):
    """Bind a resolved architecture and its verified physical coordinate transforms."""
    strategy = architecture_type(spec["architecture"])
    if isinstance(strategy, ActionPlanResidual):
        mapping = spec["action_prior"]["mapping"]
        actual = "fitted_affine" if coordinates.affine is not None else (
            "source_physical" if coordinates.action is not None else "raw_slots")
        if actual != mapping:
            raise ValueError(f"Configured action mapping {mapping} differs from available mapping {actual}")
    phoenix.eval().requires_grad_(False)
    head = build_head(spec, token_width=phoenix.action_expert.hidden_dim, device=phoenix.device)
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
