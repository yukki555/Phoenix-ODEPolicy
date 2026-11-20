"""Hardware-independent deployment API. The caller owns the command queue and acknowledgments."""
import torch

from .calibration import PriorCoordinates
from .head import ActionBlock
from .kitchen import ACTIVE, JointCoordinates, observation
from .phoenix import read_prior


class KitchenPolicy:
    def __init__(self, phoenix, head, coordinates, text, *, denoise_steps=10, seed=42, source=None, prior_affine=None):
        self.phoenix = phoenix.eval().requires_grad_(False)
        self.head = head.eval()
        self.coordinates = coordinates
        self.prior_coords = PriorCoordinates(coordinates, source, prior_affine)
        self.text = text
        self.denoise_steps, self.seed = denoise_steps, seed

    @classmethod
    def from_checkpoint(cls, phoenix, saved, text, identity):
        """Restore the exact source-state encoding and prior map used for this head."""
        from .runner import make_head
        if identity != saved["identity"]:
            raise ValueError("Deployment assets differ from head training")
        head = make_head(saved["config"], saved["family"], phoenix)
        head.load_state_dict(saved["head"])
        return cls(phoenix, head, JointCoordinates(saved["bounds"], phoenix.device), text,
                   denoise_steps=saved["config"]["policy"]["denoise_steps"],
                   seed=saved["config"]["train"]["seed"], source=saved.get("source"),
                   prior_affine=saved.get("prior_affine"))

    @torch.no_grad()
    def begin_block(self, task, rgb, measured, last_sent=None, *, seed=None):
        """RGB [3,3,H,W] uint8, original main/left/right order; q and targets in physical units."""
        if measured.shape != (14,) or (last_sent is not None and last_sent.shape != (14,)):
            raise ValueError("Expected physical joint vectors of length 14")
        if rgb.ndim != 4 or rgb.shape[:2] != (3, 3) or rgb.dtype != torch.uint8:
            raise ValueError("Expected three uint8 CHW RGB cameras")
        q = self.coordinates.normalized(measured)[None]
        initial = q if last_sent is None else self.coordinates.normalized(last_sent)[None]
        obs = observation(rgb, self.prior_coords.state.normalized(measured)[None], self.text[task])
        prior = read_prior(self.phoenix, self.head, obs, active=ACTIVE,
                           denoise_steps=self.denoise_steps, seed=self.seed if seed is None else seed)
        if self.head.family == "ab":
            prior = self.prior_coords.slots(prior)
        return KitchenBlock(ActionBlock(self.head, prior, initial), self.coordinates, q)


class KitchenBlock:
    def __init__(self, block, coordinates, initial_state):
        self.block, self.coordinates, self.initial_state = block, coordinates, initial_state

    @torch.no_grad()
    def full_chunk(self):
        head = self.block.head
        result = self.block.targets(self.initial_state, self.block.block_start, start=0, count=head.horizon)
        return self.coordinates.physical(result[0])

    @torch.no_grad()
    def next_target(self, k, measured, last_sent):
        """Compute tick k+1. Pass the actual committed target at k in physical units."""
        if measured.shape != (14,) or last_sent.shape != (14,):
            raise ValueError("Expected physical joint vectors of length 14")
        result = self.block.targets(self.coordinates.normalized(measured)[None],
                                    self.coordinates.normalized(last_sent)[None], start=k, count=1)
        return self.coordinates.physical(result[0, 0])
