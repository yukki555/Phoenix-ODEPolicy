"""Hardware-independent deployment API. The caller owns the command queue and acknowledgments."""
import torch

from .head import ActionBlock
from .kitchen import ACTIVE, observation
from .phoenix import read_prior


class KitchenPolicy:
    def __init__(self, phoenix, head, coordinates, text, *, denoise_steps=10, seed=42):
        self.phoenix = phoenix.eval().requires_grad_(False)
        self.head = head.eval()
        self.coordinates = coordinates
        self.text = text
        self.denoise_steps, self.seed = denoise_steps, seed

    @torch.no_grad()
    def begin_block(self, task, rgb, measured, last_sent=None, *, seed=None):
        """RGB [3,3,H,W] uint8, original main/left/right order; q and targets in physical units."""
        if measured.shape != (14,) or (last_sent is not None and last_sent.shape != (14,)):
            raise ValueError("Expected physical joint vectors of length 14")
        if rgb.ndim != 4 or rgb.shape[:2] != (3, 3) or rgb.dtype != torch.uint8:
            raise ValueError("Expected three uint8 CHW RGB cameras")
        q = self.coordinates.normalized(measured)[None]
        initial = q if last_sent is None else self.coordinates.normalized(last_sent)[None]
        obs = observation(rgb, q, self.text[task])
        prior = read_prior(self.phoenix, self.head, obs, active=ACTIVE,
                           denoise_steps=self.denoise_steps, seed=self.seed if seed is None else seed)
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
        """B: compute tick k+1. Pass the actual committed target at k in physical units."""
        if self.block.head.family != "ab":
            raise ValueError("Per-tick feedback requires the shared A/B head")
        if measured.shape != (14,) or last_sent.shape != (14,):
            raise ValueError("Expected physical joint vectors of length 14")
        result = self.block.targets(self.coordinates.normalized(measured)[None],
                                    self.coordinates.normalized(last_sent)[None], start=k, count=1)
        return self.coordinates.physical(result[0, 0])
