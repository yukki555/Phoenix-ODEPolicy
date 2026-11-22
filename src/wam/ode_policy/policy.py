"""Hardware-independent deployment API. The caller owns the command queue and acknowledgments."""
import torch

from .calibration import PriorCoordinates
from .kitchen import observation
from .factory import Architecture, restore_architecture
from .recipe import checkpoint_recipe


class KitchenPolicy:
    def __init__(self, phoenix, head, coordinates, text, *, denoise_steps=10, seed=42, source=None, prior_affine=None):
        if isinstance(head, Architecture):
            self.architecture = head
        else:
            self.architecture = Architecture(phoenix.eval().requires_grad_(False), head,
                                             PriorCoordinates(coordinates, source, prior_affine), denoise_steps)
        self.phoenix = self.architecture.phoenix
        self.head = self.architecture.head.eval()
        self.prior_coords = self.architecture.coordinates
        self.coordinates = self.prior_coords.kitchen
        self.text = text
        self.denoise_steps, self.seed = self.architecture.denoise_steps, seed

    @classmethod
    def from_checkpoint(cls, phoenix, saved, text, identity):
        arch = restore_architecture(saved, phoenix, identity)
        return cls(phoenix, arch, arch.coordinates.kitchen, text, seed=checkpoint_recipe(saved)["training"]["seed"])

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
        block = self.architecture.prepare_block(obs, initial, seed=self.seed if seed is None else seed, initial_q=q)
        return KitchenBlock(block, self.coordinates, q)


class KitchenBlock:
    def __init__(self, block, coordinates, initial_state):
        self.block, self.coordinates, self.initial_state = block, coordinates, initial_state

    @torch.no_grad()
    def full_chunk(self, *, feedback_control=None):
        head = self.block.head
        result = self.block.targets(self.initial_state, self.block.block_start, start=0, count=head.horizon,
                                    feedback_control=feedback_control)
        return self.coordinates.physical(result[0])

    @torch.no_grad()
    def next_target(self, k, measured, last_sent, *, feedback_control=None):
        """Pass the actual q and committed target at k, including for delayed feedback replay."""
        if measured.shape != (14,) or last_sent.shape != (14,):
            raise ValueError("Expected physical joint vectors of length 14")
        result = self.block.targets(self.coordinates.normalized(measured)[None],
                                    self.coordinates.normalized(last_sent)[None], start=k, count=1,
                                    feedback_control=feedback_control)
        return self.coordinates.physical(result[0, 0])
