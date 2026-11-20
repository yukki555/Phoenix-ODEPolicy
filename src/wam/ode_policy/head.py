"""Reference-space Euler integration. Inputs share a 14-D action normalization."""
import torch
from torch import nn

from .architectures import architecture_type


class ODEHead(nn.Module):
    def __init__(self, family, token_width, *, horizon=32, hz=30, hidden_dim=512, context_dim=128):
        super().__init__()
        if horizon < 2 or hz <= 0:
            raise ValueError("Invalid integration horizon or control rate")
        self.architecture = architecture_type(family)
        self.family, self.horizon, self.hz = self.architecture.legacy, horizon, hz
        self.token_reduction = "none"
        self.architecture.init_reader(self, token_width, context_dim)
        self.field = nn.Sequential(
            nn.Linear(context_dim + 14 * 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 14),
        )
        if self.architecture.integrates == "residual_to_mapped_plan":
            nn.init.zeros_(self.field[-1].weight)
            nn.init.zeros_(self.field[-1].bias)

    def prepare(self, prior):
        if prior.ndim != 3:
            raise ValueError("Expected a batched condition sequence")
        return self.architecture.prepare(self, prior)

    def integrate(self, context, prior, block_start, measured, reference, *, start=0, stop=None):
        """Return targets at ticks start+1 through stop, using fixed [B,14] or causal per-tick [B,H,14] measurements."""
        stop = self.horizon if stop is None else stop
        if not 0 <= start < stop <= self.horizon:
            raise ValueError("Expected a nonempty interval within the original block")
        batch = context.shape[0]
        if reference.shape != (batch, 14) or measured.shape not in {(batch, 14), (batch, self.horizon, 14)}:
            raise ValueError("Expected reference [B,14] and measured [B,14] or [B,H,14]")
        outputs = []
        for k in range(start, stop):
            state = self.architecture.to_state(prior, block_start, reference, k)
            condition = self.architecture.at_tick(self, context, k)
            phase = state.new_full((state.shape[0], 1), k / self.horizon)
            q = measured[:, k] if measured.ndim == 3 else measured
            speed = self.field(torch.cat((condition, q, state, phase), dim=-1))
            updated = state + speed / self.hz
            reference = self.architecture.to_reference(prior, updated, k)
            outputs.append(reference)
        return torch.stack(outputs, dim=1)

    def path_loss(self, prior, states, reference, targets, *, boundary=None, model_boundary=True, feedback=False):
        """states[:,k] is available before target k; reference is the block-start boundary."""
        context = self.prepare(prior)
        k = 0 if boundary is None else boundary
        if not 0 <= k < self.horizon:
            raise ValueError("Training boundary outside the block")
        initial = reference
        if k and model_boundary:
            with torch.no_grad():
                initial = self.integrate(context, prior, reference, states[:, 0], reference, stop=k)[:, -1]
        elif k:
            initial = targets[:, k-1]
        measured = states if feedback else states[:, k]
        prediction = self.integrate(context, prior, reference, measured, initial, start=k)
        return nn.functional.mse_loss(prediction, targets[:, k:])


class ActionBlock:
    """Caller-owned block. Physical reference advances only when the caller sends a target."""
    def __init__(self, head, prior, block_start):
        self.head = head
        self.prior = prior
        self.block_start = block_start
        self.context = head.prepare(prior)

    @torch.no_grad()
    def targets(self, measured, last_sent, *, start, count):
        return self.head.integrate(self.context, self.prior, self.block_start,
                                   measured, last_sent, start=start, stop=start + count)
