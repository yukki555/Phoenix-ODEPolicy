"""Reference-space Euler integration. Inputs share a 14-D action normalization."""
import torch
from torch import nn


class ODEHead(nn.Module):
    def __init__(self, family, token_width, *, horizon=32, hz=30, hidden_dim=512, context_dim=128):
        super().__init__()
        if family not in {"ab", "c1", "c2"} or horizon < 2 or hz <= 0:
            raise ValueError("Invalid ODE family, horizon or control rate")
        self.family, self.horizon, self.hz = family, horizon, hz
        self.project = nn.Linear(14 if family == "ab" else token_width, context_dim)
        if family == "c2":
            self.query = nn.Parameter(torch.randn(1, 1, token_width) * 0.02)
        else:
            self.position = nn.Parameter(torch.randn(horizon, context_dim) * 0.02)
            self.reader = nn.MultiheadAttention(context_dim, 1, batch_first=True)
        self.field = nn.Sequential(
            nn.Linear(context_dim + 14 * 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 14),
        )
        if family == "ab":
            nn.init.zeros_(self.field[-1].weight)
            nn.init.zeros_(self.field[-1].bias)

    def prepare(self, prior):
        length = 1 if self.family == "c2" else self.horizon
        if prior.ndim != 3 or prior.shape[1:] != (length, self.project.in_features):
            raise ValueError("Wrong prior sequence shape for this ODE family")
        context = self.project(prior.float())
        return context if self.family == "c2" else context + self.position

    def integrate(self, context, prior, block_start, measured, reference, *, start=0, stop=None):
        """Return targets at ticks start+1 through stop, holding measured fixed."""
        stop = self.horizon if stop is None else stop
        if not 0 <= start < stop <= self.horizon:
            raise ValueError("Expected a nonempty interval within the original block")
        if measured.shape != reference.shape or measured.shape != (context.shape[0], 14):
            raise ValueError("Expected independent measured and reference arrays [B,14]")
        outputs = []
        for k in range(start, stop):
            state = reference
            if self.family == "ab":
                boundary = prior[:, k-1] if k else block_start
                state = reference - boundary
            if self.family == "c2":
                condition = context[:, 0]
            else:
                query = context[:, k:k+1]
                read, _ = self.reader(query, context, context, need_weights=False)
                condition = (query + read)[:, 0]
            phase = state.new_full((state.shape[0], 1), k / self.horizon)
            speed = self.field(torch.cat((condition, measured, state, phase), dim=-1))
            updated = state + speed / self.hz
            reference = prior[:, k] + updated if self.family == "ab" else updated
            outputs.append(reference)
        return torch.stack(outputs, dim=1)

    def path_loss(self, prior, states, reference, targets, *, boundary=None, model_boundary=True):
        """states[:,k] is available before target k; reference is the block-start boundary."""
        context = self.prepare(prior)
        k = 0 if boundary is None else boundary
        if not 0 <= k < self.horizon or (k and self.family != "ab"):
            raise ValueError("Only A/B train on interior boundaries")
        initial = reference
        if k and model_boundary:
            with torch.no_grad():
                initial = self.integrate(context, prior, reference, states[:, 0], reference, stop=k)[:, -1]
        elif k:
            initial = targets[:, k-1]
        prediction = self.integrate(context, prior, reference, states[:, k], initial, start=k)
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
