"""Reference-space Euler integration. Inputs share a 14-D action normalization."""
from collections import deque

import torch
from torch import nn

from .architectures import ActionTokens, architecture_type
from .joint_feedback import JointHistoryFeedback


class VelocityField(nn.Sequential):
    def forward(self, inputs, *, hidden_shift=None):
        if hidden_shift is None:
            return super().forward(inputs)
        for i, layer in enumerate(self):
            inputs = layer(inputs)
            if i == 0:
                inputs = inputs + hidden_shift
        return inputs


class ODEHead(nn.Module):
    def __init__(self, family, token_width, *, horizon=32, hz=30, hidden_dim=512, context_dim=128,
                 latent_feedback=False, history_feedback=None):
        super().__init__()
        if horizon < 2 or hz <= 0:
            raise ValueError("Invalid integration horizon or control rate")
        self.architecture = architecture_type(family)
        self.family, self.horizon, self.hz = self.architecture.legacy, horizon, hz
        self.token_reduction = "none"
        self.architecture.init_reader(self, token_width, context_dim)
        self.field = VelocityField(
            nn.Linear(context_dim + 14 * 2 + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 14),
        )
        if self.architecture.integrates == "residual_to_mapped_plan":
            nn.init.zeros_(self.field[-1].weight)
            nn.init.zeros_(self.field[-1].bias)
        self.history_feedback = None
        if history_feedback is not None:
            if latent_feedback or not isinstance(self.architecture, ActionTokens):
                raise ValueError("Joint history feedback requires an action-token head without a latent adapter")
            self.history_feedback = JointHistoryFeedback(context_dim, hidden_dim, **history_feedback)
        self.feedback_adapter = None
        if latent_feedback:
            if not isinstance(self.architecture, ActionTokens):
                raise ValueError("Latent feedback requires action tokens")
            self.requires_grad_(False)
            self.feedback_adapter = nn.Sequential(
                nn.Linear(context_dim + 3 * 14 + 1, context_dim), nn.SiLU(),
                nn.Linear(context_dim, context_dim),
            )
            nn.init.zeros_(self.feedback_adapter[-1].weight)
            nn.init.zeros_(self.feedback_adapter[-1].bias)

    def prepare(self, prior):
        if prior.ndim != 3:
            raise ValueError("Expected a batched condition sequence")
        return self.architecture.prepare(self, prior)

    def integrate(self, context, prior, block_start, measured, reference, *, start=0, stop=None,
                  initial_q=None, feedback_control=None, history=None):
        """Return targets at ticks start+1 through stop, using fixed [B,14] or causal per-tick [B,H,14] measurements."""
        stop = self.horizon if stop is None else stop
        if not 0 <= start < stop <= self.horizon:
            raise ValueError("Expected a nonempty interval within the original block")
        batch = context.shape[0]
        if reference.shape != (batch, 14) or measured.shape not in {(batch, 14), (batch, self.horizon, 14)}:
            raise ValueError("Expected reference [B,14] and measured [B,14] or [B,H,14]")
        reader = self.history_feedback
        if reader is not None or self.feedback_adapter is not None:
            if initial_q is None or initial_q.shape != reference.shape:
                raise ValueError("Feedback conditioning requires the measured block-start q0")
        controls = ()
        if reader is not None:
            controls = ("feedback_off", "one_tick_delayed")
            if history is None:
                history = reader.new_history()
            if len(history) != min(start, reader.history_length) and feedback_control != "feedback_off":
                raise ValueError("A suffix needs the preceding same-tick measurement/reference history")
        elif self.feedback_adapter is not None:
            controls = ("adapter_off", "direct_c1")
        if feedback_control is not None and feedback_control not in controls:
            raise ValueError("Unknown or incompatible latent feedback control")
        outputs = []
        for k in range(start, stop):
            state = self.architecture.to_state(prior, block_start, reference, k)
            condition = self.architecture.at_tick(self, context, k)
            phase = state.new_full((state.shape[0], 1), k / self.horizon)
            q = measured[:, k] if measured.ndim == 3 else measured
            if self.feedback_adapter is not None and feedback_control != "direct_c1":
                if feedback_control != "adapter_off":
                    features = torch.cat((condition, reference, q, reference - q, phase), dim=-1)
                    condition = condition + self.feedback_adapter(features)
                q = initial_q
            if reader is None:
                speed = self.field(torch.cat((condition, q, state, phase), dim=-1))
            else:
                record = reader.observation(q, reference)
                feedback = reader.read(condition, self.position[k], history, record, k, feedback_control)
                nominal = torch.cat((condition, initial_q, state, phase), dim=-1)
                speed = reader.velocity(self.field, nominal, feedback)
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
        history = self.history_feedback.new_history() if self.history_feedback is not None else None
        measured = states if feedback else states[:, k]
        if history is not None and not feedback:
            measured = states[:, 0]
        if k and model_boundary:
            with torch.no_grad():
                prefix_measurements = measured if history is not None else states[:, 0]
                initial = self.integrate(context, prior, reference, prefix_measurements, reference,
                                         stop=k, initial_q=states[:, 0], history=history)[:, -1]
        elif k:
            initial = targets[:, k-1]
            if history is not None:
                for i in range(max(0, k - history.maxlen), k):
                    q = states[:, i] if feedback else states[:, 0]
                    accepted = targets[:, i-1] if i else reference
                    history.append(self.history_feedback.observation(q, accepted))
        prediction = self.integrate(context, prior, reference, measured, initial, start=k, initial_q=states[:, 0],
                                    history=history)
        return nn.functional.mse_loss(prediction, targets[:, k:])


class ActionBlock:
    """Caller-owned block. Physical reference advances only when the caller sends a target."""
    def __init__(self, head, prior, block_start, *, initial_q=None):
        self.head = head
        self.prior = prior
        self.block_start = block_start
        self.context = head.prepare(prior)
        self.initial_q = initial_q
        # The extra record makes retrying the last tick equivalent to its first attempt.
        self._observed = deque(maxlen=head.history_feedback.history_length + 1) if head.history_feedback else None

    @torch.no_grad()
    def targets(self, measured, last_sent, *, start, count, feedback_control=None):
        """A multi-tick preview never advances the online history. A retry replaces the pending observation."""
        history = None
        remember = False
        if self._observed is not None:
            history = self.head.history_feedback.new_history()
            last = self._observed[-1][0] if self._observed else -1
            if start in (last, last + 1):
                history.extend(record for tick, record in self._observed if tick < start)
                remember = count == 1
            elif not (start == 0 and count > 1) and feedback_control != "feedback_off":
                raise ValueError("Joint history requires consecutive ticks, a same-tick retry, or a block-start preview")
        result = self.head.integrate(self.context, self.prior, self.block_start,
                                     measured, last_sent, start=start, stop=start + count, initial_q=self.initial_q,
                                     feedback_control=feedback_control, history=history)
        if remember:
            if self._observed and self._observed[-1][0] == start:
                self._observed.pop()
            self._observed.append((start, history[-1]))
        return result
