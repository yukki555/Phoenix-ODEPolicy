"""Read same-tick measurements and tracking errors from a causal control history."""
from collections import deque

import torch
from torch import nn


def validate_history_feedback(*, history_length, query, injection):
    if type(history_length) is not int or not 1 <= history_length <= 4:
        raise ValueError("Joint feedback requires a history length between 1 and 4")
    if query not in {"action_feature", "block_step"}:
        raise ValueError(f"Unknown joint history query: {query}")
    if injection not in {"velocity_field_condition", "action_velocity_residual"}:
        raise ValueError(f"Unknown joint history injection: {injection}")


class JointHistoryFeedback(nn.Module):
    def __init__(self, context_dim, hidden_dim, *, history_length, query, injection):
        super().__init__()
        validate_history_feedback(history_length=history_length, query=query, injection=injection)

        self.history_length = history_length
        self.query = query
        self.injection = injection
        self.encode = nn.Linear(28, context_dim)
        # A shared slope encodes true sample age, including the extra tick in delayed replay.
        self.age_step = nn.Parameter(torch.randn(context_dim) * 0.02)
        self.register_buffer("ages", torch.arange(history_length, -1, -1), persistent=False)
        self.reader = nn.MultiheadAttention(context_dim, 1, batch_first=True)
        if injection == "velocity_field_condition":
            self.condition_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.residual_speed = nn.Sequential(
                nn.Linear(context_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, 14),
            )

    def new_history(self):
        return deque(maxlen=self.history_length)

    @staticmethod
    def observation(measured, reference):
        return torch.cat((measured, reference - measured), dim=-1)

    def read(self, action_feature, step_embedding, history, observation, tick, control):
        """Append this tick exactly once; delay reads only previously available pairs."""
        delayed = control == "one_tick_delayed" and tick > 0
        if not delayed:
            history.append(observation)
        if control == "feedback_off":
            return None

        samples = torch.stack(tuple(history), dim=1)
        ages = self.ages[-len(history)-1:-1] if delayed else self.ages[-len(history):]
        memory = self.encode(samples) + ages[:, None] * self.age_step
        query = action_feature if self.query == "action_feature" else step_embedding.expand_as(action_feature)
        read, _ = self.reader(query[:, None], memory, memory, need_weights=False)
        if delayed:
            history.append(observation)
        return read[:, 0]

    def velocity(self, field, nominal_inputs, feedback):
        if feedback is None:
            return field(nominal_inputs)
        if self.injection == "action_velocity_residual":
            return field(nominal_inputs) + self.residual_speed(feedback)
        # W_x x + W_c c + b is a linear layer on [x, c]. Keeping W_x separate
        # preserves identical nominal parameter initialization across the four variants.
        return field(nominal_inputs, hidden_shift=self.condition_projection(feedback))
