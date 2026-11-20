"""The three Phoenix condition readers and their reference-space dynamics."""
import torch
from torch import nn


class AbsoluteReference:
    def map_condition(self, raw, coordinates):
        return raw

    def to_state(self, prior, start, reference, k):
        return reference

    def to_reference(self, prior, updated, k):
        return updated


class SequenceReader(AbsoluteReference):
    def init_reader(self, head, width, dim):
        head.project = nn.Linear(width, dim)
        head.position = nn.Parameter(torch.randn(head.horizon, dim) * 0.02)
        head.reader = nn.MultiheadAttention(dim, 1, batch_first=True)

    def prepare(self, head, prior):
        if prior.shape[1:] != (head.horizon, head.project.in_features):
            raise ValueError("Expected one condition per tick")
        if head.token_reduction == "temporal_mean_repeat":
            prior = prior.mean(dim=1, keepdim=True).expand_as(prior)
        return head.project(prior.float()) + head.position

    def at_tick(self, head, context, k):
        query = context[:, k:k+1]
        read, _ = head.reader(query, context, context, need_weights=False)
        return (query + read)[:, 0]


class ActionTokens(SequenceReader):
    name = "action_token_ode"
    legacy = "c1"
    reads = "final_preprojection_action_tokens"
    integrates = "absolute_joint_reference"

    def read(self, model, head, obs, active, steps, seed):
        from .phoenix import sample_block
        return sample_block(model, head, obs, steps, seed, tokens=True)["action_tokens"].float()


class ActionPlanResidual(SequenceReader):
    name = "action_plan_residual_ode"
    legacy = "ab"
    reads = "final_action_chunk"
    integrates = "residual_to_mapped_plan"

    def read(self, model, head, obs, active, steps, seed):
        from .phoenix import sample_block
        result = sample_block(model, head, obs, steps, seed, tokens=False)
        return result["action"][:, active].unsqueeze(0).to(model.device)

    def map_condition(self, raw, coordinates):
        return coordinates.slots(raw)

    def init_reader(self, head, width, dim):
        super().init_reader(head, 14, dim)

    def to_state(self, prior, start, reference, k):
        boundary = prior[:, k-1] if k else start
        return reference - boundary

    def to_reference(self, prior, updated, k):
        return prior[:, k] + updated


class ActionQuery(AbsoluteReference):
    name = "action_query_ode"
    legacy = "c2"
    reads = "single_current_query"
    integrates = "absolute_joint_reference"

    def read(self, model, head, obs, active, steps, seed):
        from .phoenix import read_current_query
        return read_current_query(model, head.query, obs).float()

    def init_reader(self, head, width, dim):
        head.project = nn.Linear(width, dim)
        head.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)

    def prepare(self, head, prior):
        if prior.shape[1:] != (1, head.project.in_features):
            raise ValueError("Expected a single query condition")
        return head.project(prior.float())

    def at_tick(self, head, context, k):
        return context[:, 0]


ARCHITECTURES = {a.name: a for a in (ActionPlanResidual(), ActionTokens(), ActionQuery())}
LEGACY_NAMES = {a.legacy: a.name for a in ARCHITECTURES.values()}


def architecture_type(name):
    try:
        return ARCHITECTURES[LEGACY_NAMES.get(name, name)]
    except KeyError:
        raise ValueError(f"Unknown ODE architecture: {name}") from None
