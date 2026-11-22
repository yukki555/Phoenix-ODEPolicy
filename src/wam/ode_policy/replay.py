"""Paired offline executions and feedback diagnostics for a single action block."""
import time

import torch

from .head import ActionBlock
from .kitchen import ARMS, GRIPPERS
from .recipe import HISTORY_CONTROLS, LATENT_CONTROLS, MODE_ALIASES


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def replay(mode, head, prior, reference, states, tick_times=None, *, first_ready=None, coordinates=None):
    """Return normalized targets by default, or CPU physical targets for timed evaluation."""
    started = time.perf_counter()
    def output(target, *, first):
        value = coordinates.physical(target).cpu() if coordinates is not None else target
        if first and first_ready is not None:
            sync(value.device)
            first_ready.append((time.perf_counter() - started)*1000)
        return value

    mode = MODE_ALIASES.get(mode, mode)
    if mode == "hold_latest":
        return output(states, first=True)
    if mode == "mapped_action_plan":
        return output(prior, first=True)
    block = ActionBlock(head, prior, reference, initial_q=states[:, 0])
    control = mode if mode in (*LATENT_CONTROLS, *HISTORY_CONTROLS) else None
    if mode not in {"b_stale", "latest_joint_each_tick", "one_tick_delayed", "direct_c1"}:
        return output(block.targets(states[:, 0], reference, start=0, count=head.horizon,
                                    feedback_control=control), first=True)
    outputs = []
    for k in range(head.horizon):
        if tick_times is not None:
            sync(reference.device)
            tick_start = time.perf_counter()
        observed = states[:, 0] if mode == "b_stale" else states[:, k]
        reference = block.targets(observed, reference, start=k, count=1, feedback_control=control)[:, 0]
        outputs.append(output(reference, first=k == 0))
        if tick_times is not None:
            sync(reference.device)
            tick_times.append((time.perf_counter() - tick_start) * 1000)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def feedback_action_probe(head, prior, reference, states, coordinates, targets, *, previous=None):
    """Score one window while every mode sees the same reference at every tick.

    previous[:, k] is the reference in effect before observing states[:, k].
    Supply the same reference sequence when comparing separately trained heads.
    """
    initial_q = states[:, 0]
    if previous is None:
        off = "adapter_off" if head.feedback_adapter is not None else "feedback_off"
        nominal = replay(off, head, prior, reference, states)
        previous = torch.cat((reference[:, None], nominal[:, :-1]), dim=1)
    if previous.shape != states.shape or states.shape[0] != 1:
        raise ValueError("Expected one window and a shared [1,H,14] reference history")
    paths = {}
    modes = [("fixed", initial_q[:, None].expand_as(states), None), ("latest", states, None)]
    if head.history_feedback is not None:
        modes.append(("delayed", states, "one_tick_delayed"))
    for name, measurements, control in modes:
        block = ActionBlock(head, prior, reference, initial_q=initial_q)
        one_steps = [block.targets(measurements[:, k], previous[:, k], start=k, count=1, feedback_control=control)
                     for k in range(head.horizon)]
        paths[name] = coordinates.physical(torch.cat(one_steps, dim=1))[0].cpu()
    result = {}
    for label, indices in (("arm", ARMS), ("gripper", GRIPPERS)):
        def rms(delta):
            return delta[:, indices].square().mean().sqrt().item()
        result[label] = dict(target_change_rms=rms(paths["latest"] - paths["fixed"]),
                             fixed_rmse=rms(paths["fixed"] - targets),
                             latest_rmse=rms(paths["latest"] - targets))
        if "delayed" in paths:
            result[label].update(delayed_rmse=rms(paths["delayed"] - targets),
                                 latest_vs_delayed_change_rms=rms(paths["latest"] - paths["delayed"]))
    return result
