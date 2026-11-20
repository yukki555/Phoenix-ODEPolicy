"""Offline motion diagnostics in physical joint coordinates."""
import numpy as np
import torch

from .calibration import trajectory_metrics
from .kitchen import ARMS, GRIPPERS


def dominant_turn(path):
    """Interior peak or trough with the largest excursion relative to both endpoints."""
    from_start = path[1:-1] - path[0]
    from_end = path[1:-1] - path[-1]
    peaks = torch.minimum(from_start, from_end)
    troughs = torch.minimum(-from_start, -from_end)
    size, idx = torch.maximum(peaks, troughs).max(dim=0)
    sign = (peaks >= troughs).gather(0, idx[None])[0]
    return size.clamp_min(0), idx + 1, sign


def motion_metrics(pred, target, states, bounds):
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite replay prediction")
    errors = (pred - target).square()
    # The task labels are q[k+1]. Increment RMSE equals position RMSE;
    # direction and predicted movement size distinguish holding from moving.
    delta = (target - states)[:, ARMS]
    proposed = (pred - states)[:, ARMS]
    threshold = (0.01 * (bounds[1] - bounds[0])[ARMS]).clamp_min(1e-6)
    moving = delta.abs() > threshold
    actual_path = torch.cat((states[:1], target))[:, ARMS]
    pred_path = torch.cat((states[:1], pred))[:, ARMS]
    actual_range = actual_path.amax(0) - actual_path.amin(0)
    pred_range = pred_path.amax(0) - pred_path.amin(0)
    active = actual_range > threshold
    expected_size, expected_tick, expected_sign = dominant_turn(actual_path)
    pred_size, pred_tick, pred_sign = dominant_turn(pred_path)
    expected = expected_size > threshold
    detected = pred_size > threshold
    matched = expected & detected & (expected_sign == pred_sign)
    crosses = trajectory_metrics(pred, target, states[0], bounds)
    return dict(arm_mse=errors[:, ARMS].mean().item(), gripper_mse=errors[:, GRIPPERS].mean().item(),
                moving_steps=int(moving.sum()), correct_direction=int(((delta*proposed > 0) & moving).sum()),
                moving_expert_abs_rad=delta[moving].abs().sum().item(),
                moving_pred_abs_rad=proposed[moving].abs().sum().item(),
                moving_joints=int(active.sum()), expert_range_rad=actual_range[active].sum().item(),
                pred_range_rad=pred_range[active].sum().item(),
                expected_turns=int(expected.sum()), matched_turns=int(matched.sum()),
                extra_turns=int((detected & ~matched).sum()),
                turn_timing_ticks=int((pred_tick-expected_tick).abs()[matched].sum()),
                matched_crossings=len(crosses["gripper_timing_abs_ticks"]),
                crossing_timing_ticks=sum(crosses["gripper_timing_abs_ticks"]),
                missed_crossings=crosses["missed_gripper_crossings"], extra_crossings=crosses["extra_gripper_crossings"])


def summarize(episodes):
    """Equal episode weights for RMSE; event rates pooled over eligible observations."""
    rows = [row for episode in episodes for row in episode]
    def total(key):
        return sum(row[key] for row in rows)
    def mean(key, count):
        return total(key)/total(count) if total(count) else None
    def rmse(key):
        return float(np.sqrt(np.mean([np.mean([r[key] for r in ep]) for ep in episodes])))
    return dict(episodes=len(episodes), windows=len(rows), arm_rmse=rmse("arm_mse"),
                gripper_rmse=rmse("gripper_mse"), moving_steps=total("moving_steps"),
                direction_accuracy=mean("correct_direction", "moving_steps"),
                mean_expert_step_rad=mean("moving_expert_abs_rad", "moving_steps"),
                mean_pred_step_rad=mean("moving_pred_abs_rad", "moving_steps"), moving_joints=total("moving_joints"),
                mean_expert_range_rad=mean("expert_range_rad", "moving_joints"),
                mean_pred_range_rad=mean("pred_range_rad", "moving_joints"), expected_turns=total("expected_turns"),
                matched_turns=total("matched_turns"), extra_turns=total("extra_turns"),
                turn_timing_mae_ticks=mean("turn_timing_ticks", "matched_turns"),
                matched_crossings=total("matched_crossings"), missed_crossings=total("missed_crossings"),
                extra_crossings=total("extra_crossings"), gripper_timing_mae_ticks=mean("crossing_timing_ticks", "matched_crossings"))
