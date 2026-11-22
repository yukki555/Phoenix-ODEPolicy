"""Paired kitchen trials. The robot controller owns scheduling, transport and task scoring."""
import time

import numpy as np
import torch

from .architectures import architecture_type
from .recipe import EXECUTIONS, MODE_ALIASES


class KitchenTrial:
    """One episode. Supply a new instance after any reset or control handoff."""

    def __init__(self, policy, task, mode, pair_id, head_id, *, last_sent=None):
        self.execution = MODE_ALIASES.get(mode, mode)
        if mode not in {"fixed", "feedback", *EXECUTIONS}:
            raise ValueError("Choose block-start or per-tick execution")
        self.policy, self.task, self.mode = policy, task, mode
        self.pair_id, self.head_id = pair_id, head_id
        self.last_sent = last_sent
        self.block = self.chunk = None
        self.invalid = False
        self.k = 0
        self.tick_ms, self.first_target_ms = [], []

    def begin_block(self, rgb, measured, *, seed=None):
        if self.invalid:
            raise ValueError("Dispatch failed; reconcile controller state before starting a new trial")
        if self.block is not None:
            raise ValueError("Previous block still active")
        self.started = time.perf_counter()
        block = self.policy.begin_block(self.task, rgb, measured, self.last_sent, seed=seed)
        chunk = block.full_chunk().cpu() if self.execution == "block_start" else None
        if chunk is not None:
            self.first_target_ms.append((time.perf_counter()-self.started)*1000)
        if self.last_sent is None:
            self.last_sent = measured.detach().clone()
        self.block, self.chunk, self.k = block, chunk, 0

    def tick(self, read_state, send_target):
        """Run at a controller tick. send_target returns the actually accepted physical target."""
        if self.block is None:
            raise ValueError("No active block")
        start = time.perf_counter()
        target = (self.chunk[self.k] if self.execution == "block_start"
                  else self.block.next_target(self.k, read_state(), self.last_sent).cpu())
        if not target.isfinite().all():
            raise ValueError("Nonfinite physical target")
        ready = time.perf_counter()
        # A failed/ambiguous send must be reconciled by the controller before a new episode.
        try:
            accepted = send_target(target)
            completed = time.perf_counter()
            if not isinstance(accepted, torch.Tensor) or accepted.shape != (14,) or not accepted.isfinite().all():
                raise ValueError("send_target must acknowledge the accepted physical 14-D target")
        except Exception:
            self.block = self.chunk = None
            self.invalid = True
            raise
        self.tick_ms.append((completed-start)*1000)
        if self.k == 0 and self.execution == "latest_joint_each_tick":
            self.first_target_ms.append((ready-self.started)*1000)
        self.last_sent = accepted.detach().clone()
        self.k += 1
        if self.k == self.policy.head.horizon:
            self.block = self.chunk = None

    def result(self, success, stages):
        """Score only with the task's prespecified stage names; retain raw latency samples."""
        if type(success) is not bool or not stages or any(type(v) is not bool for v in stages.values()):
            raise ValueError("Expected a boolean task outcome and named boolean stages")
        family = self.policy.head.family
        return dict(task=self.task, pair_id=self.pair_id, architecture=architecture_type(family).name,
                    family=family, head_id=self.head_id,
                    mode=self.mode, success=success, stages=dict(stages), hz=self.policy.head.hz,
                    tick_ms=list(self.tick_ms), first_target_ms=list(self.first_target_ms),
                    dispatch_failed=self.invalid)


def summarize_pairs(trials):
    """Aggregate completed pairs by task and head; refuse unmatched or mixed-protocol trials."""
    trials = list(trials)
    pairs = {}
    for trial in trials:
        key = trial["family"], trial["head_id"], trial["task"], trial["pair_id"]
        pair = pairs.setdefault(key, {})
        execution = MODE_ALIASES.get(trial["mode"], trial["mode"])
        if execution in pair:
            raise ValueError(f"Repeated trial mode for {key}")
        pair[execution] = trial
    groups = {}
    protocols = {}
    for (family, head_id, task, pair_id), pair in pairs.items():
        if set(pair) != set(EXECUTIONS):
            raise ValueError(f"Incomplete pair {pair_id}")
        for execution, trial in pair.items():
            group = (family, head_id, task)
            protocol = (frozenset(trial["stages"]), trial["hz"])
            if protocols.setdefault(group, protocol) != protocol:
                raise ValueError(f"Scoring or control rate changed within {task}")
            groups.setdefault((family, head_id, task, execution), []).append(trial)
    result = []
    for (family, head_id, task, execution), rows in groups.items():
        labels = {row["mode"] for row in rows}
        mode = labels.pop() if len(labels) == 1 else execution
        stages, hz = rows[0]["stages"].keys(), rows[0]["hz"]
        ticks = np.asarray([v for r in rows for v in r["tick_ms"]])
        first = [v for r in rows for v in r["first_target_ms"]]
        result.append(dict(architecture=architecture_type(family).name, family=family, head_id=head_id,
                           task=task, mode=mode, trials=len(rows),
                           successes=sum(r["success"] for r in rows), dispatch_failures=sum(r["dispatch_failed"] for r in rows),
                           stages={s: sum(r["stages"][s] for r in rows) for s in stages},
                           ticks=len(ticks), tick_p50_p95_ms=np.percentile(ticks, [50, 95]).tolist() if ticks.size else None,
                           deadline_miss_fraction=float(np.mean(ticks > 1000/hz)) if ticks.size else None,
                           first_target_p50_p95_ms=np.percentile(first, [50, 95]).tolist() if first else None))
    return result
