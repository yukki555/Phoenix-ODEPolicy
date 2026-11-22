"""Train an action-token ODE head from verified kitchen shards with torchrun."""
import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .cached_windows import CachedWindows
from .factory import build_head
from .recipe import apply_learning_rate, legacy_config, resolve_recipe
from .runner import save_new


class CachedPathLoss(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, tokens, measured, initial, targets, use_latest):
        observed = torch.where(use_latest[:, None, None], measured, measured[:, :1])
        return self.head.path_loss(tokens, observed, initial, targets, feedback=True)


def sample_update(seed, step, windows, batch, probability):
    """The same global draw is partitioned among ranks and is reproducible at any step."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, step)))
    indices = rng.integers(windows, size=batch, dtype=np.int64)
    return indices, rng.random(batch) < probability


def gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def parameter_digest(parameters):
    digest = hashlib.sha256()
    for name, tensor in parameters:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_equal(parameters, label):
    hashes = gather(parameter_digest(parameters))
    if len(set(hashes)) != 1:
        raise RuntimeError(f"Ranks disagree on {label}: {hashes}")
    return hashes[0]


def train(args, rank, world, device):
    recipe = resolve_recipe(args.recipe)
    cfg = recipe["training"]
    if (recipe["architecture"] != "action_token_ode" or cfg["objective"] != "action_path_mse"
            or cfg["observation_schedule"] != "mix_block_start_and_per_tick" or cfg["trainable"] != "head"):
        raise ValueError("Expected an action-token head with mixed per-window feedback and action path supervision")
    seed, batch = cfg["seed"], cfg["batch_size"]
    stop = cfg["steps"] if args.stop_at is None else args.stop_at
    if not 1 <= stop <= cfg["steps"] or batch % world:
        raise ValueError("Invalid stopping step or global batch not divisible by world size")
    if args.log_every < 1 or args.checkpoint_every < 1:
        raise ValueError("Logging and checkpoint intervals must be positive")
    local_batch = batch // world
    rank_rows = slice(rank * local_batch, (rank + 1) * local_batch)
    saved = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    evidence = getattr(args, "cache_evidence", "auto")
    if saved is not None and evidence == "auto":
        # Keep the original proof when a full verification is published after a receipt-based training start.
        evidence = "release" if saved["cache_training"].get("proof_kind") == "shard_receipts" else "coverage"
    cache = CachedWindows(args.cache, recipe, evidence=evidence)
    torch.manual_seed(seed)
    head = build_head(recipe, token_width=cache.token_width, device=device).train()
    nominal_hash = parameter_digest((n, p) for n, p in head.named_parameters() if not n.startswith("history_feedback."))
    wrapped = CachedPathLoss(head)
    if world > 1:
        wrapped = DistributedDataParallel(
            wrapped, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, static_graph=True,
        )
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"])
    checkpoint_contract = dict(version=1, manifest=cache.manifest_sha256, proof_kind=cache.proof_kind,
                               global_batch=batch, world_size=world, device_type=device.type, seed=seed)
    checkpoint_contract[cache.evidence_name] = cache.evidence_sha256
    restored_step = 0
    counts = dict(fixed=0, feedback=0)
    sample_hash = bytes(32)
    if saved is not None:
        saved_contract = dict(saved["cache_training"])
        if "proof_kind" not in saved_contract and "coverage" in saved_contract:
            saved_contract["proof_kind"] = "full_verification"
        if (saved_contract != checkpoint_contract or saved["recipe"] != recipe
                or saved["identity"] != cache.identity or not torch.equal(saved["bounds"], cache.bounds)):
            raise ValueError("Checkpoint belongs to a different recipe, verified cache, parent or training layout")
        restored_step = saved["steps"]
        if not 0 <= restored_step < stop:
            raise ValueError("The requested stopping step must follow the resume checkpoint")
        head.load_state_dict(saved["head"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        counts = dict(saved["training_windows"])
        if counts["fixed"] + counts["feedback"] != restored_step * batch:
            raise ValueError("Inconsistent sampling counters in resume checkpoint")
        sample_hash = bytes.fromhex(saved["sample_sha256"])
        state = saved["rng_by_rank"][rank]
        torch.set_rng_state(state["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda"], device)

    def emit(event, **fields):
        if rank == 0:
            line = json.dumps(dict(event=event, **fields), allow_nan=False)
            print(line, flush=True)
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(line + "\n")

    directory = args.output / "checkpoints"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(directory.glob("step-*.pt"))
        if existing and (not args.resume or existing[-1].resolve() != args.resume.resolve()):
            raise FileExistsError("Use an empty output or resume from its latest checkpoint")
    if world > 1:
        dist.barrier()
    emit("start", architecture=recipe["architecture"], conditioning=recipe.get("conditioning"),
         world_size=world, global_batch=batch, per_rank_batch=local_batch, resumed_step=restored_step,
         stop_at=stop, nominal_initial_sha256=nominal_hash, cache_training=checkpoint_contract,
         source_windows=len(cache), torch_version=str(torch.__version__))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    interval_losses = []
    for step in range(restored_step + 1, stop + 1):
        indices, latest = sample_update(seed, step, len(cache), batch, cfg["per_tick_probability"])
        sample_hash = hashlib.sha256(sample_hash + indices.astype("<i8", copy=False).tobytes() + latest.tobytes()).digest()
        counts["feedback"] += int(latest.sum())
        counts["fixed"] += batch - int(latest.sum())
        tokens, measured, initial, targets = cache.batch(indices[rank_rows], device)
        mode = torch.from_numpy(latest[rank_rows]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = wrapped(tokens, measured, initial, targets, mode)
        loss.backward()
        # All gradients have been averaged by DDP here. Fail before saving a nonfinite update.
        norm = nn.utils.clip_grad_norm_(head.parameters(), float("inf"), error_if_nonfinite=True)
        global_loss = loss.detach().clone()
        if world > 1:
            dist.all_reduce(global_loss)
            global_loss /= world
        loss_value = global_loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Nonfinite path loss at step {step}")
        checking = args.verify_ddp and step in (restored_step + 1, stop)
        if checking:
            gradient_hash = verify_equal(((n, p.grad) for n, p in head.named_parameters()), "gradients")
        lr = apply_learning_rate(optimizer, cfg, step)
        optimizer.step()
        if checking:
            weight_hash = verify_equal(head.named_parameters(), "updated weights")
            emit("sync_check", step=step, world_size=world, grad_sha256=gradient_hash, head_sha256=weight_hash)
        interval_losses.append(loss_value)
        checkpoint = step == stop or step in args.checkpoint_at or step % args.checkpoint_every == 0
        if step % args.log_every == 0 or checkpoint:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            peak = torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.
            stats = torch.tensor([elapsed, peak], device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            seconds, peak = stats.tolist()
            updates = len(interval_losses)
            emit("train", step=step, global_batch=batch, world_size=world, loss=loss_value, lr=lr,
                 mean_loss=sum(interval_losses) / updates, grad_norm=norm.item(), timed_updates=updates,
                 windows_per_second=updates * batch / seconds, mean_update_s=seconds / updates,
                 peak_reserved_gib=peak, sample_sha256=sample_hash.hex())
            interval_losses.clear()
        if checkpoint:
            rng = {"cpu": torch.get_rng_state()}
            if device.type == "cuda":
                rng["cuda"] = torch.cuda.get_rng_state(device)
            states = gather(rng)
            path = directory / f"step-{step:06d}.pt"
            if rank == 0:
                save_new(dict(schema_version=2, architecture=recipe["architecture"], family=head.family,
                              recipe=recipe, config=legacy_config(recipe), identity=cache.identity,
                              bounds=cache.bounds, source=cache.source, prior_affine=None,
                              head=head.state_dict(), optimizer=optimizer.state_dict(),
                              steps=step, batch_size=batch, training_windows=counts,
                              sample_sha256=sample_hash.hex(), cache_training=checkpoint_contract,
                              rng_by_rank=states, last_loss=loss_value), path)
            if world > 1:
                dist.barrier()
            emit("checkpoint", step=step, path=str(path), sample_sha256=sample_hash.hex())
        if not interval_losses:
            started = time.perf_counter()
    emit("complete", steps=stop, training_windows=counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-evidence", choices=("auto", "coverage", "release"), default="auto",
                        help="Pin a published report, or prefer full verification for a new run and reuse the original proof on resume")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-at", type=int, help="Absolute final optimizer step for this invocation")
    parser.add_argument("--checkpoint-at", type=int, nargs="*", default=[1250, 5000])
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--verify-ddp", action="store_true", help="Check all ranks' gradients and updated weights at the first/last update")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / ".writer.lock").open("a") if rank == 0 else nullcontext()
        with lock as handle:
            if rank == 0:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train(args, rank, world, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
