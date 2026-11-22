"""Read verified one-step Phoenix action tokens by global kitchen window index."""
import hashlib
import json
import mmap
import re
import resource
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml

from .calibration import load_source
from .kitchen import JointCoordinates
from .recipe import CONFIG_ROOT


RELEASE_CHECKS = {"exact_shard_membership", "receipt_ranges", "file_sizes", "receipt_sha256_format"}


class CachedWindows:
    """Train-only NPY shards backed by full verification or a receipt-based release."""

    def __init__(self, root, spec, *, evidence="auto"):
        self.root = Path(root)
        manifest_bytes = (self.root / "manifest.json").read_bytes()
        if evidence not in {"auto", "coverage", "release"}:
            raise ValueError(f"Unknown cache evidence: {evidence}")
        if evidence == "auto":
            evidence = "coverage"
            if not (self.root / "coverage.json").exists() and (self.root / "release.json").is_file():
                evidence = "release"
        self.evidence_name = evidence
        proof_bytes = (self.root / f"{evidence}.json").read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.evidence_sha256 = hashlib.sha256(proof_bytes).hexdigest()
        self.coverage_sha256 = self.evidence_sha256 if evidence == "coverage" else None
        self.manifest = manifest = json.loads(manifest_bytes)
        proof = json.loads(proof_bytes)
        count = manifest["window_count"]
        if (manifest["split"] != "train" or proof["split"] != "train" or proof["manifest_sha256"] != self.manifest_sha256
                or count <= 0 or proof["expected"] != count or proof["by_task"] != manifest["task_window_count"]
                or sum(proof["by_task"].values()) != count or proof["episodes"] != manifest["episode_count"]):
            raise ValueError("Complete train-split coverage tied to this cache manifest is required")
        if evidence == "coverage":
            self.proof_kind = "full_verification"
            if (proof.get("status") != "verified" or proof.get("verified") != count or proof.get("fraction") != 1
                    or "receipt_windows" in proof or proof.get("proof_kind", self.proof_kind) != self.proof_kind):
                raise ValueError("Expected a full verification coverage report")
        else:
            self.proof_kind = "shard_receipts"
            if (proof.get("status") != "released" or proof.get("proof_kind") != self.proof_kind
                    or proof.get("receipt_windows") != count or {"verified", "fraction"}.intersection(proof)
                    or not RELEASE_CHECKS.issubset(proof.get("checks", []))):
                raise ValueError("Expected a receipt-based release report with completed release checks")

        policy, parent = spec["policy"], spec["parent"]
        if (spec["architecture"] != "action_token_ode" or manifest["feature"] != spec["phoenix_read"]
                or manifest["denoise_steps"] != policy["denoise_steps"] or manifest["denoise_steps"] != 1
                or manifest["horizon"] != policy["horizon"] or manifest["hz"] != policy["hz"]
                or manifest["base_seed"] != 42 or manifest["seed_rule"] != "42 + split_window_index"):
            raise ValueError("Recipe and cached token generation settings differ")
        dataset = (CONFIG_ROOT / (spec["dataset"] + ".yaml")).read_bytes()
        config = yaml.safe_load(dataset)
        if (hashlib.sha256(dataset).hexdigest() != manifest["kitchen_config_sha256"]
                or config["sources"] != spec["sources"] or config["instructions"] != spec["instructions"]):
            raise ValueError("Recipe and cached dataset or task instructions differ")
        profile = Path(parent["state_stats_profile"])
        if (hashlib.sha256(profile.read_bytes()).hexdigest() != manifest["state_profile_sha256"]
                or parent["state_stats_group"] != manifest["state_stats_group"]):
            raise ValueError("Recipe and cached Phoenix input scales differ")
        self.identity = manifest["identity"]
        self.source = load_source(profile, parent["state_stats_group"], self.identity, include_action=False)

        self.bounds = torch.tensor(manifest["train_bounds"], dtype=torch.float32)
        if (self.bounds.shape != (2, 14) or not self.bounds.isfinite().all()
                or (self.bounds[1] < self.bounds[0]).any()):
            raise ValueError("Invalid kitchen normalization bounds")
        self._coordinates = JointCoordinates(self.bounds, "cpu")
        horizon, self.token_width = manifest["horizon"], 1024
        self._dtype = np.dtype([
            ("index", "<i8"), ("action_tokens", "<f4", (horizon, self.token_width)),
            ("measured_q", "<f4", (horizon, 14)), ("initial_reference", "<f4", (14,)),
            ("expert_action", "<f4", (horizon, 14)),
        ])
        if manifest["record_dtype"] != json.loads(json.dumps(self._dtype.descr)):
            raise ValueError("Unexpected token cache record layout")

        self.shard_rows = manifest["shard_rows"]
        self._files = proof["files"]
        if type(self.shard_rows) is not int or self.shard_rows <= 0:
            raise ValueError("Invalid cache shard size")
        ranges = range(0, count, self.shard_rows)
        if proof["shards"] != len(ranges) or len(self._files) != len(ranges):
            raise ValueError("Incomplete shard receipts")
        for start, receipt in zip(ranges, self._files):
            if (receipt["file"] != f"shards/{start:06d}.npy" or receipt["start"] != start
                    or receipt["stop"] != min(count, start + self.shard_rows)):
                raise ValueError("Shard receipts do not cover the original window indices")
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("sha256", ""))):
                raise ValueError(f"Missing or invalid SHA-256 in shard receipt: {receipt['file']}")
            if (self.root / receipt["file"]).stat().st_size != receipt["size_bytes"]:
                raise ValueError(f"Cache shard size differs from receipt: {receipt['file']}")
        if sum(r["size_bytes"] for r in self._files) != proof["total_bytes"]:
            raise ValueError("Inconsistent cache byte count")
        if evidence == "release":
            paths = {p.name for p in (self.root / "shards").glob("*.npy")}
            if paths != {Path(r["file"]).name for r in self._files}:
                raise ValueError("Published shard membership differs from the release receipts")

        # A typical Linux worker can retain every map. On lower FD limits keep an LRU.
        fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        self._max_open = len(ranges) if fd_limit == resource.RLIM_INFINITY else max(1, fd_limit - 64)
        self._maps = OrderedDict()

    def __len__(self):
        return self.manifest["window_count"]

    def _shard(self, number):
        if number in self._maps:
            self._maps.move_to_end(number)
            return self._maps[number]
        receipt = self._files[number]
        path = self.root / receipt["file"]
        if path.stat().st_size != receipt["size_bytes"]:
            raise ValueError(f"Verified shard size changed: {path}")
        if len(self._maps) >= self._max_open:
            self._maps.popitem(last=False)
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        if rows.shape != (receipt["stop"] - receipt["start"],) or rows.dtype != self._dtype:
            raise ValueError(f"Unexpected record layout in {path}")
        if hasattr(rows._mmap, "madvise") and hasattr(mmap, "MADV_RANDOM"):
            rows._mmap.madvise(mmap.MADV_RANDOM)
        self._maps[number] = rows
        return rows

    def batch(self, indices, device="cpu"):
        """Preserve requested order and repeats. Only the three physical vectors are normalized."""
        indices = np.asarray(indices)
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty vector of integer window indices")
        if indices.min() < 0 or indices.max() >= len(self):
            raise IndexError("Window index outside the verified training split")
        shard_numbers = indices // self.shard_rows
        records = np.empty(indices.size, dtype=self._dtype)
        for number in np.unique(shard_numbers):
            positions = np.flatnonzero(shard_numbers == number)
            records[positions] = self._shard(int(number))[indices[positions] % self.shard_rows]
        if not np.array_equal(records["index"], indices):
            raise ValueError("Read window indices differ from the verified order")
        tokens = torch.from_numpy(records["action_tokens"]).contiguous().to(device)
        physical = (torch.from_numpy(records[key]) for key in ("measured_q", "initial_reference", "expert_action"))
        return (tokens, *(self._coordinates.normalized(values).to(device) for values in physical))
