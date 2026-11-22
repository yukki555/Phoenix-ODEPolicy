import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from wam.ode_policy.cached_train import train
from wam.ode_policy.cached_windows import CachedWindows, RELEASE_CHECKS
from wam.ode_policy.recipe import CONFIG_ROOT, resolve_recipe


class CachedWindowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.spec = resolve_recipe(CONFIG_ROOT / "recipes/kitchen_action_tokens_direct_feedback.yaml")
        profile = Path(self.spec["parent"]["state_stats_profile"])
        dtype = np.dtype([
            ("index", "<i8"), ("action_tokens", "<f4", (32, 1024)),
            ("measured_q", "<f4", (32, 14)), ("initial_reference", "<f4", (14,)),
            ("expert_action", "<f4", (32, 14)),
        ])
        manifest = dict(split="train", window_count=3, task_window_count={"egg": 3},
                        episode_count=1, denoise_steps=1, feature="final_preprojection_action_tokens",
                        horizon=32, hz=30, shard_rows=2, record_dtype=dtype.descr,
                        base_seed=42, seed_rule="42 + split_window_index",
                        identity={"phoenix_weights": json.loads(profile.read_bytes())["release_sha256"][0]},
                        state_stats_group=self.spec["parent"]["state_stats_group"],
                        state_profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                        kitchen_config_sha256=hashlib.sha256((CONFIG_ROOT / "kitchen.yaml").read_bytes()).hexdigest(),
                        train_bounds=[[1.] * 14, [5.] * 14])
        (self.root / "shards").mkdir()
        receipts = []
        for start, stop in ((0, 2), (2, 3)):
            ids = np.arange(start, stop)
            rows = np.zeros(len(ids), dtype=dtype)
            rows["index"] = ids
            rows["action_tokens"] = ids[:, None, None]
            rows["measured_q"] = ids[:, None, None] + 1
            rows["initial_reference"] = ids[:, None] + 2
            rows["expert_action"] = ids[:, None, None] + 3
            name = f"shards/{start:06d}.npy"
            np.save(self.root / name, rows)
            payload = (self.root / name).read_bytes()
            receipts.append(dict(file=name, start=start, stop=stop, size_bytes=len(payload),
                                 sha256=hashlib.sha256(payload).hexdigest()))
        content = json.dumps(manifest).encode()
        (self.root / "manifest.json").write_bytes(content)
        self.coverage = dict(split="train", status="verified", expected=3, verified=3, fraction=1.,
                             by_task={"egg": 3}, episodes=1, shards=2, files=receipts,
                             total_bytes=sum(r["size_bytes"] for r in receipts),
                             manifest_sha256=hashlib.sha256(content).hexdigest())
        self.publish()

    def publish(self):
        (self.root / "coverage.json").write_text(json.dumps(self.coverage))

    def publish_release(self, **fields):
        report = {k: v for k, v in self.coverage.items() if k not in ("verified", "fraction")}
        report.update(status="released", proof_kind="shard_receipts", receipt_windows=3, checks=sorted(RELEASE_CHECKS))
        report.update(fields)
        (self.root / "release.json").write_text(json.dumps(report))

    def test_random_access_preserves_global_indices_and_normalizes_once(self):
        cache = CachedWindows(self.root, self.spec)
        ids = torch.tensor([2, 0, 1, 2])
        tokens, measured, initial, targets = cache.batch(ids)
        torch.testing.assert_close(tokens[:, 0, 0], ids.float())
        torch.testing.assert_close(measured[:, 0, 0], ids.float() / 2 - 1)
        torch.testing.assert_close(initial[:, 0], ids.float() / 2 - .5)
        torch.testing.assert_close(targets[:, 0, 0], ids.float() / 2)
        self.assertEqual(tokens.shape, (4, 32, 1024))
        with self.assertRaises(IndexError):
            cache.batch([3])

    def test_no_verified_coverage_or_wrong_manifest(self):
        (self.root / "coverage.json").unlink()
        with self.assertRaises(FileNotFoundError):
            CachedWindows(self.root, self.spec)
        self.coverage["manifest_sha256"] = "wrong"
        self.publish()
        with self.assertRaisesRegex(ValueError, "coverage"):
            CachedWindows(self.root, self.spec)

    def test_complete_counts_with_overlapping_receipts(self):
        self.coverage["files"][1]["start"] = 1
        self.publish()
        with self.assertRaisesRegex(ValueError, "receipts"):
            CachedWindows(self.root, self.spec)

    def test_reject_changed_shard_and_wrong_recipe(self):
        cache = CachedWindows(self.root, self.spec)
        with (self.root / "shards/000002.npy").open("ab") as f:
            f.write(b"extra")
        with self.assertRaisesRegex(ValueError, "shard size"):
            cache.batch([2])
        self.spec["policy"]["denoise_steps"] = 10
        with self.assertRaisesRegex(ValueError, "generation"):
            CachedWindows(self.root, self.spec)

    def test_receipt_evidence_is_distinct_and_full_report_has_priority(self):
        self.publish_release()
        full = CachedWindows(self.root, self.spec)
        quick = CachedWindows(self.root, self.spec, evidence="release")
        self.assertEqual(full.proof_kind, "full_verification")
        self.assertEqual(quick.proof_kind, "shard_receipts")
        self.assertIsNone(quick.coverage_sha256)
        self.assertNotEqual(full.evidence_sha256, quick.evidence_sha256)
        (self.root / "coverage.json").unlink()
        self.assertEqual(CachedWindows(self.root, self.spec).evidence_sha256, quick.evidence_sha256)
        for fields in (dict(status="verified"), dict(verified=3), dict(fraction=1), dict(proof_kind="full_verification"),
                       dict(checks=["file_sizes"]), dict(receipt_windows=2)):
            with self.subTest(fields=fields):
                self.publish_release(**fields)
                with self.assertRaisesRegex(ValueError, "release report"):
                    CachedWindows(self.root, self.spec)
        self.publish_release(status="verified", verified=3, fraction=1)
        (self.root / "release.json").rename(self.root / "coverage.json")
        with self.assertRaisesRegex(ValueError, "full verification"):
            CachedWindows(self.root, self.spec)

    def test_release_checks_every_file_before_sampling(self):
        self.publish_release()
        with (self.root / "shards/000002.npy").open("ab") as f:
            f.write(b"extra")
        with self.assertRaisesRegex(ValueError, "shard size"):
            CachedWindows(self.root, self.spec, evidence="release")

    def check_training_resume(self, evidence, legacy=False):
        recipe = CONFIG_ROOT / "recipes/kitchen_action_tokens_action_feature_field_condition.yaml"
        overrides = yaml.safe_load(recipe.read_text())
        overrides.update(training={"batch_size": 4, "steps": 4, "lr_scheduler": {
            "type": "linear_warmup_cosine", "warmup_steps": 2, "min_lr": 1e-5,
        }}, policy={"hidden_dim": 32, "context_dim": 16})
        recipe = self.root / "short.yaml"
        recipe.write_text(yaml.safe_dump(overrides))
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        full = self.root / "full" / "checkpoints"
        saved = []
        for name, resume in (("full", None), ("resume", full / "step-000002.pt")):
            output = self.root / name
            if resume and legacy:
                checkpoint = torch.load(resume, weights_only=True)
                del checkpoint["cache_training"]["proof_kind"]
                resume = self.root / "legacy.pt"
                torch.save(checkpoint, resume)
            args = SimpleNamespace(recipe=recipe, cache=self.root, cache_evidence="auto" if resume else evidence,
                                   stop_at=3, output=output, resume=resume,
                                   log_every=1, checkpoint_at=[2], checkpoint_every=100, verify_ddp=True)
            with redirect_stdout(StringIO()):
                train(args, rank=0, world=1, device=torch.device("cpu"))
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            rates = [row["lr"] for row in rows if row["event"] == "train"]
            np.testing.assert_allclose(rates, [5.5e-5] if resume else [5e-5, 1e-4, 5.5e-5])
            saved.append(torch.load(output / "checkpoints/step-000003.pt", weights_only=True))
        full, resumed = saved
        self.assertAlmostEqual(resumed["optimizer"]["param_groups"][0]["lr"], 5.5e-5, places=15)
        self.assertIn(evidence, full["cache_training"])
        for key in ("steps", "sample_sha256", "last_loss", "training_windows", "cache_training"):
            self.assertEqual(full[key], resumed[key], key)
        self.assertEqual(full["optimizer"]["param_groups"], resumed["optimizer"]["param_groups"])
        for a, b in ((full["head"], resumed["head"]), (full["optimizer"]["state"], resumed["optimizer"]["state"]),
                     (full["rng_by_rank"], resumed["rng_by_rank"])):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_resume_matches_uninterrupted_head_optimizer_and_sampling(self):
        self.check_training_resume("coverage", legacy=True)

    def test_release_resume_pins_original_proof_when_full_coverage_is_available(self):
        self.publish_release()
        self.check_training_resume("release")
