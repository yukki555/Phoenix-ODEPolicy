import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

from test_ode_policy import TinyPhoenix
from wam.ode_policy import runner
from wam.ode_policy.calibration import PriorCoordinates, load_source
from wam.ode_policy.factory import build_architecture, restore_architecture
from wam.ode_policy.head import ActionBlock
from wam.ode_policy.kitchen import JointCoordinates
from wam.ode_policy.trial import KitchenTrial
from wam.ode_policy.recipe import CONFIG_ROOT, EXECUTIONS, HISTORY_CONTROLS, checkpoint_recipe, evaluation_modes, resolve_recipe


class RecipeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(57)
        self.model = TinyPhoenix()
        self.bounds = torch.stack((torch.zeros(14), torch.ones(14)))
        self.coords = JointCoordinates(self.bounds, "cpu")

    def test_three_presets_and_legacy_residual_restore(self):
        files = sorted((CONFIG_ROOT / "recipes").glob("*.yaml"))
        specs = list(map(resolve_recipe, files))
        self.assertEqual({s["architecture"] for s in specs}, {"action_plan_residual_ode", "action_token_ode", "action_query_ode"})
        for spec in specs:
            controls = {"joint_history_attention": HISTORY_CONTROLS, "latent_adapter": ("adapter_off", "direct_c1")}.get(
                spec.get("conditioning", {}).get("feedback_route"), ())
            self.assertEqual(evaluation_modes(spec), [*EXECUTIONS, *controls, "hold_latest"])
        cfg = dict(policy=dict(horizon=4, hz=30, solver="euler", denoise_steps=2,
                               context_dim=16, hidden_dim=32), train=dict(seed=42))
        scale = (torch.ones(14)*2, torch.arange(14))
        source = dict(group="original", state=self.bounds+1, action=self.bounds+2)
        head = runner.make_head(cfg, "ab", self.model)
        saved = dict(config=cfg, family="ab", identity={}, bounds=self.bounds, source=source,
                     prior_affine=scale, head=head.state_dict())
        arch = restore_architecture(saved, self.model, {})
        raw = torch.randn(1, 4, 14)
        with patch.object(arch.head.architecture, "read", return_value=raw):
            prior = arch.read_condition({}, seed=42)
        torch.testing.assert_close(prior, raw*2 + torch.arange(14))
        q, sent = torch.rand(1,14), torch.rand(1,14)
        block = ActionBlock(arch.head, prior, sent)
        torch.testing.assert_close(block.targets(q, sent, start=0, count=4), prior)
        torch.testing.assert_close(block.targets(q, sent, start=2, count=1)[:,0], prior[:,2]+(sent-prior[:,1]))

    def test_legacy_fixed_schedule_and_state_only_profile(self):
        old = dict(family="c1", config=dict(policy={}, train=dict(seed=42)), steps=2, batch_size=1)
        spec = checkpoint_recipe(old)
        self.assertEqual(spec["training"]["observation_schedule"], "block_start")
        self.assertEqual(spec["training"]["steps"], 2)
        profile = json.loads((CONFIG_ROOT / "robot37_256013_arx5.json").read_text())
        group = next(iter(profile["groups"]))
        del profile["groups"][group]["action"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps(profile))
            source = load_source(path, group, {"phoenix_weights": profile["release_sha256"][0]}, include_action=False)
            self.assertIsNone(PriorCoordinates(self.coords, source).action)

    def test_temporal_mean_is_before_position_and_preserves_parameters(self):
        spec = resolve_recipe(CONFIG_ROOT / "recipes/kitchen_action_tokens.yaml")
        coords = PriorCoordinates(self.coords)
        full = build_architecture(spec, self.model, coords).head
        mean = build_architecture({**spec, "conditioning": {"token_reduction": "temporal_mean_repeat"}},
                                  self.model, coords).head
        mean.load_state_dict(full.state_dict(), strict=True)
        tokens = torch.randn(2, 32, self.model.action_expert.hidden_dim)
        expected = full.prepare(tokens.mean(1, keepdim=True).expand_as(tokens))
        torch.testing.assert_close(mean.prepare(tokens), expected)
        self.assertFalse(torch.allclose(full.prepare(tokens), expected))
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in mean.parameters()))

    def test_execution_controls_when_the_robot_state_is_read(self):
        q = torch.zeros(14)
        for execution, expected_reads in zip(EXECUTIONS, (0, 2)):
            policy = Mock(head=SimpleNamespace(horizon=2, hz=30, family="ab"))
            block = policy.begin_block.return_value
            block.full_chunk.return_value = torch.zeros(2, 14)
            block.next_target.return_value = q
            read_state = Mock(return_value=q)
            trial = KitchenTrial(policy, "plate", execution, "pair", "same-head")
            trial.begin_block(None, q)
            trial.tick(read_state, lambda target: target + 1)
            trial.tick(read_state, lambda target: target)
            self.assertEqual(read_state.call_count, expected_reads)
            if execution == "latest_joint_each_tick":
                torch.testing.assert_close(block.next_target.call_args.args[2], q+1)

    def test_latent_feedback_reaches_only_causal_targets(self):
        spec = resolve_recipe(CONFIG_ROOT / "recipes/kitchen_action_tokens_latent_feedback.yaml")
        head = build_architecture(spec, self.model, PriorCoordinates(self.coords)).head
        tokens = torch.randn(1, 32, self.model.action_expert.hidden_dim)
        states = torch.randn(1, 32, 14)
        reference = torch.randn(1, 14)  # Previous target differs from measured q0.
        def predict(mode, q):
            return runner.replay(mode, head, tokens, reference, q)

        torch.testing.assert_close(predict("latest_joint_each_tick", states), predict("adapter_off", states),
                                   atol=0, rtol=0)
        head.path_loss(tokens, states, reference, states, feedback=True).backward()
        self.assertGreater(head.feedback_adapter[-1].weight.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for n, p in head.named_parameters() if not n.startswith("feedback_adapter.")))
        torch.nn.init.normal_(head.feedback_adapter[-1].weight, std=0.2)
        changed = states.clone()
        changed[:, 10] += 1
        fresh = predict("latest_joint_each_tick", states)
        altered = predict("latest_joint_each_tick", changed)
        torch.testing.assert_close(altered[:, :10], fresh[:, :10], atol=0, rtol=0)
        self.assertGreater((altered[:, 10] - fresh[:, 10]).abs().max().item(), 1e-7)
        torch.testing.assert_close(predict("adapter_off", states), predict("adapter_off", changed), atol=0, rtol=0)
        fixed = predict("block_start", states)
        torch.testing.assert_close(fixed, predict("latest_joint_each_tick", states[:, :1].expand_as(states)))
        probe = runner.feedback_action_probe(head, tokens, reference, states, self.coords, states[0])
        self.assertGreater(probe["arm"]["target_change_rms"], 0)

    def test_recipe_cli_train_restore_and_paired_replay(self):
        model = self.model
        states = np.random.default_rng(42).normal(size=(20, 14)).astype("float32")
        action = np.roll(states, -1, axis=0)
        group = {"data/state_30hz": states, "data/action_30hz": action, "meta/episode_ends": [10,20],
                 "data/camera": np.zeros((20, 16, 16, 9), dtype=np.uint8)}
        text = {"egg": (torch.randn(6,16), torch.ones(6, dtype=torch.bool))}
        latent = torch.randn(1, 4, 1, 4, 12)
        cfg = yaml.safe_load((CONFIG_ROOT / "kitchen.yaml").read_text())
        cfg.update(sources=[dict(task="egg", name="fixture", episodes=2, validation=[1])],
                   instructions={"egg": "Flip the egg."})
        cfg["policy"]["horizon"] = 4
        profile = CONFIG_ROOT / "robot37_256013_arx5.json"
        identity = json.loads(profile.read_text())["release_sha256"][0]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "architectures").symlink_to(CONFIG_ROOT / "architectures")
            (root / "recipes").mkdir()
            (root / "kitchen.yaml").write_text(yaml.safe_dump(cfg))
            recipe = yaml.safe_load((CONFIG_ROOT / "recipes/kitchen_action_tokens.yaml").read_text())
            recipe["parent"]["state_stats_profile"] = str(profile)
            recipe["policy"] = dict(hidden_dim=32, context_dim=16, denoise_steps=1)
            path = root / "recipes/fixture.yaml"
            path.write_text(yaml.safe_dump(recipe))
            ckpt = root / "head.pt"
            def invoke(operation, *extra):
                argv = ["runner",operation,"--data-root",tmp,"--model-config","parent","--phoenix-weights","weights",*extra]
                stream = io.StringIO()
                with patch("sys.argv", argv), redirect_stdout(stream):
                    runner.main()
                return [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith("{")]

            with patch.dict("sys.modules", {"zarr": SimpleNamespace(open_group=lambda *a, **kw: group)}), \
                 patch("wam.inference.frame_joint.video_codec.encode_input_image_latents_tensor", return_value=latent), \
                 patch.object(runner, "digest", return_value=identity), patch.object(runner, "load_assets", return_value=(model,text)):
                invoke("train","--recipe",str(path),"--output",str(ckpt),"--steps","1","--batch-size","2")
                saved = torch.load(ckpt, weights_only=True)
                self.assertEqual(saved["recipe"]["training"]["batch_size"], 2)
                self.assertNotIn("action_prior", saved["recipe"])
                self.assertNotIn("action", saved["source"])
                calls = []
                with model.mot.register_forward_hook(lambda *args: calls.append(1)):
                    rows = invoke("eval","--head-weights",str(ckpt),"--windows","1")
                self.assertEqual(len(calls), 1)
                episode = [r for r in rows if "window_indices" in r]
                self.assertEqual([r["mode"] for r in episode], [*EXECUTIONS, "hold_latest"])
                self.assertTrue(all(r["window_indices"] == episode[0]["window_indices"] for r in episode))
                self.assertEqual(next(r["feedback_ticks"] for r in rows if "feedback_ticks" in r), 4)
                recipe["conditioning"]["token_reduction"] = "temporal_mean_repeat"
                path.write_text(yaml.safe_dump(recipe))
                with self.assertRaisesRegex(ValueError, "conditioning"):
                    invoke("eval","--recipe",str(path),"--head-weights",str(ckpt))
                second = root / "mean.pt"
                invoke("train","--recipe",str(path),"--output",str(second),"--steps","1","--batch-size","2")
                self.assertEqual(torch.load(second, weights_only=True)["recipe"]["conditioning"],
                                 {"token_reduction": "temporal_mean_repeat"})
                latent_recipe = yaml.safe_load((CONFIG_ROOT / "recipes/kitchen_action_tokens_latent_feedback.yaml").read_text())
                latent_recipe.update(parent=recipe["parent"], policy=recipe["policy"])
                path.write_text(yaml.safe_dump(latent_recipe))
                adapted = root / "adapted.pt"
                init = ["train", "--recipe", str(path), "--output", str(adapted), "--steps", "1", "--batch-size", "2"]
                with self.assertRaisesRegex(ValueError, "matching token reduction"):
                    invoke(*init, "--base-c1-weights", str(second))
                invoke(*init, "--base-c1-weights", str(ckpt))
                trained = torch.load(adapted, weights_only=True)
                self.assertEqual(trained["recipe"]["training"]["base_c1_sha256"], identity)
                for name, weight in saved["head"].items():
                    torch.testing.assert_close(trained["head"][name], weight, atol=0, rtol=0)
                self.assertGreater(trained["head"]["feedback_adapter.2.weight"].norm().item(), 0)
                with model.mot.register_forward_hook(lambda *args: calls.append(1)):
                    paired = invoke("eval", "--head-weights", str(adapted), "--windows", "1")
                self.assertEqual(len(calls), 2)
                self.assertTrue(any(r.get("diagnostic") == "shared_reference_feedback" for r in paired))
                scores = {r["mode"]: r["arm_rmse"] for r in paired if "window_indices" in r}
                for base_row, control in zip(episode[:2], ("adapter_off", "direct_c1")):
                    self.assertEqual(scores[control], base_row["arm_rmse"])

                history_spec = resolve_recipe(
                    CONFIG_ROOT / "recipes/kitchen_action_tokens_action_feature_field_condition.yaml")
                recipe["conditioning"] = history_spec["conditioning"]
                recipe["evaluation"] = history_spec["evaluation"]
                path.write_text(yaml.safe_dump(recipe))
                history_weights = root / "joint_history.pt"
                invoke("train", "--recipe", str(path), "--output", str(history_weights),
                       "--steps", "1", "--batch-size", "2")
                with model.mot.register_forward_hook(lambda *args: calls.append(1)):
                    rows = invoke("eval", "--head-weights", str(history_weights), "--mode", "paired",
                                  "--reference-head-weights", str(ckpt), "--windows", "1")
                self.assertEqual(len(calls), 3)  # One additional parent call, shared by both heads and all modes.
                self.assertEqual({r["mode"] for r in rows if "window_indices" in r},
                                 set(evaluation_modes(history_spec)))
                probe = next(r for r in rows if r.get("diagnostic") == "shared_reference_feedback")
                self.assertIn("delayed_rmse", probe["arm"])
                self.assertEqual(probe["reference_origin"], identity)
