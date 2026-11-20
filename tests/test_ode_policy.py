import io
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from wam.ode_policy.head import ActionBlock, ODEHead
from wam.ode_policy.kitchen import ACTIVE, JointCoordinates, observation
from wam.ode_policy.policy import KitchenPolicy
from wam.ode_policy.runner import evaluate, replay, train
from wam.ode_policy.calibration import audit_prior
from wam.ode_policy.phoenix import read_prior
from wam.model.backbones.wan22.wan_video_dit import WanVideoDiT
from wam.model.backbones.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from wam.model.modules.experts.action_dit import ActionDiT
from wam.model.modules.memory.proprio_encoder import ProprioContextEncoder
from wam.model.modules.mot.mixture_of_transformers import MoT


class TinyPhoenix(torch.nn.Module):
    """Real Phoenix video/action experts and MoT. Fixture substitutes the image codec."""
    def __init__(self):
        super().__init__()
        common = dict(hidden_dim=32, ffn_dim=48, text_dim=16, freq_dim=8, eps=1e-6,
                      num_heads=2, attn_head_dim=12, num_layers=1)
        video = WanVideoDiT(**common, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
                            has_image_input=False, seperated_timestep=True)
        action = ActionDiT(**common, action_dim=80)
        self.mot = MoT({"video": video, "action": action}, attention_backend="sdpa")
        self.proprio_encoder = ProprioContextEncoder(80, 16)
        self.video_expert, self.action_expert = video, action
        self.device, self.torch_dtype = torch.device("cpu"), torch.float32
        self.vae = SimpleNamespace(temporal_downsample_factor=4)
        self.joint96_three_view, self.proprio_dim = True, 80
        self.future_delta_enabled = self.understanding_enabled = False
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler()
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler()
        self.eval().requires_grad_(False)

    def _action_token_seq_len_for_mask(self, length):
        return length


class ODETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.h = 32
        self.p = torch.randn(1, self.h, 14)
        self.q = torch.randn_like(self.p)
        self.a0 = torch.randn(1, 14)
        self.head = ODEHead("ab", 32, hidden_dim=32, context_dim=16)

    def test_boundary_and_feedback_budget(self):
        block = ActionBlock(self.head, self.p, self.a0)
        torch.testing.assert_close(block.targets(self.q[:, 0], self.a0, start=0, count=32), self.p)
        sent = self.p[:, 7] + 0.4
        torch.testing.assert_close(block.targets(self.q[:, 8], sent, start=8, count=1)[:, 0],
                                   self.p[:, 8] + 0.4)
        calls = []
        with self.head.field.register_forward_hook(lambda *args: calls.append(1)):
            pred = replay("b", self.head, self.p, self.a0, self.q)
        self.assertEqual(len(calls), 32)
        torch.testing.assert_close(pred, self.p)

    def test_shared_weights_and_stale_control(self):
        torch.nn.init.normal_(self.head.field[-1].weight, std=0.1)
        a = replay("a", self.head, self.p, self.a0, self.q)
        stale = replay("b_stale", self.head, self.p, self.a0, self.q)
        torch.testing.assert_close(a, stale)
        b = replay("b", self.head, self.p, self.a0, self.q)
        self.assertFalse(torch.allclose(a[:, 1:], b[:, 1:]))
        for boundary, model_boundary in [(0, False), (8, False), (31, True)]:
            self.head.zero_grad()
            self.head.path_loss(self.p, self.q, self.a0, self.q, boundary=boundary,
                                model_boundary=model_boundary).backward()
            self.assertGreater(self.head.field[-1].weight.grad.abs().sum().item(), 0)

    def test_joint_sampler_and_single_query(self):
        from wam.inference.frame_joint import infer_frame_joint_window
        model = TinyPhoenix()
        text = torch.randn(6, 16), torch.ones(6, dtype=torch.bool)
        obs = observation(torch.zeros(3, 3, 16, 16, dtype=torch.uint8), self.q[:, 0], text)
        latent = torch.randn(1, 4, 1, 4, 12)
        with patch("wam.inference.frame_joint.video_codec.encode_input_image_latents_tensor", return_value=latent), \
             patch("wam.ode_policy.phoenix.encode_input_image_latents_tensor", return_value=latent):
            kwargs = dict(prompt=None, **obs, num_frames=9, action_horizon=32,
                          num_inference_steps=2, seed=42, decode_video=False)
            default = infer_frame_joint_window(model, **kwargs)
            capture = infer_frame_joint_window(model, **kwargs, return_action_tokens=True)
            self.assertNotIn("action_tokens", default)
            torch.testing.assert_close(default["action"], capture["action"], rtol=0, atol=0)
            for family, passes in [("ab", 2), ("c1", 2), ("c2", 1)]:
                head = ODEHead(family, 32, hidden_dim=32, context_dim=16)
                calls = []
                with model.mot.register_forward_hook(lambda *args: calls.append(1)):
                    prior = read_prior(model, head, obs, active=ACTIVE, denoise_steps=2, seed=42)
                self.assertEqual(len(calls), passes)
                head.path_loss(prior, self.q, self.a0, self.p).backward()
                if family == "c2":
                    self.assertGreater(head.query.grad.norm().item(), 0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))


    def test_physical_tick_api(self):
        model = TinyPhoenix()
        coordinates = JointCoordinates(torch.stack((torch.arange(14), torch.arange(14) + 3)), "cpu")
        text = {"plate": (torch.randn(6, 16), torch.ones(6, dtype=torch.bool))}
        policy = KitchenPolicy(model, self.head, coordinates, text, denoise_steps=1)
        rgb = torch.zeros(3, 3, 16, 16, dtype=torch.uint8)
        with patch("wam.ode_policy.policy.read_prior", return_value=self.p) as prior_reader:
            block = policy.begin_block("plate", rgb, self.q[0, 0], self.a0[0])
            chunk = block.full_chunk()
            last = self.a0[0]
            for k in range(32):
                last = block.next_target(k, self.q[0, k], last)
                torch.testing.assert_close(last, chunk[k], rtol=2e-5, atol=2e-5)
            self.assertEqual(prior_reader.call_count, 1)

    def test_train_save_and_replay(self):
        torch.manual_seed(43)
        model = TinyPhoenix()
        horizon = 4
        states = np.random.default_rng(53).normal(size=(14, 14)).astype(np.float32)
        actions = states.copy()
        for lo, hi in ((0, 7), (7, 14)):
            actions[lo:hi-1] = states[lo+1:hi]
        group = {"data/state_30hz": states, "data/action_30hz": actions,
                 "data/camera": np.zeros((14, 16, 16, 9), dtype=np.uint8),
                 "meta/episode_ends": np.array([7, 14])}
        cfg = dict(policy=dict(horizon=horizon, hz=30, hidden_dim=32, context_dim=16,
                               denoise_steps=1, solver="euler"),
                   train=dict(seed=11, steps=1, lr=0.0001, batch_size=1, suffix_warmup_steps=0,
                              suffix_probability=1, model_boundary_probability=1),
                   sources=[dict(name="test", task="plate", episodes=2, validation=[1])])
        cfg["instructions"] = {"plate": "Wipe the plate."}
        text = {"plate": (torch.randn(6, 16), torch.ones(6, dtype=torch.bool))}
        latent = torch.randn(1, 4, 1, 4, 12)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as log, \
             patch.dict("sys.modules", {"zarr": SimpleNamespace(open_group=lambda *a, **kw: group)}), \
             patch("wam.inference.frame_joint.video_codec.encode_input_image_latents_tensor", return_value=latent), \
             patch("wam.ode_policy.phoenix.encode_input_image_latents_tensor", return_value=latent):
            audit_file = Path(tmp) / "audit.pt"
            audit_args = SimpleNamespace(data_root=Path(tmp), output=audit_file,
                                         fit_windows=1, windows=1, source_profile=None)
            audit_prior(audit_args, cfg, model, text, {"fixture": "test"})
            result = torch.load(audit_file, weights_only=True)
            self.assertEqual(result["fit_windows"], 1)
            self.assertEqual(result["report"][0]["episode"], 1)
            self.assertEqual(set(result["report"][0]["metrics"]), {"hold", "direct_slots", "fitted"})
            for family in ("ab", "c1", "c2"):
                out = Path(tmp) / (family + ".pt")
                args = SimpleNamespace(data_root=Path(tmp), family=family, output=out,
                                       head_weights=out, steps=1, batch_size=1, windows=1,
                                       prior_audit=audit_file, prior_kind="fitted")
                train(args, cfg, model, text, {"fixture": "test"})
                for mode in (("a", "b", "b_stale", "phoenix") if family == "ab" else (family,)):
                    args.mode = mode
                    evaluate(args, model, text, {"fixture": "test"})
            self.assertIn('"feedback_ticks": 4', log.getvalue())
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
