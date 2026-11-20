import json
import unittest
from pathlib import Path

import torch

from wam.ode_policy.calibration import PriorCoordinates, fit_affine, load_source, trajectory_metrics
from wam.ode_policy.kitchen import JointCoordinates


class PriorScaleTests(unittest.TestCase):
    def test_verified_group_and_round_trip(self):
        path = Path(__file__).resolve().parents[1] / "configs/ode_policy/robot37_256013_arx5.json"
        spec = json.loads(path.read_text())
        group = "raw36/arx_incremental920_20260917/arx5"
        row = load_source(path, group, {"phoenix_weights": spec["release_sha256"][0]})
        with self.assertRaises(ValueError):
            load_source(path, group, {"phoenix_weights": "wrong"})
        bounds = torch.stack((torch.ones(14), torch.full((14,), 4.)))
        kitchen = JointCoordinates(bounds, "cpu")
        coords = PriorCoordinates(kitchen, row)
        raw = torch.full((1, 32, 14), 0.5)
        physical = kitchen.physical(coords.slots(raw))
        self.assertAlmostEqual(physical[0, 0, 0].item(), -0.11415625, places=5)
        torch.testing.assert_close(coords.state.normalized(row["state"][0]), -torch.ones(14))

    def test_affine_and_timing(self):
        x = torch.arange(4.)[:, None].expand(4,14).clone()
        x[:, 0] = 4.
        y = 2*x-3
        slope, offset = fit_affine([(x,y)])
        torch.testing.assert_close(x*slope+offset,y)
        self.assertEqual(slope[0].item(), 0.)
        gt = torch.zeros(4,14)
        gt[1:,6] = 1
        pred = torch.zeros_like(gt)
        pred[2:,6] = 1
        result = trajectory_metrics(pred,gt,torch.zeros(14),torch.stack([torch.zeros(14),torch.ones(14)]))
        self.assertEqual(result["gripper_timing_abs_ticks"], [1])
