import unittest
from copy import deepcopy
from itertools import product

import torch
from torch import nn

from wam.ode_policy.calibration import PriorCoordinates
from wam.ode_policy.factory import build_architecture, restore_architecture
from wam.ode_policy.head import ActionBlock
from wam.ode_policy.kitchen import JointCoordinates
from wam.ode_policy.recipe import CONFIG_ROOT, evaluation_modes, resolve_recipe
from wam.ode_policy.replay import replay


class JointHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = {}
        for path in (CONFIG_ROOT / "recipes").rglob("*.yaml"):
            spec = resolve_recipe(path)
            conditioning = spec.get("conditioning", {})
            if conditioning.get("feedback_route") == "joint_history_attention":
                history = conditioning["history_feedback"]
                cls.specs[(history["query"], history["injection"])] = spec

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        self.model = nn.Module()
        self.model.action_expert = nn.Linear(8, 8)
        self.model.action_expert.hidden_dim = 8
        self.model.device = torch.device("cpu")
        self.bounds = torch.stack((-torch.ones(14), torch.ones(14)))
        self.coordinates = PriorCoordinates(JointCoordinates(self.bounds, "cpu"))
        self.tokens = torch.randn(2, 9, 8)
        self.states = torch.randn(2, 9, 14)
        self.reference = torch.randn(2, 14)

    def small_spec(self, query="action_feature", injection="velocity_field_condition"):
        spec = deepcopy(self.specs[query, injection])
        spec["policy"].update(horizon=self.tokens.shape[1], hidden_dim=32, context_dim=16)
        return spec

    def test_four_factory_recipes_train_and_restore(self):
        combinations = product(("action_feature", "block_step"),
                               ("velocity_field_condition", "action_velocity_residual"))
        self.assertEqual(set(self.specs), set(combinations))
        for query, injection in self.specs:
            with self.subTest(query=query, injection=injection):
                spec = self.small_spec(query, injection)
                self.assertEqual(spec["policy"]["denoise_steps"], 1)
                self.assertEqual(spec["conditioning"]["token_reduction"], "none")
                self.assertEqual(spec["conditioning"]["history_feedback"]["history_length"], 4)
                self.assertEqual(set(evaluation_modes(spec, "paired")),
                                 {"latest_joint_each_tick", "one_tick_delayed", "block_start",
                                  "hold_latest", "feedback_off"})
                head = build_architecture(spec, self.model, self.coordinates).head
                optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
                targets = torch.randn_like(self.states)
                loss = head.path_loss(self.tokens, self.states, self.reference, targets, feedback=True)
                prediction = replay("latest_joint_each_tick", head, self.tokens, self.reference, self.states)
                torch.testing.assert_close(loss, (prediction - targets).square().mean())
                loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in head.history_feedback.parameters()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
                optimizer.step()
                saved = dict(recipe=spec, identity={}, bounds=self.bounds, head=head.state_dict())
                restored = restore_architecture(saved, self.model, {}).head
                for mode in ("block_start", "latest_joint_each_tick", "one_tick_delayed", "feedback_off"):
                    expected = replay(mode, head, self.tokens, self.reference, self.states)
                    actual = replay(mode, restored, self.tokens, self.reference, self.states)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_feedback_off_matches_nominal_field_even_with_bias(self):
        for injection in ("velocity_field_condition", "action_velocity_residual"):
            with self.subTest(injection=injection):
                head = build_architecture(self.small_spec(injection=injection), self.model, self.coordinates).head
                # Any learned offset in the feedback decoder must drop out when the branch is disabled.
                with torch.no_grad():
                    for name, parameter in head.history_feedback.named_parameters():
                        if name.endswith("bias"):
                            parameter.fill_(0.7)
                reference = self.reference
                context = head.prepare(self.tokens)
                nominal = []
                for k in range(head.horizon):
                    action = head.architecture.at_tick(head, context, k)
                    phase = reference.new_full((reference.shape[0], 1), k / head.horizon)
                    inputs = torch.cat((action, self.states[:, 0], reference, phase), dim=-1)
                    reference = reference + head.field(inputs) / head.hz
                    nominal.append(reference)
                disabled = replay("feedback_off", head, self.tokens, self.reference, self.states)
                torch.testing.assert_close(disabled, torch.stack(nominal, dim=1), rtol=0, atol=0)

    def test_history_reach_on_a_common_accepted_reference(self):
        head = build_architecture(self.small_spec(), self.model, self.coordinates).head
        accepted = torch.randn_like(self.states)
        changed = self.states.clone()
        changed[:, 2] += 1.5
        different_reference = accepted.clone()
        different_reference[:, 2] += 2

        def path(states, references, control):
            block = ActionBlock(head, self.tokens, references[:, 0], initial_q=states[:, 0])
            return torch.cat([block.targets(states[:, k], references[:, k], start=k, count=1,
                                            feedback_control=control) for k in range(head.horizon)], dim=1)

        for control, affected in ((None, {2, 3, 4, 5}), ("one_tick_delayed", {3, 4, 5, 6})):
            with self.subTest(control=control):
                original = path(self.states, accepted, control)
                observed = path(changed, accepted, control)
                # Changing a measurement cannot affect outputs before it is observed or after it expires.
                different = (original != observed).any(dim=(0, 2)).nonzero().flatten().tolist()
                self.assertEqual(set(different), affected)
                # A changed, accepted command at tick 2 also enters that tick's history record.
                commanded = path(self.states, different_reference, control)
                different = (original != commanded).any(dim=(0, 2)).nonzero().flatten().tolist()
                self.assertEqual(set(different), affected | {2})

        # Keep r fixed and swap two measurements within the four records. Only their ages change.
        swapped = self.states.clone()
        swapped[:, [1, 2]] = self.states[:, [2, 1]]
        held_reference = self.reference[:, None].expand_as(self.states)
        first_order = path(self.states, held_reference, None)[:, 3]
        other_order = path(swapped, held_reference, None)[:, 3]
        self.assertGreater((first_order - other_order).abs().max().item(), 1e-8)

    def test_preview_and_retried_tick_do_not_contaminate_history(self):
        head = build_architecture(self.small_spec(), self.model, self.coordinates).head
        block = ActionBlock(head, self.tokens, self.reference, initial_q=self.states[:, 0])
        previous = self.reference
        predicted = []
        for k in range(head.horizon):
            if k == 3:
                block.targets(self.states[:, 0], self.reference, start=0, count=head.horizon)
                block.targets(self.states[:, k] + 1, previous + 0.2, start=k, count=1)
            previous = block.targets(self.states[:, k], previous, start=k, count=1)[:, 0]
            predicted.append(previous)
        fresh = replay("latest_joint_each_tick", head, self.tokens, self.reference, self.states)
        torch.testing.assert_close(torch.stack(predicted, dim=1), fresh, rtol=0, atol=0)
        fixed = replay("block_start", head, self.tokens, self.reference, self.states)
        held = replay("latest_joint_each_tick", head, self.tokens, self.reference,
                      self.states[:, :1].expand_as(self.states))
        torch.testing.assert_close(fixed, held, rtol=0, atol=0)

    def test_residual_queries_use_only_their_configured_information(self):
        other_tokens = torch.randn_like(self.tokens)
        for query in ("block_step", "action_feature"):
            with self.subTest(query=query):
                spec = self.small_spec(query, "action_velocity_residual")
                head = build_architecture(spec, self.model, self.coordinates).head

                def correction(tokens):
                    block = ActionBlock(head, tokens, self.reference, initial_q=self.states[:, 0])
                    nominal = ActionBlock(head, tokens, self.reference, initial_q=self.states[:, 0])
                    result = []
                    for k in range(4):
                        q = self.states[:, k]
                        actual = block.targets(q, self.reference, start=k, count=1)
                        disabled = nominal.targets(q, self.reference, start=k, count=1,
                                                   feedback_control="feedback_off")
                        result.append(actual - disabled)
                    return torch.cat(result, dim=1)

                first, second = correction(self.tokens), correction(other_tokens)
                # One available memory token gets attention weight one for either query.
                torch.testing.assert_close(first[:, 0], second[:, 0], atol=4e-7, rtol=0)
                if query == "block_step":
                    torch.testing.assert_close(first, second, atol=4e-7, rtol=0)
                else:
                    self.assertGreater((first[:, 1:] - second[:, 1:]).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
