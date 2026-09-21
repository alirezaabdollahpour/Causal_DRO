"""Small behavioral checks for the report's nodewise Adaptive Causal Duchi.

The losses below are pointwise functions of a two-step synthetic scenario tree.
They make the optimal downstream response explicit, so a simultaneous update of
current and future actions cannot pass the recourse test by accident.
"""

import unittest

import torch

from live_translation.attacks import (
    AttackConfig,
    NestedSolveBudgetExceeded,
    ScenarioTree,
    _adaptive_checkpoints,
    adaptive_causal_duchi_attack,
    causal_duchi_streaming_attack,
    prefix_node_ids,
)


def branching_tree():
    x = torch.tensor([[[0.0], [1.0]], [[0.0], [-1.0]]], dtype=torch.float64)
    valid = torch.ones((2, 2), dtype=torch.bool)
    return ScenarioTree(x, valid, prefix_node_ids(x, valid))


def bilinear_loss(y):
    return y[:, 0, 0] * y[:, 1, 0] + y[:, 0, 0]


class AdaptiveCausalDuchiTests(unittest.TestCase):
    def setUp(self):
        self.config = AttackConfig(lam=1.0, steps=1, tolerance=1e-12)

    def test_checkpoint_schedule_matches_report(self):
        self.assertEqual(_adaptive_checkpoints(100),
                         {22, 41, 57, 70, 80, 87, 93, 99, 100})
        self.assertEqual(_adaptive_checkpoints(1), {1})

    def test_recourse_is_reoptimized_after_current_proposal(self):
        tree = branching_tree()
        result = adaptive_causal_duchi_attack(
            tree, bilinear_loss, self.config, restarts=1, seed=7,
            max_evaluations=512)

        # At y_0=z, each child maximizes z*y_1-(y_1-x_1)^2, hence
        # y_1^*(z)=x_1+z/2. The nominal root is zero; one ascent update with
        # step 1/(2 lambda) proposes z=1/2. Its children must then be solved
        # again at that new value before the returned continuation is chosen.
        torch.testing.assert_close(result.y[:, 0, 0],
                                   torch.full((2,), 0.5, dtype=torch.float64),
                                   rtol=0, atol=1e-10)
        torch.testing.assert_close(result.y[:, 1, 0] - tree.x[:, 1, 0],
                                   torch.full((2,), 0.25, dtype=torch.float64),
                                   rtol=0, atol=1e-10)
        self.assertGreater(result.diagnostics["objective"], 0.0)

    def test_fixed_attacked_prefix_is_preserved_and_used_by_recourse(self):
        tree = branching_tree()
        fixed = torch.full((2, 1, 1), 0.4, dtype=torch.float64)
        result = adaptive_causal_duchi_attack(
            tree, bilinear_loss, self.config, fixed_prefix_y=fixed,
            restarts=1, seed=3, max_evaluations=512)
        torch.testing.assert_close(result.y[:, :1], fixed, rtol=0, atol=0)
        torch.testing.assert_close(result.y[:, 1, 0] - tree.x[:, 1, 0],
                                   torch.full((2,), 0.2, dtype=torch.float64),
                                   rtol=0, atol=1e-10)

    def test_identity_incumbent_survives_bad_restarts(self):
        x = torch.tensor([[[2.0]]], dtype=torch.float64)
        valid = torch.ones((1, 1), dtype=torch.bool)
        tree = ScenarioTree(x, valid, prefix_node_ids(x, valid))
        loss = lambda y: -100.0 * (y[:, 0, 0] - 2.0).square()
        result = adaptive_causal_duchi_attack(
            tree, loss, self.config, restarts=4, seed=19,
            max_evaluations=256)
        torch.testing.assert_close(result.y, x, rtol=0, atol=0)
        self.assertAlmostEqual(result.diagnostics["objective"], 0.0)

    def test_evaluation_budget_is_enforced(self):
        tree = branching_tree()
        with self.assertRaises(NestedSolveBudgetExceeded):
            adaptive_causal_duchi_attack(
                tree, bilinear_loss, self.config, restarts=1, seed=0,
                max_evaluations=1)

    def test_weighted_conditional_gradient_and_variable_lengths(self):
        x = torch.tensor([[[0.0], [9.0]], [[0.0], [1.0]],
                          [[0.0], [-1.0]]], dtype=torch.float64)
        valid = torch.tensor([[True, False], [True, True], [True, True]])
        probabilities = torch.tensor([.2, .3, .5], dtype=torch.float64)
        tree = ScenarioTree(x, valid, prefix_node_ids(x, valid), probabilities)
        label_gradients = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

        def loss(y):
            return label_gradients * y[:, 0, 0] + y[:, 1, 0] * valid[:, 1]

        result = adaptive_causal_duchi_attack(
            tree, loss, AttackConfig(lam=1.0, cost_scale=2.0, steps=2,
                                     tolerance=1e-12),
            restarts=1, max_evaluations=512)
        # Conditional mean gradient is .2*1 + .3*2 + .5*3 = 2.3.
        # With cost_scale=2, the initial step and exact optimum are 1*2.3.
        torch.testing.assert_close(result.y[:, 0, 0],
                                   torch.full((3,), 2.3, dtype=torch.float64),
                                   rtol=0, atol=1e-10)
        torch.testing.assert_close(result.y[0, 1], x[0, 1], rtol=0, atol=0)
        torch.testing.assert_close(result.y[1:, 1, 0] - x[1:, 1, 0],
                                   torch.ones(2, dtype=torch.float64),
                                   rtol=0, atol=1e-10)

    def test_seed_is_repeatable_and_does_not_change_global_rng(self):
        tree = branching_tree()
        before = torch.get_rng_state().clone()
        first = adaptive_causal_duchi_attack(
            tree, bilinear_loss, self.config, restarts=3, seed=23,
            max_evaluations=512)
        middle = torch.get_rng_state().clone()
        second = adaptive_causal_duchi_attack(
            tree, bilinear_loss, self.config, restarts=3, seed=23,
            max_evaluations=512)
        after = torch.get_rng_state().clone()
        torch.testing.assert_close(first.y, second.y, rtol=0, atol=0)
        torch.testing.assert_close(before, middle, rtol=0, atol=0)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_online_first_action_ignores_unseen_path_and_reference(self):
        def sampler(prefix):
            if len(prefix) == 1:
                tree = branching_tree()
            else:
                x = prefix[None].clone()
                valid = torch.ones((1, len(prefix)), dtype=torch.bool)
                tree = ScenarioTree(x, valid, prefix_node_ids(x, valid))
            return tree, bilinear_loss

        valid = torch.ones((1, 2), dtype=torch.bool)
        positive_future = torch.tensor([[[0.0], [3.0]]], dtype=torch.float64)
        negative_future = torch.tensor([[[0.0], [-3.0]]], dtype=torch.float64)
        first = causal_duchi_streaming_attack(
            positive_future, valid, sampler, lambda y: y[:, 1, 0],
            self.config, solver="adaptive", restarts=1, seed=5,
            max_evaluations=512)
        second = causal_duchi_streaming_attack(
            negative_future, valid, sampler, lambda y: -y[:, 1, 0],
            self.config, solver="adaptive", restarts=1, seed=5,
            max_evaluations=512)
        torch.testing.assert_close(first.y[:, :1], second.y[:, :1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
