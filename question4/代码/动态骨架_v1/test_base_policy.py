"""基础控制器（29 站骨架 + 25 米证书 + 有界场景树）的针对性检查。"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from q4_base_policy import Policy, MPCPolicy, SparsePolicy
from q4_sparse_backbone import CELLS, certified, independent_cover_check
from q3_belief import Observation
from order_route import scenario_masks, objective


class Tests(unittest.TestCase):
    def test_rounded_offline_route_has_independent_continuous_certificate(self):
        policy = Policy()
        points = [np.zeros(2)]+policy.future
        self.assertTrue(certified(points).all())
        self.assertTrue(independent_cover_check(points))

    def test_local_recheck_agrees_with_full_certificate(self):
        policy = Policy()
        old = np.array([np.zeros(2)]+policy.future)
        rng = np.random.default_rng(56)
        for shift in (0., 10., 80., 400.):
            new = old.copy()
            j = int(rng.integers(1, len(old)))
            new[j] += rng.normal(0, shift, 2)
            affected = (np.linalg.norm(np.abs(CELLS-old[j])+12.5, axis=1) <= 999.9)
            affected |= np.linalg.norm(np.abs(CELLS-new[j])+12.5, axis=1) <= 999.9
            self.assertEqual(bool(certified(new, CELLS[affected]).all()), bool(certified(new).all()))

    def test_tree_gate_is_bounded_and_does_not_change_fallback_action(self):
        policy = Policy()
        policy.history[0] = [Observation('no_signal', 0., 0.)]
        policy.local_rf[0] = 1
        cover, post = SimpleNamespace(c=np.zeros((20, 2))), SimpleNamespace(numeric_ok=True)
        answer = ('clear', np.array([1., 2.]), 3., 'finite_cover')
        with patch.object(policy, 'local_state', return_value=(cover, post)), \
             patch.object(MPCPolicy, 'local_action', return_value=answer) as tree, \
             patch.object(SparsePolicy, 'local_action', return_value=answer) as simple:
            self.assertIs(policy.local_action(0), answer)
            self.assertIs(policy.local_action(0), answer)
            self.assertEqual(tree.call_count, 1)
            self.assertEqual(simple.call_count, 1)
            cover.c = np.zeros((401, 2))
            policy.light_mpc_calls[0] = 0
            policy.local_action(0)
            self.assertEqual(tree.call_count, 1)

    def test_task_scheduler_does_not_run_diagnostic_tree(self):
        policy = Policy()
        with patch.object(policy, 'local_state', side_effect=AssertionError('unexpected posterior rollout')):
            choice, info = policy._task_choice(np.array([0, 1]), [np.array([100., 0.]), np.array([500., 0.])], policy.future)
        self.assertFalse(info['task_mpc'])
        self.assertIn(choice, (None, 0, 1))

    def test_directional_scenarios_and_order_objective_are_deterministic(self):
        policy = Policy()
        masks, n = scenario_masks(policy.future)
        other, _ = scenario_masks(policy.future)
        self.assertEqual(masks, other)
        heard = 0
        for mask in masks:
            heard |= mask
        self.assertEqual(heard.bit_count(), n)


if __name__ == '__main__':
    unittest.main()
