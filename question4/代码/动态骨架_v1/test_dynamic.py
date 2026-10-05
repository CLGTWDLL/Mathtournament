"""第四问默认算法的针对性检查。

默认配置（feedback_geometry=False）只保留后验概率提前停止；
可选的 12.5 米反馈几何层用 feedback_geometry=True 单独覆盖。
"""
import sys
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from q3_belief import Observation
from q4_dynamic_policy import Policy


class DynamicTests(unittest.TestCase):
    def one_unknown_policy(self):
        policy = Policy(feedback_geometry=True)
        policy.batch = None
        policy.batch_after = None
        policy.absent[1:] = True
        return policy

    def test_default_configuration_keeps_geometry_layer_inert(self):
        policy = Policy()
        policy.batch = None
        policy.batch_after = None
        policy.visited = [np.array([0.0, 0.0])]
        policy.dynamic_prune_interval = 1
        policy.future = [np.array([0.0, 0.0]), np.array([100.0, 0.0])]
        policy._mark_feedback_empty()
        policy._dynamic_prune()
        self.assertFalse(policy.feedback_geometry)
        self.assertTrue(policy.future)
        self.assertEqual(policy.dynamic_pruned_sites, 0)

    def test_repeated_future_station_is_removed(self):
        policy = self.one_unknown_policy()
        policy.dynamic_prune_interval = 1
        policy.visited = [np.array([0.0, 0.0])]
        repeated = np.array([0.0, 0.0])
        informative = np.array([100.0, 0.0])
        policy.history[0] = [Observation('no_signal', *repeated)]
        policy.future = [repeated, informative]
        policy._dynamic_prune()
        self.assertEqual(len(policy.future), 1)
        np.testing.assert_allclose(policy.future[0], informative)
        self.assertEqual(policy.dynamic_pruned_sites, 1)

    def test_empty_conservative_support_marks_channel_absent(self):
        policy = self.one_unknown_policy()
        policy._support_mask = lambda channel: np.zeros(3, dtype=bool)
        policy._mark_feedback_empty()
        self.assertTrue(policy.absent[0])
        self.assertEqual(policy.absence_reason[1], 'feedback_geometry_empty')

    def test_restricted_finish_can_drop_station_outside_remaining_support(self):
        policy = self.one_unknown_policy()
        policy.dynamic_prune_interval = 1
        policy.visited = [np.array([0.0, 0.0])]
        policy.dynamic_prune_interval = 1
        policy.visited = [np.array([0.0, 0.0])]
        support = np.zeros(len(policy._support_mask(0)), dtype=bool)
        from q4_sparse_backbone import CELLS
        center_index = int(np.argmin(np.linalg.norm(CELLS, axis=1)))
        center = CELLS[center_index]
        support[center_index] = True
        policy._support_mask = lambda channel: support
        policy._negative_positions = lambda channel: [
            center + np.array([-500.0, -500.0]),
            center + np.array([0.0, 500.0]),
            center + np.array([500.0, -500.0]),
        ]
        useless = np.array([2700.0, 0.0])
        policy.future = [useless]
        policy._dynamic_prune()
        self.assertEqual(policy.future, [])
        self.assertEqual(policy.dynamic_pruned_sites, 1)

    def test_count_prior_prevents_stop_before_ten_confirmations(self):
        policy = Policy(existence_stop_threshold=0.5, existence_samples=256)
        policy.batch = None
        policy.batch_after = None
        policy.confirmed[:9] = True
        policy.cleared[:9] = True
        policy._existence_likelihood = lambda channel: 1e-9
        posterior = policy._posterior_existence()
        self.assertAlmostEqual(posterior['probability_any_remaining'], 1.0)

    def test_probability_threshold_can_end_remaining_patrol(self):
        policy = Policy(existence_stop_threshold=0.05, existence_samples=256)
        policy.batch = None
        policy.batch_after = None
        policy.confirmed[:10] = True
        policy.cleared[:10] = True
        policy._existence_likelihood = lambda channel: 1e-12
        stopped = policy._apply_probability_stop()
        self.assertTrue(stopped)
        self.assertTrue(policy.complete())
        self.assertEqual(policy.dynamic_stop_reason, 'posterior_threshold')
        self.assertEqual(len(policy.future), 0)

    def test_next_action_preserves_probability_stop_reason(self):
        policy = Policy(existence_stop_threshold=0.05, existence_samples=256)
        policy.batch = None
        policy.batch_after = None
        policy.confirmed[:10] = True
        policy.cleared[:10] = True
        policy._existence_likelihood = lambda channel: 1e-12
        self.assertIsNone(policy.next_action())
        self.assertTrue(policy.complete())
        self.assertEqual(policy.dynamic_stop_reason, 'posterior_threshold')

    def test_no_information_stop_is_explicitly_incomplete(self):
        policy = self.one_unknown_policy()
        policy.dynamic_prune_interval = 1
        policy.visited = [np.array([1.0, 1.0])]
        station = np.array([0.0, 0.0])
        policy.history[0] = [Observation('no_signal', *station)]
        policy.future = [station]
        action = policy.next_action()
        self.assertIsNone(action)
        self.assertFalse(policy.complete())
        self.assertEqual(policy.dynamic_stop_reason, 'no_informative_future_station')

    def test_initial_action_matches_copied_light_policy(self):
        from q4_base_policy import Policy as BasePolicy
        old = BasePolicy()
        new = Policy()
        self.assertEqual(old.next_action(), new.next_action())


if __name__ == '__main__':
    unittest.main()
