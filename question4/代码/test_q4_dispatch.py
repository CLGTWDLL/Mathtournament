import unittest
import numpy as np
from q4_dispatch_policy import choose_task, Policy
from q4_route_value import insert_anchors
from q4_environment import Environment


class Tests(unittest.TestCase):
    def test_defer_when_later_gap_is_better(self):
        choice, info = choose_task([0, 0], [[1000, 0], [1000, 1000]], [[1000, 300]])
        self.assertIsNone(choice)
        self.assertEqual(info['baseline_choice'], 0)
        self.assertTrue(info['changed'])
        self.assertAlmostEqual(info['route_length_m'], 2000.)

    def test_other_source_first_can_save_return_trip(self):
        choice, info = choose_task([0, 0], [[1000, 0]], [[500, 100], [-100, 0]])
        self.assertEqual(info['baseline_choice'], 0)
        self.assertEqual(choice, 1)

    def test_anchor_insertion_never_drops_or_reorders_patrol(self):
        rng = np.random.default_rng(911)
        for _ in range(50):
            stations = rng.uniform(-1800, 1800, (8, 2))
            anchors = rng.uniform(-1500, 1500, (4, 2))
            route = insert_anchors([0, 0], stations, anchors)
            indices = [int(np.argmin(np.linalg.norm(route-p, axis=1))) for p in stations]
            self.assertEqual(indices, sorted(indices))
            self.assertEqual(len(route), len(stations)+len(anchors))
            for p in np.vstack((stations, anchors)):
                self.assertLess(np.linalg.norm(route-p, axis=1).min(), 1e-10)

    def test_complete_episode_and_feedback_clocks(self):
        env, policy = Environment(1140000), Policy('dispatch')
        for _ in range(5000):
            a = policy.next_action()
            if a is None:
                break
            result = env.act(a['action'], a['channel'], a['position'])
            policy.accept_feedback(result)
            self.assertEqual(env.channel, policy.channel)
            np.testing.assert_allclose(env.position, policy.position)
        self.assertTrue(policy.complete())
        self.assertEqual(env.successes, env.n)


if __name__ == '__main__':
    unittest.main()
