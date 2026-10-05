import unittest
from unittest.mock import patch
import numpy as np
from q4_coverage import certified_cells, domain_cells
from q4_sparse_backbone import load_backbone, certified, independent_cover_check, Policy, HALF
from q4_environment import Environment


class Tests(unittest.TestCase):
    def test_sparse_initialization_never_builds_dense_route(self):
        with patch('q4_policy.initial_stations', side_effect=AssertionError('dense route built')):
            policy = Policy('dispatch')
        self.assertEqual(len(policy.future), 29)

    def test_cached_route_does_not_share_mutable_state(self):
        route = load_backbone()
        expected = route.copy()
        route[0] = [99999., 99999.]
        np.testing.assert_array_equal(load_backbone(), expected)

    def test_independent_audit_rejects_missing_station(self):
        stations = np.vstack((np.zeros(2), load_backbone()))
        self.assertFalse(independent_cover_check(np.delete(stations, 1, axis=0)))

    def test_backbone_certificate_on_fine_and_qhull_checks(self):
        route = load_backbone()
        stations = np.vstack((np.zeros(2), route))
        self.assertTrue(certified(stations).all())
        self.assertTrue(independent_cover_check(stations))
        self.assertLess(len(route), 37)

    def test_coarse_certificate_implies_fine_certificate(self):
        # The frozen 875 m mesh is certified on 100 m squares; convexity makes
        # every 25 m sub-square certified as well (finer grid is only less conservative).
        from q4_coverage import initial_stations
        stations = np.vstack((np.zeros(2), initial_stations()))
        self.assertTrue(certified_cells(stations, domain_cells(100.), 50.).all())
        self.assertTrue(certified(stations).all())

    def test_removing_any_station_breaks_certificate(self):
        route = load_backbone()
        stations = np.vstack((np.zeros(2), route))
        for i in range(1, len(stations)):
            self.assertFalse(certified(np.delete(stations, i, axis=0)).all())

    def test_complete_episode_with_valid_certificates(self):
        env, policy = Environment(1140002), Policy('dispatch')
        for _ in range(5000):
            a = policy.next_action()
            if a is None:
                break
            policy.accept_feedback(env.act(a['action'], a['channel'], a['position']))
        self.assertTrue(policy.complete())
        self.assertEqual(env.successes, env.n)
        for channel, reason in policy.absence_reason.items():
            if reason == 'continuous_directional_cover':
                points = [np.array([o.x, o.y]) for o in policy.history[channel-1] if o.kind == 'no_signal']
                self.assertTrue(independent_cover_check(points))


if __name__ == '__main__':
    unittest.main()
