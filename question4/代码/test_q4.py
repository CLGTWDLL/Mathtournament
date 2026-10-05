import unittest
import numpy as np
from q3_belief import Observation
from q4_belief import radio_masses, likelihood, conservative_cells, radius_orientation_pieces
from q4_coverage import certified_cells, initial_stations
from q4_environment import Environment
from run_q4 import independent_cover_check


class TestQ4(unittest.TestCase):
    def test_shared_orientation_not_redrawn(self):
        g = np.array([[0., 0.]])
        one = [Observation('no_signal', 600., 0.)]
        same = one+[Observation('no_signal', 800., 0.)]
        opposite = one+[Observation('no_signal', -800., 0.)]
        self.assertAlmostEqual(radio_masses(g, one)[1][0], .5)
        self.assertAlmostEqual(radio_masses(g, same)[1][0], .5)
        self.assertAlmostEqual(radio_masses(g, opposite)[1][0], 0.)

    def test_radius_orientation_integral_against_monte_carlo(self):
        rng = np.random.default_rng(7114)
        theta, r = rng.uniform(0, 2*np.pi, 400000), rng.uniform(1000, 1500, 400000)
        u = np.column_stack((np.cos(theta), np.sin(theta)))
        g = np.array([100., 20.])
        histories = [
            [Observation('direction', 1300., 20., 180.), Observation('no_signal', -650., -300.)],
            [Observation('no_signal', 1200., 300.), Observation('no_signal', 500., 1200.)],
            [Observation('direction', 800., 400., 0.), Observation('direction', 400., 900., 0.), Observation('no_signal', -1000., 400.)]]
        for hist in histories:
            om, dr = radio_masses(g[None], hist)
            okd, oko = np.ones(len(r), bool), np.ones(len(r), bool)
            for o in hist:
                d = np.array([o.x, o.y])-g
                receive = (np.linalg.norm(d) <= r)
                direct = receive & (u@d >= 0)
                okd &= direct if o.kind != 'no_signal' else ~direct
                oko &= receive if o.kind != 'no_signal' else ~receive
            self.assertLess(abs(okd.mean()-dr[0]), .003)
            self.assertLess(abs(oko.mean()-om[0]), .003)

    def test_repeated_reading_not_double_counted(self):
        o = Observation('direction', 0., 0., 0.)
        p = np.array([[600., 0.]])
        self.assertEqual(likelihood(p, [o])[0], likelihood(p, [o, o])[0])

    def test_no_signal_does_not_delete_true_location(self):
        cells = np.array([[612.5, 12.5]])
        self.assertTrue(conservative_cells(cells, 12.5, [Observation('no_signal', 0., 0.)])[0])

    def test_zero_probability_does_not_erase_boundary_support(self):
        h = [Observation('direction', -600., 0., 0.), Observation('direction', 600., 0., 180.)]
        # Only exactly north/south orientation can face both; continuous prior
        # mass is zero, but geometric support must still retain this location.
        self.assertLess(radio_masses(np.array([[0., 0.]]), h)[1][0], 1e-14)
        self.assertTrue(conservative_cells(np.array([[0., 0.]]), 12.5, h)[0])

    def test_cover_requires_surrounding_not_overlap_count(self):
        p = np.array([[500., 0.], [500., 200.], [700., -200.]])
        self.assertFalse(certified_cells(p, np.array([[0., 0.]]), half=10)[0])
        p = np.array([[600., 0.], [-300., 520.], [-300., -520.]])
        self.assertTrue(certified_cells(p, np.array([[0., 0.]]), half=10)[0])

    def test_initial_mesh_independent_continuous_proof(self):
        stations = np.vstack(([0., 0.], initial_stations()))
        self.assertTrue(certified_cells(stations).all())
        self.assertTrue(independent_cover_check(stations))

    def test_outward_source_and_optical_clear(self):
        env = Environment(1140000)
        c = next(iter(env.sources))
        env.sources[c] = dict(position=[1800., 0.], radius=1000., directional=True, theta=0.)
        self.assertEqual(env.act('measure', c, [1799., 0.])['kind'], 'no_signal')
        self.assertEqual(env.act('measure', c, [1801., 0.])['kind'], 'near')
        self.assertEqual(env.act('measure', c, [1900., 0.])['kind'], 'direction')
        old_channel = env.channel
        self.assertEqual(env.act('clear', c, [1790., 0.])['kind'], 'success')
        self.assertEqual(env.channel, old_channel)


if __name__ == '__main__':
    unittest.main()
