import unittest
import itertools
import math
import numpy as np
from q3_belief import (Observation, SpatialPosterior, BeliefBank, likelihood,
                       occupancy_posterior, possible_cells)
from q3_environment import Environment, fixed_stations


class PosteriorTests(unittest.TestCase):
    def test_first_negative_radial_likelihood(self):
        g = np.array([[900.,0], [1100.,0], [1250.,0], [1600.,0]])
        np.testing.assert_allclose(likelihood(g, [Observation('no_signal',0,0)]), [0,.2,.5,1])

    def test_shared_radius_excludes_impossible_candidate(self):
        g = np.array([[1200.,0]])
        positive = Observation('direction',0,0,0)
        self.assertEqual(likelihood(g, [positive, Observation('no_signal',100,0)])[0], 0)
        self.assertAlmostEqual(likelihood(g, [positive, Observation('no_signal',-200,0)])[0], .4*.005)

    def test_origin_negative_matches_analytic_probability(self):
        bank = BeliefBank()
        bank.update(1, [0,0], 'no_signal')
        z = (1000**2+1000*1500+1500**2)/(3*1800**2)
        expected = .65*(1-z)/(1-.65*z)
        self.assertAlmostEqual(bank.exists[0], expected, delta=3e-5)

    def test_repeated_measurement_no_extra_evidence(self):
        bank = BeliefBank()
        bank.update(2, [0,0], 'direction', 10.)
        before = (bank.exists.copy(), bank.spatial[1].evidence)
        self.assertFalse(bank.update(2, [0,0], 'direction', 10.))
        np.testing.assert_array_equal(bank.exists, before[0])
        self.assertEqual(bank.spatial[1].evidence, before[1])
        with self.assertRaises(ValueError):
            bank.update(2, [0,0], 'direction', 11.)

    def test_count_constraint_and_original_existence(self):
        z0, z1 = np.ones(20), np.ones(20)
        z0[:16] = 0
        p, count = occupancy_posterior(z0,z1)
        np.testing.assert_allclose(p, [1]*16+[0]*4)
        self.assertEqual(count[16], 1)
        z0, z1 = np.ones(20), np.ones(20)
        z1[10:] = 0
        p, _ = occupancy_posterior(z0,z1)
        np.testing.assert_allclose(p, [1]*10+[0]*10)

    def test_count_marginals_against_exhaustive_small_unknown_set(self):
        z0,z1=np.ones(20),np.ones(20)
        z0[:9]=0
        z1[13:]=0
        z1[9:13]=[.1,.3,.6,.9]
        actual,_=occupancy_posterior(z0,z1)
        denominator=0.; numerator=np.zeros(4)
        for bits in itertools.product([0,1],repeat=4):
            n=9+sum(bits)
            if not 10 <= n <= 16:
                continue
            weight=math.prod(z1[9+j] if bit else 1 for j,bit in enumerate(bits))/math.comb(20,n)
            numerator+=weight*np.array(bits);denominator+=weight
        np.testing.assert_allclose(actual[9:13],numerator/denominator,rtol=1e-12)

    def test_near_clear_and_postclear_negative(self):
        bank = BeliefBank()
        bank.update(3, [200,100], 'near')
        self.assertAlmostEqual(bank.predict(3,[200,100])['clear'],1.)
        bank.update(3, [200,100], 'success')
        before = bank.exists.copy()
        bank.update(3, [200,100], 'no_signal')
        np.testing.assert_array_equal(bank.exists,before)
        self.assertEqual(bank.exists[2],1)
        self.assertEqual(bank.predict(3,[200,100])['receive'],0)

    def test_seven_negatives_certify_absence(self):
        h = [Observation('no_signal',*q) for q in fixed_stations()[1:]]
        post = SpatialPosterior(h)
        self.assertTrue(post.certified_empty)
        self.assertEqual(post.evidence,0.)

    def test_conservative_cell_does_not_use_only_center(self):
        c = np.array([[100.,0]])
        self.assertTrue(possible_cells(c, 10., [Observation('near',109,0)])[0])

    def test_environment_fixed_error_and_clear_channel(self):
        env = Environment(3)
        c = next(iter(env.sources))
        q = np.array(env.sources[c]['position']) + [100.,0]
        a = env.act('measure',c,q)
        b = env.act('measure',c,q)
        self.assertEqual(a['angle'],b['angle'])
        env.act('clear',c,env.sources[c]['position'])
        self.assertEqual(env.channel,c)
        other = c%20+1
        env.act('clear',other,[0,0])
        self.assertEqual(env.channel,c)
        self.assertAlmostEqual(env.time, env.distance/5+5*env.measurements+env.switches+3*env.attempts+2*env.successes)


if __name__ == '__main__':
    unittest.main()
