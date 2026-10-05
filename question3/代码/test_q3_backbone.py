import unittest
import numpy as np
from q3_belief import Observation
from q3_backbone import BackbonePolicy,ExitCover,CELLS,covered,ring,union_cover,shift_with_certificate


class BackboneTests(unittest.TestCase):
    def test_origin_and_six_ring_sites_cover_whole_target_squares(self):
        for phase in (0.,.17,.5):
            self.assertTrue(np.all(covered(np.zeros(2)) | union_cover(ring(phase))))

    def test_moving_station_keeps_essential_cells_covered(self):
        nodes=ring();required=~covered(np.zeros(2)) & ~union_cover(nodes[1:])
        self.assertTrue(required.any())
        q=shift_with_certificate(nodes[0],np.array([1800.,800.]),required)
        self.assertGreater(np.linalg.norm(q-nodes[0]),1.)
        self.assertFalse(np.any(required & ~covered(q)))

    def test_exit_cost_uses_actual_success_location(self):
        p=BackbonePolicy();p.exit_point=np.array([100.,0.])
        c=ExitCover([Observation('direction',0.,0.,0.)],p)
        g=c.c[:1];start=g[0].copy()
        cost=c.costs(start,np.ones((1,len(c.c)),bool),g)
        self.assertAlmostEqual(cost[0],5+np.linalg.norm(start-p.exit_point)/5)
        cost=c.costs(start,np.zeros((1,len(c.c)),bool),g)
        self.assertAlmostEqual(cost[0],np.linalg.norm(start-p.exit_point)/5)

    def test_insertion_charges_rejoin_and_can_append_without_return(self):
        p=BackbonePolicy()
        nodes=[np.array([100.,0.]),np.array([200.,0.])]
        plan=p.insertion_plan(nodes,{0:np.array([50.,0.]),1:np.array([250.,0.])})
        self.assertEqual([(k,i) for k,i,q in plan],[('source',0),('station',0),('station',1),('source',1)])

    def test_deletion_requires_complete_remaining_coverage(self):
        p=BackbonePolicy();p.nodes=ring();p.remaining[:]&=~covered(np.zeros(2))
        # Probability zero cannot delete any required coverage.
        p.bank.exists[:]=0
        p.optimize_nodes()
        self.assertFalse(np.any(p.unresolved_area() & ~union_cover(p.nodes)))
        self.assertFalse(p.complete())

    def test_original_batch_checks_all_channels_before_leaving(self):
        p=BackbonePolicy();p.numeric_ok=False
        for c in range(1,21):
            a=p.next_action()
            self.assertEqual(a['channel'],c)
            np.testing.assert_array_equal(a['position'],[0,0])
            # Public positive responses for ten channels; the remaining are
            # negative. Avoid invoking local planning until batch complete.
            p.accept_feedback(dict(accepted=True,kind='direction',angle=float(c)) if c<=10 else dict(accepted=True,kind='no_signal'))
        self.assertEqual(int(p.confirmed.sum()),10)
        self.assertFalse(p.complete())


if __name__=='__main__':unittest.main()
