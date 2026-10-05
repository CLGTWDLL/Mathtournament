import copy
import unittest
import numpy as np
from q3_belief import Observation
from q3_policy import Proposal
from q3_backbone import covered,ring,union_cover
from q3_joint_site import JointSitePolicy
from q3_joint_replan import JointReplanPolicy


class JointSiteTests(unittest.TestCase):
    def observed(self):
        p=JointSitePolicy()
        p.pending=Proposal('measure',1,np.zeros(2),5.,'backbone_scan')
        p.accept_feedback(dict(accepted=True,kind='direction',angle=0.))
        return p

    def test_preview_preserves_baseline_rng_position_channel_and_history(self):
        p=self.observed();rng=copy.deepcopy(p.rng.bit_generator.state)
        p.peek_local(0,np.array([300.,100.]),np.array([1250.,0.]),8)
        self.assertEqual(p.rng.bit_generator.state,rng)
        np.testing.assert_array_equal(p.position,[0.,0.])
        self.assertEqual(p.channel,1);self.assertEqual(len(p.history[0]),1)
        self.assertIsNone(p.exit_point)

    def test_near_scenario_includes_clearance_and_success_to_exit(self):
        p=self.observed();q=np.array([500.,0.]);end=np.array([1000.,0.])
        proposal=Proposal('measure',1,q,0.,'localize')
        value=p.samples_cost(0,proposal,np.zeros(2),end,8,q[None,:],np.array([1200.]),np.array([0.]))
        self.assertAlmostEqual(value[0],100+6+5+100)

    def test_shared_site_discharges_station_before_known_measurement(self):
        p=JointSitePolicy();p.numeric_ok=False
        p.confirmed[:10]=True;p.cleared[:9]=True
        p.history[9]=[Observation('direction',0.,0.,0.)]
        for c in range(10,20):p.history[c]=[Observation('no_signal',0.,0.)]
        p.remaining[:]&=~covered(np.zeros(2));p.nodes=ring()
        q=p.nodes[0].copy();end=p.nodes[1].copy()
        p.batch=q.copy();p.batch_reason='backbone_scan'
        p.transaction=dict(channel=10,position=q,exit=end,new_expected=100.)
        for c in range(11,21):
            action=p.next_action();self.assertEqual(action['channel'],c)
            np.testing.assert_array_equal(action['position'],q)
            p.accept_feedback(dict(accepted=True,kind='no_signal'))
        action=p.next_action()
        self.assertEqual(action['channel'],10);self.assertEqual(action['action'],'measure')
        np.testing.assert_array_equal(action['position'],q)
        np.testing.assert_array_equal(p.exit_point,end)
        self.assertFalse(any(np.linalg.norm(n-q)<1 for n in p.nodes))
        self.assertFalse(np.any(p.unresolved_area() & ~union_cover(p.nodes)))

    def test_joint_proposal_obeys_local_measurement_cap(self):
        p=self.observed();p.nodes=ring();p.local_measurements[0]=5
        p.planned=[('source',0,np.array([700.,0.])),('station',0,p.nodes[0])]
        self.assertIsNone(p.propose_joint())

    def test_recourse_releases_clearance_order_but_keeps_source_unresolved(self):
        p=JointReplanPolicy();p.numeric_ok=False
        p.confirmed[:10]=True;p.cleared[:9]=True
        p.history[9]=[Observation('direction',0.,0.,0.)]
        nodes=ring();q=nodes[0].copy();end=nodes[1].copy()
        p.remaining[:]&=~covered(np.zeros(2));p.remaining[:]&=~covered(q)
        p.nodes=nodes[1:];p.position=q.copy();p.batch=None
        p.transaction=dict(channel=10,position=q,exit=end,new_expected=100.)
        action=p.next_action()
        self.assertEqual(action['channel'],10)
        self.assertTrue(p.release_after_feedback)
        p.accept_feedback(dict(accepted=True,kind='direction',angle=0.))
        self.assertIsNone(p.active_source);self.assertIsNone(p.exit_point)
        self.assertTrue(p.confirmed[9]);self.assertFalse(p.cleared[9])
        self.assertFalse(p.release_after_feedback)


if __name__=='__main__':unittest.main()
