import unittest
from types import SimpleNamespace
import numpy as np
from q3_belief import Observation
from q3_policy import (AutonomousPolicy,CLEAR_CELL_HALF,CLEAR_CELL_SIZE,
                       CLEAR_GRID,Cover,Proposal,STATIONS)


class AutonomousTests(unittest.TestCase):
    def test_28m_clearance_grid_is_gap_free_and_each_cell_is_clearable(self):
        axis=np.unique(CLEAR_GRID[:,0])
        self.assertTrue(np.allclose(np.diff(axis),CLEAR_CELL_SIZE))
        self.assertLessEqual(axis[0]-CLEAR_CELL_HALF,-1800.)
        self.assertGreaterEqual(axis[-1]+CLEAR_CELL_HALF,1800.)
        self.assertLess(CLEAR_CELL_HALF*np.sqrt(2),20.)

    def test_no_posterior_probability_stopping(self):
        policy=AutonomousPolicy()
        policy.bank.exists[:]=0
        self.assertFalse(policy.complete())
        self.assertIsNotNone(policy.next_action())

    def test_rejected_action_does_not_move_or_update(self):
        policy=AutonomousPolicy()
        policy.next_action()
        with self.assertRaises(ValueError):policy.accept_feedback(dict(accepted=False))
        np.testing.assert_array_equal(policy.position,[0,0])
        self.assertEqual(policy.steps,0)
        self.assertIsNotNone(policy.pending)

    def test_clear_does_not_switch_receiver(self):
        policy=AutonomousPolicy()
        policy.channel=7
        policy.pending=Proposal('clear',2,np.array([100.,200.]),5.,'posterior_clear')
        policy.accept_feedback(dict(accepted=True,kind='success'))
        self.assertEqual(policy.channel,7)
        self.assertTrue(policy.cleared[1])
        self.assertTrue(policy.confirmed[1])

    def test_failed_cover_attempt_strictly_reduces_cells(self):
        history=[Observation('direction',0,0,0)]
        before=Cover(history)
        q=before.c[0]
        after=Cover(history+[Observation('no_target_in_range',*q)])
        self.assertLess(len(after.c),len(before.c))

    def test_near_feedback_clears_without_extra_measurement(self):
        policy=AutonomousPolicy(mode='sequential')
        policy.pending=Proposal('measure',3,np.array([100.,0]),5.,'coverage_scan')
        policy.accept_feedback(dict(accepted=True,kind='near'))
        action=policy.next_action()
        self.assertEqual(action['action'],'clear')
        self.assertEqual(action['channel'],3)
        np.testing.assert_array_equal(action['position'],[100,0])

    def test_numeric_failure_falls_back_to_geometric_clearance(self):
        policy=AutonomousPolicy(mode='sequential')
        def fail(*args,**kwargs):raise ArithmeticError('Injected integration failure')
        policy.bank.update=fail
        policy.pending=Proposal('measure',3,np.array([-100.,0]),5.,'coverage_scan')
        policy.accept_feedback(dict(accepted=True,kind='direction',angle=0.))
        self.assertFalse(policy.numeric_ok)
        action=policy.next_action()
        self.assertEqual(action['action'],'clear')
        self.assertEqual(action['reason'],'finite_cover')

    def test_original_count_upper_bound_includes_cleared_sources(self):
        policy=AutonomousPolicy()
        policy.numeric_ok=False
        policy.confirmed[:15]=True;policy.cleared[:15]=True
        policy.pending=Proposal('clear',16,np.zeros(2),5.,'posterior_clear')
        policy.accept_feedback(dict(accepted=True,kind='success'))
        self.assertTrue(policy.complete())
        self.assertTrue(policy.absent[16:].all())

    def test_posterior_tail_point_covers_every_remaining_support(self):
        policy=AutonomousPolicy(mode='joint',scan_scale=.5)
        policy.confirmed[:10]=True;policy.cleared[:10]=True
        for i in range(10,20):
            policy.bank.exists[i]=.01
            policy.station_seen[i,0]=True
            policy.history[i]=[Observation('no_signal',0,0)]
            policy.bank.spatial[i]=SimpleNamespace(
                centers=np.array([[1490.,0],[1510.,0]]),half=np.array([5.,5.]),
                predict=lambda q:dict(receive=.01,near=0.,clear=0.))
        action=policy.next_action()
        self.assertEqual(action['reason'],'posterior_certification')
        q=np.array(action['position'])
        self.assertLess(np.linalg.norm(q),np.linalg.norm(STATIONS[1]))
        for i in range(10,20):
            p=policy.bank.spatial[i]
            self.assertLessEqual(np.max(np.linalg.norm(np.abs(p.centers-q)+p.half[:,None],axis=1)),1000)
        self.assertFalse(policy.complete())


if __name__=='__main__':unittest.main()
