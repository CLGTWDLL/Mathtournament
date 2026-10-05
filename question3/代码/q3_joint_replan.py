"""Exploratory recourse variant; prior frozen joint-site code is unchanged."""
from q3_joint_site import JointSitePolicy


class JointReplanPolicy(JointSitePolicy):
    def __init__(self,mode='joint_replan',**kwargs):
        super().__init__(mode='joint_site',**kwargs)
        self.release_after_feedback=False

    def next_action(self):
        previous=self.transaction
        action=super().next_action()
        if previous is not None and self.transaction is None and action is not None:
            assert action['action']=='measure' and action['channel']==previous['channel']
            self.release_after_feedback=True
        return action

    def accept_feedback(self,result):
        super().accept_feedback(result)
        if self.release_after_feedback:
            # The shared measurement and coverage are complete. Its source
            # remains known and unresolved, but clearance order is reconsidered
            # using all new batch feedback and this new bearing.
            self.active_source=None;self.exit_point=None
            self.release_after_feedback=False
