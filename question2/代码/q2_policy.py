"""Executable Q2 observation-to-action policy; no hidden source inputs.

Second point: semi-analytic selection (or an explicitly supplied offline point).
Continuation: posterior planning and finite continuous-region cover. This is a
hybrid policy; only second-point selection is replaced by the analytic solver.
No network operations are performed by this module.
"""
import numpy as np
from solve_q2_core import CoverModel, RAD, EPS, local_origin, world_point
from solve_q2_multistep import choose_measurement
from q2_analytic import RayPrior


def feedback_cells(model, position, kind, angle=None):
    """Conservative support update from public feedback only, in local axes."""
    s = np.asarray(position)
    c, h = model.c, model.h
    active = np.ones(len(c), bool)
    if kind == 'near':
        return np.zeros(len(c), bool)
    if kind == 'direction':
        if angle is None:
            raise ValueError('direction requires an angle')
        theta = angle*RAD
        for t, sign in [(theta-EPS, 1), (theta+EPS, -1)]:
            n = sign*np.array([-np.sin(t), np.cos(t)])
            active &= (c-s)@n+h*np.abs(n).sum() >= -1e-8
        lo = np.linalg.norm(np.maximum(np.abs(c-s)-h, 0), axis=1)
        hi = np.linalg.norm(np.abs(c-s)+h, axis=1)
        active &= (lo <= 1500+1e-8) & (hi > 5)
    elif kind == 'no_signal':
        hi = np.linalg.norm(np.abs(c-s)+h, axis=1)
        active &= (hi > 1000) & (c@s-h*np.abs(s).sum() < s@s/2+1e-8)
    else:
        raise ValueError('Unknown measurement feedback')
    return active


class Q2Policy:
    def __init__(self, first_position, first_bearing_deg, channel=1,
                 max_extra_measurements=5, weight=1., second_local=None,
                 planning_particles=192, seed=20260910, cell_m=25.):
        if not 1 <= channel <= 20 or max_extra_measurements < 0:
            raise ValueError('Invalid channel or measurement budget')
        self.sc = dict(first=list(first_position), bearing=float(first_bearing_deg), key='runtime')
        self.model = CoverModel(self.sc, cell_m)
        self.channel, self.budget = channel, max_extra_measurements
        self.nplan, self.weight = planning_particles, weight
        self.rng = np.random.default_rng(seed)
        self.active = np.ones(len(self.model.c), bool)
        self.history = [(np.zeros(2), 0.)]
        self.pos = np.zeros(2)
        self.count = 0
        self.near = self.clearing = self.done = False
        self.pending = None
        self.second = None if second_local is None else np.array(second_local, float)
        if self.second is not None and np.linalg.norm(self.second) < .1:
            raise ValueError('A repeated first position supplies no new observation')

    def next_action(self):
        if self.done:
            return None
        if self.pending is not None:
            raise RuntimeError('Accept feedback before requesting another action')
        if self.near:
            kind, point = 'clear', self.pos.copy()
        elif not self.clearing and self.count < self.budget:
            if self.count == 0 and self.second is not None:
                point = self.second.copy()
            elif self.count == 0:
                try:
                    point, _, _ = RayPrior(self.sc).select(self.weight)
                    # Mirror toward the known world-disk center where asymmetric.
                    point[1] *= 1 if local_origin(self.sc)[1] >= 0 else -1
                except ValueError:
                    # A central-ray approximation may fail for an outside
                    # station whose wedge grazes the disk. Use full posterior.
                    point = choose_measurement(self.model, self.active, self.pos,
                                               self.history, self.nplan, self.rng)
            else:
                point = choose_measurement(self.model, self.active, self.pos,
                                           self.history, self.nplan, self.rng)
            if point is None:
                self.clearing = True
                return self.next_action()
            if any(np.linalg.norm(point-p) < .1 for p, _ in self.history):
                raise RuntimeError('Planner attempted a repeated measurement')
            kind = 'measure'
        else:
            self.clearing = True
            if not self.active.any():
                raise RuntimeError('Feedback contradicts the bounded-error model')
            distances = np.linalg.norm(self.model.c-self.pos, axis=1)
            index = np.argmin(np.where(self.active, distances, np.inf))
            kind, point = 'clear', self.model.c[index].copy()
        self.pending = (kind, point.copy())
        world = world_point(self.sc, point)
        return dict(action=kind, position=dict(x=float(world[0]), y=float(world[1])), channel=self.channel)

    def accept_feedback(self, result, bearing_deg=None):
        if self.pending is None:
            raise RuntimeError('There is no pending action')
        kind, point = self.pending
        # Validate before mutating state; malformed feedback can be corrected.
        if kind == 'measure':
            if result not in ['direction', 'near', 'no_signal']:
                raise ValueError('Invalid measure result')
            if result == 'direction' and (bearing_deg is None or not np.isfinite(bearing_deg)):
                raise ValueError('direction requires finite bearing_deg')
            angle = None if result != 'direction' else float((bearing_deg-self.sc['bearing']+180)%360-180)
            updated = self.active & feedback_cells(self.model, point, result, angle)
            if result != 'near' and not updated.any():
                raise RuntimeError('Feedback contradicts the bounded-error model')
            self.active = updated
            self.count += 1
            self.near = result == 'near'
            if not self.near:
                self.history.append((point.copy(), angle))
        else:
            if result == 'success':
                self.done = True
            elif result == 'no_target_in_range':
                if self.near:
                    raise RuntimeError('Clear failed after a same-position near response')
                covered = np.sum((np.abs(self.model.c-point)+self.model.h)**2, axis=1) <= 400+1e-9
                self.active &= ~covered
                if not self.active.any():
                    raise RuntimeError('No feasible cells remain after clear failure')
            else:
                raise ValueError('Invalid clear result')
        self.pos = point.copy()
        self.pending = None
