"""Select a source or patrol BEFORE committing to local clearance.

Posterior anchors approximate unresolved source tasks. Finite reference routes
keep every future patrol station; continuous certificates remain unchanged.
"""
import numpy as np
from q4_feedback_policy import Policy as GeometryPolicy
from q4_route_policy import Policy as RoutePolicy
from q4_route_value import insert_anchors


def choose_task(start, stations, anchors):
    start = np.asarray(start)
    stations = np.asarray(stations).reshape(-1, 2)
    anchors = np.asarray(anchors).reshape(-1, 2)
    plans = []
    def add(route, tag):
        length = np.linalg.norm(np.diff(np.vstack((start, route)), axis=0), axis=1).sum()
        plans.append((float(length), route, tag))
    add(insert_anchors(start, stations, anchors), 'free_insertion')
    if len(stations):
        rest = insert_anchors(stations[0], stations[1:], anchors)
        add(np.vstack((stations[0], rest)), 'patrol_first')
    # Baseline's preferred source is always among these three: it minimizes
    # this same current-to-next-station insertion distance.
    exit_point = stations[0] if len(stations) else start
    insertion = np.linalg.norm(anchors-start, axis=1)+np.linalg.norm(anchors-exit_point, axis=1)-np.linalg.norm(exit_point-start)
    for j in np.argsort(insertion)[:3]:
        add(insert_anchors(start, stations, anchors, int(j)), 'source_first')
    length, route, tag = min(plans, key=lambda row: row[0])
    if len(stations) and np.linalg.norm(route[0]-stations[0]) < 1e-7:
        choice = None
    else:
        choice = int(np.argmin(np.linalg.norm(anchors-route[0], axis=1)))
    old_choice = int(np.argmin(insertion)) if len(anchors) else None
    if len(stations) and insertion[old_choice] >= 700:
        old_choice = None
    return choice, dict(route_length_m=length, route_count=len(plans), plan=tag,
                        baseline_choice=old_choice, selected_choice=choice,
                        changed=choice != old_choice)


class Policy(RoutePolicy):
    def __init__(self, variant='dispatch', particles=32, seed=9411):
        if variant not in ('dispatch', 'dispatch_route'):
            raise ValueError(variant)
        super().__init__('route', particles, seed)
        self.dispatch_variant = variant
        self.dispatch_decision = None

    def local_action(self, i):
        if self.dispatch_variant == 'dispatch':
            self.last_decision = None
            return GeometryPolicy.local_action(self, i)
        return super().local_action(i)

    def emit(self, action, i, q, reason, estimated=0.):
        result = super().emit(action, i, q, reason, estimated)
        if self.dispatch_decision is not None:
            result['dispatch_decision'] = self.dispatch_decision
            self.pending['dispatch_decision'] = self.dispatch_decision
            self.dispatch_decision = None
        return result

    def next_action(self):
        if self.pending is not None:
            raise RuntimeError('Missing accepted feedback')
        while True:
            if self.complete():
                return None
            if self.confirmed.sum() == 16:
                for i in self.unknown():
                    self.absent[i] = True
                    self.absence_reason[int(i)+1] = 'sixteen_confirmed'
            if self.batch is not None:
                while self.batch['queue']:
                    i = self.batch['queue'].pop(0)
                    if not self.confirmed[i] and not self.absent[i]:
                        return self.emit('measure', i, self.batch['q'], 'existence_batch')
                self.visited.append(np.round(self.batch['q'], 6))
                remove = self.batch['remove']
                self.future = [p for j, p in enumerate(self.future) if j not in remove]
                self.removed_sites += len(remove)
                self.batch = None
                self.after_batch_commit()
                if len(self.unknown()) and self.certified(self.visited).all():
                    for i in self.unknown():
                        self.absent[i] = True
                        self.absence_reason[int(i)+1] = 'continuous_directional_cover'
                continue
            if self.batch_after is not None:
                q, remove = self.batch_after
                self.batch_after = None
                if len(self.unknown()):
                    self.start_batch(q, remove)
                    self.joint_batches += 1
                continue
            known = self.known()
            if len(known):
                anchors = []
                for i in known:
                    cover, post = self.local_state(int(i))
                    anchors.append(post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0))
                stations = self.future if len(self.unknown()) else []
                choice, info = choose_task(self.position, stations, anchors)
                info['known_channels'] = (known+1).tolist()
                self.dispatch_decision = info
                if choice is not None:
                    i = int(known[choice])
                    action, q, value, reason = self.local_action(i)
                    q = np.round(q, 6)
                    if action == 'measure':
                        remove = self.replacement(q)
                        if remove:
                            self.batch_after = (q.copy(), remove)
                    return self.emit(action, i, q, reason, value)
            if self.future and len(self.unknown()):
                self.start_batch(self.future[0], [0])
                continue
            if self.complete():
                return None
            if len(self.unknown()):
                raise ArithmeticError('Patrol exhausted without a directional certificate')
