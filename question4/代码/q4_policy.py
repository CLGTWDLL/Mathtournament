"""First complete Q4 prototype: continuous certificate + adaptive localization.

This is a correctness baseline with joint-site replacement, not a global
optimum or a wholesale port of every Q3 optimization. Public feedback only.
"""
import numpy as np
from q3_belief import Observation, EPS
from q3_policy import Cover as OmniCover, GRID
from q2_analytic import RayPrior
from q4_belief import KnownPosterior, conservative_cells
from q4_coverage import initial_stations, certified_cells


class Cover(OmniCover):
    def __init__(self, history):
        self.c = GRID[conservative_cells(GRID, 12.5, history)]
        if not len(self.c):
            raise ArithmeticError('Confirmed source has empty conservative cover')
        self.distance = np.linalg.norm(self.c[:, None]-self.c[None], axis=2)

    def measurement_masks(self, q, g, radius, theta, directional, noise):
        relg = g-q
        d = np.linalg.norm(relg, axis=1)
        front = np.sum((q-g)*np.column_stack((np.cos(theta), np.sin(theta))), axis=1) >= 0
        receive = (d <= radius) & (~directional | front)
        near, direction = receive & (d <= 5), receive & (d > 5)
        mask = np.ones((len(g), len(self.c)), bool)
        rel = self.c-q
        dmin = np.linalg.norm(np.maximum(np.abs(rel)-12.5, 0), axis=1)
        dmax = np.linalg.norm(np.abs(rel)+12.5, axis=1)
        ids = np.flatnonzero(direction)
        if len(ids):
            angle = np.deg2rad(np.round(np.rad2deg(np.arctan2(relg[ids, 1], relg[ids, 0]))+noise[ids], 2))
            for t, sign in ((angle-EPS, 1), (angle+EPS, -1)):
                normal = sign*np.column_stack((-np.sin(t), np.cos(t)))
                mask[ids] &= normal@rel.T+12.5*np.abs(normal).sum(axis=1)[:, None] >= -1e-9
            mask[ids] &= (dmax > 5) & (dmin <= 1500)
        # Misses do not shrink this spatial-only fallback. The probabilistic
        # posterior nevertheless uses their complete orientation/radius data.
        mask[near] = False
        return mask, near


class Policy:
    def __init__(self, mode='joint', particles=32, seed=9411):
        if mode not in ('sequential', 'joint'):
            raise ValueError(mode)
        self.mode, self.particles = mode, particles
        self.rng = np.random.default_rng(seed)
        self.position, self.channel = np.zeros(2), 1
        self.history = [[] for _ in range(20)]
        self.confirmed, self.cleared, self.absent = (np.zeros(20, bool) for _ in range(3))
        self.local_rf, self.free_clear = np.zeros(20, int), np.zeros(20, int)
        self.future = list(self.initial_stations())
        self.visited, self.cache = [], {}
        self.pending, self.batch = None, None
        self.batch_after = None
        self.steps, self.joint_batches, self.removed_sites = 0, 0, 0
        self.numeric_fallbacks = 0
        self.absence_reason = {}
        self.start_batch(np.zeros(2), [])

    def initial_stations(self):
        return initial_stations()

    def certified(self, stations):
        return certified_cells(stations)

    def after_batch_commit(self):
        """Optional patrol update after all unknown channels were measured."""

    def unknown(self):
        return np.flatnonzero(~(self.confirmed | self.absent))

    def known(self):
        return np.flatnonzero(self.confirmed & ~self.cleared)

    def complete(self):
        return bool(np.all(self.cleared | self.absent))

    def start_batch(self, q, remove):
        ids = self.unknown().tolist()
        ids.sort(key=lambda i: (i+1 != self.channel, i))
        self.batch = dict(q=np.array(q), queue=ids, remove=remove)

    def local_state(self, i):
        key = (i, len(self.history[i]))
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != i}
            cover = Cover(self.history[i])
            post = KnownPosterior(cover.c, self.history[i])
            self.cache[key] = cover, post
            self.numeric_fallbacks += int(not post.numeric_ok)
        return self.cache[key]

    def local_action(self, i):
        hist = self.history[i]
        near = next((o for o in hist if o.kind == 'near'), None)
        if near:
            return 'clear', np.array([near.x, near.y]), 5., 'near_clear'
        cover, post = self.local_state(i)
        nearest = cover.c[np.argmin(np.linalg.norm(cover.c-self.position, axis=1))]
        if not post.numeric_ok:
            return 'clear', nearest, np.linalg.norm(nearest-self.position)/5+5, 'finite_cover'
        mean = post.weights@post.points
        for q in (self.position, mean):
            if np.max(np.linalg.norm(np.abs(cover.c-q)+getattr(cover, 'half', 12.5), axis=1)) <= 19.999:
                return 'clear', q, np.linalg.norm(q-self.position)/5+5, 'certified_clear'
        g, radius, theta, directional = post.sample(self.rng, self.particles)
        active = np.ones((len(g), len(cover.c)), bool)
        best, best_cost = None, None
        candidates = [nearest]+([mean] if self.free_clear[i] < 2 else [])
        for q in candidates:
            success = np.linalg.norm(g-q, axis=1) <= 20
            remaining = active & cover.remaining_after_clear(q)
            remaining[success] = False
            cost = np.linalg.norm(q-self.position)/5+3+2*success+cover.costs(q, remaining, g)
            if best is None or cost.mean() < best_cost.mean():
                reason = 'finite_cover' if np.linalg.norm(q-nearest) < 1e-8 else 'posterior_clear'
                best, best_cost = ('clear', q, float(cost.mean()), reason), cost
        if self.local_rf[i] >= 5 or self.steps >= 1200:
            return best
        direction = mean-self.position
        direction /= max(np.linalg.norm(direction), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        points = [self.position+f*(mean-self.position)+b*normal
                  for f in (.35, .65, 1.) for b in (-60., 60.)]
        if self.future:
            points.append(self.future[0])
        first = next(o for o in hist if o.kind == 'direction')
        if self.local_rf[i] == 0:
            try:
                uv, _, _ = RayPrior(dict(first=[first.x, first.y], bearing=first.angle)).select(.25)
                t = np.deg2rad(first.angle)
                rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                points.extend(np.array([first.x, first.y])+rot@np.array([uv[0], s*uv[1]]) for s in (-1, 1))
            except ValueError:
                pass
        noise = self.rng.uniform(-1, 1, len(g))
        for q in points:
            if any(o.kind in ('direction', 'near', 'no_signal') and np.linalg.norm(q-[o.x, o.y]) < .1 for o in hist):
                continue
            masks, near = cover.measurement_masks(q, g, radius, theta, directional, noise)
            cost = np.linalg.norm(q-self.position)/5+5+(i+1 != self.channel)+5*near+cover.costs(q, masks, g)
            gain = best_cost-cost
            if gain.mean()-2*gain.std(ddof=1)/np.sqrt(len(g)) > 1.:
                best, best_cost = ('measure', q, float(cost.mean()), 'posterior_localize'), cost
        return best

    def replacement(self, q):
        """Remove future sites only with a continuous, orientation-safe proof."""
        if not len(self.unknown()) or not self.future:
            return []
        remove = []
        # Nearby sites give useful replacements; bound optimization cost.
        choices = np.argsort([np.linalg.norm(p-q) for p in self.future])[:3]
        for j in choices:
            keep = [p for k, p in enumerate(self.future) if k not in remove+[int(j)]]
            if self.certified(self.visited+[q]+keep).all():
                remove.append(int(j))
        return remove

    def emit(self, action, i, q, reason, estimated=0.):
        self.pending = dict(action=action, channel=int(i)+1,
                            position=np.round(q, 6).tolist(), reason=reason,
                            estimated_local_finish_s=float(estimated))
        return self.pending.copy()

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
                    cover, post = self.local_state(i)
                    anchors.append(post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0))
                exit_point = self.future[0] if self.future and len(self.unknown()) else self.position
                insertion = [np.linalg.norm(p-self.position)+np.linalg.norm(p-exit_point)-np.linalg.norm(exit_point-self.position)
                             for p in anchors]
                k = int(np.argmin(insertion))
                if self.mode == 'sequential' or not self.future or not len(self.unknown()) or insertion[k] < 700:
                    i = int(known[k])
                    action, q, value, reason = self.local_action(i)
                    q = np.round(q, 6)
                    if self.mode == 'joint' and action == 'measure':
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

    def accept_feedback(self, result):
        if not result.get('accepted') or self.pending is None:
            raise ValueError('Only accepted, matched feedback may update belief')
        a = self.pending
        i, q = a['channel']-1, np.array(a['position'])
        kind = result['kind']
        self.history[i].append(Observation(kind, float(q[0]), float(q[1]), result.get('angle')))
        if a['action'] == 'measure':
            if self.confirmed[i]:
                self.local_rf[i] += 1
            self.confirmed[i] |= kind in ('direction', 'near')
            self.channel = a['channel']
        else:
            self.free_clear[i] += a['reason'] == 'posterior_clear'
            if kind == 'success':
                self.confirmed[i] = self.cleared[i] = True
        self.position, self.pending = q, None
        self.steps += 1
