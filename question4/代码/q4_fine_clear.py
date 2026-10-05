"""Q4 research variant: sparse backbone + 12.5 m conservative clearance cover.

The frozen policy keeps a 25 m whole-cell cover for each confirmed source, so a
"certified clear" (whole cover inside a 19.999 m circle) is practically never
available and the finite fallback clears 25 m cells one by one. Here the
surviving 25 m cells are split into 12.5 m cells and re-tested with the same
mixed-source predicate (possible_mixed_cells), which is still a necessary
condition on the continuous square. Effects:

* certified clears become possible once two bearings cross closely;
* one 20 m attempt at a 12.5 m cell centre discharges the cell and its four
  edge neighbours (far corner <= 19.8 m), so the finite fallback needs fewer
  attempts and the scenario evaluation sees the same finer geometry.

Everything else (dispatch, joint sites, 25 m directional certificate) is
inherited from q4_sparse_backbone.Policy. Local research only.
"""
import numpy as np
from q3_belief import Observation, EPS
from q3_policy import Cover as OmniCover, GRID
from q2_analytic import RayPrior
from q4_belief import KnownPosterior, likelihood, unique_history
from q4_feedback_geometry import mixed_cells
from q4_sparse_backbone import Policy as SparsePolicy

CHILDREN = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# Cap the refined cover: the pairwise distance matrix and the quadrature grow
# quadratically with the cell count, and a very wide untouched support gains
# nothing from subdividing before the next bearing arrives.
MAX_FINE_CELLS = 1500


def refined_cells(history, half=6.25):
    """Whole-cell mixed-source support, refined from 25 m down to 2*half m."""
    centers, h = GRID[mixed_cells(GRID, 12.5, history)], 12.5
    while h > half+1e-9 and len(centers):
        if 4*len(centers) > MAX_FINE_CELLS:
            break
        centers = (centers[:, None, :]+h/2*CHILDREN).reshape(-1, 2)
        h /= 2
        centers = centers[mixed_cells(centers, h, history)]
    return centers, h


class FineCover(OmniCover):
    def __init__(self, history, half=6.25):
        self.history = tuple(history)
        self.c, self.half = refined_cells(history, half)
        if not len(self.c):
            raise ArithmeticError('Confirmed source lost continuous mixed support')
        self.distance = np.linalg.norm(self.c[:, None]-self.c[None], axis=2)

    def attach_masses(self, post):
        """W[k, j] = posterior mass of cell j's points within 20 m of centre k."""
        near = np.linalg.norm(self.c[:, None]-post.points[None], axis=2) <= 20
        onehot = np.zeros((len(post.points), len(self.c)))
        onehot[np.arange(len(post.points)), post.cell] = post.weights
        self.W = near.astype(float)@onehot

    def best_cell(self, distances, active):
        """Next cell per scenario: posterior-mass greedy only when attached.

        Without masses this is the same nearest-cell order the executed policy
        follows, so prediction and execution stay identical.
        """
        if getattr(self, 'W', None) is None:
            return np.argmin(np.where(active, distances, np.inf), axis=1)
        mass = active.astype(float)@self.W.T
        step = 3+distances/5
        score = np.where(active, mass/step, -np.inf)
        empty = ~(score > 0).any(axis=1)
        if empty.any():
            score[empty] = np.where(active[empty], -step[empty], -np.inf)
        return np.argmax(score, axis=1)

    def greedy_first(self, start, active=None):
        """Cell maximising reachable posterior mass per second from `start`."""
        active = np.ones(len(self.c), bool) if active is None else active
        mass = active@self.W.T
        cost = 3+np.linalg.norm(self.c-start, axis=1)/5
        score = np.where(active, mass/cost, -np.inf)
        if not np.isfinite(score).any() or score.max() <= 0:
            score = np.where(active, -cost, -np.inf)
        return int(np.argmax(score))

    def costs(self, start, active, g):
        """Finite clearing simulated with the same mass-greedy rule the policy uses.

        The attempt sequence of a scenario depends only on its surviving cells
        and the shared posterior masses, never on the scenario's true position;
        truth only decides when the sequence stops.
        """
        active = active.copy(); n = len(g)
        cost = np.zeros(n); alive = active.any(axis=1)
        distances = np.repeat(np.linalg.norm(self.c-start, axis=1)[None], n, axis=0)
        keep_cache = {}
        for _ in range(len(self.c)):
            ids = np.flatnonzero(alive)
            if not len(ids):
                break
            if not active[ids].any(axis=1).all():
                raise ArithmeticError('Scenario escaped the continuous clearance cover')
            choices = self.best_cell(distances[ids], active[ids])
            cost[ids] += distances[ids, choices]/5+3
            success = np.linalg.norm(g[ids]-self.c[choices], axis=1) <= 20
            cost[ids[success]] += 2
            for choice in np.unique(choices):
                choice = int(choice)
                if choice not in keep_cache:
                    keep_cache[choice] = self.remaining_after_clear(self.c[choice])
                rows = ids[(choices == choice) & ~success]
                if len(rows):
                    active[rows] &= keep_cache[choice]
            distances[ids] = self.distance[choices]
            alive[ids[success]] = False
        if alive.any():
            raise ArithmeticError('Clearance cover exhausted')
        return cost

    def certified_clear_point(self, q):
        return bool(np.max(np.linalg.norm(np.abs(self.c-q)+self.half, axis=1)) <= 19.999)

    def remaining_after_clear(self, q):
        return np.linalg.norm(np.abs(self.c-q)+self.half, axis=1) > 20

    def measurement_masks(self, q, g, radius, theta, directional, noise):
        h = self.half
        relg = g-q
        d = np.linalg.norm(relg, axis=1)
        front = np.sum((q-g)*np.column_stack((np.cos(theta), np.sin(theta))), axis=1) >= 0
        receive = (d <= radius) & (~directional | front)
        near, direction = receive & (d <= 5), receive & (d > 5)
        mask = np.ones((len(g), len(self.c)), bool)
        rel = self.c-q
        dmin = np.linalg.norm(np.maximum(np.abs(rel)-h, 0), axis=1)
        dmax = np.linalg.norm(np.abs(rel)+h, axis=1)
        ids = np.flatnonzero(direction)
        if len(ids):
            angle = np.deg2rad(np.round(np.rad2deg(np.arctan2(relg[ids, 1], relg[ids, 0]))+noise[ids], 2))
            for t, sign in ((angle-EPS, 1), (angle+EPS, -1)):
                normal = sign*np.column_stack((-np.sin(t), np.cos(t)))
                mask[ids] &= normal@rel.T+h*np.abs(normal).sum(axis=1)[:, None] >= -1e-9
            mask[ids] &= (dmax > 5) & (dmin <= 1500)
        negative = ~receive
        if negative.any():
            obs = Observation('no_signal', float(q[0]), float(q[1]))
            mask[negative] &= mixed_cells(self.c, h, self.history+(obs,))
        mask[near] = False
        return mask, near


class FinePosterior(KnownPosterior):
    def __init__(self, cells, half, history, directional_prior=.5):
        self.history = unique_history(history)
        self.directional_prior = directional_prior
        axis = np.array([-2/3, 0., 2/3])*half
        x, y = np.meshgrid(axis, axis)
        offsets = np.column_stack((x.ravel(), y.ravel()))
        p = (cells[:, None]+offsets).reshape(-1, 2)
        cell = np.repeat(np.arange(len(cells)), len(offsets))
        w = likelihood(p, self.history, directional_prior)
        self.numeric_ok = bool(w.sum() > 0)
        self.points, self.cell = p[w > 0], cell[w > 0]
        self.weights = w[w > 0]/w.sum() if self.numeric_ok else np.array([])


class Policy(SparsePolicy):
    def __init__(self, variant='dispatch', particles=32, seed=9411, half=6.25, greedy=True, receiving=False, **kwargs):
        super().__init__(variant, particles, seed, **kwargs)
        self.clear_half = half
        self.greedy = greedy        # mass-greedy finite fallback instead of nearest-cell order
        self.receiving = receiving  # False: frozen dispatch candidate set; True: add receiving-side candidates

    def local_state(self, i):
        key = (i, len(self.history[i]))
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != i}
            cover = FineCover(self.history[i], self.clear_half)
            post = FinePosterior(cover.c, cover.half, self.history[i])
            if post.numeric_ok and self.greedy:
                cover.attach_masses(post)
            else:
                cover.costs = lambda start, active, g, _c=cover: OmniCover.costs(_c, start, active, g)
            self.cache[key] = cover, post
            self.numeric_fallbacks += int(not post.numeric_ok)
        return self.cache[key]

    def local_action(self, i):
        # Frozen dispatch candidate generator (q4_policy.Policy.local_action)
        # on the fine cover; receiving-side candidates only if self.receiving.
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
            if cover.certified_clear_point(q):
                return 'clear', q, np.linalg.norm(q-self.position)/5+5, 'certified_clear'
        if self.greedy:
            nearest = cover.c[cover.greedy_first(self.position)]
        g, radius, theta, directional = post.sample(self.rng, self.particles)
        active = np.ones((len(g), len(cover.c)), bool)
        best, best_cost = None, None
        for q in [nearest]+([mean] if self.free_clear[i] < 2 else []):
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
        points = [(self.position+f*(mean-self.position)+b*normal, 'posterior_localize')
                  for f in (.35, .65, 1.) for b in (-60., 60.)]
        if self.future:
            points.append((self.future[0], 'posterior_localize'))
        positive = [o for o in hist if o.kind in ('direction', 'near')]
        first = next(o for o in hist if o.kind == 'direction')
        if self.local_rf[i] == 0:
            try:
                uv, _, _ = RayPrior(dict(first=[first.x, first.y], bearing=first.angle)).select(.25)
                t = np.deg2rad(first.angle)
                rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                points.extend((np.array([first.x, first.y])+rot@np.array([uv[0], s*uv[1]]), 'posterior_localize') for s in (-1, 1))
            except ValueError:
                pass
        for o in (positive[-2:] if self.receiving else []):
            p = np.array([o.x, o.y])
            dd = mean-p
            n = np.array([-dd[1], dd[0]])/max(np.linalg.norm(dd), 1e-9)
            points.extend((p+f*dd+b*n, 'receiving_side_approach') for f in (.35, .65) for b in (-30., 0., 30.))
            # Wide lateral offsets: a bearing that actually crosses the first wedge.
            points.extend((p+f*dd+b*n, 'receiving_side_wide') for f in (.4, .7, 1.) for b in (-300., -150., 150., 300.))
        if self.receiving and len(positive) >= 2:
            a = np.array([positive[-2].x, positive[-2].y])
            b = np.array([positive[-1].x, positive[-1].y])
            points.extend(((1-f)*a+f*b, 'receiving_chord') for f in (.25, .5, .75))
        noise = self.rng.uniform(-1, 1, len(g))
        seen = set()
        for q, reason in points:
            q = np.round(q, 6)
            key = tuple(q)
            if key in seen or any(o.kind in ('direction', 'near', 'no_signal') and np.linalg.norm(q-[o.x, o.y]) < .1 for o in hist):
                continue
            seen.add(key)
            masks, near = cover.measurement_masks(q, g, radius, theta, directional, noise)
            cost = np.linalg.norm(q-self.position)/5+5+(i+1 != self.channel)+5*near+cover.costs(q, masks, g)
            gain = best_cost-cost
            if gain.mean()-2*gain.std(ddof=1)/np.sqrt(len(g)) > 1.:
                best, best_cost = ('measure', q, float(cost.mean()), reason), cost
        return best
