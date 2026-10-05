"""Q4 second-stage ablations. Frozen v1 modules are left unchanged."""
import numpy as np
from q3_belief import Observation
from q3_policy import GRID
from q2_analytic import RayPrior
from q4_belief import KnownPosterior
from q4_policy import Policy as BasePolicy, Cover as BaseCover
from q4_feedback_geometry import mixed_cells, possible_mixed_cells


class Cover(BaseCover):
    def __init__(self, history):
        self.history = tuple(history)
        self.c = GRID[mixed_cells(GRID, 12.5, history)]
        if not len(self.c):
            raise ArithmeticError('Confirmed source lost continuous mixed support')
        self.distance = np.linalg.norm(self.c[:, None]-self.c[None], axis=2)

    def measurement_masks(self, q, g, radius, theta, directional, noise):
        mask, near = super().measurement_masks(q, g, radius, theta, directional, noise)
        delta, d = q-g, np.linalg.norm(q-g, axis=1)
        front = np.sum(delta*np.column_stack((np.cos(theta), np.sin(theta))), axis=1) >= 0
        negative = (d > radius) | (directional & ~front)
        if negative.any():
            obs = Observation('no_signal', float(q[0]), float(q[1]))
            mask[negative] &= mixed_cells(self.c, 12.5, self.history+(obs,))
        return mask, near


class Policy(BasePolicy):
    def __init__(self, variant='feedback', particles=32, seed=9411):
        if variant not in ('geometry', 'feedback'):
            raise ValueError(variant)
        super().__init__('joint', particles, seed)
        self.variant = variant

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
        # Geometry-only ablation uses the identical frozen candidate generator.
        if self.variant == 'geometry':
            return super().local_action(i)
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
        # Positive-side approach is a candidate, not a false reception promise.
        # Only chords between two previously receiving sites are guaranteed.
        for o in positive[-2:]:
            p = np.array([o.x, o.y])
            d = mean-p
            n = np.array([-d[1], d[0]])/max(np.linalg.norm(d), 1e-9)
            points.extend((p+f*d+b*n, 'receiving_side_approach')
                          for f in (.35, .65) for b in (-30., 0., 30.))
        if len(positive) >= 2:
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
