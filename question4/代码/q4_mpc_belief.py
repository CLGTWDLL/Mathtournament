"""Experimental quadrature recovery; never changes the continuous cover.

Clip each spatial square to the observed bearing wedges, then integrate its
triangles with area weights. Refinement is bounded; unresolved numerical
failure still uses the finite-clear certificate, never an absence claim.

Two additions for the MPC experiment:

* the clipping/refinement works on any cell half-width, so the same recovery
  is available to the finer clearance cover;
* ``scenarios`` returns conservative uniform-over-cover scenarios when the
  quadrature is unresolved, so candidate ranking can continue instead of
  dropping straight to cell-by-cell finite clearing.
"""
import numpy as np

from q3_belief import EPS
from q4_belief import (TAU, KnownPosterior, RF, radio_masses,
                       radius_orientation_pieces, unique_history)


def _clip(polygon, normal, origin):
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = (a-origin)@normal, (b-origin)@normal
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a+(b-a)*(da/(da-db)))
    return np.asarray(out).reshape(-1, 2)


def _log_weights(points, history, area, prior):
    logs = np.log(np.maximum(area, np.finfo(float).tiny))
    logs[np.linalg.norm(points, axis=1) > 1800] = -np.inf
    for o in history:
        delta = points-[o.x, o.y]
        d = np.linalg.norm(delta, axis=1)
        if o.kind == 'direction':
            phi = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
            e = (o.angle-phi+180) % 360-180
            factor = (d > 5)*np.maximum(0., np.minimum(1., e+.005)-np.maximum(-1., e-.005))/2
            with np.errstate(divide='ignore'):
                logs += np.log(factor)
        elif o.kind in ('near', 'success', 'no_target_in_range'):
            valid = d <= (5 if o.kind == 'near' else 20)
            if o.kind == 'no_target_in_range':
                valid = ~valid
            logs[~valid] = -np.inf
    ids = np.flatnonzero(np.isfinite(logs))
    omni, direct = radio_masses(points[ids], history)
    with np.errstate(divide='ignore'):
        logs[ids] += np.log((1-prior)*omni+prior*direct)
    good = np.isfinite(logs)
    weights = np.exp(logs[good]-logs[good].max()) if good.any() else np.array([])
    return points[good], weights/weights.sum() if len(weights) else weights


class AdaptivePosterior(KnownPosterior):
    def __init__(self, cells, history, directional_prior=.5, half=12.5):
        self.cells = np.asarray(cells, float).reshape(-1, 2)
        self.half = float(half)
        super().__init__(cells, history, directional_prior)
        self.recovered = False
        self.quadrature_level = 0
        self.initial_numeric_ok = self.numeric_ok
        ess = 1/np.sum(self.weights**2) if self.numeric_ok else 0
        if self.numeric_ok and len(self.points) >= 12 and ess >= 6:
            return
        triangles = []
        h = self.half
        corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        for c in self.cells:
            polygon = c+corners
            for o in self.history:
                if o.kind != 'direction':
                    continue
                for t, sign in ((np.deg2rad(o.angle)-EPS, 1), (np.deg2rad(o.angle)+EPS, -1)):
                    polygon = _clip(polygon, sign*np.array([-np.sin(t), np.cos(t)]), np.array([o.x, o.y]))
                    if not len(polygon):
                        break
                if not len(polygon):
                    break
            for j in range(1, len(polygon)-1):
                triangles.append([polygon[0], polygon[j], polygon[j+1]])
        if not triangles:
            return
        tri = np.asarray(triangles)
        barycentric = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
        for level in range(1, 6):
            u, v = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
            areas = np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0])/2
            tri, areas = tri[areas > 1e-18], areas[areas > 1e-18]
            if not len(tri):
                break
            points = np.einsum('ij,kjl->kil', barycentric, tri).reshape(-1, 2)
            p, w = _log_weights(points, self.history, np.repeat(areas/3, 3), directional_prior)
            if len(w):
                self.points, self.weights, self.numeric_ok = p, w, True
                self.recovered, self.quadrature_level = True, level
                if len(w) >= 12 and 1/np.sum(w*w) >= 6:
                    break
            if len(tri)*12 > 100_000:
                break
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
            tri = np.concatenate([np.stack(t, axis=1) for t in
                                  ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])

    def _sample_radio(self, rng, points):
        """Draw (radius, theta, directional) for points from public history."""
        count = len(points)
        radius = np.empty(count)
        theta = np.empty(count)
        directional = np.zeros(count, bool)
        for j, point in enumerate(points):
            (lo, hi), pieces = radius_orientation_pieces(point, self.history)
            weights = ((pieces[:, 1]-pieces[:, 0])*(pieces[:, 3]-pieces[:, 2])/(500*TAU)
                       if len(pieces) else np.zeros(0))
            omass = (1-self.directional_prior)*max(hi-lo, 0)/500
            dmass = self.directional_prior*weights.sum()
            total = omass+dmass
            directional[j] = bool(total > 0 and rng.random() < dmass/total)
            if directional[j] and weights.sum() > 0:
                row = pieces[rng.choice(len(pieces), p=weights/weights.sum())]
                theta[j], radius[j] = rng.uniform(row[0], row[1]), rng.uniform(row[2], row[3])
            else:
                directional[j] = False
                theta[j], radius[j] = 0., rng.uniform(min(lo, hi), max(lo, hi))
        return radius, theta, directional

    def fallback_scenarios(self, rng, count):
        """Uniform-over-cover scenarios used only to RANK candidates.

        The continuous cover is a strict outer bound, so these scenarios are
        conservative: they may overstate how spread the source is, but they can
        never claim a source position that the public feedback already excludes.
        """
        if not len(self.cells):
            raise ArithmeticError('No conservative cells to sample from')
        points = self.cells[rng.choice(len(self.cells), size=count)]
        radius, theta, directional = self._sample_radio(rng, points)
        return points, radius, theta, directional

    def scenarios(self, rng, count):
        """Numeric scenarios when available, conservative fallback otherwise."""
        if self.numeric_ok:
            return self.sample(rng, count)
        return self.fallback_scenarios(rng, count)
