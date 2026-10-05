"""Public-feedback-only multi-channel posterior, with analytic R integration.

Numerical probabilities use adaptive square quadrature. Conservative cell
rejection is separate: zero numerical mass is never an absence certificate.
Assumptions: area-uniform sources; independent U[1000,1500] radii; uniform
rounded bearing errors at distinct positions; uniform N in 10..16 and subsets.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

AREA = math.pi * 1800**2
EPS = np.deg2rad(1.005)


@dataclass(frozen=True)
class Observation:
    kind: str
    x: float
    y: float
    angle: float | None = None


def range_bounds(points, history):
    lower = np.full(len(points), 1000.)
    upper = np.full(len(points), 1500.)
    for o in history:
        d = np.linalg.norm(points-[o.x, o.y], axis=1)
        if o.kind == 'direction':
            lower = np.maximum(lower, d)
        elif o.kind == 'no_signal':
            upper = np.minimum(upper, d)
    return lower, upper


def likelihood(points, history):
    """Actual rounded-reading probability, including one shared radius."""
    lower, upper = range_bounds(points, history)
    value = (np.linalg.norm(points, axis=1) <= 1800).astype(float)
    for o in history:
        delta = points-[o.x, o.y]
        d = np.linalg.norm(delta, axis=1)
        if o.kind == 'direction':
            phi = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0]))
            e = (o.angle-phi+180) % 360-180
            angular = np.maximum(0., np.minimum(1., e+.005)-np.maximum(-1., e-.005))/2
            value *= (d > 5)*angular
        elif o.kind == 'near':
            value *= (d <= 5)
        elif o.kind == 'success':
            value *= (d <= 20)
        elif o.kind == 'no_target_in_range':
            value *= (d > 20)
    return value*np.maximum(upper-lower, 0)/500


def possible_cells(centers, half, history):
    """Necessary conditions for a square to intersect continuous support.

    Overestimates support; excludes a square only if an entire-square bound
    contradicts the observations. In particular, never tests centers alone.
    """
    lo = np.linalg.norm(np.maximum(np.abs(centers)-half, 0), axis=1)
    ok = lo <= 1800
    rlo, rhi = np.full(len(centers), 1000.), np.full(len(centers), 1500.)
    for o in history:
        delta = centers-[o.x, o.y]
        dmin = np.linalg.norm(np.maximum(np.abs(delta)-half, 0), axis=1)
        dmax = np.linalg.norm(np.abs(delta)+half, axis=1)
        if o.kind == 'direction':
            rlo = np.maximum(rlo, dmin)
            ok &= dmax > 5
            theta = np.deg2rad(o.angle)
            for angle, sign in [(theta-EPS, 1), (theta+EPS, -1)]:
                normal = sign*np.array([-np.sin(angle), np.cos(angle)])
                ok &= delta@normal+half*np.abs(normal).sum() >= -1e-9
        elif o.kind == 'no_signal':
            rhi = np.minimum(rhi, dmax)
        elif o.kind in ('near', 'success'):
            ok &= dmin <= (5 if o.kind == 'near' else 20)
        elif o.kind == 'no_target_in_range':
            ok &= dmax > 20
    return ok & (rlo <= rhi+1e-9)


def boundary_cells(centers, half, history):
    def circle(q, radius):
        delta = np.abs(centers-q)
        lo = np.linalg.norm(np.maximum(delta-half, 0), axis=1)
        hi = np.linalg.norm(delta+half, axis=1)
        return (lo <= radius) & (hi >= radius)
    boundary = circle(np.zeros(2), 1800)
    for o in history:
        q = np.array([o.x, o.y])
        if o.kind in ('direction', 'no_signal'):
            boundary |= circle(q,1000) | circle(q,1500)
        if o.kind == 'direction':
            theta = np.deg2rad(o.angle)
            for a in (theta-EPS, theta+EPS):
                normal = np.array([-np.sin(a),np.cos(a)])
                boundary |= np.abs((centers-q)@normal) <= half*np.abs(normal).sum()
            boundary |= circle(q,5)
        elif o.kind in ('near','success','no_target_in_range'):
            boundary |= circle(q,5 if o.kind == 'near' else 20)
    return boundary


class SpatialPosterior:
    def __init__(self, history=(), base=100., fine=3.125, order=4):
        self.history = tuple(history)
        self.base, self.fine, self.order = base, fine, order
        if abs(round(3600/base)*base-3600) > 1e-8:
            raise ValueError('base must divide 3600')
        axis = np.arange(-1800+base/2, 1800, base)
        xx, yy = np.meshgrid(axis, axis)
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        half = base/2
        samples, volumes, unresolved = [], [], []
        leaves, sizes, masses = [], [], []
        localizing = any(o.kind in ('direction', 'near', 'success') for o in history)
        target = min(base, 12.5 if localizing else base)
        offset_axis = (np.arange(order)+.5)/order*2-1
        ox, oy = np.meshgrid(offset_axis, offset_axis)
        offsets = np.column_stack((ox.ravel(), oy.ravel()))
        children = np.array([[-1,-1], [-1,1], [1,-1], [1,1]])
        while len(centers):
            centers = centers[possible_cells(centers, half, history)]
            if not len(centers):
                break
            side = 2*half
            if side > target+1e-9:
                centers = (centers[:, None, :]+half/2*children).reshape(-1, 2)
                half /= 2
                continue
            points = (centers[:, None, :]+half*offsets).reshape(-1, 2)
            values = likelihood(points, history).reshape(len(centers), -1)
            # Refine cut boundaries and possible squares missed by quadrature.
            zero = values.max(axis=1) == 0
            cut = (values.min(axis=1) == 0) & ~zero
            cut |= boundary_cells(centers, half, history)
            # A bearing wedge a few metres from its apex can be much narrower
            # than a fixed spatial tolerance. Refine relative to angular scale.
            tolerance = np.full(len(centers),fine)
            for o in history:
                if o.kind == 'direction':
                    dmin = np.linalg.norm(np.maximum(np.abs(centers-[o.x,o.y])-half,0),axis=1)
                    tolerance = np.minimum(tolerance,np.maximum(fine/32,dmin*.002))
            split = (cut & (side > tolerance+1e-9)) | (zero & (side > tolerance/16+1e-9))
            keep = ~split
            if keep.any():
                p = points.reshape(len(centers), -1, 2)[keep]
                v = values[keep]*side**2/(order**2*AREA)
                samples.append(p.reshape(-1, 2))
                volumes.append(v.ravel())
                leaves.append(centers[keep])
                sizes.append(np.full(keep.sum(), half))
                masses.append(v.sum(axis=1))
                unresolved.append(zero[keep])
            centers = (centers[split, None, :]+half/2*children).reshape(-1, 2)
            half /= 2
        self.centers = np.concatenate(leaves) if leaves else np.empty((0, 2))
        self.half = np.concatenate(sizes) if sizes else np.empty(0)
        self.mass_raw = np.concatenate(masses) if masses else np.empty(0)
        self.unresolved_cells = int(np.concatenate(unresolved).sum()) if unresolved else 0
        self.certified_empty = len(self.centers) == 0
        pts = np.concatenate(samples) if samples else np.empty((0, 2))
        wt = np.concatenate(volumes) if volumes else np.empty(0)
        positive = wt > 0
        self.points, wt = pts[positive], wt[positive]
        self.evidence = float(wt.sum()) if history else 1.
        if self.evidence > 1+1e-3:
            raise ArithmeticError('Quadrature evidence exceeds one')
        self.evidence = min(self.evidence, 1.)
        if history and not self.certified_empty and not len(wt):
            raise ArithmeticError('Possible support has no quadrature mass; refine, do not declare absent')
        self.weights = wt/wt.sum() if len(wt) else wt
        self.mass = self.mass_raw/self.mass_raw.sum() if self.mass_raw.sum() else self.mass_raw
        self.lower, self.upper = range_bounds(self.points, history)

    def predict(self, q):
        if not len(self.points):
            return dict(receive=0., near=0., clear=0.)
        d = np.linalg.norm(self.points-np.asarray(q), axis=1)
        length = self.upper-self.lower
        receive = np.clip((self.upper-np.maximum(self.lower, d))/length, 0, 1)
        return dict(receive=float(self.weights@receive),
                    near=float(self.weights@(d <= 5)),
                    clear=float(self.weights@(d <= 20)))

    def hpd_contains(self, g, level=.9):
        """Cell-average-density HPD set, with whole crossing cell included."""
        density = self.mass/(2*self.half)**2
        order = np.argsort(-density)
        chosen = order[:np.searchsorted(np.cumsum(self.mass[order]), level)+1]
        return bool(np.any(np.max(np.abs(self.centers[chosen]-g), axis=1) <= self.half[chosen]+1e-8))


@lru_cache(maxsize=48)
def spatial_cached(history, base=100., fine=3.125):
    return SpatialPosterior(history, base, fine)


def occupancy_posterior(z0, z1):
    """Exact degree-20 polynomial marginalization of the count constraint."""
    z0, z1 = np.asarray(z0, float), np.asarray(z1, float)
    scale = np.maximum(z0, z1)
    if np.any(scale <= 0):
        raise ValueError('Impossible local evidence')
    a, b = z0/scale, z1/scale
    prior = np.array([1/(7*math.comb(20, n)) if 10 <= n <= 16 else 0.
                      for n in range(21)])
    def polynomial(skip=None):
        coef = np.array([1.])
        for i in range(20):
            if i != skip:
                coef = np.convolve(coef, [a[i], b[i]])
        return coef
    count = polynomial()*prior
    total = count.sum()
    if total <= 0:
        raise ValueError('Evidence contradicts N in 10..16')
    marginal = np.array([b[i]*np.dot(polynomial(i), prior[1:])/total for i in range(20)])
    return marginal, count/total


class BeliefBank:
    """Takes only public channel, position and feedback, never source truth."""
    def __init__(self, base=100., fine=3.125):
        self.base, self.fine = base, fine
        self.history = [[] for _ in range(20)]
        self.seen = [{} for _ in range(20)]
        self.cleared = np.zeros(20, bool)
        initial = spatial_cached((), base, fine)
        self.spatial = [initial]*20
        self.z0, self.z1 = np.ones(20), np.ones(20)
        self.exists, self.count = occupancy_posterior(self.z0, self.z1)

    def predict(self, channel, q):
        i = channel-1
        if self.cleared[i]:
            return dict(receive=0., near=0., clear=0.)
        return {k: v*self.exists[i] for k, v in self.spatial[i].predict(q).items()}

    def update(self, channel, q, kind, angle=None):
        if channel not in range(1, 21) or kind not in (
                'direction', 'near', 'no_signal', 'success', 'no_target_in_range'):
            raise ValueError('Invalid public feedback')
        q = np.asarray(q, float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError('Invalid position')
        if kind == 'direction' and (angle is None or not np.isfinite(angle)):
            raise ValueError('Direction requires a finite angle')
        i = channel-1
        if self.cleared[i]:
            if kind in ('no_signal', 'no_target_in_range'):
                return False
            raise ValueError('Positive feedback after clearance')
        measurement = kind in ('direction', 'near', 'no_signal')
        key = tuple(float(x) for x in q)
        obs = Observation(kind, *key, None if angle is None else float(angle) % 360)
        if measurement and key in self.seen[i]:
            if self.seen[i][key] != obs:
                raise ValueError('Repeated position gave conflicting deterministic feedback')
            return False
        history = tuple(self.history[i]+[obs])
        posterior = spatial_cached(history, self.base, self.fine)
        z0, z1 = self.z0.copy(), self.z1.copy()
        if kind in ('direction', 'near', 'success'):
            z0[i] = 0.
        z1[i] = posterior.evidence
        exists, count = occupancy_posterior(z0, z1)
        self.spatial[i], self.history[i] = posterior, list(history)
        self.z0, self.z1, self.exists, self.count = z0, z1, exists, count
        if measurement:
            self.seen[i][key] = obs
        if kind == 'success':
            self.cleared[i] = True
        return True
