"""Mixed-source likelihood: integrate ONE fixed radius and orientation exactly.

The spatial quadrature and priors are modelling choices, not absence proofs.
Angles used internally are radians. RF readings remain rounded degrees.
"""
import numpy as np
from q3_belief import Observation, possible_cells

TAU = 2*np.pi
RF = ('direction', 'near', 'no_signal')


def unique_history(history):
    seen, out = set(), []
    for o in history:
        if o.kind in RF:
            key = (o.x, o.y)
            if key in seen:
                continue  # same location has fixed error and fixed reception
            seen.add(key)
        out.append(o)
    return tuple(out)


def geometric_factor(points, history):
    value = (np.linalg.norm(points, axis=1) <= 1800).astype(float)
    for o in history:
        delta = points-[o.x, o.y]
        d = np.linalg.norm(delta, axis=1)
        if o.kind == 'direction':
            phi = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0]))
            e = (o.angle-phi+180) % 360-180
            value *= (d > 5)*np.maximum(0., np.minimum(1., e+.005)-np.maximum(-1., e-.005))/2
        elif o.kind == 'near':
            value *= d <= 5
        elif o.kind in ('success', 'no_target_in_range'):
            value *= (d <= 20) if o.kind == 'success' else (d > 20)
    return value


def radius_orientation_pieces(g, history):
    """(omni R interval, directional angular cells and R intervals).

    Each angular cell is (theta_lo, theta_hi, R_lo, R_hi). Its interior has
    constant front/back signs for ALL past measurements. Zero-measure endpoint
    states are excluded here only for integration, never for certification.
    """
    obs = [o for o in history if o.kind in RF]
    if not obs:
        return (1000., 1500.), np.array([[0., TAU, 1000., 1500.]])
    delta = np.array([[o.x, o.y] for o in obs])-g
    dist = np.linalg.norm(delta, axis=1)
    positive = np.array([o.kind != 'no_signal' for o in obs])
    lower = max(1000., float(dist[positive].max(initial=0)))
    upper = min(1500., float(dist[~positive].min(initial=np.inf)))
    beta = np.arctan2(delta[:, 1], delta[:, 0])
    cuts = np.unique(np.r_[0., TAU, (beta-np.pi/2) % TAU, (beta+np.pi/2) % TAU])
    mid = (cuts[1:]+cuts[:-1])/2
    front = np.cos(mid[:, None]-beta[None, :]) >= 0
    # At exact coincidence, all directions count as received in this local
    # model. Neither mesh certificates nor probability integrals rely on it.
    front[:, dist < 1e-12] = True
    valid = np.all(front[:, positive], axis=1)
    hi = np.minimum(1500., np.min(np.where(front[:, ~positive], dist[None, ~positive], np.inf), axis=1, initial=np.inf))
    valid &= hi > lower
    pieces = np.column_stack((cuts[:-1], cuts[1:], np.full(len(mid), lower), hi))
    return (lower, upper), pieces[valid]


def radio_masses(points, history):
    omni, directional = np.zeros(len(points)), np.zeros(len(points))
    for j, g in enumerate(points):
        (lo, hi), p = radius_orientation_pieces(g, history)
        omni[j] = max(hi-lo, 0)/500
        directional[j] = np.sum((p[:, 1]-p[:, 0])*(p[:, 3]-p[:, 2]))/(500*TAU)
    return omni, directional


def likelihood(points, history, directional_prior=.5):
    history = unique_history(history)
    points = np.asarray(points, float).reshape(-1, 2)
    geom = geometric_factor(points, history)
    out = np.zeros(len(points))
    ids = np.flatnonzero(geom)
    omni, direct = radio_masses(points[ids], history)
    out[ids] = geom[ids]*((1-directional_prior)*omni+directional_prior*direct)
    return out


def conservative_cells(centers, half, history):
    # Positive wedges, 1500 m limit, near and optical feedback remain valid.
    # A directional miss supplies NO standalone radial upper bound.
    safe = [o for o in history if o.kind != 'no_signal']
    return possible_cells(centers, half, safe)


class KnownPosterior:
    def __init__(self, cells, history, directional_prior=.5):
        self.history = unique_history(history)
        self.directional_prior = directional_prior
        # Subclasses may refine the clearance cover (e.g. 12.5 m cells); the
        # quadrature offsets scale with the cell half-width.
        axis = np.array([-2/3, 0., 2/3])*getattr(self, 'half', 12.5)
        x, y = np.meshgrid(axis, axis)
        offsets = np.column_stack((x.ravel(), y.ravel()))
        p = (cells[:, None]+offsets).reshape(-1, 2)
        w = likelihood(p, self.history, directional_prior)
        self.numeric_ok = bool(w.sum() > 0)
        self.points = p[w > 0]
        self.weights = w[w > 0]/w.sum() if self.numeric_ok else np.array([])

    def sample(self, rng, count):
        if not self.numeric_ok:
            raise ArithmeticError('Zero quadrature mass is not empty support')
        ids = rng.choice(len(self.points), size=count, p=self.weights)
        g = self.points[ids]
        radius, theta, directional = np.empty(count), np.empty(count), np.zeros(count, bool)
        for j, point in enumerate(g):
            (lo, hi), pieces = radius_orientation_pieces(point, self.history)
            weights = (pieces[:, 1]-pieces[:, 0])*(pieces[:, 3]-pieces[:, 2])/(500*TAU)
            omass = (1-self.directional_prior)*max(hi-lo, 0)/500
            dmass = self.directional_prior*weights.sum()
            directional[j] = rng.random() < dmass/(omass+dmass)
            if directional[j]:
                row = pieces[rng.choice(len(pieces), p=weights/weights.sum())]
                theta[j] = rng.uniform(row[0], row[1])
                radius[j] = rng.uniform(row[2], row[3])
            else:
                theta[j], radius[j] = 0., rng.uniform(lo, hi)
        return g, radius, theta, directional
