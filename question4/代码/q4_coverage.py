"""Continuous directional-discovery certificates, independent of probability."""
import numpy as np


def domain_cells(side=100.):
    axis = np.arange(-1800+side/2, 1800, side)
    x, y = np.meshgrid(axis, axis)
    cells = np.column_stack((x.ravel(), y.ravel()))
    return cells[np.linalg.norm(np.maximum(np.abs(cells)-side/2, 0), axis=1) <= 1800]


def certified_cells(stations, cells=None, half=50., radius=999.9):
    """Every corner strictly inside hull of stations covering the ENTIRE cell.

    Convexity then covers the continuous square, not just four sample points.
    The max-angle-gap test is vectorized. Strict inequalities deliberately
    reject marginal configurations that would be valid only on boundaries.
    """
    cells = domain_cells(2*half) if cells is None else np.asarray(cells)
    stations = np.asarray(stations, float).reshape(-1, 2)
    if len(stations) < 3:
        return np.zeros(len(cells), bool)
    eligible = np.linalg.norm(np.abs(cells[:, None]-stations[None])+half, axis=2) <= radius
    good = eligible.sum(axis=1) >= 3
    for offset in ((-half, -half), (-half, half), (half, -half), (half, half)):
        delta = stations[None]-(cells+offset)[:, None]
        corner_eligible = eligible & (np.linalg.norm(delta, axis=2) > 1e-8)
        counts = corner_eligible.sum(axis=1)
        ang = np.arctan2(delta[:, :, 1], delta[:, :, 0]) % (2*np.pi)
        ordered = np.sort(np.where(corner_eligible, ang, np.inf), axis=1)
        with np.errstate(invalid='ignore'):
            gaps = np.diff(ordered, axis=1)
        gaps = np.where(np.arange(len(stations)-1)[None] < counts[:, None]-1, gaps, 0.)
        last = ordered[np.arange(len(cells)), np.maximum(counts-1, 0)]
        with np.errstate(invalid='ignore'):
            wrap = ordered[:, 0]+2*np.pi-last
        good &= (counts >= 3) & (np.maximum(gaps.max(axis=1), wrap) < np.pi-1e-10)
    return good


def initial_stations():
    # Triangular mesh; exterior vertices handle outward-facing boundary
    # sources. Its validity is checked on continuous squares, not assumed.
    spacing = 875.
    stations = np.array([[spacing*(i+j/2)+17, spacing*np.sqrt(3)*j/2+23]
                         for j in range(-5, 6) for i in range(-7, 8)])
    stations = stations[np.linalg.norm(stations, axis=1) < 2800]
    stations = np.vstack((np.zeros(2), stations))
    if not certified_cells(stations).all():
        raise AssertionError('Initial continuous directional cover failed')
    # Deterministic backward deletion maintains the complete certificate.
    for i in range(len(stations)-1, 0, -1):
        trial = np.delete(stations, i, axis=0)
        if certified_cells(trial).all():
            stations = trial
    return open_route(stations[1:], np.zeros(2))


def open_route(stations, start):
    left = list(np.asarray(stations, float).reshape(-1, 2))
    ordered, p = [], np.asarray(start)
    while left:
        j = int(np.argmin([np.linalg.norm(q-p) for q in left]))
        p = left.pop(j)
        ordered.append(p)
    route = np.array(ordered).reshape(-1, 2)
    # Open-path 2-opt; endpoint may change and no return-to-origin charge.
    improved = True
    while improved:
        improved = False
        for i in range(len(route)):
            a = start if i == 0 else route[i-1]
            for j in range(i+1, len(route)):
                before = np.linalg.norm(a-route[i])
                after = np.linalg.norm(a-route[j])
                if j+1 < len(route):
                    before += np.linalg.norm(route[j]-route[j+1])
                    after += np.linalg.norm(route[i]-route[j+1])
                if after < before-1e-8:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
    return route
