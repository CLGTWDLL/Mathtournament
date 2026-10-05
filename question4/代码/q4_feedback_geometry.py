"""Conservative mixed-source cell pruning using joint RF feedback.

Closed angular arcs are intersected analytically: isolated feasible angles and
radius endpoints are retained. Probability mass is never used to delete cells.
"""
from functools import lru_cache

import numpy as np
from q4_belief import conservative_cells, RF, unique_history

TAU = 2*np.pi


def intersect_arc(intervals, center, halfwidth):
    if halfwidth >= np.pi:
        return intervals
    # Outward rounding margin preserves boundary states near floating limits.
    halfwidth += 2e-12
    lo, hi = center-halfwidth, center+halfwidth
    pieces = []
    for k in range(-2, 3):
        a, b = max(0., lo+k*TAU), min(TAU, hi+k*TAU)
        if a <= b:
            pieces.append((a, b))
    result = []
    for a, b in intervals:
        for c, d in pieces:
            low, high = max(a, c), min(b, d)
            if low <= high:
                result.append((low, high))
    return result


def possible_mixed_cells(centers, half, history):
    """Necessary (not sufficient) conditions on a continuous square.

    L_C = max(1000, positive d_min). A negative whose d_max < L_C
    MUST be on the back side for any directional state in this square; it
    rules out omni states. Positive/front and forced-negative/back directions
    are broadened by asin(sqrt(2)*half / distance-to-center) before intersection.
    Spatial, angular and radius correlations are relaxed only outwards.
    """
    centers = np.asarray(centers, float).reshape(-1, 2)
    hist = unique_history(history)
    keep = conservative_cells(centers, half, hist)
    obs = [o for o in hist if o.kind in RF]
    if not obs or not keep.any():
        return keep
    stations = np.array([[o.x, o.y] for o in obs])
    positive = np.array([o.kind != 'no_signal' for o in obs])
    for i in np.flatnonzero(keep):
        delta = stations-centers[i]
        dmin = np.linalg.norm(np.maximum(np.abs(delta)-half, 0), axis=1)
        dmax = np.linalg.norm(np.abs(delta)+half, axis=1)
        lower = max(1000., dmin[positive].max(initial=0.))
        forced_back = ~positive & (dmax < lower-1e-8)
        # With no forced negative, an omnidirectional state survives the
        # relaxed cell bounds, regardless of directional feasibility.
        if not forced_back.any():
            continue
        intervals = [(0., TAU)]
        for j in np.flatnonzero(positive | forced_back):
            dist = np.linalg.norm(delta[j])
            if dist <= np.sqrt(2)*half:
                continue
            beta = np.arctan2(delta[j, 1], delta[j, 0]) % TAU
            center = (beta+(np.pi if forced_back[j] else 0.)) % TAU
            width = np.pi/2+np.arcsin(min(1., np.sqrt(2)*half/dist))
            intervals = intersect_arc(intervals, center, width)
            if not intervals:
                keep[i] = False
                break
    return keep


@lru_cache(maxsize=512)
def _cached_mixed_cells(centers_bytes, shape, half, history):
    centers = np.frombuffer(centers_bytes, float).reshape(shape)
    result = possible_mixed_cells(centers, half, history)
    result.setflags(write=False)
    return result


def mixed_cells(centers, half, history):
    """Same predicate as :func:`possible_mixed_cells`, memoised by content.

    The two-stage rollout evaluates the same coupled history for many receiver
    candidates, so the whole-cell predicate is the dominant cost. The returned
    array is read-only and must not be modified in place.
    """
    centers = np.ascontiguousarray(centers, float).reshape(-1, 2)
    return _cached_mixed_cells(centers.tobytes(), centers.shape, float(half),
                               tuple(history))
