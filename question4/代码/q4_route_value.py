"""Finite remaining-route proxy independent of latent source positions.

Other source means are scheduling anchors, not guaranteed clear positions.
Only current-source finite clearing is simulated. This is NOT exact V_pi0.
"""
import numpy as np


def insert_anchors(start, stations, anchors, forced_first=None):
    start = np.asarray(start, float)
    anchors = np.asarray(anchors, float).reshape(-1, 2)
    path = [np.array(p) for p in stations]
    remaining = list(range(len(anchors)))
    locked = 0
    if forced_first is not None:
        path.insert(0, np.array(anchors[forced_first]))
        remaining.remove(forced_first)
        locked = 1
    while remaining:
        best = None
        for j in remaining:
            q = anchors[j]
            for k in range(locked, len(path)+1):
                a = start if k == 0 else path[k-1]
                gain = np.linalg.norm(a-q)
                if k < len(path):
                    gain += np.linalg.norm(q-path[k])-np.linalg.norm(a-path[k])
                if best is None or gain < best[0]:
                    best = (gain, j, k)
        _, j, k = best
        path.insert(k, np.array(anchors[j])); remaining.remove(j)
    return np.asarray(path).reshape(-1, 2)


class TailValue:
    def __init__(self, start, stations, anchors, unknown, reorder=True, station_services=None):
        self.unknown = set(map(int, unknown))  # one-based channels
        self.services = 5*len(anchors)
        self.batches = len(stations)
        k = len(self.unknown)
        if not k:
            pass
        elif station_services is None:
            # Uniform proxy: every future stop is assumed to scan all channels.
            self.services += len(stations)*(6*k-1)
        else:
            station_services = np.asarray(station_services, float).reshape(-1)
            if len(station_services) != len(stations):
                raise ValueError('One service estimate per future station is required')
            # Per-station proxy driven by the effective-contribution filter.
            self.services += float(station_services.sum())
        paths = [insert_anchors(start, stations, anchors)]
        if reorder:
            paths += [insert_anchors(start, stations, anchors, j) for j in range(len(anchors))]
        self.first, self.internal = [], []
        for p in paths:
            if len(p):
                self.first.append(p[0])
                self.internal.append(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()/5)
        self.first = np.asarray(self.first).reshape(-1, 2)
        self.internal = np.asarray(self.internal)

    def __call__(self, endpoints, channel):
        endpoints = np.asarray(endpoints).reshape(-1, 2)
        if not len(self.first):
            return np.zeros(len(endpoints))
        values = np.linalg.norm(endpoints[:, None]-self.first[None], axis=2)/5+self.internal[None]
        switch = int(bool(self.batches and self.unknown) and channel not in self.unknown)
        return values.min(axis=1)+self.services+switch


def finish_costs(cover, start, active, sources):
    """Same feedback-only NN clearing as v1, additionally return actual endpoints.

    Source truth only determines success; it never determines the next center.
    Inactive rows already succeeded/are near; endpoint remains start.

    A failed ``clear`` is public feedback too: the executed policy rebuilds the
    cover with ``possible_mixed_cells`` and therefore drops every cell that lies
    ENTIRELY inside the 20 m attempt disk, not only the cell that was aimed at.
    The rollout must remove the same cells, otherwise the predicted number of
    follow-up attempts (and hence every measured-versus-clear comparison) is
    inconsistent with the simulator-facing policy.
    """
    active = active.copy()
    costs = np.zeros(len(sources))
    endpoints = np.repeat(np.asarray(start)[None], len(sources), axis=0)
    alive = active.any(axis=1)
    distances = np.repeat(np.linalg.norm(cover.c-start, axis=1)[None], len(sources), axis=0)
    keep_cache = {}
    for _ in range(len(cover.c)):
        ids = np.flatnonzero(alive)
        if not len(ids):
            break
        if not active[ids].any(axis=1).all():
            raise ArithmeticError('Scenario escaped finite clearing support')
        # Covers may define their own next-cell rule (e.g. posterior-mass
        # greedy). The fallback is the nearest surviving cell, which is what
        # the executed geometry policy uses when no masses are attached.
        if hasattr(cover, 'best_cell'):
            choices = cover.best_cell(distances[ids], active[ids])
        else:
            choices = np.argmin(np.where(active[ids], distances[ids], np.inf), axis=1)
        costs[ids] += distances[ids, choices]/5+3
        success = np.linalg.norm(sources[ids]-cover.c[choices], axis=1) <= 20
        costs[ids[success]] += 2
        endpoints[ids] = cover.c[choices]
        for choice in np.unique(choices):
            choice = int(choice)
            if choice not in keep_cache:
                keep_cache[choice] = cover.remaining_after_clear(cover.c[choice])
            rows = ids[(choices == choice) & ~success]
            if len(rows):
                active[rows] &= keep_cache[choice]
        distances[ids] = cover.distance[choices]
        alive[ids[success]] = False
    if alive.any():
        raise ArithmeticError('Finite clearing support exhausted')
    return costs, endpoints
