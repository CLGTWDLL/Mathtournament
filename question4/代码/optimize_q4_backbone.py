"""Offline local search for a short Q4 patrol backbone under the fine certificate.

Objective: open-route seconds (length/5) + STATION_S per station, subject to
q4_sparse_backbone.certified(...) on 25 m squares. Start point: the 875 m
triangular mesh pruned on the fine grid. Randomised; keeps the best found and
re-verifies it with the independent Qhull check before writing the JSON.

    python3 optimize_q4_backbone.py --seconds 1500 --station-s 120 --seed 7
"""
import argparse
import json
import time
import numpy as np
from q4_coverage import open_route
from q4_sparse_backbone import certified, independent_cover_check, BACKBONE_FILE


def route_len(route, start=np.zeros(2)):
    return float(np.linalg.norm(np.diff(np.vstack((start, route)), axis=0), axis=1).sum())


def lattice(spacing=850., offset=(437., 0.), rmax=2800.):
    pts = np.array([[spacing*(i+j/2), spacing*np.sqrt(3)*j/2] for j in range(-6, 7) for i in range(-8, 9)])
    pts = pts+np.asarray(offset)
    pts = pts[np.linalg.norm(pts, axis=1) < rmax]
    return np.vstack((np.zeros(2), pts))


def prune(stations):
    if not certified(stations).all():
        raise AssertionError('Initial mesh is not a fine-grid certificate')
    for i in range(len(stations)-1, 0, -1):
        trial = np.delete(stations, i, axis=0)
        if certified(trial).all():
            stations = trial
    return stations


def objective(stations, station_s):
    route = open_route(stations[1:], np.zeros(2))
    length = route_len(route)
    return length/5+station_s*(len(stations)-1), length, route


def search(stations, seconds, station_s, rng, log=print):
    best, best_len, best_route = objective(stations, station_s)
    t0, it = time.time(), 0
    while time.time()-t0 < seconds:
        it += 1
        trial = stations.copy()
        move = rng.random()
        if move < .2 and len(trial) > 4:
            trial = np.delete(trial, rng.integers(1, len(trial)), axis=0)
            for _ in range(rng.integers(0, 4)):
                trial[rng.integers(1, len(trial))] += rng.normal(0, 120, 2)
        elif move < .6:
            trial[rng.integers(1, len(trial))] += rng.normal(0, rng.choice([20., 60., 150.]), 2)
        elif move < .8:
            trial[rng.integers(1, len(trial))] *= 1-rng.uniform(-.03, .06)
        else:
            for j in rng.choice(np.arange(1, len(trial)), 2, replace=False):
                trial[j] += rng.normal(0, 60, 2)
        if not certified(trial).all():
            continue
        value, length, route = objective(trial, station_s)
        if value < best-1e-6:
            stations, best, best_len, best_route = trial, value, length, route
            log(f'it={it} t={time.time()-t0:.0f}s stations={len(stations)-1} route={best_len:.0f}m objective={best:.0f}')
    return stations, best_len, best_route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=1500)
    parser.add_argument('--station-s', type=float, default=120., help='penalty seconds per station')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--start', help='existing backbone JSON to continue from')
    parser.add_argument('--out', default=str(BACKBONE_FILE))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.start:
        stations = np.vstack((np.zeros(2), np.asarray(json.loads(open(args.start).read())['route'], float)))
    else:
        stations = prune(lattice())
        print(f'pruned mesh: {len(stations)-1} stations, {route_len(open_route(stations[1:], np.zeros(2))):.0f} m')
    stations, length, route = search(stations, args.seconds, args.station_s, rng)
    if not independent_cover_check(np.vstack((np.zeros(2), route))):
        raise AssertionError('Qhull audit rejected the optimised backbone')
    json.dump(dict(route=np.round(route, 6).tolist(), stations=len(route), open_route_m=round(length, 3),
                   certificate='q4_coverage.certified_cells on 25 m squares, radius 999.9',
                   objective=dict(station_s=args.station_s, seconds=args.seconds, seed=args.seed)),
              open(args.out, 'w'), indent=1)
    print(f'wrote {args.out}: {len(route)} stations, {length:.0f} m')


if __name__ == '__main__':
    main()
