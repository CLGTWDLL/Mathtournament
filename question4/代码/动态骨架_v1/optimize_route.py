"""Deterministic offline local search; never overwrite the original backbone.

Single-site proposals recheck all and only possibly affected certificate cells.
Deletion and accepted output also get full checks. Environment seeds are unused.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from q4_sparse_backbone import BACKBONE_FILE, CELLS, certified, independent_cover_check


def length(route):
    return float(np.linalg.norm(np.diff(np.vstack((np.zeros(2), route)), axis=0), axis=1).sum())


def two_opt(route):
    route = np.asarray(route).copy()
    for _ in range(12):
        changed = False
        for i in range(len(route)):
            a = route[i-1] if i else np.zeros(2)
            for j in range(i+1, len(route)):
                old, new = np.linalg.norm(a-route[i]), np.linalg.norm(a-route[j])
                if j+1 < len(route):
                    old += np.linalg.norm(route[j]-route[j+1])
                    new += np.linalg.norm(route[i]-route[j+1])
                if new < old-1e-7:
                    route[i:j+1] = route[i:j+1][::-1]
                    changed = True
        if not changed:
            break
    return route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--output', type=Path, default=HERE/'backbone.json')
    args = parser.parse_args()
    if args.output.resolve() == BACKBONE_FILE.resolve() or args.output.exists():
        raise ValueError('Output must be a new file, never the original route')
    raw = BACKBONE_FILE.read_bytes()
    route = two_opt(np.asarray(json.loads(raw)['route'], float))
    start_length = length(route)
    if not certified(np.vstack((np.zeros(2), route))).all():
        raise AssertionError('Initial route invalid')
    rng = np.random.default_rng(args.seed)
    began = time.perf_counter()
    progress = []
    best = length(route)/5+100*len(route)
    for it in range(args.iterations):
        trial = route.copy()
        j = int(rng.integers(len(trial)))
        old = trial[j].copy()
        delete = it % 13 == 0
        if delete:
            trial = np.delete(trial, j, axis=0)
        else:
            previous = route[j-1] if j else np.zeros(2)
            following = route[j+1] if j+1 < len(route) else previous
            if it % 3 == 0:
                trial[j] += rng.uniform(.02, .3)*((previous+following)/2-old)
            else:
                trial[j] += rng.normal(0, rng.choice([3., 8., 20., 50.]), 2)
            trial[j] = np.round(trial[j], 6)
            if np.linalg.norm(trial[j]) > 2800:
                continue
        # Cheap objective test before geometry. Order is retained here.
        value = length(trial)/5+100*len(trial)
        if value >= best-1e-7:
            continue
        affected = np.linalg.norm(np.abs(CELLS-old)+12.5, axis=1) <= 999.9
        if not delete:
            affected |= np.linalg.norm(np.abs(CELLS-trial[j])+12.5, axis=1) <= 999.9
        stations = np.vstack((np.zeros(2), trial))
        if not certified(stations, CELLS[affected]).all():
            continue
        route = two_opt(trial)
        best = length(route)/5+100*len(route)
        progress.append(dict(iteration=it+1, stations=len(route), length_m=length(route)))
        if len(progress) % 10 == 0:
            print(json.dumps(progress[-1]), flush=True)
    route = np.round(route, 6)
    stations = np.vstack((np.zeros(2), route))
    if not certified(stations).all() or not independent_cover_check(stations):
        raise AssertionError('Final rounded route fails independent continuous audit')
    payload = dict(route=route.tolist(), stations=len(route), open_route_m=length(route),
                   original_open_route_m=start_length, original_sha256=hashlib.sha256(raw).hexdigest(),
                   optimizer_seed=args.seed, iterations=args.iterations,
                   offline_wall_s=time.perf_counter()-began, accepted_changes=progress,
                   certificate='25 m squares, whole-cell radius 999.9 m, Qhull audit',
                   environment_seeds_used=False)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in payload.items() if k not in ('route', 'accepted_changes')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
