"""Offline only. The planner receives no seed, count, type or hidden position."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from q4_environment import Environment
from q4_policy import Policy
from q4_coverage import domain_cells

ROOT = Path(__file__).resolve().parent


def version():
    files = ['q4_belief.py', 'q4_coverage.py', 'q4_environment.py', 'q4_policy.py', 'run_q4.py',
             'q3_belief.py', 'q3_policy.py', 'q3_environment.py', 'q2_analytic.py', 'solve_q2.py']
    return hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in files)).hexdigest()[:16]


def independent_cover_check(positions):
    """Qhull supporting halfplanes, independent of planner's angular-gap code."""
    stations = np.asarray(positions, float)
    cells = domain_cells()
    corners = np.array([[-50, -50], [-50, 50], [50, -50], [50, 50]])
    for c in cells:
        p = stations[np.linalg.norm(np.abs(stations-c)+50, axis=1) <= 999.9]
        if len(p) < 3:
            return False
        try:
            hull = ConvexHull(p)
        except Exception:
            return False
        eq = hull.equations
        if np.max((c+corners)@eq[:, :2].T+eq[:, 2]) >= -1e-7:
            return False
    return True


def audit(record):
    s = record['summary']
    env = Environment(s['seed'], s['distribution'])
    positive, cleared, negatives = set(), set(), {c: [] for c in range(1, 21)}
    for entry in record['logs']:
        result = env.act(entry['action'], entry['channel'], entry['position'])
        if result != entry['feedback']:
            raise AssertionError('Independent replay mismatch')
        kind, channel = result['kind'], entry['channel']
        if kind in ('direction', 'near'):
            positive.add(channel)
        if kind == 'success':
            positive.add(channel)
            cleared.add(channel)
        if kind == 'no_signal':
            negatives[channel].append(entry['position'])
    reasons = s['absence_reason']
    memo = {}
    for channel, reason in reasons.items():
        channel = int(channel)
        if channel in positive:
            raise AssertionError('Confirmed original source declared absent')
        if reason == 'sixteen_confirmed':
            assert len(positive) == 16
        elif reason == 'continuous_directional_cover':
            key = tuple(tuple(p) for p in negatives[channel])
            if key not in memo:
                memo[key] = independent_cover_check(negatives[channel])
            assert memo[key], 'Invalid directional absence certificate'
        else:
            raise AssertionError('Unknown certificate type')
    assert set(range(1, 21)) == cleared | set(map(int, reasons))
    assert len(cleared) == env.n == s['cleared']
    assert abs(env.time-s['virtual_s']) < 1e-8
    return dict(actions=len(record['logs']), absent_certificates=len(reasons),
                replay=True, continuous_hull_check=True)


def episode(seed, mode='joint', distribution='uniform', particles=32):
    env = Environment(seed, distribution)
    started = time.perf_counter()
    policy = Policy(mode, particles)
    logs, error = [], None
    try:
        for step in range(5000):
            a = policy.next_action()
            if a is None:
                break
            feedback = env.act(a['action'], a['channel'], a['position'])
            policy.accept_feedback(feedback)
            logs.append(dict(step=step+1, **a, feedback=feedback))
            assert np.linalg.norm(policy.position-env.position) < 1e-8
            assert policy.channel == env.channel
            if env.time > 360000 or time.perf_counter()-started > 1200:
                raise RuntimeError('Virtual/real time cap')
        else:
            raise RuntimeError('Action guard')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    summary = dict(seed=seed, mode=mode, distribution=distribution, particles=particles,
                   source_count=env.n, directional_count=sum(s['directional'] for s in env.sources.values()),
                   cleared=env.successes, clear_fraction=env.successes/env.n,
                   complete=policy.complete() and env.successes == env.n and error is None,
                   virtual_s=env.time, average_per_cleared_s=env.time/env.successes if env.successes else None,
                   move_m=env.distance, measurements=env.measurements, switches=env.switches,
                   clear_attempts=env.attempts, joint_batches=policy.joint_batches,
                   numeric_fallbacks=policy.numeric_fallbacks, absence_reason=policy.absence_reason,
                   wall_s=time.perf_counter()-started, error=error)
    record = dict(version=version(), summary=summary, logs=logs, truth=env.truth_for_evaluation())
    if summary['complete']:
        record['audit'] = audit(record)
    return record


def balanced_seeds(start):
    buckets = {}
    while len(buckets) < 7:
        n = Environment(start).n
        buckets.setdefault(n, start)
        start += 1
    return [buckets[n] for n in range(10, 17)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--start', type=int, default=1140000)
    parser.add_argument('--modes', nargs='+', default=['sequential', 'joint'])
    parser.add_argument('--distributions', nargs='+', default=['uniform'])
    parser.add_argument('--particles', type=int, default=32)
    parser.add_argument('--label', default='baseline')
    args = parser.parse_args()
    folder = ROOT.parent/'数据'/'实验结果'/args.label
    folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for distribution in args.distributions:
        for seed in args.seeds or balanced_seeds(args.start):
            for mode in args.modes:
                path = folder/f'{distribution}_{seed}_{mode}.json'
                record = json.loads(path.read_text()) if path.exists() else None
                if record is None or record['version'] != version() or record['summary']['particles'] != args.particles:
                    record = episode(seed, mode, distribution, args.particles)
                    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
                s = record['summary']
                summaries.append(s)
                print(f"{distribution} {seed} {mode}: {s['cleared']}/{s['source_count']}, "
                      f"{s['virtual_s']:.1f}s, CPU {s['wall_s']:.2f}s, "
                      f"joint batches {s['joint_batches']}, error={s['error']}", flush=True)
    (folder/'summaries.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
