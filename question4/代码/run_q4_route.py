"""Serial fresh-seed comparison of geometry, validated-local and route scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import run_q4 as base
from run_q4_feedback import version as feedback_version
from q4_feedback_policy import Policy as GeometryPolicy
from q4_route_policy import Policy as RoutePolicy

ROOT = Path(__file__).resolve().parent
BASE_VERSION = feedback_version()


def version():
    return hashlib.sha256(BASE_VERSION.encode()+b''.join((ROOT/f).read_bytes() for f in
        ('q4_route_value.py', 'q4_route_policy.py', 'run_q4_route.py'))).hexdigest()[:16]


def episode(seed, variant, distribution, particles):
    old_policy, old_version = base.Policy, base.version
    base.Policy = lambda mode, count: GeometryPolicy('geometry', count) if variant == 'geometry' else RoutePolicy(variant, count)
    base.version = version
    try:
        result = base.episode(seed, 'joint', distribution, particles)
    finally:
        base.Policy, base.version = old_policy, old_version
    result['summary']['mode'] = variant
    decisions = [a['route_decision'] for a in result['logs'] if 'route_decision' in a]
    result['summary']['rescored_actions'] = len(decisions)
    result['summary']['accepted_changes'] = sum(d['accepted'] for d in decisions)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--start', type=int, default=1340000)
    p.add_argument('--modes', nargs='+', choices=['geometry', 'validated', 'route'], default=['geometry', 'validated', 'route'])
    p.add_argument('--distributions', nargs='+', default=['uniform', 'outward', 'cluster'])
    p.add_argument('--particles', type=int, default=32)
    p.add_argument('--label', default='route_holdout')
    args = p.parse_args()
    folder = ROOT.parent/'数据'/'实验结果'/args.label
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in args.distributions:
        for seed in args.seeds or base.balanced_seeds(args.start):
            for mode in args.modes:
                path = folder/f'{d}_{seed}_{mode}.json'
                r = json.loads(path.read_text()) if path.exists() else None
                if r is None or r['version'] != version() or r['summary']['particles'] != args.particles:
                    r = episode(seed, mode, d, args.particles)
                    path.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
                s = r['summary']; rows.append(s)
                print(f"{d} {seed} {mode}: {s['cleared']}/{s['source_count']}, T={s['virtual_s']:.2f}s, "
                      f"changes={s['accepted_changes']}, CPU={s['wall_s']:.2f}s, error={s['error']}", flush=True)
    (folder/'summaries.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
