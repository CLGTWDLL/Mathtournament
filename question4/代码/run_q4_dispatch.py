"""Fresh-seed task-dispatch comparison, after terminal-value development."""
import argparse
import hashlib
import json
from pathlib import Path
import run_q4 as base
from run_q4_route import version as route_version
from q4_feedback_policy import Policy as GeometryPolicy
from q4_dispatch_policy import Policy as DispatchPolicy

ROOT = Path(__file__).resolve().parent
BASE_VERSION = route_version()


def version():
    return hashlib.sha256(BASE_VERSION.encode()+b''.join((ROOT/f).read_bytes() for f in
        ('q4_dispatch_policy.py', 'run_q4_dispatch.py'))).hexdigest()[:16]


def episode(seed, mode, distribution, particles):
    old_policy, old_version = base.Policy, base.version
    base.Policy = lambda unused, count: GeometryPolicy('geometry', count) if mode == 'geometry' else DispatchPolicy(mode, count)
    base.version = version
    try:
        r = base.episode(seed, 'joint', distribution, particles)
    finally:
        base.Policy, base.version = old_policy, old_version
    s = r['summary']; s['mode'] = mode
    s['dispatch_decisions'] = sum('dispatch_decision' in a for a in r['logs'])
    s['dispatch_changes'] = sum(a.get('dispatch_decision', {}).get('changed', False) for a in r['logs'])
    s['route_changes'] = sum(a.get('route_decision', {}).get('accepted', False) for a in r['logs'])
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--start', type=int, default=1360000)
    p.add_argument('--label', default='dispatch_holdout')
    p.add_argument('--modes', nargs='+', default=['geometry', 'dispatch', 'dispatch_route'])
    p.add_argument('--distributions', nargs='+', default=['uniform', 'outward', 'cluster'])
    p.add_argument('--particles', type=int, default=32)
    args = p.parse_args()
    folder = ROOT.parent/'数据'/'实验结果'/args.label; folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for d in args.distributions:
        for seed in args.seeds or base.balanced_seeds(args.start):
            for mode in args.modes:
                path = folder/f'{d}_{seed}_{mode}.json'
                r = json.loads(path.read_text()) if path.exists() else None
                if r is None or r['version'] != version() or r['summary']['particles'] != args.particles:
                    r = episode(seed, mode, d, args.particles)
                    path.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
                s = r['summary']; summaries.append(s)
                print(f"{d} {seed} {mode}: {s['cleared']}/{s['source_count']}, T={s['virtual_s']:.2f}s, "
                      f"dispatch changes={s['dispatch_changes']}, CPU={s['wall_s']:.2f}s, error={s['error']}", flush=True)
    (folder/'summaries.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
