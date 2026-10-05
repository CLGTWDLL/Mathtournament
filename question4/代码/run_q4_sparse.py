"""Paired local comparison: frozen dispatch policy vs sparse-backbone variant.

Same seeds/distributions as run_q4_dispatch.py (1360000/1370000 series). The
absence-certificate audit for the sparse variant uses the fine-grid Qhull check
(q4_sparse_backbone.independent_cover_check); the frozen policy keeps the
100 m audit from run_q4.py. Never touches the official simulator.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import run_q4 as base
from run_q4_dispatch import version as dispatch_version
from q4_dispatch_policy import Policy as DispatchPolicy
import q4_sparse_backbone as sparse

ROOT = Path(__file__).resolve().parent


def version():
    return hashlib.sha256(dispatch_version().encode()+b''.join((ROOT/f).read_bytes() for f in
        ('q4_sparse_backbone.py', 'q4_sparse_backbone.json', 'run_q4_sparse.py'))).hexdigest()[:16]


def episode(seed, mode, distribution, particles):
    if mode not in ('sparse', 'dispatch'):
        raise ValueError(mode)
    old = base.Policy, base.version, base.independent_cover_check, base.audit
    audit_wall_s = 0.
    def timed_audit(record):
        nonlocal audit_wall_s
        started = time.perf_counter()
        try:
            return old[3](record)
        finally:
            audit_wall_s = time.perf_counter()-started
    base.audit = timed_audit
    base.Policy = lambda unused, count: sparse.Policy('dispatch', count) if mode == 'sparse' else DispatchPolicy('dispatch', count)
    base.version = version
    if mode == 'sparse':
        base.independent_cover_check = sparse.independent_cover_check
    try:
        r = base.episode(seed, 'joint', distribution, particles)
    finally:
        base.Policy, base.version, base.independent_cover_check, base.audit = old
    r['summary']['mode'] = mode
    r['summary']['audit_wall_s'] = audit_wall_s
    r['summary']['dispatch_decisions'] = sum('dispatch_decision' in a for a in r['logs'])
    r['summary']['dispatch_changes'] = sum(a.get('dispatch_decision', {}).get('changed', False) for a in r['logs'])
    r['summary']['route_changes'] = sum(a.get('route_decision', {}).get('accepted', False) for a in r['logs'])
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--starts', type=int, nargs='+', default=[1360000, 1370000])
    p.add_argument('--label', default='sparse_backbone')
    p.add_argument('--modes', nargs='+', default=['dispatch', 'sparse'])
    p.add_argument('--distributions', nargs='+', default=['uniform', 'outward', 'cluster'])
    p.add_argument('--particles', type=int, default=32)
    args = p.parse_args()
    folder = ROOT.parent/'数据'/'实验结果'/args.label
    folder.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds or [s for start in args.starts for s in base.balanced_seeds(start)]
    summaries = []
    for d in args.distributions:
        for seed in seeds:
            for mode in args.modes:
                path = folder/f'{d}_{seed}_{mode}.json'
                r = json.loads(path.read_text()) if path.exists() else None
                if r is None or r['version'] != version() or r['summary']['particles'] != args.particles:
                    r = episode(seed, mode, d, args.particles)
                    path.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
                s = r['summary']
                summaries.append(s)
                print(f"{d} {seed} {mode}: {s['cleared']}/{s['source_count']}, T={s['virtual_s']:.1f}s, "
                      f"move={s['move_m']/5:.0f}s meas={5*s['measurements']}s wall={s['wall_s']:.1f}s "
                      f"audit={'ok' if r.get('audit') else 'none'} error={s['error']}", flush=True)
    (folder/'summaries.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')
    for d in args.distributions:
        pairs = {}
        for s in summaries:
            if s['distribution'] == d:
                pairs.setdefault(s['seed'], {})[s['mode']] = s['virtual_s']
        rows = [(v['dispatch'], v['sparse']) for v in pairs.values() if 'dispatch' in v and 'sparse' in v]
        if rows:
            a, b = np.array(rows).T
            diff = a-b
            se = diff.std(ddof=1)/np.sqrt(len(diff))
            print(f"{d}: dispatch {a.mean():.1f}s, sparse {b.mean():.1f}s, saved {diff.mean():.1f}s "
                  f"({diff.mean()/a.mean()*100:.1f}%), approx 95% interval [{diff.mean()-2*se:.0f}, {diff.mean()+2*se:.0f}], "
                  f"faster {(diff > 0).sum()}/{len(diff)}")


if __name__ == '__main__':
    main()
