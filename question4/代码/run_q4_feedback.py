"""Serial paired Q4 feedback ablations, keeping frozen first-stage files intact."""
import argparse
import hashlib
import json
from pathlib import Path
import run_q4 as base
from q4_policy import Policy as FrozenPolicy
from q4_feedback_policy import Policy as FeedbackPolicy

ROOT = Path(__file__).resolve().parent
BASE_VERSION = base.version()


def version():
    return hashlib.sha256(BASE_VERSION.encode()+b''.join((ROOT/f).read_bytes() for f in
        ('q4_feedback_geometry.py', 'q4_feedback_policy.py', 'run_q4_feedback.py'))).hexdigest()[:16]


def episode(seed, variant, distribution, particles):
    # No file mutation. Base episode is serial and restores its factory even
    # on failure; a future concurrent runner must pass an explicit factory.
    old_policy, old_version = base.Policy, base.version
    base.Policy = lambda mode, count: FrozenPolicy('joint', count) if variant == 'frozen' else FeedbackPolicy(variant, count)
    base.version = version
    try:
        record = base.episode(seed, 'joint', distribution, particles)
    finally:
        base.Policy, base.version = old_policy, old_version
    record['summary']['mode'] = variant
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--start', type=int, default=1240000)
    p.add_argument('--modes', nargs='+', choices=['frozen', 'geometry', 'feedback'], default=['frozen', 'geometry', 'feedback'])
    p.add_argument('--distributions', nargs='+', default=['uniform', 'outward', 'cluster'])
    p.add_argument('--particles', type=int, default=32)
    p.add_argument('--label', default='feedback_holdout')
    a = p.parse_args()
    folder = ROOT.parent/'数据'/'实验结果'/a.label
    folder.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dist in a.distributions:
        for seed in a.seeds or base.balanced_seeds(a.start):
            for variant in a.modes:
                path = folder/f'{dist}_{seed}_{variant}.json'
                r = json.loads(path.read_text()) if path.exists() else None
                if r is None or r['version'] != version() or r['summary']['particles'] != a.particles:
                    r = episode(seed, variant, dist, a.particles)
                    path.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n')
                s = r['summary']; summaries.append(s)
                print(f"{dist} {seed} {variant}: {s['cleared']}/{s['source_count']}, "
                      f"T={s['virtual_s']:.2f}s, attempts={s['clear_attempts']}, "
                      f"CPU={s['wall_s']:.2f}s, error={s['error']}", flush=True)
    (folder/'summaries.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
