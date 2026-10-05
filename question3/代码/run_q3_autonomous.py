"""Paired local complete-task experiments. Never calls the official simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from q3_environment import Environment
from q3_policy import AutonomousPolicy

OUT=Path(__file__).resolve().parent.parent/'数据'/'实验结果'/'autonomous'


def version():
    root=Path(__file__).resolve().parent
    return hashlib.sha256(b''.join((root/f).read_bytes() for f in
        ('q3_policy.py','q3_belief.py','q3_environment.py','run_q3_autonomous.py'))).hexdigest()[:16]


VERSION=version()


def episode(seed,mode,particles=48,scan_scale=1.,distribution='uniform'):
    env=Environment(seed)
    if distribution=='cluster':
        # Offline stress environment, deliberately different from the prior.
        rng=np.random.default_rng(seed+8700000)
        center=np.array([1050.,650.])
        for source in env.sources.values():
            p=center+rng.normal(0,180,2)
            p*=min(1.,1790/max(np.linalg.norm(p),1.))
            source['position']=p.tolist()
    elif distribution=='edge':
        rng=np.random.default_rng(seed+8700000)
        for source in env.sources.values():
            angle=rng.uniform(0,2*np.pi)
            source['position']=(rng.uniform(1650,1800)*np.array([np.cos(angle),np.sin(angle)])).tolist()
    elif distribution!='uniform':raise ValueError('Unknown environment distribution')
    # The planner seed is fixed and carries no information about the hidden
    # environment seed. Different worlds reach it only through public feedback.
    policy=AutonomousPolicy(mode=mode,seed=9101,particles=particles,scan_scale=scan_scale)
    started=time.perf_counter();logs=[];planning=[];updates=[];error=None
    try:
        for step in range(5000):
            t=time.perf_counter();action=policy.next_action();planning.append(time.perf_counter()-t)
            if action is None:break
            result=env.act(action['action'],action['channel'],action['position'])
            t=time.perf_counter();policy.accept_feedback(result);updates.append(time.perf_counter()-t)
            logs.append(dict(step=step+1,**action,feedback=result,
                             confirmed=int(policy.confirmed.sum()),cleared=int(policy.cleared.sum()),
                             absent=int(policy.absent.sum()),receiver_channel=policy.channel,
                             exists=policy.bank.exists.tolist() if policy.numeric_ok else None))
            assert np.linalg.norm(policy.position-env.position)<1e-9
            assert policy.channel==env.channel
            if env.time>360000:raise RuntimeError('Exceeded 100 virtual hours')
            if time.perf_counter()-started>1200:raise RuntimeError('Exceeded 20 real minutes')
        else:raise RuntimeError('Exceeded finite episode action guard')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    wall=time.perf_counter()-started
    actual_all=env.successes==env.n
    certificate=policy.complete()
    if certificate and not actual_all:
        error='Incorrect completion certificate'
    summary=dict(seed=seed,mode=mode,particles=particles,scan_scale=scan_scale,distribution=distribution,
                 source_count=env.n,cleared=env.successes,clear_fraction=env.successes/env.n,
                 complete=actual_all and certificate and error is None,certificate=certificate,
                 virtual_s=env.time,average_per_cleared_s=env.time/env.successes if env.successes else None,
                 move_m=env.distance,move_s=env.distance/5,measurements=env.measurements,
                 measurement_s=5*env.measurements,switches=env.switches,switch_s=env.switches,
                 clear_attempts=env.attempts,clear_s=3*env.attempts+2*env.successes,
                 actions=len(logs),opportunistic_scans=policy.extra_scans,
                 adaptive_discovery_scans=policy.frontier_scans,
                 target_switches=policy.target_switches,numerical_fallbacks=policy.numeric_failures,
                 completion_evidence=dict(cleared_channels=(np.flatnonzero(policy.cleared)+1).tolist(),
                                          absent_channels=policy.absence_reason,
                                          confirmed_original_sources=int(policy.confirmed.sum())),
                 wall_s=wall,planning_median_ms=float(np.median(planning)*1000),
                 planning_p95_ms=float(np.quantile(planning,.95)*1000),
                 update_p95_ms=float(np.quantile(updates,.95)*1000) if updates else 0,error=error)
    return dict(version=VERSION,summary=summary,logs=logs,truth=env.truth_for_evaluation())


def balanced_seeds(start,per_count):
    buckets={n:[] for n in range(10,17)}
    seed=start
    while any(len(v)<per_count for v in buckets.values()):
        n=Environment(seed).n
        if len(buckets[n])<per_count:buckets[n].append(seed)
        seed+=1
    return [s for n in range(10,17) for s in buckets[n]]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,nargs='*')
    parser.add_argument('--start',type=int,default=930000)
    parser.add_argument('--per-count',type=int,default=2)
    parser.add_argument('--modes',nargs='+',default=['sequential','discover_first','joint'])
    parser.add_argument('--particles',type=int,default=48)
    parser.add_argument('--scan-scale',type=float,default=1.)
    parser.add_argument('--distribution',default='uniform',choices=['uniform','cluster','edge'])
    parser.add_argument('--label',default='test')
    args=parser.parse_args()
    folder=OUT/args.label;folder.mkdir(parents=True,exist_ok=True)
    seeds=args.seeds or balanced_seeds(args.start,args.per_count)
    summaries=[]
    for seed in seeds:
        for mode in args.modes:
            path=folder/f'{args.distribution}_{seed}_{mode}.json'
            old=json.loads(path.read_text()) if path.exists() else None
            if old and old['version']==VERSION and old['summary']['particles']==args.particles and old['summary']['scan_scale']==args.scan_scale:
                result=old
            else:
                result=episode(seed,mode,args.particles,args.scan_scale,args.distribution)
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
            s=result['summary'];summaries.append(s)
            print(f"{seed} {mode}: {s['cleared']}/{s['source_count']}, {s['virtual_s']:.1f}s, "
                  f"{s['actions']} actions, CPU {s['wall_s']:.2f}s, error={s['error']}",flush=True)
    (folder/'summaries.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
