"""Independent replay of saved actions, accounting and absence certificates."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from q3_environment import Environment,fixed_stations

OUT=Path(__file__).resolve().parent.parent/'数据'/'实验结果'/'autonomous'


@lru_cache(maxsize=128)
def disk_cover_certificate(positions):
    """Prove A subset union B(q,1000) without posterior weights or particles."""
    c=np.array([[0.,0.]])
    half=1800.
    children=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    for _ in range(24):
        c=c[np.linalg.norm(np.maximum(np.abs(c)-half,0),axis=1)<=1800]
        if not len(c):return True
        possible=np.ones(len(c),bool)
        for q in positions:
            possible &= np.linalg.norm(np.abs(c-q)+half,axis=1)>1000
        c=c[possible]
        if not len(c):return True
        c=(c[:,None,:]+half/2*children).reshape(-1,2)
        half/=2
    return False


def audit(path):
    data=json.loads(path.read_text());summary=data['summary'];truth=data['truth']
    env=Environment(summary['seed'])
    env.n=truth['source_count']
    env.sources={int(c):v for c,v in truth['sources'].items()}
    positives=set();negative={c:[] for c in range(1,21)}
    for row in data['logs']:
        response=env.act(row['action'],row['channel'],row['position'])
        expected=row['feedback']
        assert response['kind']==expected['kind'],(path,row['step'],'feedback')
        if 'angle' in response:assert response['angle']==expected['angle']
        assert abs(response['virtual_time_s']-expected['virtual_time_s'])<1e-7
        assert env.channel==row['receiver_channel']
        if response['kind'] in ('direction','near','success'):positives.add(row['channel'])
        if response['kind']=='no_signal':negative[row['channel']].append(tuple(row['position']))
    assert summary['complete'] and summary['cleared']==env.n==env.successes
    assert abs(env.time-summary['virtual_s'])<1e-7
    assert abs(summary['move_s']+summary['measurement_s']+summary['switch_s']+summary['clear_s']-env.time)<1e-7
    certificate=summary['completion_evidence']
    assert set(certificate['cleared_channels'])==env.cleared
    absent={int(c):reason for c,reason in certificate['absent_channels'].items()}
    assert set(absent)|env.cleared==set(range(1,21))
    assert not (set(absent)&env.cleared)
    assert certificate['confirmed_original_sources']==len(positives)
    for c,reason in absent.items():
        assert c not in positives
        if reason=='count_16':assert len(positives)==16
        elif reason=='seven_negative_stations':
            assert all(any(np.linalg.norm(q-np.array(p))<.01 for p in negative[c]) for q in fixed_stations()[1:])
        elif reason=='geometry_empty':
            assert disk_cover_certificate(tuple(sorted(set(negative[c])))),(path,c,'unverified cover')
        else:raise AssertionError('Unknown certificate type')
    return dict(file=str(path.relative_to(OUT)),actions=len(data['logs']),cleared=env.successes,
                absence_certificates=len(absent),verified=True)


def main():
    paths=list((OUT/'final').glob('uniform_*_*.json'))
    for d in ('cluster','edge'):paths+=list((OUT/f'stress_{d}').glob(f'{d}_*_joint.json'))
    results=[audit(p) for p in sorted(paths)]
    summary=dict(episodes=len(results),actions=sum(r['actions'] for r in results),
                 cleared=sum(r['cleared'] for r in results),
                 absence_certificates=sum(r['absence_certificates'] for r in results),
                 all_verified=all(r['verified'] for r in results),records=results)
    (OUT/'independent_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k!='records'})


if __name__=='__main__':main()
