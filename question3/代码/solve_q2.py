"""Q2: conditional Monte Carlo for one additional bearing measurement.

Only NumPy is required. Synthetic experiments, not official simulator tests.
All decisions depend on observations and conservative feasible cells; hidden
source coordinates are used only by the simulated sensor/clear oracle.
"""
from __future__ import annotations
import csv
import json
import math
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / '问题2结果'
RAD = np.pi / 180
EPS = 1.005 * RAD  # ±1 degree before rounding plus 0.005 degree quantization
SCENARIOS = [
    dict(key='center', name='圆心首次检测，朝东', first=[0., 0.], bearing=0.),
    dict(key='outward', name='近边界首次检测，朝外', first=[1500., 0.], bearing=0.),
    dict(key='tangent', name='近边界首次检测，沿切向', first=[1500., 0.], bearing=90.),
]


def local_origin(sc):
    t = sc['bearing'] * RAD
    u, v = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    p = np.array(sc['first'])
    return np.array([-p @ u, -p @ v])


def world_point(sc, local):
    t = sc['bearing'] * RAD
    a = np.asarray(local)
    return (np.array(sc['first']) + a[..., 0, None] * np.array([np.cos(t), np.sin(t)])
            + a[..., 1, None] * np.array([-np.sin(t), np.cos(t)]))


def sample_conditional(sc, n, seed):
    """Uniform world-disk location and U[1000,1500] radius, conditioned on
    rounded first bearing and successful direction (>5m). Exact rejection.
    Unrounded first error U[-1,1]; observed first angle has a 0.01deg bin.
    """
    rng = np.random.default_rng(seed)
    origin = local_origin(sc)
    pts, radii = [], []
    total = 0
    while total < n:
        batch = max(4096, (n-total)*8)
        r = np.sqrt(25 + rng.random(batch)*(1500**2-25))
        phi_deg = rng.uniform(-1.005, 1.005, batch)
        phi = phi_deg * RAD
        p = np.column_stack([r*np.cos(phi), r*np.sin(phi)])
        R = rng.uniform(1000, 1500, batch)
        overlap = np.maximum(0, np.minimum(1., .005-phi_deg)
                             - np.maximum(-1., -.005-phi_deg))
        ok = ((np.linalg.norm(p-origin, axis=1) <= 1800)
              & (R >= r) & (rng.random(batch) < overlap/.01))
        pts.append(p[ok]); radii.append(R[ok]); total += int(ok.sum())
    p, R = np.concatenate(pts)[:n], np.concatenate(radii)[:n]
    noise = rng.uniform(-1, 1, n)
    return dict(g=p, radius=R, noise=noise)


class CoverModel:
    def __init__(self, sc, cell=25.):
        self.sc, self.cell, self.h = sc, cell, cell/2
        # Cells are aligned to the first bearing, with one row centered on it.
        ys = np.arange(-math.ceil(1500*np.sin(EPS)/cell)-1,
                       math.ceil(1500*np.sin(EPS)/cell)+2)*cell
        xs = np.arange(-1, math.ceil(1500/cell)+2)*cell
        xx, yy = np.meshgrid(xs, ys)
        c = np.column_stack([xx.ravel(), yy.ravel()])
        h = self.h
        lo = np.maximum(np.abs(c)-h, 0)
        hi = np.abs(c)+h
        n = np.array([[np.sin(EPS), np.cos(EPS)], [np.sin(EPS), -np.cos(EPS)]])
        feasible = ((np.sum(lo*lo, axis=1) <= 1500**2+1e-8)
                    & (np.sum(hi*hi, axis=1) > 25)
                    & np.all(c@n.T+h*np.sum(np.abs(n), axis=1) >= -1e-9, axis=1))
        d = np.maximum(np.abs(c-local_origin(sc))-h, 0)
        feasible &= np.sum(d*d, axis=1) <= 1800**2+1e-8
        self.c = c[feasible]
        delta = self.c[:, None, :]-self.c[None, :, :]
        self.dist = np.linalg.norm(delta, axis=2)
        self.fully_covered = np.sum((np.abs(delta)+h)**2, axis=2) <= 400+1e-9
        assert cell/np.sqrt(2) <= 20
        assert len(self.c) > 0

    def posterior_cells(self, s, samples):
        g, R, noise = samples['g'], samples['radius'], samples['noise']
        n = len(g)
        delta = g-s
        d = np.linalg.norm(delta, axis=1)
        near = d <= 5
        signal = d <= R
        near &= signal
        direction = signal & ~near
        no_signal = ~signal
        active = np.ones((n, len(self.c)), dtype=bool)
        if np.linalg.norm(s) < 1e-9:
            # Repeated first measurement has the same error and same readout.
            return active, np.zeros(n, bool), np.ones(n, bool), np.zeros(n, bool)
        theta = np.round(np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))+noise, 2)*RAD
        ids = np.flatnonzero(direction)
        if len(ids):
            rel = self.c-s
            for angles, sign in [(theta[ids]-EPS, 1), (theta[ids]+EPS, -1)]:
                normals = np.column_stack([-sign*np.sin(angles), sign*np.cos(angles)])
                active[ids] &= normals@rel.T + self.h*np.abs(normals).sum(axis=1)[:, None] >= -1e-8
            dmin = np.linalg.norm(np.maximum(np.abs(rel)-self.h, 0), axis=1)
            dmax = np.linalg.norm(np.abs(rel)+self.h, axis=1)
            active[ids] &= (dmin <= 1500+1e-8) & (dmax > 5)
        if no_signal.any():
            # First signal implies R>=d1; second no_signal implies d2>R.
            # Thus d2>1000 AND d2>d1. The latter is a linear halfplane in g.
            dmax = np.linalg.norm(np.abs(self.c-s)+self.h, axis=1)
            min_projection = self.c@s-self.h*np.abs(s).sum()
            possible = (dmax > 1000) & (min_projection < (s@s)/2+1e-8)
            active[no_signal] &= possible
        active[near] = False
        if np.any(~active.any(axis=1) & ~near):
            raise AssertionError('A non-near observation removed every feasible cell')
        return active, near, direction, no_signal

    def clear_route(self, position, active, g):
        """A deterministic executable continuation: nearest remaining cell
        center, then remove cells wholly covered by that clear disk. Stop
        immediately when the oracle confirms success. Covers the continuous
        feasible set, not merely the sampled source locations.
        """
        active = active.copy()
        n = len(g)
        cost, counts, length = np.zeros(n), np.zeros(n, int), np.zeros(n)
        alive = active.any(axis=1)
        dists = np.repeat(np.linalg.norm(self.c-position, axis=1)[None, :], n, axis=0)
        for _ in range(len(self.c)):
            ids = np.flatnonzero(alive)
            if not len(ids): break
            if np.any(~active[ids].any(axis=1)):
                raise AssertionError('Cover exhausted without clearing the source')
            choice = np.argmin(np.where(active[ids], dists[ids], np.inf), axis=1)
            travel = dists[ids, choice]
            center = self.c[choice]
            success = np.linalg.norm(g[ids]-center, axis=1) <= 20+1e-9
            length[ids] += travel
            counts[ids] += 1
            cost[ids] += travel/5 + 3 + 2*success
            active[ids] &= ~self.fully_covered[choice]
            dists[ids] = self.dist[choice]
            alive[ids[success]] = False
        if alive.any(): raise AssertionError('Cover failed')
        return cost, counts, length

    def evaluate(self, s, samples, baseline=False):
        s = np.array(s, float)
        g = samples['g']; n = len(g)
        if baseline:
            active = np.ones((n, len(self.c)), bool)
            cost, count, length = self.clear_route(np.zeros(2), active, g)
            return dict(time=cost, clears=count, length=length,
                        signal=np.ones(n, bool), near=np.zeros(n, bool))
        active, near, direction, no = self.posterior_cells(s, samples)
        cost, count, length = self.clear_route(s, active, g)
        cost += np.linalg.norm(s)/5 + 5
        length += np.linalg.norm(s)
        cost[near] += 5
        count[near] = 1
        return dict(time=cost, clears=count, length=length, signal=~no, near=near)


def statistics(result):
    t = result['time']; n = len(t)
    se = float(t.std(ddof=1)/np.sqrt(n))
    return dict(n=n, mean_s=float(t.mean()), se_s=se,
                ci95_mean_s=[float(t.mean()-1.96*se), float(t.mean()+1.96*se)],
                sd_s=float(t.std(ddof=1)), p50_s=float(np.quantile(t,.5)),
                p90_s=float(np.quantile(t,.9)), mean_clears=float(result['clears'].mean()),
                p_one_clear=float((result['clears']==1).mean()),
                max_clears=int(result['clears'].max()), mean_path_m=float(result['length'].mean()),
                p_signal=float(result['signal'].mean()), p_near=float(result['near'].mean()))


def evaluate_grid(model, samples, xs, ys, description):
    z=np.empty((len(ys),len(xs)))
    for j,y in enumerate(ys):
        for i,x in enumerate(xs): z[j,i]=model.evaluate([x,y],samples)['time'].mean()
        if j % 8 == 0: print(f'{description}: row {j+1}/{len(ys)}',flush=True)
    return z


def run():
    OUT.mkdir(exist_ok=True)
    start=time.time(); results=[]
    for k,sc in enumerate(SCENARIOS):
        print('SCENARIO',sc['name'],flush=True)
        model=CoverModel(sc)
        train=sample_conditional(sc,1024,20260910+100*k)
        length=float(np.max(model.c[:,0]))
        xs=np.unique(np.r_[np.linspace(-200,length+200,25),0.])
        ys=np.linspace(-1000,1000,25)
        z=evaluate_grid(model,train,xs,ys,sc['key']+' coarse')
        base_train=model.evaluate([0,0],train,True)['time'].mean()
        j,i=np.unravel_index(np.argmin(z),z.shape)
        dx=(length+400)/24;dy=2000/24
        fx=np.linspace(xs[i]-dx,xs[i]+dx,17)
        fy=np.linspace(ys[j]-dy,ys[j]+dy,17)
        tune=sample_conditional(sc,2048,20260911+100*k)
        fz=evaluate_grid(model,tune,fx,fy,sc['key']+' refine')
        rank=np.argsort(fz,axis=None)[:8]
        candidates=[]
        for index in rank:
            jj,ii=np.unravel_index(index,fz.shape)
            candidates.append([float(fx[ii]),float(fy[jj])])
        # Coarse alternatives from distinct neighborhoods protect against one
        # noisy basin; they enter the independent selection batch as well.
        coarse_rank=np.argsort(z,axis=None)
        for index in coarse_rank:
            jj,ii=np.unravel_index(index,z.shape); p=[float(xs[ii]),float(ys[jj])]
            if all(np.linalg.norm(np.array(p)-q)>max(dx,dy) for q in candidates):
                candidates.append(p)
            if len(candidates)>=12:break
        selection=sample_conditional(sc,4096,20260912+100*k)
        choices=[model.evaluate(p,selection) for p in candidates]
        winner=int(np.argmin([r['time'].mean() for r in choices]));best=candidates[winner]
        # Independent evaluation: never report training/selection confidence as
        # out-of-sample performance of the chosen point.
        test=sample_conditional(sc,16384,20260913+100*k)
        base=model.evaluate([0,0],test,True)
        selected=model.evaluate(best,test)
        difference=base['time']-selected['time']
        se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
        basetune=float(model.evaluate([0,0],tune,True)['time'].mean())
        se_table=[]
        for p,r in zip(candidates,choices):
            d=r['time']-choices[winner]['time']
            se_table.append(dict(local=p,mean_s=float(r['time'].mean()),
                                 difference_to_selected_s=float(d.mean()),
                                 paired_se_s=float(d.std(ddof=1)/np.sqrt(len(d)))))
        altmodel=CoverModel(sc,12.5)
        sensitivity_samples=sample_conditional(sc,4096,20260914+100*k)
        sensitive=altmodel.evaluate(best,sensitivity_samples)
        sensitivebase=altmodel.evaluate([0,0],sensitivity_samples,True)
        same_selected=model.evaluate(best,sensitivity_samples)
        same_base=model.evaluate([0,0],sensitivity_samples,True)
        cell_delta=sensitive['time']-same_selected['time']
        cell_se=float(cell_delta.std(ddof=1)/np.sqrt(len(cell_delta)))
        jj,ii=np.where(fz<=fz.min()+5)
        region=dict(tolerance_s=5,reference_min_s=float(fz.min()),count=len(jj),
                    local_x_range=[float(fx[ii].min()),float(fx[ii].max())],
                    local_y_range=[float(fy[jj].min()),float(fy[jj].max())],
                    interpretation='Bounding box of qualifying refined grid points, not a guarantee for every interior point.')
        record={**sc,'cell_m':model.cell,'cell_count':len(model.c),
                'best_local':best,'best_world':world_point(sc,best).tolist(),
                'baseline':statistics(base),'selected':statistics(selected),
                'gain_s':float(difference.mean()),'gain_ci95_s':[float(difference.mean()-1.96*se),float(difference.mean()+1.96*se)],
                'selection_candidates':se_table,
                'grid':dict(x=xs.tolist(),y=ys.tolist(),time=z.tolist(),baseline_s=float(base_train),n=1024),
                'refined_grid':dict(x=fx.tolist(),y=fy.tolist(),time=fz.tolist(),baseline_s=basetune,n=2048),
                'candidate_region':region,
                'cell_sensitivity':dict(cell_m=12.5,cell_count=len(altmodel.c),baseline=statistics(sensitivebase),selected=statistics(sensitive),
                                        paired_25m_selected=statistics(same_selected),paired_25m_baseline=statistics(same_base),
                                        paired_change_s=float(cell_delta.mean()),paired_change_ci95_s=[float(cell_delta.mean()-1.96*cell_se),float(cell_delta.mean()+1.96*cell_se)]),
                'source_samples_world':world_point(sc,test['g'][::32]).tolist()}
        results.append(record)
        with (OUT/'results.json').open('w') as f:json.dump(dict(scenarios=results,elapsed_s=time.time()-start),f,ensure_ascii=False,indent=2)
        for kind in ['grid','refined_grid']:
            g=record[kind]
            with (OUT/f'{sc["key"]}_{kind}.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['local_x_m','local_y_m','world_x_m','world_y_m','expected_time_s','gain_vs_cover_baseline_s'])
                for yy,row in zip(g['y'],g['time']):
                    for xx,tt in zip(g['x'],row):
                        ww=world_point(sc,[xx,yy]);writer.writerow([xx,yy,*ww,tt,g['baseline_s']-tt])
        print('RESULT',sc['key'],best,statistics(selected),'gain',difference.mean(),flush=True)
    print('DONE',time.time()-start,flush=True)


def check():
    for k,sc in enumerate(SCENARIOS):
        model=CoverModel(sc)
        s=sample_conditional(sc,256,17+k)
        # Each true source lies in at least one of the retained closed cells.
        belongs=np.all(np.abs(s['g'][:,None,:]-model.c[None,:,:])<=model.h+1e-8,axis=2)
        assert belongs.any(axis=1).all()
        for p in [[0,0],[400,100],[1000,-100],[1800,500],[-200,200]]:
            active,near,_,_=model.posterior_cells(np.array(p),s)
            assert ((active&belongs).any(axis=1)|near).all()
            r=model.evaluate(p,s)
            assert np.all(r['time']>=np.linalg.norm(p)/5+10-1e-8)
            assert r['clears'].min()>=1
        base=model.evaluate([0,0],s,True)
        repeated=model.evaluate([0,0],s)
        assert np.allclose(repeated['time']-base['time'],5)
        # Post-measurement near case: 5 s detection + 5 s successful clear.
        p=s['g'][0]
        tiny={key:value[:1] for key,value in s.items()}
        r=model.evaluate(p,tiny)
        assert abs(r['time'][0]-(np.linalg.norm(p)/5+10))<1e-8
    print('CHECKS PASSED: support preservation, continuous-cell cover, repeated-point error, action timing.',flush=True)


if __name__=='__main__':
    import sys
    check()
    if '--check-only' not in sys.argv:run()
