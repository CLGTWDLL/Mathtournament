"""Offline second-point optimization against a complete adaptive continuation.

The finite policy class uses posterior one-step decisions at subsequent stops.
The second point is evaluated through the whole executed trajectory, for each
budget 1..5 extra measurements, including all later observation branches.
This is rollout policy optimization, not an exact global POMDP solution.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from solve_q2_core import CoverModel, SCENARIOS, OUT, sample_conditional, world_point
from solve_q2_multistep import choose_measurement, observe
from q2_analytic import RayPrior

JOINT = OUT/'joint'
WEIGHTS = [.25, .5, 1., 2., 4.]


def episode(model, second, g, radius, noise, index, nplan, horizon=5, timing=False, noise_kind='uniform'):
    rng = np.random.default_rng(1740000+index)
    sensor = np.random.default_rng(1840000+index)
    active = np.ones(len(model.c), bool)
    pos = np.array(second, float)
    if np.linalg.norm(pos) < 1e-8:
        raise ValueError('Second point must differ from first')
    history = [(np.zeros(2), 0.)]
    active, response = observe(model, pos, g, radius, noise, active)
    length = np.linalg.norm(pos)
    spent = length/5+5
    measures, planner_s = 1, 0.
    rows = []
    for stage in range(horizon):
        if response == 'near':
            c, count, tail = np.array([5.]), np.array([1]), np.array([0.])
        else:
            c, count, tail = model.clear_route(pos, active[None, :], g[None, :])
        rows.append([spent+float(c[0]), measures, int(count[0]), length+float(tail[0]), planner_s])
        assert abs(rows[-1][0]-(rows[-1][3]/5+5*measures+3*rows[-1][2]+2)) < 1e-7
        if stage == horizon-1:
            break
        if response == 'near':
            rows.extend([rows[-1]]*(horizon-stage-1)); break
        history.append((pos.copy(), response))
        start = time.perf_counter()
        candidate = choose_measurement(model, active, pos, history, nplan, rng)
        planner_s += time.perf_counter()-start
        if candidate is None:
            # Include the computation needed to decide stopping.
            stopped = rows[-1].copy(); stopped[4] = planner_s
            rows.extend([stopped]*(horizon-stage-1)); break
        distance = np.linalg.norm(candidate-pos)
        spent += distance/5+5; length += distance
        pos = candidate; measures += 1
        new_noise = sensor.triangular(-1, 0, 1) if noise_kind == 'triangular' else sensor.uniform(-1, 1)
        active, response = observe(model, pos, g, radius, new_noise, active)
    return np.array(rows)


def evaluate(model, point, samples, seed, nplan, horizon=5, noise_kind='uniform'):
    return np.array([episode(model, point, g, radius, noise, seed+i, nplan, horizon, noise_kind=noise_kind)
                     for i, (g, radius, noise) in enumerate(zip(samples['g'], samples['radius'], samples['noise']))])


def stats(x):
    t = x[:, 0]
    se = float(t.std(ddof=1)/np.sqrt(len(t)))
    return dict(mean_s=float(t.mean()), ci95_s=[float(t.mean()-1.96*se), float(t.mean()+1.96*se)],
                p90_s=float(np.quantile(t, .9)), mean_measurements_after_first=float(x[:, 1].mean()),
                mean_clears=float(x[:, 2].mean()), p_one_clear=float((x[:, 2] == 1).mean()),
                mean_path_m=float(x[:, 3].mean()), planning_ms_mean=float(x[:, 4].mean()*1000),
                planning_ms_p95=float(np.quantile(x[:, 4], .95)*1000))


def paired(a, b):
    d = a-b; se = d.std(ddof=1)/np.sqrt(len(d))
    return dict(mean_s=float(d.mean()), ci95_s=[float(d.mean()-1.96*se), float(d.mean()+1.96*se)])


def optimize(sc, k, nplan=96):
    destination = JOINT/sc['key']; destination.mkdir(parents=True, exist_ok=True)
    model = CoverModel(sc)
    ray = RayPrior(sc)
    old = next(s for s in json.loads((OUT/'results.json').read_text())['scenarios'] if s['key'] == sc['key'])
    L = ray.limit
    # Broad initial window, followed by refinement around different horizons.
    xs = np.linspace(-.2*L, 1.2*L, 9)
    ys = np.linspace(-.6*L, .6*L, 9)
    points = [np.array([x, y]) for y in ys for x in xs if np.hypot(x, y) > 1e-8]
    analytical = []
    for weight in WEIGHTS:
        point, proxy, info = ray.select(weight)
        for sign in [1, -1]:
            q = point*np.array([1, sign])
            analytical.append(dict(weight=weight, sign=sign, point=q.tolist(), proxy_s=proxy, solver=info))
            points.append(q)
    points.append(np.array(old['best_local']))
    records = []
    train = sample_conditional(sc, 64, 31100+k)
    start = time.time()
    for i, q in enumerate(points):
        values = evaluate(model, q, train, 510000+k*10000, nplan)
        records.append(dict(point=q.tolist(), means=values[:, :, 0].mean(axis=0).tolist(), phase='coarse'))
        if (i+1) % 10 == 0:
            print(sc['key'], 'coarse', i+1, '/', len(points), 's', round(time.time()-start, 1), flush=True)
    (destination/'coarse.json').write_text(json.dumps(records))
    initial = np.array([r['means'] for r in records])
    centers = []
    for h in [0, 1, 2, 4]:
        q = np.array(records[np.argmin(initial[:, h])]['point'])
        if all(np.linalg.norm(q-c) > .05*L for c in centers): centers.append(q)
    refined = []
    for center in centers:
        for dx in np.linspace(-.12*L, .12*L, 5):
            for dy in np.linspace(-.09*L, .09*L, 5):
                q = center+np.array([dx, dy])
                if np.linalg.norm(q) > 1 and all(np.linalg.norm(q-p) > 1 for p in refined): refined.append(q)
    tune = sample_conditional(sc, 128, 32100+k)
    for i, q in enumerate(refined):
        values = evaluate(model, q, tune, 520000+k*10000, nplan)
        records.append(dict(point=q.tolist(), means=values[:, :, 0].mean(axis=0).tolist(), phase='refine'))
        if (i+1) % 10 == 0:
            print(sc['key'], 'refine', i+1, '/', len(refined), 's', round(time.time()-start, 1), flush=True)
    (destination/'maps.json').write_text(json.dumps(records))
    # Independent selection includes analytical candidates, old point, and top
    # points for every horizon from each search phase; never use test outcomes.
    finalists = [np.array(a['point']) for a in analytical]+[np.array(old['best_local'])]
    for phase in ['coarse', 'refine']:
        group = [r for r in records if r['phase'] == phase]
        vals = np.array([r['means'] for r in group])
        for h in [0, 1, 2, 4]:
            for index in np.argsort(vals[:, h])[:2]:
                q = np.array(group[index]['point'])
                if all(np.linalg.norm(q-p) > 1 for p in finalists): finalists.append(q)
    select = sample_conditional(sc, 384, 33100+k)
    scores = []
    for i, q in enumerate(finalists):
        values = evaluate(model, q, select, 530000+k*10000, nplan)
        scores.append(values[:, :, 0].mean(axis=0))
        print(sc['key'], 'select', i+1, '/', len(finalists), flush=True)
    scores = np.array(scores)
    winners = scores.argmin(axis=0)
    # Semi-analytic policy: tune a single uncertainty-cost weight AND mirror
    # sign on the selection sample; retain untuned weight=1 for error auditing.
    analytic_winners = scores[:len(analytical)].argmin(axis=0)
    result = dict(scenario=sc, old_point=old['best_local'], nplan=nplan, ray_limit=L,
                  coarse_x=xs.tolist(), coarse_y=ys.tolist(), analytical=analytical,
                  finalists=[p.tolist() for p in finalists], selection_scores=scores.tolist(),
                  selected_points=[finalists[i].tolist() for i in winners],
                  analytic_selected=[analytical[i] for i in analytic_winners],
                  offline_optimization_s=time.time()-start)
    (destination/'selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def test(sc, k, selected, ntest=2048, nplan=192):
    destination = JOINT/sc['key']
    model = CoverModel(sc)
    samples = sample_conditional(sc, ntest, 34100+k)
    points = [np.array(selected['old_point'])]
    points += [np.array(p) for p in selected['selected_points']]
    points += [np.array(a['point']) for a in selected['analytic_selected']]
    points += [np.array(a['point']) for a in selected['analytical'] if a['weight'] == 1. and a['sign'] == 1]
    unique = []
    for q in points:
        if all(np.linalg.norm(q-p) > 1e-6 for p in unique): unique.append(q)
    cache = {}
    for i, q in enumerate(unique):
        print(sc['key'], 'test', i+1, '/', len(unique), q.tolist(), flush=True)
        values = evaluate(model, q, samples, 540000+k*10000, nplan)
        key = tuple(q); cache[key] = values
        np.save(destination/f'test_{i}.npy', values)
    get = lambda p: cache[tuple(p)]
    old = get(selected['old_point'])
    horizons = []
    for h in range(5):
        q = selected['selected_points'][h]
        a = selected['analytic_selected'][h]
        ref, approx = get(q)[:, h], get(a['point'])[:, h]
        untuned_point = next(a['point'] for a in selected['analytical'] if a['weight'] == 1 and a['sign'] == 1)
        untuned = get(untuned_point)[:, h]
        horizons.append(dict(max_total_measurements=h+2, point_local=q, point_world=world_point(sc, q).tolist(),
                             reference=stats(ref), old_point=stats(old[:, h]),
                             gain_vs_fixed_second=paired(old[:, h, 0], ref[:, 0]),
                             analytic=a, analytic_stats=stats(approx),
                             analytic_regret=paired(approx[:, 0], ref[:, 0]),
                             analytic_point_distance_m=float(np.linalg.norm(np.array(q)-a['point'])),
                             untuned_stats=stats(untuned), untuned_regret=paired(untuned[:, 0], ref[:, 0])))
    baseline = model.evaluate([0, 0], samples, True)
    # Clairvoyant travel lower bound, valid for any policy after the first
    # observation: reach some point within 20m and perform one successful clear.
    lb = np.maximum(np.linalg.norm(samples['g'], axis=1)-20, 0)/5+5
    result = dict(n=ntest, nplan=nplan, horizons=horizons, baseline_s=float(baseline['time'].mean()),
                  oracle_lower_bound_s=float(lb.mean()), test_points=[q.tolist() for q in unique])
    (destination/'test.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['optimize', 'test', 'all'], default='all')
    parser.add_argument('--scenario', default='all')
    parser.add_argument('--test-n', type=int, default=2048)
    args = parser.parse_args()
    JOINT.mkdir(exist_ok=True)
    for k, sc in enumerate(SCENARIOS):
        if args.scenario != 'all' and args.scenario != sc['key']: continue
        if args.phase in ['all', 'optimize']: selection = optimize(sc, k)
        else: selection = json.loads((JOINT/sc['key']/'selection.json').read_text())
        if args.phase in ['all', 'test']: test(sc, k, selection, args.test_n)
