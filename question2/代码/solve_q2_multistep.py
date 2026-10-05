"""Adaptive extra measurements, with an independent posterior planning sample.

Keeps the previously selected second point fixed. A finite-candidate one-step
lookahead is repeated at most twice, with stopping allowed at each step.
This is a policy evaluation, not a globally optimal Bellman solution.
"""
import argparse
import json
import time
import numpy as np
from solve_q2_core import CoverModel, SCENARIOS, OUT, RAD, local_origin, sample_conditional

SAMPLER_DIAGNOSTICS = dict(polygon_proposals=0, safe_stops=0)


def polygon_proposal(model, active, history):
    centers = model.c[active]
    lo, hi = centers.min(axis=0)-model.h, centers.max(axis=0)+model.h
    polygon = np.array([[lo[0],lo[1]], [hi[0],lo[1]], [hi[0],hi[1]], [lo[0],hi[1]]])
    for pos, response in history:
        if response is None:
            continue
        for angle, sign in [(response-1.0, 1), (response+1.0, -1)]:
            angle *= RAD
            normal = sign*np.array([-np.sin(angle), np.cos(angle)])
            values = (polygon-pos)@normal+1e-9
            output = []
            for j in range(len(polygon)):
                prev, current = polygon[j-1], polygon[j]
                a, b = values[j-1], values[j]
                if (a >= 0) != (b >= 0):
                    output.append(prev+(current-prev)*(a/(a-b)))
                if b >= 0:
                    output.append(current)
            polygon = np.array(output)
            if len(polygon) < 3:
                return None
    a = polygon[0]
    triangles = np.stack([np.repeat(a[None,:],len(polygon)-2,axis=0),polygon[1:-1],polygon[2:]],axis=1)
    u, v = triangles[:,1]-a, triangles[:,2]-a
    areas = np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2
    if areas.sum() <= 1e-20:
        return None
    return triangles, areas/areas.sum()


def angle_likelihood(g, pos, bearing):
    angle = np.degrees(np.arctan2(g[:, 1]-pos[1], g[:, 0]-pos[0]))
    error = (bearing-angle+180) % 360-180
    return (np.abs(error) <= 1.0).astype(float)


def posterior_sample(model, active, history, n, rng):
    """Exact rejection from uniform area cells and the joint (g,R) prior.

    History contains distinct-position direction/no_signal observations. R is
    fixed across all observations. Integrating R gives an interval length;
    independent angular errors give a product of rounded-bin likelihoods.
    """
    centers = model.c[active]
    found, points, radii, proposals = 0, [], [], 0
    triangles = None
    while found < n:
        if proposals >= 131072 and triangles is None:
            proposal = polygon_proposal(model, active, history)
            if proposal is None:
                SAMPLER_DIAGNOSTICS['safe_stops'] += 1
                return None
            triangles, probabilities = proposal
            SAMPLER_DIAGNOSTICS['polygon_proposals'] += 1
        batch = min(131072, max(4096, (n-found)*128))
        if triangles is None:
            g = centers[rng.integers(len(centers), size=batch)] + rng.uniform(-model.h, model.h, (batch, 2))
        else:
            selected = triangles[rng.choice(len(triangles), size=batch, p=probabilities)]
            uv = rng.random((batch,2)); root = np.sqrt(uv[:,0])
            g = ((1-root)[:,None]*selected[:,0]+(root*(1-uv[:,1]))[:,None]*selected[:,1]
                 +(root*uv[:,1])[:,None]*selected[:,2])
        lo, hi = np.full(batch, 1000.), np.full(batch, 1500.)
        likelihood = (np.linalg.norm(g-local_origin(model.sc), axis=1) <= 1800).astype(float)
        for pos, response in history:
            d = np.linalg.norm(g-pos, axis=1)
            if response is None:
                hi = np.minimum(hi, d)
            else:
                lo = np.maximum(lo, d)
                likelihood *= (d > 5)*angle_likelihood(g, pos, response)
        likelihood *= np.maximum(hi-lo, 0)/500
        ok = rng.random(batch) < likelihood
        count = int(ok.sum())
        if count:
            points.append(g[ok])
            radii.append(rng.uniform(lo[ok], hi[ok]))
            found += count
        proposals += batch
        if proposals > 2000000:
            # The decision policy stops measuring rather than introducing a
            # biased posterior approximation. Continuous cover still succeeds.
            SAMPLER_DIAGNOSTICS['safe_stops'] += 1
            return None
    return dict(g=np.concatenate(points)[:n], radius=np.concatenate(radii)[:n], noise=rng.uniform(-1, 1, n))


def candidate_points(pos, samples, history):
    g = samples['g']
    mean = g.mean(axis=0)
    # Candidate anchors depend exclusively on posterior samples.
    anchors = [mean]
    q = np.quantile(g[:, 0], [.25, .75])
    anchors.extend([g[g[:, 0] <= q[0]].mean(axis=0), g[g[:, 0] >= q[1]].mean(axis=0)])
    direction = mean-pos
    direction /= max(np.linalg.norm(direction), 1e-12)
    normal = np.array([-direction[1], direction[0]])
    candidates = [pos + a*direction + b*normal for a in [0., 25.] for b in [-25., 25.]]
    for anchor in anchors:
        candidates.extend(anchor + off*normal for off in [-25., 0., 25.])
    return [p for p in candidates if all(np.linalg.norm(p-hp) > .1 for hp, _ in history)]


def choose_measurement(model, active, pos, history, nplan, rng):
    samples = posterior_sample(model, active, history, nplan, rng)
    if samples is None:
        return None
    prior = np.repeat(active[None, :], nplan, axis=0)
    stop = model.clear_route(pos, prior, samples['g'])[0]
    best_gain, best_pos = 0., None
    for candidate in candidate_points(pos, samples, history):
        mask, near, _, _ = model.posterior_cells(candidate, samples)
        mask &= prior
        costs = model.clear_route(candidate, mask, samples['g'])[0]
        costs += np.linalg.norm(candidate-pos)/5 + 5 + 5*near
        gain = stop-costs
        # Conservative numerical stop rule; this is not a global confidence
        # guarantee after comparing multiple candidates.
        adjusted_gain = gain.mean()-2*gain.std(ddof=1)/np.sqrt(nplan)-1.
        if adjusted_gain > best_gain:
            best_gain, best_pos = adjusted_gain, candidate
    return best_pos


def observe(model, pos, g, radius, noise, active):
    sample = dict(g=g[None, :], radius=np.array([radius]), noise=np.array([noise]))
    mask, near, _, no = model.posterior_cells(pos, sample)
    mask = mask[0] & active
    if near[0]:
        return mask, 'near'
    if no[0]:
        return mask, None
    bearing = float(np.degrees(np.arctan2(*(g-pos)[::-1])) + noise)
    # Every surviving true source must lie in a retained cell.
    contains = np.max(np.abs(model.c-g), axis=1) <= model.h+1e-8
    assert np.any(mask & contains)
    return mask, bearing


def evaluate_episode(model, second, g, radius, first_noise, index, nplan):
    rng = np.random.default_rng(704000+index)
    sensor_rng = np.random.default_rng(804000+index)
    active = np.ones(len(model.c), bool)
    pos = np.array(second)
    history = [(np.zeros(2), 0.)]
    active, response = observe(model, pos, g, radius, first_noise, active)
    spent = np.linalg.norm(pos)/5+5
    measures = 1
    rows = []
    for budget in range(3):
        if response == 'near':
            final_cost, clears = spent+5, 1
        else:
            c, count, _ = model.clear_route(pos, active[None, :], g[None, :])
            final_cost, clears = spent+float(c[0]), int(count[0])
        rows.append([final_cost, measures, clears])
        if budget == 2:
            break
        if response == 'near':
            rows.extend([rows[-1]]*(2-budget))
            break
        history.append((pos.copy(), response))
        candidate = choose_measurement(model, active, pos, history, nplan, rng)
        if candidate is None:
            rows.extend([rows[-1]]*(2-budget))
            break
        spent += np.linalg.norm(candidate-pos)/5+5
        pos = candidate
        measures += 1
        active, response = observe(model, pos, g, radius, sensor_rng.uniform(-1, 1), active)
    return rows


def summary(values):
    t = values[:, 0]
    se = t.std(ddof=1)/np.sqrt(len(t))
    return dict(mean_s=float(t.mean()), ci95_s=[float(t.mean()-1.96*se), float(t.mean()+1.96*se)],
                p90_s=float(np.quantile(t, .9)), mean_extra_measurements=float(values[:, 1].mean()),
                mean_clears=float(values[:, 2].mean()), p_one_clear=float((values[:, 2] == 1).mean()))


def run(n, nplan):
    previous = json.loads((OUT/'results.json').read_text())
    if isinstance(previous, dict):
        previous = previous['scenarios']
    output = dict(n=n, planning_particles=nplan, cell_m=25, method='Fixed second point; repeated finite-candidate one-step lookahead; optional stopping', scenarios=[])
    start = time.time()
    for k, sc in enumerate(SCENARIOS):
        model = CoverModel(sc)
        old = next(r for r in previous if r['key'] == sc['key'])
        second = np.array(old['best_local'])
        sample = sample_conditional(sc, n, 950100+k)
        episodes = []
        for i in range(n):
            episodes.append(evaluate_episode(model, second, sample['g'][i], sample['radius'][i], sample['noise'][i], k*100000+i, nplan))
            if (i+1) % 64 == 0:
                print(sc['key'], i+1, 'elapsed', round(time.time()-start, 1), flush=True)
        values = np.array(episodes)
        # Exact pairing with the established one-additional-measurement code.
        baseline = model.evaluate(second, sample)['time']
        assert np.allclose(baseline, values[:, 0, 0], atol=1e-8)
        stats = [summary(values[:, j, :]) for j in range(3)]
        gains = []
        for j in [1, 2]:
            d = values[:, 0, 0]-values[:, j, 0]
            se = d.std(ddof=1)/np.sqrt(n)
            gains.append(dict(mean_s=float(d.mean()), ci95_s=[float(d.mean()-1.96*se), float(d.mean()+1.96*se)]))
        output['scenarios'].append(dict(key=sc['key'], name=sc['name'], second_local=second.tolist(), stats=stats, gains_vs_one_extra=gains))
        np.save(OUT/(sc['key']+'_multistep_samples.npy'), values)
        print(json.dumps(output['scenarios'][-1], ensure_ascii=False), flush=True)
        (OUT/'multistep.json').write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print('DONE', round(time.time()-start, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=512)
    parser.add_argument('--planning', type=int, default=192)
    args = parser.parse_args()
    run(args.n, args.planning)
