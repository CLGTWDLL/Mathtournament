"""Offline discovery-aware ordering under an explicit uniform design prior.

The geometric station set is unchanged. 4096 fixed Sobol source scenarios
estimate how many still-unknown channels require scanning at each stop. Travel
time plus expected scan service is minimized by 2-opt and one-site relocation.
This is a surrogate, not a prediction of full clearance time or a proof of
optimality. Empty channels and directional half-planes are both represented.
"""
import json
from pathlib import Path
import sys
import time
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
from scipy.stats import qmc
from q4_sparse_backbone import certified, independent_cover_check


def scenario_masks(route):
    u = qmc.Sobol(5, scramble=True, seed=20260914).random_base2(12)
    rho, angle = 1800*np.sqrt(u[:, 0]), 2*np.pi*u[:, 1]
    g = rho[:, None]*np.column_stack((np.cos(angle), np.sin(angle)))
    radius, theta, direct = 1000+500*u[:, 2], 2*np.pi*u[:, 3], u[:, 4] < .5
    direction = np.column_stack((np.cos(theta), np.sin(theta)))
    masks = []
    for q in np.vstack((np.zeros(2), route)):
        delta = q-g
        hear = (np.linalg.norm(delta, axis=1) <= radius) & (~direct | ((delta*direction).sum(axis=1) >= 0))
        masks.append(int.from_bytes(np.packbits(hear, bitorder='little').tobytes(), 'little'))
    return masks, len(g)


def objective(order, distances, masks, count):
    heard, previous, value = masks[0], 0, 0.
    for j in order:
        # Prior expected N=13 sources and 7 empty channels out of 20.
        unknown = 7+13*(1-heard.bit_count()/count)
        value += distances[previous, j]/5 + 6*unknown-1
        heard |= masks[j]
        previous = j
    return float(value)


def main():
    destination = HERE/'backbone_final.json'
    if destination.exists():
        raise ValueError('Final route already exists; select a new version for reruns')
    began = time.perf_counter()
    data = json.loads((HERE/'backbone.json').read_text(encoding='utf-8'))
    route = np.asarray(data['route'], float)
    points = np.vstack((np.zeros(2), route))
    distances = np.linalg.norm(points[:, None]-points[None], axis=2)
    masks, count = scenario_masks(route)
    order = list(range(1, len(route)+1))
    initial = best = objective(order, distances, masks, count)
    changes = 0
    for _ in range(20):
        selected, value = order, best
        for i in range(len(order)):
            for j in range(i+1, len(order)):
                trial = order[:i]+list(reversed(order[i:j+1]))+order[j+1:]
                score = objective(trial, distances, masks, count)
                if score < value-1e-8:
                    selected, value = trial, score
        for i in range(len(order)):
            rest = order[:i]+order[i+1:]
            for j in range(len(order)):
                trial = rest[:j]+[order[i]]+rest[j:]
                score = objective(trial, distances, masks, count)
                if score < value-1e-8:
                    selected, value = trial, score
        if value >= best-1e-8:
            break
        order, best = selected, value
        changes += 1
    route = points[order]
    stations = np.vstack((np.zeros(2), route))
    assert certified(stations).all() and independent_cover_check(stations)
    data.update(route=route.tolist(),
                open_route_m=float(np.linalg.norm(np.diff(stations, axis=0), axis=1).sum()),
                discovery_order=dict(initial_proxy_s=initial, final_proxy_s=best,
                                     accepted_changes=changes, scenarios=count,
                                     prior='uniform area, R uniform 1000..1500, half directional, E[N]=13',
                                     full_clearance_cost_included=False,
                                     offline_wall_s=time.perf_counter()-began))
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(data['discovery_order'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
