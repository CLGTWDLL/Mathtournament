"""Experimental sparse policy with GDOP candidates and two-stage MPC.

The frozen sparse policy remains the fallback.  This module changes only
action selection:

* bearing geometry proposes a few high-information receiver locations;
* a two-stage scenario tree chooses a contingent second action for each
  first-feedback branch, but executes only the first action and replans;
* future patrol stations with zero marginal contribution are removed only
  after the same continuous certificate still passes.

No simulated truth is available to the policy.  All scenarios come from the
feedback posterior, and the strict finite-clear and absence certificates stay
in force.

Consistency and robustness additions:

* the clearance cover is refined from 25 m to 12.5 m cells, so one 20 m
  attempt discharges the aimed cell and its four edge neighbours, which cuts
  the length of the cell-by-cell finite fallback;
* an unresolved quadrature switches to conservative uniform-over-cover
  scenarios for ranking instead of abandoning MPC;
* observation candidates never repeat an already-measured position (the
  simulator returns the identical deterministic reading there);
* every MPC decision stores a labelled second-step branch log.
"""
from __future__ import annotations

import numpy as np

from q4_route_value import TailValue, finish_costs
from q4_sparse_backbone import Policy as SparsePolicy
from q4_feedback_policy import Cover
from q4_fine_clear import FineCover
from q4_mpc_belief import AdaptivePosterior
from q4_mpc_rollout import ANGLE_BIN_DEG, Rollout
from q4_mpc_tasks import Tasks


GDOP_RADII_M = (120.0, 220.0, 340.0)
GDOP_ANGLE_STEPS = 16
FINE_HALF = 6.25


def _deduplicate(proposals, history, limit=None):
    """Drop exact duplicates, repeats at already-measured points, out-of-range.

    Returns ``(kept, skipped_repeats)``. A repeated RF reading at the same
    position is deterministic in the simulator, so such a probe carries no
    information and must never enter the rollout.
    """
    result, seen, skipped = [], set(), 0
    measured = [np.array([o.x, o.y]) for o in history
                if o.kind in ('direction', 'near', 'no_signal')]
    for action, q, reason in proposals:
        q = np.round(np.asarray(q, float), 6)
        key = (action, float(q[0]), float(q[1]))
        if key in seen:
            continue
        if action == 'measure' and any(np.linalg.norm(q-p) < .1 for p in measured):
            skipped += 1
            continue
        if np.linalg.norm(q) > 2800.0:
            continue
        seen.add(key)
        result.append((action, q, reason))
        if limit is not None and len(result) >= limit:
            break
    return result, skipped


def _bearing_fim(sources, stations):
    """Per-particle 2x2 bearing Fisher information, up to a common variance."""
    sources = np.asarray(sources, float)
    a = np.full(len(sources), 1.0/1800.0**2)
    b = np.zeros(len(sources))
    c = a.copy()
    for station in np.asarray(stations, float).reshape(-1, 2):
        delta = sources-station
        d2 = np.maximum(np.sum(delta*delta, axis=1), 25.0)
        hx, hy = -delta[:, 1]/d2, delta[:, 0]/d2
        a += hx*hx
        b += hx*hy
        c += hy*hy
    return a, b, c


def _gdop_values(base_fim, sources, q, radius, theta, directional):
    delta = sources-q
    d2 = np.maximum(np.sum(delta*delta, axis=1), 25.0)
    receive = np.sqrt(d2) <= radius
    if np.any(directional):
        outward = np.column_stack((np.cos(theta), np.sin(theta)))
        receive &= (~directional) | (np.sum((q-sources)*outward, axis=1) >= 0)
    hx, hy = -delta[:, 1]/d2, delta[:, 0]/d2
    a, b, c = (part.copy() for part in base_fim)
    a += receive*hx*hx
    b += receive*hx*hy
    c += receive*hy*hy
    determinant = np.maximum(a*c-b*b, 1e-24)
    gdop = np.sqrt(np.maximum((a+c)/determinant, 0.0))
    return gdop, receive


class Policy(Tasks, Rollout, SparsePolicy):
    """Sparse backbone plus conservative GDOP/MPC action proposals."""

    def __init__(self, variant='dispatch', particles=32, seed=9411, horizon=2,
                 clear_half=FINE_HALF, task_margin=None):
        if horizon != 2:
            raise ValueError('Only the validated two-stage horizon is supported')
        if clear_half <= 0 or clear_half > 12.5:
            raise ValueError('clear_half must lie in (0, 12.5]')
        self._cert_cache = {}
        self.skipped_channel_observations = 0
        self.clear_half = float(clear_half)
        if task_margin is not None:
            if task_margin < 0:
                raise ValueError('task_margin must be non-negative')
            self.task_accept_margin_s = float(task_margin)
        super().__init__(variant, particles, seed)
        self.horizon = horizon
        self.mpc_rng = np.random.default_rng(seed+2_700_001)
        self.check_rng = np.random.default_rng(seed+3_700_001)
        self.mpc_decision = None
        self.pruned_sites = 0
        self._prune_signature = None
        self.posterior_recoveries = 0
        self.posterior_fallbacks = 0
        self.repeated_observation_blocks = 0
        self._original_sites = {tuple(np.round(q, 6)) for q in [np.zeros(2)]+self.future}
        self.task_rng = np.random.default_rng(seed+4_700_001)
        self.task_check_rng = np.random.default_rng(seed+5_700_001)
        self._task_samples = {}

    def local_state(self, i):
        key = (i, len(self.history[i]))
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != i}
            if self.clear_half < 12.5-1e-9:
                cover = FineCover(self.history[i], self.clear_half)
            else:
                cover = Cover(self.history[i])
            post = AdaptivePosterior(cover.c, self.history[i],
                                     half=getattr(cover, 'half', 12.5))
            self.cache[key] = cover, post
            self.numeric_fallbacks += int(not post.numeric_ok)
            self.posterior_recoveries += int(post.recovered)
        return self.cache[key]

    def accept_feedback(self, result):
        pending = self.pending
        extra_clear = bool(pending and pending['action'] == 'clear'
                           and pending['reason'] == 'mpc_posterior_clear')
        super().accept_feedback(result)
        if extra_clear:
            self.free_clear[pending['channel']-1] += 1

    def emit(self, action, i, q, reason, estimated=0.):
        result = super().emit(action, i, q, reason, estimated)
        if self.mpc_decision is not None:
            result['gdop_mpc_decision'] = self.mpc_decision
            self.pending['gdop_mpc_decision'] = self.mpc_decision
            self.mpc_decision = None
        return result

    def _tail(self, current):
        anchors = []
        for j in self.known():
            if int(j) == current:
                continue
            cover, post = self.local_state(int(j))
            anchors.append(post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0))
        stations = self.future if len(self.unknown()) else []
        return TailValue(self.position, stations, anchors, self.unknown()+1)

    def _gdop_candidates(self, cover, post, history, samples):
        g, radius, theta, directional = samples
        mean = post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0)
        positive = [np.array([o.x, o.y]) for o in history
                    if o.kind in ('direction', 'near')]
        base = _bearing_fim(g, positive)
        pool = []
        for distance in GDOP_RADII_M:
            for angle in np.linspace(0.0, 2*np.pi, GDOP_ANGLE_STEPS, endpoint=False):
                q = mean+distance*np.array([np.cos(angle), np.sin(angle)])
                if np.linalg.norm(q) > 2800.0:
                    continue
                gdop, receive = _gdop_values(base, g, q, radius, theta, directional)
                # GDOP is useful only when the receiver is likely to hear the
                # source.  Travel is expressed in metres to keep the ranking
                # from selecting a distant, slightly prettier geometry.
                score = (np.mean(np.minimum(gdop, 5000.0))
                         +1200.0*(1.0-receive.mean())
                         +0.35*np.linalg.norm(q-self.position))
                pool.append((float(score), q, float(receive.mean()), float(np.mean(gdop))))
        for p in positive[-2:]:
            for f in (.35, .65):
                q = p+f*(mean-p)
                gdop, receive = _gdop_values(base, g, q, radius, theta, directional)
                score = (np.mean(np.minimum(gdop, 5000.0))
                         +1200.0*(1.0-receive.mean())
                         +0.35*np.linalg.norm(q-self.position))
                pool.append((float(score), q, float(receive.mean()), float(np.mean(gdop))))
        # Cross the support sideways: a point offset from the current wedge
        # gives a bearing that actually intersects the surviving band.
        spread = np.linalg.norm(cover.c-mean, axis=1)
        extent = float(np.quantile(spread, .9)) if len(spread) else 0.
        if extent > 1.0:
            direction = mean-self.position
            direction = direction/max(np.linalg.norm(direction), 1e-9)
            normal = np.array([-direction[1], direction[0]])
            for side in (-1.0, 1.0):
                for scale in (.6, 1.0):
                    q = mean+side*max(60., scale*extent)*normal
                    if np.linalg.norm(q) > 2800.0:
                        continue
                    gdop, receive = _gdop_values(base, g, q, radius, theta, directional)
                    score = (np.mean(np.minimum(gdop, 5000.0))
                             +1200.0*(1.0-receive.mean())
                             +0.35*np.linalg.norm(q-self.position))
                    pool.append((float(score), q, float(receive.mean()), float(np.mean(gdop))))
        chosen = []
        for score, q, receive, gdop in sorted(pool, key=lambda row: row[0]):
            if any(np.linalg.norm(q-old[0]) < 70.0 for old in chosen):
                continue
            chosen.append((q, score, receive, gdop))
            if len(chosen) == 6:
                break
        return chosen

    @staticmethod
    def _feedback_keys(q, samples, noise):
        g, radius, theta, directional = samples
        delta = g-q
        distance = np.linalg.norm(delta, axis=1)
        outward = np.column_stack((np.cos(theta), np.sin(theta)))
        receive = (distance <= radius) & ((~directional) |
                  (np.sum((q-g)*outward, axis=1) >= 0))
        near = receive & (distance <= 5.0)
        angle = np.round(np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))+noise, 2) % 360.0
        keys = np.floor(angle/ANGLE_BIN_DEG).astype(int)
        keys[~receive] = -2
        keys[near] = -1
        return keys, near

    def _terminal(self, cover, q, active, sources, tail, channel):
        cost, endpoints = finish_costs(cover, q, active, sources)
        return cost+tail(endpoints, channel)


    def local_action(self, i):
        baseline = super().local_action(i)
        self.mpc_decision = None
        if baseline[3] in ('near_clear', 'certified_clear') or self.local_rf[i] >= 5 or self.steps >= 1200:
            return baseline
        cover, post = self.local_state(i)
        if not post.numeric_ok:
            # Quadrature is unresolved. Fall back to conservative
            # uniform-over-cover scenarios for ranking instead of dropping
            # straight to cell-by-cell finite clearing.
            self.posterior_fallbacks += 1
        self._mpc_channel = int(i)+1
        train = post.scenarios(self.mpc_rng, self.particles)
        noise1 = self.mpc_rng.uniform(-1.0, 1.0, self.particles)
        noise2 = self.mpc_rng.uniform(-1.0, 1.0, self.particles)
        mean = post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0)
        nearest = cover.c[np.argmin(np.linalg.norm(cover.c-self.position, axis=1))]
        gdop = self._gdop_candidates(cover, post, self.history[i], train)
        proposals = [(baseline[0], baseline[1], baseline[3]),
                     ('clear', nearest, 'mpc_finite_clear')]
        if self.free_clear[i] < 2:
            proposals.append(('clear', mean, 'mpc_posterior_clear'))
        proposals.extend(('measure', row[0], 'gdop_localize') for row in gdop)
        if self.future:
            proposals.append(('measure', self.future[0], 'mpc_patrol_measure'))
        proposals, skipped = _deduplicate(proposals, self.history[i])
        self.repeated_observation_blocks += skipped
        first = proposals  # Keep the patrol candidate even when GDOP fills the pool.
        second, skipped = _deduplicate([
            ('clear', nearest, 'mpc_finite_clear'),
            *[p for p in proposals if p[0] == 'measure'],
        ], self.history[i], limit=7)
        self.repeated_observation_blocks += skipped
        if not first or not second:
            return baseline
        tail = self._tail(i)
        trained = [self._train_plan(p, second, cover, train, noise1, noise2, tail)
                   for p in first]
        means = [scores.mean() for _, scores in trained]
        chosen = int(np.argmin(means))
        realized = None
        if self.history[i]:
            realized = self._observation_branch(self.history[i][-1])
        decision = dict(horizon=self.horizon, candidate_count=len(first),
                        second_action_count=len(second), selected_index=chosen,
                        selected_reason=first[chosen][2], train_gain_s=float(means[0]-means[chosen]),
                        selected_second_branches=self._plan_log(trained[chosen][0]),
                        baseline_second_branches=self._plan_log(trained[0][0]),
                        posterior_recovered=post.recovered,
                        posterior_numeric=post.numeric_ok,
                        clear_half=getattr(cover, 'half', 12.5),
                        previous_feedback_branch=realized,
                        previous_feedback_label=self.branch_label(realized),
                        skipped_repeat_observations=skipped,
                        quadrature_level=post.quadrature_level,
                        gdop_candidates=[dict(position=row[0].tolist(), rank_score=row[1],
                                              receive_probability=row[2], expected_gdop=row[3])
                                         for row in gdop], accepted=False)
        if chosen:
            count = max(48, 2*self.particles)
            check = post.scenarios(self.check_rng, count)
            check_noise1 = self.check_rng.uniform(-1.0, 1.0, count)
            check_noise2 = self.check_rng.uniform(-1.0, 1.0, count)
            old = self._apply_plan(first[0], trained[0][0], second, cover, check,
                                   check_noise1, check_noise2, tail)
            new = self._apply_plan(first[chosen], trained[chosen][0], second, cover, check,
                                   check_noise1, check_noise2, tail)
            gain = old-new
            se = gain.std(ddof=1)/np.sqrt(count)
            accepted = bool(gain.mean()-2.0*se > 1.0)
            decision.update(validation_count=count, validation_gain_s=float(gain.mean()),
                            validation_se_s=float(se), accepted=accepted)
            if accepted:
                action, q, reason = first[chosen]
                self.mpc_decision = decision
                return action, q, float(new.mean()), reason
        self.mpc_decision = decision
        return baseline
