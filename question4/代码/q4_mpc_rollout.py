"""Feedback-conditioned, bounded two-action rollout for the experiment.

The rollout must predict exactly what the simulator-facing policy would see:

* RF feedback at a repeated position is deterministic, so a second observation
  at an already-measured location carries zero information and is never a
  branch candidate;
* a ``clear`` already attempted at a position is never proposed again, because
  the public failure has already excluded the 20 m disk around it.
"""
import numpy as np

from q3_belief import Observation
from q4_feedback_geometry import mixed_cells

ANGLE_BIN_DEG = 30.0


class Rollout:
    def _observations(self, first, samples, noise):
        action, q, _ = first
        if action == 'clear':
            success = np.linalg.norm(samples[0]-q, axis=1) <= 20
            return [Observation('success' if yes else 'no_target_in_range', *q) for yes in success]
        keys, _ = self._feedback_keys(q, samples, noise)
        delta = samples[0]-q
        angles = np.round(np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))+noise, 2) % 360
        return [Observation('near' if key == -1 else 'no_signal' if key == -2 else 'direction',
                            *q, float(angle) if key >= 0 else None)
                for key, angle in zip(keys, angles)]

    def _first_state(self, first, cover, samples, noise, tail):
        action, q, _ = first
        count = len(samples[0])
        obs = self._observations(first, samples, noise)
        if action == 'clear':
            done = np.array([o.kind == 'success' for o in obs])
            active = np.repeat(cover.remaining_after_clear(q)[None], count, axis=0)
            active[done] = False
            immediate = np.linalg.norm(q-self.position)/5+3+2*done
            keys = np.full(count, -3)
            channel = self.channel  # Optical clear does not switch the RF channel.
        else:
            active, done = cover.measurement_masks(q, *samples, noise)
            keys, _ = self._feedback_keys(q, samples, noise)
            immediate = np.linalg.norm(q-self.position)/5+5+int(self.channel != self._mpc_channel)+5*done
            channel = self._mpc_channel
        result = np.zeros(count)
        result[done] = immediate[done]+tail(np.repeat(q[None], done.sum(), axis=0), channel)
        return active, keys, done, immediate, result, obs, channel

    def _branch_candidates(self, first, second, cover, active, samples):
        action, q1, reason = first
        half = float(getattr(cover, 'half', 12.5))
        union = active.any(axis=0)
        cells = cover.c[union] if union.any() else cover.c
        nearest = cells[np.argmin(np.linalg.norm(cells-q1, axis=1))]
        mean = samples[0].mean(axis=0)
        pool = [('clear', nearest, 'mpc_finite_clear')]
        spent = int(reason in ('posterior_clear', 'mpc_posterior_clear'))
        if self.free_clear[self._mpc_channel-1]+spent < 2:
            pool.append(('clear', mean, 'mpc_posterior_clear'))
        may_measure = self.local_rf[self._mpc_channel-1]+int(action == 'measure') < 5
        if may_measure:
            direction = mean-q1
            normal = np.array([-direction[1], direction[0]])/max(np.linalg.norm(direction), 1e-9)
            pool.extend(('measure', q1+.65*direction+s*normal, 'mpc_branch_localize') for s in (-60., 60.))
            pool.extend(p for p in second if p[0] == 'measure')
        # Same-location RF errors are fixed in the simulator, so a second
        # measurement there is pure waste. A clear that already failed at a
        # position cannot succeed there on a retry either.
        result, seen = [], set()
        measured = [np.array([o.x, o.y]) for o in cover.history if o.kind in ('direction', 'near', 'no_signal')]
        failed_clears = [np.array([o.x, o.y]) for o in cover.history if o.kind == 'no_target_in_range']
        for kind, q, why in pool:
            q = np.round(q, 6)
            key = (kind, *q)
            if key in seen or np.linalg.norm(q) > 2800:
                continue
            if kind == 'measure' and (np.linalg.norm(q-q1) < .1 or any(np.linalg.norm(q-p) < .1 for p in measured)):
                continue
            if kind == 'clear' and (np.linalg.norm(q-q1) < .1
                                    or any(np.linalg.norm(q-p) < .1 for p in failed_clears)):
                continue
            seen.add(key)
            result.append((kind, q, why))
            if len(result) == 6:
                break
        return result

    def _second_scores(self, cover, q1, active1, samples, noise2, tail, proposal, channel, observations):
        action, q2, _ = proposal
        g = samples[0]
        travel = np.linalg.norm(q2-q1)/5
        half = float(getattr(cover, 'half', 12.5))
        if action == 'clear':
            success = np.linalg.norm(g-q2, axis=1) <= 20
            active = active1 & cover.remaining_after_clear(q2)[None]
            active[success] = False
            return travel+3+2*success+self._terminal(cover, q2, active, g, tail, channel)
        masks, near = cover.measurement_masks(q2, *samples, noise2)
        second_obs = self._observations(proposal, samples, noise2)
        # The real belief uses BOTH feedbacks, including coupled fixed
        # radius/orientation constraints. Do not evaluate step 2 on stale h.
        for j, (o1, o2) in enumerate(zip(observations, second_obs)):
            if near[j]:
                continue
            ids = np.flatnonzero(masks[j])
            if len(ids):
                # Restrict the joint radius/orientation predicate to the cells
                # that survived the second, single-observation mask; the
                # predicate is a necessary condition, so this is equivalent to
                # intersecting with the full-cover mask but much cheaper.
                masks[j][ids] = mixed_cells(cover.c[ids], half, cover.history+(o1, o2))
        active = active1 & masks
        return (travel+5+int(channel != self._mpc_channel)+5*near
                +self._terminal(cover, q2, active, g, tail, self._mpc_channel))

    def _train_plan(self, first, second, cover, samples, noise1, noise2, tail):
        active, keys, done, immediate, result, obs, channel = self._first_state(first, cover, samples, noise1, tail)
        q1 = first[1]
        # A no-second-action finite fallback also covers unseen validation
        # branches, including the all-training-scenarios-near corner case.
        plan = {'branches': {}, 'counts': {}, 'default': None,
                'default_reason': 'terminal_fallback'}
        for key in np.unique(keys[~done]):
            ids = np.flatnonzero((keys == key) & ~done)
            sub = tuple(v[ids] for v in samples)
            candidates = self._branch_candidates(first, second, cover, active[ids], sub)
            costs = [self._terminal(cover, q1, active[ids], sub[0], tail, channel)]
            costs += [self._second_scores(cover, q1, active[ids], sub, noise2[ids], tail,
                                         p, channel, [obs[j] for j in ids]) for p in candidates]
            choice = int(np.argmin([c.mean() for c in costs]))
            plan['branches'][int(key)] = candidates[choice-1] if choice else None
            plan['counts'][int(key)] = len(ids)
            result[ids] = immediate[ids]+costs[choice]
        return plan, result

    def _apply_plan(self, first, plan, second, cover, samples, noise1, noise2, tail):
        active, keys, done, immediate, result, obs, channel = self._first_state(first, cover, samples, noise1, tail)
        q1 = first[1]
        for key in np.unique(keys[~done]):
            ids = np.flatnonzero((keys == key) & ~done)
            sub = tuple(v[ids] for v in samples)
            proposal = plan['branches'].get(int(key), plan['default'])
            if proposal is None:
                cost = self._terminal(cover, q1, active[ids], sub[0], tail, channel)
            else:
                cost = self._second_scores(cover, q1, active[ids], sub, noise2[ids], tail,
                                           proposal, channel, [obs[j] for j in ids])
            result[ids] = immediate[ids]+cost
        return result

    @staticmethod
    def branch_label(key, bin_deg=ANGLE_BIN_DEG):
        """Human-readable meaning of a first-step feedback branch key."""
        if key is None:
            return 'default'
        key = int(key)
        if key == -1:
            return 'near'
        if key == -2:
            return 'no_signal'
        if key == -3:
            return 'clear'
        lo = (key*bin_deg) % 360.0
        return f'direction_{lo:.0f}_{(lo+bin_deg) % 360:.0f}'

    @staticmethod
    def _observation_branch(observation, bin_deg=ANGLE_BIN_DEG):
        """Branch key of an actually observed first-step feedback."""
        if observation.kind == 'near':
            return -1
        if observation.kind == 'no_signal':
            return -2
        if observation.kind == 'direction':
            return int((float(observation.angle) % 360.0)//bin_deg)
        return -3

    @classmethod
    def _plan_log(cls, plan, bin_deg=ANGLE_BIN_DEG):
        """Per-first-feedback branch log for the contingent second action."""
        rows = [dict(feedback_branch=int(key), feedback_label=cls.branch_label(key, bin_deg),
                     training_count=int(plan['counts'][key]),
                     second_action=p[0] if p else 'terminal_fallback',
                     second_position=p[1].tolist() if p else None,
                     second_reason=p[2] if p else plan.get('default_reason', 'terminal_fallback'))
                for key, p in sorted(plan['branches'].items())]
        default = plan.get('default')
        rows.append(dict(feedback_branch=None, feedback_label='default', training_count=0,
                         second_action=default[0] if default else 'terminal_fallback',
                         second_position=default[1].tolist() if default else None,
                         second_reason=default[2] if default else plan.get('default_reason', 'terminal_fallback')))
        return rows
