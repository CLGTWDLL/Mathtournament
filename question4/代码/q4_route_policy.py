"""Third-stage paired scoring: baseline action + separate validation samples."""
import numpy as np
from q4_feedback_policy import Policy as GeometryPolicy
from q4_route_value import TailValue, finish_costs


class Policy(GeometryPolicy):
    def __init__(self, variant='route', particles=32, seed=9411):
        if variant not in ('validated', 'route'):
            raise ValueError(variant)
        super().__init__('geometry', particles, seed)
        self.route_variant = variant
        self.score_rng = np.random.default_rng(seed+900001)
        self.check_rng = np.random.default_rng(seed+1900001)
        self.last_decision = None

    def scores(self, i, proposal, cover, samples, noise, tail):
        action, q, _, _ = proposal
        g, radius, theta, directional = samples
        travel = np.linalg.norm(q-self.position)/5
        if action == 'clear':
            success = np.linalg.norm(g-q, axis=1) <= 20
            mask = np.repeat(cover.remaining_after_clear(q)[None], len(g), axis=0)
            mask[success] = False
            cost, endpoints = finish_costs(cover, q, mask, g)
            cost += travel+3+2*success
            channel = self.channel
        else:
            mask, near = cover.measurement_masks(q, g, radius, theta, directional, noise)
            cost, endpoints = finish_costs(cover, q, mask, g)
            cost += travel+5+(i+1 != self.channel)+5*near
            channel = i+1
        if tail is not None:
            cost += tail(endpoints, channel)
        return cost

    def local_action(self, i):
        # This call consumes exactly the old policy RNG stream. Separate RNGs
        # below avoid changing old decisions merely by evaluating extra plans.
        baseline = super().local_action(i)
        self.last_decision = None
        cover, post = self.local_state(i)
        if not post.numeric_ok or baseline[3] in ('near_clear', 'certified_clear') or self.local_rf[i] >= 5 or self.steps >= 1200:
            return baseline
        mean = post.weights@post.points
        tail = None
        if self.route_variant == 'route':
            anchors = []
            for j in self.known():
                if j == i:
                    continue
                c, p = self.local_state(int(j))
                anchors.append(p.weights@p.points if p.numeric_ok else c.c.mean(axis=0))
            stations = self.future if len(self.unknown()) else []
            tail = TailValue(self.position, stations, anchors, self.unknown()+1)
        candidates = [baseline]
        nearest = cover.c[np.argmin(np.linalg.norm(cover.c-self.position, axis=1))]
        candidates.append(('clear', nearest, 0., 'finite_cover'))
        if self.free_clear[i] < 2:
            candidates.append(('clear', mean, 0., 'posterior_clear'))
        direction = mean-self.position
        direction /= max(np.linalg.norm(direction), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        for f in (.35, .65, 1.):
            for b in (-60., 60.):
                candidates.append(('measure', self.position+f*(mean-self.position)+b*normal, 0., 'route_localize'))
        if self.future:
            candidates.append(('measure', self.future[0], 0., 'route_localize'))
        unique, seen = [], set()
        for a, q, value, reason in candidates:
            q = np.round(q, 6)
            key = (a, *q)
            if key in seen:
                continue
            if a == 'measure' and any(o.kind in ('direction', 'near', 'no_signal') and np.linalg.norm(q-[o.x, o.y]) < .1 for o in self.history[i]):
                continue
            seen.add(key); unique.append((a, q, value, reason))
        candidates = unique
        if not candidates or candidates[0][0] != baseline[0] or np.linalg.norm(candidates[0][1]-baseline[1]) > .001:
            return baseline
        train = post.sample(self.score_rng, self.particles)
        noise = self.score_rng.uniform(-1, 1, self.particles)
        scores = [self.scores(i, p, cover, train, noise, tail) for p in candidates]
        chosen = int(np.argmin([s.mean() for s in scores]))
        decision = dict(variant=self.route_variant, candidate_count=len(candidates), selected_index=chosen,
                        train_gain_s=float((scores[0]-scores[chosen]).mean()), accepted=False,
                        baseline_action=baseline[0], baseline_position=np.asarray(baseline[1]).tolist())
        if chosen:
            # Candidate selected using training samples; only that pair is
            # tested on a fresh posterior draw, avoiding reuse of the minimum.
            count = 96
            check = post.sample(self.check_rng, count)
            noise = self.check_rng.uniform(-1, 1, count)
            old_cost = self.scores(i, candidates[0], cover, check, noise, tail)
            new_cost = self.scores(i, candidates[chosen], cover, check, noise, tail)
            gain = old_cost-new_cost
            se = gain.std(ddof=1)/np.sqrt(count)
            accepted = gain.mean()-2*se > 1.
            decision.update(validation_count=count, validation_gain_s=float(gain.mean()),
                            validation_se_s=float(se), accepted=bool(accepted))
            if accepted:
                a, q, _, reason = candidates[chosen]
                self.last_decision = decision
                return a, q, float(new_cost.mean()), reason
        self.last_decision = decision
        return baseline

    def emit(self, action, i, q, reason, estimated=0.):
        result = super().emit(action, i, q, reason, estimated)
        if self.last_decision is not None and reason != 'existence_batch':
            result['route_decision'] = self.last_decision
            self.pending['route_decision'] = self.last_decision
        self.last_decision = None
        return result
