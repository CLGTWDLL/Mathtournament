"""Experimental bounded task lookahead and per-channel certified patrol.

Task lookahead compares source-first and patrol-then-source macro actions; the
patrol alternative is evaluated AT ITS OWN STATION and with the per-channel
effective-contribution filter, so a station that cannot contribute to any
unresolved channel is never selected and never travelled to.

Unknown discoveries are not simulated; the remaining route is a proxy, not an
exact global value function. No probability is used to delete a station.
"""
import numpy as np

from q4_dispatch_policy import choose_task
from q4_route_value import TailValue, finish_costs
from q4_sparse_backbone import CELLS, certified


# Mirrors q4_dispatch_policy.choose_task: the validated default only prefers a
# known source when inserting it between the current point and the next patrol
# station costs less than this detour. The MPC may rank eligible sources with a
# richer proxy, but it must not invent unbounded detours that the proxy cannot
# price (local 40-run comparison showed those lengthen the route).
SOURCE_DETOUR_M = 700.0


class Tasks:
    # Deviation from the validated dispatch action must beat its independent
    # validation by this margin. The route/clearing proxy is only a proxy, so
    # a 1 s margin overfits its own error: on the local paired comparison the
    # task layer predicted 100-500 s gains while the realised route grew by
    # kilometres. Conservative default; override per policy instance.
    task_accept_margin_s = 20.0
    # Conservative participation: the task layer EVALUATES and LOGS the
    # source-versus-patrol macro choice, but by default the executed task order
    # stays the validated dispatch rule. The local paired comparison showed the
    # route proxy predicting 100-500 s gains that did not materialise, and
    # making it binding lengthened the travelled route on 4/4 probed seeds.
    # Enable only with new independent evidence.
    task_override_enabled = False
    # The task proxy simulates finite clearing for every proposal. On a very
    # wide surviving support that dominates the episode, so above this many
    # cover cells the evaluation is skipped and logged as such; the executed
    # action is the validated dispatch choice either way.
    task_eval_cell_budget = 1500

    def _negative_positions(self, i):
        return [np.array([o.x, o.y]) for o in self.history[i] if o.kind == 'no_signal']

    def _cert(self, stations):
        key = tuple(sorted(set(tuple(np.round(q, 6)) for q in stations)))
        if key not in self._cert_cache:
            if len(self._cert_cache) >= 96:
                self._cert_cache.clear()
            self._cert_cache[key] = certified(list(key))
        return self._cert_cache[key]

    def _effective_channels(self, q):
        """Channels for which a measurement at ``q`` can still change the belief.

        This is the per-channel effective-contribution filter:

        * a channel whose whole remaining support is already excluded by the
          strict continuous certificate of its own no-signal sites cannot be
          learned from again;
        * any possible reception needs distance <= 1500 m, so a station with no
          surviving square inside that range carries no information;
        * a position already probed for the channel returns the identical
          deterministic reading and is skipped (same-point repeat guard).
        """
        q = np.round(np.asarray(q, float), 6)
        unknown = tuple(int(i) for i in self.unknown())
        signature = (float(q[0]), float(q[1]), unknown,
                     tuple(tuple(np.round(np.asarray(p, float), 6))
                           for i in unknown for p in self._negative_positions(i)))
        cache = getattr(self, '_channel_cache', None)
        if cache is None:
            cache = self._channel_cache = {}
        if signature in cache:
            return list(cache[signature])
        # Any possible reception requires distance <= 1500 m. Only squares
        # already excluded by a strict continuous certificate may be ignored.
        reachable = self._reachable(q)
        ids = []
        for i in unknown:
            negatives = self._negative_positions(int(i))
            if any(np.linalg.norm(q-p) < 1e-8 for p in negatives):
                continue
            if np.any(reachable & ~self._cert(negatives)):
                ids.append(int(i))
        if len(cache) >= 512:
            cache.clear()
        cache[signature] = tuple(ids)
        return list(ids)

    def _patrol_service(self, q):
        """Immediate seconds and channel list for one patrol stop at ``q``."""
        ids = self._effective_channels(q)
        ids.sort(key=lambda i: (i+1 != self.channel, i))
        seconds, channel = 0., self.channel
        for i in ids:
            seconds += 5+int(channel != i+1)
            channel = i+1
        return seconds, ids, channel

    def _reachable(self, q):
        """Cached 25 m cells whose nearest point is within 1500 m of ``q``."""
        key = (round(float(q[0]), 6), round(float(q[1]), 6))
        cache = getattr(self, '_reachable_cache', None)
        if cache is None:
            cache = self._reachable_cache = {}
        if key not in cache:
            if len(cache) >= 256:
                cache.clear()
            cache[key] = np.linalg.norm(np.maximum(np.abs(CELLS-np.asarray(q, float))-12.5, 0),
                                        axis=1) <= 1500.000001
        return cache[key]

    def _can_finish(self, future, extra=()):
        for i in self.unknown():
            if not self._cert(self._negative_positions(int(i))+list(extra)+list(future)).all():
                return False
        return True

    def _mark_certified_absent(self):
        for i in self.unknown():
            if self._cert(self._negative_positions(int(i))).all():
                self.absent[i] = True
                self.absence_reason[int(i)+1] = 'continuous_directional_cover'

    def start_batch(self, q, remove):
        q = np.round(np.asarray(q, float), 6)
        ids = self._effective_channels(q)
        skipped = sorted(set(map(int, self.unknown()))-set(ids))
        self.skipped_channel_observations += len(skipped)
        ids.sort(key=lambda i: (i+1 != self.channel, i))
        self.batch = dict(q=q, queue=ids, remove=remove, skipped=[i+1 for i in skipped])

    def replacement(self, q):
        if not len(self.unknown()) or not self.future:
            return []
        remove = []
        for j in np.argsort([np.linalg.norm(p-q) for p in self.future])[:3]:
            keep = [p for k, p in enumerate(self.future) if k not in remove+[int(j)]]
            if self._can_finish(keep, [q]):
                remove.append(int(j))
        return remove

    def _prune_redundant_future(self):
        if not self.future or not len(self.unknown()):
            return
        histories = tuple(sorted(set(tuple(tuple(q) for q in self._negative_positions(int(i)))
                                     for i in self.unknown())))
        signature = (tuple(tuple(q) for q in self.future), histories)
        if signature == self._prune_signature:
            return
        self._prune_signature = signature
        # Immediate no-contribution sites are removable without travelling.
        # A final full certificate check is retained as an implementation guard.
        for j in range(len(self.future)-1, -1, -1):
            if not self._effective_channels(self.future[j]):
                keep = self.future[:j]+self.future[j+1:]
                if self._can_finish(keep):
                    self.future = keep
                    self.pruned_sites += 1
        # Only new joint sites can change geometric redundancy of the stored
        # irredundant backbone. Avoid repeating 29 full deletion tests at every
        # ordinary patrol batch. Duplicated coordinates are cheap to detect.
        combined = [tuple(np.round(q, 6)) for q in self.visited+self.future]
        novel = any(q not in self._original_sites for q in combined)
        duplicate = len(set(combined)) != len(combined)
        if not (novel or duplicate):
            return
        for j in range(len(self.future)-1, -1, -1):
            keep = self.future[:j]+self.future[j+1:]
            if self._can_finish(keep):
                self.future = keep
                self.pruned_sites += 1

    def _task_choice(self, known, anchors, stations):
        old, info = choose_task(self.position, stations, anchors)
        info['known_channels'] = (known+1).tolist()
        if len(known) == 1 and not len(stations):
            return old, info
        covers, posts, train = [], [], []
        for i in known:
            cover, post = self.local_state(int(i))
            if not post.numeric_ok:
                info['task_mpc_fallback'] = 'unresolved_quadrature'
            covers.append(cover)
            posts.append(post)
            key = (int(i), len(self.history[i]))
            if key not in self._task_samples:
                self._task_samples = {k: v for k, v in self._task_samples.items() if k[0] != int(i)}
                self._task_samples[key] = post.scenarios(self.task_rng, self.particles)[0]
            train.append(self._task_samples[key])
        total_cells = sum(len(c.c) for c in covers)
        if total_cells > self.task_eval_cell_budget:
            info.update(task_mpc=False, task_mpc_fallback='cost_budget',
                        task_eval_cells=int(total_cells))
            return old, info
        order = list(np.argsort(np.linalg.norm(np.asarray(anchors)-self.position, axis=1))[:3])
        if old is not None and old not in order:
            order[-1] = old
        detour = (np.linalg.norm(np.asarray(anchors)-self.position, axis=1)
                  + np.linalg.norm(np.asarray(anchors)-(stations[0] if len(stations) else self.position), axis=1)
                  - np.linalg.norm((stations[0] if len(stations) else self.position)-self.position))
        eligible = [int(j) for j in order if detour[j] < SOURCE_DETOUR_M]
        if old is not None and old not in eligible:
            eligible.append(int(old))
        proposals = [('source', int(j)) for j in eligible]
        contributions, station_order = {}, []
        station_services = []
        if len(stations):
            # Per-channel effective-contribution filter for the patrol option:
            # a station is a candidate only if some unresolved channel can
            # still be learned from at that exact position. The station itself
            # stays the route-first one: reordering the offline-optimised
            # backbone by a route proxy was not independently validated and
            # lengthened the local 40-run route.
            for j, station in enumerate(stations):
                seconds, ids, _ = self._patrol_service(station)
                station_services.append(seconds)
                if int(j) == 0:
                    contributions[int(j)] = [int(i)+1 for i in ids]
                    if ids:
                        station_order.append(int(j))
            info['task_patrol_candidates'] = [
                dict(future_index=j, channels=contributions[j]) for j in station_order]
            proposals.extend(('patrol', int(j)) for j in station_order)
        residual = []
        for c, a, g in zip(covers, anchors, train):
            cost, _ = finish_costs(c, a, np.ones((len(g), len(c.c)), bool), g)
            residual.append(float(cost.mean())-5)
        tail_cache = {}

        def tail_value(start, remaining, others, services):
            key = (round(float(start[0]), 6), round(float(start[1]), 6),
                   tuple(tuple(np.round(p, 6)) for p in remaining),
                   tuple(tuple(np.round(a, 6)) for a in others),
                   tuple(np.round(services, 6)))
            if key not in tail_cache:
                tail_cache[key] = TailValue(start, remaining, others, self.unknown()+1,
                                            station_services=services)
            return tail_cache[key]

        def source_first(j, start, remaining, services, channel, samples):
            c, g = covers[j], samples[j]
            cost, endpoints = finish_costs(c, start, np.ones((len(g), len(c.c)), bool), g)
            others = [a for k, a in enumerate(anchors) if k != j]
            # Give every alternative the same expected service allowance for
            # other unresolved sources; anchor-at-5-seconds alone biases order.
            return (cost+tail_value(start, remaining, others, services)(endpoints, channel)
                    +sum(residual)-residual[j])

        def scores(proposal, samples):
            kind, j = proposal
            if kind == 'source':
                return source_first(j, self.position, stations, station_services,
                                    self.channel, samples)
            start = stations[j]
            remaining = [p for k, p in enumerate(stations) if k != j]
            services = [s for k, s in enumerate(station_services) if k != j]
            _, _, channel = self._patrol_service(start)
            service = station_services[j]
            immediate = np.linalg.norm(start-self.position)/5+service
            # Patrol first, then the best of the two nearest source tasks from
            # the new station; the route tail keeps every remaining station.
            local = list(np.argsort(np.linalg.norm(np.asarray(anchors)-start, axis=1))[:2])
            best = None
            for k in local:
                value = source_first(int(k), start, remaining, services, channel, samples)
                best = value if best is None else np.minimum(best, value)
            return immediate+best

        values = [scores(p, train) for p in proposals]
        patrol_ids = [j for j, p in enumerate(proposals) if p[0] == 'patrol']
        if old is None:
            if not patrol_ids:
                info.update(task_mpc=False, task_mpc_fallback='no_contributing_patrol_station')
                return old, info
            baseline = min(patrol_ids, key=lambda j: values[j].mean())
        else:
            baseline = proposals.index(('source', old))
        selected = int(np.argmin([v.mean() for v in values]))
        accepted = False
        info.update(task_mpc=True, task_candidates=len(proposals), task_model='finite_source_or_patrol_then_source',
                    unknown_discovery_value_included=False,
                    task_train_gain_s=float((values[baseline]-values[selected]).mean()))
        if proposals[selected][0] != proposals[baseline][0] or (proposals[selected][0] == 'source' and proposals[selected] != proposals[baseline]):
            count = max(48, 2*self.particles)
            check = [p.scenarios(self.task_check_rng, count)[0] for p in posts]
            gain = scores(proposals[baseline], check)-scores(proposals[selected], check)
            se = float(gain.std(ddof=1)/np.sqrt(count))
            accepted = bool(gain.mean()-2*se > self.task_accept_margin_s)
            info.update(task_validation_count=count, task_validation_gain_s=float(gain.mean()), task_validation_se_s=se)
        override = bool(accepted and self.task_override_enabled)
        executed = proposals[selected] if override else proposals[baseline]
        station = None if executed[0] == 'source' else int(executed[1])
        choice = int(executed[1]) if executed[0] == 'source' else None
        info.update(task_mpc_accepted=accepted, selected_choice=choice, changed=choice != old,
                    task_selected=executed[0], task_next_station=station,
                    task_next_source_channel=(int(known[choice])+1 if choice is not None else None),
                    task_selected_channels=(contributions.get(station) if station is not None else None),
                    task_override_enabled=bool(self.task_override_enabled),
                    task_override_applied=override,
                    task_proxy_selected=proposals[selected][0],
                    task_proxy_next_station=(None if proposals[selected][0] == 'source'
                                             else int(proposals[selected][1])),
                    task_proxy_index=int(selected), task_baseline_index=int(baseline),
                    task_would_change=bool(selected != baseline))
        return choice, info

    def next_action(self):
        if self.pending is not None:
            raise RuntimeError('Missing accepted feedback')
        while True:
            if self.complete():
                return None
            if self.confirmed.sum() == 16:
                for i in self.unknown():
                    self.absent[i] = True
                    self.absence_reason[int(i)+1] = 'sixteen_confirmed'
            if self.batch is not None:
                while self.batch['queue']:
                    i = self.batch['queue'].pop(0)
                    if not self.confirmed[i] and not self.absent[i]:
                        result = self.emit('measure', i, self.batch['q'], 'existence_batch')
                        result['skipped_channels_at_station'] = self.batch['skipped']
                        self.pending['skipped_channels_at_station'] = self.batch['skipped']
                        return result
                self.visited.append(self.batch['q'].copy())
                remove = self.batch['remove']
                self.future = [q for j, q in enumerate(self.future) if j not in remove]
                self.removed_sites += len(remove)
                self.batch = None
                self._mark_certified_absent()
                self._prune_redundant_future()
                continue
            if self.batch_after is not None:
                q, remove = self.batch_after
                self.batch_after = None
                if len(self.unknown()):
                    self.start_batch(q, remove)
                    self.joint_batches += 1
                continue
            known = self.known()
            if len(known):
                anchors = []
                for i in known:
                    cover, post = self.local_state(int(i))
                    anchors.append(post.weights@post.points if post.numeric_ok else cover.c.mean(axis=0))
                stations = self.future if len(self.unknown()) else []
                choice, self.dispatch_decision = self._task_choice(known, anchors, stations)
                if choice is not None:
                    i = int(known[choice])
                    action, q, value, reason = self.local_action(i)
                    q = np.round(q, 6)
                    if action == 'measure':
                        remove = self.replacement(q)
                        if remove:
                            self.batch_after = (q.copy(), remove)
                    return self.emit(action, i, q, reason, value)
                info = self.dispatch_decision or {}
                station = info.get('task_next_station')
                if info.get('task_selected') == 'patrol' and station is not None and self.future:
                    # MPC decided the next channel-investigation station; visit
                    # exactly that station instead of the first route element.
                    station = int(station)
                    self.start_batch(self.future[station], [station])
                    continue
            if self.future and len(self.unknown()):
                self.start_batch(self.future[0], [0])
                continue
            if self.complete():
                return None
            if len(self.unknown()):
                raise ArithmeticError('Patrol exhausted without per-channel certificate')
