"""第四问默认算法：29 站骨架 + 反馈驱动动态调整。

两层结构：

1. 基础控制器（``q4_base_policy.Policy``）：29 个后续站加原点的稀疏骨架、
   25 米连续方向覆盖证书、一步前瞻为主并对定位受阻状态触发有界两步场景树。
   这一层保证排查不漏源。
2. 动态层（本文件）：每步获得必要反馈后，用固定 Sobol 先验估计
   "至少还有一个未发现源"的后验概率；确认至少 10 个源且该概率不超过
   ``existence_stop_threshold``（默认 1%）时，提前结束剩余站点访问。

``feedback_geometry``（默认 False）控制可选的 12.5 米反馈几何层：
开启后，额外的保守支撑域计算可以证明"某频道不可能有源"从而提前关闭该频道，
同时启用基于可行域的严格删站。40 局同种子配对实测显示，该层在均匀分布实例上
从未触发过（``dynamic_empty_channels = 0``），虚拟时间与动作序列和关闭时逐局
完全一致，却使平均 CPU 时间由 15.4 秒升到 29.8 秒，因此默认关闭。
原始策略与路线文件按只读方式复用，不做修改。
"""
from pathlib import Path
import sys
from functools import lru_cache
from math import comb

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from q4_feedback_geometry import possible_mixed_cells
from q4_sparse_backbone import CELLS, certified
from q4_base_policy import Policy as BasePolicy


@lru_cache(maxsize=4)
def _prior_scenarios(count):
    """Fixed Sobol prior samples; independent of simulator seeds and truth."""
    count = int(count)
    if count < 256:
        raise ValueError('existence_samples must be at least 256')
    power = int(np.ceil(np.log2(count)))
    design = qmc.Sobol(d=5, scramble=True, seed=20260913).random_base2(power)[:count]
    angle = 2 * np.pi * design[:, 0]
    radial = 1800 * np.sqrt(design[:, 1])
    positions = radial[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    radii = 1000 + 500 * design[:, 2]
    directional = design[:, 3] < 0.5
    theta = 2 * np.pi * design[:, 4]
    outward = np.column_stack((np.cos(theta), np.sin(theta)))
    for array in (positions, radii, directional, outward):
        array.setflags(write=False)
    return positions, radii, directional, outward
class Policy(BasePolicy):
    """基础控制器 + 后验概率提前停止（可选 12.5 米反馈几何层）。"""

    def __init__(self, particles=32, seed=9411, backbone=None,
                 existence_stop_threshold=0.01, existence_samples=16384,
                 strict_prune_interval=0, feedback_geometry=False):
        if existence_stop_threshold is not None and not 0 <= existence_stop_threshold < 1:
            raise ValueError('existence_stop_threshold must lie in [0, 1) or be None')
        self.feedback_geometry = bool(feedback_geometry)
        self.dynamic_pruned_sites = 0
        self.dynamic_empty_channels = 0
        self.dynamic_stop_reason = None
        self.dynamic_route_events = []
        self._dynamic_support_cache = {}
        self._dynamic_prune_signature = None
        self._last_dynamic_prune_visit_count = 0
        self.dynamic_prune_interval = int(strict_prune_interval)
        self.existence_stop_threshold = existence_stop_threshold
        self.existence_samples = int(existence_samples)
        self.existence_updates = []
        self.probability_stopped_channels = 0
        self._existence_likelihood_cache = {}
        super().__init__(particles=particles, seed=seed, backbone=backbone)

    def _support_mask(self, channel):
        """Necessary 25 m-square support after all public RF feedback."""
        history = tuple(self.history[int(channel)])
        key = history
        if key not in self._dynamic_support_cache:
            if len(self._dynamic_support_cache) >= 128:
                self._dynamic_support_cache.clear()
            mask = possible_mixed_cells(CELLS, 12.5, history).copy()
            mask.setflags(write=False)
            self._dynamic_support_cache[key] = mask
        return self._dynamic_support_cache[key]

    def _support_groups(self):
        """Group unresolved channels with identical feedback histories."""
        groups = {}
        for channel in map(int, self.unknown()):
            history = tuple(self.history[channel])
            groups.setdefault(history, []).append(channel)
        return [
            (channels, self._support_mask(channels[0]), self._negative_positions(channels[0]))
            for channels in groups.values()
        ]

    def _existence_likelihood(self, channel):
        """P(all observed misses | the channel contains a source)."""
        positions = tuple(sorted({
            (round(float(observation.x), 6), round(float(observation.y), 6))
            for observation in self.history[int(channel)]
            if observation.kind == 'no_signal'
        }))
        if positions in self._existence_likelihood_cache:
            return self._existence_likelihood_cache[positions]
        if not positions:
            return 1.0
        sources, radii, directional, outward = _prior_scenarios(self.existence_samples)
        survives = np.ones(len(sources), dtype=bool)
        for q in positions:
            delta = np.asarray(q) - sources
            receive = np.linalg.norm(delta, axis=1) <= radii
            receive &= (~directional) | (np.sum(delta * outward, axis=1) >= 0)
            survives &= ~receive
        # Jeffreys smoothing avoids turning a finite quadrature miss into a
        # literal zero probability.
        likelihood = float((survives.sum() + 0.5) / (len(survives) + 1.0))
        self._existence_likelihood_cache[positions] = likelihood
        return likelihood

    @staticmethod
    def _polynomial(weights, skip=None):
        coefficients = np.zeros(len(weights) + 1)
        coefficients[0] = 1.0
        used = 0
        for index, weight in enumerate(weights):
            if index == skip:
                continue
            coefficients[1:used + 2] += coefficients[:used + 1] * weight
            used += 1
        return coefficients[:used + 1]

    def _posterior_existence(self):
        """Coupled existence posterior under the uniform 10--16 count prior."""
        unknown = list(map(int, self.unknown()))
        fixed = int(self.confirmed.sum())
        likelihoods = np.array([self._existence_likelihood(i) for i in unknown])
        coefficients = self._polynomial(likelihoods)

        denominator = 0.0
        probability_none_mass = 0.0
        for count_unknown, coefficient in enumerate(coefficients):
            total = fixed + count_unknown
            if 10 <= total <= 16:
                mass = coefficient / comb(20, total)
                denominator += mass
                if count_unknown == 0:
                    probability_none_mass += mass
        if denominator <= 0:
            return dict(valid=False, channels=unknown, likelihoods=likelihoods.tolist())

        marginals = []
        for index, likelihood in enumerate(likelihoods):
            without = self._polynomial(likelihoods, skip=index)
            numerator = 0.0
            for count_without, coefficient in enumerate(without):
                total = fixed + 1 + count_without
                if 10 <= total <= 16:
                    numerator += likelihood * coefficient / comb(20, total)
            marginals.append(float(numerator / denominator))
        distribution = []
        for count_unknown, coefficient in enumerate(coefficients):
            total = fixed + count_unknown
            if 10 <= total <= 16:
                distribution.append((count_unknown, float(
                    coefficient / comb(20, total) / denominator
                )))
        return dict(
            valid=True,
            channels=unknown,
            likelihoods=likelihoods.tolist(),
            marginals=marginals,
            probability_any_remaining=float(1.0 - probability_none_mass / denominator),
            expected_remaining=float(sum(k * mass for k, mass in distribution)),
            remaining_count_distribution=distribution,
        )

    def _apply_probability_stop(self):
        threshold = self.existence_stop_threshold
        if threshold is None or threshold <= 0 or not len(self.unknown()):
            return False
        # The count prior forces at least ten sources, so no early stop is
        # possible before ten channels have been confirmed.
        if self.confirmed.sum() < 10:
            return False
        posterior = self._posterior_existence()
        if not posterior.get('valid'):
            return False
        event = dict(
            step=int(self.steps),
            confirmed=int(self.confirmed.sum()),
            threshold=float(threshold),
            **posterior,
        )
        self.existence_updates.append(event)
        if posterior['probability_any_remaining'] > threshold:
            return False
        unresolved = list(map(int, self.unknown()))
        for channel in unresolved:
            self.absent[channel] = True
            self.absence_reason[channel + 1] = 'posterior_remaining_below_threshold'
        self.probability_stopped_channels += len(unresolved)
        self.dynamic_pruned_sites += len(self.future)
        self.pruned_sites += len(self.future)
        self.future = []
        self.dynamic_stop_reason = 'posterior_threshold'
        return True

    def _station_channels(self, q):
        """Unresolved channels for which measuring at q can change feedback."""
        q = np.round(np.asarray(q, float), 6)
        reachable = self._reachable(q)
        useful = []
        for channels, support, _ in self._support_groups():
            history = self.history[channels[0]]
            if any(
                observation.kind in ('direction', 'near', 'no_signal')
                and np.linalg.norm(q - np.array([observation.x, observation.y])) < 1e-8
                for observation in history
            ):
                continue
            if np.any(support & reachable):
                useful.extend(channels)
        return useful

    def _mark_feedback_empty(self):
        """Close a channel only after a restricted continuous proof."""
        if not self.feedback_geometry:
            return
        for channels, support, negatives in self._support_groups():
            empty = not np.any(support)
            covered = bool(
                len(negatives) >= 3
                and certified(negatives, CELLS[support]).all()
            )
            if empty or covered:
                for channel in channels:
                    self.absent[channel] = True
                    self.absence_reason[channel + 1] = (
                        'feedback_geometry_empty' if empty
                        else 'restricted_directional_cover'
                    )
                    self.dynamic_empty_channels += 1

    def _dynamic_can_finish(self, future):
        """Whether future visits suffice on each channel's current support."""
        for _, support, negatives in self._support_groups():
            if not np.any(support):
                continue
            stations = negatives + list(future)
            if len(stations) < 3:
                return False
            if not certified(stations, CELLS[support]).all():
                return False
        return True

    def _dynamic_prune(self):
        """Drop future stations with zero possible contribution.

        A station is retained whenever at least one unresolved channel still
        has a feasible square within the maximum 1500 m reception range.  Thus
        pruning never relies on posterior probability or an expected-value
        threshold.
        """
        if not self.feedback_geometry:
            return
        if not self.future or not len(self.unknown()):
            return
        if self.dynamic_prune_interval <= 0:
            return
        visit_count = len(self.visited)
        if (
            len(self.future) > 4
            and visit_count - self._last_dynamic_prune_visit_count < self.dynamic_prune_interval
        ):
            return
        self._last_dynamic_prune_visit_count = visit_count
        signature = (
            tuple(tuple(np.round(q, 6)) for q in self.future),
            tuple(
                (channel, tuple(self.history[channel]))
                for channel in map(int, self.unknown())
            ),
        )
        if signature == self._dynamic_prune_signature:
            return
        self._dynamic_prune_signature = signature

        kept, removed = [], []
        for station in self.future:
            if self._station_channels(station):
                kept.append(station)
            else:
                removed.append(np.round(np.asarray(station, float), 6).tolist())
        for index in range(len(kept) - 1, -1, -1):
            trial = kept[:index] + kept[index + 1:]
            if self._dynamic_can_finish(trial):
                removed.append(np.round(np.asarray(kept[index], float), 6).tolist())
                kept = trial
        if removed:
            self.future = kept
            count = len(removed)
            self.dynamic_pruned_sites += count
            self.pruned_sites += count
            self.dynamic_route_events.append({
                'step': int(self.steps),
                'removed': removed,
                'remaining_stations': len(self.future),
                'unresolved_channels': (self.unknown() + 1).tolist(),
                'reason': 'redundant_on_current_feasible_regions',
            })

    def _prune_redundant_future(self):
        """Keep the inherited proof-based pruning, then apply feedback pruning."""
        super()._prune_redundant_future()
        self._mark_feedback_empty()
        if not self._apply_probability_stop():
            self._dynamic_prune()

    def next_action(self):
        if self.pending is None and self.batch is None and self.batch_after is None:
            self._mark_feedback_empty()
            if not self._apply_probability_stop():
                self._dynamic_prune()
            if self.complete():
                if self.dynamic_stop_reason is None:
                    self.dynamic_stop_reason = 'all_channels_resolved'
                return None
        try:
            action = super().next_action()
        except ArithmeticError:
            if not self.future and len(self.unknown()):
                self.dynamic_stop_reason = 'no_informative_future_station'
                return None
            raise
        if action is None:
            if self.complete():
                if self.dynamic_stop_reason is None:
                    self.dynamic_stop_reason = 'all_channels_resolved'
            elif not self.future and len(self.unknown()):
                self.dynamic_stop_reason = 'no_informative_future_station'
        return action
