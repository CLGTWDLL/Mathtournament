"""Certified offline route + one-step control with bounded event-triggered MPC.

Original modules are imported read-only. No environment seed or truth reaches
this policy. A 25 m conservative cover, count rule and per-channel continuous
certificates retain the original completion conditions.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
from q4_gdop_mpc_policy import Policy as MPCPolicy
from q4_sparse_backbone import Policy as SparsePolicy, load_backbone
from q4_dispatch_policy import choose_task


class Policy(MPCPolicy):
    """Spend tree-search effort only on bounded, stalled localization states."""

    def __init__(self, particles=32, seed=9411, backbone=None):
        self.light_backbone = Path(backbone) if backbone else HERE/'backbone_final.json'
        self.light_mpc_calls = np.zeros(20, dtype=int)
        self.light_one_step_calls = 0
        self.light_decision = None
        super().__init__('dispatch', particles, seed, clear_half=12.5)

    def initial_stations(self):
        return load_backbone(self.light_backbone)

    def _task_choice(self, known, anchors, stations):
        # The original task-tree override is disabled by default. Avoid doing
        # expensive diagnostic rollouts whose result cannot change the action.
        choice, info = choose_task(self.position, stations, anchors)
        info.update(known_channels=(known+1).tolist(), task_mpc=False,
                    task_model='geometric_route_insertion',
                    unknown_discovery_value_included=False)
        return choice, info

    def local_action(self, i):
        cover, post = self.local_state(i)
        hist = self.history[i]
        latest = hist[-1].kind if hist else None
        # Trigger only after unproductive feedback or at least two additional
        # bearings, with bounded support and at most one tree per source.
        stalled = latest in ('no_signal', 'no_target_in_range') or self.local_rf[i] >= 2
        use_mpc = bool(post.numeric_ok and stalled and 4 <= len(cover.c) <= 400
                       and self.local_rf[i] < 5 and self.steps < 1200
                       and self.light_mpc_calls[i] < 1
                       and not any(o.kind == 'near' for o in hist))
        self.light_decision = dict(tree_triggered=use_mpc, cover_cells=len(cover.c),
                                   latest_feedback=latest, clear_grid_m=25.0)
        if use_mpc:
            self.light_mpc_calls[i] += 1
            return MPCPolicy.local_action(self, i)
        self.light_one_step_calls += 1
        self.mpc_decision = None
        return SparsePolicy.local_action(self, i)

    def emit(self, action, i, q, reason, estimated=0.):
        result = super().emit(action, i, q, reason, estimated)
        if self.light_decision is not None:
            result['light_decision'] = self.light_decision
            self.pending['light_decision'] = self.light_decision
            self.light_decision = None
        return result
