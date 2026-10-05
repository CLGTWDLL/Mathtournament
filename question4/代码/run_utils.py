"""第四问批量运行共用的种子均衡与统计口径。"""
from __future__ import annotations

import json

from q4_environment import Environment


def balanced_run_seeds(start, runs):
    """按真实源数 10—16 均匀摊派候选种子。"""
    counts = range(10, 17)
    base, extra = divmod(runs, len(counts))
    quota = {count: base + int(i < extra) for i, count in enumerate(counts)}
    seeds = []
    seed = start
    while len(seeds) < runs:
        count = Environment(seed, "uniform").n
        if quota[count]:
            seeds.append(seed)
            quota[count] -= 1
        seed += 1
    return seeds


def enrich_summary(summary):
    """把动作计数换算成第四问时间口径的分解项。"""
    summary = dict(summary)
    summary["move_s"] = summary["move_m"] / 5
    summary["measurement_s"] = 5 * summary["measurements"]
    summary["switch_s"] = summary["switches"]
    summary["clear_s"] = 3 * summary["clear_attempts"] + 2 * summary["cleared"]
    summary["actions"] = summary["measurements"] + summary["clear_attempts"]
    return summary


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
