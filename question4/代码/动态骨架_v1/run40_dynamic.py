"""40 次批量运行：第四问默认算法（动态站点版）。

仅本地离线环境，不连接官方模拟器。每局分别记录：

  virtual_s  环境虚拟时钟推进的时间（第四问的优化目标）
  cpu_s      该局算法本体消耗的进程 CPU 时间（time.process_time，不含审计）
  wall_s     该局墙钟时间（time.perf_counter，不含审计）

独立回放与确定性覆盖证书审计单独计时，写入 audit_wall_s / audit_cpu_s，
不计入 cpu_s，保证两种算法比较的都是算法本体的 CPU 时间。
概率提前停止的局只做逐动作回放审计，不声明确定性证书。
可选的 12.5 米反馈几何层默认关闭，用 --feedback-geometry 打开。

用法：
  python -B run40_dynamic.py --runs 40 --start 1480000 --out <批次目录>/dynamic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
for _path in (str(SOURCE), str(HERE)):
    if _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE))

import numpy as np
import scipy
from q4_environment import Environment
from q4_dynamic_policy import Policy
from q4_sparse_backbone import independent_cover_check
import run_q4 as audit_module
from run_utils import balanced_run_seeds, enrich_summary

ARM = "dynamic"
POLICY_DESCRIPTION = ("q4_dynamic_policy.Policy(existence_stop_threshold=0.01)：29 站骨架出发，"
                      "反馈更新后按 1% 后验阈值提前停止（12.5 米几何层默认关闭）")
THRESHOLD = 0.01
FEEDBACK_GEOMETRY = False


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def hashes(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir()) if p.is_file()}


def make_policy():
    return Policy(existence_stop_threshold=THRESHOLD, feedback_geometry=FEEDBACK_GEOMETRY)


def replay_audit(record):
    """逐动作独立回放，不把概率当作证明。"""
    summary = record["summary"]
    environment = Environment(summary["seed"], summary["distribution"])
    for entry in record["logs"]:
        feedback = environment.act(entry["action"], entry["channel"], entry["position"])
        if feedback != entry["feedback"]:
            raise AssertionError("Independent replay mismatch")
    if environment.successes != summary["cleared"]:
        raise AssertionError("Cleared-source count mismatch")
    if abs(environment.time - summary["virtual_s"]) >= 1e-8:
        raise AssertionError("Virtual-time mismatch")
    return dict(entries=len(record["logs"]), replay=True, cleared_matches=True,
                virtual_time_matches=True)


def strict_audit(record):
    """旧算法的确定性无遗漏证书审计（独立 Qhull 半平面检查）。"""
    audit_module.independent_cover_check = independent_cover_check
    return audit_module.audit(record)


def run_one(seed):
    environment = Environment(seed, "uniform")
    began, cpu_began = time.perf_counter(), time.process_time()
    policy = make_policy()
    init_s = time.perf_counter() - began
    init_cpu_s = time.process_time() - cpu_began
    logs, error = [], None
    try:
        for step in range(5000):
            action = policy.next_action()
            if action is None:
                break
            feedback = environment.act(action["action"], action["channel"], action["position"])
            policy.accept_feedback(feedback)
            logs.append(dict(step=step + 1, **action, feedback=feedback))
            if environment.time > 360000 or time.perf_counter() - began > 3600:
                raise RuntimeError("Virtual/real time cap")
        else:
            raise RuntimeError("Action cap")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - began
    cpu_s = time.process_time() - cpu_began
    policy_complete = bool(policy.complete())
    truth_complete = bool(environment.successes == environment.n)
    probability_stopped = int(getattr(policy, "probability_stopped_channels", 0))
    summary = enrich_summary(dict(
        arm=ARM, seed=seed, distribution="uniform", planner_seed=9411, particles=32,
        complete=bool(policy_complete and truth_complete and error is None),
        policy_complete=policy_complete, truth_complete=truth_complete,
        strict_guarantee=probability_stopped == 0,
        probability_stopped_channels=probability_stopped,
        source_count=environment.n, cleared=environment.successes,
        virtual_s=environment.time, move_m=environment.distance,
        measurements=environment.measurements, switches=environment.switches,
        clear_attempts=environment.attempts, absence_reason=policy.absence_reason,
        init_s=init_s, init_cpu_s=init_cpu_s, wall_s=wall_s, cpu_s=cpu_s,
        log_entries=len(logs), error=error,
        joint_batches=policy.joint_batches, pruned_sites=policy.pruned_sites,
        dynamic_pruned_sites=int(getattr(policy, "dynamic_pruned_sites", 0)),
        dynamic_empty_channels=int(getattr(policy, "dynamic_empty_channels", 0)),
        dynamic_stop_reason=getattr(policy, "dynamic_stop_reason", None),
        existence_stop_threshold=THRESHOLD,
        feedback_geometry=FEEDBACK_GEOMETRY,
        final_probability_any_remaining=(
            policy.existence_updates[-1]["probability_any_remaining"]
            if getattr(policy, "existence_updates", []) else None),
        remaining_future_stations=len(policy.future),
    ))
    record = dict(summary=summary, logs=logs, truth=environment.truth_for_evaluation())
    audit_began = audit_cpu_began = None
    if summary["complete"]:
        audit_began, audit_cpu_began = time.perf_counter(), time.process_time()
        try:
            record["replay_audit"] = replay_audit(record)
            if summary["strict_guarantee"]:
                record["strict_audit"] = strict_audit(record)
                summary["certificate_verified"] = True
            else:
                summary["certificate_verified"] = False
        except Exception as exc:
            summary["complete"] = False
            summary["error"] = f"Audit: {type(exc).__name__}: {exc}"
    summary["audit_wall_s"] = 0.0 if audit_began is None else time.perf_counter() - audit_began
    summary["audit_cpu_s"] = 0.0 if audit_began is None else time.process_time() - audit_cpu_began
    summary["cpu_total_s"] = summary["cpu_s"] + summary["audit_cpu_s"]
    return record


def batch_summary(rows, seeds, requested):
    def stat(key, sample):
        values = [float(row[key]) for row in sample if row.get(key) is not None]
        if not values:
            return None
        return dict(n=len(values), mean=statistics.fmean(values),
                    median=statistics.median(values), minimum=min(values),
                    maximum=max(values), stdev=statistics.stdev(values) if len(values) > 1 else 0.0)

    complete = [row for row in rows if row["complete"]]
    return dict(
        arm=ARM, algorithm=POLICY_DESCRIPTION, distribution="uniform",
        execution="local_offline_environment", official_simulator_used=False,
        feedback_geometry=FEEDBACK_GEOMETRY,
        planner_seed=9411, particles=32, requested_runs=requested,
        attempted_runs=len(rows), completed_runs=len(complete),
        all_completed=len(complete) == requested,
        seeds=list(seeds),
        source_counts={str(row["seed"]): row["source_count"] for row in rows},
        cleared_total=sum(int(row["cleared"]) for row in rows),
        sources_total=sum(int(row["source_count"]) for row in rows),
        probability_stopped_runs=sum(1 for row in rows if row.get("probability_stopped_channels")),
        stats=dict(
            virtual_s=stat("virtual_s", complete),
            cpu_s=stat("cpu_s", rows),
            wall_s=stat("wall_s", rows),
            audit_wall_s=stat("audit_wall_s", rows),
            cpu_total_s=stat("cpu_total_s", rows),
            move_m=stat("move_m", complete),
            measurements=stat("measurements", complete),
            log_entries=stat("log_entries", complete),
        ),
        total_virtual_s=sum(float(row["virtual_s"]) for row in complete),
        total_cpu_s=sum(float(row["cpu_s"]) for row in rows),
        total_wall_s=sum(float(row["wall_s"]) for row in rows),
        rows=rows,
    )


def main():
    global FEEDBACK_GEOMETRY
    parser = argparse.ArgumentParser(description="动态站点版 40 次本地批量运行")
    parser.add_argument("--runs", type=int, default=40)
    parser.add_argument("--start", type=int, default=1480000)
    parser.add_argument("--seeds", type=int, nargs="+", help="显式指定种子，覆盖 --runs/--start")
    parser.add_argument("--out", type=Path, help="本算法的输出目录")
    parser.add_argument("--feedback-geometry", action="store_true",
                        help="启用可选的 12.5 米反馈几何层（默认关闭，实测无收益且 CPU 翻倍）")
    args = parser.parse_args()
    FEEDBACK_GEOMETRY = bool(args.feedback_geometry)

    seeds = args.seeds if args.seeds else balanced_run_seeds(args.start, args.runs)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or (SOURCE.parent / "数据" / "对比40次" / f"{stamp}-{ARM}")
    out.mkdir(parents=True, exist_ok=False)

    manifest = dict(
        arm=ARM, algorithm=POLICY_DESCRIPTION, runs=len(seeds), seeds=seeds,
        distribution="uniform", planner_seed=9411, particles=32,
        existence_stop_threshold=THRESHOLD,
        feedback_geometry=FEEDBACK_GEOMETRY,
        execution="local_offline_environment", official_simulator_used=False,
        timing="virtual_s=solver clock; cpu_s=time.process_time of algorithm only; "
               "audit timed separately",
        thread_limit=1, python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(), cpu_count=os.cpu_count(),
        thread_env={key: os.environ.get(key) for key in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS")},
        source_hashes=hashes(HERE), inherited_source_hashes=hashes(SOURCE),
        started_at=stamp,
    )
    write_json(out / "manifest.json", manifest)
    write_json(out / "seeds.json", seeds)
    print(f"输出目录：{out}", flush=True)
    print(f"算法：{POLICY_DESCRIPTION}", flush=True)
    print(f"种子（{len(seeds)} 个）：{seeds}", flush=True)

    rows = []
    for index, seed in enumerate(seeds, 1):
        record = run_one(seed)
        row = record["summary"]
        write_json(out / f"seed-{seed}.json", record)
        write_json(out / f"seed-{seed}.summary.json", row)
        rows.append(row)
        write_json(out / "batch.json", batch_summary(rows, seeds, len(seeds)))
        print(f"[{index:02d}/{len(seeds)}] seed={seed} n={row['source_count']} "
              f"cleared={row['cleared']} complete={row['complete']} "
              f"virtual={row['virtual_s']:.2f}s cpu={row['cpu_s']:.3f}s "
              f"wall={row['wall_s']:.3f}s audit={row['audit_wall_s']:.3f}s "
              f"error={row['error']}", flush=True)

    final = batch_summary(rows, seeds, len(seeds))
    final["finished_at"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    final["source_hashes_unchanged"] = hashes(HERE) == manifest["source_hashes"]
    write_json(out / "batch.json", final)
    print(json.dumps(dict(arm=ARM, completed=final["completed_runs"],
                          mean_virtual_s=final["stats"]["virtual_s"]["mean"],
                          mean_cpu_s=final["stats"]["cpu_s"]["mean"],
                          sources_unchanged=final["source_hashes_unchanged"]),
                     ensure_ascii=False), flush=True)
    return 0 if final["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
