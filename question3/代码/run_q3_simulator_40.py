"""离线运行问题3联合重规划算法40次，并保存逐动作日志。

文件名为兼容原来的启动方式而保留。脚本只调用本地 Environment，既不访问
HTTP 接口，也不启动、控制或连接官方模拟器。默认挑选40个近似均衡覆盖
10至16个目标数量的独立种子，并为每一局保存完整记录、JSONL动作日志、
单局摘要以及可逐动作重放的审计结果。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import uuid

import audit_q3_autonomous as checker
from q3_environment import Environment
from q3_joint_replan import JointReplanPolicy
import run_q3_autonomous as runner


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "数据" / "本地40次运行"
VERSION_FILES = (
    "q3_joint_replan.py",
    "q3_joint_site.py",
    "q3_backbone.py",
    "q3_policy.py",
    "q3_belief.py",
    "q3_environment.py",
    "run_q3_autonomous.py",
    "run_q3_simulator_40.py",
)


def algorithm_version() -> str:
    return hashlib.sha256(
        b"".join((ROOT / name).read_bytes() for name in VERSION_FILES)
    ).hexdigest()[:16]


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, result: dict, index: int) -> None:
    with path.open("x", encoding="utf-8") as stream:
        entries = [
            {
                "event": "run_start",
                "run_index": index,
                "version": result["version"],
                "seed": result["summary"]["seed"],
                "mode": result["summary"]["mode"],
                "official_simulator_used": False,
            },
            *({"event": "action", **row} for row in result["logs"]),
            {"event": "run_summary", **result["summary"]},
        ]
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")


def balanced_run_seeds(start: int, runs: int) -> list[int]:
    """Select exactly runs seeds, balancing source counts 10..16."""
    counts = range(10, 17)
    base, extra = divmod(runs, len(counts))
    quota = {count: base + int(i < extra) for i, count in enumerate(counts)}
    selected: list[int] = []
    seed = start
    while len(selected) < runs:
        count = Environment(seed).n
        if quota[count] > 0:
            selected.append(seed)
            quota[count] -= 1
        seed += 1
    return selected


def run_local_episode(seed: int, particles: int, scan_scale: float, version: str) -> dict:
    created: list[JointReplanPolicy] = []

    def factory(**kwargs):
        policy = JointReplanPolicy(**kwargs)
        created.append(policy)
        return policy

    previous_factory, previous_version = runner.AutonomousPolicy, runner.VERSION
    runner.AutonomousPolicy, runner.VERSION = factory, version
    try:
        result = runner.episode(seed, "joint_replan", particles, scan_scale, "uniform")
    finally:
        runner.AutonomousPolicy, runner.VERSION = previous_factory, previous_version

    if created:
        policy = created[-1]
        result["summary"]["joint_trials"] = policy.joint_trials
        result["summary"]["joint_accepted"] = policy.joint_accepted
        result["joint_history"] = policy.joint_history
    return result


def batch_summary(records: list[dict], requested: int, version: str) -> dict:
    completed = [r for r in records if r["status"] == "completed_with_certificate"]
    virtual = [float(r["virtual_s"]) for r in completed]
    wall = [float(r["wall_s"]) for r in records]
    return {
        "execution": "local_offline_environment",
        "official_simulator_used": False,
        "algorithm": "q3_joint_replan",
        "version": version,
        "requested_runs": requested,
        "attempted_runs": len(records),
        "completed_runs": len(completed),
        "all_completed": len(completed) == requested,
        "mean_virtual_s": statistics.fmean(virtual) if virtual else None,
        "median_virtual_s": statistics.median(virtual) if virtual else None,
        "stdev_virtual_s": statistics.stdev(virtual) if len(virtual) > 1 else 0.0 if virtual else None,
        "minimum_virtual_s": min(virtual) if virtual else None,
        "maximum_virtual_s": max(virtual) if virtual else None,
        "mean_wall_s": statistics.fmean(wall) if wall else None,
        "total_actions": sum(int(r["actions"]) for r in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=40, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--start", type=int, default=1200000, help="候选环境种子的起点")
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--scan-scale", type=float, default=0.5)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-fast", action="store_true", help="首局失败后立即停止；默认仍跑完其余局")
    parser.add_argument("--dry-run", action="store_true", help="只显示将使用的本地种子，不运行算法")
    args = parser.parse_args()

    if args.start < 0:
        parser.error("--start 不能为负数")
    if args.particles < 2:
        parser.error("--particles 至少为2")
    if args.scan_scale < 0:
        parser.error("--scan-scale 不能为负数")

    seeds = balanced_run_seeds(args.start, args.runs)
    source_counts = [Environment(seed).n for seed in seeds]
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_only",
            "execution": "local_offline_environment",
            "official_simulator_used": False,
            "runs": args.runs,
            "seeds": seeds,
            "source_counts": source_counts,
            "output_root": str(args.output_root),
        }, ensure_ascii=False, indent=2))
        return 0

    version = algorithm_version()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = args.output_root / f"batch-{stamp}-{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=False)
    checker.OUT = args.output_root
    batch_file = folder / "batch_summary.json"
    records: list[dict] = []

    print(f"日志目录：{folder}")
    print(f"离线运行问题3联合重规划 {args.runs} 局；不会连接官方模拟器。")

    for index, seed in enumerate(seeds, 1):
        name = f"run-{index:02d}-seed-{seed}"
        full_path = folder / f"{name}.json"
        log_path = folder / f"{name}.jsonl"
        summary_path = folder / f"{name}.summary.json"
        print(f"[{index:02d}/{args.runs:02d}] 本地种子 {seed}，目标数 {Environment(seed).n}……", flush=True)

        result = run_local_episode(seed, args.particles, args.scan_scale, version)
        write_json(full_path, result)
        write_jsonl(log_path, result, index)

        audit_verified = False
        audit_error = None
        if result["summary"]["complete"]:
            try:
                audit_verified = bool(checker.audit(full_path)["verified"])
            except Exception as error:
                audit_error = f"{type(error).__name__}: {error}"

        summary = result["summary"]
        complete = bool(summary["complete"] and audit_verified)
        record = {
            "run_index": index,
            "seed": seed,
            "status": "completed_with_certificate" if complete else "error_or_incomplete",
            "audit_verified": audit_verified,
            "audit_error": audit_error,
            "full_record_file": full_path.name,
            "log_file": log_path.name,
            "summary_file": summary_path.name,
            **summary,
        }
        write_json(summary_path, record)
        records.append(record)
        write_json(batch_file, batch_summary(records, args.runs, version))

        print(
            f"  清除 {summary['cleared']}/{summary['source_count']}，"
            f"虚拟 {summary['virtual_s']:.2f} 秒，CPU {summary['wall_s']:.2f} 秒，"
            f"动作 {summary['actions']}，审计 {'通过' if audit_verified else '未通过'}。",
            flush=True,
        )
        if not complete and args.fail_fast:
            break

    final = batch_summary(records, args.runs, version)
    write_json(batch_file, final)
    print(f"完成 {final['attempted_runs']} 局，通过 {final['completed_runs']} 局。")
    if final["mean_virtual_s"] is not None:
        print(f"平均虚拟耗时：{final['mean_virtual_s']:.2f} 秒")
    print(f"批次汇总：{batch_file}")
    return 0 if final["all_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
