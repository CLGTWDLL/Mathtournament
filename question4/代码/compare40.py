"""汇总两个 40 次批量目录，输出同种子配对对比。

用法：
  python -B compare40.py --light <light目录> --dynamic <dynamic目录>
结果写入批次目录：comparison.json、对比报告.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import t


def load(folder):
    return json.loads((folder / "batch.json").read_text(encoding="utf-8"))


def describe(values):
    values = [float(v) for v in values]
    if not values:
        return None
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
                minimum=min(values), maximum=max(values))


def paired(values):
    values = np.asarray([float(v) for v in values], float)
    mean = float(values.mean()) if len(values) else None
    half = (float(t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0)
    return dict(n=len(values), mean=mean, ci95=[mean - half, mean + half],
                wins=int((values > 1e-9).sum()), losses=int((values < -1e-9).sum()),
                ties=int((np.abs(values) <= 1e-9).sum()), total=float(values.sum()))


def main():
    parser = argparse.ArgumentParser(description="同种子配对对比两套算法的 40 次运行日志")
    parser.add_argument("--light", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    light_batch, dynamic_batch = load(args.light), load(args.dynamic)
    light = {row["seed"]: row for row in light_batch["rows"]}
    dynamic = {row["seed"]: row for row in dynamic_batch["rows"]}
    same_seeds = light_batch["seeds"] == dynamic_batch["seeds"]
    assert same_seeds, "两个批次的种子列表不一致，无法配对"
    seeds = list(light_batch["seeds"])
    assert all(seed in dynamic for seed in seeds), "存在缺失的局，无法完整配对"

    pairs = []
    for seed in seeds:
        old, new = light[seed], dynamic[seed]
        pairs.append(dict(
            seed=seed, source_count=old["source_count"],
            light_cleared=old["cleared"], dynamic_cleared=new["cleared"],
            light_complete=old["complete"], dynamic_complete=new["complete"],
            light_certificate=old.get("certificate_verified", False),
            dynamic_probability_stop=bool(new.get("probability_stopped_channels")),
            light_virtual_s=old["virtual_s"], dynamic_virtual_s=new["virtual_s"],
            saved_virtual_s=old["virtual_s"] - new["virtual_s"],
            light_cpu_s=old["cpu_s"], dynamic_cpu_s=new["cpu_s"],
            saved_cpu_s=old["cpu_s"] - new["cpu_s"],
            cpu_ratio=new["cpu_s"] / old["cpu_s"] if old["cpu_s"] else None,
            light_wall_s=old["wall_s"], dynamic_wall_s=new["wall_s"],
            saved_wall_s=old["wall_s"] - new["wall_s"],
            light_audit_wall_s=old["audit_wall_s"], dynamic_audit_wall_s=new["audit_wall_s"],
            light_move_m=old["move_m"], dynamic_move_m=new["move_m"],
            light_measurements=old["measurements"], dynamic_measurements=new["measurements"],
            light_log_entries=old["log_entries"], dynamic_log_entries=new["log_entries"],
        ))

    saved_virtual = [p["saved_virtual_s"] for p in pairs]
    saved_cpu = [p["saved_cpu_s"] for p in pairs]
    saved_wall = [p["saved_wall_s"] for p in pairs]

    def total(arm, key):
        return sum(float(p[f"{arm}_{key}"]) for p in pairs)

    result = dict(
        light_dir=str(args.light), dynamic_dir=str(args.dynamic),
        runs=len(pairs), same_seeds=same_seeds,
        light_algorithm=light_batch["algorithm"], dynamic_algorithm=dynamic_batch["algorithm"],
        light_all_completed=light_batch["all_completed"],
        dynamic_all_completed=dynamic_batch["all_completed"],
        light_total_virtual_s=total("light", "virtual_s"),
        dynamic_total_virtual_s=total("dynamic", "virtual_s"),
        light_total_cpu_s=total("light", "cpu_s"),
        dynamic_total_cpu_s=total("dynamic", "cpu_s"),
        light_total_wall_s=total("light", "wall_s"),
        dynamic_total_wall_s=total("dynamic", "wall_s"),
        mean_light_virtual_s=statistics.fmean([p["light_virtual_s"] for p in pairs]),
        mean_dynamic_virtual_s=statistics.fmean([p["dynamic_virtual_s"] for p in pairs]),
        mean_light_cpu_s=statistics.fmean([p["light_cpu_s"] for p in pairs]),
        mean_dynamic_cpu_s=statistics.fmean([p["dynamic_cpu_s"] for p in pairs]),
        mean_saved_virtual_s=statistics.fmean(saved_virtual),
        mean_saved_cpu_s=statistics.fmean(saved_cpu),
        mean_cpu_ratio=float(np.mean([p["cpu_ratio"] for p in pairs])),
        virtual_time_shorter_arm="dynamic" if statistics.fmean(saved_virtual) > 0 else "light",
        cpu_time_shorter_arm="dynamic" if statistics.fmean(saved_cpu) > 0 else "light",
        saved_virtual_s=paired(saved_virtual),
        saved_cpu_s=paired(saved_cpu),
        saved_wall_s=paired(saved_wall),
        light_virtual_s=describe([p["light_virtual_s"] for p in pairs]),
        dynamic_virtual_s=describe([p["dynamic_virtual_s"] for p in pairs]),
        light_cpu_s=describe([p["light_cpu_s"] for p in pairs]),
        dynamic_cpu_s=describe([p["dynamic_cpu_s"] for p in pairs]),
        by_source_count={}, pairs=pairs,
    )
    for count in sorted({p["source_count"] for p in pairs}):
        group = [p for p in pairs if p["source_count"] == count]
        result["by_source_count"][str(count)] = dict(
            runs=len(group),
            light_virtual_s=statistics.fmean([p["light_virtual_s"] for p in group]),
            dynamic_virtual_s=statistics.fmean([p["dynamic_virtual_s"] for p in group]),
            light_cpu_s=statistics.fmean([p["light_cpu_s"] for p in group]),
            dynamic_cpu_s=statistics.fmean([p["dynamic_cpu_s"] for p in group]),
            mean_saved_virtual_s=statistics.fmean([p["saved_virtual_s"] for p in group]),
            mean_saved_cpu_s=statistics.fmean([p["saved_cpu_s"] for p in group]),
            probability_stop_runs=sum(1 for p in group if p["dynamic_probability_stop"]),
        )

    out = args.out or args.dynamic.parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    vr, cr = result["saved_virtual_s"], result["saved_cpu_s"]
    lines = ["# 第四问新旧两套算法 40 次同种子配对对比", "",
             f"- 老算法（light）：{result['light_algorithm']}",
             f"- 新算法（dynamic）：{result['dynamic_algorithm']}",
             f"- 配对局数：{len(pairs)}，种子列表一致：{same_seeds}",
             f"- 全部完成：light={result['light_all_completed']}，"
             f"dynamic={result['dynamic_all_completed']}", "",
             "## 总体结果", "",
             "| 指标 | 老算法（29 站静态骨架） | 新算法（动态站点） | 差值（老-新） |",
             "| --- | ---: | ---: | ---: |",
             f"| 平均虚拟时间 / s | {result['mean_light_virtual_s']:.2f} | "
             f"{result['mean_dynamic_virtual_s']:.2f} | {result['mean_saved_virtual_s']:+.2f} |",
             f"| 虚拟时间合计 / s | {result['light_total_virtual_s']:.2f} | "
             f"{result['dynamic_total_virtual_s']:.2f} | "
             f"{result['light_total_virtual_s'] - result['dynamic_total_virtual_s']:+.2f} |",
             f"| 平均 CPU 时间 / s | {result['mean_light_cpu_s']:.3f} | "
             f"{result['mean_dynamic_cpu_s']:.3f} | {result['mean_saved_cpu_s']:+.3f} |",
             f"| CPU 时间合计 / s | {result['light_total_cpu_s']:.3f} | "
             f"{result['dynamic_total_cpu_s']:.3f} | "
             f"{result['light_total_cpu_s'] - result['dynamic_total_cpu_s']:+.3f} |",
             f"| 墙钟合计 / s | {result['light_total_wall_s']:.3f} | "
             f"{result['dynamic_total_wall_s']:.3f} | "
             f"{result['light_total_wall_s'] - result['dynamic_total_wall_s']:+.3f} |",
             "", "## 配对差值（老 - 新）的 95% 置信区间", "",
             f"- 虚拟时间：均值 {vr['mean']:.2f} s，"
             f"95%CI [{vr['ci95'][0]:.2f}, {vr['ci95'][1]:.2f}]，"
             f"新算法更短 {vr['wins']}/{len(pairs)} 局，相同 {vr['ties']} 局",
             f"- CPU 时间：均值 {cr['mean']:.3f} s，"
             f"95%CI [{cr['ci95'][0]:.3f}, {cr['ci95'][1]:.3f}]，"
             f"新算法更短 {cr['wins']}/{len(pairs)} 局，相同 {cr['ties']} 局",
             f"- 新/老 CPU 时间比：{result['mean_cpu_ratio']:.3f}", "",
             "## 按源数分组", "",
             "| 源数 | 局数 | 老算法平均虚拟 / s | 新算法平均虚拟 / s | "
             "老算法平均 CPU / s | 新算法平均 CPU / s | 新算法触发概率停止 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for count, group in result["by_source_count"].items():
        lines.append(f"| {count} | {group['runs']} | {group['light_virtual_s']:.2f} | "
                     f"{group['dynamic_virtual_s']:.2f} | {group['light_cpu_s']:.3f} | "
                     f"{group['dynamic_cpu_s']:.3f} | {group['probability_stop_runs']} |")
    lines += ["", "## 逐局明细", "",
              "| # | 种子 | 源数 | 老虚拟 / s | 新虚拟 / s | 虚拟差 / s | "
              "老CPU / s | 新CPU / s | CPU差 / s | 新算法概率停止 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, p in enumerate(pairs, 1):
        lines.append(
            f"| {index} | {p['seed']} | {p['source_count']} | {p['light_virtual_s']:.2f} | "
            f"{p['dynamic_virtual_s']:.2f} | {p['saved_virtual_s']:+.2f} | "
            f"{p['light_cpu_s']:.3f} | {p['dynamic_cpu_s']:.3f} | {p['saved_cpu_s']:+.3f} | "
            f"{'是' if p['dynamic_probability_stop'] else '否'} |")
    lines += ["", "## 结论", "",
              f"- 虚拟时间更短："
              f"{'新算法（动态站点）' if result['virtual_time_shorter_arm'] == 'dynamic' else '老算法（29 站静态骨架）'}",
              f"- CPU 时间更短："
              f"{'老算法（29 站静态骨架）' if result['cpu_time_shorter_arm'] == 'light' else '新算法（动态站点）'}",
              "- 全部结果为本地离线规则环境，未使用官方模拟器。", ""]
    (out / "对比报告.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({key: result[key] for key in
                      ("runs", "mean_light_virtual_s", "mean_dynamic_virtual_s",
                       "mean_light_cpu_s", "mean_dynamic_cpu_s", "mean_cpu_ratio",
                       "virtual_time_shorter_arm", "cpu_time_shorter_arm")},
                     ensure_ascii=False, indent=2))
    print(f"报告：{out/'对比报告.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
