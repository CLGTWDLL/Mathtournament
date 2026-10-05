"""生成第二问帕累托主结果、独立复评、图和表。"""

from pathlib import Path
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "tmp" / "question2_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from hybrid_q2 import select_hybrid
from model import Problem
from solve_q2_core import CoverModel, sample_conditional
from solve_q2_joint import evaluate as rollout_evaluate

DATA = ROOT / "数据"
FIGURES = ROOT / "图片"
TABLES = ROOT / "表格"
INPUT = DATA / "标准算例输入.json"
MAIN_RESULT = DATA / "标准算例主推荐结果.json"
AUDIT_RESULT = DATA / "独立复评结果.json"


def setup_font():
    choices = [Path(os.environ["CHINESE_FONT"])] if os.environ.get("CHINESE_FONT") else []
    installed_fonts = {Path(p).name.lower(): Path(p) for p in font_manager.findSystemFonts()}
    choices += [installed_fonts[name] for name in ("msyh.ttc", "simhei.ttf", "notosanscjk-regular.ttc") if name in installed_fonts]
    path = next((p for p in choices if p.is_file()), None)
    if path is None:
        raise RuntimeError("未找到中文字体；可通过 CHINESE_FONT 指定")
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family": font_manager.FontProperties(fname=str(path)).get_name(),
        "axes.unicode_minus": False,
    })


def summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "n": len(values),
        "mean_s": mean,
        "se_s": se,
        "ci95_s": [mean - 1.96 * se, mean + 1.96 * se],
        "p90_s": float(np.quantile(values, 0.9)),
    }


def paired(left, right):
    result = summary(np.asarray(left) - np.asarray(right))
    result["meaning"] = "前者耗时减后者耗时；负值表示前者更快"
    return result


def independent_rollout(points, scenario, settings):
    model = CoverModel(scenario)
    samples = sample_conditional(scenario, settings["n_samples"], settings["seed"])
    result = {}
    for name, point in points.items():
        values = rollout_evaluate(
            model,
            np.asarray(point, dtype=float),
            samples,
            settings["seed"] + 1_000_000,
            settings["nplan"],
            horizon=settings["horizon"],
        )
        result[name] = values[:, -1, 0]
    return result


def compact(candidate):
    return {
        "point": candidate["point"],
        "selection_rollout_mean_s": candidate["rollout_mean_s"],
        "action_time_s": candidate["finite_metric"]["action_time_s"],
        "finite_worst_diameter_m": candidate["finite_metric"]["worst_diameter_m"],
        "continuous_diameter_upper_m": candidate["continuous_metric"]["diameter_upper_m"],
    }


def main():
    source = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    problem = Problem(**source["first_observation"])
    fused = select_hybrid(problem, **source["hybrid_settings"])
    validation = source["validation_settings"]

    choices = {
        "最快前沿点": fused["fastest_front_point"],
        "帕累托拐点": fused["selected"],
        "高精度前沿点": fused["most_accurate_front_point"],
    }
    points = {name: row["point"] for name, row in choices.items()}
    scenario = {"first": [problem.x, problem.y], "bearing": problem.angle_deg}
    values = independent_rollout(points, scenario, validation)

    rows = []
    for name, candidate in choices.items():
        rows.append({
            "name": name,
            "point": candidate["point"],
            "independent_rollout": summary(values[name]),
            "action_time_s": candidate["finite_metric"]["action_time_s"],
            "finite_worst_diameter_m": candidate["finite_metric"]["worst_diameter_m"],
            "continuous_diameter_upper_m": candidate["continuous_metric"]["diameter_upper_m"],
            "physical_receive_feasible": candidate["physical_receive_feasible"],
        })

    audit = {
        "source": "自建中心场景独立同批复评；不是官方模拟器结果",
        "settings": validation,
        "rows": rows,
        "paired": {
            "拐点减最快": paired(values["帕累托拐点"], values["最快前沿点"]),
            "拐点减高精度": paired(values["帕累托拐点"], values["高精度前沿点"]),
        },
    }
    AUDIT_RESULT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = fused["selected"]
    selected_audit = next(row for row in rows if row["name"] == "帕累托拐点")
    main_result = {
        "source": "自建中心场景；不是官方模拟器或正式演练结果",
        "first_observation": source["first_observation"],
        "method": fused["method"],
        "performance_hard_limits": None,
        "physical_constraints": fused["physical_constraints"],
        "selection_rule": fused["selection_rule"],
        "selection_settings": source["hybrid_settings"],
        "second_point_local": selected["point"],
        "second_point_world": list(problem.world(selected["point"])),
        "selection_rollout_mean_s": selected["rollout_mean_s"],
        "independent_rollout": selected_audit["independent_rollout"],
        "action_time_s": selected["finite_metric"]["action_time_s"],
        "finite_worst_diameter_m": selected["finite_metric"]["worst_diameter_m"],
        "continuous_diameter_upper_m": selected["continuous_metric"]["diameter_upper_m"],
        "physical_receive_feasible": selected["physical_receive_feasible"],
        "pareto_options": [compact(row) for row in fused["pareto_front"]],
        "interpretation": "无D_lim、T_lim和lambda_risk；第三问可按剩余时间在前沿上动态切换。",
    }
    MAIN_RESULT.write_text(json.dumps(main_result, ensure_ascii=False, indent=2), encoding="utf-8")

    setup_font()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in rows]
    colors = ["#CB7638", "#3777A8", "#228569"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    bars = axes[0].bar(names, [r["independent_rollout"]["mean_s"] for r in rows], color=colors)
    axes[0].bar_label(bars, fmt="%.1f")
    axes[0].set(title="完整任务耗时", ylabel="秒")
    bars = axes[1].bar(names, [r["continuous_diameter_upper_m"] for r in rows], color=colors)
    axes[1].bar_label(bars, fmt="%.1f")
    axes[1].set(title="连续直径证书上界", ylabel="米")
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("第二问无性能硬阈值帕累托选点", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "融合主推荐对比.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# 表一：第二问帕累托主推荐",
        "",
        f"同一批 {validation['n_samples']} 个独立样本复评；不设 D_lim、T_lim 或风险权重。",
        "",
        "| 方案 | 第二点 | 完整耗时及95%区间（秒） | 动作耗时（秒） | 有限直径（米） | 连续上界（米） |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        point = row["point"]
        rollout = row["independent_rollout"]
        ci = rollout["ci95_s"]
        lines.append(
            f"| {row['name']} | ({point[0]:.3f}, {point[1]:.3f}) | "
            f"{rollout['mean_s']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
            f"{row['action_time_s']:.2f} | {row['finite_worst_diameter_m']:.2f} | "
            f"{row['continuous_diameter_upper_m']:.2f} |"
        )
    lines += ["", "三点是同一帕累托前沿上的不同工作档位，第三问可按剩余时间动态选用。"]
    (TABLES / "表一_融合主推荐.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(main_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
