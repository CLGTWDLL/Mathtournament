"""第二问无人为性能硬阈值的选点主链。

候选点只需满足题设的保证接收条件；完整任务 rollout 耗时与连续
后验直径上界同时进入帕累托比较。默认选择无量纲前沿上相对两端连线
下凹最大的拐点，不使用 D_lim、T_lim 或风险权重。
"""

import math

import numpy as np

from continuous_region import certify_rectangle
from model import Problem, evaluate, is_safe, prior_polygon, target_scenarios
from rollout_q2 import select_second_point
from solve_q2_core import CoverModel, local_origin, sample_conditional
from solve_q2_joint import evaluate as rollout_evaluate


def _validate_rollout_constants(problem):
    """完整 rollout 当前对应的题设常量。"""
    expected = {
        "error_deg": 1.0,
        "target_radius": 1800.0,
        "receive_min": 1000.0,
        "receive_max": 1500.0,
        "near_radius": 5.0,
        "speed": 5.0,
        "measure_time": 5.0,
    }
    mismatched = [
        name
        for name, value in expected.items()
        if abs(float(getattr(problem, name)) - value) > 1e-9
    ]
    if mismatched:
        raise ValueError("完整rollout暂不支持修改这些题设常量：" + ", ".join(mismatched))


def _finite_metric(point, problem):
    return evaluate(
        point,
        problem,
        prior_polygon(problem),
        target_scenarios(problem),
        noise_count=5,
        weight=0.0,
    )


def _continuous_metric(point, problem, cell_size, safety_buffer):
    """对候选点周围整块矩形给出连续直径上界。

    证书只校验题设接收可行性并返回原始直径上界，不施加性能阈值。
    """
    a, b = float(point[0]), abs(float(point[1]))
    half = float(cell_size) / 2.0
    if a < half or b < half:
        return {
            "available": False,
            "diameter_upper_m": None,
            "reason": "候选点过于靠近坐标轴，当前局部矩形证书不适用",
        }
    certificate = certify_rectangle(
        (a - half, a + half, b - half, b + half),
        problem,
        diameter_limit=None,
        time_limit=None,
        safety_buffer=safety_buffer,
    )
    available = bool(
        certificate["reception_verified"]
        and certificate["diameter_upper_m"] is not None
    )
    return {
        "available": available,
        "diameter_upper_m": certificate["diameter_upper_m"] if available else None,
        "time_upper_s": certificate["time_upper_s"],
        "receive_margin_lower_m": certificate["receive_margin_lower_m"],
        "rectangle_upper_half": certificate["rectangle_upper_half"],
    }


def _pareto_front(candidates):
    eligible = [
        row
        for row in candidates
        if row["continuous_metric"]["available"]
        and math.isfinite(row["rollout_mean_s"])
    ]
    front = [
        row
        for row in eligible
        if not any(
            other["rollout_mean_s"] <= row["rollout_mean_s"]
            and other["continuous_metric"]["diameter_upper_m"]
            <= row["continuous_metric"]["diameter_upper_m"]
            and (
                other["rollout_mean_s"] < row["rollout_mean_s"]
                or other["continuous_metric"]["diameter_upper_m"]
                < row["continuous_metric"]["diameter_upper_m"]
            )
            for other in eligible
        )
    ]
    return sorted(
        front,
        key=lambda row: (
            row["rollout_mean_s"],
            row["continuous_metric"]["diameter_upper_m"],
        ),
    )


def _select_knee(front):
    if not front:
        raise ValueError("没有同时具备连续误差上界和完整耗时的候选点")
    times = np.array([row["rollout_mean_s"] for row in front], dtype=float)
    bounds = np.array(
        [row["continuous_metric"]["diameter_upper_m"] for row in front], dtype=float
    )
    time_span = max(float(times.max() - times.min()), 1e-12)
    bound_span = max(float(bounds.max() - bounds.min()), 1e-12)
    time_norm = (times - times.min()) / time_span
    bound_norm = (bounds - bounds.min()) / bound_span
    knee_scores = 1.0 - time_norm - bound_norm
    for row, score in zip(front, knee_scores):
        row["pareto_knee_score"] = float(score)
    return max(front, key=lambda row: (row["pareto_knee_score"], -row["rollout_mean_s"]))


def _candidate_points(problem, scenario):
    """从题设尺度生成无性能阈值的确定性候选网格。"""
    scale = float(problem.receive_min)
    orientation = 1.0 if local_origin(scenario)[1] >= 0 else -1.0
    points = []
    for u in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        for v in (0.125, 0.175, 0.225, 0.275, 0.325):
            points.append(([u * scale, orientation * v * scale], "尺度化帕累托网格"))
    rollout = select_second_point((problem.x, problem.y), problem.angle_deg)
    points.append((rollout["local_point"], rollout["method"]))
    return points


def select_hybrid(
    problem,
    *,
    cell_size=12.5,
    safety_buffer=1.0,
    nplan=24,
    n_samples=128,
    seed=71100,
    horizon=5,
):
    problem.validate()
    _validate_rollout_constants(problem)
    if cell_size <= 0 or safety_buffer < 0 or nplan < 2 or n_samples < 2 or horizon < 1:
        raise ValueError("证书网格、接收余量或rollout参数无效")
    scenario = {
        "key": "runtime",
        "first": [float(problem.x), float(problem.y)],
        "bearing": float(problem.angle_deg),
    }
    model = CoverModel(scenario)
    samples = sample_conditional(scenario, n_samples, seed)
    candidates = []
    seen = set()
    for point, source in _candidate_points(problem, scenario):
        point = [float(point[0]), float(point[1])]
        key = (round(point[0], 8), round(point[1], 8))
        if key in seen or not is_safe(point, problem, safety_buffer):
            continue
        seen.add(key)
        values = rollout_evaluate(
            model,
            np.asarray(point, dtype=float),
            samples,
            seed + 1_000_000,
            nplan,
            horizon=horizon,
        )[:, -1, 0]
        se = float(values.std(ddof=1) / math.sqrt(len(values)))
        candidates.append(
            {
                "point": point,
                "source": source,
                "physical_receive_feasible": True,
                "rollout_mean_s": float(values.mean()),
                "rollout_se_s": se,
                "rollout_ci95_s": [
                    float(values.mean() - 1.96 * se),
                    float(values.mean() + 1.96 * se),
                ],
                "rollout_p90_s": float(np.quantile(values, 0.9)),
                "finite_metric": _finite_metric(point, problem),
                "continuous_metric": _continuous_metric(
                    point, problem, cell_size, safety_buffer
                ),
            }
        )
    front = _pareto_front(candidates)
    selected = _select_knee(front)
    return {
        "method": "物理可行域 + 完整耗时/连续误差帕累托拐点",
        "problem": problem.__dict__,
        "performance_hard_limits": None,
        "physical_constraints": [
            "目标圆域与首次±1度观测",
            "接收半径、近距离阈值、速度和操作耗时",
            "第二点对首次可行目标具有保证接收性",
        ],
        "selection_rule": "在耗时与连续直径的无量纲帕累托前沿上选最大下凹拐点",
        "certificate_cell_size_m": cell_size,
        "safety_buffer_m": safety_buffer,
        "rollout_settings": {
            "nplan": nplan,
            "n_samples": n_samples,
            "seed": seed,
            "horizon": horizon,
        },
        "selected": selected,
        "fastest_front_point": front[0],
        "most_accurate_front_point": min(
            front, key=lambda row: row["continuous_metric"]["diameter_upper_m"]
        ),
        "pareto_front": front,
        "candidate_count": len(candidates),
    }
