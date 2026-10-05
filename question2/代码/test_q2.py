"""第二问当前主链路回归测试。"""

import inspect
import json
from pathlib import Path
import unittest

from continuous_region import certify_rectangle
from hybrid_q2 import _pareto_front, _select_knee, _validate_rollout_constants, select_hybrid
from model import Problem, is_safe
from rollout_q2 import select_second_point

ROOT = Path(__file__).resolve().parents[1]


def row(time_s, bound_m):
    return {
        "rollout_mean_s": time_s,
        "continuous_metric": {"available": True, "diameter_upper_m": bound_m},
    }


class Q2Tests(unittest.TestCase):
    def test_runtime_sources_are_self_contained(self):
        for name in ["hybrid_q2.py", "rollout_q2.py", "make_fused_results.py"]:
            text = (Path(__file__).parent / name).read_text(encoding="utf-8")
            self.assertNotIn(' / "B题"', text)
            self.assertNotIn("bq3_dependencies", text)
            self.assertNotIn(' / "wait"', text)

    def test_dependencies_are_declared(self):
        requirements = (Path(__file__).parent / "requirements.txt").read_text(encoding="utf-8")
        for package in ["numpy", "scipy", "matplotlib"]:
            self.assertIn(package, requirements.lower())

    def test_second_point_policy_is_local(self):
        result = select_second_point((0.0, 0.0), 0.0)
        self.assertEqual(result["source"], "代码/q2_policy.py")

    def test_main_has_no_performance_threshold_arguments(self):
        parameters = inspect.signature(select_hybrid).parameters
        for name in ["diameter_limit", "time_limit", "lambda_risk", "selection_mode"]:
            self.assertNotIn(name, parameters)

    def test_pareto_removes_dominated_point(self):
        front = _pareto_front([row(10, 100), row(12, 120), row(15, 50)])
        self.assertEqual([(x["rollout_mean_s"], x["continuous_metric"]["diameter_upper_m"]) for x in front],
                         [(10, 100), (15, 50)])

    def test_knee_selects_middle_tradeoff(self):
        front = [row(10, 100), row(12, 50), row(20, 40)]
        self.assertIs(_select_knee(front), front[1])

    def test_rollout_uses_one_degree(self):
        with self.assertRaises(ValueError):
            _validate_rollout_constants(Problem(error_deg=0.9))
        _validate_rollout_constants(Problem(error_deg=1.0))

    def test_input_has_no_performance_hard_limits(self):
        source = json.loads((ROOT / "数据" / "标准算例输入.json").read_text(encoding="utf-8"))
        self.assertEqual(source["first_observation"]["error_deg"], 1.0)
        settings = source["hybrid_settings"]
        for name in ["diameter_limit", "time_limit", "lambda_risk", "selection_mode"]:
            self.assertNotIn(name, settings)

    def test_continuous_certificate_accepts_no_performance_limits(self):
        result = certify_rectangle((443.75, 456.25, 168.75, 181.25), Problem())
        self.assertFalse(result["diameter_limit_applied"])
        self.assertFalse(result["time_limit_applied"])
        self.assertIsNotNone(result["diameter_upper_m"])

    def test_saved_result_is_physical_and_on_front(self):
        result = json.loads((ROOT / "数据" / "标准算例主推荐结果.json").read_text(encoding="utf-8"))
        self.assertIsNone(result["performance_hard_limits"])
        self.assertTrue(result["physical_receive_feasible"])
        self.assertTrue(is_safe(result["second_point_local"], Problem(), 1.0))
        points = [option["point"] for option in result["pareto_options"]]
        self.assertIn(result["second_point_local"], points)
        self.assertGreater(result["continuous_diameter_upper_m"], 0)
        self.assertEqual(result["independent_rollout"]["n"], 512)


if __name__ == "__main__":
    unittest.main(verbosity=2)
