# 第四问代码

补充优化实验算法是 `动态骨架_v1/q4_dynamic_policy.Policy()`：29 个后续站加原点、约 17.51 公里开放路线、25 米连续方向覆盖证书，局部决策以一步前瞻为主并对定位受阻状态触发有界两步场景树；在此之上，每次反馈更新后估计"至少还有一个未发现源"的后验概率，确认至少 10 个源且概率不超过 1% 时提前结束剩余站点访问。

12.5 米反馈几何层（几何闭环与严格可行域删站）**默认关闭**，用 `Policy(feedback_geometry=True)` 或 `run40_dynamic.py --feedback-geometry` 打开。40 局同种子实测该层从未触发，虚拟时间与动作序列和关闭时逐局一致，CPU 时间却翻倍。

继承链未改动：`q4_dynamic_policy.Policy` → `动态骨架_v1/q4_base_policy.Policy`（原 `q4_light_policy.Policy`）→ `q4_gdop_mpc_policy.Policy` → `q4_dispatch_policy.Policy`。旧算法作为独立入口（`轻量优化_v1/`）已删除，被继承的基础模块保留。

## 目录

- `动态骨架_v1/`：当前算法本体。
  - `q4_dynamic_policy.py`：补充优化实验算法（后验提前停止 + 可选几何层）。
  - `q4_base_policy.py`：被继承的基础控制器（29 站骨架、25 米证书、有界场景树）。
  - `run40_dynamic.py`：40 局批量入口，`--feedback-geometry` 打开几何层。
  - `optimize_route.py`、`order_route.py`：离线路线与发现顺序优化，输出 `backbone.json`、`backbone_final.json`。
  - `test_dynamic.py`、`test_base_policy.py`：针对性正确性检查。
  - `results/`：阈值敏感性等开发期记录。
- 共享模块：`q4_environment.py`（本地环境）、`q4_sparse_backbone.py`（29 站骨架、25 米整格证书与独立 Qhull 审计）、`q4_dispatch_policy.py`（调度流程与 37 站密集骨架对照）、`q4_gdop_mpc_policy.py` / `q4_feedback_policy.py` / `q4_route_policy.py` / `q4_fine_clear.py` / `q4_mpc_*.py`（后验、覆盖与场景树基础实现）、`q4_policy.py` 与 `run_q4.py`（审计与基线 episode 流程）。
- 第二问、第三问依赖副本：`q2_analytic.py`、`solve_q2.py`、`q3_belief.py`、`q3_environment.py`、`q3_policy.py`。
- `run_utils.py`：批量运行的种子均衡与时间分解口径；`compare40.py`：由两个 40 局批次生成配对报告。
- `run_q4_sparse.py`：29 站/37 站骨架对照实验入口（`数据/实验结果/sparse_backbone/` 的复现入口）。

## 入口

```powershell
# 补充优化实验算法的针对性检查
python -B 动态骨架_v1\test_dynamic.py
python -B 动态骨架_v1\test_base_policy.py

# 40 局批量运行（输出目录必须不存在；不覆盖已有批次）
python -B 动态骨架_v1\run40_dynamic.py --runs 40 --start 1480000 --out ..\数据\对比40次\新批次\dynamic

# 由两个批次生成新旧配对报告
python -B compare40.py --light ..\数据\对比40次\20260913-145508-pair40\light --dynamic ..\数据\对比40次\20260913-145508-pair40\dynamic

# 三类场景下的稀疏/密集骨架配对（结果写入 ../数据/实验结果/sparse_backbone/）
python run_q4_sparse.py
```

每局分别记录虚拟耗时、算法计算时间与独立审计时间。骨架按文件内容缓存验证结果；静态路线直接加载，避免重复生成后丢弃。

## 测试

```powershell
python -m unittest test_q4.py test_q4_dispatch.py test_q4_sparse.py
```

全部脚本只使用本地环境，未连接官方模拟器。

