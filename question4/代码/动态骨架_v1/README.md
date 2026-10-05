# 第四问当前算法：动态站点版

本目录是第四问现在使用的（也是唯一的）算法。入口是 `q4_dynamic_policy.Policy()`，它在被继承的基础控制器之上加了两件事：把旧的 29 个后续站当作**初始**路线，并在每次有效反馈之后用后验概率判断是否可以提前收工。

## 两层结构

1. **基础控制器** `q4_base_policy.Policy`（原 `q4_light_policy.Policy`，本次只改名）：29 个后续站加原点的稀疏骨架、25 米连续方向覆盖证书、一步前瞻为主、定位受阻时每源至多一次的有界两步场景树、逐频道扫描贡献过滤与有限清除后备。这一层保证排查不漏源。
2. **动态层** `q4_dynamic_policy.Policy`：每次获得无信号反馈后，用固定 Sobol 先验估计"至少还有一个未发现源"的后验概率。只有在已确认至少 10 个源、且该概率不超过 `existence_stop_threshold`（默认 1%）时，才停止访问剩余站点并结束本局；若确认源数达到 16，仍沿用基础控制器的确定性计数规则结束。

## 12.5 米反馈几何层：可选，默认关闭

动态层原本每步都对未解频道重算 `possible_mixed_cells(CELLS, 12.5, history)` 支撑域，用来"证明某频道不可能有源"从而提前关闭频道（`feedback_geometry=True` 还额外启用基于可行域的严格删站）。40 局同种子实测：

- 该层从未触发过（`dynamic_empty_channels = 0`；不存在证书类型只有 `posterior_remaining_below_threshold` 与 `sixteen_confirmed`）；
- 打开与关闭，40/40 局的虚拟时间与动作序列逐局完全一致，40/40 局清除全部实际源；
- 平均进程 CPU 时间由 15.4 秒升到 29.8 秒，其中绝大部分耗在这张 12.5 米网格的重复计算上。

因此 `feedback_geometry` 默认取 `False`，代码保留，需要几何闭环或严格删站时再打开。

## 参数

```python
Policy(particles=32, seed=9411, backbone=None,
       existence_stop_threshold=0.01, existence_samples=16384,
       strict_prune_interval=0, feedback_geometry=False)
```

- `existence_stop_threshold`：后验提前停止阈值；设为 `0` 或 `None` 可关闭概率停止，只剩基础控制器的确定性结束条件。
- `existence_samples`：后验求积的固定 Sobol 样本数（默认 16384），与实验种子无关。
- `strict_prune_interval`：严格可行域删站的检查间隔；`0` 表示关闭。该功能依赖 12.5 米几何层。
- `feedback_geometry`：12.5 米反馈几何层总开关，默认关闭。

## 保证范围

概率停止表示：在上述先验和观测模型成立的条件下，程序估计"仍有未发现源"的概率不超过设定阈值。它不提供确定性无遗漏证明，1% 也不是由有限次仿真直接证明的实际漏检率。基础控制器原有的连续方向覆盖证书、数量上界规则仍然有效，程序分别记录：

- `truth_complete`：离线实验中是否清除了模拟器实际生成的全部源；
- `replay_audit`：逐动作独立回放是否一致；
- `strict_guarantee`：本次结束是否完全没有使用概率证书；
- `strict_certificate_audit`：是否执行并通过基础控制器的确定性证书审计。

因此采用概率阈值的成功仿真不会被旧审计器误报为程序错误，也不会被写成已经取得确定性证明。

## 文件

- `q4_dynamic_policy.py`：补充优化实验算法。后验提前停止与可选几何层在此实现。
- `q4_base_policy.py`：被继承的基础控制器（本次由 `q4_light_policy.py` 改名）。
- `run40_dynamic.py`：40 局批量入口，每局单独记录虚拟时间、算法 CPU 时间与审计时间。
- `optimize_route.py`：离线路线优化，输出 `backbone.json`（中间结果）。
- `order_route.py`：离线发现顺序优化，输出 `backbone_final.json`（最终路线）。
- `test_dynamic.py`、`test_base_policy.py`：本算法的针对性正确性检查。
- `results/`：阈值敏感性等开发期记录。

## 运行

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
python -B 'question4\代码\动态骨架_v1\test_dynamic.py'
python -B 'question4\代码\动态骨架_v1\test_base_policy.py'
python -B 'question4\代码\动态骨架_v1\run40_dynamic.py' --runs 40 --start 1480000 --out 'question4\数据\对比40次\新批次\dynamic'
python -B 'question4\代码\动态骨架_v1\run40_dynamic.py' --runs 40 --start 1480000 --feedback-geometry --out 'question4\数据\对比40次\新批次\dynamic_geo'
```

直接使用策略接口：构造 `Policy()`，反复调用 `next_action()`，执行返回的公开动作后调用 `accept_feedback(feedback)`。批量入口拒绝覆盖已存在的输出目录。

## 与旧算法的关系与本次对照

旧算法（29 站静态骨架的独立版本）已删除，但新算法仍继承它的代码，继承链未改动，被继承的模块保留。两者在同一批 40 个候选种子（源数 10—16，均匀分布）上逐局配对：

| 指标 | 旧算法 | 动态提前停止方案（补充） |
| --- | ---: | ---: |
| 平均虚拟时间 / s | 6328.33 | 6183.91 |
| 平均进程 CPU 时间 / s | 15.41 | 持平（串行 A/B：12.97 → 13.18） |
| 清除全部实际源 | 40/40 | 40/40 |

虚拟时间平均节省 144.42 秒（2.28%），近似 95% Student-t 区间 [123.03, 165.81] 秒，不跨零；35/40 局更短，5 局相同（都是 16 个源、未触发概率停止的实例）。记录见 `question4/数据/对比40次/20260913-145508-pair40/`，变更细节见 `question4/更新说明.md`。

优化器拒绝覆盖已存在的输出。提供的路线已经生成且审核通过，通常无需重新优化；若要开展新一轮路线搜索，应另建版本目录，保留旧路线和旧实验。全部运行使用本地环境，未连接官方模拟器。

