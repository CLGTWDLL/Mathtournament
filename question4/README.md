# 第四问：混合源搜索、定位与清除

主解答见[第四问解答.md](第四问解答.md)。论文正式主方案为 V5 / sparse / 29站稀疏骨架；补充优化实验算法为
`代码/动态骨架_v1/q4_dynamic_policy.Policy()`：

- 基础层：29 个后续站加原点的稀疏骨架、25 米连续方向覆盖证书、一步前瞻为主、定位受阻时每源至多一次的有界两步场景树；
- 动态层：每次反馈更新后估计"至少还有一个未发现源"的后验概率，确认至少 10 个源且该概率不超过 1% 时，提前结束剩余站点访问。

12.5 米反馈几何层（几何闭环与严格删站）为**可选功能，默认关闭**：40 局同种子实测中它一次都没有触发过，虚拟时间与动作序列和关闭时逐局完全一致，却让平均 CPU 时间由 15.4 秒升到 29.8 秒。需要时用 `Policy(feedback_geometry=True)` 或 `run40_dynamic.py --feedback-geometry` 打开。

旧算法（29 站静态骨架的独立版本 `轻量优化_v1`）已删除；新算法仍继承它的代码，继承链未改动：`q4_dynamic_policy.Policy` → `q4_base_policy.Policy`（原 `q4_light_policy.Policy`）→ `q4_gdop_mpc_policy.Policy`。本次更新改了什么见[更新说明.md](更新说明.md)。

## 目录

- `代码/`：第四问算法、测试与实验入口，包含使其可独立运行所需的第二问、第三问依赖副本。
- `数据/对比40次/20260913-145508-pair40/`：本次更新依据——40 局新旧同种子配对、删除几何层的对照运行、串行 CPU A/B。
- `数据/实验结果/sparse_backbone/`：三类场景下 29 站与 37 站骨架的 84 份配对记录、汇总与两张图（稀疏臂为早于动态层的实现，只用于说明骨架选择）。
- `图片/`：从实验结果汇集的图，保留原相对目录。
- `表格/`：核心结果汇总与数据表索引。

## 主要代码

- `代码/动态骨架_v1/q4_dynamic_policy.py`：补充优化实验算法。
- `代码/动态骨架_v1/q4_base_policy.py`：被继承的基础控制器（29 站骨架、25 米证书、有界场景树），不再是独立算法入口。
- `代码/动态骨架_v1/run40_dynamic.py`：40 局批量入口。
- `代码/动态骨架_v1/optimize_route.py`、`order_route.py`：离线路线与发现顺序优化。
- `代码/q4_sparse_backbone.py`、`代码/q4_dispatch_policy.py`：骨架与调度基础模块，也是骨架对照实验的两个对照臂。
- `代码/run_q4_sparse.py`：骨架对照实验入口。
- `代码/compare40.py`：由两个 40 局批次生成配对报告。

## 本地复核

```powershell
python -m pip install -r "question4\代码\requirements.txt"
Push-Location "question4\代码"
python -m unittest test_q4.py test_q4_dispatch.py test_q4_sparse.py
Pop-Location
python -B "question4\代码\动态骨架_v1\test_dynamic.py"
python -B "question4\代码\动态骨架_v1\test_base_policy.py"
python -B "question4\代码\动态骨架_v1\run40_dynamic.py" --runs 40 --start 1480000 --out "question4\数据\对比40次\新批次\dynamic"
```

全部脚本只使用本地环境，没有连接模拟器、没有启动演练，也没有改动正式演练材料。

