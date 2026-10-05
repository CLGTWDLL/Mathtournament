# 第三问：联合重规划与保底清除证书

本目录只保留第三问最终解答和可运行核心。主解答见[第三问解答.md](第三问解答.md)，冻结主策略为 `q3_joint_replan.JointReplanPolicy`。

## 目录

- `代码/`：主算法、必要依赖、本地与接口入口、独立审计和核心测试。
- `代码/冻结版_28米联合重规划_20260912/`：带SHA-256清单的只读冻结副本。

## 主要代码

- `代码/q3_joint_replan.py`：联合检测点与反馈后任务重排的主策略。
- `代码/q3_backbone.py`：连续候选域覆盖骨架和保底证书。
- `代码/q3_joint_site.py`：检测点与清除任务的联合安排。
- `代码/run_q3_simulator_40.py`：本地连续运行40次并生成日志。
- `代码/q3_simulator_client.py`：协议客户端，仅作交付保留。

本地40次脚本会在需要时自动重建 `数据/本地40次运行/`。

## 本地复核

```powershell
python -m pip install -r "question3/代码/requirements.txt"
Push-Location "question3/代码"
python -m unittest test_q3_belief.py test_q3_policy.py test_q3_backbone.py test_q3_joint_site.py test_q3_simulator_client.py
Pop-Location
```

研究阶段的报告生成器、诊断脚本、图片、表格和实验日志已清理。正式演练材料未改动。
