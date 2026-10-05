# 第三问代码

主策略入口为 `q3_joint_replan.py`，本地批量评估入口为 `run_q3_joint_replan.py`。`q2_analytic.py` 与 `solve_q2.py` 是为保持第三问代码自包含而迁入的第二问依赖。

各脚本的实验输出已统一改到 `../数据/实验结果/`。`q3_simulator_client.py` 仅作为协议交付文件保留；本次迁移未执行其联网运行参数。

## 本地连续运行40次

直接运行下面的命令。它只使用本地环境，不连接或控制官方模拟器；每局保存
完整 JSON、逐动作 JSONL、单局摘要和批次汇总。

```powershell
python run_q3_simulator_40.py
```

文件名为了兼容原来的启动方式而保留，日志默认写入
`../数据/本地40次运行/`。

只查看将使用的40个本地种子、不真正运行算法时，可执行：

```powershell
python run_q3_simulator_40.py --dry-run
```

当前固定情景版统一使用48个后验情景。覆盖、清除和完成证书不依赖该数量。
