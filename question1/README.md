# 第一问：交会定位区域

完整数学解答见 [第一问解答.md](第一问解答.md)。当前全部统一使用题设 `±1°` 误差：将每次测向写成前向误差楔形，取半平面交得到可行区域，再枚举顶点对计算直径。

## 必要文件

- `代码/geometry.py`：纯 Python 核心求解。
- `代码/solve.py`：自定义 JSON 观测的命令行入口。
- `代码/examples.py`：正文与测试使用的确定性算例。
- `代码/make_results.py`：重算图、表和汇总结果。
- `代码/test_geometry.py`：18 项测试。
- `数据/正交双点定位.json`：最小示例输入。
- `数据/全部求解结果.json`：完整精度汇总结果。

## 复现

```powershell
python -m pip install -r "question1/代码/requirements.txt"
python "question1/代码/test_geometry.py"
python "question1/代码/make_results.py"
python "question1/代码/solve.py" "question1/数据/正交双点定位.json"
```

题目没有给出第一问的具体检测点数据，数值结果来自明确标注的自建算例，不是官方模拟器或正式演练结果。
