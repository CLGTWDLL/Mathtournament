# 论文绘图脚本

统一重绘入口集中在此文件夹；读取本项目的现有数据，不运行演练、不修改算法或原始JSON。

```bash
python -m pip install -r 绘图脚本/requirements.txt
python 绘图脚本/build_all.py
python 绘图脚本/integrate_manuscripts.py
```

推荐 Python 3.10 或以上。自动查找 macOS 宋体、Windows 中易宋体、Linux Noto Serif CJK；未找到时设置环境变量 `PAPER_FONT` 为中文字体文件的绝对路径。

- `common.py`：字体、色板、尺寸、PNG/SVG导出、来源哈希。
- `question1.py`：6幅几何及算法图；只调用确定性几何求解，不做随机试验。
- `question23.py`：第二问2幅图；第三问1幅理论机制图。
- `question3_results.py`：读取用户补充的40次运行，验证版本、路程与计时，生成3幅结果/轨迹图及汇总表。
- `question4.py`：当前骨架与实际轨迹、主要收益、耗时分解、动态增量4幅图。
- `tables.py`：5张精简结果表（第三问脚本另生成1张），输出Markdown和HTML三线表预览。
- `build_all.py`：批量生成至项目根目录 `论文图表/`。
- `integrate_manuscripts.py`：将图片和独立图注接入各问正文；首次保存原稿与原图备份，再次运行更新已有图块。保留其他正文。

图内文字统一黑色；蓝与砖红只区分图形对象。解释、统计口径、来源和限制作为图片下方的独立图注。通栏默认6.3英寸（160 mm），字9 pt；狭长候选域图单独使用较窄画幅。SVG文字转路径以保证跨设备显示，文字修改通过脚本重绘。

图号按各问内部行文次序排列，合并全文时再统一编号。输出清单及输入文件哈希保存在 `论文图表/图表清单.json`。若修改正文后希望哈希与当前文件完全一致，可最后再运行一次 `build_all.py`。
