"""Report terminal-scoring and subsequent task-dispatch stages separately."""
import hashlib
import json
from report_layout import write_report
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from build_q3_autonomous_report import stratified_interval
from run_q4_route import version as route_version
from run_q4_dispatch import version as dispatch_version

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent/'数据'/'实验结果'/'route_analysis'
DIST = dict(uniform='均匀', outward='边缘朝外', cluster='聚集')


def collect(folders, modes, expected):
    paths = [p for f in folders for p in sorted((ROOT.parent/'数据'/'实验结果'/f).glob('*_*.json')) if p.name != 'summaries.json']
    records = [json.loads(p.read_text()) for p in paths]
    assert len(records) == 126 and {r['version'] for r in records} == {expected}
    assert all(r['summary']['complete'] and r['audit']['continuous_hull_check'] for r in records)
    groups = {}
    for dist in DIST:
        lookup = {(r['summary']['seed'], r['summary']['mode']): r for r in records if r['summary']['distribution'] == dist}
        seeds = sorted({s for s, m in lookup})
        assert len(seeds) == 14 and len(lookup) == 42
        counts = [lookup[s, 'geometry']['summary']['source_count'] for s in seeds]
        assert all(counts.count(n) == 2 for n in range(10, 17))
        group = {}
        for mode in modes:
            rows = [lookup[s, mode]['summary'] for s in seeds]
            value = dict(cases=14, cleared=sum(r['cleared'] for r in rows), sources=sum(r['source_count'] for r in rows))
            for key in ('virtual_s', 'average_per_cleared_s', 'move_m', 'measurements', 'switches', 'clear_attempts', 'wall_s'):
                value[key] = float(np.mean([r[key] for r in rows]))
            value['movement_s'] = value['move_m']/5
            value['measurement_s'] = 5*value['measurements']
            value['optical_and_clear_s'] = 3*value['clear_attempts']+26
            if mode != 'geometry':
                delta = np.array([lookup[s, 'geometry']['summary']['virtual_s']-lookup[s, mode]['summary']['virtual_s'] for s in seeds])
                value['paired'] = dict(saved_s=float(delta.mean()), fraction=float(delta.mean()/group['geometry']['virtual_s']),
                                       ci95=stratified_interval(delta, counts), faster=int((delta > 1e-6).sum()),
                                       tied=int((np.abs(delta) <= 1e-6).sum()), slower=int((delta < -1e-6).sum()))
            group[mode] = value
        a, b = modes[1:]
        delta = np.array([lookup[s, a]['summary']['virtual_s']-lookup[s, b]['summary']['virtual_s'] for s in seeds])
        group['last_vs_middle'] = dict(saved_s=float(delta.mean()), ci95=stratified_interval(delta, counts))
        groups[dist] = group
    return dict(version=expected, groups=groups, episodes=126,
                sources=sum(r['summary']['source_count'] for r in records),
                cleared=sum(r['summary']['cleared'] for r in records),
                actions=sum(r['audit']['actions'] for r in records),
                absent_certificates=sum(r['audit']['absent_certificates'] for r in records)), paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, ap = collect(('route_holdout', 'route_holdout_b'), ('geometry', 'validated', 'route'), route_version())
    b, bp = collect(('dispatch_holdout', 'dispatch_holdout_b'), ('geometry', 'dispatch', 'dispatch_route'), dispatch_version())
    result = dict(terminal_stage=a, dispatch_stage=b, official_tests=False)
    (OUT/'statistics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    code = [ROOT/f for f in ('q4_route_value.py', 'q4_route_policy.py', 'run_q4_route.py', 'q4_dispatch_policy.py',
                             'run_q4_dispatch.py', 'test_q4_route.py', 'test_q4_dispatch.py', 'build_q4_route_report.py',
                             'q4_feedback_geometry.py', 'q4_feedback_policy.py', 'run_q4_feedback.py',
                             'q4_belief.py', 'q4_policy.py', 'q4_coverage.py', 'q4_environment.py', 'run_q4.py',
                             'q3_belief.py', 'q3_policy.py', 'q3_environment.py', 'q2_analytic.py', 'solve_q2.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ap+bp+code}
    (OUT/'evidence_sha256.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2)+'\n')
    font = Path('/System/Library/Fonts/STHeiti Light.ttc')
    if font.exists():
        plt.rcParams['font.family'] = FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1))
    for stage, ax, modes, names in ((a, axes[0], ('validated', 'route'), ('独立复核局部评分', '再加入出口与重排')),
                                   (b, axes[1], ('dispatch', 'dispatch_route'), ('首任务调度', '首任务调度＋出口评分'))):
        for j, (mode, name) in enumerate(zip(modes, names)):
            means = np.array([stage['groups'][d][mode]['paired']['saved_s'] for d in DIST])
            ci = np.array([stage['groups'][d][mode]['paired']['ci95'] for d in DIST])
            ax.errorbar(np.arange(3)+(j-.5)*.16, means, yerr=np.vstack((means-ci[:, 0], ci[:, 1]-means)), fmt='o',
                        color=('#4f839f', '#bf7755')[j], capsize=4, label=name)
        ax.axhline(0, color='#777777', lw=1)
        ax.set_xticks(range(3), list(DIST.values())); ax.set_ylabel('相对该批几何基线节省（秒）')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=.15)
    axes[0].set_title('先清除同一个源：出口影响有限\n134/135系列，每类14例')
    axes[1].set_title('先决定处理哪个任务：比较整条剩余路线\n另取136/137系列，每类14例')
    fig.tight_layout(); fig.savefig(OUT/'出口评分与任务调度对比.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    old = np.array([[0, 0], [1000, 300], [1000, 0], [1000, 1000]])
    new = np.array([[0, 0], [1000, 0], [1000, 300], [1000, 1000]])
    ax.plot(old[:, 0], old[:, 1], '--o', color='#bd775c', lw=2, label='先做源：约2344米')
    ax.plot(new[:, 0], new[:, 1], '-o', color='#4f839f', lw=2, label='先去第一站：2000米')
    for p, t in [([0, 0], '当前位置'), ([1000, 0], '排查站1'), ([1000, 300], '源锚点'), ([1000, 1000], '排查站2')]:
        ax.annotate(t, p, xytext=(10, 8), textcoords='offset points')
    ax.set_xlim(-120, 1400); ax.set_ylim(-100, 1150); ax.set_aspect('equal')
    ax.set_title('低于700米的当前插入，也可能应该推迟')
    ax.set_xlabel('x（米）'); ax.set_ylabel('y（米）'); ax.legend(loc='upper left'); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(OUT/'稍后顺路处理示例.png', dpi=170); plt.close(fig)
    def table(stage, modes):
        return '\n'.join(f"| {DIST[d]} | "+' | '.join(f"{g[m]['virtual_s']:.2f}" for m in modes)+' |'
                         for d, g in stage['groups'].items())
    comparisons = '\n'.join(f"| {DIST[d]} | {name} | {g[m]['paired']['saved_s']:.2f} | {g[m]['paired']['fraction']*100:.2f}% | [{g[m]['paired']['ci95'][0]:.2f}, {g[m]['paired']['ci95'][1]:.2f}] | {g[m]['paired']['faster']}/{g[m]['paired']['tied']}/{g[m]['paired']['slower']} |"
                            for d, g in b['groups'].items() for m, name in (('dispatch', '首任务调度'), ('dispatch_route', '调度＋出口评分')))
    components = '\n'.join(f"| {DIST[d]} | {name} | {g[m]['movement_s']:.2f} | {g[m]['measurement_s']:.2f} | {g[m]['switches']:.2f} | {g[m]['optical_and_clear_s']:.2f} | {g[m]['wall_s']:.2f} |"
                            for d, g in b['groups'].items() for m, name in (('geometry', '基线'), ('dispatch', '首任务调度'), ('dispatch_route', '调度＋出口')))
    extra = '\n'.join(f"| {DIST[d]} | {g['last_vs_middle']['saved_s']:.2f} | [{g['last_vs_middle']['ci95'][0]:.2f}, {g['last_vs_middle']['ci95'][1]:.2f}] |" for d, g in b['groups'].items())
    report = f'''# 第四问第三轮：续策评分与清除前任务调度

数学推导与实现范围见 [出口价值与续策改进边界](问题4_出口价值与续策改进边界.md)。本轮先检验清除后的出口评分，发现影响有限后，再用另外的新样本检验清除前的首任务选择；两阶段不混为同一批独立试验。

**当前主研究版本：`q4_dispatch_policy.Policy(variant="dispatch")`，整程比较版本指纹为 `{b['version']}`。** 它保留负反馈几何裁剪和原局部策略，仅用完整未来路线插入替换700米首任务门槛。三类新样本中，相对同批第二轮几何基线的平均耗时均降低，配对近似区间均不跨零；额外出口评分的边际区间均跨零，因此暂不将更复杂版本设为主方案。此选择在比较后作出，不应把本批数据再当作未来未查看的留出集。

## 1. 进展的核心

增加出口和反馈后路线重排，还不足以完整表达联合搜索。若两个方案都先清除同一个源，它们的成功清除点最多相距40米；在同一剩余路线库下，出口评分对动作相对成本的修正最多8秒，接收频道差异最多再加1秒。

本轮进一步把决定前移：**现在清除当前源、先处理其他源、还是先去排查站，再回来处理？** 新策略将所有已知源后验锚点插入完整的未来排查序列，比较有限参考路线的首任务，收到每次反馈后重新选择。取消“当前绕路小于700米就先做”的固定规则。

![稍后顺路处理示例](问题4结果/route_analysis/稍后顺路处理示例.png)

三种新模式始终保留连续覆盖证书、负反馈几何裁剪和有限清除后备。路线锚点是后验均值，不是泄露的真实坐标。路线最短只指当前有限参考路线库中的几何代理，不宣称整程全局最优。

## 2. 阶段A：清除后的出口评分

版本 `{a['version']}`。1340000、1350000两批全新种子，每种分布按N=10—16各2例，共14组配对。候选选择用32个后验情景，新动作再由另外96个情景与旧动作单独比较；新增随机数流与旧策略隔离。三种模式各完成42/42例、清除546/546个源。

| 分布 | 第二轮几何基线总时间/秒 | 独立复核局部评分/秒 | 再加出口与重排/秒 |
|---|---:|---:|---:|
{table(a, ('geometry', 'validated', 'route'))}

该阶段整体收益有限，不能据此声称已经计算旧策略的真实剩余期望价值。出口代理尚未模拟未知源的发现、其他源的后续定位以及联合扫描中的新反馈。独立验证控制了一部分候选选择噪声，却没有消除这些模型偏差。

## 3. 阶段B：清除前的首任务调度

版本 `{b['version']}`。这项实现是在阶段A分析后确定的，因此另取1360000、1370000新种子，仍每种分布14组配对。比较基线、仅改变任务调度、任务调度加出口评分。三种模式各完成42/42例、清除546/546个源。

| 分布 | 该批几何基线总时间/秒 | 首任务调度/秒 | 首任务调度＋出口评分/秒 |
|---|---:|---:|---:|
{table(b, ('geometry', 'dispatch', 'dispatch_route'))}

| 分布 | 新策略 | 平均节省/秒 | 节省比例 | 95%分层配对近似区间/秒 | 更快/相同/更慢 |
|---|---|---:|---:|---|---|
{comparisons}

每种分布按N分层，每层仅两个种子，区间采用分层配对均值和t近似；不是分布无关或多次比较校正后的保证。三种分布由同一组14种子派生，所以不把42个环境当成42个完全独立种子。

![出口评分与任务调度对比](问题4结果/route_analysis/出口评分与任务调度对比.png)

## 4. 收益来自哪部分，额外出口评分是否值得

| 分布 | 策略 | 移动/秒 | 检测/秒 | 切频/秒 | 光学及清除/秒 | 本地运行/秒 |
|---|---|---:|---:|---:|---:|---:|
{components}

运行时间不含事后审计；两批有并行运行时段，因此本机小幅耗时差异不作为严格性能证据。

在相同新调度下，再加入出口评分的边际收益如下（正数表示进一步节省）：

| 分布 | 额外平均节省/秒 | 95%分层配对近似区间/秒 |
|---|---:|---|
{extra}

不能把任务调度带来的改善都归因于出口评分。下一阶段应优先保留有独立比较支持的调度结构，谨慎决定是否值得为额外评分付出计算成本。当前均值路线仍忽略空间不确定性对实际行进与定位的全部影响；论文中应保留其近似性质。

## 5. 验证、原始日志与复现

两阶段共252次新样本整程，清除{a['cleared']+b['cleared']}/{a['sources']+b['sources']}个源，全部完成；共{a['actions']+b['actions']}条动作、{a['absent_certificates']+b['absent_certificates']}个空频道证书通过独立回放与连续凸包审计。原1140002案例用于流程回归，未计入独立样本统计。

本轮新增11项测试，加上既有13项共24项：包含清除成本与终点一致性、8秒界、路线库不依赖情景真值、频道首批差价、强制拒绝新动作时与旧策略逐条一致、稍后顺路处理的反例、切换到另一源先处理的反例、排查站保持完整有序及整程完成检查。

复现命令（在本目录）：

```bash
python3 -m unittest test_q4 test_q4_feedback test_q4_route test_q4_dispatch -v
python3 run_q4_route.py --start 1340000 --label route_holdout
python3 run_q4_route.py --start 1350000 --label route_holdout_b
python3 run_q4_dispatch.py --start 1360000 --label dispatch_holdout
python3 run_q4_dispatch.py --start 1370000 --label dispatch_holdout_b
python3 build_q4_route_report.py
```

原始数据在 `问题4结果/route_holdout*` 和 `问题4结果/dispatch_holdout*`；汇总、图片、代码与日志哈希在 `问题4结果/route_analysis/`。旧版代码和数据保留。官方模拟器演练与正式测试仍未进行。
'''
    write_report('问题4_第三轮续策评分验证.md', report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
