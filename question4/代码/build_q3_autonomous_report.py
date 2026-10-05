"""Generate the complete-task local benchmark report from retained logs."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent.parent/'question3'/'数据'/'实验结果'/'autonomous'
MODES=['sequential','discover_first','joint']
NAMES={'sequential':'逐源完成','discover_first':'先发现后清除','joint':'联合调度'}


def read_folder(label):
    return [json.loads(p.read_text()) for p in sorted((OUT/label).glob('uniform_*_*.json'))]


def stratified_interval(values,counts):
    groups=[np.asarray(values)[np.asarray(counts)==n] for n in range(10,17)]
    if min(map(len,groups))<2:return [None,None]
    components=np.array([g.var(ddof=1)/len(g)/49 for g in groups])
    variance=components.sum()
    df=variance**2/sum(v*v/(len(g)-1) for v,g in zip(components,groups)) if variance else 1000
    mean=float(np.mean([g.mean() for g in groups]))
    margin=float(student_t.ppf(.975,df)*np.sqrt(variance))
    return [mean-margin,mean+margin]


def build():
    data=read_folder('final')
    if not data:raise ValueError('No final evaluations')
    versions={r['version'] for r in data}
    if len(versions)!=1:raise ValueError('Do not pool different implementation versions')
    cases=sorted({r['summary']['seed'] for r in data})
    lookup={(r['summary']['seed'],r['summary']['mode']):r for r in data}
    if len(lookup)!=3*len(cases):raise ValueError('Incomplete paired comparison')
    aggregates={}
    for mode in MODES:
        rows=[lookup[s,mode]['summary'] for s in cases]
        aggregate=dict(cases=len(rows),complete=sum(r['complete'] for r in rows),
                       source_count=sum(r['source_count'] for r in rows),cleared=sum(r['cleared'] for r in rows))
        for key in ('virtual_s','average_per_cleared_s','move_s','measurement_s','switch_s','clear_s',
                    'measurements','clear_attempts','wall_s','opportunistic_scans','target_switches'):
            aggregate[key]=float(np.mean([r[key] for r in rows]))
        aggregate['max_wall_s']=max(r['wall_s'] for r in rows)
        aggregate['max_virtual_s']=max(r['virtual_s'] for r in rows)
        aggregate['fallback_cases']=sum(bool(r['numerical_fallbacks']) for r in rows)
        tails=[]
        for s in cases:
            log=lookup[s,mode]['logs']
            last=max((a['feedback']['virtual_time_s'] for a in log if a['feedback']['kind']=='success'),default=0)
            tails.append(lookup[s,mode]['summary']['virtual_s']-last)
        aggregate['mean_certification_tail_s']=float(np.mean(tails))
        aggregate['mean_extra_task_resumptions']=float(np.mean([
            max(0,r['target_switches']-r['source_count']+1) for r in rows]))
        aggregates[mode]=aggregate
    differences={}
    counts=[lookup[s,'joint']['summary']['source_count'] for s in cases]
    for baseline in MODES[:2]:
        d=np.array([lookup[s,baseline]['summary']['virtual_s']-lookup[s,'joint']['summary']['virtual_s'] for s in cases])
        differences[baseline]=dict(mean_saved_s=float(d.mean()),ci95=stratified_interval(d,counts),
                                   ratio_of_mean_times_saved=float(d.mean()/aggregates[baseline]['virtual_s']),
                                   joint_faster_cases=int((d>0).sum()),cases=len(cases))
    stress={}
    for distribution in ('cluster','edge'):
        rows=[json.loads(p.read_text())['summary'] for p in (OUT/f'stress_{distribution}').glob(f'{distribution}_*_joint.json')]
        if rows:
            stress[distribution]=dict(cases=len(rows),complete=sum(r['complete'] for r in rows),
                                      cleared=sum(r['cleared'] for r in rows),sources=sum(r['source_count'] for r in rows),
                                      mean_s=float(np.mean([r['virtual_s'] for r in rows])),
                                      max_s=max(r['virtual_s'] for r in rows),max_wall_s=max(r['wall_s'] for r in rows),
                                      fallback_cases=sum(bool(r['numerical_fallbacks']) for r in rows))
    by_count={str(n):{m:float(np.mean([lookup[s,m]['summary']['virtual_s'] for s in cases
                                      if lookup[s,m]['summary']['source_count']==n])) for m in MODES}
              for n in range(10,17)}
    stats=dict(implementation_version=versions.pop(),aggregates=aggregates,differences=differences,
               stress=stress,by_source_count=by_count)
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    font=Path('/System/Library/Fonts/STHeiti Light.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    # The displayed case is selected by median paired improvement, not best gain.
    chosen=sorted(cases,key=lambda s:lookup[s,'discover_first']['summary']['virtual_s']-lookup[s,'joint']['summary']['virtual_s'])[len(cases)//2]
    fig,axes=plt.subplots(1,3,figsize=(15,5.6))
    for mode,ax in zip(MODES,axes):
        run=lookup[chosen,mode];rows=run['logs'];s=run['summary']
        route=np.vstack((np.zeros(2),np.array([a['position'] for a in rows])))
        ax.plot(route[:,0],route[:,1],color='#3a75a7',lw=.9,alpha=.7)
        ax.plot(0,0,'s',color='#283746',ms=5)
        for source in run['truth']['sources'].values():
            ax.plot(*source['position'],'x',color='#b44332',ms=6,mew=1.3)
        successful=np.array([a['position'] for a in rows if a['feedback']['kind']=='success'])
        if len(successful):ax.scatter(successful[:,0],successful[:,1],s=30,facecolors='none',edgecolors='#2e846e',lw=1)
        ax.add_patch(Circle((0,0),1800,fill=False,color='#9aa5ae',lw=.8))
        ax.set(xlim=(-2050,2050),ylim=(-2050,2050),aspect='equal',xlabel='东向 x（米）',ylabel='北向 y（米）')
        ax.set_title(f"{NAMES[mode]}\n{s['virtual_s']:.0f}秒 · 清除{s['cleared']}/{s['source_count']}",fontsize=12)
        ax.tick_params(labelsize=9);ax.grid(alpha=.1)
    fig.suptitle(f'同一环境的完整任务路线（案例 {chosen}）',fontsize=16)
    fig.text(.5,.025,'方块：原点；红叉：仅用于评价的源真值；空心圆：成功清除位置；折线：实际行程',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.12,1,.94])
    fig.savefig(OUT/'完整任务路线对比.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    keys=['move_s','measurement_s','switch_s','clear_s'];labels=['移动','检测','切频','光学定位与清除'];colors=['#467aa5','#e3a04c','#987ba7','#5a9c88']
    bottom=np.zeros(3)
    for key,label,color in zip(keys,labels,colors):
        values=np.array([aggregates[m][key] for m in MODES])
        axes[0].bar(np.arange(3),values,bottom=bottom,label=label,color=color,width=.6)
        bottom+=values
    for i,value in enumerate(bottom):axes[0].text(i,value+60,f'{value:.0f}',ha='center',fontsize=10)
    axes[0].set(xticks=range(3),xticklabels=[NAMES[m] for m in MODES],ylabel='平均任务耗时（秒）',ylim=(0,bottom.max()*1.16),title='平均时间的组成')
    axes[0].legend(fontsize=9)
    for seed in cases:
        axes[1].plot(range(3),[lookup[seed,m]['summary']['virtual_s'] for m in MODES],'-o',lw=.7,ms=3,alpha=.38,color='#57758e')
    axes[1].plot(range(3),[aggregates[m]['virtual_s'] for m in MODES],'-o',lw=2,ms=6,color='#b44332',label='均值')
    axes[1].set(xticks=range(3),xticklabels=[NAMES[m] for m in MODES],ylabel='单案例任务耗时（秒）',title=f'{len(cases)}个配对案例')
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.15)
    fig.tight_layout();fig.savefig(OUT/'完整任务耗时对比.png',dpi=160);plt.close(fig)
    selection=json.loads((OUT/'parameter_selection.json').read_text())
    table='\n'.join(f"| {NAMES[m]} | {a['complete']}/{a['cases']} | {a['cleared']}/{a['source_count']} | {a['virtual_s']:.2f} | {a['average_per_cleared_s']:.2f} | {a['wall_s']:.2f} |" for m,a in aggregates.items())
    comparisons='\n'.join(f"- 相对{NAMES[m]}，联合策略平均节省{d['mean_saved_s']:.2f}秒，分层配对95%区间[{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]秒；均值耗时降低{d['ratio_of_mean_times_saved']:.2%}，{d['joint_faster_cases']}/{d['cases']}个案例更快。" for m,d in differences.items())
    stress_text='\n'.join(f"| {'局部聚集' if k=='cluster' else '靠近边界'} | {v['complete']}/{v['cases']} | {v['cleared']}/{v['sources']} | {v['mean_s']:.2f} | {v['max_s']:.2f} |" for k,v in stress.items())
    count_table='\n'.join(f"| {n} | {v['sequential']:.1f} | {v['discover_first']:.1f} | {v['joint']:.1f} |" for n,v in by_count.items())
    report=fr'''# 第三问：自主搜索、定位清除与本地完整任务测试

已完成可运行的自动策略，能够从原点、初始频道1开始，仅根据公开反馈完成发现、定位、清除和无遗漏确认。本报告包含本地模拟实验；没有连接官方模拟器，也没有进行正式测试。该策略是受后验驱动的近似策略，不是连续空间全局最优解。

## 1. 状态、动作与停止条件

一个全局机器人状态保存实际位置和测向机频道；20个频道分别保存全部观测、条件位置/半径后验、是否确认有源、是否清除及几何排查进度。源数后验仍受10—16约束。处理另一源后恢复当前源时，所有移动费用从机器狗的实际新位置计算。

动作只有指定位置和频道的检测或清除。每收到一条`accepted=true`反馈才更新状态，清除不改变测向机频道。全过程核对

$$T=L/5+5N_m+N_{{switch}}+3N_c+2N_{{success}}.$$

只有每个频道都已清除或确定不存在源时停止。不存在证明来自完整连续可行集被排除、七个保证覆盖位置均测得阴性，或已经确认16个源。已清除的源也计入原始确认源数。小概率、空粒子集及后验均值都不是完成证书。

## 2. 局部定位与清除动作怎样选择

沿用已验证的二维自适应后验和解析半径区间。对已确认源，独立于环境真值抽取{data[0]['summary']['particles']}个后验位置样本，按各位置允许的半径区间抽半径，并预测新位置的测角反馈。

测量候选由后验均值、沿接近方向的四分位附近位置、横向偏移及沿途位置构成；第一次后续定位还加入第二问的半解析候选，但以完整历史后验重新评价，不把它直接当作最优点。清除候选包括当前位置、后验均值和最近的保守覆盖方格中心。

每个候选动作执行后，使用“对剩余连续可行方格逐次就近清除”这一明确续策估计完成当前源的耗时。收到真实反馈后重新计算下一步。因此属于单步情景评价和反复决策，没有展开完整多步决策树，也没有简单把每个源的独立最优成本相加。

每个源确认后最多再测5次。测量相对当前候选的样本平均节省，扣除两倍配对标准误后超过1秒，才接受；该门槛是启发式筛选，不是搜索全部候选后的统一置信保证。至多允许两次非保底的后验清除尝试。所有数值估计均不能取代实际成功反馈。

## 3. 保底清除与数值异常处理

使用边长25米的固定方格保守覆盖源的连续可能区域，每格外接半径$25/\sqrt2<20$。若选择一格中心清除且失败，该整格必被排除，故失败保底动作严格减少有限格数。`near`后直接到提示点清除；若整个保守后验支持已在某个20米圆内，可直接到圆心清除。

若后验积分失败或数值证据与计数模型冲突，保留全部公开历史，停用概率调度，转入几何覆盖清除与固定发现排查。该分支不会把数值失败解释为不存在源。原始可能格数有限，未知频道的站点任务有限，额外扫描和测量也有上限，因此理想规则下可有限步结束；现实20分钟和虚拟100小时限制另由运行器检查。

## 4. 三种调度方式

| 策略 | 行为 |
| --- | --- |
| 逐源完成 | 按预设发现站及频道顺序排查；一旦发现源，就完成它，再返回尚未执行的排查任务。 |
| 先发现后清除 | 完成所有频道的存在性排查，跳过已确认有源的频道；再逐个定位清除，选择离当前保守可行区域较近的任务。 |
| 联合调度 | 在实际停靠点比较是否顺带检测未知频道；每次反馈后可重选已知源，对最近的三个源做局部耗时评价；没有已知待处理源时选择下一个发现站。 |

联合策略的顺带扫描估值为

$$B_c=\lambda\,p_{{recv}}(s,c)\left(6k_c+\frac{{\widehat L_{{remaining}}}}{{5u}}\right)-(5+\mathbf1_{{c\ne h}}).$$

$k_c$为该频道尚未访问的七站任务数，$u$为未解决频道数，$\widehat L_{{remaining}}$为以近邻路线估计的剩余发现站行程。仅当最大$B_c>0$时在当前位置顺带扫描；已测位置不重复检测，额外扫描上限为100次。这里是对发现后可能省去排查服务的启发式估值，不是精确的信息价值。它没有把“检测到源”直接等同于“清除源”。

无已知源待处理时，发现站按“行程与扫描费用 / 预期接收数及排查进度”排序，其中每个尚未测过的频道贡献$1/7$的排查进度。该项鼓励完成无源频道的验证，即使预期阳性概率已经很低。

收尾阶段还加入一条保守的后验换点规则：已确认至少10个源，且剩余频道的后验期望源数不超过0.5时，尝试寻找能把所有剩余连续位置支持纳入999.5米范围的共同检测点。先检查支持包围盒中心是否满足条件，再沿当前位置方向移动至仍满足全部距离约束的位置。只有该点比常规下一站近至少5米才采用，并在那里逐频道完成一批存在性检测。阳性确认新源，阴性给出不存在证明；0.5阈值只用于触发换点，绝不用于宣布无源。不能满足这些几何条件时仍使用原发现站。

这一收尾点可沿给定方向解析计算。令$a_j$为所有剩余可能方格的顶点，$q_0$为包围盒中心，$v$为由$q_0$指向当前位置$s$的单位向量，$r_0=999.5$。当$q_0$已满足所有距离约束时，取

$$t_* = \min\left\{{\|s-q_0\|,\ \min_j\left[-v\cdot(q_0-a_j)+\sqrt{{(v\cdot(q_0-a_j))^2+r_0^2-\|q_0-a_j\|^2}}\right]\right\}},\qquad q=q_0+t_*v.$$

这是在该线段上最靠近当前位置的可行点，不是对整个平面最近点的全局求解。它覆盖全部保守方格顶点，进而覆盖方格整体，留出0.5米的接收余量。

## 5. 调参与独立评价

开发阶段发现，无条件积极顺带扫描会增加大量检测，有时比先发现后清除更慢。因此调整扫描估值系数$\lambda$，候选值和各自调参结果保存在`parameter_selection.json`，选择$\lambda={selection['selected_scale']}$。调参环境种子从{selection['training_start']}开始。之后用已查看的开发案例检验收尾换点规则，并固定沿用该系数，未对新增规则重新寻找最优系数。独立最终评价从970000开始；930000、940000系列的开发记录和调参结果均不混入下表。

最终测试按原始源数10—16分层，每层3个独立环境，共21个配对案例。每个案例的三个策略使用相同真实源和固定位置误差，规划器的随机种子固定为9101，不接收环境随机种子或真值。源位置按面积均匀、半径均匀、不同位置测角误差均匀的生成假设与后验模型一致。

| 策略 | 完成案例 | 清除源数/总数 | 平均总时间/秒 | 平均单案例每源时间/秒 | 平均本地计算时间/秒 |
| --- | --- | --- | ---: | ---: | ---: |
{table}

“平均单案例每源时间”先按每次测试$T/N_{{cleared}}$计算，再对案例取平均。原始记录保留每个案例的清除比例与分项计时，没有仅对成功案例筛选计算。

{comparisons}

区间根据10—16七个源数层内的配对差计算均值方差，采用Welch–Satterthwaite自由度的$t$近似，避免把固定分层样本当成随意增加的独立观测。区间跨零时不能声称具有明确的平均优势。

![完整任务耗时](完整任务耗时对比.png)

![完整任务路线](完整任务路线对比.png)

路线图选择联合策略相对先发现后清除的收益处于中位的案例{chosen}，没有挑选收益最大的案例。图中的真值只供评价，不是策略输入。

按原始源数分层的平均总时间如下，每格均来自3个环境：

| 源数 | 逐源完成/秒 | 先发现后清除/秒 | 联合调度/秒 |
| --- | ---: | ---: | ---: |
{count_table}

源数越多不必然要求更多排查时间：一旦确认16个源，其他频道可直接由上界排除；不足16个时还要证明剩余频道没有遗漏。联合策略最后一次清除后的平均排查耗时为{aggregates['joint']['mean_certification_tail_s']:.2f}秒，这部分已计入任务总时间。

## 6. 分布偏离时的完整性检查

另外保持在线后验假设不变，生成源集中在局部区域及源集中于1650—1800米边缘带的环境，每种按源数10—16各测一次。它们用于暴露先验不匹配时的问题，不与均匀环境的速度指标混算。

| 环境 | 完成案例 | 清除源数/总数 | 平均任务时间/秒 | 最慢任务时间/秒 |
| --- | --- | --- | ---: | ---: |
{stress_text}

这些有限实验不是对所有允许环境的性能证明。算法的完成依据仍是公开成功反馈与几何证书，而不是概率校准成立。各模式的数值保底次数、最慢现实时间及最后一次清除后的排查耗时见`statistics.json`。

## 7. 文件与复现

- `q3_policy.py`：在线自主策略，不读取源真值。
- `run_q3_autonomous.py`：本地传感器执行、逐动作日志、时间限制与完整性核对。
- `test_q3_policy.py`：状态更新、频道计时、保底进度、数值失败和完成条件验证。
- `final/uniform_*_*.json`：21个案例、三种策略的完整动作与反馈，共63份记录。
- `stress_cluster/`、`stress_edge/`：分布偏离检查记录。
- `statistics.json`：汇总、分层配对区间及分项统计。
- `independent_audit.json`：独立回放每条传感器反馈与计时，并用另一个圆覆盖检查器验证无源证书。

在`B题/solution`目录执行：

```bash
python3 -m unittest -v test_q3_belief.py test_q3_policy.py
python3 run_q3_autonomous.py --start 970000 --per-count 3 --label final --scan-scale {selection['selected_scale']}
python3 run_q3_autonomous.py --start 950000 --per-count 1 --label stress_cluster --distribution cluster --modes joint --scan-scale {selection['selected_scale']}
python3 run_q3_autonomous.py --start 960000 --per-count 1 --label stress_edge --distribution edge --modes joint --scan-scale {selection['selected_scale']}
python3 build_q3_autonomous_report.py
python3 audit_q3_autonomous.py
```

第三问的自动算法和本地完整任务验证现已具备。官方通信适配、官方演练及三次正式日志仍未开展；本地记录不冒充正式日志。

本版19项单元与几何验证通过；日志的独立逐动作回放和圆覆盖证明检查另见`independent_audit.json`。
'''
    (OUT/'自主策略实验报告.md').write_text(report,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
