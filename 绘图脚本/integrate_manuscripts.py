"""Integrate generated figures once, keep immutable first-run source backups."""
from pathlib import Path
import json,re,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'论文图表';BACK=OUT/'原稿备份'
DOCS=['question1/第一问解答.md','question2/融合第二问说明.md','question3/第三问解答.md','question4/第四问解答.md']
rows=json.loads((OUT/'图表清单.json').read_text())['figures'];lookup={r['name']:r for r in rows}
def backup(p):
 q=BACK/p.relative_to(ROOT)
 if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for doc in DOCS:backup(ROOT/doc)
for p in ROOT.glob('question*/图片/**/*.png'):backup(p)

def block(name):
 r=lookup[name];stem=name.split('/')[1];return f'<!-- paper-figure:{name} -->\n![{stem}](../论文图表/{name}.png)\n\n**{stem.replace("_"," ")}。** {r["caption"]}\n<!-- /paper-figure -->'
def clean(s):return re.sub(r'\n*<!-- paper-figure:.*?<!-- /paper-figure -->\n*','\n\n',s,flags=re.S)
def before(s,anchor,names):
 assert anchor in s,anchor
 return s.replace(anchor,'\n\n'.join(block(n) for n in names)+'\n\n'+anchor,1)
# For the first integration, replace original Q1 image references and preserve their location.
p=ROOT/DOCS[0];s=p.read_text()
if '<!-- paper-figure:' not in s:
 mapping={'图一_双点交会定位与直径':'图1_交会定位','图五_第一问求解流程':'图2_算法流程','图二_定位区域直径圆覆盖反例':'图3_三角形反例','图六_两次测向四边形反例':'图4_两次测向反例','图三_多点观测递进收缩':'图5_递进定位','图四_交会角敏感性分析':'图6_交会角敏感性'}
 for old,new in mapping.items():s,n=re.subn(r'!\[[^\]]*\]\(图片/'+old+r'\.png\)',lambda _:block('question1/'+new),s);assert n==1,old
 s=s.replace('图一给出','图1给出').replace('图二左图展示','图3左图展示')
else:
 for name,r in lookup.items():
  if name.startswith('question1/'):
   s=re.sub(r'<!-- paper-figure:'+re.escape(name)+r' -->.*?<!-- /paper-figure -->',lambda _:block(name),s,flags=re.S)
p.write_text(s)
# Add the newly supplied single-policy batch as its own section, not a replacement for the old paired trials.
p=ROOT/DOCS[2];s=p.read_text()
if '### 7.3 新增40次本地运行' not in s:
 note="""### 7.3 新增40次本地运行

新增批次 `batch-20260913-183116-6270521a` 的40个均匀场景使用联合重排策略，版本哈希 `ede0b2c65ab649f3`，与本文件夹当前代码一致。40/40次全部完成，共清除515/515个源；平均任务时间3187.65秒，中位数3164.49秒，标准差206.67秒。每例平均清除时间T/K的算术平均为252.47秒；全部时间除以全部清除数为247.58秒，两者采用不同权重。

日志包含4618个动作，接受联合检测点135次、候选评估345次。本轮绘图核对了版本、动作数、累计移动距离、计时分量与末次反馈时间；完成证书状态来自原日志，本轮未重新运行连续几何证书审计。该批仅含一种策略，用于补充绝对性能与执行过程，不能替代7.1节的独立配对比较，也不是官方模拟器成绩。

配套记录见[新增40次运行结果表](../论文图表/question3/表1_新增四十次运行.md)。

"""
 s=s.replace('## 8. 新理论在定稿中的位置',note+'## 8. 新理论在定稿中的位置');p.write_text(s)
for doc,inserts in [(DOCS[1],[('## 5. 如何衡量第二次检测后的准确度',['question2/图1_保证接收候选域']),('### 11.3 与旧推荐点的时长对比',['question2/图2_时间与精度'])]),(DOCS[2],[('## 5. 单源定位、清除与有限后备',['question3/图1_覆盖与反馈机制']),('## 8. 新理论在定稿中的位置',['question3/图2_四十次运行结果','question3/图3_实际轨迹与反馈','question3/图4_平均耗时分解'])]),(DOCS[3],[('### 6.2 局部决策',['question4/图1_当前骨架']),('### 7.2 本次更新：默认算法加入动态层',['question4/图2_骨架优化主要收益','question4/图3_耗时分解']),('### 7.3 限制',['question4/图4_动态层增量收益'])])]:
 p=ROOT/doc;s=clean(p.read_text())
 for anchor,names in inserts:s=before(s,anchor,names)
 p.write_text(s)
# Link publication tables without duplicating existing analytical content.
for doc,table in [(DOCS[0],'question1/表1_区域与覆盖判定.md'),(DOCS[1],'question2/表1_推荐方案.md'),(DOCS[3],'question4/表3_CPU串行复核.md')]:
 p=ROOT/doc;s=p.read_text();line=f'配套排版表：[三线表数据](../论文图表/{table})。'
 if line not in s:s+='\n\n'+line+'\n'
 p.write_text(s)
print('Integrated 16 figures and captions; original manuscripts and 16 PNGs backed up')
