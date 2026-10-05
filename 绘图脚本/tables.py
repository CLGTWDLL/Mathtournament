"""Publication tables as Markdown plus semantic HTML (three horizontal rules)."""
from common import *
import html

def emit(q,name,headers,rows,note):
 p=OUT/f'question{q}';p.mkdir(exist_ok=True)
 md=f'# {name}\n\n'+ '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |\n'
 md+=''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)+f'\n{note}\n';(p/(name+'.md')).write_text(md)
 table='<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'
 table+=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
 return '<section><h2>'+name+'</h2>'+table+'<p>'+html.escape(note)+'</p></section>'
def run():
 sections=[];d=read('question1/数据/全部求解结果.json');rows=[]
 for k,v in d.items():
  dia=v['diameter_m'];r=v['enclosing_radius_m'];finite=dia is not None and r is not None
  rows.append([k, v['status'],f'{dia:.2f}' if finite else ('∞' if v['status']=='无界' else '未定义'),f'{r:.2f}' if finite else '—',f'{max(0,(r-dia/2)*1000):.2f}' if finite else '—','是' if v['diameter_circle_covers'] is True else ('否' if v['diameter_circle_covers'] is False else '—')])
 sections.append(emit(1,'表1_区域与覆盖判定',['算例','状态','D / m','R / m','R−D/2 / mm','可覆盖'],rows,'一般距离保留两位；覆盖余量另列毫米单位，避免四边形反例的2.66 mm差异被舍入掩盖。数值容差内的负零显示为零；判定保留原始完整精度结果。'))
 d=read('question2/数据/独立复评结果.json')['rows'];rows=[]
 for r in d:
  t=r['independent_rollout'];lo,hi=t['ci95_s'];rows.append([r['name'],f"{t['mean_s']:.2f}",f'[{lo:.2f}, {hi:.2f}]',f"{r['continuous_diameter_upper_m']:.2f}"])
 sections.append(emit(2,'表1_推荐方案',['方案','完整耗时 / s','均值95%区间 / s','连续直径上界 / m'],rows,'同一批512个独立样本复评。坐标和有限样本直径保留在原参数表，正文主表集中呈现时间—精度权衡。'))
 d=read('question4/表格/问题四结果汇总.json')['three_scenario_paired'];rows=[]
 for k,l in [('uniform','均匀'),('outward','边缘朝外'),('cluster','聚集')]:
  r=d[k];rows.append([l,f"{r['dense_mean_virtual_s']:.2f}",f"{r['sparse_mean_virtual_s']:.2f}",f"{r['percent_saved']:.2f}%",'14/14'])
 sections.append(emit(4,'表1_骨架主要收益',['场景','密集 / s','稀疏 / s','节省比例','更快组数'],rows,'每类14组配对，两臂各清除182/182个源。该实验早于动态层，不与40局动态实验合并计算累计节省。'))
 data=read('question4/数据/实验结果/sparse_backbone/summaries.json');rows=[]
 for k,l in [('uniform','均匀'),('outward','边缘朝外'),('cluster','聚集')]:
  for m,ml in [('dispatch','密集'),('sparse','稀疏')]:
   rs=[r for r in data if r['distribution']==k and r['mode']==m];vals=[np.mean([r['move_m']/5 for r in rs]),np.mean([r['measurements']*5 for r in rs]),np.mean([r['switches'] for r in rs]),np.mean([3*r['clear_attempts']+2*r['cleared'] for r in rs])];rows.append([l,ml]+[f'{v:.2f}' for v in vals]+[f'{sum(vals):.2f}'])
 sections.append(emit(4,'表2_耗时分解',['场景','骨架','移动 / s','检测 / s','切频 / s','光学与清除 / s','合计 / s'],rows,'各分量由逐例记录计算后取平均；显示值因舍入可能有0.01 s差异。'))
 cp='question4/数据/对比40次/20260913-145508-pair40/cpu_ab_sequential.json';d=read(cp);rows=[[r['seed'],r['source_count'],f"{r['base_cpu_s']:.3f}",f"{r['new_cpu_s']:.3f}"] for r in d['rows']];rows.append(['均值','—',f"{d['mean_base_cpu_s']:.3f}",f"{d['mean_new_cpu_s']:.3f}"])
 sections.append(emit(4,'表3_CPU串行复核',['种子','源数','静态CPU / s','动态CPU / s'],rows,'当前默认配置关闭几何层。仅4个种子的同进程串行复核，不是40局CPU统计；不据此作普遍等效结论。'))
 css='body{font-family:"Songti SC","SimSun",serif;color:#000;max-width:960px;margin:40px auto;padding:0 24px}section{margin-bottom:50px}h2{font-size:18px}table{border-collapse:collapse;width:100%;border-top:1.5px solid black;border-bottom:1.5px solid black;font-size:15px;font-variant-numeric:tabular-nums}th{border-bottom:1px solid black}td,th{text-align:right;padding:10px 8px}td:first-child,th:first-child{text-align:left}p{font-size:14px;line-height:1.7}@media print{section{break-inside:avoid}body{margin:0}}'
 (OUT/'表格排版预览.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>论文三线表</title><style>'+css+'</style>'+''.join(sections)+'</html>')
