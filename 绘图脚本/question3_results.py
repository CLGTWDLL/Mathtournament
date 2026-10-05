"""Read and validate the user-supplied 40-run batch; never simulate new trials."""
from common import *
from tables import emit
BATCH='question3/数据/本地40次运行/batch-20260913-183116-6270521a'
def run():
 batch=read(BATCH+'/batch_summary.json');rows=batch['records'];assert len(rows)==40
 files=['q3_joint_replan.py','q3_joint_site.py','q3_backbone.py','q3_policy.py','q3_belief.py','q3_environment.py','run_q3_autonomous.py','run_q3_simulator_40.py']
 version=hashlib.sha256(b''.join(source('question3/代码/'+f).read_bytes() for f in files)).hexdigest()[:16]
 assert version==batch['version'],(version,batch['version'])
 episodes=[]
 for row in rows:
  p=BATCH+'/'+row['full_record_file'];d=read(p);s=d['summary'];assert d['version']==version
  assert s['complete'] and s['certificate'] and s['cleared']==s['source_count'];assert len(d['logs'])==s['actions']
  pts=np.vstack([[0,0],[r['position'] for r in d['logs']]]);dist=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum();assert abs(dist-s['move_m'])<1e-6
  assert abs(s['move_s']+s['measurement_s']+s['switch_s']+s['clear_s']-s['virtual_s'])<1e-6
  assert abs(3*s['clear_attempts']+2*s['cleared']-s['clear_s'])<1e-6
  assert abs(d['logs'][-1]['feedback']['virtual_time_s']-s['virtual_s'])<1e-6
  assert abs(s['virtual_s']-row['virtual_s'])<1e-6
  episodes.append((p,d))
 assert abs(np.mean([r['virtual_s'] for r in rows])-batch['mean_virtual_s'])<1e-7
 # Absolute results, no fabricated comparison arm.
 fig,ax=panels(h=3.0)
 for j,a in enumerate(ax):
  grid(a);a.set_yticks(range(10,17));a.set_ylabel('实际源数' if j==0 else '')
  for n in range(10,17):
   rs=[r for r in rows if r['source_count']==n];y=n+np.linspace(-.17,.17,len(rs));values=[r['virtual_s' if j==0 else 'average_per_cleared_s'] for r in rs];a.scatter(values,y,s=17,color=B if j==0 else R,edgecolors='white',linewidths=.3)
  a.set_title(['(a) 完整任务时间','(b) 每例平均清除时间'][j],loc='left',fontsize=10);a.set_xlabel(['任务时间 / s','任务时间 / 清除源数 / s'][j])
 total=sum(r['cleared'] for r in rows);per=np.mean([r['average_per_cleared_s'] for r in rows])
 record(fig,'question3/图2_四十次运行结果',f'新增40次本地均匀场景运行，版本{version}。每点为一次运行，同源数内纵向错开。40/40次全部完成，共清除{total}/{total}个源。平均任务时间{batch["mean_virtual_s"]:.2f} s，中位数{batch["median_virtual_s"]:.2f} s；每例T/K的算术平均为{per:.2f} s，全部时间除以全部清除数为{sum(r["virtual_s"] for r in rows)/total:.2f} s。该批仅包含联合重排策略，不能独立计算相对其他策略的提速。',[BATCH+'/batch_summary.json'])
 # Pick the lower of the two central ranked runs; deterministic, not a best-case selection.
 median=np.median([r['virtual_s'] for r in rows]);p,d=min(episodes,key=lambda pd:(abs(pd[1]['summary']['virtual_s']-median),pd[1]['summary']['virtual_s']));s=d['summary'];logs=d['logs']
 pts=np.vstack([[0,0],[r['position'] for r in logs]]);pts=pts[np.r_[True,np.any(np.diff(pts,axis=0)!=0,axis=1)]]
 fig,ax=panels(h=4.2);fig.subplots_adjust(bottom=.38);mapax(ax[0],'(a) 实际轨迹与联合检测点');pathplot(ax[0],pts)
 for r in logs:
  if r['action']=='clear' and r['feedback']['kind']=='success':ax[0].scatter(*r['position'],marker='x',s=23,color=R,zorder=6)
 for r in d['joint_history']:
  old=np.array(r['old_station']);new=np.array(r['position']);ax[0].scatter(*new,marker='D',s=25,facecolors='white',edgecolors=B,zorder=7);ax[0].annotate('',new,old,arrowprops=dict(arrowstyle='->',lw=.8,color=R))
 ax[0].set_xlim(-2200,2200);ax[0].set_ylim(-2200,2200);ax[0].set_xticks([-2000,0,2000]);ax[0].set_yticks([-2000,0,2000])
 t=[0]+[r['feedback']['virtual_time_s'] for r in logs]
 for key,col,ls,label in [('confirmed',B,'-','已确认源'),('cleared',R,'--','已清除源'),('absent',G,':','已排除频道')]:ax[1].step(t,[0]+[r[key] for r in logs],where='post',color=col,ls=ls,label=label,lw=1.1)
 ax[1].set_xlabel('累计任务时间 / s');ax[1].set_ylabel('数量');ax[1].set_title('(b) 反馈与任务完成进度',loc='left',fontsize=10);fig.legend(*ax[1].get_legend_handles_labels(),frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(.61,.24));ax[1].set_ylim(0,20);ax[1].set_yticks([0,5,10,15,20])
 fig.legend(handles=[symbol('动作位置与轨迹','o',B,'-'),symbol('出发点','*','black'),symbol('任务终点','s',B,hollow=True),symbol('成功清除位置','x',R),symbol('联合检测点','D',B,hollow=True),symbol('目标区域边界',None,G,'--'),symbol('原站→联合点','>',R,'-')],ncol=2,loc='upper left',bbox_to_anchor=(.04,.24),frameon=False,fontsize=8,columnspacing=1)
 record(fig,'question3/图3_实际轨迹与反馈',f'按任务时间距样本中位数最近的规则选择种子{s["seed"]}，若并列取时间较短者，非择优展示。该例任务时间{s["virtual_s"]:.2f} s，清除{s["cleared"]}/{s["source_count"]}个源。左图蓝色圆点为动作执行位置、蓝线为行进轨迹（蓝色箭头表示行进方向），并非源的位置；砖红叉为成功清除位置，空心菱形为接受的联合检测点，砖红箭头从原站指向联合点；星号为出发点，空心方形为终点。右图按逐动作反馈画阶梯曲线，已排除频道数与已清除源数含义不同。',[p])
 fig,ax=panels(1,2.8);a=ax[0];fig.subplots_adjust(left=.23);names=['移动','检测','切频','光学定位','激光清除'];values=[np.mean([r['move_s'] for r in rows]),np.mean([r['measurement_s'] for r in rows]),np.mean([r['switch_s'] for r in rows]),np.mean([3*r['clear_attempts'] for r in rows]),np.mean([2*r['cleared'] for r in rows])]
 a.barh(range(5),values,color=[B,B,G,R,R],height=.5);a.set_yticks(range(5),names);a.invert_yaxis();a.set_xlabel('平均耗时 / s');a.set_xlim(0,max(values)*1.25)
 for i,v in enumerate(values):a.text(v+30,i,f'{v:.2f}',va='center',fontsize=8)
 record(fig,'question3/图4_平均耗时分解','新增40次运行的平均任务耗时分解。光学定位时间包含失败尝试，激光清除仅按成功清除计时；五项未舍入均值之和等于3187.65 s。墙钟计算时间不属于题目任务时间，未混入本图。',[BATCH+'/batch_summary.json'])
 table_rows=[['运行数','40'],['全部完成','40/40'],['清除源数',f'{total}/{total}'],['平均任务时间 / s',f'{batch["mean_virtual_s"]:.2f}'],['中位任务时间 / s',f'{batch["median_virtual_s"]:.2f}'],['标准差 / s',f'{batch["stdev_virtual_s"]:.2f}'],['每例T/K的平均 / s',f'{per:.2f}'],['接受联合点 / 候选评估',f'{sum(r["joint_accepted"] for r in rows)}/{sum(r["joint_trials"] for r in rows)}']]
 html=emit(3,'表1_新增四十次运行',['指标','结果'],table_rows,f'单策略本地批次，版本{version}；不与正文21例配对实验混合。')
 (OUT/'question3/数据核对.json').write_text(json.dumps({'version':version,'matches_current_code':True,'runs':40,'sources':total,'actions':sum(r['actions'] for r in rows),'representative_seed':s['seed'],'checks':['version hash','action count','route distance','time components','final feedback time','batch and per-run summary'],'certificate_scope':'读取原日志certificate及audit_verified；本轮未重新运行连续几何证书审计'},ensure_ascii=False,indent=2))
 return html
