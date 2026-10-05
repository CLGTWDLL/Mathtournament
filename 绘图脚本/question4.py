from common import *
def run():
 p='question4/表格/问题四结果汇总.json';d=read(p);stats=d['three_scenario_paired'];keys=['uniform','outward','cluster'];names=['均匀','边缘朝外','聚集']
 fig,ax=panels(h=3.4);fig.subplots_adjust(left=.13,wspace=.46,bottom=.34)
 for a in ax:a.set_yticks(range(3),names);a.set_ylim(2.5,-.6);grid(a)
 ax[1].set_yticklabels([])
 for i,k in enumerate(keys):
  r=stats[k];x,y=r['dense_mean_virtual_s'],r['sparse_mean_virtual_s'];ax[0].plot([y,x],[i,i],color=G,lw=1.1);ax[0].plot(x,i,'o',mfc='white',mec=B);ax[0].plot(y,i,'s',color=R,ms=4)
  ax[0].annotate(f'{x:.0f}',(x,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8);ax[0].annotate(f'{y:.0f}',(y,i),xytext=(0,-14),textcoords='offset points',ha='center',fontsize=8)
  lo,hi=r['ci95_saved_s'];m=r['mean_saved_s'];ax[1].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=B,capsize=3,ms=4);ax[1].annotate(f'{r["percent_saved"]:.2f}%',(m,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
 ax[0].set_xlim(5300,10900);ax[0].set_xticks([6000,8000,10000]);ax[0].set_xlabel('平均任务时间 / s');ax[1].set_xlim(2350,3650);ax[1].set_xticks([2500,3000,3500]);ax[1].set_xlabel('配对时间节省 / s')
 ax[0].set_title('(a) 密集与稀疏骨架',loc='left',fontsize=10);ax[1].set_title('(b) 节省均值与 95% 区间',loc='left',fontsize=10)
 fig.legend(handles=[symbol('37站密集骨架','o',B,hollow=True),symbol('29站稀疏骨架','s',R)],loc='upper left',bbox_to_anchor=(.07,.17),frameon=False,fontsize=8)
 from matplotlib.container import ErrorbarContainer
 eb=next(c for c in ax[1].containers if isinstance(c,ErrorbarContainer))
 fig.legend(handles=[eb],labels=['节省均值及95%区间'],loc='upper left',bbox_to_anchor=(.56,.17),frameon=False,fontsize=8)
 record(fig,'question4/图2_骨架优化主要收益','三类场景各14组配对。左图空心圆为37站密集骨架，砖红方点为29站稀疏骨架；右图为配对节省均值和数据汇总中的95%区间，上方标注相对平均总时间的节省比例。42/42组稀疏方案更快，两臂各清除546/546个源。该实验同时改变骨架和认证网格；稀疏臂早于当前动态层，不能当作当前方案对同一基线的完整累计收益。',[p])
 pp='question4/数据/实验结果/sparse_backbone/summaries.json';rows=read(pp);fig,ax=panels(1,3.0);a=ax[0];colors=[B,'#84A4B4',G,R];labels=['移动','检测','切频','光学与清除'];x=0
 for k in keys:
  for mode in ['dispatch','sparse']:
   rs=[r for r in rows if r['distribution']==k and r['mode']==mode];assert len(rs)==14
   mean=lambda f:np.mean([f(r) for r in rs]);vals=[mean(lambda r:r['move_m']/5),mean(lambda r:5*r['measurements']),mean(lambda r:r['switches']),mean(lambda r:3*r['clear_attempts']+2*r['cleared'])];total=mean(lambda r:r['virtual_s']);assert abs(sum(vals)-total)<1e-5
   bot=0
   for j,v in enumerate(vals):a.bar(x,v,bottom=bot,color=colors[j],width=.6,label=labels[j] if x==0 else None);bot+=v
   a.text(x,total+130,f'{total:.0f}',ha='center',fontsize=8);x+=1
  x+=.5
 a.set_xticks([0,1,2.5,3.5,5,6],['密集','稀疏']*3);a.set_ylim(0,12200);a.set_ylabel('平均任务时间 / s');a.legend(ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.19),fontsize=8)
 for x,l in zip([.5,3,5.5],names):a.text(x,-.23,l,ha='center',transform=a.get_xaxis_transform())
 fig.subplots_adjust(bottom=.27)
 record(fig,'question4/图3_耗时分解','同一批三类场景骨架对照的平均耗时分解。各柱总高等于移动、检测、切频、光学定位及清除时间之和；小分量不在柱内堆叠标注，精确数值见配套表格。',[pp])
 pairpath='question4/数据/对比40次/20260913-145508-pair40/comparison.json';pairs=read(pairpath)['pairs'];fig,ax=panels(1,3);a=ax[0];grid(a);a.axvline(0,color=G,lw=.8)
 for n in range(10,17):
  rs=[r for r in pairs if r['source_count']==n];a.scatter([r['saved_virtual_s'] for r in rs],n+np.linspace(-.17,.17,len(rs)),color=B,s=19,edgecolors='white',linewidths=.3)
 a.set_yticks(range(10,17));a.set_ylabel('实际源数');a.set_xlabel('静态方案时间 − 动态方案时间 / s');a.set_xlim(-12,300)
 record(fig,'question4/图4_动态层增量收益','40个同种子配对的动态层增量收益，每点为一组配对，同源数内纵向错开以便辨认。平均6328.33→6183.91 s，节省144.42 s（2.28%），95%区间[123.03,165.81] s；35组更快，5组相同且均为16源。动态方案默认配置关闭几何层，更新说明记录该变化不改变这40组动作与任务时间。该批实验与图2的骨架对照不是同一批种子。',[pairpath,'question4/更新说明.md'])
 # Current initial route and actual recorded movement, using each action position.
 bp='question4/代码/动态骨架_v1/backbone_final.json';route=np.vstack([[0,0],read(bp)['route']])
 lp='question4/数据/对比40次/20260913-145508-pair40/dynamic/seed-1480000.json';log=read(lp)
 pts=np.vstack([[0,0],[r['position'] for r in log['logs']]])
 length=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum();assert abs(length-log['summary']['move_m'])<1e-5
 pts=pts[np.r_[True,np.any(np.diff(pts,axis=0)!=0,axis=1)]]
 fig,ax=panels(h=4.0);fig.subplots_adjust(wspace=.65,bottom=.34);mapax(ax[0],'(a) 当前初始骨架');pathplot(ax[0],route)
 mapax(ax[1],'(b) 实际执行轨迹');pathplot(ax[1],pts)
 for row in log['logs']:
  if row['action']=='clear' and row['feedback']['kind']=='success':ax[1].scatter(*row['position'],marker='x',s=22,color=R,zorder=6)
 for a in ax:a.set_xlim(-2400,2400);a.set_ylim(-2400,2400);a.set_xticks([-2000,0,2000]);a.set_yticks([-2000,0,2000])
 fig.legend(handles=[symbol('骨架站点与规划路线','o',B,'-'),symbol('出发点','*','black'),symbol('规划末站','s',B,hollow=True),symbol('目标区域边界',None,G,'--')],loc='upper left',bbox_to_anchor=(.05,.2),ncol=2,frameon=False,fontsize=7.5,columnspacing=.8)
 fig.legend(handles=[symbol('动作位置与实际轨迹','o',B,'-'),symbol('出发点','*','black'),symbol('执行终点','s',B,hollow=True),symbol('成功清除位置','x',R)],loc='upper left',bbox_to_anchor=(.55,.2),ncol=2,frameon=False,fontsize=7.5,columnspacing=.8)
 record(fig,'question4/图1_当前骨架','左图蓝色圆点为规划骨架站点、蓝线为预设访问路线；右图蓝色圆点为动作执行位置、蓝线为实际行进路线，蓝色箭头表示行进方向，这些圆点不是源位置。左：当前默认初始开放骨架，原点后29站，长17505.97 m。右：种子1480000动态方案逐动作位置连接得到的实际轨迹，行进22777.62 m，清除14/14个源，任务时间6466.52 s；砖红叉为成功清除位置。星号为起点，空心方形为终点，虚线圆为目标区域。右图来自开启几何层的配对记录；更新说明确认关闭该层后该批动作与任务时间保持一致。该例用于说明执行形态，性能结论使用完整配对样本。',[bp,lp,'question4/更新说明.md'])

