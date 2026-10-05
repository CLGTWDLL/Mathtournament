from common import *
import sys
sys.path.insert(0,str(ROOT/'question1/代码'))
from examples import cases
from geometry import solve
DATA='question1/数据/全部求解结果.json'
def run():
 d=read(DATA);cs=cases();source('question1/代码/examples.py');source('question1/代码/geometry.py')
 def geom(name,key):
  r=d[key];fig,ax=panels()
  for a in ax:mapax(a,'')
  for i,o in enumerate(cs[key]):
   p=np.array([o['x'],o['y']]);ang=np.radians(o['angle_deg']);length=1000 if '正交' in key else 1300
   vv=[p]+[p+length*np.array([np.cos(ang+e),np.sin(ang+e)]) for e in np.radians([-1,1])]
   poly(ax[0],vv,[B,R][i]);ax[0].scatter(*p,s=20,color=[B,R][i]);ax[0].annotate(f'S{i+1}',p,xytext=(-5,9) if '四边形' in key and i==0 else (5,7),ha='right' if '四边形' in key and i==0 else 'left',textcoords='offset points')
  poly(ax[0],r['vertices']);poly(ax[1],r['vertices']);v=np.array(r['vertices']);c=np.array(r['enclosing_center']);rr=r['enclosing_radius_m']
  ax[1].add_patch(Circle(c,rr,fill=False,color=B,lw=1));ends=np.array(r['diameter_endpoints']);ax[1].plot(*ends.T,color=R,lw=1.1)
  ax[0].set_title('(a) 测向约束',loc='left',fontsize=10);ax[1].set_title('(b) 定位区域与包围圆',loc='left',fontsize=10)
  ax[1].set_xlim(c[0]-rr*1.35,c[0]+rr*1.35);ax[1].set_ylim(c[1]-rr*1.35,c[1]+rr*1.35)
  if '四边形' in key:
   gap=(rr-r['diameter_m']/2)*1000
   ax[1].add_patch(Circle(ends.mean(0),r['diameter_m']/2,fill=False,color=R,ls='--',lw=.8))
   cap=f'两次测向形成的四边形反例。蓝线为最小包围圆，红色虚线为以最长对角线为直径的圆；D={r["diameter_m"]:.6f} m，R={rr:.6f} m，R−D/2={gap:.2f} mm。两圆非常接近，覆盖失败以数值判据确认，不能仅靠视觉判断。'
  else:cap=f'正交双点测向算例。检测点为(−700,0)、(0,−700)，误差半角为1°；右图放大左图原点附近的交集。区域直径D={r["diameter_m"]:.2f} m，最小包围圆半径R={rr:.2f} m。红线为直径端点连线。'
  record(fig,'question1/'+name,cap,[DATA,'question1/代码/examples.py'])
 geom('图1_交会定位','正交双点定位');geom('图4_两次测向反例','两次测向四边形反例')
 # Accepted geometric style, explanatory sentence moved to the manuscript caption for consistent publication layout.
 fig,ax=panels(h=3.0);v=np.array([[-20,0],[20,0],[0,20*np.sqrt(3)]])
 for a in ax:
  a.axis('off');a.set_aspect('equal');a.set_xlim(-27,27);a.set_ylim(-23,41);poly(a,v)
  a.scatter(*v.T,s=18,color=B)
  for p,l,off in zip(v,'ABC',[(-4,-1),(3,-1),(0,3)]):a.text(*(p+off),l,ha='center')
 ax[0].add_patch(Circle((0,0),20,fill=False,color=R,ls='--',lw=1.1));ax[0].plot([-20,20],[0,0],color=R);ax[0].text(0,-5,'D = 40 m',ha='center');ax[0].plot([0,0],[20,v[2,1]],color=R);ax[0].annotate('圆外顶点',v[2],xytext=(-23,30),fontsize=8,arrowprops=dict(arrowstyle='->',lw=.7,color=R))
 c=[0,20/np.sqrt(3)];ax[1].add_patch(Circle(c,40/np.sqrt(3),fill=False,color=B));ax[1].scatter(*c,s=15,color=B);ax[1].plot([0,20],[c[1],0],color=B,ls='--',lw=.7);ax[1].text(2,10,'R = 23.09 m')
 ax[0].set_title('(a) 以最长边为直径',loc='left',fontsize=10);ax[1].set_title('(b) 允许自由选择圆心',loc='left',fontsize=10)
 record(fig,'question1/图3_三角形反例','可由三组合法测向约束构造的正三角形定位区域。为便于标注，图形作了平移；边长及区域直径均为40 m，最小包围圆半径为40/√3≈23.09 m，大于D/2=20 m。',[DATA,'question1/代码/examples.py'])
 fig,ax=panels();rs=[solve(cs['多点递进定位'][:n]) for n in (2,3,4)]
 for i,r in enumerate(rs):
  poly(ax[0],r.vertices,[G,B,R][i]);ax[1].plot([2,3,4][i],r.diameter_m,'o',color=[G,B,R][i]);ax[1].annotate(f'{r.diameter_m:.2f}',(i+2,r.diameter_m),xytext=(0,8),textcoords='offset points',ha='center')
 ax[1].plot([2,3,4],[r.diameter_m for r in rs],color=B,lw=.9,zorder=0);ax[1].set_xticks([2,3,4]);ax[1].set_ylim(0,43);ax[1].set_xlim(1.7,4.3);ax[1].set_xlabel('累计测向次数');ax[1].set_ylabel('区域直径 / m');ax[1].set_title('(b) 直径变化',loc='left',fontsize=10)
 mapax(ax[0],'(a) 交集递进');ax[0].plot([],[],color=G,label='2 次');ax[0].plot([],[],color=B,label='3 次');ax[0].plot([],[],color=R,label='4 次');fig.subplots_adjust(bottom=.31);fig.legend(*ax[0].get_legend_handles_labels(),frameon=False,fontsize=8,ncol=3,loc='lower center',bbox_to_anchor=(.29,.005));record(fig,'question1/图5_递进定位','按同一算例依次加入第二、第三、第四次测向约束。新增观测可以缩小可行集，但不保证每次都严格减小直径；第三次观测后的直径保持不变。',['question1/代码/examples.py','question1/代码/geometry.py'])
 lines=source('question1/表格/表四_交会角敏感性数据.md').read_text().splitlines();rows=[]
 for l in lines:
  try:rows.append([float(x.strip()) for x in l.strip('|').split('|')])
  except ValueError:pass
 a=np.array(rows);fig,axs=panels(1,3.5);ax=axs[0]
 ax.plot(a[:,0],a[:,1],color=B,lw=1.2);ax.set_xlabel('交会角 / °');ax.set_ylabel('区域直径 / m');ax.set_xlim(0,180);ax.set_ylim(0,720)
 # Explicit ROI and coordinate-linked inset; no reference line through inset labels.
 ax.add_patch(Rectangle((30,30),120,70,fill=False,edgecolor=R,linestyle='--',linewidth=.9))
 ax.annotate('放大范围',(150,100),xytext=(150,235),ha='center',fontsize=8,arrowprops=dict(arrowstyle='->',color=R,lw=.7))
 ins=ax.inset_axes([.33,.48,.36,.41]);sel=a[(a[:,0]>=30)&(a[:,0]<=150)]
 ins.plot(sel[:,0],sel[:,1],color=B,lw=1);ins.set_xlim(30,150);ins.set_ylim(30,100);ins.set_xticks([30,90,150]);ins.set_yticks([40,60,80,100]);ins.tick_params(labelsize=7,length=2)
 ins.set_title('局部放大：30°—150°',fontsize=8,pad=7);ins.set_xlabel('交会角 / °',fontsize=7,labelpad=2);ins.set_ylabel('区域直径 / m',fontsize=7,labelpad=3)
 for spine in ins.spines.values():spine.set_visible(True);spine.set_linewidth(.65)
 from matplotlib.patches import ConnectionPatch
 for x,corner in [(30,(0,0)),(150,(1,0))]:
  line=ConnectionPatch(xyA=(x,100),coordsA=ax.transData,xyB=corner,coordsB=ins.transAxes,linestyle='--',linewidth=.7,color=G,zorder=0)
  ax.add_artist(line)
 record(fig,'question1/图6_交会角敏感性','固定两检测点距目标均为700 m，真实读数误差为零，建模误差半角为1°。主图展示5°—175°范围的计算结果；砖红虚线框选出交会角30°—150°、区域直径30—100 m的局部，灰色虚线将该范围连接到上方放大图。放大图的横、纵坐标与主图含义及单位相同，仅改变显示尺度，并非另一组实验。局部图表明交会角接近90°时区域直径较小；该图为确定性几何实验。',['question1/表格/表四_交会角敏感性数据.md'])

 fig,ax=panels(1,3.1);a=ax[0];a.axis('off');a.set_xlim(0,10);a.set_ylim(0,6)
 boxes=[(3.4,5,'输入测向约束'),(3.4,3.65,'可行性与有界性判定'),(.3,2.1,'空集 / 无界'),(5.8,2.1,'有界可行集'),(.3,.4,'报告状态与直径'),(5.8,.4,'凸包 → 直径 → 包围圆')]
 for x,y,t in boxes:a.add_patch(Rectangle((x,y),3,0.7,facecolor='white',edgecolor=B,lw=.8));a.text(x+1.5,y+.35,t,ha='center',va='center',fontsize=9)
 for p,q in [((4.9,5),(4.9,4.35)),((4.2,3.65),(1.8,2.8)),((5.6,3.65),(7.3,2.8)),((1.8,2.1),(1.8,1.1)),((7.3,2.1),(7.3,1.1))]:a.annotate('',q,p,arrowprops=dict(arrowstyle='->',lw=.8))
 record(fig,'question1/图2_算法流程','第一问求解流程。先判断空集和无界情形，再对有界交集计算凸包、直径及最小包围圆；点与线段按退化凸集处理，不使用人为包围框代替无界判定。',['question1/代码/geometry.py'])
