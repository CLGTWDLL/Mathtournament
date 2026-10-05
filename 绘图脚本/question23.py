from common import *
def run():
 p='question2/数据/独立复评结果.json';rows=read(p)['rows'];fig,ax=panels(h=2.7);fig.subplots_adjust(left=.19,wspace=.4)
 for a in ax:a.set_ylim(2.55,-.55);a.set_yticks(range(3));a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0);grid(a)
 ax[0].set_yticklabels([r['name'] for r in rows]);ax[1].set_yticklabels([])
 for i,row in enumerate(rows):
  m=row['independent_rollout']['mean_s'];lo,hi=row['independent_rollout']['ci95_s'];c=R if i==1 else B
  ax[0].errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color=c,capsize=3,ms=4);ax[0].text(m,i-.25,f'{m:.1f}',ha='center',fontsize=8)
  v=row['continuous_diameter_upper_m'];ax[1].plot([0,v],[i,i],color=PALE,lw=2);ax[1].plot(v,i,'o',color=c,ms=4);ax[1].text(v+55,i,f'{v:,.1f}',va='center',fontsize=8)
 ax[0].set_xlim(185,262);ax[0].set_xticks([190,210,230,250]);ax[0].set_xlabel('完整任务耗时 / s');ax[0].set_title('(a) 均值与 95% 区间',loc='left',fontsize=10)
 ax[1].set_xlim(0,2300);ax[1].set_xticks([0,1000,2000]);ax[1].set_xlabel('连续直径上界 / m');ax[1].set_title('(b) 几何精度保证',loc='left',fontsize=10)
 record(fig,'question2/图2_时间与精度','三种代表方案在同一批512个独立样本上的复评结果。左侧误差棒表示任务耗时均值的95%置信区间；右侧为连续定位直径的几何上界，不是统计区间。砖红色为帕累托拐点。',[p])
 fig,ax=panels(1,3.1);fig.set_size_inches(6.3,3.4);fig.subplots_adjust(left=.08,right=.54,bottom=.18,top=.95);a=ax[0];mapax(a,'');xx,yy=np.meshgrid(np.linspace(-80,1100,460),np.linspace(-950,950,460));eps=np.radians(1);centers=np.array([[0,0],[1000*np.cos(eps),1000*np.sin(eps)],[1000*np.cos(eps),-1000*np.sin(eps)]])
 valid=np.ones(xx.shape,bool)
 for c in centers:valid&=np.hypot(xx-c[0],yy-c[1])<=999
 a.contourf(xx,yy,valid.astype(float),levels=[.5,1.5],colors=[PALE]);a.contour(xx,yy,valid.astype(float),levels=[.5],colors=[B],linewidths=.8)
 a.scatter(*centers[0],color='black',marker='*',s=40);a.scatter(*centers[1:].T,edgecolors=B,facecolors='white',marker='^',s=24);a.annotate('O',(0,0),xytext=(5,-12),textcoords='offset points');a.annotate('A±',centers[1],xytext=(6,0),textcoords='offset points')
 for i,r in enumerate(rows):a.plot(*r['point'],'o',color=R if i==1 else B,ms=4);a.annotate(['最快','拐点','高精度'][i],r['point'],xytext=(6,-4),textcoords='offset points',fontsize=8)
 a.set_xlim(-80,1150);a.set_ylim(-920,920)
 from matplotlib.patches import Patch
 handles=[symbol('O：首次检测点','*','black'),symbol('A±：辅助构造圆心','^',B,hollow=True),symbol('最快：第二点候选','o',B),symbol('拐点：第二点候选','o',R),symbol('高精度：第二点候选','o',B),Patch(facecolor=PALE,edgecolor=B,label='保证接收候选域')]
 fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.56,.54),frameon=False,fontsize=9,labelspacing=1.1)
 record(fig,'question2/图1_保证接收候选域','标准局部坐标下的保证接收候选域，由以O、A+、A−为圆心、半径999 m的三个圆盘交集构成；其中1 m为数值安全余量。A±相距约34.9 m，在本比例下接近。星号O为首次检测点，空心三角形A±为构造保证接收域的辅助圆心，不是信号源；三个实心圆点为第二次检测的代表候选位置，砖红点为拐点方案，浅蓝填充表示保证接收候选域。区域边界以网格显示，候选可行性仍由解析条件判定。',[p,'question2/融合第二问说明.md'])
 # Q3 geometry is a mathematical illustration, not an invented realized route.
 fig,ax=panels();a=ax[0];mapax(a,'(a) 初始覆盖布局');pts=np.vstack([[0,0],1250*np.column_stack([np.cos(np.arange(6)*np.pi/3),np.sin(np.arange(6)*np.pi/3)])])
 for i,pt in enumerate(pts):a.add_patch(Circle(pt,1000,fill=False,color=B if i==0 else G,lw=.7,ls='-' if i==0 else '--'))
 a.scatter(*pts.T,s=13,color=B);a.add_patch(Circle((0,0),1800,fill=False,color='black',lw=1));a.set_xlim(-2300,2300);a.set_ylim(-2300,2300);a.set_xticks([-2000,0,2000]);a.set_yticks([-2000,0,2000])
 a=ax[1];a.axis('off');a.set_xlim(0,10);a.set_ylim(0,10)
 texts=['维护各频道剩余区域','在可行域内选择联合检测点','同点完成排查与目标测向','更新后验，释放旧优先级','重排剩余定位与清除任务']
 for i,t in enumerate(texts):
  y=8.5-i*1.65;a.add_patch(Rectangle((.05,y-.42),9.9,.84,fill=False,edgecolor=B,lw=.8));a.text(5,y,t,ha='center',va='center',fontsize=8)
  if i<4:a.annotate('',(5,y-1.2),(5,y-.5),arrowprops=dict(arrowstyle='->',lw=.8))
 a.set_title('(b) 联合检测与反馈重排',loc='left',fontsize=10)
 record(fig,'question3/图1_覆盖与反馈机制','左图为正文指定的原点加六个外围站的初始覆盖布局，外围站半径1250 m、间隔60°，接收圆半径1000 m。右图表示联合检测与反馈重排的逻辑；本图是算法机制示意，不是实际行进轨迹，也不主张必须绕一圈。',['question3/第三问解答.md'])
