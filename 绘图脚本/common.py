from pathlib import Path
import json,hashlib,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Circle,Polygon,Rectangle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'论文图表';OUT.mkdir(exist_ok=True)
installed_fonts={Path(p).name.lower():p for p in fm.findSystemFonts()}
font_candidates=[os.environ.get('PAPER_FONT','')] + [installed_fonts.get(name,'') for name in ('songti.ttc','simsun.ttc','notoserifcjk-regular.ttc')]
font=next((p for p in font_candidates if p and Path(p).exists()),None)
if not font: raise RuntimeError('请用 PAPER_FONT 指定可用的中文字体文件')
fm.fontManager.addfont(font)
plt.rcParams.update({'font.family':fm.FontProperties(fname=font).get_name(),'font.size':9,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','text.color':'black','svg.fonttype':'path','savefig.facecolor':'white'})
B='#326881';R='#B45B42';G='#9DA7AD';PALE='#E8EFF2'
manifest=[];sources={}
def source(p):
 p=ROOT/p;sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(source(p).read_text())
def record(fig,name,caption,data):
 from matplotlib.text import Text
 for t in fig.findobj(Text):t.set_color('black')
 target=OUT/name;target.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(target.with_suffix('.png'),dpi=300);fig.savefig(target.with_suffix('.svg'));plt.close(fig)
 for p in data:source(p)
 manifest.append(dict(name=name,caption=caption,sources=data))
def panels(n=2,h=2.9):
 fig,axs=plt.subplots(1,n,figsize=(6.3,h),squeeze=False);fig.subplots_adjust(left=.1,right=.98,top=.86,bottom=.2,wspace=.42);return fig,axs[0]
def mapax(ax,title):
 ax.set_aspect('equal');ax.set_xlabel('东向 / m');ax.set_ylabel('北向 / m');ax.set_title(title,loc='left',fontsize=10);ax.tick_params(length=3)
def grid(ax):ax.grid(axis='x',lw=.5,color='#E5E8EA');ax.set_axisbelow(True)
def poly(ax,v,color=B,alpha=.15):
 ax.add_patch(Polygon(v,closed=True,facecolor=color,alpha=alpha));v=np.asarray(v);ax.plot(*np.vstack([v,v[0]]).T,color=color,lw=1)
def pathplot(ax,pts,color=B):
 pts=np.asarray(pts);ax.plot(*pts.T,color=color,lw=.9);ax.scatter(*pts.T,s=8,color=color)
 ax.scatter(*pts[0],marker='*',s=55,color='black',zorder=5);ax.scatter(*pts[-1],marker='s',s=22,facecolor='white',edgecolor=color,zorder=5)
 ax.add_patch(Circle((0,0),1800,fill=False,ls='--',color=G,lw=.8))
 for i in [len(pts)//3,2*len(pts)//3]:
  if i+1<len(pts):
   a=pts[i]*.6+pts[i+1]*.4;b=pts[i]*.4+pts[i+1]*.6;ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=color,lw=.9))

# Explicit symbol keys, placed outside data axes by each plot.
def symbol(label,marker=None,color=B,ls='None',hollow=False):
 from matplotlib.lines import Line2D
 return Line2D([],[],label=label,marker=marker,color=color,linestyle=ls,markersize=5,markerfacecolor='white' if hollow else color,linewidth=1)
