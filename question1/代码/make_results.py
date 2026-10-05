"""一键生成第一问自建算例、中文图件、结果表与机器可读结果。"""
from pathlib import Path
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT.parent/'tmp'/'question1_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, FancyBboxPatch
from geometry import solve, bearing
from examples import cases, observations

FIGURES,TABLES,DATA = ROOT/'图片',ROOT/'表格',ROOT/'数据'
BLUE,GREEN,RED,ORANGE = '#2463a7','#228569','#c94d51','#cf902e'
COLORS = [BLUE,ORANGE,GREEN,'#8859a5']


def setup():
    for folder in (FIGURES,TABLES,DATA):
        folder.mkdir(parents=True,exist_ok=True)
    supplied = os.environ.get('CHINESE_FONT')
    options = [Path(supplied)] if supplied else []
    installed_fonts={Path(p).name.lower():Path(p) for p in font_manager.findSystemFonts()}
    options += [installed_fonts[name] for name in ('msyh.ttc','simhei.ttf','notosanscjk-regular.ttc') if name in installed_fonts]
    path = next((p for p in options if p.is_file()),None)
    if path is None:
        raise RuntimeError('未找到中文字体；请将 CHINESE_FONT 环境变量设为中文字体文件路径')
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams.update({'font.family':family,'font.size':11,'axes.titlesize':14,
                         'axes.labelsize':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white',
                         'axes.formatter.useoffset':False})


def save(fig,name):
    # 所有可见文本必须无拉丁字母，数值、数学符号和中文允许。
    import re
    fig.canvas.draw()
    from matplotlib.text import Text
    for obj in fig.findobj(match=Text):
        if obj.get_visible() and re.search(r'[A-Za-z]',obj.get_text()):
            raise AssertionError(f'图片存在非中文文字：{obj.get_text()}')
    fig.savefig(FIGURES/(name+'.png'),dpi=300,bbox_inches='tight',pad_inches=.18)
    plt.close(fig)


def axes_map(ax,title):
    ax.set_title(title,pad=14)
    ax.set_xlabel('东西方向坐标（米）')
    ax.set_ylabel('南北方向坐标（米）')
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.18,zorder=0)
    ax.ticklabel_format(style='plain',axis='both',useOffset=False)


def polygon(ax,result,color=BLUE,label='定位区域',alpha=.20):
    p = result.vertices
    ax.add_patch(Polygon(p,closed=True,facecolor=color,edgecolor=color,alpha=alpha,linewidth=1.7,label=label))
    ax.plot([x for x,y in p]+[p[0][0]],[y for x,y in p]+[p[0][1]],color=color,lw=1.4)


def table(filename,title,headers,rows,note):
    def cell(value):
        if value is None:
            return '不适用'
        if isinstance(value,bool):
            return '是' if value else '否'
        if isinstance(value,float):
            return f'{value:.6f}'
        return str(value)
    lines = [f'# {title}','',note,'','| '+' | '.join(headers)+' |',
             '| '+' | '.join(['---']*len(headers))+' |']
    lines += ['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]
    (TABLES/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def figure_geometry(obs,result):
    fig,axs = plt.subplots(1,2,figsize=(12.2,5.4),layout='constrained')
    ax = axs[0]
    axes_map(ax,'双点交会定位：观测全景')
    for i,o in enumerate(obs):
        x,y,theta = o['x'],o['y'],o['angle_deg']
        a,b = math.radians(theta-1),math.radians(theta+1)
        boundary = [(x,y),(x+900*math.cos(a),y+900*math.sin(a)),(x+900*math.cos(b),y+900*math.sin(b))]
        ax.add_patch(Polygon(boundary,color=COLORS[i],alpha=.17,label=f'检测点{["一","二"][i]}的误差楔形'))
        angle = math.radians(theta)
        ax.plot([x,x+900*math.cos(angle)],[y,y+900*math.sin(angle)],color=COLORS[i],lw=1.2)
        ax.scatter([x],[y],color=COLORS[i],s=45)
        ax.annotate(f'检测点{["一","二"][i]}',(x,y),xytext=(8,10),textcoords='offset points')
    polygon(ax,result)
    ax.scatter([0],[0],marker='*',color=RED,s=110,label='构造目标真值')
    ax.set_xlim(-800,240)
    ax.set_ylim(-800,240)
    ax.legend(loc='lower left',fontsize=9,framealpha=.95)
    ax = axs[1]
    axes_map(ax,'局部放大：直径与最小包围圆')
    polygon(ax,result)
    p,q = result.diameter_endpoints
    ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=2,label='区域直径线段')
    ax.add_patch(Circle(result.enclosing_center,result.enclosing_radius_m,fill=False,edgecolor=GREEN,lw=1.8,linestyle='--',label='最小包围圆'))
    ax.scatter([v[0] for v in result.vertices],[v[1] for v in result.vertices],s=22,color=BLUE)
    ax.scatter([0],[0],marker='*',color=RED,s=95)
    ax.set_xlim(-23,23)
    ax.set_ylim(-23,23)
    ax.legend(loc='lower right',fontsize=9)
    ax.text(.03,.97,f'区域直径：{result.diameter_m:.3f} 米\n包围圆半径：{result.enclosing_radius_m:.3f} 米',transform=ax.transAxes,va='top',fontsize=10)
    fig.suptitle('图一  误差半角为一度的交会定位示例',fontsize=17)
    save(fig,'图一_双点交会定位与直径')


def figure_counterexample(result):
    fig,axs = plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
    p,q = result.diameter_endpoints
    midpoint = ((p[0]+q[0])/2,(p[1]+q[1])/2)
    for ax in axs:
        axes_map(ax,'')
        polygon(ax,result,color=BLUE,label='正三角形定位区域',alpha=.15)
        ax.scatter([v[0] for v in result.vertices],[v[1] for v in result.vertices],s=38,color=BLUE)
        ax.set_xlim(-34,34)
        ax.set_ylim(-40,34)
    ax = axs[0]
    ax.set_title('直径对应的圆不能覆盖第三个顶点')
    ax.add_patch(Circle(midpoint,result.diameter_m/2,fill=False,edgecolor=RED,linestyle='--',lw=2,label='以最长边为直径的圆'))
    ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=2)
    ax.annotate('未被覆盖的顶点',xy=(0,40/math.sqrt(3)),xytext=(-29,30),arrowprops=dict(arrowstyle='->',color=RED),color=RED,fontsize=10)
    ax.text(.04,.04,'区域直径：40 米\n圆的半径：20 米',transform=ax.transAxes,fontsize=11)
    ax.legend(loc='upper right',bbox_to_anchor=(1.0,.88),fontsize=9)
    ax = axs[1]
    ax.set_title('改变圆心也无法用半径二十米覆盖')
    ax.add_patch(Circle(result.enclosing_center,result.enclosing_radius_m,fill=False,edgecolor=GREEN,lw=2,label='最小包围圆'))
    ax.scatter([0],[0],s=30,color=GREEN,label='最小包围圆圆心')
    for v in result.vertices:
        ax.plot([0,v[0]],[0,v[1]],color=GREEN,alpha=.55,linestyle=':')
    ax.text(.04,.04,f'最小包围圆半径：{result.enclosing_radius_m:.3f} 米\n所需半径大于二十米',transform=ax.transAxes,fontsize=11)
    ax.legend(loc='upper right',fontsize=9)
    fig.suptitle('图二  三组合法测向楔形构成的圆覆盖反例',fontsize=17)
    save(fig,'图二_定位区域直径圆覆盖反例')


def figure_quadrilateral(obs,result):
    fig,axs = plt.subplots(1,2,figsize=(12.2,5.5),layout='constrained')
    ax = axs[0]
    axes_map(ax,'两次测向构造的典型四边形')
    for i,o in enumerate(obs):
        x,y,theta = o['x'],o['y'],o['angle_deg']
        a,b = math.radians(theta-1),math.radians(theta+1)
        ax.add_patch(Polygon([(x,y),(x+1300*math.cos(a),y+1300*math.sin(a)),(x+1300*math.cos(b),y+1300*math.sin(b))],
                             color=COLORS[i],alpha=.16,label=f'检测点{["一","二"][i]}的误差楔形'))
        ax.plot([x,x+1300*math.cos(math.radians(theta))],[y,y+1300*math.sin(math.radians(theta))],color=COLORS[i],lw=1.2)
        ax.scatter([x],[y],color=COLORS[i],s=48)
        ax.annotate(f'检测点{["一","二"][i]}',(x,y),xytext=(8,10),textcoords='offset points')
    polygon(ax,result)
    ax.scatter([500],[500*math.tan(math.radians(44.5))],marker='*',color=RED,s=120,label='构造目标真值')
    ax.set_xlim(-120,1120)
    ax.set_ylim(-80,760)
    ax.legend(loc='lower left',fontsize=9,framealpha=.95)
    ax = axs[1]
    axes_map(ax,'局部放大：直径圆不能覆盖')
    polygon(ax,result)
    p,q = result.diameter_endpoints
    midpoint = ((p[0]+q[0])/2,(p[1]+q[1])/2)
    ax.add_patch(Circle(midpoint,result.diameter_m/2,fill=False,edgecolor=RED,linestyle='--',lw=2,label='以直径为直径的圆'))
    ax.add_patch(Circle(result.enclosing_center,result.enclosing_radius_m,fill=False,edgecolor=GREEN,lw=1.8,label='最小包围圆'))
    ax.scatter([v[0] for v in result.vertices],[v[1] for v in result.vertices],s=26,color=BLUE)
    ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=2)
    ax.set_xlim(472,528)
    ax.set_ylim(465,520)
    ax.legend(loc='lower right',fontsize=9)
    ax.text(.03,.97,f'区域直径：{result.diameter_m:.3f} 米\n直径圆覆盖：否\n最小包围圆半径：{result.enclosing_radius_m:.3f} 米',
            transform=ax.transAxes,va='top',fontsize=10)
    fig.suptitle('图六  两次测向四边形反例',fontsize=17)
    save(fig,'图六_两次测向四边形反例')


def figure_incremental(obs):
    rs = [solve(obs[:n]) for n in (2,3,4)]
    fig,axs = plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
    ax = axs[0]
    axes_map(ax,'新增有效观测逐步缩小可行区域')
    for i,r in enumerate(rs):
        polygon(ax,r,color=COLORS[i],label=f'{["两","三","四"][i]}个检测点',alpha=.15)
    ax.scatter([0],[0],color=RED,marker='*',s=95,label='构造目标真值')
    ax.set_xlim(-17,25)
    ax.set_ylim(-24,25)
    ax.legend(loc='lower right',fontsize=10)
    ax = axs[1]
    ax.set_title('直径与最小包围圆半径均不增大',pad=14)
    ax.plot([2,3,4],[r.diameter_m for r in rs],'o-',color=BLUE,label='定位区域直径',lw=2)
    ax.plot([2,3,4],[r.enclosing_radius_m for r in rs],'s-',color=GREEN,label='最小包围圆半径',lw=2)
    for i,r in enumerate(rs):
        ax.annotate(f'{r.diameter_m:.2f}',(i+2,r.diameter_m),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks([2,3,4],['两个','三个','四个'])
    ax.set_xlabel('累计检测点数量')
    ax.set_ylabel('长度（米）')
    ax.grid(alpha=.2)
    ax.set_ylim(0,max(r.diameter_m for r in rs)*1.22)
    ax.legend(loc='upper right',fontsize=10)
    fig.suptitle('图三  多点观测对定位区域的约束作用',fontsize=17)
    save(fig,'图三_多点观测递进收缩')
    return rs


def sensitivity():
    rows = []
    for angle in range(5,176):
        rad = math.radians(angle)
        obs = observations([(-700.,0.),(-700*math.cos(rad),-700*math.sin(rad))])
        result = solve(obs)
        rows.append((angle,result.diameter_m,result.enclosing_radius_m))
    fig,axs = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,lo,hi,title in [(axs[0],5,175,'完整角度范围'),(axs[1],30,150,'中间角度范围放大')]:
        sel = [row for row in rows if lo <= row[0] <= hi]
        ax.plot([r[0] for r in sel],[r[1] for r in sel],color=BLUE,lw=2,label='定位区域直径')
        ax.axvline(90,color=GREEN,linestyle='--',lw=1.2,label='九十度交会角')
        ax.set_xlabel('两条真实观测方向的夹角（度）')
        ax.set_ylabel('定位区域直径（米）')
        ax.set_title(title,pad=14)
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    fig.suptitle('图四  固定距离下交会角对区域直径的影响',fontsize=17)
    fig.supxlabel('自建几何实验：目标距离均为七百米，示向度误差取零，误差界为正负一度',fontsize=10)
    save(fig,'图四_交会角敏感性分析')
    return rows


def figure_flow():
    fig,ax = plt.subplots(figsize=(10,8))
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.axis('off')
    labels = ['输入检测点坐标、示向度与误差界',
              '将每次测向转换为两个前向半平面约束',
              '求可行点并判断空集或无界情形',
              '枚举边界交点，筛选可行点并构造凸包',
              '枚举顶点对，取得最大距离及对应端点',
              '求最小包围圆，比较其半径与直径的一半',
              '输出区域顶点、直径、覆盖判定和诊断状态']
    for i,label in enumerate(labels):
        y = 8.9-i*1.25
        rect = FancyBboxPatch((1.15,y-.37),7.7,.74,boxstyle='round,pad=0.12',facecolor='#edf4fa' if i%2==0 else '#edf7f2',edgecolor=BLUE if i%2==0 else GREEN,linewidth=1.3)
        ax.add_patch(rect)
        ax.text(5,y,label,ha='center',va='center',fontsize=13)
        if i < len(labels)-1:
            ax.annotate('',xy=(5,y-.78),xytext=(5,y-.5),arrowprops=dict(arrowstyle='->',lw=1.5,color='#617487'))
    ax.text(5,.45,'空集：报告观测不相容；无界：报告直径无穷大\n点和线段按退化凸集处理，不使用人为包围框截断区域',ha='center',va='center',fontsize=11,color='#536273')
    ax.set_title('图五  第一问求解算法流程',fontsize=18,pad=18)
    save(fig,'图五_第一问求解流程')


def main():
    setup()
    all_cases = cases()
    results = {name:solve(obs) for name,obs in all_cases.items()}
    sample = all_cases['正交双点定位']
    (DATA/'正交双点定位.json').write_text(
        json.dumps({'source':'第一问自建验证算例，非官方模拟器数据','observations':sample},
                   ensure_ascii=False,indent=2),encoding='utf-8')
    (DATA/'全部求解结果.json').write_text(json.dumps({name:r.to_dict() for name,r in results.items()},ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    input_rows = []
    for name,obs in all_cases.items():
        for i,o in enumerate(obs,1):
            input_rows.append((name,i,o['x'],o['y'],o['angle_deg'],o.get('error_deg',1.)))
    table('表一_算例输入参数.md','表一：算例输入参数',['算例','检测点序号','横坐标（米）','纵坐标（米）','示向度（度）','误差半角（度）'],input_rows,
          '来源：代码确定性构造。正交、递进和正三角形算例的目标真值均为原点；两次测向四边形反例的目标真值在四边形内部；后三个为几何边界测试，其中空集和同点相向观测用于验证异常处理，不表示真实合法测量。计算使用完整精度，表格仅显示六位小数。')
    table('表二_定位区域与圆覆盖结果.md','表二：定位区域与圆覆盖结果',
          ['算例','区域状态','有限顶点数','直径（米）','最小包围圆半径（米）','是否存在直径相同的覆盖圆'],
          [(name,r.status,len(r.vertices),('无穷大' if r.status=='无界' else ('未定义' if r.status=='空集' else r.diameter_m)),r.enclosing_radius_m,r.diameter_circle_covers) for name,r in results.items()],
          '来源：geometry.py 对表一对应完整精度数据的计算。空集直径未定义；无界区域直径为正无穷，在机器可读结果中均用空值配合区域状态表示。无界区域的有限顶点不构成完整区域。')
    table('表三_定位多边形顶点.md','表三：定位多边形顶点',['算例','顶点序号','横坐标（米）','纵坐标（米）'],
          [(name,i,x,y) for name,r in results.items() if r.status=='有界多边形' for i,(x,y) in enumerate(r.vertices,1)],
          '来源：半平面交算法。顶点按逆时针顺序排列。')
    figure_geometry(all_cases['正交双点定位'],results['正交双点定位'])
    figure_counterexample(results['可实现的正三角形反例'])
    figure_quadrilateral(all_cases['两次测向四边形反例'],results['两次测向四边形反例'])
    incremental = figure_incremental(all_cases['多点递进定位'])
    rows = sensitivity()
    table('表四_交会角敏感性数据.md','表四：交会角敏感性数据',['交会角（度）','区域直径（米）','最小包围圆半径（米）'],rows,
          '来源：自建确定性几何实验。两个检测点到原点目标的距离均为七百米，真实读数误差取零，建模误差界为正负一度。不是模拟器测试或随机分布推断。')
    table('表六_多点递进定位结果.md','表六：多点递进定位结果',['累计检测点数','顶点数','区域直径（米）','最小包围圆半径（米）'],
          [(i+2,len(r.vertices),r.diameter_m,r.enclosing_radius_m) for i,r in enumerate(incremental)],
          '按表一的多点递进定位算例顺序依次加入观测。新增约束只能缩小或保持可行集，不能保证每次都严格缩小。')
    figure_flow()
    print('已生成六幅全中文 PNG 图、五张计算结果表、一个示例输入及汇总结果。')
    for name,r in results.items():
        print(name,r.status,r.diameter_m,r.enclosing_radius_m)


if __name__ == '__main__':
    main()
