"""连续候选区域：整矩形的有理数外向界验证，不依赖目标/误差采样。"""
from dataclasses import asdict
from fractions import Fraction as F
from functools import lru_cache
from math import factorial, isqrt
from model import Problem, prior_polygon, target_scenarios, evaluate

SCALE = 10**18


def rational(value):
    return value if isinstance(value,F) else F(str(value))


def lower_decimal(value):
    return F(value.numerator*SCALE//value.denominator,SCALE)


def upper_decimal(value):
    return F(-((-value.numerator*SCALE)//value.denominator),SCALE)


def sqrt_upper(value):
    """精确整数平方根构造有理上界；不调用浮点开方。"""
    value=rational(value)
    if value<0:
        raise ValueError('平方根自变量不能为负')
    numerator=value.numerator*SCALE*SCALE
    denominator=value.denominator
    root=isqrt(numerator//denominator)
    if root*root*denominator<numerator:
        root+=1
    return F(root,SCALE)


def alternating_atan_bounds(inverse,terms=24):
    x=F(1,inverse)
    value=sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)),F(0))
    next_value=value+(-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(value,next_value),max(value,next_value)


def taylor_bounds(x,kind,terms=12):
    offset=1 if kind=='sin' else 0
    value=sum(((-1)**k*x**(2*k+offset)/factorial(2*k+offset) for k in range(terms)),F(0))
    next_value=value+(-1)**terms*x**(2*terms+offset)/factorial(2*terms+offset)
    return min(value,next_value),max(value,next_value)


@lru_cache(maxsize=16)
def trig_bounds(error_deg):
    """Machin 恒等式和交错级数给出可审计的正弦、余弦有理区间。"""
    error=rational(error_deg)
    if not 0<error<=5:
        raise ValueError('连续认证模块目前支持大于零且不超过五度的误差半角')
    a0,a1=alternating_atan_bounds(5)
    b0,b1=alternating_atan_bounds(239)
    pi0,pi1=16*a0-4*b1,16*a1-4*b0
    x0,x1=error*pi0/180,error*pi1/180
    # 本区间小于一弧度，正弦递增、余弦递减，交错项绝对值递减。
    sin0=taylor_bounds(x0,'sin')[0]
    sin1=taylor_bounds(x1,'sin')[1]
    cos0=taylor_bounds(x1,'cos')[0]
    cos1=taylor_bounds(x0,'cos')[1]
    return dict(sin_lower=lower_decimal(sin0),sin_upper=upper_decimal(sin1),
                cos_lower=lower_decimal(cos0),cos_upper=upper_decimal(cos1))


def max_interval_abs(lo,hi):
    return max(abs(lo),abs(hi))


def certify_rectangle(rect,problem,diameter_limit=None,time_limit=None,safety_buffer=1.):
    """验证上半平面闭矩形 [a0,a1]×[b0,b1] 内的全部点。

    返回的 accepted 是有理数不等式结论。未通过仅表示本证书不足，
    不表示整个矩形不满足真实性能。真实目标及第二读数均不采样。
    """
    problem.validate()
    a0,a1,b0,b1=map(rational,rect)
    if not (0<=a0<=a1 and 0<=b0<=b1):
        raise ValueError('认证矩形必须位于局部上半平面，且坐标区间有序')
    limit = None if diameter_limit is None else rational(diameter_limit)
    time = None if time_limit is None else rational(time_limit)
    margin = rational(safety_buffer)
    L,R,near,speed,measure=map(rational,(problem.receive_min,problem.receive_max,problem.near_radius,problem.speed,problem.measure_time))
    if ((limit is not None and limit <= 0)
            or (time is not None and time < measure) or not 0 <= margin < L):
        raise ValueError('直径、时间或接收余量阈值无效')
    tr=trig_bounds(str(problem.error_deg))
    sn,cl,ch=tr['sin_upper'],tr['cos_lower'],tr['cos_upper']
    norm2=a1*a1+b1*b1
    da=max_interval_abs(a0-L*ch,a1-L*cl)
    receiver2=da*da+(b1+L*sn)**2
    reception=max(norm2,receiver2)<=(L-margin)**2
    temporal = time is None or norm2 <= (speed * (time - measure))**2
    H=R*sn
    # 全扇形最远距离在 r=0 或 r=R 的两个角端点达到；对矩形也取外向界。
    far_x=max_interval_abs(a0-R*ch,a1-R*cl)
    M=sqrt_upper(max(norm2,far_x*far_x+(b1+H)**2))
    s0=(b0-H)/M if M else F(0)
    bound2=None
    sine_lower=None
    if 0<s0<=1:
        sine_lower=lower_decimal(s0*cl-sqrt_upper(1-s0*s0)*sn)
        if 0<sine_lower<=1:
            cosine_upper=sqrt_upper(1-sine_lower*sine_lower)
            dx=upper_decimal((2*M*sn+2*H*cosine_upper)/sine_lower)
            bound2=max(dx*dx+(2*H)**2,(2*near)**2)
    performance = bound2 is not None and (limit is None or bound2 <= limit * limit)
    accepted=reception and temporal and performance
    result=dict(accepted=accepted,reception_verified=reception,time_verified=temporal,
                performance_verified=performance,
                diameter_limit_applied=limit is not None,time_limit_applied=time is not None,
                rectangle_upper_half=[float(x) for x in (a0,a1,b0,b1)],
                diameter_upper_m=float(sqrt_upper(bound2)) if bound2 is not None else None,
                time_upper_s=float(sqrt_upper(norm2)/speed+measure),
                receive_margin_lower_m=float(L-sqrt_upper(max(norm2,receiver2))),
                transverse_sine_lower=float(sine_lower) if sine_lower is not None else None)
    # 可独立重算的精确有理数证书；显示浮点数只用于表格和绘图。
    if accepted:
        result['exact_certificate']={
            'norm_squared_upper':str(norm2),'receiver_squared_upper':str(receiver2),
            'diameter_squared_upper':str(bound2),'sine_lower':str(sine_lower),
            'distance_upper':str(M),'first_half_width_upper':str(H),
            'reception_squared_limit':str((L-margin)**2)}
        if limit is not None:
            result['exact_certificate']['diameter_limit_squared'] = str(limit * limit)
        if time is not None:
            result['exact_certificate']['time_distance_squared_limit'] = str((speed * (time - measure))**2)
    return result


def build_region(problem,diameter_limit,time_limit,cell_size=25.,safety_buffer=1.):
    """最终区域为通过整块认证的闭矩形并集，而不是网格中心点集。"""
    problem.validate()
    step=rational(cell_size)
    L=rational(problem.receive_min)
    if step<=0:
        raise ValueError('矩形边长必须为正')
    boundaries=[F(0)]
    while boundaries[-1]<L:
        boundaries.append(min(L,boundaries[-1]+step))
    cells=[]
    stats=dict(total_upper_half=0,reception_pass=0,time_pass=0,performance_pass=0)
    for a0,a1 in zip(boundaries,boundaries[1:]):
        for b0,b1 in zip(boundaries,boundaries[1:]):
            cert=certify_rectangle((a0,a1,b0,b1),problem,diameter_limit,time_limit,safety_buffer)
            stats['total_upper_half']+=1
            for field,stat in [('reception_verified','reception_pass'),('time_verified','time_pass'),('performance_verified','performance_pass')]:
                stats[stat]+=int(cert[field])
            if cert['accepted']:
                for sign in (1,-1):
                    rect=[a0,a1,b0,b1] if sign==1 else [a0,a1,-b1,-b0]
                    cells.append({'bounds':[float(x) for x in rect],
                                  'exact_bounds':[str(x) for x in rect],
                                  'reflection':sign,'certificate':cert})
    area=sum(((rational(c['exact_bounds'][1])-rational(c['exact_bounds'][0]))*
              (rational(c['exact_bounds'][3])-rational(c['exact_bounds'][2])) for c in cells),F(0))
    return dict(problem=asdict(problem),settings=dict(diameter_limit=diameter_limit,time_limit=time_limit,
                cell_size=cell_size,safety_buffer=safety_buffer),
                domain_definition='最终连续候选区域为全部通过整矩形接收、性能及耗时证明的闭矩形并集。',
                guarantee='每个区域内位置、每个符合首次观测的连续目标位置、每个允许的第二测向误差均满足阈值。',
                proof_method='有理数外向界；Machin 恒等式与交错级数界；整数平方根上界；不使用目标或误差采样作证书。',
                area_m2=float(area),exact_area_m2=str(area),cell_count=len(cells),cells=cells,stats=stats,
                upper_diameter_m=max((c['certificate']['diameter_upper_m'] for c in cells),default=None),
                upper_time_s=max((c['certificate']['time_upper_s'] for c in cells),default=None),
                lower_receive_margin_m=min((c['certificate']['receive_margin_lower_m'] for c in cells),default=None),
                trig_intervals={k:str(v) for k,v in trig_bounds(str(problem.error_deg)).items()})


def containing_cell(region,point):
    a,b=map(rational,point)
    for index,cell in enumerate(region['cells']):
        a0,a1,b0,b1=map(F,cell['exact_bounds'])
        if a0<=a<=a1 and b0<=b<=b1:
            return index
    return None


def recommend_in_region(region,radial_count=25,angle_count=5,noise_count=5,weight=0.,sides=128):
    """保证先由连续区域取得；情景评分仅用于在其中偏好排序。"""
    if not region['cells']:
        return None
    problem=Problem(**region['problem'])
    prior=prior_polygon(problem,sides)
    targets=target_scenarios(problem,radial_count,angle_count)
    choices=[]
    for index,cell in enumerate(region['cells']):
        a0,a1,b0,b1=cell['bounds']
        point=((a0+a1)/2,(b0+b1)/2)
        result=evaluate(point,problem,prior,targets,noise_count,weight,sides)
        result.update(certified_cell_index=index,
                      continuous_diameter_upper_m=cell['certificate']['diameter_upper_m'],
                      continuous_time_upper_s=cell['certificate']['time_upper_s'])
        choices.append(result)
    choices.sort(key=lambda x:(x['score'],x['move_m'],x['local_point']))
    best=choices[0]
    return dict(best=best,ranking=choices,interpretation='推荐点仅按有限情景排序，但所在整块闭矩形已有连续性能保证；不宣称连续全局最优。')
