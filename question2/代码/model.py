"""第二问：保证接收可行域、带接收距离约束的后验与有限情景选点。"""
from dataclasses import dataclass, asdict
from itertools import combinations
import math

TOL = 1e-8


@dataclass(frozen=True)
class Problem:
    x: float = 0.
    y: float = 0.
    angle_deg: float = 0.
    error_deg: float = 1.
    target_radius: float = 1800.
    receive_min: float = 1000.
    receive_max: float = 1500.
    near_radius: float = 5.
    speed: float = 5.
    measure_time: float = 5.

    def validate(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('输入必须为有限数值')
        if not 0 < self.error_deg < 90:
            raise ValueError('误差半角必须介于零与九十度之间')
        if not 0 < self.receive_min <= self.receive_max:
            raise ValueError('接收半径区间无效')
        if not 0 <= self.near_radius < self.receive_max:
            raise ValueError('近距离阈值无效')
        if min(self.target_radius,self.speed) <= 0 or self.measure_time < 0:
            raise ValueError('区域半径、速度或检测时间无效')

    def world(self,p):
        t = math.radians(self.angle_deg)
        return (self.x+p[0]*math.cos(t)-p[1]*math.sin(t),
                self.y+p[0]*math.sin(t)+p[1]*math.cos(t))

    def local(self,p):
        t = math.radians(self.angle_deg)
        dx,dy = p[0]-self.x,p[1]-self.y
        return (dx*math.cos(t)+dy*math.sin(t),-dx*math.sin(t)+dy*math.cos(t))


def linspace(start,stop,count):
    if count < 2:
        raise ValueError('采样点数至少为二')
    return [start+(stop-start)*i/(count-1) for i in range(count)]


def diameter(points):
    return max((math.dist(a,b) for a,b in combinations(points,2)),default=0.)


def clip(polygon,plane):
    """凸多边形裁剪：a*x+b*y<=c。输入和输出坐标均使用米。"""
    if not polygon:
        return []
    a,b,c = plane
    out = []
    for i,p in enumerate(polygon):
        q = polygon[(i+1)%len(polygon)]
        fp,fq = a*p[0]+b*p[1]-c,a*q[0]+b*q[1]-c
        pin,qin = fp <= TOL,fq <= TOL
        if pin:
            out.append(p)
        if pin != qin:
            # 按原始直线求交；容差只用于内外分类。
            denominator = fp-fq
            if abs(denominator)>1e-15:
                weight = min(1.,max(0.,fp/denominator))
                out.append((p[0]+weight*(q[0]-p[0]),p[1]+weight*(q[1]-p[1])))
    compact = []
    for p in out:
        if not compact or math.dist(p,compact[-1]) > TOL:
            compact.append(p)
    if len(compact)>1 and math.dist(compact[0],compact[-1]) <= TOL:
        compact.pop()
    return compact


def wedge_planes(station,theta,error):
    low,high = math.radians(theta-error),math.radians(theta+error)
    pairs = [(math.sin(low),-math.cos(low)),(-math.sin(high),math.cos(high))]
    return [(a,b,a*station[0]+b*station[1]) for a,b in pairs]


def disk_planes(center,radius,sides):
    if sides < 8:
        raise ValueError('外切多边形至少使用八条边')
    return [(math.cos(2*math.pi*i/sides),math.sin(2*math.pi*i/sides),
             radius+center[0]*math.cos(2*math.pi*i/sides)+center[1]*math.sin(2*math.pi*i/sides)) for i in range(sides)]


def prior_polygon(problem,sides=128):
    """第一观测的保守外多边形；圆域采用外切多边形，绝不内缩丢掉真值。"""
    problem.validate()
    bound = problem.receive_max/math.cos(math.pi/sides)+1
    p = [(-bound,-bound),(bound,-bound),(bound,bound),(-bound,bound)]
    planes = wedge_planes((0.,0.),0.,problem.error_deg)
    planes += disk_planes((0.,0.),problem.receive_max,sides)
    planes += disk_planes(problem.local((0.,0.)),problem.target_radius,sides)
    for plane in planes:
        p = clip(p,plane)
    return p


def safe_centers(problem):
    r = problem.receive_min
    e = math.radians(problem.error_deg)
    return [(0.,0.),(r*math.cos(e),r*math.sin(e)),(r*math.cos(e),-r*math.sin(e))]


def safety_margin(point,problem):
    """到三个安全圆边界的最小内侧距离；非负则具有连续接收保证。"""
    return problem.receive_min-max(math.dist(point,c) for c in safe_centers(problem))


def is_safe(point,problem,margin=0.):
    return safety_margin(point,problem) >= margin-TOL


def safe_lateral_limit(a,problem):
    """三个圆盘交集在给定前进距离 a 处的上下边界 ±b_max。"""
    r = problem.receive_min
    e = math.radians(problem.error_deg)
    if not 0 <= a <= r:
        return None
    first = math.sqrt(max(0.,r*r-a*a))
    term = r*r-(a-r*math.cos(e))**2
    if term < -TOL:
        return None
    second = math.sqrt(max(0.,term))-r*math.sin(e)
    if second < -TOL:
        return None
    return max(0.,min(first,second))


def target_scenarios(problem,radial_count=25,angle_count=5):
    """在实际圆域与首次误差扇形交集内采样；并非目标概率分布。"""
    problem.validate()
    result = []
    for delta in linspace(-problem.error_deg,problem.error_deg,angle_count):
        global_angle = math.radians(problem.angle_deg+delta)
        ux,uy = math.cos(global_angle),math.sin(global_angle)
        projection = problem.x*ux+problem.y*uy
        discriminant = projection*projection+problem.target_radius**2-problem.x**2-problem.y**2
        if discriminant < 0:
            continue
        span = math.sqrt(max(0.,discriminant))
        lo = max(problem.near_radius+1e-4,-projection-span)
        hi = min(problem.receive_max,-projection+span)
        if hi < lo:
            continue
        for radius in linspace(lo,hi,radial_count):
            angle = math.radians(delta)
            result.append((radius*math.cos(angle),radius*math.sin(angle)))
    if not result:
        raise ValueError('观测方向与目标圆域无可采样交集；请检查输入或加密角度采样')
    return result


def received_polygon(prior,point,receive_max=1500.,sides=128):
    """成功接收后加入 B(第二点,最大接收半径) 的外切多边形约束。"""
    if not math.isfinite(receive_max) or receive_max <= 0:
        raise ValueError('接收半径必须为正有限数值')
    p = prior
    for plane in disk_planes(point,receive_max,sides):
        p = clip(p,plane)
    return p


def direction_polygon(received,point,measured_angle,error):
    """在已加入接收距离约束的区域上叠加第二次误差楔形。"""
    p = received
    for plane in wedge_planes(point,measured_angle,error):
        p = clip(p,plane)
    return p


def posterior_polygon(prior,point,measured_angle,error,receive_max=1500.,sides=128):
    return direction_polygon(received_polygon(prior,point,receive_max,sides),point,measured_angle,error)


def evaluate(point,problem,prior,targets,noise_count=5,weight=0.,sides=128):
    """比较有限目标和误差情景下的保守直径。不是连续全局最优证明。"""
    if noise_count < 2 or weight < 0:
        raise ValueError('误差采样数至少为二，移动权重不能为负')
    prior_diameter = diameter(prior)
    move = math.hypot(*point)
    action_time = move/problem.speed+problem.measure_time
    if move <= TOL:
        # 同地点测向误差固定，不能生成一个独立的新读数来虚假缩小区域。
        return dict(local_point=list(point),world_point=list(problem.world(point)),
                    certified=True,safety_margin_m=safety_margin(point,problem),
                    move_m=move,action_time_s=action_time,worst_diameter_m=prior_diameter,
                    grid_mean_diameter_m=prior_diameter,no_signal_target_count=0,
                    target_count=len(targets),scenario_count=len(targets),near_target_count=0,
                    score=prior_diameter+weight*action_time,worst_target=list(targets[-1]),
                    worst_noise_deg=None)
    worst,total,count = -1.,0.,0
    lost,near = 0,0
    worst_target,worst_noise = None,None
    noises = linspace(-problem.error_deg,problem.error_deg,noise_count)
    # 这一圆域与未知示向度无关，每个候选仅裁剪一次，再供全部成功情景复用。
    received = received_polygon(prior,point,problem.receive_max,sides)
    near_bound = None
    for target in targets:
        distance = math.dist(point,target)
        # 首次信号已收到，对每个可能真值，最不利合法半径为 max(r_min,r_1)。
        radius = max(problem.receive_min,math.hypot(*target))
        if distance > radius+1e-7:
            samples = [(None,prior_diameter)]
            lost += 1
        elif distance <= problem.near_radius:
            if near_bound is None:
                near_polygon = received_polygon(received,point,problem.near_radius,sides)
                near_bound = min(diameter(near_polygon),2*problem.near_radius)
            samples = [(None,near_bound)]
            near += 1
        else:
            theta = math.degrees(math.atan2(target[1]-point[1],target[0]-point[0]))
            samples = []
            for noise in noises:
                polygon = direction_polygon(received,point,theta+noise,problem.error_deg)
                if not polygon:
                    raise ArithmeticError('合法观测产生空可行域，请检查精度或模型')
                samples.append((noise,diameter(polygon)))
        # 每个目标权重相同；近距离和失联分支复制同一上界以对齐误差网格。
        if len(samples)==1:
            samples = samples*noise_count
        for noise,value in samples:
            total += value
            count += 1
            if value > worst:
                worst,worst_target,worst_noise = value,list(target),noise
    return dict(local_point=list(point),world_point=list(problem.world(point)),
                certified=is_safe(point,problem),safety_margin_m=safety_margin(point,problem),
                move_m=move,action_time_s=action_time,worst_diameter_m=worst,
                grid_mean_diameter_m=total/count,no_signal_target_count=lost,
                target_count=len(targets),scenario_count=count,near_target_count=near,
                score=worst+weight*action_time,worst_target=worst_target,worst_noise_deg=worst_noise)


def select_point(problem,grid_step=100.,refine_step=25.,radial_count=25,angle_count=5,
                 noise_count=5,weight=0.,safety_buffer=1.,sides=128):
    if grid_step <= 0 or refine_step <= 0 or safety_buffer < 0 or weight < 0:
        raise ValueError('网格步长必须为正，安全余量和权重不能为负')
    problem.validate()
    prior = prior_polygon(problem,sides)
    targets = target_scenarios(problem,radial_count,angle_count)
    radius = problem.receive_min
    evaluated = {}
    def add(a,b):
        key = (round(a,8),round(b,8))
        if key in evaluated or not is_safe(key,problem,safety_buffer):
            return
        evaluated[key] = evaluate(key,problem,prior,targets,noise_count,weight,sides)
    steps = int(math.ceil(radius/grid_step))
    for i in range(steps+1):
        for j in range(-steps,steps+1):
            add(i*grid_step,j*grid_step)
    # 保证粗网格过大时仍有可行起点。
    for sign in (-1,1):
        add(.6*radius,sign*.4*radius)
    if not evaluated:
        raise ValueError('安全余量过大或候选网格为空')
    order = lambda r: (r['score'],r['move_m'],r['local_point'][0],r['local_point'][1])
    half = int(math.ceil(grid_step/refine_step))
    for _ in range(3):
        seeds = sorted(evaluated.values(),key=order)[:4]
        old_count = len(evaluated)
        for seed in seeds:
            a,b = seed['local_point']
            for i in range(-half,half+1):
                for j in range(-half,half+1):
                    add(a+i*refine_step,b+j*refine_step)
        if len(evaluated)==old_count:
            break
    ranking = sorted(evaluated.values(),key=order)
    # C0 只保证接收。实际评估集合 Q 还需比较定位直径和动作耗时，
    # 去掉被其他点同时在这两个指标上支配的点，才形成离散非支配点集。
    nondominated_candidates = [r for r in ranking if not any(
        other['worst_diameter_m'] <= r['worst_diameter_m'] and
        other['action_time_s'] <= r['action_time_s'] and
        (other['worst_diameter_m'] < r['worst_diameter_m'] or other['action_time_s'] < r['action_time_s'])
        for other in ranking)]
    return {'problem':asdict(problem),'settings':dict(grid_step=grid_step,refine_step=refine_step,
            radial_count=radial_count,angle_count=angle_count,noise_count=noise_count,weight=weight,
            safety_buffer=safety_buffer,sides=sides),'best':min(nondominated_candidates,key=order),
            'candidate_count':len(ranking),'ranking':ranking,'prior_polygon':prior,
            'nondominated_candidate_count':len(nondominated_candidates),'nondominated_candidates':nondominated_candidates,
            'posterior_model':'首次可行域 ∩ 第二次误差楔形 ∩ 第二点最大接收距离圆；圆盘使用外切多边形。',
            'domain_definitions':{'C0':'保证接收可行域，不能直接当作最终候选域',
                                 'Q':'在 C0 内保留接收余量后生成并实际评分的离散点集',
                                 'Q_nd':'Q 中关于后验直径和动作耗时的非支配点，即离散非支配点集'},
            'prior_diameter_m':diameter(prior),
            'interpretation':'C0 仅保证连续接收；在离散评估集合中先形成离散非支配候选，再按目标函数推荐。最优仅针对有限候选及情景网格。'}
