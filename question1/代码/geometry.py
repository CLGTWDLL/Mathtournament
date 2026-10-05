"""第一问：有界角误差下的半平面交、直径及最小包围圆。仅依赖标准库。"""
from dataclasses import dataclass, asdict
from itertools import combinations
import math

EPS = 1e-9
DET_EPS = 1e-12


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def bearing(station, target):
    return math.degrees(math.atan2(target[1]-station[1], target[0]-station[0])) % 360


def make_halfplanes(observations, origin=(0., 0.), scale=1.):
    """返回单位法向量约束 a*x+b*y<=c；角度为度。"""
    planes = []
    for obs in observations:
        x, y = (float(obs['x'])-origin[0])/scale, (float(obs['y'])-origin[1])/scale
        theta, error = float(obs['angle_deg']), float(obs.get('error_deg', 1.))
        if not all(math.isfinite(t) for t in (x, y, theta, error)):
            raise ValueError('坐标、示向度和误差必须是有限数值')
        if not 0 < error < 90:
            raise ValueError('误差半角必须严格介于零度和九十度之间')
        low, high = math.radians((theta-error) % 360), math.radians((theta+error) % 360)
        # cross(u_low, q-s)>=0；cross(u_high, q-s)<=0。
        for a, b in ((math.sin(low), -math.cos(low)), (-math.sin(high), math.cos(high))):
            planes.append((a, b, a*x+b*y))
    return planes


def satisfies(p, planes, tol=EPS):
    return all(a*p[0]+b*p[1] <= c+tol for a,b,c in planes)


def line_intersection(p, q):
    a,b,c = p
    d,e,f = q
    det = a*e-b*d
    if abs(det) <= DET_EPS:
        return None
    return ((c*e-b*f)/det, (a*f-c*d)/det)


def convex_hull(points):
    """单调链；共线时只保留两个端点。"""
    pts = []
    for p in sorted(points):
        if not pts or math.dist(p, pts[-1]) > EPS:
            pts.append(p)
    if len(pts) <= 1:
        return pts
    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and cross(sub(out[-1], out[-2]), sub(p, out[-1])) <= 0:
                out.pop()
            out.append(p)
        return out
    return half(pts)[:-1] + half(reversed(pts))[:-1]


def halfplane_region(planes):
    """不添加人为包围框。返回状态、有限顶点、一个可行点和无界方向。"""
    intersections = []
    for p,q in combinations(planes, 2):
        v = line_intersection(p,q)
        if v is not None and satisfies(v, planes):
            intersections.append(v)
    candidates = [(0.,0.)] + [(a*c/(a*a+b*b), b*c/(a*a+b*b)) for a,b,c in planes]
    candidates += intersections
    witness = next((p for p in candidates if satisfies(p, planes)), None)
    if witness is None:
        return '空集', [], None, None
    # 非平凡二维多面锥含边界方向。检查每个法向量的两个垂直方向。
    directions = [(1.,0.),(-1.,0.),(0.,1.),(0.,-1.)]
    directions += [(sign*b, -sign*a) for a,b,_ in planes for sign in (-1,1)]
    recession = next((d for d in directions if all(a*d[0]+b*d[1] <= DET_EPS for a,b,_ in planes)), None)
    if recession is not None:
        return '无界', convex_hull(intersections), witness, recession
    vertices = convex_hull(intersections)
    if not vertices:
        raise ArithmeticError('未找到有界区域顶点，请检查病态输入或调整数值精度')
    status = '点' if len(vertices)==1 else ('线段' if len(vertices)==2 else '有界多边形')
    return status, vertices, witness, None


def diameter(vertices):
    if not vertices:
        raise ValueError('空顶点集没有直径')
    best, pair = 0., (vertices[0],vertices[0])
    for p,q in combinations(vertices, 2):
        d = math.dist(p,q)
        if d > best:
            best, pair = d, (p,q)
    return best, pair


def circumcircle(p,q,r):
    """平移后计算外接圆，减少大坐标引起的消去误差。"""
    u,v = sub(q,p), sub(r,p)
    det = 2*cross(u,v)
    if abs(det) <= DET_EPS * max(math.hypot(*u)*math.hypot(*v), 1e-30):
        return None
    uu,vv = dot(u,u),dot(v,v)
    offset = ((uu*v[1]-vv*u[1])/det, (u[0]*vv-v[0]*uu)/det)
    center = (p[0]+offset[0],p[1]+offset[1])
    return center, math.hypot(*offset)


def minimum_enclosing_circle(vertices):
    """枚举一、二、三支撑点，小规模问题直接验证全部顶点。"""
    if not vertices:
        raise ValueError('空集不调用包围圆算法')
    best = None
    tol = EPS * max(1., max(math.dist(vertices[0], p) for p in vertices))
    def consider(center, radius):
        nonlocal best
        if best is not None and radius >= best[1]:
            return
        needed = max(math.dist(center,p) for p in vertices)
        if needed <= radius + tol:
            best = (center, max(radius, needed))
    for p in vertices:
        consider(p, 0.)
    for p,q in combinations(vertices,2):
        consider(((p[0]+q[0])/2,(p[1]+q[1])/2),math.dist(p,q)/2)
    for p,q,r in combinations(vertices,3):
        circle = circumcircle(p,q,r)
        if circle is not None:
            consider(*circle)
    if best is None:
        raise ArithmeticError('包围圆求解失败')
    return best


@dataclass
class Result:
    status: str
    vertices: list
    diameter_m: float | None
    diameter_endpoints: list | None
    enclosing_center: tuple | None
    enclosing_radius_m: float | None
    diameter_circle_covers: bool | None
    diameter_midpoint_max_distance_m: float | None
    feasible_point: tuple | None
    recession_direction: tuple | None
    numerical_tolerance_m: float

    def to_dict(self):
        return asdict(self)


def solve(observations):
    if not observations:
        raise ValueError('至少需要一个检测点')
    for o in observations:
        if not all(math.isfinite(float(o[k])) for k in ('x','y','angle_deg')):
            raise ValueError('输入必须是有限数值')
    origin = (sum(float(o['x']) for o in observations)/len(observations),
              sum(float(o['y']) for o in observations)/len(observations))
    scale = max(1., max(math.dist((float(o['x']),float(o['y'])), origin) for o in observations))
    planes = make_halfplanes(observations, origin, scale)
    status, local, witness, direction = halfplane_region(planes)
    restore = lambda p: (origin[0]+scale*p[0], origin[1]+scale*p[1])
    vertices = [restore(p) for p in local]
    witness = restore(witness) if witness is not None else None
    if status in ('空集','无界'):
        return Result(status,vertices,None,None,None,None,None,None,witness,direction,EPS*scale)
    d,pair = diameter(local)
    center,radius = minimum_enclosing_circle(local)
    midpoint = ((pair[0][0]+pair[1][0])/2,(pair[0][1]+pair[1][1])/2)
    midpoint_max = max(math.dist(midpoint,p) for p in local)
    return Result(status,vertices,d*scale,[restore(p) for p in pair],restore(center),radius*scale,
                  radius <= d/2+10*EPS,midpoint_max*scale,witness,None,EPS*scale)
