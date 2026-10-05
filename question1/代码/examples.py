"""全部为自建确定性算例，不是官方模拟器数据。"""
import math
from geometry import bearing


def observations(stations, target=(0.,0.), errors=None):
    errors = errors if errors is not None else [0.]*len(stations)
    return [dict(x=x,y=y,angle_deg=(bearing((x,y),target)+e)%360,error_deg=1.)
            for (x,y),e in zip(stations,errors)]


def equilateral_case():
    # 三条逆时针边各作为一个测向楔形的下边界；上边界冗余。
    vertices = [(-20.,-20/math.sqrt(3)),(20.,-20/math.sqrt(3)),(0.,40/math.sqrt(3))]
    obs = []
    for i,a in enumerate(vertices):
        b = vertices[(i+1)%3]
        u = ((b[0]-a[0])/40,(b[1]-a[1])/40)
        station = (a[0]-1000*u[0],a[1]-1000*u[1])
        obs.append(dict(x=station[0],y=station[1],angle_deg=(math.degrees(math.atan2(u[1],u[0]))+1)%360,error_deg=1.))
    return obs


def quadrilateral_case():
    """B题的两测向四边形反例；不是官方实测数据。"""
    return [
        dict(x=0., y=0., angle_deg=44.5, error_deg=1.),
        dict(x=1000., y=0., angle_deg=135.5, error_deg=1.),
    ]


def cases():
    return {
        '正交双点定位': observations([(-700.,0.),(0.,-700.)]),
        '多点递进定位': observations([(-700.,0.),(0.,-700.),(600.,500.),(-550.,650.)], errors=[.6,-.4,.3,-.7]),
        '可实现的正三角形反例': equilateral_case(),
        '两次测向四边形反例': quadrilateral_case(),
        '同向观测无界': observations([(-800.,0.),(-400.,0.)]),
        '相背观测空集': [dict(x=1.,y=0.,angle_deg=0.),dict(x=-1.,y=0.,angle_deg=180.)],
        '相向同点退化': [dict(x=0.,y=0.,angle_deg=0.),dict(x=0.,y=0.,angle_deg=180.)],
    }
