"""解析结果、异常几何、变换不变性与独立裁剪算法交叉验证。"""
import json
import math
from pathlib import Path
import random
import unittest
from geometry import solve, make_halfplanes, satisfies, halfplane_region, diameter, minimum_enclosing_circle
from examples import cases, observations, equilateral_case, quadrilateral_case


def clipping_oracle(planes):
    """仅用于已知小范围有界算例的独立参考算法，不是正式求解算法。"""
    polygon = [(-10000.,-10000.),(10000.,-10000.),(10000.,10000.),(-10000.,10000.)]
    for a,b,c in planes:
        out = []
        for i,p in enumerate(polygon):
            q = polygon[(i+1)%len(polygon)]
            fp,fq = a*p[0]+b*p[1]-c,a*q[0]+b*q[1]-c
            if fp <= 0:
                out.append(p)
            if (fp <= 0) != (fq <= 0):
                t = fp/(fp-fq)
                out.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
        polygon = out
    return polygon


class GeometryTests(unittest.TestCase):
    def test_orthogonal_analytic(self):
        result = solve(cases()['正交双点定位'])
        tangent = math.tan(math.radians(1))
        expected = 2*math.sqrt(2)*700*tangent/(1-tangent*tangent)
        self.assertAlmostEqual(result.diameter_m,expected,places=8)
        self.assertTrue(result.diameter_circle_covers)

    def test_realizable_triangle(self):
        obs = equilateral_case()
        result = solve(obs)
        self.assertEqual(len(result.vertices),3)
        self.assertAlmostEqual(result.diameter_m,40.,places=8)
        self.assertAlmostEqual(result.enclosing_radius_m,40/math.sqrt(3),places=8)
        self.assertFalse(result.diameter_circle_covers)
        self.assertTrue(satisfies((0.,0.),make_halfplanes(obs)))
        self.assertTrue(all(5 < math.hypot(o['x'],o['y']) <= 1500 for o in obs))

    def test_two_measurement_quadrilateral_counterexample(self):
        obs = quadrilateral_case()
        result = solve(obs)
        self.assertEqual(len(result.vertices), 4)
        self.assertAlmostEqual(result.diameter_m, 34.904812875, places=8)
        self.assertAlmostEqual(result.diameter_midpoint_max_distance_m, 17.759697816, places=8)
        self.assertAlmostEqual(result.enclosing_radius_m, 17.455064928, places=8)
        self.assertFalse(result.diameter_circle_covers)
        self.assertTrue(all(math.dist(result.enclosing_center, v) <= result.enclosing_radius_m + 1e-6 for v in result.vertices))

    def test_empty(self):
        result = solve(cases()['相背观测空集'])
        self.assertEqual(result.status,'空集')
        self.assertIsNone(result.diameter_m)

    def test_unbounded(self):
        obs = cases()['同向观测无界']
        result = solve(obs)
        self.assertEqual(result.status,'无界')
        self.assertTrue(satisfies((result.feasible_point[0]+1e8,result.feasible_point[1]),make_halfplanes(obs),1e-6))

    def test_single_station(self):
        self.assertEqual(solve([dict(x=200,y=300,angle_deg=359.8)]).status,'无界')

    def test_degenerate_point(self):
        result = solve(cases()['相向同点退化'])
        self.assertEqual(result.status,'点')
        self.assertEqual(result.diameter_m,0.)

    def test_degenerate_segment(self):
        obs = [dict(x=0,y=0,angle_deg=1),dict(x=0,y=0,angle_deg=359),dict(x=10,y=0,angle_deg=180)]
        result = solve(obs)
        self.assertEqual(result.status,'线段')
        self.assertAlmostEqual(result.diameter_m,10.,places=8)

    def test_angle_wrapping(self):
        obs = cases()['正交双点定位']
        shifted = [dict(o,angle_deg=o['angle_deg']+720) for o in obs]
        self.assertAlmostEqual(solve(obs).diameter_m,solve(shifted).diameter_m,places=8)

    def test_no_backward_extension(self):
        planes = make_halfplanes([dict(x=0,y=0,angle_deg=0)])
        self.assertTrue(satisfies((100.,0.),planes))
        self.assertFalse(satisfies((-100.,0.),planes))

    def test_nearly_parallel_but_bounded(self):
        phi = math.radians(2.01)
        obs = observations([(-700.,0.),(-700*math.cos(phi),-700*math.sin(phi))])
        result = solve(obs)
        self.assertEqual(result.status,'有界多边形')
        self.assertGreater(result.diameter_m,100000.)

    def test_translation_rotation_scale(self):
        obs = cases()['多点递进定位']
        original = solve(obs)
        angle = math.radians(127)
        c,s = math.cos(angle),math.sin(angle)
        factor = 3.2
        changed = [dict(x=1e6+factor*(c*o['x']-s*o['y']),y=-2e6+factor*(s*o['x']+c*o['y']),angle_deg=o['angle_deg']+127) for o in obs]
        self.assertAlmostEqual(solve(changed).diameter_m,original.diameter_m*factor,places=6)

    def test_redundant_observations(self):
        obs = cases()['多点递进定位']
        self.assertAlmostEqual(solve(obs).diameter_m,solve(obs+obs[:2]).diameter_m,places=7)

    def test_incremental_monotonicity(self):
        obs = cases()['多点递进定位']
        ds = [solve(obs[:n]).diameter_m for n in range(2,len(obs)+1)]
        self.assertTrue(all(ds[i+1] <= ds[i]+1e-7 for i in range(len(ds)-1)))

    def test_enclosing_circle_known_shapes(self):
        for points,expected in [([(-1,-1),(1,-1),(1,1),(-1,1)],math.sqrt(2)), ([(0,0),(4,0),(1,1)],2.)]:
            _,radius = minimum_enclosing_circle(points)
            self.assertAlmostEqual(radius,expected,places=9)

    def test_invalid_input(self):
        for obs in [[],[dict(x=0,y=0,angle_deg=float('nan'))],[dict(x=0,y=0,angle_deg=0,error_deg=0)]]:
            with self.assertRaises(ValueError):
                solve(obs)

    def test_halfplane_strip_without_vertices(self):
        status,_,_,_ = halfplane_region([(1.,0.,1.),(-1.,0.,0.)])
        self.assertEqual(status,'无界')

    def test_randomized_independent_clipping(self):
        rng = random.Random(20260910)
        for trial in range(120):
            n = rng.randint(3,8)
            stations = []
            for i in range(n):
                angle = 2*math.pi*i/n+rng.uniform(-.1,.1)
                radius = rng.uniform(200,1200)
                stations.append((radius*math.cos(angle),radius*math.sin(angle)))
            obs = observations(stations,errors=[rng.uniform(-.9,.9) for _ in stations])
            planes = make_halfplanes(obs)
            result = solve(obs)
            oracle = clipping_oracle(planes)
            self.assertEqual(result.status,'有界多边形',trial)
            self.assertTrue(all(max(abs(x),abs(y)) < 9000 for x,y in oracle))
            self.assertAlmostEqual(result.diameter_m,diameter(oracle)[0],places=6)
            self.assertTrue(all(satisfies(v,planes,1e-6) for v in result.vertices))
            self.assertTrue(satisfies((0.,0.),planes))
            self.assertTrue(all(math.dist(result.enclosing_center,v) <= result.enclosing_radius_m+1e-6 for v in result.vertices))
            self.assertGreaterEqual(result.enclosing_radius_m+1e-6,result.diameter_m/2)
            self.assertLessEqual(result.enclosing_radius_m,result.diameter_m/math.sqrt(3)+1e-6)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = Path(__file__).resolve().parents[1]
    (root/'表格').mkdir(parents=True,exist_ok=True)
    lines = ['# 表五：算法验证结果','', '测试数据均为解析构造或固定种子的自建数据，不是官方演练结果。','',
             '| 项目 | 结果 |','|---|---|',f'| 测试方法数 | {result.testsRun} |',
             '| 独立随机交叉验证算例数 | 120 |',f'| 失败数 | {len(result.failures)} |',f'| 错误数 | {len(result.errors)} |',
             f'| 总体结论 | {"全部通过" if result.wasSuccessful() else "存在失败，不能交付"} |','',
             '验证范围：解析直径、可实现反例、空集、无界、点、线段、角度跨零、前向性、近乎平行、平移旋转缩放、冗余观测、递进单调性、已知包围圆、非法输入以及独立裁剪算法。','']
    (root/'表格'/'表五_算法验证结果.md').write_text('\n'.join(lines),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
