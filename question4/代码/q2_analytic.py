"""Semi-analytic second-point surrogate; exact two-wedge geometry diagnostics.

The surrogate is NOT the exact expected multi-step task time. Expectations over
the thin-ray prior are integrated in closed form; a two-variable continuous
minimizer uses analytic gradients. Keeping this distinction is intentional.
"""
import numpy as np
from scipy.optimize import minimize
from solve_q2 import RAD, EPS


class RayPrior:
    def __init__(self, scenario):
        s = np.array(scenario['first'], float)
        angle = scenario['bearing']*RAD
        d = np.array([np.cos(angle), np.sin(angle)])
        t = s@d
        discriminant = t*t+1800**2-s@s
        if discriminant < 0:
            raise ValueError('Central ray misses the disk; use the full wedge posterior')
        self.lower = max(5., -t-np.sqrt(discriminant))
        self.limit = min(1500., -t+np.sqrt(discriminant))
        if self.limit <= self.lower:
            raise ValueError('Central ray has no direction-support interval')
        # Unnormalized density: r on [5,1000], (1500r-r²)/500 above 1000.
        self.parts = []
        if self.lower < 1000:
            self.parts.append((self.lower, min(1000., self.limit), 1., 0.))
        if self.limit > 1000:
            self.parts.append((max(1000., self.lower), self.limit, 3., -.002))
        moments = [sum(a*(hi**(j+2)-lo**(j+2))/(j+2)+b*(hi**(j+3)-lo**(j+3))/(j+3)
                       for lo, hi, a, b in self.parts) for j in range(5)]
        self.mass = moments[0]
        self.m = np.array(moments)/self.mass

    @staticmethod
    def primitives(t, v):
        z = np.hypot(t, v)
        a = np.arcsinh(t/v)
        F = np.array([.5*(t*z+v*v*a), z**3/3, (t*z*(2*t*t+v*v)-v**4*a)/8])
        G = np.array([a, z, .5*(t*z-v*v*a), z**3/3-v*v*z])
        return F, G

    def distance(self, u, v):
        value, du, dv = 0., 0., 0.
        for lo, hi, a, b in self.parts:
            Fh, Gh = self.primitives(hi-u, v)
            Fl, Gl = self.primitives(lo-u, v)
            c = np.array([a*u+b*u*u, a+2*b*u, b])
            value += c@(Fh-Fl)
            du -= c@(Gh[1:]-Gl[1:])
            dv += v*(c@(Gh[:3]-Gl[:3]))
        return value/self.mass, np.array([du, dv])/self.mass

    def mse(self, u, v):
        _, m1, m2, m3, m4 = self.m
        A = 2*m4-6*u*m3+7*u*u*m2-4*u**3*m1+u**4
        B = 3*m2-4*u*m1+2*u*u
        Ap = -6*m3+14*u*m2-12*u*u*m1+4*u**3
        Bp = -4*m1+4*u
        Z = A/v**2+B+v*v
        # U[-1°,1°] variance; rounding error is omitted only in this surrogate.
        variance = RAD**2/3
        return variance*Z, variance*np.array([Ap/v**2+Bp, -2*A/v**3+2*v])

    def objective(self, xy, weight=1.):
        u, v = xy
        length = np.hypot(u, v)
        tail, grad = self.distance(u, v)
        mse, mg = self.mse(u, v)
        rms = np.sqrt(max(mse, 1e-20))
        # 10 is one measurement plus the final successful clear. Extra future
        # actions are represented imperfectly by weight*rms/5, not concealed.
        value = (length+tail+weight*rms)/5+10
        derivative = (np.array([u, v])/length+grad+weight*mg/(2*rms))/5
        return value, derivative

    def select(self, weight=1.):
        L = self.limit
        starts = [[.2*L, .1*L], [.5*L, .15*L], [.8*L, .1*L]]
        solutions = [minimize(lambda q:self.objective(q, weight), start, jac=True,
                              method='L-BFGS-B', bounds=[(5., 1.25*L), (5., min(1000., L))],
                              options=dict(ftol=1e-11, gtol=1e-8, maxiter=100)) for start in starts]
        best = min(solutions, key=lambda r:r.fun)
        return best.x, float(best.fun), dict(success=bool(best.success), iterations=int(best.nit),
                                           evaluations=int(best.nfev))


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def wedge_vertices(s1, theta1, s2, theta2, eps=EPS):
    """Exact bounded intersection of two forward wedges, else None.

    No circular priors are used, so a bounded result conservatively contains
    the region after those extra restrictions. Angles are radians.
    """
    positions = [np.asarray(s1), np.asarray(s2)]
    angles = [theta1, theta2]
    normals, intercepts = [], []
    for p, a in zip(positions, angles):
        for t, sign in [(a-eps, 1), (a+eps, -1)]:
            n = sign*np.array([-np.sin(t), np.cos(t)])
            normals.append(n); intercepts.append(n@p)
    N, c = np.array(normals), np.array(intercepts)
    # A nonzero recession direction exists iff it can be chosen along one of
    # the halfplane boundaries. Two wedges here have aperture strictly < pi.
    for n in N:
        for sign in [-1, 1]:
            v = sign*np.array([-n[1], n[0]])
            if np.all(N@v >= -1e-12):
                return None
    vertices = []
    for i in range(4):
        for j in range(i):
            matrix = N[[i, j]]
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            p = np.linalg.solve(matrix, c[[i, j]])
            if np.all(N@p >= c-1e-7):
                if not any(np.linalg.norm(p-q) < 1e-7 for q in vertices):
                    vertices.append(p)
    if not vertices:
        return np.empty((0, 2))
    vertices = np.array(vertices)
    mean = vertices.mean(axis=0)
    order = np.argsort(np.arctan2(vertices[:, 1]-mean[1], vertices[:, 0]-mean[0]))
    return vertices[order]


def enclosing_circle(vertices):
    """Finite exact candidate enumeration for the <=4-vertex two-wedge polygon."""
    V = np.asarray(vertices)
    if len(V) == 0:
        raise ValueError('Empty region')
    best, center = np.inf, None
    candidates = list(V)
    for i in range(len(V)):
        for j in range(i):
            candidates.append((V[i]+V[j])/2)
            for k in range(j):
                A = 2*np.array([V[j]-V[i], V[k]-V[i]])
                if abs(np.linalg.det(A)) > 1e-10:
                    rhs = np.array([V[j]@V[j]-V[i]@V[i], V[k]@V[k]-V[i]@V[i]])
                    candidates.append(np.linalg.solve(A, rhs))
    for c in candidates:
        radius = float(np.max(np.linalg.norm(V-c, axis=1)))
        if radius < best:
            best, center = radius, c
    return center, best


def linear_and_safe_radius(s1, theta1, s2, theta2, eps=EPS):
    """Parallelogram linearization and a conservative perturbation bound.

    The bound encloses the nonlinear feasible set about the nominal line
    intersection whenever eta*||A^-1||<1; otherwise it returns infinity.
    """
    d1 = np.array([np.cos(theta1), np.sin(theta1)])
    d2 = np.array([np.cos(theta2), np.sin(theta2)])
    A = np.array([[-d1[1], d1[0]], [-d2[1], d2[0]]])
    sine = abs(cross(d1, d2))
    if sine < 1e-12:
        return None, np.inf, np.inf
    nominal = np.linalg.solve(A, [A[0]@s1, A[1]@s2])
    r1, r2 = np.linalg.norm(nominal-s1), np.linalg.norm(nominal-s2)
    cosine = abs(d1@d2)
    linear = eps*np.sqrt(r1*r1+r2*r2+2*r1*r2*cosine)/sine
    inverse_norm = 1/np.sqrt(max(1-cosine, 1e-30))
    eta = 2*np.sqrt(2)*np.sin(eps/2)
    factor = inverse_norm*eta
    safe = inverse_norm*np.sin(eps)*np.hypot(r1, r2)/(1-factor) if factor < 1 else np.inf
    return nominal, linear, safe
