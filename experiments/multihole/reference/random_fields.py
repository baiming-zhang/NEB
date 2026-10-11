"""Analytic random scalar fields, unchanged paper sampling and conditioning."""
import math
import numpy as np
from scipy.interpolate import RectBivariateSpline
from . import geometry as geo
CASES=("one_circle","circle_square","circle_square_triangle")
COLLAR=.35
LENGTH=4.
class Jet:
    """Taylor coefficients d_x^i d_y^j f/(i!j!), total order <= n."""

    def __init__(self, values, n=5):
        self.n = n
        self.keys = [(i, total - i) for total in range(n + 1) for i in range(total + 1)]
        self.index = {key: k for k, key in enumerate(self.keys)}
        values = np.asarray(values, dtype=float)
        self.a = np.zeros((len(values), len(self.keys)))
        self.a[:, 0] = values

    @classmethod
    def variable(cls, values, axis, n=5):
        obj = cls(values, n)
        obj.a[:, obj.index[(1, 0) if axis == 0 else (0, 1)]] = 1.0
        return obj

    def truncate(self, n):
        obj = Jet(self.a[:, 0], n)
        obj.a[:] = self.a[:, :len(obj.keys)]
        return obj

    def derivative(self, axis):
        obj = Jet(self.a[:, 0] * 0, self.n - 1)
        for k, (i, j) in enumerate(obj.keys):
            src = (i + 1, j) if axis == 0 else (i, j + 1)
            obj.a[:, k] = self.a[:, self.index[src]] * (i + 1 if axis == 0 else j + 1)
        return obj

    def __add__(self, other):
        obj = self.truncate(self.n)
        if isinstance(other, Jet):
            assert self.n == other.n
            obj.a += other.a
        else:
            obj.a[:, 0] += other
        return obj
    __radd__ = __add__

    def __neg__(self):
        obj = self.truncate(self.n)
        obj.a *= -1
        return obj

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        obj = Jet(self.a[:, 0] * 0, self.n)
        if not isinstance(other, Jet):
            obj.a = self.a * np.asarray(other)[..., None] if np.ndim(other) else self.a * other
            return obj
        assert self.n == other.n
        for k, (i, j) in enumerate(self.keys):
            for p in range(i + 1):
                for q in range(j + 1):
                    obj.a[:, k] += self.a[:, self.index[p, q]] * other.a[:, self.index[i - p, j - q]]
        return obj
    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * (other.power(-1) if isinstance(other, Jet) else 1 / other)

    def compose(self, coefficients):
        residual = self.truncate(self.n)
        residual.a[:, 0] = 0
        result = Jet(coefficients[0], self.n)
        term = Jet(np.ones(len(self.a)), self.n)
        for k in range(1, self.n + 1):
            term = term * residual
            result.a += term.a * coefficients[k][:, None]
        return result

    def power(self, exponent):
        if int(exponent) == exponent and exponent >= 0:
            result = Jet(np.ones(len(self.a)), self.n)
            base = self
            count = int(exponent)
            while count:
                if count & 1:
                    result = result * base
                count >>= 1
                if count:
                    base = base * base
            return result
        value = self.a[:, 0]
        coefficients = []
        factor = 1.0
        for k in range(self.n + 1):
            if k:
                factor *= (exponent - k + 1) / k
            coefficients.append(factor * value ** (exponent - k))
        return self.compose(coefficients)

    def log(self):
        value = self.a[:, 0]
        return self.compose([np.log(value)] + [(-1) ** (k - 1) / k * value ** (-k) for k in range(1, self.n + 1)])

    def exp(self):
        value = np.exp(self.a[:, 0])
        return self.compose([value / math.factorial(k) for k in range(self.n + 1)])

def coordinates(points, case):
    x, y = (Jet.variable(points[:, i], i) for i in range(2))
    q = [1 - ((x - 5) / 4).power(12) - (y / 4).power(12)]
    for spec in geo.get_obstacles(case):
        c, s = (np.cos(spec.rotation), np.sin(spec.rotation))
        xx = c * (x - spec.x) + s * (y - spec.y)
        yy = -s * (x - spec.x) + c * (y - spec.y)
        if spec.kind == 'circle':
            value = (xx.power(2) + yy.power(2)) / spec.size ** 2 - 1
        elif spec.kind == 'square':
            value = (xx / spec.size).power(spec.power) + (yy / spec.size).power(spec.power) - 1
        else:
            planes = [(np.cos(a) * xx + np.sin(a) * yy) / spec.size - 1 for a in np.pi / 2 + 2 * np.pi * np.arange(3) / 3]
            shift = np.max(np.stack([p.a[:, 0] for p in planes]), axis=0)
            value = sum((((p - shift) * spec.smooth_k).exp() for p in planes)).log() / spec.smooth_k + shift
        q.append(value)
    return q

def loops_for(case, n=1536):
    return [geo.outer_boundary(0.0, n)] + [geo.obstacle_boundary(s, n) for s in geo.get_obstacles(case)]

class RandomField:

    def __init__(self, case, sample=0):
        self.case = case
        self.sample = sample
        self.seed = [20260928, CASES.index(case), sample]
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        self.x = np.linspace(-1, 11, 25)
        self.y = np.linspace(-6, 6, 25)
        self.raw = self.rng.standard_normal((25, 25))
        self.spline = RectBivariateSpline(self.x, self.y, self.raw, kx=5, ky=5, s=0)
        self.loops = loops_for(case)
        self.constants = []
        for loop in self.loops:
            ds = 0.5 * (np.linalg.norm(loop - np.roll(loop, 1, axis=0), axis=1) + np.linalg.norm(loop - np.roll(loop, -1, axis=0), axis=1))
            self.constants.append(float(np.average(self.spline.ev(loop[:, 0], loop[:, 1]), weights=ds)))
        self.scale = 1.0

    def spline_jet(self, points):
        result = Jet(np.zeros(len(points)), 4)
        for k, (i, j) in enumerate(result.keys):
            result.a[:, k] = self.spline.ev(points[:, 0], points[:, 1], dx=i, dy=j) / (math.factorial(i) * math.factorial(j))
        return result

    def evaluate_block(self, points):
        q = coordinates(points, self.case)
        s = [(v / (1 + v.power(2)).power(0.5)).truncate(4) for v in q]
        products = []
        for j in range(len(s)):
            prod = Jet(np.ones(len(points)), 4)
            for i, value in enumerate(s):
                if i != j:
                    prod = prod * value
            products.append(prod)
        denominator = sum(products)
        raw = self.spline_jet(points)
        preliminary = raw
        correction_factor = Jet(np.zeros(len(points)), 2)
        for j, value in enumerate(q):
            ell = (1 + value).log()
            gx, gy = (ell.derivative(0), ell.derivative(1))
            ell = ell.truncate(4)
            norm2 = gx.power(2) + gy.power(2) + (ell / LENGTH).power(2) + 1e-30
            distance2 = ell.power(2) / norm2
            weight = (products[j] / denominator).power(3)
            collar = (-(distance2 / COLLAR ** 2).power(2)).exp()
            blend = weight * collar
            preliminary = preliminary + blend * (self.constants[j] - raw)
            correction_factor = correction_factor + (distance2 * blend * 0.5).truncate(2)
        lap = preliminary.derivative(0).derivative(0) + preliminary.derivative(1).derivative(1)
        field = preliminary.truncate(2) - correction_factor * lap
        field.a *= self.scale
        psi = field.a[:, 0]
        ux = field.a[:, field.index[0, 1]]
        uy = -field.a[:, field.index[1, 0]]
        omega = -2 * (field.a[:, field.index[2, 0]] + field.a[:, field.index[0, 2]])
        return np.column_stack((psi, ux, uy, omega))

    def evaluate(self, points, chunk=1024):
        points = np.asarray(points)
        output = np.empty((len(points), 4))
        for start in range(0, len(points), chunk):
            output[start:start + chunk] = self.evaluate_block(points[start:start + chunk])
        return output

def checks(field):
    records = []
    for j, loop in enumerate(loops_for(field.case, 2053)):
        values = field.evaluate(loop)
        grad = geo.outer_gradient(loop) if j == 0 else geo.obstacle_gradient(loop, geo.get_obstacles(field.case)[j - 1])
        normal = grad / np.linalg.norm(grad, axis=1)[:, None]
        records.append(dict(component=j, trace_spread=float(np.ptp(values[:, 0])), trace_expected=field.scale * field.constants[j], trace_max_error=float(np.max(abs(values[:, 0] - field.scale * field.constants[j]))), normal_speed_max=float(np.max(abs(np.sum(values[:, 1:3] * normal, axis=1)))), boundary_vorticity_max=float(np.max(abs(values[:, 3])))))
    rng = np.random.default_rng(47321)
    candidates = np.column_stack((rng.uniform(1.1, 8.9, 60), rng.uniform(-3.9, 3.9, 60)))
    points = candidates[geo.fluid_mask(candidates, field.case)][:10]
    eps = 2e-05
    center = field.evaluate(points)
    shifted = [field.evaluate(points + direction) for direction in (np.array([eps, 0]), np.array([-eps, 0]), np.array([0, eps]), np.array([0, -eps]))]
    px, mx, py, my = shifted
    numerical_velocity = np.column_stack(((py[:, 0] - my[:, 0]) / (2 * eps), -(px[:, 0] - mx[:, 0]) / (2 * eps)))
    numerical_omega = (px[:, 2] - mx[:, 2] - py[:, 1] + my[:, 1]) / (2 * eps)
    verror = np.linalg.norm(center[:, 1:3] - numerical_velocity) / max(np.linalg.norm(center[:, 1:3]), 1e-15)
    werror = np.linalg.norm(center[:, 3] - numerical_omega) / max(np.linalg.norm(center[:, 3]), 1e-15)
    assert max((r['normal_speed_max'] for r in records)) < 1e-08
    assert max((r['boundary_vorticity_max'] for r in records)) < 1e-07
    assert max((r['trace_max_error'] for r in records)) < 1e-09
    assert verror < 1e-05 and werror < 1e-05, (verror, werror)
    return dict(boundaries=records, analytic_velocity_relative_check=float(verror), analytic_vorticity_relative_check=float(werror), passed=True, divergence='Mixed derivatives of the same C2 scalar cancel analytically', scope='Initial-field kinematics only; no NS trajectory, GT convergence, harmonic energy or method errors evaluated')
