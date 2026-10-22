from _neb_paths import Path
import sys
import math
import itertools
import json
from types import SimpleNamespace
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
LEGACY_BACKEND = WORK
sys.path.insert(0, str(LEGACY_BACKEND))
from native_kernels.basisnet.analytic_jets import hardcoded_backend as old
Jet1 = old.Jet1
OLD = WORK / 'metaball_reference/cases/3D_Metaball/results/k2048_siren_freeslip_e2000_w128_d5/run_001'
DNS = WORK / 'metaball_reference/cases/3D_Metaball/results/independent_freeslip_dns_gt_128'
OUT = ROOT / 'results/smoke32_native_cupy_v1'

def write(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    q.replace(p)

def status(phase, progress):
    import os, time
    write(ROOT / 'status.json', dict(updated_unix=time.time(), phase=phase, process=os.getpid(), backend='native_cupy', progress=progress, run=str(OUT), new_K2048_held=True, next='finite K32/2000 then immediate full evaluation'))
    (ROOT / 'STATUS.md').write_text('# Metaball native CuPy\n\n' + phase + '\n\n' + json.dumps(progress, indent=2), encoding='utf-8')

def config_namespace():
    return SimpleNamespace(**json.loads((OLD / 'config.json').read_text()))

def levels(x, geo, c):
    outer = 1 - cp.sum(((x - 0.5) / c.outer_half_extent) ** c.outer_superellipsoid_power, axis=1)
    d = cp.linalg.norm(x[:, None] - geo[None, :, :3], axis=2) - geo[None, :, 3]
    low = cp.min(d, axis=1)
    inner = low - c.metaball_softmin_tau * cp.log(cp.sum(cp.exp(-(d - low[:, None]) / c.metaball_softmin_tau), axis=1))
    return (outer, inner)

def fluid_points(c, geometry, count, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    pieces = []
    remain = count
    while remain:
        points = rng.random((max(2 * remain, 256), 3), dtype=np.float32)
        o, i = levels(cp.asarray(points), cp.asarray(geometry), c)
        accepted = points[cp.asnumpy((o > 0) & (i > 0))][:remain]
        pieces.append(accepted)
        remain -= len(accepted)
    return np.concatenate(pieces)

def envelope1(x, geo, c):
    q = (x - 0.5) / c.outer_half_extent
    out = 1 - cp.sum(q ** c.outer_superellipsoid_power, 1)
    go = -c.outer_superellipsoid_power / c.outer_half_extent * q ** (c.outer_superellipsoid_power - 1)
    delta = x[:, None] - geo[None, :, :3]
    length = cp.linalg.norm(delta, axis=2)
    d = length - geo[None, :, 3]
    small = cp.min(d, 1)
    weights = cp.exp(-(d - small[:, None]) / c.metaball_softmin_tau)
    weights /= weights.sum(1)[:, None]
    inner = small - c.metaball_softmin_tau * cp.log(cp.exp(-(d - small[:, None]) / c.metaball_softmin_tau).sum(1))
    gi = cp.sum(weights[:, :, None] * delta / length[:, :, None], 1)
    fo = out / (c.outer_envelope_transition + out)
    fi = inner / (c.obstacle_envelope_transition + inner)
    dfo = c.outer_envelope_transition * go / (c.outer_envelope_transition + out)[:, None] ** 2
    dfi = c.obstacle_envelope_transition * gi / (c.obstacle_envelope_transition + inner)[:, None] ** 2
    e = fo * fi
    de = dfo * fi[:, None] + dfi * fo[:, None]
    chi = fo / (fo + fi)
    dc = (dfo * fi[:, None] - dfi * fo[:, None]) / (fo + fi)[:, None] ** 2
    return (Jet1(e[:, None], de[:, None, :]), chi, dc)

def features1(x, geo, c):
    cond = 2 * cp.concatenate(((geo[:, :3] - c.sphere_center_min) / (c.sphere_center_max - c.sphere_center_min), (geo[:, 3:] - c.sphere_radius_min) / (c.sphere_radius_max - c.sphere_radius_min)), axis=1).reshape(-1) - 1
    value = cp.concatenate((2 * x - 1, cp.broadcast_to(cond, (len(x), 12))), axis=1)
    grad = cp.zeros((len(x), 15, 3), dtype=x.dtype)
    for j in range(3):
        grad[:, j, j] = 2
    return Jet1(value, grad)

def linear1(source, W, b):
    axis = 1 if source.value.ndim == 3 else 0
    n = source.value.shape[axis]
    pack = cp.concatenate([source.value, *[source.grad[..., j] for j in range(3)]], axis=axis)
    y = cp.matmul(pack, W.transpose(0, 2, 1)).reshape(len(W), 4, n, W.shape[1])
    return Jet1(y[:, 0] + b[:, None, :], y[:, 1:].transpose(0, 2, 3, 1))

def scale1(x, s):
    return Jet1(x.value * s, x.grad * s)

def product1(e, f):
    return Jet1(e.value * f.value, e.value[..., None] * f.grad + f.value[..., None] * e.grad)

def curl1(field):
    g = field.grad
    return cp.stack((g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]), axis=-1)

def curl_adjoint(u):
    g = cp.zeros((*u.shape[:-1], 3, 3), dtype=u.dtype)
    g[..., 2, 1] = u[..., 0]
    g[..., 1, 2] = -u[..., 0]
    g[..., 0, 2] = u[..., 1]
    g[..., 2, 0] = -u[..., 1]
    g[..., 1, 0] = u[..., 2]
    g[..., 0, 1] = -u[..., 2]
    return g

class Model:

    def __init__(self, c, K=32, seed=0):
        self.c = c
        self.K = K
        self.epoch = 0
        self.W = []
        self.b = []
        shapes = [15] + [c.hidden] * c.hidden_layers + [3]
        generators = [np.random.Generator(np.random.PCG64(seed + 7919 * (i + 1))) for i in range(K)]
        for j, (inp, out) in enumerate(zip(shapes[:-1], shapes[1:])):
            bound = 1 / inp if j == 0 else math.sqrt(6 / inp) / c.siren_hidden_omega0
            self.W.append(cp.asarray(np.stack([r.uniform(-bound, bound, (out, inp)).astype('f4') for r in generators])))
            self.b.append(cp.asarray(np.stack([r.uniform(-bound, bound, out).astype('f4') for r in generators])))
        self.mW = [cp.zeros_like(w) for w in self.W]
        self.vW = [cp.zeros_like(w) for w in self.W]
        self.mb = [cp.zeros_like(b) for b in self.b]
        self.vb = [cp.zeros_like(b) for b in self.b]

    def forward(self, data, start, stop, cache=False):
        x = data.features
        history = []
        for j in range(len(self.W) - 1):
            omega = self.c.siren_first_omega0 if j == 0 else self.c.siren_hidden_omega0
            pre = scale1(linear1(x, self.W[j][start:stop], self.b[j][start:stop]), omega)
            if cache:
                history.append((x, pre, omega))
            x = old.sine_forward_jet1(pre)
        raw = linear1(x, self.W[-1][start:stop], self.b[-1][start:stop])
        return (product1(data.envelope, raw), (x, history) if cache else None)

    def backward(self, data, start, stop, cache, adj):
        source, history = cache
        a = old.envelope_product_backward_jet1(data.envelope, adj)
        a, w, b = old.linear_backward_batch_jet1(source, a, self.W[-1][start:stop])
        weights = [w]
        biases = [b]
        for j in range(len(history) - 1, -1, -1):
            source, pre, omega = history[j]
            a = scale1(old.sine_backward_jet1(pre, a), omega)
            a, w, b = old.linear_backward_batch_jet1(source, a, self.W[j][start:stop])
            weights.append(w)
            biases.append(b)
        return (weights[::-1], biases[::-1])

    def adam(self, gW, gb, lr, clip=8.0):
        norms = cp.sqrt(sum((cp.sum(g.reshape(self.K, -1) ** 2, axis=1) for g in [*gW, *gb])))
        if not bool(cp.all(cp.isfinite(norms) & (norms > 0))):
            raise FloatingPointError('Invalid parameter gradients; no optimizer update')
        scales = cp.minimum(1, clip / cp.maximum(norms, 1e-30))
        t = self.epoch + 1
        for pars, moms, vars, grads in [(self.W, self.mW, self.vW, gW), (self.b, self.mb, self.vb, gb)]:
            for p, m, v, g in zip(pars, moms, vars, grads):
                g = g * scales.reshape((self.K,) + (1,) * (g.ndim - 1))
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                p -= lr * (m / (1 - 0.9 ** t)) / (cp.sqrt(v / (1 - 0.999 ** t)) + 1e-08)
        self.epoch = t
        return norms

    def save(self, path, configuration=None):
        arrays = {f'{name}_{j}': cp.asnumpy(a) for name in ['W', 'b', 'mW', 'vW', 'mb', 'vb'] for j, a in enumerate(getattr(self, name))}
        arrays.update(epoch=np.asarray(self.epoch))
        tmp = Path(path).with_suffix('.tmp.npz')
        np.savez(tmp, **arrays)
        tmp.replace(path)

    def load(self, path, configuration=None):
        with np.load(path) as z:
            self.epoch = int(z['epoch'])
            for name in ['W', 'b', 'mW', 'vW', 'mb', 'vb']:
                for j, a in enumerate(getattr(self, name)):
                    a[:] = cp.asarray(z[f'{name}_{j}'])

class Data:

    def __init__(self, c, geometry, count, seed, order=5, points=None):
        self.points = fluid_points(c, geometry, count, seed) if points is None else points
        self.n = len(self.points)
        self.x = cp.asarray(self.points)
        self.geo = cp.asarray(geometry, dtype='f4')
        self.features = features1(self.x, self.geo, c)
        self.envelope, _, _ = envelope1(self.x, self.geo, c)
        x = self.x.astype('f8')
        e, _, dc = envelope1(x, self.geo.astype('f8'), c)
        y = 2 * x - 1
        vals = [cp.ones_like(y), y]
        ders = [cp.zeros_like(y), 2 * cp.ones_like(y)]
        for m in range(1, order - 1):
            vals.append(((2 * m + 1) * y * vals[-1] - m * vals[-2]) / (m + 1))
            ders.append(((2 * m + 1) * (2 * vals[-2] + y * ders[-1]) - m * ders[-2]) / (m + 1))
        v = cp.stack(vals[:order], 1)
        d = cp.stack(ders[:order], 1)
        mi = cp.asarray(list(itertools.product(range(order), repeat=3)))
        p = cp.stack([v[:, mi[:, j], j] for j in range(3)], 2)
        dp = cp.stack([d[:, mi[:, j], j] for j in range(3)], 2)
        value = p.prod(2)
        gradient = cp.stack([dp[:, :, j] * p[:, :, (j + 1) % 3] * p[:, :, (j + 2) % 3] for j in range(3)], 2)
        g = e.grad[:, 0, None, :] * value[:, :, None] + e.value[:, 0, None, None] * gradient
        self.G = cp.concatenate((g, dc[:, None, :]), axis=1).transpose(0, 2, 1).reshape(3 * self.n, -1)
        self.scales = cp.linalg.norm(self.G, axis=0)
        self.B = self.G / self.scales
        self.gram = self.B.T @ self.B
        self.factor = cp.linalg.cholesky(self.gram)

    def solve(self, rhs):
        return solve_triangular(self.factor.T, solve_triangular(self.factor, rhs, lower=True), lower=False)

    def project(self, a, coeff=False):
        a = a.astype('f8', copy=False)
        c = self.solve(self.B.T @ a)
        p = a - self.B @ c
        return (p, c / self.scales[:, None]) if coeff else p

def peel(A, U, data, adjoints=False):
    n = data.n
    K = A.shape[1]
    P = data.project(A)
    order = cp.argsort(cp.sum(U * U, axis=0) / cp.sum(P * P, axis=0), kind='stable')
    Q = cp.empty_like(P)
    V = cp.empty_like(U)
    T = cp.zeros((K, K), dtype='f8')
    GA = cp.zeros_like(P)
    GU = cp.zeros_like(U)
    losses = cp.empty(K)
    masses = cp.empty(K)
    for rank, index in enumerate(cp.asnumpy(order)):
        p = P[:, index]
        v = U[:, index]
        normal = cp.sqrt(cp.dot(p, p) / n)
        a = p / normal
        u = v / normal
        t = cp.zeros(K, dtype='f8')
        t[index] = 1 / normal
        q = Q[:, :rank]
        w = V[:, :rank]
        for _ in range(2):
            alpha = q.T @ a / n
            a -= q @ alpha
            u -= w @ alpha
            t -= T[:, :rank] @ alpha
        mass = cp.dot(a, a) / n
        stiff = cp.dot(u, u) / n
        losses[rank] = stiff / mass
        masses[rank] = mass
        if adjoints:
            ga = -2 * stiff * a / (n * mass * mass)
            gu = 2 * u / (n * mass)
            for _ in range(2):
                ga -= q @ (q.T @ ga + w.T @ gu) / n
            gp = ga / normal - p * (cp.dot(ga, p) + cp.dot(gu, v)) / (n * normal ** 3)
            GA[:, index] = gp
            GU[:, index] = gu / normal
        norm = cp.sqrt(mass)
        Q[:, rank] = a / norm
        V[:, rank] = u / norm
        T[:, rank] = t / norm
    if not bool(cp.all(cp.isfinite(losses) & (masses > 1e-12))):
        raise FloatingPointError('Invalid canonical quotient/rank')
    return dict(losses=losses, order=order, T=T, Q=Q, U=V, GA=data.project(GA), GU=GU, gram_max=cp.max(cp.abs(Q.T @ Q / n - cp.eye(K))))

def train_step(model, data, lr, block=4):
    K = model.K
    A = cp.empty((3 * data.n, K), dtype='f8')
    U = cp.empty_like(A)
    for start in range(0, K, block):
        stop = min(K, start + block)
        f, _ = model.forward(data, start, stop)
        A[:, start:stop] = f.value.reshape(stop - start, -1).T
        U[:, start:stop] = curl1(f).reshape(stop - start, -1).T
    result = peel(A, U, data, True)
    gW = [cp.empty_like(w) for w in model.W]
    gb = [cp.empty_like(b) for b in model.b]
    for start in range(0, K, block):
        stop = min(K, start + block)
        _, cache = model.forward(data, start, stop, True)
        ga = result['GA'][:, start:stop].T.reshape(stop - start, data.n, 3).astype('f4')
        gu = result['GU'][:, start:stop].T.reshape(stop - start, data.n, 3).astype('f4')
        dw, db = model.backward(data, start, stop, cache, Jet1(ga, curl_adjoint(gu)))
        for j in range(len(gW)):
            gW[j][start:stop] = dw[j]
            gb[j][start:stop] = db[j]
    norms = model.adam(gW, gb, lr)
    return dict(mean_rayleigh=float(cp.mean(result['losses'])), max_rayleigh=float(cp.max(result['losses'])), min_rayleigh=float(cp.min(result['losses'])), gram_max=float(result['gram_max']), gradient_norm_min=float(cp.min(norms)), gradient_norm_max=float(cp.max(norms)), order=cp.asnumpy(result['order']).tolist())
MULTI = [m for total in range(4) for m in itertools.product(range(4), repeat=3) if sum(m) == total]
INDEX = {m: i for i, m in enumerate(MULTI)}
PAIRS = [[(INDEX[b], INDEX[tuple((a[j] - b[j] for j in range(3)))]) for b in MULTI if all((b[j] <= a[j] for j in range(3)))] for a in MULTI]

def jconst(value):
    return cp.concatenate((cp.asarray(value)[..., None], cp.zeros((*cp.asarray(value).shape, 19), dtype=cp.asarray(value).dtype)), axis=-1)

def jmul(a, b):
    return cp.stack([sum((a[..., i] * b[..., j] for i, j in pairs)) for pairs in PAIRS], axis=-1)

def junary(a, values):
    d = a.copy()
    d[..., 0] = 0
    d2 = jmul(d, d)
    d3 = jmul(d2, d)
    return jconst(values[0]) + values[1][..., None] * d + values[2][..., None] * d2 / 2 + values[3][..., None] * d3 / 6

def jsin(a):
    s = cp.sin(a[..., 0])
    c = cp.cos(a[..., 0])
    return junary(a, (s, c, -s, -c))

def jexp(a):
    e = cp.exp(a[..., 0])
    return junary(a, (e, e, e, e))

def jlog(a):
    x = a[..., 0]
    return junary(a, (cp.log(x), 1 / x, -1 / x ** 2, 2 / x ** 3))

def jpow(a, p):
    x = a[..., 0]
    return junary(a, (x ** p, p * x ** (p - 1), p * (p - 1) * x ** (p - 2), p * (p - 1) * (p - 2) * x ** (p - 3)))

def jdiv(a, b):
    return jmul(a, jpow(b, -1))

def jet_geometry(points, geometry, c):
    xyz = []
    for j in range(3):
        x = jconst(points[:, j])
        x[:, INDEX[tuple((int(k == j) for k in range(3)))]] = 1
        xyz.append(x)
    out = jconst(cp.ones(len(points), dtype=points.dtype))
    for x in xyz:
        out -= jpow((x - jconst(cp.full(len(points), 0.5))) / c.outer_half_extent, c.outer_superellipsoid_power)
    distances = []
    for center in geometry:
        r = sum((jmul(x - jconst(cp.full(len(points), center[j])), x - jconst(cp.full(len(points), center[j]))) for j, x in enumerate(xyz)))
        distances.append(jpow(r, 0.5) - jconst(cp.full(len(points), center[3])))
    shift = cp.min(cp.stack([d[:, 0] for d in distances]), axis=0)
    inner = jconst(shift) - c.metaball_softmin_tau * jlog(sum((jexp(-(d - jconst(shift)) / c.metaball_softmin_tau) for d in distances)))
    fo = jdiv(out, out + jconst(cp.full(len(points), c.outer_envelope_transition)))
    fi = jdiv(inner, inner + jconst(cp.full(len(points), c.obstacle_envelope_transition)))
    return (xyz, jmul(fo, fi), jdiv(fo, fo + fi))

def higher_fields(model, points, geometry, start=0, stop=None):
    stop = model.K if stop is None else stop
    x = cp.asarray(points, dtype='f8')
    geo = cp.asarray(geometry, dtype='f8')
    c = model.c
    xyz, e, _ = jet_geometry(x, geo, c)
    first = features1(x, geo, c).value
    f = jconst(first)
    for j in range(3):
        f[:, j, INDEX[tuple((int(k == j) for k in range(3)))]] = 2
    for layer in range(len(model.W)):
        f = cp.einsum('...ni,bfn->b...fi', f, model.W[layer][start:stop].astype('f8')) if layer == 0 else cp.einsum('bnfi,bof->bnoi', f, model.W[layer][start:stop].astype('f8'))
        f[..., 0] += model.b[layer][start:stop, None, :]
        if layer < len(model.W) - 1:
            f = jsin(f * (c.siren_first_omega0 if layer == 0 else c.siren_hidden_omega0))
    return jmul(e[None, :, None, :], f)

def derivative(field, component, multi):
    return field[..., component, INDEX[tuple(multi)]] * math.prod((math.factorial(i) for i in multi))

def jet_velocity(field, extra=(0, 0, 0)):
    values = []
    for j in range(3):
        a = (j + 1) % 3
        b = (j + 2) % 3
        da = list(extra)
        db = list(extra)
        da[a] += 1
        db[b] += 1
        values.append(derivative(field, b, da) - derivative(field, a, db))
    return cp.stack(values, axis=-1)

def jet_vorticity(field, extra=(0, 0, 0)):
    out = []
    for j in range(3):
        a = (j + 1) % 3
        b = (j + 2) % 3
        da = list(extra)
        db = list(extra)
        da[a] += 1
        db[b] += 1
        out.append(jet_velocity(field, da)[..., b] - jet_velocity(field, db)[..., a])
    return cp.stack(out, axis=-1)

def scalar_gauge_jets(points, geometry, c, order):
    x = cp.asarray(points, dtype='f8')
    xyz, e, chi = jet_geometry(x, cp.asarray(geometry, dtype='f8'), c)
    values = []
    for coordinate in xyz:
        y = 2 * coordinate - jconst(cp.ones(len(x)))
        p = [jconst(cp.ones(len(x))), y]
        for m in range(1, order - 1):
            p.append(((2 * m + 1) * jmul(y, p[-1]) - m * p[-2]) / (m + 1))
        values.append(p)
    fields = [jmul(e, jmul(values[0][i], jmul(values[1][j], values[2][k]))) for i, j, k in itertools.product(range(order), repeat=3)]
    return cp.stack([*fields, chi], axis=1)
