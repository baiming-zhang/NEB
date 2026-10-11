





import numpy as np


def features(x, xp, periodic=False, descriptor=None):
    n, d = x.shape
    if periodic:
        assert d == 2
        f = xp.stack((xp.sin(x[:, 0]), xp.cos(x[:, 0]),
                      xp.sin(x[:, 1]), xp.cos(x[:, 1])), axis=1)
        j = xp.zeros((n, d, 4), dtype=x.dtype)
        j[:, 0, 0] = xp.cos(x[:, 0]); j[:, 0, 1] = -xp.sin(x[:, 0])
        j[:, 1, 2] = xp.cos(x[:, 1]); j[:, 1, 3] = -xp.sin(x[:, 1])
    else:
        f = 2 * x - 1
        j = xp.broadcast_to(2 * xp.eye(d, dtype=x.dtype), (n, d, d)).copy()
    if descriptor is not None:
        desc = xp.asarray(descriptor, dtype=x.dtype).ravel()
        f = xp.concatenate((f, xp.broadcast_to(desc, (n, len(desc)))), axis=1)
        j = xp.concatenate((j, xp.zeros((n, d, len(desc)), dtype=x.dtype)), axis=2)
    return f, j


class Model:
    def __init__(self, xp, input_dim, dim, modes, seed=0, width=256, hidden=7, dtype='float32'):
        self.xp, self.dim, self.modes = xp, dim, modes
        self.width, self.hidden, self.dtype, self.step = width, hidden, dtype, 0
        rng = np.random.default_rng(seed)
        sizes = [input_dim] + [width] * hidden + [dim * modes]
        self.W = [xp.asarray(rng.normal(size=(a, b)) * np.sqrt(2 / ((1 + .01**2) * a)), dtype=dtype)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [xp.zeros(b, dtype=dtype) for b in sizes[1:]]
        self.params = self.W + self.b
        self.first = [xp.zeros_like(x) for x in self.params]
        self.second = [xp.zeros_like(x) for x in self.params]

    def forward(self, f, j=None, record=False):
        xp = self.xp
        tape = []
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = f @ w + b
            dz = None if j is None else j.reshape(-1, j.shape[-1]) @ w
            if dz is not None: dz = dz.reshape(len(f), self.dim, -1)
            if i == len(self.W) - 1:
                e = xp.exp(xp.minimum(z, 0))
                p = xp.where(z >= 0, 1, e)
                p2 = xp.where(z >= 0, 0, e)
                a = xp.where(z >= 0, z, e - 1)
            else:
                p = xp.where(z >= 0, 1., .01).astype(z.dtype)
                p2 = None
                a = z * p
            if record: tape.append((f, j, dz, p, p2))
            f = a
            j = None if dz is None else dz * p[:, None, :]
        v = f.reshape(len(f), self.modes, self.dim)
        jac = None if j is None else j.reshape(len(f), self.dim, self.modes, self.dim).transpose(0, 2, 3, 1)
        return v, jac, tape

    def backward(self, tape, gv, gj):
        xp = self.xp
        ga = gv.reshape(len(gv), -1)
        gd = gj.transpose(0, 3, 1, 2).reshape(len(gv), self.dim, -1)
        gw, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            a, j, dz, p, p2 = tape[i]
            gz = ga * p
            if p2 is not None: gz += xp.sum(gd * dz, axis=1) * p2
            gdz = gd * p[:, None, :]
            gw[i] = a.T @ gz + j.reshape(-1, j.shape[-1]).T @ gdz.reshape(-1, gdz.shape[-1])
            gb[i] = gz.sum(axis=0)
            if i:
                ga = gz @ self.W[i].T
                gd = (gdz.reshape(-1, gdz.shape[-1]) @ self.W[i].T).reshape(len(a), self.dim, -1)
        return gw + gb

    def adam(self, gradients, lr):
        xp = self.xp
        self.step += 1
        norm = xp.sqrt(sum(xp.sum(g.astype('float64')**2) for g in gradients))
        if not bool(xp.isfinite(norm)): raise FloatingPointError('Nonfinite NKB parameter gradient')
        for p, m, v, g in zip(self.params, self.first, self.second, gradients):
            m *= .9; m += .1 * g
            v *= .999; v += .001 * g * g
            p -= lr * (m / (1 - .9**self.step)) / (xp.sqrt(v / (1 - .999**self.step)) + 1e-8)
        return float(norm)

    def save(self, path):
        from _neb_paths import Path
        path = Path(path)
        host = lambda x: np.asarray(x) if self.xp is np else self.xp.asnumpy(x)
        arrays = {f'{kind}_{i}': host(v) for kind, vs in [('W', self.W), ('b', self.b),
                  ('first', self.first), ('second', self.second)] for i, v in enumerate(vs)}
        arrays['step'] = np.int64(self.step)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, **arrays)
        import time
        deadline = time.monotonic()+10
        while True:
            try:
                tmp.replace(path); break
            except PermissionError:
                if time.monotonic() >= deadline: raise
                time.sleep(.05)

    def load(self, path):
        with np.load(path) as z:
            for kind, vs in [('W', self.W), ('b', self.b), ('first', self.first), ('second', self.second)]:
                for i, v in enumerate(vs): v[...] = self.xp.asarray(z[f'{kind}_{i}'])
            self.step = int(z['step'])


def losses(v, j, weight, boundary_weight, normal, xp, need_grad=True):
    n, k, d = v.shape
    w = weight / xp.maximum(weight.sum(), 1e-8)
    wb = boundary_weight / xp.maximum(boundary_weight.sum(), 1e-8)
    div = sum(j[:, :, a, a] for a in range(d))
    length = xp.sqrt(xp.sum(v * v, axis=2))
    safe = xp.maximum(length, 1e-8)
    mean_length = xp.sum(w[:, None] * length, axis=0)
    dot = xp.sum(v * normal[:, None, :], axis=2)
    cosine = dot / safe

    gram = xp.zeros((k, k), dtype='float64')
    for a in range(d):
        va = v[:, :, a].astype('float64')
        gram += va.T @ (w[:, None] * va)
    off = gram.copy()
    off[xp.arange(k), xp.arange(k)] = 0
    pair_den = max(1, k * (k - 1))
    raw = dict(smooth=xp.sum(w[:, None, None, None] * j*j) / k,
               div=xp.sum(w[:, None] * div*div) / k,
               orth=xp.sum(off*off) / pair_den,
               bc=xp.sum(wb[:, None] * cosine*cosine) / k,
               length=xp.sum((mean_length - .37)**2) / k,
               small=xp.sum(w[:, None] * xp.maximum(.05 - length, 0)) / k)
    weights = dict(smooth=.01, div=5., orth=100., bc=30., length=100., small=100.)
    raw['loss'] = sum(weights[key] * raw[key] for key in weights)
    raw['mean_basis_length'] = mean_length.mean()
    raw['gram_diagonal_min'] = xp.diag(gram).min()
    if not need_grad: return raw, None, None
    gv = xp.zeros_like(v)
    gj = 2 * .01 / k * w[:, None, None, None] * j
    for a in range(d):
        gj[:, :, a, a] += 10 / k * w[:, None] * div
        gv[:, :, a] += (400 / pair_den * w[:, None] * (v[:, :, a].astype('float64') @ off)).astype(v.dtype)
    radial = 200 / k * w[:, None] * (mean_length[None, :] - .37)
    radial -= 100 / k * w[:, None] * (length < .05)
    gv += radial[:, :, None] * v / safe[:, :, None]

    bcgrad = 2 * dot[:, :, None] / safe[:, :, None]**2 * normal[:, None, :]
    bcgrad -= (2 * dot*dot / safe**4 * (length > 1e-8))[:, :, None] * v
    gv += 30 / k * wb[:, None, None] * bcgrad
    return raw, gv, gj
