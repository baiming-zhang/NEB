import numpy as np,math,itertools
class Space:

    def __init__(self, n):
        self.nside = n
        self.dx = 1 / n
        self.radius = 2.5 / n
        xyz = (cp.indices((n,) * 3).reshape(3, -1).T + 0.5) / n
        o, i, *_ = mfs.levels(xyz)
        keep = (o > 0) & (i > 0)
        self.centers = xyz[keep]
        self.n = len(self.centers)
        self.lookup = cp.full(n ** 3, -1, dtype='i4')
        self.lookup[cp.flatnonzero(keep)] = cp.arange(self.n, dtype='i4')
        self.ce = shape_geometry(self.centers)[0]
        self.offset = cp.asarray(list(itertools.product(range(-3, 4), repeat=3)), dtype='i4')

    def operators(self, x, kinds=('grad', 'lap', 'value')):
        rows = {k: [] for k in kinds}
        cols = {k: [] for k in kinds}
        vals = {k: [] for k in kinds}
        for off in range(0, len(x), 512):
            xx = x[off:off + 512]
            idx = cp.floor(xx / self.dx - 0.5).astype('i4')[:, None, :] + self.offset[None, :, :]
            valid = cp.all((idx >= 0) & (idx < self.nside), 2)
            flat = (idx[:, :, 0] * self.nside + idx[:, :, 1]) * self.nside + idx[:, :, 2]
            col = self.lookup[cp.clip(flat, 0, self.nside ** 3 - 1)]
            valid &= col >= 0
            delta = xx[:, None, :] - self.centers[cp.maximum(col, 0)]
            r = cp.linalg.norm(delta, axis=2) / self.radius
            valid &= r < 1
            a, b = cp.nonzero(valid)
            cc = col[a, b]
            d = delta[a, b]
            r = r[a, b]
            t = 1 - r
            f = t ** 8 * (32 * r ** 3 + 25 * r * r + 8 * r + 1)
            g = -22 * t ** 7 * (16 * r * r + 7 * r + 1)
            hh = 528 * t ** 6 * (6 * r + 1)
            gf = g[:, None] * d / self.radius ** 2
            lf = (3 * g + hh * r * r) / self.radius ** 2
            e, de, le = shape_geometry(xx)
            for kind in kinds:
                if kind == 'grad':
                    v = (de[a] * f[:, None] + e[a, None] * gf) / self.ce[cc, None]
                    rr = (3 * (a + off)[:, None] + cp.arange(3)).ravel()
                    co = cp.repeat(cc, 3)
                    v = v.ravel()
                else:
                    v = (e[a] * f if kind == 'value' else le[a] * f + 2 * cp.sum(de[a] * gf, 1) + e[a] * lf) / self.ce[cc]
                    rr = a + off
                    co = cc
                vals[kind].append(v)
                rows[kind].append(rr)
                cols[kind].append(co)
        return {k: csr_matrix((cp.concatenate(vals[k]), (cp.concatenate(rows[k]), cp.concatenate(cols[k]))), shape=((3 if k == 'grad' else 1) * len(x), self.n)) for k in kinds}
class Normal:

    def __init__(self, G, w):
        self.G = G
        self.GT = G.T.tocsr()
        self.w = w
        self.diag = cp.asarray(G.multiply(G).T @ w).ravel()

    def diagonal(self):
        return self.diag

    def __matmul__(self, x):
        return self.GT @ (self.w[:, None] * (self.G @ x))
class SplineSpace:
    """Envelope times tensor cardinal cubic B-splines; C2 and analytic Laplacian."""

    def __init__(self, n):
        self.nside = n
        self.side = n + 3
        self.n = self.side ** 3
        self.active = None
        self.offset = cp.asarray(list(itertools.product(range(-1, 3), repeat=3)), dtype='i4')

    def operators(self, x, kinds=('grad', 'lap', 'value')):
        cols = {k: [] for k in kinds}
        vals = {k: [] for k in kinds}
        for off in range(0, len(x), 1024):
            xx = x[off:off + 1024]
            idx = cp.floor(xx * self.nside).astype('i4')[:, None, :] + self.offset[None, :, :]
            t = xx[:, None, :] * self.nside - idx
            a = cp.abs(t)
            sign = cp.sign(t)
            z = cp.maximum(2 - a, 0)
            v = cp.maximum(1 - a, 0)
            b = (z ** 3 - 4 * v ** 3) / 6
            db = sign * (-3 * z * z + 12 * v * v) / 6 * self.nside
            ddb = (6 * z - 24 * v) / 6 * self.nside ** 2
            phi = cp.prod(b, 2)
            grad = cp.stack([db[:, :, j] * b[:, :, (j + 1) % 3] * b[:, :, (j + 2) % 3] for j in range(3)], 2)
            lap = sum((ddb[:, :, j] * b[:, :, (j + 1) % 3] * b[:, :, (j + 2) % 3] for j in range(3)))
            e, de, le = shape_geometry(xx)
            cc = ((idx[:, :, 0] + 1) * self.side + idx[:, :, 1] + 1) * self.side + idx[:, :, 2] + 1
            rr = cp.broadcast_to(cp.arange(off, off + len(xx))[:, None], cc.shape)
            for kind in kinds:
                if kind == 'grad':
                    val = de[:, None, :] * phi[:, :, None] + e[:, None, None] * grad
                    col = cp.broadcast_to(cc[:, None, :], (len(xx), 3, 64)).ravel()
                    val = val.transpose(0, 2, 1).ravel()
                else:
                    val = (e[:, None] * phi if kind == 'value' else le[:, None] * phi + 2 * cp.sum(de[:, None, :] * grad, 2) + e[:, None] * lap).ravel()
                    col = cc.ravel()
                cols[kind].append(col.astype('i4'))
                vals[kind].append(val)
        out = {}
        for k in kinds:
            nr = (3 if k == 'grad' else 1) * len(x)
            out[k] = csr_matrix((cp.concatenate(vals[k]), cp.concatenate(cols[k]), cp.arange(nr + 1, dtype='i4') * 64), shape=(nr, self.side ** 3))
            vals[k].clear()
            cols[k].clear()
        if self.active is not None:
            out = {k: a[:, self.active].tocsr() for k, a in out.items()}
        return out
def two_level(K, s, xf, w):
    """Symmetric two-level preconditioner; exact cubic refinement prolongation."""
    from cupyx.scipy.linalg import solve_triangular
    nc = s.nside // 2
    coarse = SplineSpace(nc)
    GG = coarse.operators(xf, ('grad',))['grad']
    diag = cp.asarray(GG.multiply(GG).T @ w).ravel()
    coarse.active = cp.flatnonzero(diag > float(diag.max()) * 1e-05)
    coarse.n = len(coarse.active)
    GG = GG[:, coarse.active].tocsr()
    KC = None
    for off in range(0, GG.shape[0], 2048):
        Z = GG[off:off + 2048]
        part = (Z.T @ Z.multiply(w[off:off + 2048, None]).tocsr()).tocsr()
        KC = part if KC is None else KC + part
    scale = cp.sqrt(KC.diagonal())
    dense = KC.toarray() / scale[:, None] / scale[None, :]
    L = cp.linalg.cholesky(dense)
    del dense, KC, GG
    cp.get_default_memory_pool().free_all_blocks()
    fc = cp.asnumpy(s.active)
    lookup = np.full(s.side ** 3, -1)
    lookup[fc] = np.arange(len(fc))
    ids = cp.asnumpy(coarse.active)
    coords = np.stack((ids // coarse.side ** 2 - 1, ids // coarse.side % coarse.side - 1, ids % coarse.side - 1), 1)
    rr = []
    cc = []
    vv = []
    weights = [1, 4, 6, 4, 1]
    for d in itertools.product(range(-2, 3), repeat=3):
        fi = 2 * coords + np.array(d)
        ok = np.all((fi >= -1) & (fi <= s.nside + 1), 1)
        co = np.flatnonzero(ok)
        fi = fi[ok] + 1
        flat = (fi[:, 0] * s.side + fi[:, 1]) * s.side + fi[:, 2]
        rows = lookup[flat]
        good = rows >= 0
        rr.extend(rows[good])
        cc.extend(co[good])
        vv.extend([np.prod([weights[a + 2] for a in d]) / 512] * int(good.sum()))
    P = csr_matrix((cp.asarray(vv), (cp.asarray(rr, dtype='i4'), cp.asarray(cc, dtype='i4'))), shape=(s.n, coarse.n))
    PT = P.T.tocsr()
    diagonal = K.diagonal()

    def smooth(r):
        z = 0.08 * r / diagonal[:, None]
        res = r - K @ z
        rhs = PT @ res
        y = solve_triangular(L.T, solve_triangular(L, rhs / scale[:, None], lower=True), lower=False) / scale[:, None]
        z += P @ y
        z += 0.08 * (r - K @ z) / diagonal[:, None]
        return z
    K.precondition = smooth
    return dict(type='symmetric_two_level_cubic_refinement', coarse_resolution=nc, coarse_dof=coarse.n, jacobi_damping=0.08)
def solve(K, b, tol=2e-09, maxiter=1200):
    diag = K.diagonal()
    assert bool(cp.all(diag > 0))
    scale = cp.maximum(cp.linalg.norm(b, axis=0), 1e-30)
    b = b / scale
    pre = getattr(K, 'precondition', lambda r: r / diag[:, None])
    x = cp.zeros_like(b)
    r = b.copy()
    z = pre(r)
    p = z.copy()
    rz = cp.sum(r * z, 0)
    for it in range(maxiter):
        q = K @ p
        den = cp.sum(p * q, 0)
        active = cp.linalg.norm(r, axis=0) > tol
        alpha = cp.where(active, rz / cp.where(den != 0, den, 1), 0)
        x += p * alpha
        r -= q * alpha
        if it % 20 == 0 or not bool(cp.any(active)):
            actual = cp.linalg.norm(b - K @ x, axis=0)
            if float(actual.max()) < tol * 5:
                break
        z = pre(r)
        nrz = cp.sum(r * z, 0)
        p = z + p * cp.where(active, nrz / cp.where(rz != 0, rz, 1), 0)
        rz = nrz
    err = float(cp.linalg.norm(b - K @ x, axis=0).max())
    if err > tol * 5:
        raise RuntimeError(f'CG residual {err} after {it + 1}')
    return (x * scale, dict(iterations=it + 1, residual=err))
