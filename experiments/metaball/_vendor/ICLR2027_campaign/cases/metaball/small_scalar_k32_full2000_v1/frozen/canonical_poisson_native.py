





from native_cupy_core import *
from cupyx.scipy.sparse import csr_matrix
import time


def geometry2(x, geo, c):
    _, e, chi = jet_geometry(x, geo, c)

    grad = lambda a: cp.stack([a[:, INDEX[tuple(int(k == j) for k in range(3))]] for j in range(3)], 1)
    lap = lambda a: sum(2*a[:, INDEX[tuple(2*int(k == j) for k in range(3))]] for j in range(3))
    hess = lambda a: cp.stack([cp.stack([a[:, INDEX[tuple(int(k == j)+int(k == l) for k in range(3))]]*(2 if j == l else 1) for l in range(3)], 1) for j in range(3)], 1)
    return e[:, 0], grad(e), lap(e), hess(e), chi[:, 0], grad(chi), lap(chi)


class ScalarPoisson:
    def __init__(self, c, geo, resolution, radius_cells=3., budget_seconds=420.):
        self.c, self.geo = c, geo
        self.nside = resolution
        self.dx, self.radius = 1/resolution, radius_cells/resolution
        self.deadline = time.perf_counter()+budget_seconds
        ijk = cp.indices((resolution,)*3).reshape(3, -1).T
        x = (ijk+.5)/resolution
        o, i = levels(x, geo, c)
        keep = (o > 0) & (i > 0)
        self.centers = x[keep]
        self.lookup = cp.full(resolution**3, -1, dtype='i4')
        self.lookup[cp.flatnonzero(keep)] = cp.arange(int(keep.sum()), dtype='i4')
        self.n = len(self.centers)
        self.center_e = envelope1(self.centers, geo, c)[0].value[:, 0]
        width = int(math.ceil(radius_cells))
        self.offsets = cp.asarray(list(itertools.product(range(-width, width+1), repeat=3)), dtype='i4')
        self.L = self.operator(self.centers, 'lap')
        self.diagonal = self.L.diagonal()
        if bool(cp.any(cp.abs(self.diagonal) < 1e-12)):
            raise FloatingPointError('Zero collocation diagonal; no regularization applied')
        self.LT = self.L.T.tocsr()
        self.solve_records = []

    def entries(self, x):
        ijk = cp.floor(x/self.dx-.5).astype('i4')[:, None, :]+self.offsets[None, :, :]
        valid = cp.all((ijk >= 0) & (ijk < self.nside), 2)
        flat = ((ijk[:, :, 0]*self.nside)+ijk[:, :, 1])*self.nside+ijk[:, :, 2]
        col = self.lookup[cp.clip(flat, 0, self.nside**3-1)]
        valid &= col >= 0
        delta = x[:, None, :]-self.centers[cp.maximum(col, 0)]
        r = cp.linalg.norm(delta, axis=2)/self.radius
        valid &= r < 1
        row, slot = cp.nonzero(valid)
        col = col[row, slot]
        delta, r = delta[row, slot], r[row, slot]
        t = 1-r

        f = t**8*(32*r**3+25*r**2+8*r+1)
        g = -22*t**7*(16*r**2+7*r+1)
        hh = 528*t**6*(6*r+1)
        grad = g[:, None]*delta/self.radius**2
        lap = (3*g+hh*r**2)/self.radius**2
        return row, col, delta, f, grad, lap, g, hh

    def operator(self, points, kind, chunk=512):
        vals, rows, cols = [], [], []
        for start in range(0, len(points), chunk):
            x = points[start:start+chunk]
            row, col, d, f, gf, lf, g, hh = self.entries(x)
            e, de, le, he, _, _, _ = geometry2(x, self.geo, self.c)
            if kind == 'lap':
                value = le[row]*f+2*cp.sum(de[row]*gf, 1)+e[row]*lf
                rr, cc = row+start, col
            elif kind == 'grad':
                value = (de[row]*f[:, None]+e[row, None]*gf).reshape(-1)
                rr = (3*(row+start)[:, None]+cp.arange(3)).reshape(-1)
                cc = cp.repeat(col, 3)
            elif kind == 'value':
                value, rr, cc = e[row]*f, row+start, col
            else:
                raise ValueError(kind)
            vals.append(value/self.center_e[cc]); rows.append(rr); cols.append(cc)
        return csr_matrix((cp.concatenate(vals), (cp.concatenate(rows), cp.concatenate(cols))),
                          shape=((3 if kind == 'grad' else 1)*len(points), self.n))

    def solve(self, b, transpose=False, tol=2e-10, maxiter=700):





        tic = time.perf_counter()
        mat = self.LT if transpose else self.L
        b = cp.asarray(b, dtype='f8')
        if b.ndim == 1:
            b = b[:, None]
        scale = cp.maximum(cp.linalg.norm(b, axis=0), 1e-300)
        rhs = b/scale
        x = cp.zeros_like(rhs); r = rhs.copy(); shadow = r.copy()
        p = cp.zeros_like(r); v = cp.zeros_like(r)
        rho_old = cp.ones(b.shape[1]); alpha = rho_old.copy(); omega = rho_old.copy()
        active = cp.linalg.norm(r, axis=0) > tol
        for iteration in range(1, maxiter+1):
            if time.perf_counter() > self.deadline:
                raise TimeoutError('Declared scalar component deadline exceeded')
            rho = cp.sum(shadow*r, axis=0)
            safe = lambda z: cp.where(cp.abs(z) > 1e-290, z, 1.)
            beta = rho/safe(rho_old)*alpha/safe(omega)
            p = cp.where(active, r+beta*(p-omega*v), 0.)
            ph = p/self.diagonal[:, None]
            v = mat@ph
            alpha = cp.where(active, rho/safe(cp.sum(shadow*v, axis=0)), 0.)
            s = r-alpha*v
            sh = s/self.diagonal[:, None]
            t = mat@sh
            omega = cp.where(active, cp.sum(t*s, axis=0)/safe(cp.sum(t*t, axis=0)), 0.)
            x += alpha*ph+omega*sh
            r = s-omega*t
            active &= cp.linalg.norm(r, axis=0) > tol
            rho_old = rho
            if iteration % 20 == 0 or not bool(cp.any(active)):
                actual = cp.linalg.norm(rhs-mat@x, axis=0)
                if not bool(cp.all(cp.isfinite(actual))):
                    raise FloatingPointError('Nonfinite Poisson residual')
                if float(cp.max(actual)) <= tol*5:
                    break
        actual = cp.linalg.norm(rhs-mat@x, axis=0)
        record = dict(transpose=transpose, rhs=b.shape[1], iterations=iteration,
                      max_relative_residual=float(actual.max()), seconds=time.perf_counter()-tic,
                      passed=bool(cp.all(actual <= tol*5)))
        self.solve_records.append(record)
        if not record['passed']:
            raise RuntimeError('Poisson inverse did not converge: '+json.dumps(record))
        return x*scale


class CanonicalProjection:
    def __init__(self, scalar, points):
        self.scalar = scalar
        self.G = scalar.operator(points, 'grad')
        self.L_eval = scalar.operator(points, 'lap')
        *_, dc, lc = geometry2(scalar.centers, scalar.geo, scalar.c)
        self.hcoeff = scalar.solve(lc)
        *_, dq, lq = geometry2(points, scalar.geo, scalar.c)
        self.h = dq.reshape(-1, 1)-self.G@self.hcoeff
        self.hdiv = lq[:, None]-self.L_eval@self.hcoeff
        self.hmass = cp.sum(self.h*self.h)
        if float(self.hmass) <= 0:
            raise FloatingPointError('Degenerate relative harmonic potential')

    def remove_harmonic(self, a):
        return a-self.h@((self.h.T@a)/self.hmass)

    def forward(self, a, div_centers):
        coeff = self.scalar.solve(div_centers)
        intermediate = a-self.G@coeff
        amplitude = (self.h.T@intermediate)/self.hmass
        return intermediate-self.h@amplitude, coeff, amplitude

    def backward(self, adj):





        ga = self.remove_harmonic(adj)
        gd = -self.scalar.solve(self.G.T@ga, transpose=True)
        return ga, gd
