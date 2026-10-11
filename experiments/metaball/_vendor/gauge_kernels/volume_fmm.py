import numpy as np,math,itertools
import cupy as cp
'Cartesian volume FMM: exact constant-cuboid near field; harmonic far Taylor.\n\nP2M -> M2M -> M2L -> L2L -> L2P, with an exact transpose of this discretization.\nVolume density is piecewise constant, separately assessed from FMM truncation.\nThis is a uniform-tree FMM; optional source sampling adaptation is independent.\n'
def powers(p):
    return [a for degree in range(p + 1) for a in itertools.product(range(degree + 1), repeat=3) if sum(a) == degree]
def choose(a, b):
    return math.prod((math.comb(x, y) if x >= y else 0 for x, y in zip(a, b)))
def kernel_series(d, p):
    """Taylor coefficients of 1/|d+z| via binomial polynomial expansion."""
    aa = powers(p)
    index = {a: j for j, a in enumerate(aa)}
    rr = sum((t * t for t in d))
    terms = []
    for j in range(3):
        a = [0] * 3
        a[j] = 1
        terms.append((tuple(a), 2 * d[j] / rr))
        a[j] = 2
        terms.append((tuple(a), 1 / rr))
    result = np.zeros(len(aa))
    poly = {(0, 0, 0): 1.0}
    coef = 1.0
    for k in range(p + 1):
        for a, v in poly.items():
            result[index[a]] += coef * v / math.sqrt(rr)
        nxt = {}
        for a, v in poly.items():
            for b, w in terms:
                c = tuple((x + y for x, y in zip(a, b)))
                if sum(c) <= p:
                    nxt[c] = nxt.get(c, 0.0) + v * w
        poly = nxt
        coef *= (-0.5 - k) / (k + 1)
    return (result, index)
_cube = cp.RawKernel('\nextern "C" __global__ void cube(const double* x,const int* cells,double* out,\n int nq,int ns,int n,double lo,double dx) {\n long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;\n if(id>=(long long)nq*ns)return;int q=id/ns,s=id%ns,cell=cells[id];\n if(cell<0){for(int j=0;j<4;j++)out[((long long)q*4+j)*ns+s]=0.;return;}\n int ix=cell/(n*n),iy=(cell/n)%n,iz=cell%n;double v=0.,gx=0.,gy=0.,gz=0.;\n for(int a=0;a<2;a++)for(int b=0;b<2;b++)for(int c=0;c<2;c++){\n  double X=lo+(ix+a)*dx-x[3*q],Y=lo+(iy+b)*dx-x[3*q+1],Z=lo+(iz+c)*dx-x[3*q+2];\n  double R=sqrt(X*X+Y*Y+Z*Z);double lx=log(fmax(R+X,1e-300)),ly=log(fmax(R+Y,1e-300)),lz=log(fmax(R+Z,1e-300));\n  double ax=(X==0?0.:atan((Y*Z)/(X*R))),ay=(Y==0?0.:atan((X*Z)/(Y*R))),az=(Z==0?0.:atan((X*Y)/(Z*R)));\n  double sign=((a+b+c)%2?1.:-1.);\n  v+=sign*(X*Y*lz+Y*Z*lx+Z*X*ly-.5*(X*X*ax+Y*Y*ay+Z*Z*az));\n  gx+=sign*(Y*lz+Z*ly-X*ax);gy+=sign*(Z*lx+X*lz-Y*ay);gz+=sign*(X*ly+Y*lx-Z*az);\n }\n double pi=3.14159265358979323846;\n out[((long long)q*4)*ns+s]=-v/(4*pi);\n out[((long long)q*4+1)*ns+s]=gx/(4*pi);\n out[((long long)q*4+2)*ns+s]=gy/(4*pi);\n out[((long long)q*4+3)*ns+s]=gz/(4*pi);\n}\n', 'cube')
def cube_matrix(x, cells, n, lo, length):
    x = cp.ascontiguousarray(x, dtype='f8')
    cells = cp.ascontiguousarray(cells, dtype='i4')
    q, s = cells.shape
    out = cp.empty((q, 4, s), dtype='f8')
    _cube(((q * s + 255) // 256,), (256,), (x, cells, out, q, s, n, float(lo), float(length / n)))
    return out
class VolumeFMM:

    def __init__(self, n, p=4, lo=-0.02, length=1.04):
        assert n >= 8 and n % 4 == 0 and (n // 4 & n // 4 - 1 == 0)
        self.n = n
        self.p = p
        self.lo = lo
        self.length = length
        self.depth = int(math.log2(n // 4))
        self.a = powers(p)
        self.P = len(self.a)
        self.bin = np.array([[choose(a, b) for b in self.a] for a in self.a])
        self.children = {}
        self.up = []
        self.down = []
        for child in itertools.product(range(2), repeat=3):
            off = np.array(child) * 0.5 - 0.25
            T = np.zeros((self.P, self.P))
            for ia, a in enumerate(self.a):
                for ib, b in enumerate(self.a):
                    if all((y <= x for x, y in zip(a, b))):
                        T[ia, ib] = choose(a, b) * np.prod(off ** (np.array(a) - b)) * 0.5 ** sum(b)
            self.up.append(cp.asarray(T))
            self.down.append(cp.asarray(T.T))
        for level in range(1, self.depth + 1):
            m = 2 ** level
            par = np.indices((m // 2,) * 3).reshape(3, -1).T
            self.children[level] = [cp.asarray(((2 * par[:, 0] + a) * m + 2 * par[:, 1] + b) * m + 2 * par[:, 2] + c, dtype='i4') for a, b, c in itertools.product(range(2), repeat=3)]
        self.offsets = [d for d in itertools.product(range(-3, 4), repeat=3) if max((abs(t) for t in d)) > 1]
        self.translation = {}
        for d in self.offsets:
            coeff, ind = kernel_series(d, 2 * p)
            T = np.empty((self.P, self.P))
            for ib, b in enumerate(self.a):
                for ia, a in enumerate(self.a):
                    ab = tuple((x + y for x, y in zip(a, b)))
                    T[ib, ia] = (-1) ** sum(a) * choose(ab, a) * coeff[ind[ab]] / (-4 * np.pi)
            self.translation[d] = cp.asarray(T)
        self.pairs = {}
        for level in range(2, self.depth + 1):
            m = 2 ** level
            target = np.indices((m,) * 3).reshape(3, -1).T
            groups = []
            for d in self.offsets:
                source = target - np.array(d)
                ok = np.all((source >= 0) & (source < m), 1) & np.all(np.abs(target // 2 - source // 2) <= 1, 1)
                it = np.flatnonzero(ok)
                s = source[ok]
                iss = (s[:, 0] * m + s[:, 1]) * m + s[:, 2]
                if len(it):
                    groups.append((d, cp.asarray(it, dtype='i4'), cp.asarray(iss, dtype='i4')))
            self.pairs[level] = groups
        m = 2 ** self.depth
        box = np.indices((m,) * 3).reshape(3, -1).T
        voxel = np.array(list(itertools.product(range(4), repeat=3)))
        grid = 4 * box[:, None, :] + voxel[None, :, :]
        self.voxel_ids = cp.asarray((grid[:, :, 0] * n + grid[:, :, 1]) * n + grid[:, :, 2], dtype='i4')
        z = (voxel + 0.5) / 4 - 0.5
        half = 0.125
        H = length / m
        self.moment = cp.asarray(np.array([[np.prod(((zz + half) ** (np.array(a) + 1) - (zz - half) ** (np.array(a) + 1)) / (np.array(a) + 1)) * H ** 3 for zz in z] for a in self.a]))
        self.near_offsets = cp.asarray(np.array(list(itertools.product(range(-4, 8), repeat=3))), dtype='i4')

    def centers(self):
        ids = cp.indices((self.n,) * 3).reshape(3, -1).T
        return self.lo + self.length * (ids + 0.5) / self.n

    def forward(self, f):
        f = cp.asarray(f, dtype='f8')
        k = f.shape[1]
        M = {}
        L = {}
        M[self.depth] = cp.einsum('av,bvk->bak', self.moment, f[self.voxel_ids])
        for level in range(self.depth, 0, -1):
            M[level - 1] = cp.zeros((8 ** (level - 1), self.P, k))
            for child, ids in enumerate(self.children[level]):
                M[level - 1] += cp.einsum('ab,nbk->nak', self.up[child], M[level][ids])
        for level in range(self.depth + 1):
            L[level] = cp.zeros_like(M[level])
        for level in range(1, self.depth + 1):
            for child, ids in enumerate(self.children[level]):
                L[level][ids] += cp.einsum('ab,nbk->nak', self.down[child], L[level - 1])
            H = self.length / 2 ** level
            for d, it, iss in self.pairs.get(level, []):
                L[level][it] += cp.einsum('ab,nbk->nak', self.translation[d] / H, M[level][iss])
        return (f, L[self.depth])

    def query(self, x):
        m = 2 ** self.depth
        H = self.length / m
        xyz = cp.clip(cp.floor((x - self.lo) / H).astype('i4'), 0, m - 1)
        leaf = (xyz[:, 0] * m + xyz[:, 1]) * m + xyz[:, 2]
        z = (x - (self.lo + (xyz + 0.5) * H)) / H
        vv = []
        gg = [[], [], []]
        ll = []
        for a in self.a:
            aa = cp.asarray(a)
            vv.append(cp.prod(z ** aa, 1))
            lap = cp.zeros(len(x))
            for j in range(3):
                b = list(a)
                b[j] -= 1
                gg[j].append(a[j] * cp.prod(z ** cp.asarray(b), 1) / H if a[j] else cp.zeros(len(x)))
                if a[j] > 1:
                    b[j] -= 1
                    lap += a[j] * (a[j] - 1) * cp.prod(z ** cp.asarray(b), 1) / H ** 2
            ll.append(lap)
        E = cp.stack([cp.stack(vv, 1)] + [cp.stack(g, 1) for g in gg], axis=1)
        return (xyz, leaf, E, cp.stack(ll, 1))

    def near(self, x, xyz):
        coords = 4 * xyz[:, None, :] + self.near_offsets[None, :, :]
        valid = cp.all((coords >= 0) & (coords < self.n), 2)
        ids = (coords[:, :, 0] * self.n + coords[:, :, 1]) * self.n + coords[:, :, 2]
        ids = cp.where(valid, ids, -1).astype('i4')
        return (ids, cube_matrix(x, ids, self.n, self.lo, self.length))

    def evaluate(self, state, x):
        f, L = state
        out = []
        laps = []
        for off in range(0, len(x), 128):
            xx = x[off:off + 128]
            xyz, leaf, E, D = self.query(xx)
            ids, B = self.near(xx, xyz)
            out.append(cp.einsum('qap,qpk->qak', E, L[leaf]) + cp.einsum('qas,qsk->qak', B, f[cp.maximum(ids, 0)]))
            cell = cp.floor((xx - self.lo) * self.n / self.length).astype('i4')
            inside = cp.all((cell >= 0) & (cell < self.n), 1)
            cell = cp.clip(cell, 0, self.n - 1)
            flat = (cell[:, 0] * self.n + cell[:, 1]) * self.n + cell[:, 2]
            laps.append(f[flat] * inside[:, None] + cp.einsum('qp,qpk->qk', D, L[leaf]))
        out = cp.concatenate(out)
        return dict(value=out[:, 0], gradient=out[:, 1:], lap=cp.concatenate(laps))

    def transpose(self, queries):
        k = queries[0][1].shape[-1]
        fbar = cp.zeros((self.n ** 3, k))
        Lbar = {j: cp.zeros((8 ** j, self.P, k)) for j in range(self.depth + 1)}
        for x, g in queries:
            for off in range(0, len(x), 128):
                xx = x[off:off + 128]
                gg = g[off:off + 128]
                xyz, leaf, E, D = self.query(xx)
                ids, B = self.near(xx, xyz)
                cp.add.at(Lbar[self.depth], leaf, cp.einsum('qap,qak->qpk', E, gg))
                cot = cp.einsum('qas,qak->qsk', B, gg)
                valid = ids >= 0
                cp.add.at(fbar, ids[valid], cot[valid])
        Mbar = {j: cp.zeros_like(Lbar[j]) for j in Lbar}
        for level in range(self.depth, 0, -1):
            H = self.length / 2 ** level
            for d, it, iss in self.pairs.get(level, []):
                Mbar[level][iss] += cp.einsum('ba,nbk->nak', self.translation[d] / H, Lbar[level][it])
            for child, ids in enumerate(self.children[level]):
                Lbar[level - 1] += cp.einsum('ba,nbk->nak', self.down[child], Lbar[level][ids])
        for level in range(1, self.depth + 1):
            for child, ids in enumerate(self.children[level]):
                Mbar[level][ids] += cp.einsum('ba,nbk->nak', self.up[child], Mbar[level - 1])
        fbar[self.voxel_ids.ravel()] += cp.einsum('av,bak->bvk', self.moment, Mbar[self.depth]).reshape(-1, k)
        return fbar
def qualification():
    """Independent cuboid quadrature, harmonicity, FMM/direct and transpose."""
    from common import rel
    n = 16
    lo = -0.02
    length = 1.04
    rng = cp.random.RandomState(72)
    x = cp.asarray([[2.0, 1.7, 2.5], [-1.0, 2.0, -2.0]], dtype='f8')
    ids = cp.zeros((2, 1), dtype='i4')
    B = cube_matrix(x, ids, n, lo, length)[:, :, 0]
    g, w = np.polynomial.legendre.leggauss(12)
    qq = np.array(list(itertools.product(range(12), repeat=3)))
    yy = lo + (g[qq] + 1) * length / n / 2
    ww = np.prod(w[qq], 1) * (length / n / 2) ** 3
    d = x[:, None, :] - cp.asarray(yy)[None, :, :]
    r = cp.linalg.norm(d, axis=2)
    expected = cp.concatenate(((-cp.sum(cp.asarray(ww) / r, 1) / (4 * np.pi))[:, None], cp.sum(cp.asarray(ww)[None, :, None] * d / r[:, :, None] ** 3, 1) / (4 * np.pi)), 1)
    cube_error = rel(B, expected)
    assert cube_error < 1e-06, cube_error
    x = rng.uniform(0.01, 0.99, (48, 3))
    f = rng.normal(size=(n ** 3, 2))
    result = {'cube_gauss_relative': cube_error}
    direct = []
    for off in range(0, len(x), 8):
        xx = x[off:off + 8]
        B = cube_matrix(xx, cp.broadcast_to(cp.arange(n ** 3, dtype='i4'), (len(xx), n ** 3)), n, lo, length)
        direct.append(cp.einsum('qas,sk->qak', B, f))
    direct = cp.concatenate(direct)
    for p in (4, 6):
        op = VolumeFMM(n, p)
        st = op.forward(f)
        ev = op.evaluate(st, x)
        y = cp.concatenate((ev['value'][:, None], ev['gradient']), axis=1)
        cot = rng.normal(size=y.shape)
        fb = op.transpose([(x, cot)])
        adj = abs(float(cp.sum(y * cot) - cp.sum(f * fb))) / max(abs(float(cp.sum(y * cot))), 1e-30)
        local_lap = cp.einsum('qp,qpk->qk', op.query(x)[3], st[1][op.query(x)[1]])
        result[f'p{p}'] = dict(value_relative=rel(y[:, 0], direct[:, 0]), gradient_relative=rel(y[:, 1:], direct[:, 1:]), adjoint_relative=adj, far_laplacian_max=float(cp.abs(local_lap).max()))
        assert adj < 1e-09 and float(cp.abs(local_lap).max()) < 1e-08, result
    assert result['p6']['gradient_relative'] < result['p4']['gradient_relative'], result
    return result
