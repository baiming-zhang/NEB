





import numpy as np


def storage_estimate(k):
    pairs = k * (k + 1) // 2
    return dict(K=k, pairs=pairs, dense_fp64_bytes=8*k**3,
                packed_fp64_bytes=8*k*pairs, matrix_fp64_bytes=8*k*k)


class CoefficientROM:
    def __init__(self, mass, linear, quadratic, xp=np, *, block_bytes=64*2**20):
        self.xp = xp
        self.mass = xp.asarray(mass, dtype='f8')
        self.linear = xp.asarray(linear, dtype='f8')
        self.K = len(self.mass)
        self.j, self.k = xp.triu_indices(self.K)
        self.pairs = len(self.j)
        if self.mass.shape != (self.K, self.K) or self.linear.shape != self.mass.shape:
            raise ValueError('Mass and linear operators must be K by K')
        if quadratic.shape != (self.K, self.pairs):
            raise ValueError('Q columns must follow triu_indices(K), with summed off-diagonal terms')
        self.quadratic = quadratic
        self.rows = max(1, block_bytes // (8*self.pairs))
        self.cholesky = xp.linalg.cholesky((self.mass+self.mass.T)*.5)
        if hasattr(xp, 'cuda'):
            from cupyx.scipy.linalg import solve_triangular
        else:
            from scipy.linalg import solve_triangular
        self.triangular = solve_triangular

    def solve(self, load):
        y = self.triangular(self.cholesky, load, lower=True)
        return self.triangular(self.cholesky.T, y, lower=False)

    def rhs(self, c):

        products = c[self.j] * c[self.k]
        load = self.linear @ c
        for first in range(0, self.K, self.rows):
            q = self.quadratic[first:first+self.rows]
            if isinstance(q, np.ndarray) and hasattr(self.xp, 'cuda'):
                q = self.xp.asarray(q, dtype='f8', blocking=True)
            else:
                q = self.xp.asarray(q, dtype='f8')
            load[first:first+len(q)] += q @ products
        return self.solve(load)

    def step(self, c, dt):
        k1 = self.rhs(c)
        k2 = self.rhs(c+.5*dt*k1)
        k3 = self.rhs(c+.5*dt*k2)
        k4 = self.rhs(c+dt*k3)
        return c+dt/6*(k1+2*k2+2*k3+k4)

    def relative_error(self, c, reference_load, reference_norm_squared):

        energy = float(c @ self.mass @ c)
        cross = float(c @ reference_load)
        ref = float(reference_norm_squared)
        if ref <= 0:
            raise ValueError('Positive reference norm required')
        error_squared = energy-2*cross+ref
        scale = abs(energy)+2*abs(cross)+ref
        if error_squared < -64*np.finfo('f8').eps*scale:
            raise ValueError('Inconsistent full-mass/reference inner products')
        return float(np.sqrt(max(0, error_squared)/ref))


def assemble_small_reference(k, linear_load, quadratic_load):

    if not 1 <= k <= 16:
        raise ValueError('Pairwise polarization is restricted to a small verification')
    eye = np.eye(k)
    a = np.column_stack([linear_load(e) for e in eye])
    diagonal = [quadratic_load(e) for e in eye]
    q = []
    for j, m in zip(*np.triu_indices(k)):
        q.append(diagonal[j] if j == m else
                 quadratic_load(eye[j]+eye[m])-diagonal[j]-diagonal[m])
    return a, np.column_stack(q)
