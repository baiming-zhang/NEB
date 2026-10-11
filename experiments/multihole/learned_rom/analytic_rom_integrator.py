"""Full-mass native-CuPy IMEX midpoint for analytic weak Hodge operators."""
import cupy as cp
from .rom_fem import cholsolve


class AnalyticROM:
    def __init__(self, G, D, Q, nu=1/750):
        self.G = cp.asarray(G, dtype=cp.float64)
        self.D = cp.asarray(D, dtype=cp.float64)
        self.Q = cp.asarray(Q, dtype=cp.float64)
        self.K = self.G.shape[0]
        assert self.G.shape == self.D.shape == (self.K, self.K)
        assert self.Q.shape == (self.K, self.K, self.K)
        assert nu >= 0
        self.nu = nu
        for value in (self.G, self.D, self.Q):
            assert bool(cp.isfinite(value).all().item())
        for value in (self.G, self.D):
            assert float(cp.linalg.norm(value-value.T).item()) < 1e-10*max(float(cp.linalg.norm(value).item()), 1e-30)
        cp.linalg.cholesky(self.G)
        assert float(cp.linalg.eigvalsh(self.D).min().item()) >= -1e-10*max(float(cp.linalg.norm(self.D).item()), 1e-30)
        assert float(cp.linalg.norm(self.Q+self.Q.swapaxes(0, 1)).item()) < 1e-10*max(float(cp.linalg.norm(self.Q).item()), 1e-30)
        self.factors = {}

    def nonlinear(self, c):
        """Columns are independent initial states, never coupled to each other."""
        return cp.einsum('ijk,js,ks->is', self.Q, c, c, optimize=True)

    def step(self, c, dt):
        assert dt > 0 and c.ndim == 2 and c.shape[0] == self.K
        if dt not in self.factors:
            self.factors[dt] = cp.linalg.cholesky(self.G+.5*dt*self.nu*self.D)
        L = self.factors[dt]
        half = cholsolve(L, self.G@c+.5*dt*self.nonlinear(c))
        return cholsolve(L, (self.G-.5*dt*self.nu*self.D)@c+dt*self.nonlinear(half))

    def trajectory(self, initial, dt, final_time=1., frame_interval=.02):
        steps = round(final_time/dt)
        stride = round(frame_interval/dt)
        assert steps > 0 and stride > 0 and steps % stride == 0
        assert abs(steps*dt-final_time) < 1e-12 and abs(stride*dt-frame_interval) < 1e-12
        c = cp.asarray(initial, dtype=cp.float64).copy()
        frames = [c.copy()]
        for step in range(1, steps+1):
            c = self.step(c, dt)
            if step % stride == 0:
                assert bool(cp.isfinite(c).all().item())
                frames.append(c.copy())
        return cp.stack(frames)
