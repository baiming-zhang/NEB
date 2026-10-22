from pathlib import Path
from types import SimpleNamespace
import ast,json,numpy as np
from train import V
LEGACY=V/'3d_metaball_reproduction_latest/cases/3D_Metaball'
GENERATOR=LEGACY/'physical_dns/scripts/generate_independent_dns_gt_128.py'
CONFIG=Path(__file__).resolve().parents[2]/'configs/vrq_capacity_k256.json'
def _original_functions(path, names, namespace):
    """Compile selected pure functions, never execute the legacy module imports."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Original source API changed')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace
def geometry_grid(geometry):
    """Exact original NumPy geometry function, 128 cubed and float32."""
    cfg = SimpleNamespace(**json.loads(CONFIG.read_text()))
    training = _original_functions(LEGACY / 'scripts/train_validate.py', ['outer_level_numpy', 'obstacle_level_numpy'], {'np': np})
    ns = _original_functions(GENERATOR, ['cell_center_geometry'], {'np': np, 'training': SimpleNamespace(**training)})
    grid = ns['cell_center_geometry'](cfg, 128, np.asarray(geometry, dtype=np.float32))
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(grid['fluid_mask'].astype(np.float32), sigma=0.75, mode='wrap')
    grid['solid_fraction'] = np.clip(1.0 - smooth, 0.0, 1.0).astype(np.float32)
    return grid
class Spectral:
    """Native CuPy translation of original SpectralOperators/PseudoSpectralDNS."""

    def __init__(self, grid):
        import cupy as cp
        self.cp = cp
        k = cp.float32(2 * np.pi) * cp.fft.fftfreq(128, d=1 / 128).astype('f4')
        self.kz, self.ky, self.kx = (k[:, None, None], k[None, :, None], k[None, None, :])
        self.k2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        self.safe = self.k2.copy()
        self.safe[0, 0, 0] = 1
        valid = cp.abs(cp.fft.fftfreq(128, d=1 / 128)) <= 42
        self.dealias = valid[:, None, None] & valid[None, :, None] & valid[None, None, :]
        self.normal = cp.asarray(grid['normal_xyz'].transpose(3, 0, 1, 2))
        self.mask = cp.asarray(grid['fluid_mask'])
        self.band = cp.asarray(grid['boundary_band'])
        self.half_penalty = cp.exp(-0.5 * 0.005 * cp.asarray(grid['solid_fraction']) / 0.001)
        self.visc = cp.exp(-0.5 * 0.00025 * self.k2 * 0.005)

    def fft(self, u):
        return self.cp.fft.fftn(u, axes=(1, 2, 3))

    def real(self, v):
        return self.cp.fft.ifftn(v, axes=(1, 2, 3)).real

    def project(self, v):
        cp = self.cp
        dot = self.kx * v[0] + self.ky * v[1] + self.kz * v[2]
        r = cp.stack((v[0] - self.kx * dot / self.safe, v[1] - self.ky * dot / self.safe, v[2] - self.kz * dot / self.safe))
        r *= self.dealias
        r[:, 0, 0, 0] = 0
        return r

    def curl(self, v):
        return self.cp.stack((1j * (self.ky * v[2] - self.kz * v[1]), 1j * (self.kz * v[0] - self.kx * v[2]), 1j * (self.kx * v[1] - self.ky * v[0])))

    def nonlinear(self, v):
        cp = self.cp
        return self.project(self.fft(cp.cross(self.real(v), self.real(self.curl(v)), axisa=0, axisb=0, axisc=0)))

    def penalty(self, v):
        cp = self.cp
        u = self.real(v)
        return self.project(self.fft(u + (self.half_penalty[None] - 1) * cp.sum(u * self.normal, axis=0, keepdims=True) * self.normal))

    def step(self, v):
        v = self.penalty(v) * self.visc
        k1 = self.nonlinear(v)
        k2 = self.nonlinear(v + 0.5 * 0.005 * k1)
        k3 = self.nonlinear(v + 0.5 * 0.005 * k2)
        k4 = self.nonlinear(v + 0.005 * k3)
        v = self.project(v + 0.005 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)) * self.visc
        return self.penalty(v)

    def initialize(self, phases):
        cp = self.cp
        a = (cp.arange(128, dtype='f4') + 0.5) / 128
        z, y, x = cp.meshgrid(a, a, a, indexing='ij')
        tg = cp.stack((cp.sin(2 * np.pi * x) * cp.cos(2 * np.pi * y) * cp.cos(2 * np.pi * z), -cp.cos(2 * np.pi * x) * cp.sin(2 * np.pi * y) * cp.cos(2 * np.pi * z), cp.zeros_like(x)))
        abc = cp.stack((cp.sin(4 * np.pi * z) + 0.8 * cp.cos(4 * np.pi * y), 0.9 * cp.sin(4 * np.pi * x) + cp.cos(4 * np.pi * z), 0.8 * cp.sin(4 * np.pi * y) + 0.9 * cp.cos(4 * np.pi * x)))
        p = cp.asarray(phases, dtype='f4')
        perturb = cp.stack((cp.sin(6 * np.pi * y + p[0]) * cp.cos(4 * np.pi * z), cp.sin(6 * np.pi * z + p[1]) * cp.cos(4 * np.pi * x), cp.sin(6 * np.pi * x + p[2]) * cp.cos(4 * np.pi * y)))
        v = self.project(self.fft(0.68 * tg + 0.26 * abc + 0.06 * perturb))
        for _ in range(8):
            v = self.penalty(v)
        physical = self.real(v)
        rms = cp.sqrt(cp.mean(cp.sum(physical[:, self.mask] ** 2, axis=0)))
        v = v * (0.32 / rms)
        for _ in range(4):
            v = self.step(v)
        return v