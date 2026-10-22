




from _neb_paths import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LEGACY = ROOT / '3d_metaball_reproduction_latest/cases/3D_Metaball'
GENERATOR = LEGACY / 'physical_dns/scripts/generate_independent_dns_gt_128.py'
CONFIG = LEGACY / 'configs/metaball_geometry.json'
DNS = LEGACY / 'results/independent_freeslip_dns_gt_128'
PROTOCOL = ROOT / 'ICLR2027_campaign/cases/metaball/results/velocity_k1024_full2000_v1/checkpoint_comparison20/protocol.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _original_functions(path, names, namespace):

    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError('Original source API changed')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace


def geometry_grid(geometry):

    cfg = SimpleNamespace(**json.loads(CONFIG.read_text()))
    training = _original_functions(LEGACY / 'scripts/train_validate.py',
        ['outer_level_numpy', 'obstacle_level_numpy'], {'np': np})
    ns = _original_functions(GENERATOR, ['cell_center_geometry'],
        {'np': np, 'training': SimpleNamespace(**training)})
    grid = ns['cell_center_geometry'](cfg, 128, np.asarray(geometry, dtype=np.float32))
    from scipy.ndimage import gaussian_filter
    smooth = gaussian_filter(grid['fluid_mask'].astype(np.float32), sigma=.75, mode='wrap')
    grid['solid_fraction'] = np.clip(1.0 - smooth, 0.0, 1.0).astype(np.float32)
    return grid


def capture_original_phases():






    import torch
    generator = torch.Generator(device='cuda').manual_seed(20260826)
    phases = (2.0 * np.pi * torch.rand(3, generator=generator, device='cuda')).cpu().numpy()
    return phases.tolist()


class Spectral:

    def __init__(self, grid):
        import cupy as cp
        self.cp = cp

        k = cp.float32(2 * np.pi) * cp.fft.fftfreq(128, d=1 / 128).astype('f4')
        self.kz, self.ky, self.kx = k[:, None, None], k[None, :, None], k[None, None, :]
        self.k2 = self.kx ** 2 + self.ky ** 2 + self.kz ** 2
        self.safe = self.k2.copy(); self.safe[0, 0, 0] = 1
        valid = cp.abs(cp.fft.fftfreq(128, d=1 / 128)) <= 42
        self.dealias = valid[:, None, None] & valid[None, :, None] & valid[None, None, :]
        self.normal = cp.asarray(grid['normal_xyz'].transpose(3, 0, 1, 2))
        self.mask = cp.asarray(grid['fluid_mask'])
        self.band = cp.asarray(grid['boundary_band'])
        self.half_penalty = cp.exp(-.5 * .005 * cp.asarray(grid['solid_fraction']) / .001)
        self.visc = cp.exp(-.5 * .00025 * self.k2 * .005)

    def fft(self, u): return self.cp.fft.fftn(u, axes=(1, 2, 3))
    def real(self, v): return self.cp.fft.ifftn(v, axes=(1, 2, 3)).real

    def project(self, v):
        cp = self.cp
        dot = self.kx * v[0] + self.ky * v[1] + self.kz * v[2]
        r = cp.stack((v[0] - self.kx * dot / self.safe,
                      v[1] - self.ky * dot / self.safe,
                      v[2] - self.kz * dot / self.safe))
        r *= self.dealias; r[:, 0, 0, 0] = 0
        return r

    def curl(self, v):
        return self.cp.stack((1j * (self.ky * v[2] - self.kz * v[1]),
            1j * (self.kz * v[0] - self.kx * v[2]), 1j * (self.kx * v[1] - self.ky * v[0])))

    def nonlinear(self, v):
        cp = self.cp
        return self.project(self.fft(cp.cross(self.real(v), self.real(self.curl(v)), axisa=0, axisb=0, axisc=0)))

    def penalty(self, v):
        cp = self.cp
        u = self.real(v)
        return self.project(self.fft(u + (self.half_penalty[None] - 1) * cp.sum(u * self.normal, axis=0, keepdims=True) * self.normal))

    def step(self, v):
        v = self.penalty(v) * self.visc
        k1 = self.nonlinear(v); k2 = self.nonlinear(v + .5 * .005 * k1)
        k3 = self.nonlinear(v + .5 * .005 * k2); k4 = self.nonlinear(v + .005 * k3)
        v = self.project(v + (.005 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)) * self.visc
        return self.penalty(v)

    def initialize(self, phases):
        cp = self.cp
        a = (cp.arange(128, dtype='f4') + .5) / 128
        z, y, x = cp.meshgrid(a, a, a, indexing='ij')
        tg = cp.stack((cp.sin(2*np.pi*x)*cp.cos(2*np.pi*y)*cp.cos(2*np.pi*z),
            -cp.cos(2*np.pi*x)*cp.sin(2*np.pi*y)*cp.cos(2*np.pi*z), cp.zeros_like(x)))
        abc = cp.stack((cp.sin(4*np.pi*z)+.8*cp.cos(4*np.pi*y),
            .9*cp.sin(4*np.pi*x)+cp.cos(4*np.pi*z), .8*cp.sin(4*np.pi*y)+.9*cp.cos(4*np.pi*x)))
        p = cp.asarray(phases, dtype='f4')
        perturb = cp.stack((cp.sin(6*np.pi*y+p[0])*cp.cos(4*np.pi*z),
            cp.sin(6*np.pi*z+p[1])*cp.cos(4*np.pi*x), cp.sin(6*np.pi*x+p[2])*cp.cos(4*np.pi*y)))
        v = self.project(self.fft(.68*tg+.26*abc+.06*perturb))
        for _ in range(8): v = self.penalty(v)
        physical = self.real(v)
        rms = cp.sqrt(cp.mean(cp.sum(physical[:, self.mask] ** 2, axis=0)))
        v = v * (.32 / rms)
        for _ in range(4): v = self.step(v)
        return v


def build_reference(case_record, output_dir):






    frozen = json.loads(PROTOCOL.read_text())
    record = next((r for r in frozen['cases'] if r['case'] == case_record['case']), None)
    if record is None or record['geometry_token'] != case_record['geometry_token'] or not np.array_equal(record['geometry'], case_record['geometry']):
        raise ValueError('case_record differs from frozen protocol')
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    phases = case_record.get('original_phases')
    if phases is None: phases = capture_original_phases()
    phases = np.asarray(phases, dtype='f4')
    if phases.shape != (3,) or not np.isfinite(phases).all(): raise ValueError('three finite original phases required')
    grid = geometry_grid(record['geometry'])
    solver = Spectral(grid); cp = solver.cp
    v = solver.initialize(phases)
    u = solver.real(v)
    div = 1j * (solver.kx*v[0]+solver.ky*v[1]+solver.kz*v[2])
    ub = u[:, solver.band]; nb = solver.normal[:, solver.band]
    diagnostics = dict(divergence_relative_L2=float(cp.linalg.norm(div)/cp.linalg.norm(cp.sqrt(solver.k2)[None]*v)),
        boundary_normal_relative_rms=float(cp.sqrt(cp.mean(cp.sum(ub*nb,axis=0)**2)/cp.mean(cp.sum(ub**2,axis=0)))),
        fluid_rms_speed=float(cp.sqrt(cp.mean(cp.sum(u[:,solver.mask]**2,axis=0)))))
    velocity = cp.asnumpy(cp.where(solver.mask[None], u, cp.float32(0)))
    if velocity.shape != (3,128,128,128) or not np.isfinite(velocity).all(): raise ValueError('invalid generated velocity')
    arrays = dict(velocity=velocity, fluid_mask=grid['fluid_mask'], coordinates=grid['coordinates'],
        boundary_normal_xyz=grid['normal_xyz'], boundary_band=grid['boundary_band'],
        solid_fraction=grid['solid_fraction'], geometry_xyzr=np.asarray(record['geometry'],dtype='f4'))
    paths = {}
    for name, value in arrays.items():
        path = out / (name + '.npy'); np.save(path, value); paths[name] = str(path.resolve())
    cp.cuda.Stream.null.synchronize()
    meta = dict(case=record, paths=paths, layout='velocity: component_xyz,z,y,x; mask: z,y,x',
        dtype=str(velocity.dtype), original_phases=phases.tolist(), seed=20260826,
        resolution=128, dt=.005, initial_penalty_half_steps=8, warmup_substeps=4,
        viscosity=.00025, penalty_time_scale=.001, penalty_smoothing_cells=.75,
        time_label=0.0, physical_warmup_duration=.02, initial_rms_before_warmup=.32,
        diagnostics=diagnostics, seconds=time.perf_counter()-started,
        source_hashes={str(p):sha(p) for p in [GENERATOR,CONFIG,LEGACY/'scripts/train_validate.py',Path(__file__),PROTOCOL]},
        output_hashes={name:sha(path) for name,path in paths.items()},
        qualification='Independent per-geometry normal-penalty FFT NS numerical snapshot; not certified continuous Hodge truth; CuPy port reproduction validation pending')
    temporary=out/'metadata.tmp.json'
    temporary.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    temporary.replace(out/'metadata.json')
    return meta


def validate_case1_reproduction(output_dir, tolerance=2e-5):

    out = Path(output_dir); meta = json.loads((out/'metadata.json').read_text())
    if meta['case']['geometry_token'] != 12000001: raise ValueError('original case1 required')
    original = np.load(DNS/'velocity/frame_000.npy').transpose(3,0,1,2).astype('f8')
    actual = np.load(out/'velocity.npy').astype('f8')
    exact = {name:bool(np.array_equal(np.load(out/(name+'.npy')),np.load(DNS/(name+'.npy'))))
        for name in ['fluid_mask','coordinates','boundary_normal_xyz','boundary_band','solid_fraction','geometry_xyzr']}
    relative = float(np.linalg.norm(actual-original)/np.linalg.norm(original))
    return dict(passed=all(exact.values()) and relative<=tolerance, exact_geometry_arrays=exact,
        velocity_relative_L2=relative, velocity_max_abs=float(np.max(np.abs(actual-original))),
        tolerance=tolerance, meaning='cross-library FP32 port reproduction, not bitwise identity or continuum convergence')
