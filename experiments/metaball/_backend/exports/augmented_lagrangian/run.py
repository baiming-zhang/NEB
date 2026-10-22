import os, sys, time, json, threading, traceback, gc
from _neb_paths import Path
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CASE = ROOT / 'flow/cases/metaball'
DATA = Path('_data/metaball/augmented_lagrangian')
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'exports/potential_rayleigh'))
import train_potential as train
cp, np, rt = (train.cp, train.np, train.rt)
from common import guard, read, write
from equivalent_harmonic_lift import lift_fields
from canonical_poisson_native import geometry2
from threadpoolctl import threadpool_limits
import psutil
STAGE = sys.argv[1]
DEST = HERE / STAGE
DEST.mkdir(exist_ok=True)
assert STAGE in ('qualification', 'unconstrained', 'penalty', 'augmented_lagrangian')

def clear():
    cp.cuda.Stream.null.synchronize()
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

def values(x):
    return cp.asnumpy(x).tolist()

def distribution(x):
    a = cp.asnumpy(x)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))

def inputs():
    cfg = read(Path('_data/metaball/potential_capacity_20261005/potential_k128/config.json'))
    c = rt.config_namespace()
    geo = cp.asarray(cfg['geometry'], 'f8')
    path = CASE / 'results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz'
    with np.load(path) as z:
        x = z['fit'].copy()
    q = rt.fluid_points(c, geo, 16384, 919000)
    assert len(x) == 103244 and len(q) == 16384
    return (c, geo, x, q)

def manufactured(points, geo, c):
    x = cp.asarray(points, 'f8')
    e, de, le, *_ = geometry2(x, geo, c)
    A = []
    D = []
    for f in (1, 4, 12, 32):
        k = 2 * np.pi * f * cp.asarray([1.0, 2.0, 3.0])
        s = cp.sin(x @ k)
        cs = cp.cos(x @ k)
        A.append((2 * e[:, None] * de * s[:, None] + e[:, None] ** 2 * cs[:, None] * k).reshape(-1))
        D.append(2 * (cp.sum(de * de, 1) + e * le) * s + 4 * e * (de @ k) * cs - e * e * cp.sum(k * k) * s)
    return (cp.stack(A, 1), cp.stack(D, 1))

def wall_test_scalar(points, geo, c):
    return lift_fields(cp.asarray(points, 'f8'), geo, c)[1].reshape(-1, 1)

def augmented(a, d, h, lam, eta, rho, scale):
    n = d.shape[0]
    mass = cp.sum(a * a, axis=0) / n
    s = cp.sqrt(mass)
    assert bool(cp.all(cp.isfinite(mass) & (mass > 0)))
    r = d / s[None, :]
    sh = cp.sqrt(cp.sum(h * h) / n)
    moment = (h.T @ a).reshape(-1) / (n * s * sh)
    t = lam + rho * r
    th = eta + rho * moment
    cost = cp.mean(lam * r + 0.5 * rho * r * r, axis=0) + scale * (eta * moment + 0.5 * rho * moment * moment)
    gd = t / (n * s[None, :])
    ga = -a * (cp.mean(t * r, axis=0) / (n * mass))[None, :]
    ga += scale * th[None, :] * (h / (n * s[None, :] * sh) - a * (moment / (n * mass))[None, :])
    return (cost, ga, gd, r, moment)

def metrics(model, data, h):
    f = rt.grouped8.network_fields(model, data, 'aud')
    mass = cp.sum(f['a'] * f['a'], axis=0) / data.n
    curlmass = cp.sum(f['u'] * f['u'], axis=0) / data.n
    divmass = cp.sum(f['d'] * f['d'], axis=0) / data.n
    moment = (h.T @ f['a']).reshape(-1) / (data.n * cp.sqrt(mass) * cp.sqrt(cp.sum(h * h) / data.n))
    obj = train.bounded_potential_objective(f['a'], f['u'], data.n)
    row = dict(points=data.n, all_modes=model.K, ANN_precision=str(data.x.dtype), mean_potential_Rayleigh=float(cp.mean(obj['losses'])), gram_max=float(obj['gram_max']), divergence_over_potential_mass=values(divmass / mass), divergence_over_curl_mass=values(divmass / cp.maximum(curlmass, 1e-300)), divergence_over_curl_summary=distribution(divmass / cp.maximum(curlmass, 1e-300)), weak_wall_scalar_moment=values(moment), weak_wall_scalar_moment_max=float(cp.max(cp.abs(moment))), potential_mass=values(mass), curl_mass=values(curlmass))
    del f, obj
    clear()
    return row
