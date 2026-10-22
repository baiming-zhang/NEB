import os, sys, time, gc
from _neb_paths import Path
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'flow/nkb'))
from common import guard, read, write
import train_potential as train
rt, cp, np = (train.rt, train.cp, train.np)
DEST = HERE / 'potential256_diagnosis'
RUN = Path('_data/metaball/potential_capacity_20261005/potential_k256')

def model_from(path, c, dtype):
    m = rt.Model.__new__(rt.Model)
    m.c = c
    m.K = 256
    m.epoch = 2000
    with np.load(path) as z:
        m.W = [cp.asarray(z[f'W_{i}'], dtype=dtype) for i in range(6)]
        m.b = [cp.asarray(z[f'b_{i}'], dtype=dtype) for i in range(6)]
    return m

def summary(a):
    a = cp.asnumpy(a)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()))

def canonical_masses(model, points, geo, c, resolution, deadline):
    import small
    factory = small.ScalarPoisson
    if resolution != 8:
        small.ScalarPoisson = lambda config, g, n, **kw: factory(config, g, resolution, **kw)
    try:
        pipe = train.Pipeline(c, geo)
    finally:
        small.ScalarPoisson = factory
    pipe.fit = rt.NeuralPoints(cp.asnumpy(pipe.x), geo, c, 'f8')
    data, G, h = pipe.batch(points, 'f8')
    raw = cp.empty((3 * len(points), 256), dtype='f8')
    velocity = cp.empty_like(raw)
    canonical = cp.empty_like(raw)
    for first in range(0, 256, 8):
        if time.perf_counter() > deadline:
            raise RuntimeError('Finite diagnosis7200s deadline')
        last = first + 8
        m = train.block_model(model, first, last)
        af = train.source_values(m, pipe.fit)
        co = pipe.solve(af, 'diagnostic_forward')
        pf = af - pipe.fit_field(co)
        amp = pipe.hf.T @ pf / pipe.hm
        f = rt.grouped8.network_fields(m, data, 'au')
        raw[:, first:last] = f['a']
        velocity[:, first:last] = f['u']
        canonical[:, first:last] = f['a'] - G @ co - h @ amp
        del af, co, pf, amp, f, m
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    denom = cp.sum(canonical * canonical, axis=0) / len(points)
    curlmass = cp.sum(velocity * velocity, axis=0) / len(points)
    rawmass = cp.sum(raw * raw, axis=0) / len(points)
    obj = train.bounded_potential_objective(canonical, velocity, len(points))
    result = dict(resolution=resolution, scalar_directions=pipe.scalar.n, source_points=pipe.fit.n, points=len(points), all_modes=256, raw_potential_mass=cp.asnumpy(rawmass).tolist(), canonical_potential_mass=cp.asnumpy(denom).tolist(), curl_mass=cp.asnumpy(curlmass).tolist(), raw_to_canonical_mass=summary(denom / cp.maximum(rawmass, 1e-300)), sorted_GS_mean_rayleigh=float(cp.mean(obj['losses'])), GS_gram_max=float(obj['gram_max']), solvers=list(pipe.scalar.records))
    del pipe, data, G, h, raw, velocity, canonical, obj
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result
