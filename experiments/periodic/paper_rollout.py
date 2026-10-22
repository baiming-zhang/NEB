"""Frozen-basis re-evaluation of the single manuscript vortex-merger case.

No training. Original native CuPy fields and curl-form ETDRK4, all K1024
coordinates, full mass, d=1.35/T18pi and the same 241 reference states.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import argparse, ast, gc, importlib.util, json, sys, time, threading
import numpy as np
import scipy.linalg as la
from scipy.signal import resample
import cupy as cp
WORK = Path(__file__).resolve().parents[1] / '2d/_backend'
_runtime_spec = importlib.util.spec_from_file_location('neb_periodic_runtime', Path(__file__).with_name('runtime.py'))
_runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(_runtime)
guard, read, write = (_runtime.guard, _runtime.read, _runtime.write)
SOURCE = WORK / 'exports/2d_vortex_animation_20260923/wider'
OUT = Path('outputs/periodic')
OLD = WORK / 'flow/cases/vortex_merger/results/neural_training/formal'
K = 1024

def stop_check(dest):
    for base in (OUT, dest):
        for name in ('user_requested_stop.json', 'user_budget_stop.json', 'scientific_failure.json', 'budget_stop.json'):
            if (base / name).exists():
                raise RuntimeError('Persistent stop requires review: ' + str(base / name))

def rollout(gu, gv, gw, vals, gt, dt, dest, tag):
    """Dynamic frame count only; equations match the inherited evaluator."""
    n = int(round(np.sqrt(gu.shape[0])))
    weight = (2 * np.pi / n) ** 2
    c = weight * (gu.T @ cp.asarray(gt['velocity_u'][0].ravel()) + gv.T @ cp.asarray(gt['velocity_v'][0].ravel()))

    def rhs(a):
        u = gu @ a
        v = gv @ a
        w = gw @ a
        return gu.T @ (weight * w * v) - gv.T @ (weight * w * u)
    rng = cp.asarray(np.random.default_rng(42).normal(size=len(c)))
    skew = float(cp.abs(rng @ rhs(rng)) / cp.linalg.norm(rng) ** 3)
    assert skew < 1e-09, skew
    states = []
    current = 0.0
    started = time.perf_counter()
    for index, target in enumerate(gt['time']):
        stop_check(dest)
        if time.perf_counter() - started > 7200:
            write(dest / 'budget_stop.json', dict(phase=tag, elapsed=time.perf_counter() - started, hard_limit_seconds=7200))
            raise TimeoutError('7200-second trajectory cap; do not automatically restart')
        steps = max(0, int(np.ceil((target - current) / dt - 1e-10)))
        if steps:
            h = (target - current) / steps
            linear = -gt['nu'] * vals
            e = cp.exp(h * linear)
            e2 = cp.exp(0.5 * h * linear)
            roots = cp.exp(1j * cp.pi * (cp.arange(1, 33) - 0.5) / 32)
            lr = h * linear[:, None] + roots
            q = h * cp.real(cp.mean((cp.exp(lr / 2) - 1) / lr, axis=-1))
            f1 = h * cp.real(cp.mean((-4 - lr + cp.exp(lr) * (4 - 3 * lr + lr * lr)) / lr ** 3, axis=-1))
            f2 = h * cp.real(cp.mean((2 + lr + cp.exp(lr) * (-2 + lr)) / lr ** 3, axis=-1))
            f3 = h * cp.real(cp.mean((-4 - 3 * lr - lr * lr + cp.exp(lr) * (4 - lr)) / lr ** 3, axis=-1))
            for _ in range(steps):
                nv = rhs(c)
                a = e2 * c + q * nv
                na = rhs(a)
                b = e2 * c + q * na
                nb = rhs(b)
                z = e2 * a + q * (2 * nb - nv)
                c = e * c + f1 * nv + 2 * f2 * (na + nb) + f3 * rhs(z)
        current = target
        states.append(cp.asnumpy(c))
        if index % 10 == 0 or index == len(gt['time']) - 1:
            row = dict(phase=tag, frame=index + 1, frames=len(gt['time']), dt=dt, elapsed=time.perf_counter() - started, pid=os.getpid(), unix=time.time())
            write(dest / 'progress.json', row)
            print(json.dumps(row), flush=True)
    cp.cuda.Stream.null.synchronize()
    online = time.perf_counter() - started
    tick = time.perf_counter()
    states = np.array(states)
    device = cp.asarray(states)
    stride = n // 128
    fields = [cp.asnumpy(device @ a.T).reshape(len(states), n, n)[:, ::stride, ::stride] for a in (gu, gv, gw)]
    assert all((np.isfinite(a).all() for a in fields))
    return (fields, states, skew, online, time.perf_counter() - tick)

def inherited_check(gu, gv, gw, vals, gt, dest, n):
    """Compare one short trajectory to the exact original evaluator function."""
    source = WORK / 'periodic_core/vortex_merger/canonical_gauge/scripts/evaluate_large_comparison.py'
    code = source.read_text()
    node = next((x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name == 'matrixfree_rollout'))
    body = ast.get_source_segment(code, node).replace('128', str(n))
    ns = dict(np=np, cp=cp)
    exec(compile(body, str(source) + '::matrixfree_rollout', 'exec'), ns)
    times = np.linspace(0, 0.04, 61)
    short = dict(gt, time=times)
    actual = rollout(gu, gv, gw, vals, short, 0.005, dest, 'original_equation_check')
    u, v, w, c, skew = ns['matrixfree_rollout'](gu, gv, gw, vals, short, 0.005)
    defect = float(np.linalg.norm(c - actual[1]) / max(np.linalg.norm(c), 1e-30))
    assert defect < 1e-10, defect
    return dict(coefficient_relative_difference=defect, all_modes=True, all_operator_points=True)

def build_fields(method, seed, n, dest):
    axis = np.arange(n) * 2 * np.pi / n
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    if method == 'NKB':
        nkb = Path(__file__).resolve().parents[1] / '2d/nkb'
        sys.path.insert(0, str(nkb))
        from core import Model
        from domains import Domain
        meta = read(explicit_checkpoint.with_suffix('.json'))
        cfg = meta['config']
        assert meta['epoch'] == 2000 and cfg['case'] == 'periodic' and (cfg['K'] == K)
        model = Model(cp, cfg['input_dim'], cfg['dim'], cfg['K'], cfg['seed'], cfg['width'], cfg['hidden_layers'], dtype='float64')
        model.load(explicit_checkpoint)
        domain = Domain('periodic', cp)
        arrays = [cp.empty((n * n, K), dtype='f8') for _ in range(4)]
        for first in range(0, n * n, 2048):
            p = cp.asarray(points[first:first + 2048])
            f, j = domain.encode(p)
            v, jac, _ = model.forward(f, j)
            rows = slice(first, first + len(p))
            arrays[0][rows] = v[:, :, 0]
            arrays[1][rows] = v[:, :, 1]
            arrays[2][rows] = jac[:, :, 1, 0] - jac[:, :, 0, 1]
            arrays[3][rows] = jac[:, :, 0, 0] + jac[:, :, 1, 1]
        return (arrays, explicit_checkpoint, None)
    checkpoint = explicit_checkpoint
    source = WORK / 'periodic_core/vortex_merger/canonical_gauge/scripts/evaluate_large_comparison.py'
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location('merger_original_evaluator', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = source.read_text()
    node = next((x for x in ast.parse(code).body if isinstance(x, ast.FunctionDef) and x.name == 'fields'))
    body = ast.get_source_segment(code, node).replace('128', str(n))
    namespace = dict(vars(module))
    exec(compile(body, str(source) + '::fields', 'exec'), namespace)
    cache = dest / 'analytic_jets'
    cache.mkdir(exist_ok=True)
    cfg, meta, raw = namespace['fields'](checkpoint, cache)
    assert cfg.modes == K and meta['epoch'] == 2000
    arrays = [cp.asarray(raw[k]) for k in (1, 2, 3)]
    training = None
    return (arrays, checkpoint, training)

def main(method, seed, n):
    dest = OUT / f'{method.lower()}_seed{seed}_n{n}'
    dest.mkdir(parents=True, exist_ok=True)
    stop_check(dest)
    if (dest / 'complete.json').exists():
        return
    done = guard(cp, dest)
    started = time.perf_counter()
    try:
        summary = read(SOURCE / 'summary.json')
        assert summary['separation'] == 1.35 and summary['frames'] == 241 and (summary['K'] == K)
        with np.load(SOURCE / 'dns.npz') as z:
            gt = dict(time=z['time'], velocity_u=z['u'], velocity_v=z['v'], vorticity=z['omega'], nu=summary['viscosity'])
        initial = {k: resample(resample(gt[k][0], n, axis=0), n, axis=1)[None].real for k in ('velocity_u', 'velocity_v')}
        refined = dict(gt, **initial)
        tick = time.perf_counter()
        arrays, checkpoint, training = build_fields(method, seed, n, dest)
        cp.cuda.Stream.null.synchronize()
        sampling = time.perf_counter() - tick
        projection_started = time.perf_counter()
        gu, gv, gw = arrays[:3]
        weight = (2 * np.pi / n) ** 2
        if method != 'NKB':
            norm = cp.sqrt(weight * cp.sum(gu * gu + gv * gv, axis=0))
            assert bool(cp.all(norm > 0))
            for field in arrays:
                field /= norm
            del norm
        mass = weight * (gu.T @ gu + gv.T @ gv)
        stiff = weight * (gw.T @ gw)
        mh = cp.asnumpy((mass + mass.T) / 2)
        sh = cp.asnumpy((stiff + stiff.T) / 2)
        ev = np.linalg.eigvalsh(mh)
        rank = int(np.sum(ev > ev[-1] * 1e-12))
        if rank != K:
            write(dest / 'scientific_failure.json', dict(reason='Full mass rank failed', rank=rank, K=K, ridge=0, modes_removed=0))
            raise ArithmeticError('Full-K rank gate')
        vals, transform = la.eigh(sh, mh)
        assert vals.min() > -1e-09
        del gu, gv, gw
        T = cp.asarray(transform)
        for i in range(len(arrays)):
            arrays[i] = arrays[i] @ T
        gu, gv, gw = arrays[:3]
        del T, mass, stiff
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        c0 = weight * (gu.T @ cp.asarray(initial['velocity_u'][0].ravel()) + gv.T @ cp.asarray(initial['velocity_v'][0].ravel()))
        divergence = float(cp.linalg.norm(arrays[3] @ c0) / cp.maximum(cp.linalg.norm(gw @ c0), 1e-30)) if method == 'NKB' else 0.0
        arrays = arrays[:3]
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Stream.null.synchronize()
        projection_seconds = time.perf_counter() - projection_started
        write(dest / 'projection.json', dict(rank=rank, K=K, ridge=0, modes_removed=0, condition=float(ev[-1] / ev[0]), initial_divergence_to_curl_L2=divergence, checkpoint=str(checkpoint), basis_sampling_seconds=sampling, projection_and_ritz_seconds=projection_seconds))
        equation = inherited_check(gu, gv, gw, cp.asarray(vals), refined, dest, n)
        write(dest / 'original_equation_check.json', equation)
        probe_results = []
        for dt in (0.005, 0.0025):
            tick = time.perf_counter()
            rollout(gu, gv, gw, cp.asarray(vals), dict(refined, time=np.array([0.0, 0.1])), dt, dest, 'timing_probe')
            measured = time.perf_counter() - tick
            estimate = measured / 0.1 * float(gt['time'][-1]) + 30
            probe_results.append(dict(dt=dt, probe_seconds=measured, estimate_seconds=estimate, admitted=estimate * 1.25 + 60 <= 7200))
        write(dest / 'duration_admission.json', dict(probes=probe_results, hard_cap_each_trajectory_seconds=7200))
        if not all((r['admitted'] for r in probe_results)):
            write(dest / 'budget_stop.json', dict(reason='Original duration admission not met', probes=probe_results))
            return
        results = []
        for dt in (0.005, 0.0025):
            marker = dest / f'metrics_dt{dt}.json'
            fieldsfile = dest / f'fields_dt{dt}.npz'
            if not marker.exists() or not fieldsfile.exists():
                fields, states, skew, online, reconstruct = rollout(gu, gv, gw, cp.asarray(vals), refined, dt, dest, 'autonomous_ETDRK4')
                u, v, w = fields
                square = np.sum((u - gt['velocity_u']) ** 2 + (v - gt['velocity_v']) ** 2, axis=(1, 2))
                norm = np.sum(gt['velocity_u'] ** 2 + gt['velocity_v'] ** 2, axis=(1, 2))
                errors = np.sqrt(square / norm)
                tick = time.perf_counter()
                np.savez_compressed(fieldsfile, u=u, v=v, omega=w, coefficients=states, time=gt['time'])
                write(marker, dict(dt=dt, relative_velocity_L2=errors.tolist(), velocity_mse=(square / (2 * 128 ** 2)).tolist(), initial_error=float(errors[0]), final_error=float(errors[-1]), mean_relative_velocity_L2=float(errors.mean()), mean_velocity_mse=float((square / (2 * 128 ** 2)).mean()), online_with_coefficient_capture_seconds=online, reconstruction_seconds=reconstruct, write_seconds=time.perf_counter() - tick, energy_skew_defect=skew))
            with np.load(fieldsfile) as z:
                states = z['coefficients'].copy()
            results.append((read(marker), states))
        delta = np.linalg.norm(results[0][1] - results[1][1], axis=1) / np.maximum(np.linalg.norm(results[1][1], axis=1), 1e-30)
        if float(delta.max()) > 0.001:
            write(dest / 'scientific_failure.json', dict(reason='Time-step gate failed', maximum=float(delta.max()), threshold=0.001))
            return
        record = dict(case='periodic', method=method, K=K, seed=seed, initial_separation=1.35, final_time=18 * np.pi, frames=241, quadrature_N=n, observation_N=128, nu=gt['nu'], reference=str(SOURCE / 'dns.npz'), checkpoint=str(checkpoint), full_rank=True, modes_removed=0, ridge=0, training_seconds=training, assessment_dt=0.005, metrics=results[0][0], fine_metrics=results[1][0], max_mass_norm_dt_difference=float(delta.max()), initial_error=results[0][0]['initial_error'], final_error=results[0][0]['final_error'], mean_relative_velocity_L2=results[0][0]['mean_relative_velocity_L2'], mean_velocity_mse=results[0][0]['mean_velocity_mse'], temporal_differences=delta.tolist(), mass_condition=float(ev[-1] / ev[0]), initial_divergence_to_curl_L2=divergence, autonomous=True, future_reference_injection=False, current_evaluation_seconds=time.perf_counter() - started, original_equation_check=equation, basis_sampling_seconds=sampling, projection_and_ritz_seconds=projection_seconds)
        write(dest / 'complete.json', record)
        write(dest / 'progress.json', dict(phase='complete', pid=os.getpid(), unix=time.time(), method=method, seed=seed))
        print(json.dumps(record), flush=True)
    except BaseException:
        import traceback
        write(dest / 'failure.json', dict(error=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        done.set()
