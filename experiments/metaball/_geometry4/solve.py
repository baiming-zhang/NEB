from __future__ import annotations
import os,sys,time,json,hashlib,ast,gc,itertools
from pathlib import Path
import numpy as np,scipy.sparse as sp,scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet,Basis,ElementTetN1
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
import importlib.util
from .curved import complex_from
def spectrum(chart):
    assert not (OUT / 'eigensolve_complete.json').exists(), 'Completed spectrum must not be repeated'
    M, S = chart.assemble(progress)
    progress('canonical_deflation')
    c = complex_from(chart, M, S)
    del M, S
    gc.collect()
    spec = importlib.util.spec_from_file_location('curved_pardiso', Path(__file__).parent / 'pardiso_ooc.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OOC = OUT / 'pardiso_ooc'
    module.OOC.mkdir(exist_ok=True)
    os.environ['MKL_PARDISO_OOC_PATH'] = str(module.OOC)
    os.environ['MKL_PARDISO_OOC_FILE_NAME'] = f'curved_case4_{os.getpid()}'
    progress('shifted_consistent_factorization', free_edges=len(c.free))
    tick = time.perf_counter()
    factor = module.PardisoOOC(c.S + c.M, progress)
    factor_seconds = time.perf_counter() - tick
    write('factorization.json', dict(seconds=factor_seconds, **factor.analysis))
    calls = [0]
    elapsed = [0.0]

    def apply(x):
        tick = time.perf_counter()
        y = c.project(factor.solve(c.project_transpose(x)))
        calls[0] += 1
        elapsed[0] += time.perf_counter() - tick
        if calls[0] == 1 or calls[0] % 25 == 0:
            progress('generalized_eigensolve', operator_calls=calls[0], operator_mean_seconds=elapsed[0] / calls[0], requested_modes=256, ritz_vectors=264)
        return y
    op = sla.LinearOperator(c.S.shape, matvec=apply, dtype=np.float64)
    tick = time.perf_counter()
    try:
        lam, A = sla.eigsh(c.S, k=264, M=c.M, sigma=-1, which='LM', OPinv=op, tol=1e-09, v0=c.project(np.random.default_rng(20260907).normal(size=len(c.free))))
    finally:
        factor.close()
    eigseconds = time.perf_counter() - tick
    del factor, op
    gc.collect()
    A = c.project(A)
    gram = A.T @ (c.M @ A)
    w, v = np.linalg.eigh((gram + gram.T) * 0.5)
    assert w.min() > 0
    A = A @ (v / np.sqrt(w) @ v.T)
    reduced = A.T @ (c.S @ A)
    lam, rot = np.linalg.eigh((reduced + reduced.T) * 0.5)
    A = A @ rot[:, :256]
    lam = lam[:256]
    assert lam.min() > 0
    residual = []
    for off in range(0, 256, 16):
        a = A[:, off:off + 16]
        ma = c.M @ a * lam[off:off + 16]
        residual.extend((np.linalg.norm(c.S @ a - ma, axis=0) / np.linalg.norm(ma, axis=0)).tolist())
    orth = float(np.max(np.abs(A.T @ (c.M @ A) - np.eye(256))))
    assert max(residual) < 1e-06 and orth < 1e-08
    np.savez(OUT / 'modes.npz', free_edges=c.free, potential=A, eigenvalues=lam, source_sha=np.array(chart.source_sha))
    write('eigensolve_complete.json', dict(status='complete', K=256, ritz_vectors=264, tolerance=1e-09, maximum_residual=max(residual), residuals=residual, M_orthogonality_max=orth, factor_seconds=factor_seconds, eigensolve_seconds=eigseconds, operator_calls=calls[0], source_sha=chart.source_sha, method='Fresh quadratic-chart physical consistent Nedelec S/M, original canonical gradient/harmonic deflation and shift-invert ARPACK, FP64 Rayleigh-Ritz'))

def evaluate(chart):
    assert not (OUT / 'result.json').exists(), 'Do not repeat complete reconstruction'
    r = read(OUT / 'eigensolve_complete.json')
    assert r['source_sha'] == chart.source_sha and r['maximum_residual'] < 1e-06
    with np.load(OUT / 'modes.npz') as z:
        A = z['potential']
        lam = z['eigenvalues']
        free = z['free_edges']
    slots = np.full(len(chart.edges), -1, dtype=int)
    slots[free] = np.arange(len(free))
    q = read(CHART / 'qualification.json')
    samples = Path(q['source_samples'])
    assert sha(samples) == q['source_samples_sha256']
    with np.load(samples) as z:
        truth = {k: z[k] for k in z.files}

    def matrix(z, off):
        ids = z['cell_ids'][off:off + 1024]
        cells = chart.t.T[ids]
        v = chart.p.T[cells]
        iv = np.linalg.inv((v[:, 1:] - v[:, :1]).transpose(0, 2, 1))
        grad = np.concatenate((-iv.sum(1)[:, None], iv), axis=1)
        curls = []
        for a, b in PAIRS:
            curls.append(2 * np.cross(grad[:, a], grad[:, b]) * (1 - 2 * (cells[:, a] > cells[:, b]))[:, None])
        loc = slots[chart.mesh.t2e.T[ids]]
        pot = A[np.maximum(loc, 0)] * (loc >= 0)[:, :, None]
        U = np.einsum('ani,nak->nik', np.array(curls), pot) / np.sqrt(lam)[None, None, :]
        U = np.einsum('nij,njk->nik', z['J'][off:off + 1024] / z['det'][off:off + 1024, None, None], U)
        return U.reshape(-1, 256)
    mass = np.zeros((256, 256))
    rhs = np.zeros(256)
    with np.load(CHART / 'fit_points_mapped.npz') as z:
        assert len(z['cell_ids']) == 16384
        for off in range(0, 16384, 1024):
            U = matrix(z, off)
            y = truth['fit_target'][off:off + 1024].reshape(-1)
            mass += U.T @ U
            rhs += U.T @ y
    scale = np.sqrt(np.diag(mass))
    e = np.linalg.eigvalsh(mass / scale[:, None] / scale[None, :])
    assert e[0] > 1e-10
    coeff = np.linalg.solve(mass, rhs)
    normal = float(np.linalg.norm(mass @ coeff - rhs) / np.linalg.norm(rhs))
    assert normal < 1e-08
    fit_difference = 0.0
    fit_prediction_energy = 0.0
    fit_target_energy = float(np.sum(truth['fit_target'] ** 2))
    with np.load(CHART / 'fit_points_mapped.npz') as z:
        for off in range(0, 16384, 1024):
            prediction = (matrix(z, off) @ coeff).reshape(-1, 3)
            y = truth['fit_target'][off:off + 1024]
            fit_difference += float(np.sum((prediction - y) ** 2))
            fit_prediction_energy += float(np.sum(prediction ** 2))
    fit_error = float(np.sqrt(fit_difference / fit_target_energy))
    fit_capture = fit_prediction_energy / fit_target_energy
    cMc = float(coeff @ mass @ coeff)
    mass_energy_relative = abs(cMc - fit_prediction_energy) / max(fit_prediction_energy, 1e-30)
    assert mass_energy_relative < 1e-08 and abs(fit_capture - (1 - fit_error ** 2)) < 1e-08
    pred = []
    with np.load(CHART / 'assessment_points_mapped.npz') as z:
        assert len(z['cell_ids']) == 16384
        for off in range(0, 16384, 1024):
            pred.append((matrix(z, off) @ coeff).reshape(-1, 3))
    pred = np.concatenate(pred)
    target = truth['assessment_target']
    error = float(np.linalg.norm(pred - target) / np.linalg.norm(target))
    assert np.isfinite(error)
    learned = None
    np.savez(OUT / 'reconstruction_arrays.npz', mass=mass, rhs=rhs, coefficients=coeff, prediction=pred)
    evaluation_seconds = time.perf_counter() - START
    timing = {stage: read(OUT / f'{stage}_stage_complete.json')['seconds'] for stage in ('assemble', 'solve')}
    write('result.json', dict(status='complete', case=4, K=256, rank=256, scaled_gram_min_eigenvalue=float(e[0]), learned_error=learned, numerical_error=error, fit_points=16384, assessment_points=16384, normal_residual=normal, fit_direct_error=fit_error, fit_captured_energy=fit_capture, cMc_vs_direct_energy_relative=mass_energy_relative, scaled_condition=float(e[-1] / e[0]), seconds=sum(timing.values()) + evaluation_seconds, evaluation_seconds=evaluation_seconds, stage_seconds=timing, source_samples=str(samples), source_samples_sha256=sha(samples), source_chart_sha256=chart.source_sha, boundary_geometric_distance_max=q['boundary_geometric_distance_max'], method='Target-specific consistent physical Nedelec K256 on globally qualified quadratic approximation of the original smooth geometry, full256 fit and direct disjoint assessment on all original targets/points', all_original_points_retained=True, original_failed_radial_results_preserved=True))
