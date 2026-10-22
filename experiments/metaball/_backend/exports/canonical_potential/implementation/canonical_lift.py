import argparse
import json
import os
from _neb_paths import Path
import threading
import time
import traceback
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import numpy as np
import psutil
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
from pardiso_multi import PardisoSPD
HERE = Path(__file__).resolve().parent
ASSETS = Path('_data/metaball/physical_complex/route2_physical_complex_q1')
OLD_CERTIFICATE = HERE.parents[1] / 'physical_complex/route2_physical_complex_q1/complete.json'

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)

class CanonicalLift:

    def __init__(self, output, log=None, *, projected_preconditioner=True):
        self.log = log or (lambda *a, **kw: None)
        self.projected_preconditioner = projected_preconditioner
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        certificate = json.loads(OLD_CERTIFICATE.read_text(encoding='utf-8-sig'))
        if not (certificate['qualified'] and certificate['CG_exact_integer_zero'] and (certificate['scalar_dof'] == 29352) and (certificate['relative_edges'] == 239415) and (certificate['harmonic_dimension'] == 1)):
            raise ValueError('The actual full relative-complex certificate changed')
        self.inputs = {str(p): None for p in [OLD_CERTIFICATE, ASSETS / 'M1.npz', ASSETS / 'S1.npz', ASSETS / 'G.npz', ASSETS / 'C.npz', ASSETS / 'complex.npz']}
        self.M = sp.load_npz(ASSETS / 'M1.npz').tocsr()
        self.S = sp.load_npz(ASSETS / 'S1.npz').tocsr()
        self.G = sp.load_npz(ASSETS / 'G.npz').tocsr()
        self.C = sp.load_npz(ASSETS / 'C.npz').tocsr()
        with np.load(ASSETS / 'complex.npz') as saved:
            self.H = saved['H'].copy()
        n = self.M.shape[0]
        if self.M.shape != (239415, 239415) or self.S.shape != self.M.shape:
            raise ValueError('Incomplete physical mass/stiffness')
        if self.G.shape != (n, 29352) or self.H.shape != (n, 1):
            raise ValueError('Incomplete scalar or relative harmonic kernel')
        K = (self.G.T @ self.M @ self.G).tocsc()
        started = time.perf_counter()
        self.scalar = sla.splu(K)
        self.scalar_factor_seconds = time.perf_counter() - started
        self.log('full_scalar_factor_complete', dof=29352, seconds=self.scalar_factor_seconds, L_nnz=self.scalar.L.nnz, U_nnz=self.scalar.U.nnz)
        started = time.perf_counter()
        self.shifted = PardisoSPD(self.S + self.M, self.output / 'pardiso_ooc', self.log)
        self.shifted_factor_seconds = time.perf_counter() - started
        self.log('shifted_preconditioner_complete', seconds=self.shifted_factor_seconds)
        self.records = []

    def project(self, a):
        z = a - self.G @ self.scalar.solve(np.asarray(self.G.T @ (self.M @ a)))
        return z - self.H @ (self.H.T @ (self.M @ z))

    def project_transpose(self, a):
        z = a - self.M @ (self.G @ self.scalar.solve(np.asarray(self.G.T @ a)))
        return z - self.M @ (self.H @ (self.H.T @ z))

    def precondition(self, residual):
        if self.projected_preconditioner:
            return self.project(self.shifted.solve(self.project_transpose(residual)))
        return self.shifted.solve(residual)

    def solve(self, rhs, *, rtol=1e-12, maxiter=80, label='canonical_forward'):
        started = time.perf_counter()
        rhs = np.asarray(rhs, dtype='f8')
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.M.shape[0] or (not np.isfinite(rhs).all()):
            raise ValueError('Invalid full physical RHS')
        b = self.project_transpose(rhs)
        norm = np.linalg.norm(b, axis=0)
        active = norm > 0
        x = np.zeros_like(b)
        if not active.any():
            record = dict(label=label, columns=rhs.shape[1], iterations=0, relative_residuals=[0.0] * rhs.shape[1], exact_zero_rhs=True, seconds=time.perf_counter() - started, pure_kernel_does_not_produce_finite_mode=True)
            self.records.append(record)
            return (x[:, 0] if vector else x, record)
        scaled_b = np.divide(b, norm, out=np.zeros_like(b), where=norm[None, :] > 0)
        r = scaled_b.copy()
        w = self.precondition(r)
        p = w.copy()
        rho = np.sum(r * w, axis=0)
        history = []
        iteration = 0
        for iteration in range(1, maxiter + 1):
            ap = self.S @ p
            pap = np.sum(p * ap, axis=0)
            if np.any(active & ((pap <= 0) | ~np.isfinite(pap) | (rho <= 0))):
                raise FloatingPointError('Loss of positive coexact PCG direction')
            alpha = np.divide(rho, pap, out=np.zeros_like(rho), where=active)
            x += p * alpha
            r -= ap * alpha
            relative = np.linalg.norm(r, axis=0)
            actual = np.linalg.norm(scaled_b - self.S @ x, axis=0)
            history.append(float(actual.max()))
            self.log('canonical_PCG', label=label, iteration=iteration, columns=b.shape[1], maximum_relative_residual=float(actual.max()))
            active &= actual > rtol
            if not active.any():
                break
            w = self.precondition(r)
            next_rho = np.sum(r * w, axis=0)
            beta = np.divide(next_rho, rho, out=np.zeros_like(rho), where=active)
            p = w + p * beta
            p[:, ~active] = 0
            rho = next_rho
        x = self.project(x) * norm
        actual = np.linalg.norm(b - self.S @ x, axis=0) / np.maximum(norm, 1e-300)
        mx = self.M @ x
        scale = np.maximum(np.linalg.norm(mx, axis=0), 1e-300)
        gauge = np.linalg.norm(self.G.T @ mx, axis=0) / scale
        harmonic = np.linalg.norm(self.H.T @ mx, axis=0) / scale
        record = dict(label=label, columns=rhs.shape[1], iterations=iteration, relative_residuals=actual.tolist(), maximum_relative_residual=float(actual.max()), gauge_relative=gauge.tolist(), harmonic_relative=harmonic.tolist(), projected_preconditioner=self.projected_preconditioner, zero_start=True, history=history, actual_operator='unshifted full physical S', preconditioner_only='P (S+M)^-1 P^T', seconds=time.perf_counter() - started)
        self.records.append(record)
        if max(float(actual.max()), float(gauge.max()), float(harmonic.max())) > rtol * 10:
            raise RuntimeError('Actual curl inverse/canonical qualification failure: ' + json.dumps(record))
        return (x[:, 0] if vector else x, record)

    def adjoint(self, cotangent, **kw):
        return self.solve(cotangent, label='canonical_adjoint', **kw)

    def close(self):
        self.shifted.close()

def selftest(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        raise RuntimeError('Completed bounded CPU qualification must not be repeated')
    started = time.perf_counter()
    done = threading.Event()
    creation = psutil.Process().create_time()
    resources = []

    def log(phase, **kw):
        row = dict(phase=phase, pid=os.getpid(), creation=creation, unix=time.time(), stage_elapsed_seconds=time.perf_counter() - started, **kw)
        write(output / 'progress.json', row)
        print(json.dumps(row, ensure_ascii=True), flush=True)

    def watch():
        while not done.is_set():
            resource = dict(pid=os.getpid(), creation=creation, unix=time.time(), seconds=time.perf_counter() - started, rss=psutil.Process().memory_info().rss, hostfree=psutil.virtual_memory().available, CPU_only=True)
            resources.append(resource)
            with (output / 'resources.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(resource) + '\n')
            if resource['rss'] >= 8 * 2 ** 30 or resource['hostfree'] <= 9 * 2 ** 30 or resource['seconds'] >= args.timeout:
                write(output / 'bounded_stop.json', resource)
                os._exit(124)
            done.wait(2)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    solver = None
    try:
        with threadpool_limits(limits=2):
            solver = CanonicalLift(output, log, projected_preconditioner=not args.raw_preconditioner)
            rng = np.random.default_rng(20261008)
            rows = []
            for columns in args.columns:
                known = solver.project(rng.normal(size=(solver.M.shape[0], columns)))
                rhs = solver.S @ known
                result, record = solver.solve(rhs, label=f'known_coexact_{columns}')
                error = float(np.linalg.norm(result - known) / np.linalg.norm(known))
                if error >= 1e-08:
                    raise AssertionError(('Known full coexact solution mismatch', error))
                rows.append(dict(record, known_solution_relative=error))
            first = solver.project(rng.normal(size=(solver.M.shape[0], 1)))
            second = solver.project(rng.normal(size=(solver.M.shape[0], 1)))
            a, b = (solver.S @ first, solver.S @ second)
            fa, _ = solver.solve(a, label='symmetric_forward')
            fb, _ = solver.adjoint(b)
            dot = abs(float(np.sum(fa * b) - np.sum(a * fb)))
            dot /= max(float(np.linalg.norm(fa) * np.linalg.norm(b)), 1e-300)
            if dot >= 1e-09:
                raise AssertionError(('Canonical inverse adjoint failure', dot))
            zero, zero_record = solver.solve(np.zeros((solver.M.shape[0], 4)), label='exact_curl_free_rhs')
            if np.any(zero != 0):
                raise AssertionError('Pure gradient curl=0 input produced nonzero canonical lift')
            result = dict(status='complete', qualified=True, scope='Full physical discrete canonical inverse only; no continuous ANN qualification or training', zero_ANN_curl_maps_to_exact_zero_rhs_and_lift=True, zero_lift_cannot_be_accepted_as_finite_Rayleigh_mode=True, pure_gradient_manufactured_ANN_not_executed=True, original_scalar_dof=29352, original_harmonic_dof=1, original_relative_edges=239415, forward_and_adjoint_same_symmetric_inverse=True, adjoint_relative=dot, rows=rows, inputs=solver.inputs, factor_analysis=solver.shifted.analysis, scalar_factor_seconds=solver.scalar_factor_seconds, shifted_preconditioner_factor_seconds=solver.shifted_factor_seconds, seconds=time.perf_counter() - started, resources=len(resources), pid=os.getpid(), creation=creation, optimizer_updates=0, CUDA_imported=False, complete_K32_full2000_budget_not_measured=True)
            write(output / 'complete.json', result)
            log('complete', qualified=True, adjoint_relative=dot)
    except BaseException:
        write(output / 'failure.json', dict(error=traceback.format_exc(), seconds=time.perf_counter() - started, pid=os.getpid(), creation=creation))
        raise
    finally:
        if solver is not None:
            solver.close()
        done.set()
        thread.join(timeout=3)
