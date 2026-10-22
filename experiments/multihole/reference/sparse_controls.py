"""Portable full-space, full-consistent-mass Laplacian P/PH controls, including K1024."""
from pathlib import Path
import argparse, ast, json, time, warnings
import numpy as np
from .run import load_mesh, write_json, native_guard

def assemble_p1(xy, tri, free, cp, coo_matrix):
    v = cp.asarray(xy[tri])
    a = v[:, 1] - v[:, 0]
    b = v[:, 2] - v[:, 0]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    assert bool(cp.all(determinant > 0))
    edge = cp.stack((v[:, 1] - v[:, 2], v[:, 2] - v[:, 0], v[:, 0] - v[:, 1]), axis=1)
    grad = cp.stack((edge[:, :, 1], -edge[:, :, 0]), axis=2) / determinant[:, None, None]
    area = determinant / 2
    nodes = cp.asarray(tri)
    row = cp.broadcast_to(nodes[:, :, None], (len(tri), 3, 3)).ravel()
    col = cp.broadcast_to(nodes[:, None, :], (len(tri), 3, 3)).ravel()
    stiffness = coo_matrix(((area[:, None, None] * cp.einsum('tid,tjd->tij', grad, grad)).ravel(), (row, col)), shape=(len(xy), len(xy))).tocsr()
    mass = coo_matrix(((area[:, None, None] * (cp.eye(3) + 1) / 12).ravel(), (row, col)), shape=(len(xy), len(xy))).tocsr()
    F = cp.asarray(free)
    S = stiffness[F][:, F]
    M = mass[F][:, F]
    return (stiffness, mass, S, M)

def solve_modes(S, M, initial, count, iterations, cp, diags, lobpcg, history, guard=lambda: None):
    if initial.shape != (S.shape[0], count) or S.shape[0] <= 5 * count:
        raise ValueError('Complete block and full sparse space (>5K free nodes) required')
    guard()
    rates, vectors, residual_history = lobpcg(S, cp.asarray(initial), B=M, M=diags(1 / S.diagonal()), largest=False, tol=1e-12, maxiter=iterations, retResidualNormsHistory=True)
    guard()
    order = cp.argsort(rates)
    rates = rates[order]
    vectors = vectors[:, order]
    residual = S @ vectors - M @ vectors * rates
    relative = cp.linalg.norm(residual, axis=0) / cp.linalg.norm(S @ vectors, axis=0)
    if not bool(cp.all(cp.isfinite(rates))) or not bool(cp.all(cp.isfinite(vectors))) or (not bool(cp.all(rates > 0))):
        raise FloatingPointError('Invalid complete eigenpairs')
    vectors = vectors / cp.sqrt(rates)[None, :]
    gram = float(cp.max(abs(vectors.T @ (S @ vectors) - cp.eye(count))))
    if float(relative.max()) >= 1e-08 or gram >= 1e-08:
        raise ArithmeticError(f'Full true-residual/energy-Gram gates failed: {float(relative.max())}, {gram}')
    return (rates, vectors, dict(relative_residuals=cp.asnumpy(relative), true_relative_history=np.asarray(history), energy_gram_max=gram, iterations_limit=iterations, history_entries=len(residual_history)))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--case', choices=('one_circle', 'circle_square', 'circle_square_triangle'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', type=int, default=1024)
    parser.add_argument('--positive-only', action='store_true')
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--initial-seed', type=int, default=20260928)
    parser.add_argument('--iterations', type=int, default=800)
    parser.add_argument('--timeout-seconds', type=float, default=7200)
    parser.add_argument('--pool-bytes', type=int, default=5 * 2 ** 30)
    a = parser.parse_args(argv)
    if a.modes < 1 or a.iterations < 1 or a.timeout_seconds <= 0 or (not 0 < a.pool_bytes <= 15500000000):
        parser.error('Positive mode/iteration/deadline and valid pool limit required')
    mesh = load_mesh(a.mesh, a.case)
    D = np.unique(mesh.boundary_edges)
    free = np.setdiff1d(np.arange(len(mesh.xy)), D)
    h = 0 if a.positive_only else len(mesh.obstacles)
    count = a.modes - h
    if count < 1 or len(free) <= 5 * count:
        parser.error('Full sparse mesh must have more than 5 positive-mode counts of free nodes')
    if a.initial:
        with np.load(a.initial, allow_pickle=False) as z:
            initial = z['free_vectors'] * np.sqrt(z['eigenvalues'])[None, :]
        if initial.shape != (len(free), count):
            parser.error('Warm start must match full mesh and exact positive-mode block')
    else:
        initial = np.random.default_rng(a.initial_seed).standard_normal((len(free), count))
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    import cupy as cp
    from cupyx.scipy.sparse import coo_matrix, diags
    from cupyx.scipy.sparse.linalg import LinearOperator
    from .lobpcg_true_residual import lobpcg, TRUE_RESIDUAL_HISTORY
    cp.get_default_memory_pool().set_limit(size=a.pool_bytes)
    sample = native_guard(cp)

    def guard():
        sample()
        if time.perf_counter() - started > a.timeout_seconds:
            raise TimeoutError('Bounded Laplacian stage deadline')
    stiffness, mass, S, M = assemble_p1(mesh.xy, mesh.tri, free, cp, coo_matrix)

    def apply(A, X):
        guard()
        return A @ X
    SA = LinearOperator(S.shape, matvec=lambda X: apply(S, X), matmat=lambda X: apply(S, X), dtype=S.dtype)
    SA.diagonal = S.diagonal
    rates, vectors, facts = solve_modes(SA, M, initial, count, a.iterations, cp, diags, lobpcg, TRUE_RESIDUAL_HISTORY, guard)
    potentials = cp.zeros((len(mesh.xy), a.modes))
    potentials[cp.asarray(free), :count] = vectors
    if h:
        from .solver_sparse import SparseFactor, sparse_solve
        H = cp.zeros((len(mesh.xy), h))
        LS = SparseFactor(S)
        for j in range(h):
            H[cp.asarray(mesh.boundary_nodes[j + 1]), j] = 1.0
            H[cp.asarray(free), j] = sparse_solve(LS, -(stiffness @ H[:, j])[cp.asarray(free)])
        gram = H.T @ stiffness @ H
        R = cp.linalg.cholesky(gram)
        from cupyx.scipy.linalg import solve_triangular
        transform = solve_triangular(R.T, cp.eye(h, dtype=cp.float64), lower=False)
        potentials[:, count:] = H @ transform
    allrates = cp.concatenate((rates, cp.zeros(h)))
    guard()
    cp.cuda.Stream.null.synchronize()
    fullgram = float(cp.max(abs(potentials.T @ (stiffness @ potentials) - cp.eye(a.modes))))
    if fullgram >= 1e-08:
        raise ArithmeticError('Full P/PH energy Gram gate failed')
    np.savez_compressed(a.output / 'controls.npz', xy=mesh.xy, tri=mesh.tri, positive_free_nodes=free, free_vectors=cp.asnumpy(vectors), potentials=cp.asnumpy(potentials), eigenvalues=cp.asnumpy(rates), all_rates=cp.asnumpy(allrates), relative_residuals=facts.pop('relative_residuals'), true_relative_history=facts.pop('true_relative_history'))
    write_json(a.output / 'controls.json', dict(case=a.case, total_modes=a.modes, positive_modes=count, harmonic_modes=h, full_consistent_mass=True, full_true_residual_threshold=1e-08, full_energy_Gram_max=fullgram, actual_native_seconds=time.perf_counter() - started, initial='complete supplied iterate' if a.initial else 'declared PCG64 fullspace initial block', initial_seed=a.initial_seed, **facts))
    print(json.dumps({'output': str(a.output.resolve()), 'modes': a.modes}))
if __name__ == '__main__':
    main()
