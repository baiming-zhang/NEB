from pathlib import Path
from contextlib import contextmanager
import json, os, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components, breadth_first_order
from train import V, load, resource
import sys
sys.path.insert(0, str(V / 'exports/canonical_potential/implementation'))
from native_cudss import NativeCuDSSPair
_pardiso = load('public_pardiso', V / 'exports/canonical_potential/implementation/pardiso_incore_scoped.py')
ScopedPardisoSPD = _pardiso.InCoreScopedPardisoSPD
ASSETS = Path('data/metaball/physical')
CERTIFICATE = ASSETS / 'complete.json'
PAIRS = np.array([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)])

class NativeFactor:

    def __init__(self, matrix, output, log):
        import cupy as cp
        import cupyx.scipy.sparse as csp
        self.cp = cp
        self.output = output
        host = matrix.tocsr()
        transpose = host.T.tocsr()
        self.pair = NativeCuDSSPair(cp, csp.csr_matrix(host), csp.csr_matrix(transpose), host, transpose, lambda: resource(cp, output), dll=os.environ['NEB_CUDSS_LIBRARY'], log=log)
        self.analysis = self.pair.analysis

    @contextmanager
    def local_threads(self, threads):
        yield

    def solve(self, rhs):
        return self.pair.solve(rhs)

    def close(self):
        self.pair.close()

def relative_tree(p, t, free, vn, bn, G, C):
    edges = np.unique(np.sort(t[:, PAIRS].reshape(-1, 2), axis=1), axis=0)
    wall_edges = edges[np.setdiff1d(np.arange(len(edges)), free)]
    wall_graph = sp.coo_matrix((np.ones(2 * len(wall_edges)), (np.r_[wall_edges[:, 0], wall_edges[:, 1]], np.r_[wall_edges[:, 1], wall_edges[:, 0]])), shape=(len(p), len(p))).tocsr()[bn][:, bn]
    count, labels = connected_components(wall_graph)
    if count != 2:
        raise ValueError('Original relative harmonic dimension requires two closed wall components')
    outer = int(np.argmax([np.ptp(p[bn[labels == i]], axis=0).max() for i in range(count)]))
    nv = len(vn)
    node = np.full(len(p), -1, dtype='i8')
    node[vn] = np.arange(nv)
    node[bn[labels == outer]] = nv
    node[bn[labels != outer]] = nv + 1
    if np.any(node < 0):
        raise ValueError('An original vertex has no relative graph node')
    ends = node[edges[free]]
    n = nv + 2
    nonloops = ends[:, 0] != ends[:, 1]
    graph = sp.coo_matrix((np.ones(2 * nonloops.sum()), (np.r_[ends[nonloops, 0], ends[nonloops, 1]], np.r_[ends[nonloops, 1], ends[nonloops, 0]])), shape=(n, n)).tocsr()
    order, pred = breadth_first_order(graph, nv, directed=False, return_predecessors=True)
    if len(order) != n:
        raise ValueError('Original full relative edge graph is disconnected')
    pair = np.sort(ends, axis=1)
    code = pair[:, 0] * n + pair[:, 1]
    unique, first = np.unique(code, return_index=True)
    children = order[1:]
    tree_code = np.minimum(children, pred[children]) * n + np.maximum(children, pred[children])
    tree = first[np.searchsorted(unique, tree_code)]
    if len(tree) != 29353 or len(np.unique(tree)) != len(tree):
        raise ValueError('The complete relative scalar/harmonic tree size changed')
    cotree = np.setdiff1d(np.arange(len(free)), tree)
    if len(cotree) != 210062:
        raise ValueError('Incomplete physical cotree quotient')
    column = ends.copy()
    ground = column == nv
    column[column == nv + 1] = nv
    rows = np.repeat(np.arange(len(free)), 2).reshape(-1, 2)
    values = np.broadcast_to([-1.0, 1.0], column.shape)
    D = sp.coo_matrix((values[~ground], (rows[~ground], column[~ground])), shape=(len(free), nv + 1)).tocsr()
    D.sum_duplicates()
    D.eliminate_zeros()
    difference = D[:, :nv] - G
    difference.eliminate_zeros()
    CD = C @ D
    CD.eliminate_zeros()
    if difference.nnz or CD.nnz:
        raise ValueError('Relative tree incidence does not match all original G/harmonic curl kernel')
    graph_column = np.where(children == nv + 1, nv, children)
    rooted = D[tree][:, graph_column].tocsr()
    above = sp.triu(rooted, k=1).tocsr()
    above.eliminate_zeros()
    if above.nnz or not np.all(np.abs(rooted.diagonal()) == 1):
        raise ValueError('Rooted complete relative tree incidence is not unimodular triangular')
    depth = np.zeros(n, dtype='i8')
    for child in children:
        depth[child] = depth[pred[child]] + 1
    metadata = dict(boundary_components=count, collapsed_relative_vertices=n, interior_scalar_dof=nv, additional_inner_wall_scalar=1, tree_edges=len(tree), cotree_edges=len(cotree), tree_max_depth=int(depth.max()), all_original_relative_edges=len(free), self_loops_retained_in_cotree=int((~nonloops).sum()), original_G_exactly_equal_to_D_interior=True, CD_exact_integer_zero=True, grounded_tree_unimodular_triangular=True, grounding='outer boundary component; inner boundary retained as one additional node')
    return (tree, cotree, D, metadata)

class TreeCanonicalLift:

    def __init__(self, output, log=None, *, scalar_backend='superlu'):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda *a, **kw: None)
        certificate = json.loads(CERTIFICATE.read_text(encoding='utf-8-sig'))
        if not (certificate['qualified'] and certificate['CG_exact_integer_zero'] and (certificate['relative_edges'] == 239415) and (certificate['scalar_dof'] == 29352) and (certificate['harmonic_dimension'] == 1)):
            raise ValueError('The actual complete relative physical kernel certificate changed')
        self.inputs = {str(f): None for f in [CERTIFICATE, ASSETS / 'M1.npz', ASSETS / 'S1.npz', ASSETS / 'G.npz', ASSETS / 'C.npz', ASSETS / 'complex.npz']}
        self.M = sp.load_npz(ASSETS / 'M1.npz').tocsr()
        self.S = sp.load_npz(ASSETS / 'S1.npz').tocsr()
        self.G = sp.load_npz(ASSETS / 'G.npz').tocsr()
        C = sp.load_npz(ASSETS / 'C.npz').tocsr()
        with np.load(ASSETS / 'complex.npz') as saved:
            p, t, free, vn, bn, self.H = [saved[k].copy() for k in ('p', 't', 'free', 'vn', 'bn', 'H')]
        started = time.perf_counter()
        self.tree, self.cotree, self.D, self.topology = relative_tree(p, t, free, vn, bn, self.G, C)
        self.tree_setup_seconds = time.perf_counter() - started
        self.SFF = self.S[self.cotree][:, self.cotree].tocsr()
        np.savez(self.output / 'relative_tree.npz', tree=self.tree, cotree=self.cotree)
        self.log('complete_relative_tree', seconds=self.tree_setup_seconds, S_cotree_nnz=self.SFF.nnz, **self.topology)
        self.scalar_backend = scalar_backend
        self.scalar = None
        self.factor = None
        K = (self.G.T @ self.M @ self.G).tocsc()
        started = time.perf_counter()
        if scalar_backend == 'superlu':
            self.scalar = sla.splu(K)
            self.scalar_analysis = dict(backend='SuperLU', L_nnz=self.scalar.L.nnz, U_nnz=self.scalar.U.nnz)
        elif scalar_backend == 'pardiso':
            self.scalar = ScopedPardisoSPD(K, self.output / 'scalar_pardiso_ooc', self.log)
            self.scalar_analysis = dict(backend='PardisoSPD', **self.scalar.analysis)
        else:
            raise ValueError('Unknown declared scalar inverse backend')
        self.scalar_factor_seconds = time.perf_counter() - started
        self.log('complete_scalar_factor', seconds=self.scalar_factor_seconds, **self.scalar_analysis)
        started = time.perf_counter()
        self.factor = NativeFactor(self.SFF, self.output, self.log)
        self.cotree_factor_seconds = time.perf_counter() - started
        self.log('complete_unshifted_cotree_factor', seconds=self.cotree_factor_seconds, **self.factor.analysis)
        self.records = []
        self.projection_records = []

    def _project(self, a, transpose=False):
        started = time.perf_counter()
        a = np.asarray(a, dtype='f8')
        vector = a.ndim == 1
        if vector:
            a = a[:, None]
        if transpose:
            t = time.perf_counter()
            rhs = np.asarray(self.G.T @ a)
            rhs_seconds = time.perf_counter() - t
            t = time.perf_counter()
            solution = self.scalar.solve(rhs)
            scalar_seconds = time.perf_counter() - t
            t = time.perf_counter()
            z = a - self.M @ (self.G @ solution)
            z -= self.M @ (self.H @ (self.H.T @ z))
            finish_seconds = time.perf_counter() - t
        else:
            t = time.perf_counter()
            rhs = np.asarray(self.G.T @ (self.M @ a))
            rhs_seconds = time.perf_counter() - t
            t = time.perf_counter()
            solution = self.scalar.solve(rhs)
            scalar_seconds = time.perf_counter() - t
            t = time.perf_counter()
            z = a - self.G @ solution
            z -= self.H @ (self.H.T @ (self.M @ z))
            finish_seconds = time.perf_counter() - t
        self.projection_records.append(dict(transpose=transpose, columns=a.shape[1], scalar_backend=self.scalar_backend, rhs_sparse_seconds=rhs_seconds, scalar_solve_seconds=scalar_seconds, finish_sparse_seconds=finish_seconds, total_seconds=time.perf_counter() - started))
        return z[:, 0] if vector else z

    def project(self, a):
        return self._project(a, False)

    def project_transpose(self, a):
        return self._project(a, True)

    def solve(self, rhs, *, rtol=1e-12, max_refinement=3, minimum_refinement=0, threads=2, label='tree_forward'):
        started = time.perf_counter()
        rhs = np.asarray(rhs, dtype='f8')
        vector = rhs.ndim == 1
        if vector:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != len(self.tree) + len(self.cotree) or (not np.isfinite(rhs).all()):
            raise ValueError('Invalid full physical RHS')
        proj_start = len(self.projection_records)
        b = self.project_transpose(rhs)
        norm = np.linalg.norm(b, axis=0)
        if not np.any(norm):
            zero = np.zeros_like(b)
            record = dict(label=label, columns=b.shape[1], exact_zero_rhs=True, zero_lift_cannot_be_finite_Rayleigh_mode=True, seconds=time.perf_counter() - started, refinement_passes=0, threads=threads)
            self.records.append(record)
            return (zero[:, 0] if vector else zero, record)
        gathered = np.asfortranarray(b[self.cotree])
        t = time.perf_counter()
        with self.factor.local_threads(threads):
            aF = self.factor.solve(gathered)
        direct_seconds = time.perf_counter() - t
        refinements = 0
        history = []
        refinement_seconds = 0.0
        while True:
            a = np.zeros_like(b)
            a[self.cotree] = aF
            x = self.project(a)
            actual = np.linalg.norm(b - self.S @ x, axis=0) / np.maximum(norm, 1e-300)
            history.append(float(actual.max()))
            if float(actual.max()) <= rtol and refinements >= minimum_refinement:
                break
            if refinements >= max_refinement:
                raise RuntimeError('Unshifted full physical S residual failed after exact refinement: ' + json.dumps(history))
            defect = gathered - self.SFF @ aF
            t = time.perf_counter()
            with self.factor.local_threads(threads):
                aF += self.factor.solve(defect)
            refinement_seconds += time.perf_counter() - t
            refinements += 1
        mx = self.M @ x
        scale = np.maximum(np.linalg.norm(mx, axis=0), 1e-300)
        gauge = np.linalg.norm(self.G.T @ mx, axis=0) / scale
        harmonic = np.linalg.norm(self.H.T @ mx, axis=0) / scale
        if max(float(gauge.max()), float(harmonic.max())) >= 1e-10:
            raise RuntimeError('Complete original M-canonical gauge failed')
        sections = self.projection_records[proj_start:]
        record = dict(label=label, columns=b.shape[1], threads=threads, actual_operator='unshifted complete original physical S', exact_equivalent_factor='cotree principal S_FF; no shift or penalty', residuals=actual.tolist(), maximum_relative_residual=float(actual.max()), gauge_relative=gauge.tolist(), harmonic_relative=harmonic.tolist(), refinement_passes=refinements, refinement_full_S_history=history, direct_cotree_solve_seconds=direct_seconds, refinement_cotree_solve_seconds=refinement_seconds, projections=sections, seconds=time.perf_counter() - started, scalar_backend=self.scalar_backend, all_original_edges_restored=True)
        self.records.append(record)
        self.log('complete_tree_inverse', **record)
        return (x[:, 0] if vector else x, record)

    def adjoint(self, rhs, **kw):
        kw.setdefault('label', 'tree_adjoint')
        return self.solve(rhs, **kw)

    def close(self):
        if self.factor is not None:
            self.factor.close()
        if self.scalar_backend == 'pardiso' and self.scalar is not None:
            self.scalar.close()
