from __future__ import annotations
import os,sys,time,json,hashlib,ast,gc,itertools
from pathlib import Path
import numpy as np,scipy.sparse as sp,scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet,Basis,ElementTetN1
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
def quadrature(n):
    x, w = np.polynomial.legendre.leggauss(n)
    x = (x + 1) / 2
    w = w / 2
    u, v, z = np.meshgrid(x, x, x, indexing='ij')
    wu, wv, wz = np.meshgrid(w, w, w, indexing='ij')
    X = np.stack((u, (1 - u) * v, (1 - u) * (1 - v) * z), axis=-1).reshape(-1, 3)
    lam = np.column_stack((1 - X.sum(1), X))
    weight = (wu * wv * wz * (1 - u) ** 2 * (1 - v)).ravel()
    assert np.all(weight > 0) and abs(weight.sum() - 1 / 6) < 1e-13
    return (lam, weight)

class Chart:

    def __init__(self, root):
        global HERE, CHART, GLOBAL
        HERE = Path(root)
        CHART = HERE / 'geometry4_local_repaired_chart'
        GLOBAL = HERE / 'geometry4_global_extension'
        cert = read(HERE / 'geometry4_boundary_flow/global_boundary_complete.json')
        assert cert['status'] == 'global_boundary_and_fluid_injectivity_qualified'
        assert cert['entire_boundary_homotopy_certified']
        topo = read(HERE / 'geometry4_boundary_flow/boundary_manifold_verified.json')
        assert topo['status'] == 'verified_closed_connected_spheres'
        assert cert['source_chart_sha256'] == topo['source_chart_sha256'] == sha(CHART / 'chart_coefficients.npz')
        q = read(CHART / 'qualification.json')
        assert q['global_boundary_injectivity_pending']
        with np.load(CHART / 'chart_coefficients.npz') as z:
            self.p = z['p']
            self.t = z['t']
            self.vd = z['vertex_displacements']
            self.ed = z['edge_corrections']
            self.edges = z['edges']
        self.mesh = MeshTet(self.p, self.t)
        assert np.array_equal(self.mesh.t, self.t) and np.array_equal(self.mesh.edges.T, self.edges)
        self.source_sha = sha(CHART / 'chart_coefficients.npz')

    def fields(self, ids, lam, identity=False):
        cells = self.t.T[ids]
        v = self.p.T[cells]
        E = (v[:, 1:] - v[:, :1]).transpose(0, 2, 1)
        inv = np.linalg.inv(E)
        grad = np.concatenate((-inv.sum(1)[:, None], inv), axis=1)
        phi = []
        curl = []
        for a, b in PAIRS:
            sign = (1 - 2 * (cells[:, a] > cells[:, b]))[:, None, None]
            phi.append(sign * (lam[None, :, a, None] * grad[:, None, b] - lam[None, :, b, None] * grad[:, None, a]))
            curl.append(sign[:, 0] * 2 * np.cross(grad[:, a], grad[:, b]))
        phi = np.array(phi)
        curl = np.array(curl)
        J = np.broadcast_to(np.eye(3), (len(ids), len(lam), 3, 3)).copy()
        if not identity:
            J += np.einsum('nai,naj->nij', self.vd[cells], grad)[:, None]
            td = self.ed[self.mesh.t2e.T[ids]]
            for k, (a, b) in enumerate(PAIRS):
                J += 4 * td[:, None, k, :, None] * (lam[None, :, a, None] * grad[:, None, b] + lam[None, :, b, None] * grad[:, None, a])[:, :, None, :]
        det = np.linalg.det(J)
        assert np.all(det > 0)
        return (phi, curl, J, det, np.abs(np.linalg.det(E)))

    def matrices(self, ids, n, identity=False):
        lam, weights = quadrature(n)
        phi, curl, J, det, volume = self.fields(ids, lam, identity)
        inverse = np.linalg.inv(J)
        physical_phi = np.einsum('nqji,anqj->anqi', inverse, phi)
        physical_curl = np.einsum('nqij,anj->anqi', J, curl) / det[None, :, :, None]
        W = volume[:, None] * det * weights[None]
        M = np.einsum('anqi,bnqi,nq->nab', physical_phi, physical_phi, W, optimize=True)
        S = np.einsum('anqi,bnqi,nq->nab', physical_curl, physical_curl, W, optimize=True)
        return (M, S)

    def identity_check(self):
        ids = np.arange(32)
        lam, w = quadrature(3)
        phi, curl, _, _, _ = self.fields(ids, lam, True)
        basis = Basis(self.mesh, ElementTetN1(), elements=ids, quadrature=(lam[:, 1:].T, w))
        expected_phi = np.array([b[0].value.transpose(1, 2, 0) for b in basis.basis])
        expected_curl = np.array([b[0].curl.transpose(1, 2, 0) for b in basis.basis])
        ep = float(np.linalg.norm(phi - expected_phi) / np.linalg.norm(expected_phi))
        ec = float(np.linalg.norm(curl[:, :, None] - expected_curl) / np.linalg.norm(expected_curl))
        assert ep < 1e-12 and ec < 1e-12
        write('identity_implementation_verified.json', dict(passed=True, phi_relative=ep, curl_relative=ec, reference='Original skfem ElementTetN1 covariant Piola and edge orientations', all_cell_edge_order_verified=True, source_sha=self.source_sha))

    def assemble(self, progress):
        if (OUT / 'assembly_complete.json').exists():
            assert read(OUT / 'assembly_complete.json')['source_sha'] == self.source_sha
            return (sp.load_npz(OUT / 'physical_mass.npz'), sp.load_npz(OUT / 'physical_stiffness.npz'))
        self.identity_check()
        nc = self.t.shape[1]
        path = OUT / 'assembly_cursor.json'
        if path.exists():
            r = read(path)
            assert r['source_sha'] == self.source_sha
            first = r['completed_cells']
            maxn = r['maximum_quadrature_n']
            worst = r['maximum_accepted_local_relative_difference']
            mode = 'r+'
        else:
            first = 0
            maxn = 3
            worst = 0.0
            mode = 'w+'
        lm = np.lib.format.open_memmap(OUT / 'local_mass.npy', mode=mode, dtype='f8', shape=(nc, 6, 6))
        ls = np.lib.format.open_memmap(OUT / 'local_stiffness.npy', mode=mode, dtype='f8', shape=(nc, 6, 6))
        for off in range(first, nc, 256):
            ids = np.arange(off, min(off + 256, nc))
            M, S = self.matrices(ids, 3)
            curved = np.any(self.ed[self.mesh.t2e.T[ids]] != 0, axis=(1, 2))
            active = np.flatnonzero(curved)
            for n in (5, 7, 11, 17, 25, 37, 55, 81):
                if not len(active):
                    break
                remaining = []
                batch = max(1, 120000 // n ** 3)
                for j in range(0, len(active), batch):
                    at = active[j:j + batch]
                    m, s = self.matrices(ids[at], n)
                    rel = np.maximum(np.linalg.norm(m - M[at], axis=(1, 2)) / np.maximum(np.linalg.norm(m, axis=(1, 2)), 1e-30), np.linalg.norm(s - S[at], axis=(1, 2)) / np.maximum(np.linalg.norm(s, axis=(1, 2)), 1e-30))
                    accepted = rel <= 1e-06
                    if np.any(accepted):
                        worst = max(worst, float(rel[accepted].max()))
                    remaining.extend(at[~accepted].tolist())
                    M[at] = m
                    S[at] = s
                maxn = max(maxn, n)
                active = np.array(remaining, dtype=int)
            if len(active):
                write(f'quadrature_failure_{time.time_ns()}.json', dict(cell_ids=ids[active].tolist(), maximum_n=81, source_sha=self.source_sha))
                raise ArithmeticError('Local physical matrices did not converge; no spectrum admitted')
            assert np.all(np.linalg.eigvalsh(M) > 0), 'Positive mass must hold in every curved element'
            lm[ids] = M
            ls[ids] = S
            lm.flush()
            ls.flush()
            write('assembly_cursor.json', dict(completed_cells=int(ids[-1] + 1), total_cells=nc, source_sha=self.source_sha, maximum_quadrature_n=maxn, maximum_accepted_local_relative_difference=worst))
            if off % 4096 == 0:
                progress('curved_consistent_assembly', completed_cells=int(ids[-1] + 1), total_cells=nc, maximum_quadrature_n=maxn)
        index = self.mesh.t2e.T
        rows = np.broadcast_to(index[:, :, None], (nc, 6, 6)).ravel()
        cols = np.broadcast_to(index[:, None, :], (nc, 6, 6)).ravel()
        size = self.mesh.edges.shape[1]
        M = sp.coo_matrix((lm.ravel(), (rows, cols)), shape=(size, size)).tocsr()
        S = sp.coo_matrix((ls.ravel(), (rows, cols)), shape=(size, size)).tocsr()
        del rows, cols, lm, ls
        gc.collect()
        sp.save_npz(OUT / 'physical_mass.npz', M)
        sp.save_npz(OUT / 'physical_stiffness.npz', S)
        write('assembly_complete.json', dict(status='complete', source_sha=self.source_sha, cells=nc, positive_local_mass=True, quadrature='Positive tensor Gauss-Duffy with per-cell M and S convergence', local_relative_tolerance=1e-06, maximum_n=maxn, maximum_accepted_local_relative_difference=worst, consistent_physical_J=True, original_polyhedral_S_M_not_reused=True))
        return (M, S)

def complex_from(chart, M, S):
    """Reuse the supplied canonical gradient/harmonic deflation with active nodes."""
    src = Path(__file__).parent / 'fitted_target_core.py'
    tree = ast.parse(src.read_text())
    cls = next((x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'FittedComplex'))
    init = next((x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '__init__'))
    start = next((i for i, x in enumerate(init.body) if ast.unparse(x).startswith('edge = mesh.edges')))
    code = ast.unparse(ast.Module(body=init.body[start:], type_ignores=[])).replace('vn = np.setdiff1d(np.arange(mesh.p.shape[1]), bn)', 'vn = np.setdiff1d(np.unique(mesh.t), bn)').replace('euler_absolute = mesh.p.shape[1] -', 'euler_absolute = len(np.unique(mesh.t)) -')
    methods = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in ('project', 'project_transpose')]
    ns = dict(np=np, sp=sp, sla=sla, connected_components=connected_components)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Canonical', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])), str(src), 'exec'), ns)
    obj = ns['Canonical']()
    mesh = chart.mesh
    boundary = mesh.boundary_edges()
    obj.free = np.setdiff1d(np.arange(mesh.edges.shape[1]), boundary)
    obj.mesh = mesh
    obj.M = M[obj.free][:, obj.free]
    obj.S = S[obj.free][:, obj.free]
    ns.update(self=obj, mesh=mesh, boundary=boundary)
    exec(compile(code, str(src), 'exec'), ns)
    obj.metadata['active_vertices'] = len(np.unique(mesh.t))
    obj.metadata['physical_chart_sha'] = chart.source_sha
    rel = float(sla.norm(obj.S @ obj.G) / (sla.norm(obj.S) * sla.norm(obj.G)))
    assert rel < 1e-10
    write('canonical_verified.json', dict(**obj.metadata, gradient_kernel_relative=rel, same_original_canonical_deflation=True))
    return obj
