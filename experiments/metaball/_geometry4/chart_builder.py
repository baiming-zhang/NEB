from __future__ import annotations
import os, sys, time, json, ast, gc, itertools, traceback
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet, Basis, ElementTetN1
PAIRS = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
GBAR = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

def levels(x, geo, c):
    q = (x - 0.5) / c['outer_half_extent']
    p = c['outer_superellipsoid_power']
    fo = 1 - np.sum(q ** p, axis=-1)
    go = -p / c['outer_half_extent'] * q ** (p - 1)
    d = x[..., None, :] - geo[:, :3]
    r = np.linalg.norm(d, axis=-1)
    distance = r - geo[:, 3]
    low = distance.min(-1)
    w = np.exp(-(distance - low[..., None]) / c['metaball_softmin_tau'])
    total = w.sum(-1)
    fi = low - c['metaball_softmin_tau'] * np.log(total)
    w /= total[..., None]
    gi = np.sum(w[..., None] * d / r[..., None], axis=-2)
    return (fo, fi, go, gi)

def main():
    guard()
    try:
        cfg = target_config
        geo = np.array(cfg['geometry'])
        c = legacy_config
        with np.load(MESH) as z:
            mesh = MeshTet(z['p'], z['t'])
        vertices = mesh.p.T[mesh.t.T]
        edges = mesh.edges.T
        mid = mesh.p.T[edges].mean(1)
        delta = np.zeros_like(mid)
        facets = mesh.facets.T[mesh.boundary_facets()]
        centers = mesh.p.T[facets].mean(1)
        fo, fi, go, gi = levels(centers, geo, c)
        outer = np.abs(fo) / np.linalg.norm(go, axis=1) < np.abs(fi) / np.linalg.norm(gi, axis=1)
        edge_key = edges[:, 0].astype('i8') * len(mesh.p.T) + edges[:, 1]
        flags = np.full(len(edges), -1, dtype='i1')
        for face, flag in zip(facets, outer):
            pairs = np.sort(np.array([face[[0, 1]], face[[1, 2]], face[[0, 2]]]), axis=1)
            key = pairs[:, 0].astype('i8') * len(mesh.p.T) + pairs[:, 1]
            ids = np.searchsorted(edge_key, key)
            assert np.array_equal(edge_key[ids], key)
            assert np.all((flags[ids] == -1) | (flags[ids] == int(flag)))
            flags[ids] = int(flag)
        for flag in (0, 1):
            ids = np.flatnonzero(flags == flag)
            x = mid[ids].copy()
            for _ in range(12):
                fo, fi, go, gi = levels(x, geo, c)
                f = fo if flag else fi
                g = go if flag else gi
                x -= f[:, None] * g / np.sum(g * g, axis=1)[:, None]
            fo, fi, go, gi = levels(x, geo, c)
            assert np.max(np.abs(fo if flag else fi)) < 1e-11
            delta[ids] = x - mid[ids]
        vd = np.zeros_like(mesh.p.T)
        if PROJECT_VERTICES:
            vertex_flags = np.full(mesh.p.shape[1], -1, dtype='i1')
            for flag in (0, 1):
                ids = np.unique(facets[outer == bool(flag)])
                assert np.all(vertex_flags[ids] == -1)
                vertex_flags[ids] = flag
                x = mesh.p.T[ids].copy()
                for _ in range(12):
                    fo, fi, go, gi = levels(x, geo, c)
                    f = fo if flag else fi
                    g = go if flag else gi
                    x -= f[:, None] * g / np.sum(g * g, axis=1)[:, None]
                fo, fi, go, gi = levels(x, geo, c)
                assert np.max(np.abs(fo if flag else fi)) < 1e-11
                vd[ids] = x - mesh.p.T[ids]
            delta -= vd[edges].mean(1)
        if EXTEND_INTERIOR:
            assert PROJECT_VERTICES
            from scipy.spatial import cKDTree
            boundary_vertices = np.flatnonzero(vertex_flags >= 0)
            boundary_edges = np.flatnonzero(flags >= 0)
            bp = np.concatenate((mesh.p.T[boundary_vertices], mid[boundary_edges]))
            bv = np.concatenate((vd[boundary_vertices], delta[boundary_edges] + vd[edges[boundary_edges]].mean(1)))
            tree = cKDTree(bp)

            def extend(points):
                values = np.empty_like(points)
                for off in range(0, len(points), 8192):
                    x = points[off:off + 8192]
                    dist, idx = tree.query(x, k=16, workers=2)
                    weights = 1 / np.maximum(dist, 1e-10) ** 2
                    weights /= weights.sum(1)[:, None]
                    values[off:off + len(x)] = np.sum(weights[:, :, None] * bv[idx], axis=1) * np.exp(-(dist[:, 0] / 0.025) ** 2)[:, None]
                return values
            ids = np.flatnonzero(vertex_flags < 0)
            vd[ids] = extend(mesh.p.T[ids])
            ids = np.flatnonzero(flags < 0)
            physical_mid = extend(mid[ids])
            delta[boundary_edges] = bv[len(boundary_vertices):] - vd[edges[boundary_edges]].mean(1)
            delta[ids] = physical_mid - vd[edges[ids]].mean(1)
        if STRAIGHT_INTERIOR_EDGES:
            assert PROJECT_VERTICES
            delta[flags < 0] = 0
        E = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
        inv = np.linalg.inv(E)
        grad = np.concatenate((-inv.sum(1)[:, None, :], inv), axis=1)
        td = delta[mesh.t2e.T]
        vc = vd[mesh.t.T]
        linear_jac = np.einsum('nai,naj->nij', vc, grad)
        for i, (a, b) in enumerate(PAIRS):
            assert np.array_equal(np.sort(mesh.t.T[:, [a, b]], axis=1), edges[mesh.t2e.T[:, i]])
        corners = np.zeros((len(vertices), 4, 3, 3))
        corners += linear_jac[:, None]
        for j, (a, b) in enumerate(PAIRS):
            corners[:, a] += 4 * td[:, j, :, None] * grad[:, b, None, :]
            corners[:, b] += 4 * td[:, j, :, None] * grad[:, a, None, :]
        max_norm = 0.0
        min_det = 1.0
        negative_corners = 0
        worst_cell = -1
        worst_corner = -1
        bad_corners = []
        for first in range(0, len(vertices), 2048):
            a = corners[first:first + 2048]
            max_norm = max(max_norm, float(np.linalg.svd(a, compute_uv=False)[..., 0].max()))
            det = np.linalg.det(np.eye(3) + a)
            negative_corners += int(np.count_nonzero(det <= 0))
            bad_corners.extend([[first + int(i), int(j)] for i, j in np.argwhere(det <= 0)])
            if float(det.min()) < min_det:
                min_det = float(det.min())
                local = np.unravel_index(det.argmin(), det.shape)
                worst_cell = first + int(local[0])
                worst_corner = int(local[1])
        norms = np.linalg.norm(delta, axis=1)
        physical_mid_norms = np.linalg.norm(delta + vd[edges].mean(1), axis=1)
        write('jacobian_diagnostics.json', dict(tetrahedra=len(vertices), max_J_minus_identity_norm=max_norm, min_corner_det=min_det, nonpositive_corner_count=negative_corners, worst_cell=worst_cell, worst_corner=worst_corner, worst_cell_vertices=vertices[worst_cell].tolist(), worst_cell_edge_displacements=td[worst_cell].tolist(), max_boundary_midpoint_displacement=float(norms.max()), boundary_displacement_quantiles=np.quantile(norms[flags >= 0], [0, 0.5, 0.9, 0.99, 1]).tolist(), source_mesh=str(MESH), seconds=time.perf_counter() - START, qualification_passed=max_norm < 1))
        np.savez(OUT / 'chart_coefficients.npz', p=mesh.p, t=mesh.t, vertex_displacements=vd, edge_corrections=delta, edges=edges)
        write('chart_method.json', dict(project_boundary_vertices=PROJECT_VERTICES, extend_interior_nodes=EXTEND_INTERIOR, straight_interior_edges=STRAIGHT_INTERIOR_EDGES, nonpositive_corners=bad_corners, worst_cell_vertex_ids=mesh.t.T[worst_cell].tolist(), worst_cell_boundary_vertices=np.isin(mesh.t.T[worst_cell], np.unique(facets)).tolist(), worst_cell_polyhedral_volume=abs(float(np.linalg.det(E[worst_cell]))) / 6, max_physical_midpoint_displacement=float(physical_mid_norms.max()), max_edge_correction=float(norms.max()), spectral_solve_started=False))
        if PRE_REPAIR:
            return
        det_lower = (1 - max_norm) ** 3 if max_norm < 1 else None
        if det_lower is None and min_det > 0:
            import itertools
            groups = {}
            for assignment in itertools.product(range(4), repeat=3):
                key = tuple((assignment.count(i) for i in range(4)))
                groups.setdefault(key, []).append(assignment)
            det_lower = float('inf')
            worst_bernstein = None
            for key, assignments in groups.items():
                for first in range(0, len(vertices), 2048):
                    j = np.eye(3) + corners[first:first + 2048]
                    coefficient = np.zeros(len(j))
                    for a, b, c0 in assignments:
                        coefficient += np.linalg.det(np.stack((j[:, a, :, 0], j[:, b, :, 1], j[:, c0, :, 2]), axis=2))
                    coefficient /= len(assignments)
                    if float(coefficient.min()) < det_lower:
                        det_lower = float(coefficient.min())
                        worst_bernstein = dict(cell=first + int(coefficient.argmin()), multi_index=key)
            write('bernstein_determinant_certificate.json', dict(certified=det_lower > 1e-10, coefficient_min=det_lower, worst=worst_bernstein, degree=3, all_cells=len(vertices), coefficients_per_cell=20))
        assert det_lower is not None and det_lower > 1e-10, 'Quadratic curved-cell all-cell Jacobian positivity certificate fails'
        displacement_bound = np.linalg.norm(vc, axis=2).max(1) + 1.5 * np.linalg.norm(td, axis=2).max(1)
        progress('continuous_quadratic_chart_constructed', tetrahedra=len(vertices), max_boundary_midpoint_displacement=float(np.linalg.norm(delta + vd[edges].mean(1), axis=1).max()), max_J_minus_identity_norm=max_norm, all_cell_det_lower_bound=det_lower)
        beta = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
        faceverts = mesh.p.T[facets]
        faceids = []
        for face in facets:
            pairs = np.sort(np.array([face[[0, 1]], face[[1, 2]], face[[0, 2]]]), axis=1)
            key = pairs[:, 0].astype('i8') * len(mesh.p.T) + pairs[:, 1]
            faceids.append(np.searchsorted(edge_key, key))
        fd = delta[np.array(faceids)]
        max_distance = 0.0
        for b in beta:
            x = np.einsum('i,nij->nj', b, faceverts + vd[facets]) + 4 * (fd[:, 0] * b[0] * b[1] + fd[:, 1] * b[1] * b[2] + fd[:, 2] * b[0] * b[2])
            fo, fi, go, gi = levels(x, geo, c)
            distance = np.where(outer, np.abs(fo) / np.linalg.norm(go, axis=1), np.abs(fi) / np.linalg.norm(gi, axis=1))
            max_distance = max(max_distance, float(distance.max()))
        nbin = 32
        lo = np.clip(np.floor((vertices.min(1) - displacement_bound[:, None] - 1e-10) * nbin).astype('i4'), 0, nbin - 1)
        hi = np.clip(np.floor((vertices.max(1) + displacement_bound[:, None] + 1e-10) * nbin).astype('i4'), 0, nbin - 1)
        dims = hi - lo + 1
        counts = np.prod(dims, 1)
        ends = np.cumsum(counts)
        cells = np.repeat(np.arange(len(vertices)), counts)
        off = np.arange(int(ends[-1])) - np.repeat(ends - counts, counts)
        dd = dims[cells]
        xyz = lo[cells] + np.stack((off % dd[:, 0], off // dd[:, 0] % dd[:, 1], off // (dd[:, 0] * dd[:, 1])), 1)
        bins = xyz[:, 0] + nbin * xyz[:, 1] + nbin ** 2 * xyz[:, 2]
        order = np.argsort(bins)
        bins = bins[order]
        cells = cells[order]
        cnt = np.bincount(bins, minlength=nbin ** 3)
        starts = np.cumsum(cnt) - cnt
        slot = np.arange(len(cells)) - starts[bins]
        table = np.full((nbin ** 3, int(cnt.max())), -1, dtype='i4')
        table[bins, slot] = cells
        summary = {}
        original = Path(cfg['source_samples'])
        samples = np.load(original)
        for label in ('fit_points', 'assessment_points'):
            points = samples[label]
            result = {key: [] for key in ('points', 'J', 'det', 'cell_ids')}
            max_residual = 0.0
            missing = []
            for first in range(0, len(points), 32):
                q = points[first:first + 32]
                bb = np.clip(np.floor(q * nbin).astype('i4'), 0, nbin - 1)
                ids = table[bb[:, 0] + nbin * bb[:, 1] + nbin ** 2 * bb[:, 2]]
                safe = np.maximum(ids, 0)
                vv = vertices[safe]
                iv = inv[safe]
                dd = td[safe]
                bar = np.einsum('ncij,ncj->nci', iv, q[:, None] - vv[:, :, 0])
                for _ in range(8):
                    lam = np.concatenate((1 - bar.sum(2)[..., None], bar), axis=2)
                    d = np.einsum('nca,ncai->nci', lam, vc[safe])
                    jac = np.broadcast_to(np.eye(3), (*bar.shape[:2], 3, 3)).copy() + linear_jac[safe]
                    for j, (a, b) in enumerate(PAIRS):
                        d += 4 * dd[:, :, j] * lam[:, :, a, None] * lam[:, :, b, None]
                        gg = lam[:, :, a, None] * grad[safe, b] + lam[:, :, b, None] * grad[safe, a]
                        jac += 4 * dd[:, :, j, :, None] * gg[:, :, None, :]
                    x = vv[:, :, 0] + np.einsum('ncij,ncj->nci', E[safe], bar) + d
                    res = x - q[:, None]
                    det = np.linalg.det(jac)
                    good = np.isfinite(det) & (np.abs(det) > 1e-08)
                    jac = np.where(good[..., None, None], jac, np.eye(3))
                    dx = np.linalg.solve(jac, res[..., None])[..., 0]
                    bar -= np.einsum('ncij,ncj->nci', iv, dx)
                residual = np.linalg.norm(res, axis=2)
                inside = (ids >= 0) & np.all(bar >= -1e-10, axis=2) & (bar.sum(2) <= 1 + 1e-10) & (residual < 1e-10) & (det > 0)
                valid = inside.any(1)
                missing.extend((first + np.flatnonzero(~valid)).tolist())
                pick = inside.argmax(1)
                row = np.arange(len(q))
                chosen = ids[row, pick]
                max_residual = max(max_residual, float(residual[row[valid], pick[valid]].max()) if valid.any() else 0.0)
                poly = vv[row, pick, 0] + np.einsum('nij,nj->ni', E[safe[row, pick]], bar[row, pick])
                for key, value in [('points', poly), ('J', jac[row, pick]), ('det', det[row, pick]), ('cell_ids', chosen)]:
                    result[key].append(value)
                if first % 2048 == 0:
                    progress('inverse_curved_cells', set=label, points=first + len(q), total=len(points), missing=len(missing))
            summary[label] = dict(points=len(points), located=len(points) - len(missing), missing_indices=missing, inverse_residual_max=max_residual)
            np.savez(OUT / f'{label}_mapped.npz', **{k: np.concatenate(v) for k, v in result.items()})
        qualified = all((r['located'] == r['points'] for r in summary.values()))
        record = dict(status='qualified_for_review' if qualified else 'not_qualified', all_points_located=qualified, all_cell_local_det_positive=True, max_J_minus_identity_norm=max_norm, det_lower_bound=det_lower, boundary_geometric_distance_max=max_distance, boundary_test_points=len(facets) * len(beta), source_mesh=str(MESH), source_samples=str(original), sets=summary, seconds=time.perf_counter() - START, method='Continuous quadratic isoparametric chart using original target mesh and implicit-wall node projection; approximate smooth geometry, no star-shapedness assumption', project_boundary_vertices=PROJECT_VERTICES, extend_interior_nodes=EXTEND_INTERIOR, global_boundary_injectivity_pending=True, spectral_solve_started=False)
        write('qualification.json', record)
        progress('qualification_complete', all_points_located=qualified, boundary_distance_max=max_distance)
    except BaseException:
        write(f'failure_{time.time_ns()}.json', dict(error=traceback.format_exc()))
        raise
    finally:
        DONE.set()
