from __future__ import annotations
import os, sys, time, json, ast, gc, itertools
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet, Basis, ElementTetN1
PAIRS = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
from scipy.optimize import minimize

def main():
    assert not (OUT / 'mesh_complete.json').exists(), 'Do not repeat a completed repair'
    with np.load(CHART / 'chart_coefficients.npz') as z:
        p = z['p'].copy()
        t = z['t']
        vd = z['vertex_displacements']
        delta = z['edge_corrections']
        edges = z['edges']
    mesh = MeshTet(p, t)
    boundary = np.unique(mesh.facets.T[mesh.boundary_facets()])
    method = json.loads((CHART / 'chart_method.json').read_text())
    bad = np.unique([row[0] for row in method['nonpositive_corners']])
    free = np.setdiff1d(np.unique(t[:, bad]), boundary)
    assert 0 < len(free) <= 8, 'Repair must remain a small patch of interior nodes'
    patch = np.flatnonzero(np.any(np.isin(t.T, free), axis=1))
    local = t.T[patch]
    original = p.T[local]
    sign = np.sign(np.linalg.det((original[:, 1:] - original[:, :1]).transpose(0, 2, 1)))
    diameter = np.max(np.linalg.norm(original[:, :, None] - original[:, None, :], axis=-1), axis=(1, 2))
    scale = diameter ** 3
    td = delta[mesh.t2e.T[patch]]
    gb = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pairs = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
    curve = np.zeros((len(patch), 4, 3, 3))
    for j, (a, b) in enumerate(pairs):
        curve[:, a] += 4 * td[:, j, :, None] * gb[b]
        curve[:, b] += 4 * td[:, j, :, None] * gb[a]
    assignments = np.array(list(itertools.product(range(4), repeat=3)))
    groups = {}
    for i, a in enumerate(assignments):
        groups.setdefault(tuple((list(a).count(k) for k in range(4))), []).append(i)
    groups = list(groups.values())
    slots = [np.argwhere(local == node) for node in free]

    def constraints(q):
        v = original.copy()
        move = q.reshape(-1, 3) * 0.025
        for i, s in enumerate(slots):
            v[s[:, 0], s[:, 1]] += move[i]
        E = (v[:, 1:] - v[:, :1]).transpose(0, 2, 1)
        physical = v + vd[local]
        j = (physical[:, 1:] - physical[:, :1]).transpose(0, 2, 1)[:, None] + curve
        mixed = np.stack(tuple((np.take(j[..., col], assignments[:, col], axis=1) for col in range(3))), axis=3)
        determinants = np.linalg.det(mixed)
        coeff = np.stack([determinants[:, group].mean(1) for group in groups], axis=1) * sign[:, None] / scale[:, None]
        linear = np.linalg.det(E) * sign / scale
        return np.concatenate((linear - 1e-06, (coeff - 1e-06).ravel()))
    initial = constraints(np.zeros(3 * len(free)))
    write('preflight.json', dict(free_vertex_ids=free.tolist(), affected_cells=patch.tolist(), bad_cells=bad.tolist(), initial_constraint_min=float(initial.min()), scope='Interior positions only; all original boundary coordinates/faces and all original physical fit/assessment points fixed', max_position_change=0.005, limit_seconds=600))
    result = minimize(lambda q: float(q @ q), np.zeros(3 * len(free)), jac=lambda q: 2 * q, method='SLSQP', bounds=[(-0.2, 0.2)] * (3 * len(free)), constraints=[dict(type='ineq', fun=constraints)], options=dict(maxiter=400, ftol=1e-12))
    final = constraints(result.x)
    write('optimizer_result.json', dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit), constraint_min=float(final.min()), displacements=(result.x.reshape(-1, 3) * 0.025).tolist(), objective=float(result.fun), seconds=time.perf_counter() - START))
    assert result.success and final.min() >= -1e-10, 'Local repair feasibility not established; no candidate admitted'
    repaired = p.copy()
    repaired[:, free] += result.x.reshape(-1, 3).T * 0.025
    assert np.array_equal(repaired[:, boundary], p[:, boundary])
    v = repaired.T[t.T]
    ov = p.T[t.T]
    det = np.linalg.det((v[:, 1:] - v[:, :1]).transpose(0, 2, 1))
    old = np.linalg.det((ov[:, 1:] - ov[:, :1]).transpose(0, 2, 1))
    assert np.all(det * old > 0), 'A cell outside the patch became invalid'
    assert abs(np.abs(det).sum() - np.abs(old).sum()) / np.abs(old).sum() < 1e-12
    np.savez(OUT / 'mesh_arrays.npz', p=repaired, t=t)
    write('mesh_complete.json', dict(status='local_quality_repair_complete_global_chart_qualification_pending', interior_vertices_moved=free.tolist(), affected_cells=len(patch), boundary_identical=True, all_polyhedral_cell_orientations_preserved=True, all_local_cubic_Bernstein_coefficients_positive=True, original_h=0.025, surface_resolution=80, seconds=time.perf_counter() - START, spectral_solve_started=False))
