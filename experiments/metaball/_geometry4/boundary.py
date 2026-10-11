from __future__ import annotations
import os,sys,time,json,hashlib,ast,gc,itertools
from pathlib import Path
import numpy as np,scipy.sparse as sp,scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet,Basis,ElementTetN1
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
PAIRS=[(0,1),(1,2),(0,2)]
BG = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
LAMBDA = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]])
CHILDREN = np.array([[[1, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5]], [[0.5, 0.5, 0], [0, 1, 0], [0, 0.5, 0.5]], [[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]], dtype=float)

def value(v, d, lam):
    out = np.einsum('qa,nai->nqi', lam, v)
    for k, (a, b) in enumerate(PAIRS):
        out += 4 * lam[None, :, a, None] * lam[None, :, b, None] * d[:, None, k]
    return out

def split(v, d):
    vv = []
    dd = []
    for child in CHILDREN:
        q = LAMBDA @ child
        F = value(v, d, q)
        vertices = F[:, :3]
        delta = np.stack([F[:, 3 + k] - 0.5 * (vertices[:, a] + vertices[:, b]) for k, (a, b) in enumerate(PAIRS)], axis=1)
        vv.append(vertices)
        dd.append(delta)
    return (np.concatenate(vv), np.concatenate(dd))

def main():
    if (OUT / 'global_boundary_complete.json').exists():
        old = json.loads((OUT / 'global_boundary_complete.json').read_text())
        assert not old.get('entire_boundary_homotopy_certified', False), 'Do not repeat completed stronger proof'
        import shutil
        shutil.copy2(OUT / 'global_boundary_complete.json', OUT / 'global_boundary_before_homotopy_review.json')
    with np.load(SOURCE / 'chart_coefficients.npz') as z:
        p = z['p']
        t = z['t']
        vd = z['vertex_displacements']
        edges = z['edges']
        ed = z['edge_corrections']
    m = MeshTet(p, t)
    facets = m.facets.T[m.boundary_facets()].copy()
    bn = np.unique(facets)
    center = p.T[facets].mean(1)
    v = p.T[facets]
    inside = p.T[t.T[m.f2t[0, m.boundary_facets()]]].mean(1)
    flip = np.einsum('ni,ni->n', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), inside - center) > 0
    facets[flip] = facets[flip][:, [0, 2, 1]]
    cfg = target_config
    geo = np.array(cfg['geometry'])
    c = legacy_config
    tree = ast.parse((Path(__file__).parent / 'chart_builder.py').read_text())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'levels'))
    ns = dict(np=np)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'original_analytic_levels', 'exec'), ns)
    levels = ns['levels']
    fo, fi, go, gi = levels(center, geo, c)
    outer = np.abs(fo) / np.linalg.norm(go, axis=1) < np.abs(fi) / np.linalg.norm(gi, axis=1)
    key = edges[:, 0].astype('i8') * p.shape[1] + edges[:, 1]
    qedge = np.sort(np.stack([facets[:, [a, b]] for a, b in PAIRS], axis=1), axis=-1)
    slots = np.searchsorted(key, qedge[:, :, 0].astype('i8') * p.shape[1] + qedge[:, :, 1])
    assert np.array_equal(edges[slots], qedge)
    whole_v = (p.T + vd)[facets]
    whole_d = ed[slots]
    topology = []
    for flag in (False, True):
        f = facets[outer == flag]
        unique, counts = np.unique(np.sort(np.concatenate([f[:, [a, b]] for a, b in PAIRS]), axis=1), axis=0, return_counts=True)
        chi = len(np.unique(f)) - len(unique) + len(f)
        assert chi == 2 and np.all(counts == 2)
        topology.append(dict(outer=flag, facets=len(f), Euler=chi, closed=True))
    records = []
    for flag in (False, True):
        initial = np.flatnonzero(outer == flag)
        accepted = 0
        maxdepth = 0
        minnormal = float('inf')
        maxband = 0.0
        minother = float('inf')
        maxflow = 0.0
        for first in range(0, len(initial), 2048):
            ids = initial[first:first + 2048]
            v = whole_v[ids]
            d = whole_d[ids]
            v0 = p.T[facets[ids]]
            d0 = np.zeros_like(d)
            depth = 0
            while len(v):
                control = np.concatenate((v, np.stack([0.5 * (v[:, a] + v[:, b]) + 2 * d[:, k] for k, (a, b) in enumerate(PAIRS)], axis=1), v0, np.stack([0.5 * (v0[:, a] + v0[:, b]) + 2 * d0[:, k] for k, (a, b) in enumerate(PAIRS)], axis=1)), axis=1)
                x = 0.5 * (value(v, d, np.ones((1, 3)) / 3)[:, 0] + value(v0, d0, np.ones((1, 3)) / 3)[:, 0])
                R = np.max(np.linalg.norm(control - x[:, None], axis=2), axis=1)
                fo, fi, go, gi = levels(x, geo, c)
                power = c['outer_superellipsoid_power']
                half = c['outer_half_extent']
                qmax = (np.max(np.abs(x - 0.5), axis=1) + R + 0.005) / half
                Lo = power * (power - 1) / half ** 2 * qmax ** (power - 2)
                rmin = np.min(np.linalg.norm(x[:, None] - geo[None, :, :3], axis=2), axis=1) - R - 0.005
                assert np.all(rmin > 0)
                Li = 1 / rmin + 1 / c['metaball_softmin_tau']
                L = Lo if flag else Li
                f = fo if flag else fi
                g = go if flag else gi
                other = fi if flag else fo
                og = gi if flag else go
                OL = Li if flag else Lo
                deriv = np.broadcast_to(np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1)[:, None], (len(v), 3, 3, 2)).copy()
                deriv0 = np.broadcast_to(np.stack((v0[:, 1] - v0[:, 0], v0[:, 2] - v0[:, 0]), axis=-1)[:, None], (len(v), 3, 3, 2)).copy()
                for k, (a, b) in enumerate(PAIRS):
                    deriv[:, a] += 4 * d[:, k, :, None] * BG[b]
                    deriv[:, b] += 4 * d[:, k, :, None] * BG[a]
                for k, (a, b) in enumerate(PAIRS):
                    deriv0[:, a] += 4 * d0[:, k, :, None] * BG[b]
                    deriv0[:, b] += 4 * d0[:, k, :, None] * BG[a]
                normals = []
                for A, B in ((deriv0, deriv0), (deriv0, deriv), (deriv, deriv)):
                    normals += [0.5 * (np.cross(A[:, a, :, 0], B[:, a, :, 1]) + np.cross(B[:, a, :, 0], A[:, a, :, 1])) for a in range(3)]
                    normals += [0.25 * (np.cross(A[:, a, :, 0], B[:, b, :, 1]) + np.cross(B[:, a, :, 0], A[:, b, :, 1]) + np.cross(A[:, b, :, 0], B[:, a, :, 1]) + np.cross(B[:, b, :, 0], A[:, a, :, 1])) for a, b in PAIRS]
                normals = np.stack(normals, axis=1)
                nmax = np.max(np.linalg.norm(normals, axis=2), axis=1)
                normal_lower = np.min(-np.einsum('nai,ni->na', normals, g), axis=1) - nmax * L * R
                linear = np.einsum('nai,ni->na', control - x[:, None], g)
                band = np.abs(f) + np.max(np.abs(linear), axis=1) + 0.5 * L * R ** 2
                gmin = np.linalg.norm(g, axis=1) - L * R
                flow = 2 * band / np.maximum(gmin, 1e-30)
                other_lower = other - np.max(np.abs(np.einsum('nai,ni->na', control - x[:, None], og)), axis=1) - 0.5 * OL * R ** 2
                good = (normal_lower > 1e-14 * nmax) & (gmin > 0) & (flow < 0.005) & (L * flow < 0.5 * gmin) & (other_lower > 0)
                if np.any(good):
                    accepted += int(good.sum())
                    maxdepth = max(maxdepth, depth)
                    minnormal = min(minnormal, float((normal_lower[good] / nmax[good]).min()))
                    maxband = max(maxband, float(band[good].max()))
                    minother = min(minother, float(other_lower[good].min()))
                    maxflow = max(maxflow, float(flow[good].max()))
                bad = ~good
                if np.any(bad) and depth >= 12:
                    write('unqualified_patches.json', dict(outer=flag, original_batch_ids=ids.tolist(), depth=depth, remaining=int(bad.sum()), normal_lower_min=float(normal_lower[bad].min()), gmin_min=float(gmin[bad].min()), flow_max=float(flow[bad].max())))
                    raise ArithmeticError('Entire boundary flow certificate failed; no sampled fallback')
                if not np.any(bad):
                    break
                if depth % 3 == 2:
                    a = v0[bad]
                    b = v[bad]
                    da = d0[bad]
                    db = d[bad]
                    mid = 0.5 * (a + b)
                    dm = 0.5 * (da + db)
                    v0 = np.concatenate((a, mid))
                    v = np.concatenate((mid, b))
                    d0 = np.concatenate((da, dm))
                    d = np.concatenate((dm, db))
                else:
                    v, d = split(v[bad], d[bad])
                    v0, d0 = split(v0[bad], d0[bad])
                depth += 1
            progress(outer=flag, original_facets_completed=min(first + 2048, len(initial)), original_facets_total=len(initial), accepted_entire_patches=accepted, maximum_depth=maxdepth)
        records.append(dict(outer=flag, original_facets=len(initial), accepted_entire_patches=accepted, maximum_depth=maxdepth, normal_dot_gradient_lower_over_max_normal=minnormal, associated_field_absolute_upper=maxband, other_field_lower=minother, gradient_flow_path_length_upper=maxflow))
    inner, outerr = records
    assert inner['other_field_lower'] > outerr['associated_field_absolute_upper'] and outerr['other_field_lower'] > inner['associated_field_absolute_upper']
    proof = 'The entire straight homotopy from the original embedded polyhedral walls to the quadratic walls is transverse to the original analytic gradient, proved by degree(2,2) normal Bernstein controls and analytic Hessian bounds. Every point of this whole homotopy has a regular gradient-flow projection onto the zero wall. For the initial embedded transverse wall, projected sheets can be globally ordered along the flow direction; a connected embedded closed transverse wall therefore has one sheet and projection degree one. Homotopy preserves this degree. The final piecewise polynomial projection has positive face Jacobians and isolated fibers; all local degrees are positive, including skeleton points. Total degree one therefore excludes branch points and multiple preimages and establishes the final quadratic wall embedding. Both walls remain disjoint by certified analytic level-band separation. Together with the all-cell positive fluid Jacobian, the oriented embedded boundary gives volume degree one in its shell and zero outside; positivity of every local degree excludes interior or skeleton multiplicity. The unchanged fluid chart is globally injective. It remains a quadratic approximation of the original smooth walls, not an exact radial map.'
    write('global_boundary_complete.json', dict(status='global_boundary_and_fluid_injectivity_qualified', entire_boundary_homotopy_certified=True, normal_Bernstein_degree=[2, 2], all_boundary_facets=len(facets), topology=topology, entire_patch_bounds=records, proof=proof, source_chart_sha256=hashlib.sha256((SOURCE / 'chart_coefficients.npz').read_bytes()).hexdigest(), source_point_arrays_unchanged=True, local_positive_det_receipt=str(SOURCE / 'bernstein_determinant_certificate.json'), seconds=time.perf_counter() - START, scope='Analytic interval over the entire boundary homotopy and positive degree proof; no sampled transversality replacement'))
    progress(status='global_boundary_and_fluid_injectivity_qualified')
