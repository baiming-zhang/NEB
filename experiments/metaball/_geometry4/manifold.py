from __future__ import annotations
import os, sys, time, json, ast, gc, itertools
from pathlib import Path
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet, Basis, ElementTetN1
PAIRS = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

def main():
    """Independent connectedness and complete vertex-link checks for both walls."""
    with np.load(CHART / 'chart_coefficients.npz') as z:
        p = z['p']
        t = z['t']
    m = MeshTet(p, t)
    faces = m.facets.T[m.boundary_facets()]
    nf = len(faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1)
    keys = edges[:, 0].astype('i8') * p.shape[1] + edges[:, 1]
    order = np.argsort(keys)
    assert np.all(keys[order[::2]] == keys[order[1::2]])
    unique, counts = np.unique(keys, return_counts=True)
    assert np.all(counts == 2)
    fidx = np.tile(np.arange(nf), 3)
    a = fidx[order[::2]]
    b = fidx[order[1::2]]
    paired = edges[order[::2]]
    graph = sp.csr_matrix((np.ones(len(a) * 2), (np.r_[a, b], np.r_[b, a])), shape=(nf, nf))
    ncomp, labels = connected_components(graph)
    assert ncomp == 2
    parent = np.arange(3 * nf)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for axis in (0, 1):
        vertex = paired[:, axis]
        ca = 3 * a + np.argmax(faces[a] == vertex[:, None], axis=1)
        cb = 3 * b + np.argmax(faces[b] == vertex[:, None], axis=1)
        for i, j in zip(ca, cb):
            parent[find(i)] = find(j)
    roots = np.fromiter((find(i) for i in range(3 * nf)), dtype='i8')
    links = np.unique(np.column_stack((faces.ravel(), roots)), axis=0)
    assert len(links) == len(np.unique(faces)), 'Each vertex link must be one connected cycle'
    components = []
    for i in range(ncomp):
        f = faces[labels == i]
        e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1), axis=0)
        chi = len(np.unique(f)) - len(e) + len(f)
        assert chi == 2
        components.append(dict(vertices=len(np.unique(f)), edges=len(e), faces=len(f), Euler=chi, connected=True))
    r = dict(status='verified_closed_connected_spheres', components=components, every_edge_incidence=2, every_vertex_link_single_cycle=True)
    path = HERE / 'geometry4_boundary_flow/boundary_manifold_verified.json'
    path.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
