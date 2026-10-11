







from __future__ import annotations

import importlib.util
import json
import sys
from _neb_paths import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CASE = WORKSPACE / "3d_metaball_reproduction_latest/cases/3D_Metaball"
THEORY = WORKSPACE / "ICLR2027_Potential_Eigenfluids/ICLR2027_Potential_Eigenfluids.tex"
CONFIG = CASE / "configs/k2048_three_sphere_connected_siren_freeslip_2000_w128_d5.json"
GT = CASE / "results/independent_freeslip_dns_gt_128"


def legacy():
    name = "gauge_legacy_geometry"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, CASE / "scripts/train_validate.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def geometry_mask(n, token=12000001):
    old = legacy()
    cfg = old.load_config(CONFIG, {})
    geom = old.geometry_from_token(cfg, token)
    xyz = (np.indices((n, n, n)).reshape(3, -1).T + 0.5) / n
    return old.fluid_mask(cfg, xyz, geom).reshape(n, n, n), geom


def plus(p, axis, amount=1):
    q = list(p)
    q[axis] += amount
    return tuple(q)


def face_edges(face):
    a, x, y, z = face
    p = (x, y, z)
    b, c = (a + 1) % 3, (a + 2) % 3
    return [((b, *p), 1), ((c, *plus(p, b)), 1), ((b, *plus(p, c)), -1), ((c, *p), -1)]


class Complex:
    def __init__(self, mask):
        self.mask = np.asarray(mask, bool)
        self.n = mask.shape[0]
        self.h = 1 / self.n
        self.cells = np.argwhere(mask)
        face_count = {}
        for p in map(tuple, self.cells):
            for axis in range(3):
                for q in (p, plus(p, axis)):
                    f = (axis, *q)
                    face_count[f] = face_count.get(f, 0) + 1
        all_faces = sorted(face_count)
        boundary_faces = [f for f in all_faces if face_count[f] == 1]
        boundary_edges = {e for f in boundary_faces for e, _ in face_edges(f)}
        all_edges = sorted({e for f in all_faces for e, _ in face_edges(f)})
        all_vertices = sorted(
            {v for a, *p in all_edges for v in (tuple(p), plus(p, a))}
        )
        boundary_vertices = {
            v for a, *p in boundary_edges for v in (tuple(p), plus(p, a))
        }
        self.edges = np.array([e for e in all_edges if e not in boundary_edges], int)
        self.faces = np.array([f for f in all_faces if face_count[f] == 2], int)
        self.vertices = np.array(
            [v for v in all_vertices if v not in boundary_vertices], int
        )
        emap = {tuple(e): i for i, e in enumerate(self.edges)}
        fmap = {tuple(f): i for i, f in enumerate(self.faces)}
        avmap = {v: i for i, v in enumerate(all_vertices)}
        rows, cols, vals = [], [], []
        for i, (a, *p) in enumerate(self.edges):
            for v, sign in ((tuple(p), -1), (plus(p, a), 1)):
                rows.append(i)
                cols.append(avmap[v])
                vals.append(sign / self.h)
        g_all = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.edges), len(all_vertices))
        )
        self.G = g_all[:, [avmap[tuple(v)] for v in self.vertices]].tocsr()
        rows, cols, vals = [], [], []
        for i, f in enumerate(self.faces):
            for e, sign in face_edges(f):
                if e in emap:
                    rows.append(i)
                    cols.append(emap[e])
                    vals.append(sign / self.h)
        self.C = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.faces), len(self.edges))
        )
        rows, cols, vals = [], [], []
        for i, p in enumerate(map(tuple, self.cells)):
            for a in range(3):
                for q, sign in ((p, -1), (plus(p, a), 1)):
                    if (a, *q) in fmap:
                        rows.append(i)
                        cols.append(fmap[(a, *q)])
                        vals.append(sign / self.h)
        self.D = sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(self.cells), len(self.faces))
        )
        self.poisson = sla.splu((self.G.T @ self.G).tocsc())


        bv = sorted(boundary_vertices)
        bvmap = {v: i for i, v in enumerate(bv)}
        br, bc = [], []
        for a, *p in boundary_edges:
            left, right = bvmap[tuple(p)], bvmap[plus(p, a)]
            br.extend((left, right))
            bc.extend((right, left))
        graph = sp.csr_matrix((np.ones(len(br)), (br, bc)), shape=(len(bv), len(bv)))
        components, labels = connected_components(graph)
        outer = labels[np.argmin(np.asarray(bv).sum(axis=1))]
        harmonics = []
        for comp in range(components):
            if comp == outer:
                continue
            boundary_values = np.zeros(len(all_vertices))
            for v, label in zip(bv, labels):
                if label == comp:
                    boundary_values[avmap[v]] = 1
            g = g_all @ boundary_values
            g -= self.G @ self.poisson.solve(self.G.T @ g)
            harmonics.append(g)
        self.H = (
            np.linalg.qr(np.column_stack(harmonics))[0]
            if harmonics
            else np.zeros((len(self.edges), 0))
        )
        self.boundary_components = int(components)
        self.canonical_dimension = (
            len(self.edges) - len(self.vertices) - self.H.shape[1]
        )

        fluid_components = connected_components((self.D @ self.D.T).astype(bool))[0]
        self.absolute_harmonic_dimension = (
            len(self.faces)
            - self.canonical_dimension
            - len(self.cells)
            + fluid_components
        )
        if fluid_components != 1 or self.absolute_harmonic_dimension != 0:
            raise ValueError(
                f"Pilot requires connected, tunnel-free fluid: components={fluid_components}, b1={self.absolute_harmonic_dimension}"
            )
        self.edge_points = (
            self.edges[:, 1:].astype(float) + 0.5 * np.eye(3)[self.edges[:, 0]]
        ) * self.h
        self.face_points = (
            self.faces[:, 1:].astype(float) + 0.5 * (1 - np.eye(3)[self.faces[:, 0]])
        ) * self.h
        self.cell_points = (self.cells + 0.5) * self.h
        self._emap, self._fmap = emap, fmap

    def project(self, a, harmonic=True):
        a = np.asarray(a)
        result = a - self.G @ self.poisson.solve(np.asarray(self.G.T @ a))
        if harmonic and self.H.shape[1]:
            result -= self.H @ (self.H.T @ result)
        return result

    def spectrum(self, k, tol=1e-9):
        if not 0 < k < self.canonical_dimension:
            raise ValueError("Requested modes must be below the canonical dimension")


        hodge = self.C.T @ self.C + self.G @ self.G.T
        factor = sla.splu((hodge + sp.eye(len(self.edges))).tocsc())

        def inverse(x):
            return self.project(factor.solve(self.project(x)))

        op = sla.LinearOperator(
            hodge.shape, matvec=inverse, matmat=inverse, dtype=np.float64
        )
        rng = np.random.default_rng(20260907)



        solve_k = min(k + 8, self.canonical_dimension - 1)
        inv, a = sla.eigsh(
            op,
            k=solve_k,
            which="LA",
            tol=tol,
            v0=self.project(rng.normal(size=len(self.edges))),
        )
        order = np.argsort(1 / inv - 1)
        lam, a = (1 / inv - 1)[order], a[:, order]
        a = self.project(a)

        a = np.linalg.qr(a)[0]
        ca = self.C @ a
        lam, rotation = np.linalg.eigh(ca.T @ ca)
        lam, a = lam[:k], a @ rotation[:, :k]
        u = self.C @ a / np.sqrt(lam)
        return lam, a, u

    def to_cells(self, u):
        is_vector = u.ndim == 1
        if is_vector:
            u = u[:, None]
        result = np.zeros((len(self.cells), 3, u.shape[1]))
        for i, p in enumerate(map(tuple, self.cells)):
            for a in range(3):
                for q in (p, plus(p, a)):
                    j = self._fmap.get((a, *q))
                    if j is not None:
                        result[i, a] += 0.5 * u[j]
        return result[..., 0] if is_vector else result

    def metadata(self):
        return dict(
            resolution=self.n,
            fluid_cells=len(self.cells),
            interior_vertices=len(self.vertices),
            potential_edges=len(self.edges),
            velocity_faces=len(self.faces),
            boundary_components=self.boundary_components,
            relative_harmonic_potentials=self.H.shape[1],
            absolute_harmonic_velocities=self.absolute_harmonic_dimension,
            canonical_dimension=self.canonical_dimension,
            mass_per_dof=self.h**3,
            discretization="relative cubical incidence complex; voxel boundary; uniform Cartesian Hodge mass",
        )


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
