from __future__ import annotations
import os,sys,time,json,hashlib,ast,gc,itertools
from pathlib import Path
import numpy as np,scipy.sparse as sp,scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet,Basis,ElementTetN1
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
import gmsh,meshio
from skimage.measure import marching_cubes
from scipy.special import logsumexp
def mesh_metaball(geometry, h, n, out):
    axis = np.linspace(0, 1, n)
    x = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    conf = cfg()
    outer = 1 - np.sum(((x - 0.5) / conf.outer_half_extent) ** conf.outer_superellipsoid_power, axis=-1)
    geom = np.asarray(geometry)
    distance = np.linalg.norm(x[..., None, :] - geom[None, None, None, :, :3], axis=-1) - geom[:, 3]
    obstacle = -conf.metaball_softmin_tau * logsumexp(-distance / conf.metaball_softmin_tau, axis=-1)
    verts, faces, _, _ = marching_cubes(np.minimum(outer, obstacle), level=0, spacing=(1 / (n - 1),) * 3, allow_degenerate=False)
    meshio.write(out / 'surface.stl', meshio.Mesh(verts, [('triangle', faces)]), binary=True)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1)
    gmsh.option.setNumber('General.NumThreads', 2)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(len(verts), len(verts))).tocsr()
    count, labels = connected_components(graph)
    if count != 2:
        raise ValueError(f'Surface components {count}; expected outer and one obstacle')
    components = sorted(range(count), key=lambda c: np.ptp(verts[labels == c], axis=0).max(), reverse=True)
    loops = []
    for index, component in enumerate(components, 1):
        selected = faces[labels[faces[:, 0]] == component]
        nodes = np.unique(selected)
        tag = gmsh.model.addDiscreteEntity(2, index)
        gmsh.model.mesh.addNodes(2, tag, (nodes + 1).tolist(), verts[nodes].reshape(-1).tolist())
        gmsh.model.mesh.addElementsByType(tag, 2, [], (selected + 1).reshape(-1).tolist())
        loops.append(gmsh.model.geo.addSurfaceLoop([tag]))
    gmsh.option.setNumber('Mesh.MeshOnlyEmpty', 1)
    gmsh.model.geo.addVolume(loops)
    gmsh.model.geo.synchronize()
    for key, value in [('Mesh.MeshSizeMin', h), ('Mesh.MeshSizeMax', h), ('Mesh.MeshSizeFromPoints', 0), ('Mesh.MeshSizeFromCurvature', 0), ('Mesh.MeshSizeExtendFromBoundary', 0)]:
        gmsh.option.setNumber(key, value)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(out / 'mesh.msh'))
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    mapper = {int(t): i for i, t in enumerate(tags)}
    _, tet = gmsh.model.mesh.getElementsByType(4)
    cells = np.array([mapper[int(i)] for i in tet]).reshape(-1, 4).T
    mesh = MeshTet(np.asarray(xyz).reshape(-1, 3).T, cells)
    gmsh.finalize()
    return mesh
