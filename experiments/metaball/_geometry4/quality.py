from __future__ import annotations
import os,sys,time,json,hashlib,ast,gc,itertools
from pathlib import Path
import numpy as np,scipy.sparse as sp,scipy.sparse.linalg as sla
from scipy.sparse.csgraph import connected_components
from skfem import MeshTet,Basis,ElementTetN1
PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]
import gmsh
def boundary_coordinates(mesh):
    xyz = mesh.p.T[mesh.facets.T[mesh.boundary_facets()]]
    order = np.lexsort((xyz[:, :, 2], xyz[:, :, 1], xyz[:, :, 0]), axis=1)
    xyz = np.take_along_axis(xyz, order[:, :, None], axis=1).reshape(-1, 9)
    xyz = xyz[np.lexsort(tuple((xyz[:, i] for i in range(8, -1, -1))))]
    return xyz

def mesh_from_gmsh():
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    mapper = {int(t): i for i, t in enumerate(tags)}
    _, tet = gmsh.model.mesh.getElementsByType(4)
    cells = np.fromiter((mapper[int(i)] for i in tet), dtype='i8').reshape(-1, 4).T
    return MeshTet(np.asarray(xyz).reshape(-1, 3).T, cells)

def main():
    assert not (OUT / 'mesh_complete.json').exists(), 'Do not repeat an already completed quality repair'
    with np.load(OLD / 'mesh_arrays.npz') as z:
        original = MeshTet(z['p'], z['t'])
    original_boundary = boundary_coordinates(original)
    gmsh.initialize()
    gmsh.option.setNumber('General.NumThreads', 2)
    gmsh.open(str(OLD / 'mesh.msh'))
    loaded = mesh_from_gmsh()
    before_boundary = boundary_coordinates(loaded)
    assert before_boundary.shape == original_boundary.shape
    roundtrip_difference = float(np.max(np.abs(before_boundary - original_boundary)))
    assert roundtrip_difference < 1e-14, 'Original MSH and original mesh arrays differ beyond text serialization rounding'
    before = hashlib.sha256(before_boundary.tobytes()).hexdigest()
    del loaded, original
    gmsh.model.mesh.optimize('Netgen', force=True, niter=5)
    mesh = mesh_from_gmsh()
    after_boundary = boundary_coordinates(mesh)
    after = hashlib.sha256(after_boundary.tobytes()).hexdigest()
    np.savez(OUT / 'candidate_mesh_arrays.npz', p=mesh.p, t=mesh.t)
    write('boundary_diagnostics.json', dict(original_msh_serialization_difference=roundtrip_difference, before_faces=len(before_boundary), after_faces=len(after_boundary), before_sha=before, after_sha=after, optimizer_changed_boundary=before != after, coordinate_max_difference=float(np.max(np.abs(before_boundary - after_boundary))) if before_boundary.shape == after_boundary.shape else None))
    assert before == after, 'Optimizer changed the prescribed boundary; reject this mesh'
    gmsh.write(str(OUT / 'mesh.msh'))
    np.savez(OUT / 'mesh_arrays.npz', p=mesh.p, t=mesh.t)
    write('mesh_complete.json', dict(status='quality_repair_complete_chart_qualification_pending', boundary_coordinates_and_faces_identical=True, boundary_sha256=before, source_mesh=str(OLD / 'mesh.msh'), vertices=mesh.p.shape[1], tetrahedra=mesh.t.shape[1], original_h=0.025, surface_resolution=80, seconds=time.perf_counter() - START, optimizer='Gmsh Netgen, five iterations, fixed boundary', spectral_solve_started=False))
