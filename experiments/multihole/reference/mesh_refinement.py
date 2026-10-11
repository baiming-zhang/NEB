"""Original analytic interpolation-error refinement of the polygon mesh."""
import numpy as np
import meshpy.triangle as triangle
from .geometry import outer_boundary,obstacle_boundary,get_obstacles,fluid_mask
from .mesh_builder import DomainMesh
BARY=np.array([[1/3,1/3,1/3],[.059715871789770,.470142064105115,.470142064105115],[.470142064105115,.059715871789770,.470142064105115],[.470142064105115,.470142064105115,.059715871789770],[.797426985353087,.101286507323456,.101286507323456],[.101286507323456,.797426985353087,.101286507323456],[.101286507323456,.101286507323456,.797426985353087]])
WEIGHTS=np.array([.225]+[.132394152788506]*3+[.125939180544827]*3)
def use_native():
    global cp
    import cupy as cp

def mesh_from(info, case):
    xy = np.asarray(info.points, dtype='f8')
    tri = np.asarray(info.elements, dtype='i4')
    edges = np.asarray(info.facets, dtype='i4')
    labels = np.asarray(info.facet_markers, dtype='i4') - 1
    pts = xy[tri]
    det = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    neg = det < 0
    tri[neg, 1], tri[neg, 2] = (tri[neg, 2].copy(), tri[neg, 1].copy())
    nodes = tuple((np.unique(edges[labels == i]) for i in range(len(get_obstacles(case)) + 1)))
    return DomainMesh(xy, tri, edges, labels, nodes, case, get_obstacles(case), 0.0, 0.004)

def initial_mesh(case):
    loops = [outer_boundary(0.0, 2048)] + [obstacle_boundary(s, 1024)[::-1] for s in get_obstacles(case)]
    pts = []
    facets = []
    markers = []
    for j, loop in enumerate(loops):
        first = len(pts)
        pts.extend(loop.tolist())
        facets.extend(((first + i, first + (i + 1) % len(loop)) for i in range(len(loop))))
        markers.extend([j + 1] * len(loop))
    info = triangle.MeshInfo()
    info.set_points(pts)
    info.set_facets(facets, facet_markers=markers)
    info.set_holes([s.anchor.tolist() for s in get_obstacles(case)])
    return triangle.build(info, max_volume=0.004, min_angle=28, allow_boundary_steiner=False, generate_faces=True)

def interpolation_metrics(mesh, field, source):
    xy = cp.asarray(mesh.xy)
    tri = cp.asarray(mesh.tri)
    nodal = field.evaluate(xy)[:, 0]
    correction = 0.0
    for j, nodes in enumerate(mesh.boundary_nodes):
        idx = cp.asarray(nodes)
        constant = source.scale * source.constants[j]
        correction = max(correction, float(cp.max(abs(nodal[idx] - constant))))
        nodal[idx] = constant
    error = np.empty(len(tri))
    norm = np.empty(len(tri))
    areas = np.empty(len(tri))
    for start in range(0, len(tri), 2048):
        cells = tri[start:start + 2048]
        p = xy[cells]
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        edges = cp.stack((p[:, 1] - p[:, 2], p[:, 2] - p[:, 0], p[:, 0] - p[:, 1]), axis=1)
        grad = cp.stack((edges[:, :, 1], -edges[:, :, 0]), axis=2) / det[:, None, None]
        g = cp.einsum('ti,tij->tj', nodal[cells], grad)
        velocity = cp.stack((g[:, 1], -g[:, 0]), axis=1)
        points = cp.einsum('qi,tij->tqj', cp.asarray(BARY), p)
        values = field.evaluate(points.reshape(-1, 2)).reshape(-1, 7, 4)
        target = values[:, :, 1:3]
        weight = det[:, None] / 2 * cp.asarray(WEIGHTS)[None, :]
        e = cp.sum(weight * cp.sum((velocity[:, None] - target) ** 2, axis=2), axis=1)
        n = cp.sum(weight * cp.sum(target ** 2, axis=2), axis=1)
        error[start:start + len(cells)] = cp.asnumpy(e)
        norm[start:start + len(cells)] = cp.asnumpy(n)
        areas[start:start + len(cells)] = cp.asnumpy(det / 2)
    return (dict(relative_velocity_error=float(np.sqrt(error.sum() / norm.sum())), analytic_energy_twice=float(norm.sum()), boundary_trace_correction_max=correction), error, areas)
