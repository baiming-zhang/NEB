def mesh_metaball(geometry, h, n, out):
    axis = np.linspace(0, 1, n)
    x = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    conf = cfg()
    outer = 1 - np.sum(((x - 0.5) / conf.outer_half_extent) ** conf.outer_superellipsoid_power, axis=-1)
    geom = np.asarray(geometry)
    distance = np.linalg.norm(x[..., None, :] - geom[None, None, None, :, :3], axis=-1) - geom[:, 3]
    obstacle = -conf.metaball_softmin_tau * logsumexp(-distance / conf.metaball_softmin_tau, axis=-1)
    verts, faces, _, _ = marching_cubes(np.minimum(outer, obstacle), level=0, spacing=(1 / (n - 1),) * 3)
    meshio.write(out / 'surface.stl', meshio.Mesh(verts, [('triangle', faces)]), binary=True)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 1)
    gmsh.option.setNumber('General.NumThreads', 2)
    gmsh.merge(str(out / 'surface.stl'))
    gmsh.model.mesh.classifySurfaces(40 * np.pi / 180, True, True, np.pi)
    gmsh.model.mesh.createGeometry()
    surfaces = gmsh.model.getEntities(2)
    curve_sets = [{tag for _, tag in gmsh.model.getBoundary([s], oriented=False)} for s in surfaces]
    remaining = set(range(len(surfaces)))
    groups = []
    while remaining:
        group = {remaining.pop()}
        front = list(group)
        while front:
            i = front.pop()
            near = {j for j in remaining if curve_sets[i] & curve_sets[j]}
            remaining -= near
            group |= near
            front.extend(near)
        groups.append(group)
    if len(groups) != 2:
        raise ValueError(f'Surface components {len(groups)}; expected outer and one obstacle')
    bounds = []
    for group in groups:
        boxes = np.asarray([gmsh.model.getBoundingBox(*surfaces[i]) for i in group])
        bounds.append(float(np.max(boxes[:, 3:].max(axis=0) - boxes[:, :3].min(axis=0))))
    groups = [groups[i] for i in np.argsort(bounds)[::-1]]
    loops = [gmsh.model.geo.addSurfaceLoop([surfaces[i][1] for i in group]) for group in groups]
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
