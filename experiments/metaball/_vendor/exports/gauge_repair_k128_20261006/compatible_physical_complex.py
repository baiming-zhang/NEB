import numpy as np
def quadrature(level):
    a = (5 + 3 * np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    bary = np.full((4, 4), b)
    np.fill_diagonal(bary, a)
    if level == 0:
        return (bary, np.full(4, 0.25))
    points = []
    for i in range(4):
        vertices = np.eye(4)
        vertices[i] = 0.25
        points.extend(bary @ vertices)
    return (np.array(points), np.full(16, 1 / 16))

def main(dest):
    source = ROOT / 'ICLR2027_campaign/cases/metaball/results/fitted_metaball_reference_v2/h0.025_surface80_K1024/modes.npz'
    with np.load(source) as z:
        p = z['p'].T
        t = z['t'].T
        free = z['free_edges']
    mesh = MeshTet(p.T, t.T)
    edges = mesh.edges.T
    local_edges = mesh.t2e.T
    bn = mesh.boundary_nodes()
    vn = np.setdiff1d(np.arange(len(p)), bn)
    ga = sp.csr_matrix((np.tile([-1.0, 1.0], len(free)), (np.repeat(np.arange(len(free)), 2), edges[free].ravel())), shape=(len(free), len(p)))
    G = ga[:, vn].tocsr()
    faces = mesh.facets.T
    codes = edges[:, 0].astype('i8') * len(p) + edges[:, 1]
    order = np.argsort(codes)
    fe = np.stack([faces[:, [1, 2]], faces[:, [0, 2]], faces[:, [0, 1]]], 1)
    edgeids = order[np.searchsorted(codes[order], fe[:, :, 0] * len(p) + fe[:, :, 1])]
    invfree = np.full(len(edges), -1, dtype='i8')
    invfree[free] = np.arange(len(free))
    ci = invfree[edgeids]
    keep = ci >= 0
    row = np.broadcast_to(np.arange(len(faces))[:, None], ci.shape)
    signs = np.broadcast_to(np.array([1.0, -1.0, 1.0]), ci.shape)
    C = sp.csr_matrix((signs[keep], (row[keep], ci[keep])), shape=(len(faces), len(free)))
    cg = C @ G
    cg.eliminate_zeros()
    assert cg.nnz == 0
    cfg = read(Path('_data/metaball/potential_capacity_20261005/potential_k128/config.json'))
    c = config_namespace()
    geo = cp.asarray(cfg['geometry'], 'f8')
    mapping = RadialMap(mesh, geo, c)
    write(dest / 'geometry_certificate.json', mapping.certificate)
    points, weights = quadrature(LEVEL)
    v = p[t]
    B = (v[:, 1:] - v[:, :1]).transpose(0, 2, 1)
    inv = np.linalg.inv(B)
    grad = np.concatenate((-inv.sum(1)[:, None, :], inv), 1)
    vol = np.abs(np.linalg.det(B)) / 6
    orient = np.stack([1 - 2 * (t[:, a] > t[:, b]) for a, b in PAIRS], 1)
    lm = np.zeros((len(t), 6, 6))
    ls = np.zeros_like(lm)
    geometry_checks = []
    for off in range(0, len(t), 512):
        stop = min(off + 512, len(t))
        vv = cp.asarray(v[off:stop])
        gg = cp.asarray(grad[off:stop])
        sg = cp.asarray(orient[off:stop])
        lc = cp.stack([2 * cp.cross(gg[:, a], gg[:, b]) * sg[:, j, None] for j, (a, b) in enumerate(PAIRS)], 1)
        mm = cp.zeros((stop - off, 6, 6))
        ss = cp.zeros_like(mm)
        for bary, w in zip(points, weights):
            x = cp.einsum('v,nva->na', cp.asarray(bary), vv)
            y, J, det, check = mapping.map(x)
            assert check['inverse_point_max'] < 1e-10 and check['inverse_J_max'] < 1e-08 and (float(det.min()) > 0), check
            phi = cp.stack([(bary[a] * gg[:, b] - bary[b] * gg[:, a]) * sg[:, j, None] for j, (a, b) in enumerate(PAIRS)], 1)
            phys = cp.einsum('nai,nei->nea', cp.linalg.inv(J).transpose(0, 2, 1), phi)
            curl = cp.einsum('nai,nei->nea', J, lc) / det[:, None, None]
            wt = cp.asarray(vol[off:stop]) * det * w
            mm += cp.einsum('nei,nfi,n->nef', phys, phys, wt)
            ss += cp.einsum('nei,nfi,n->nef', curl, curl, wt)
        lm[off:stop] = cp.asnumpy(mm)
        ls[off:stop] = cp.asnumpy(ss)
        if off % (512 * 8) == 0:
            progress(dest, 'all_cell_true_metric_assembly', cells_done=stop, cells_total=len(t), quadrature_per_cell=len(points))
            resource_check(dest)
    ii = np.broadcast_to(local_edges[:, :, None], lm.shape).ravel()
    jj = np.broadcast_to(local_edges[:, None, :], lm.shape).ravel()
    M = sp.coo_matrix((lm.ravel(), (ii, jj)), shape=(len(edges), len(edges))).tocsr()[free][:, free]
    S = sp.coo_matrix((ls.ravel(), (ii, jj)), shape=(len(edges), len(edges))).tocsr()[free][:, free]
    del lm, ls, ii, jj
    data_dir = Path('_data/metaball/gauge_repair_k128_20261006') / dest.name
    data_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(data_dir / 'M1.npz', M)
    sp.save_npz(data_dir / 'S1.npz', S)
    sp.save_npz(data_dir / 'G.npz', G)
    sp.save_npz(data_dir / 'C.npz', C)
    progress(dest, 'factor_physical_scalar_poisson', scalar_dof=len(vn), edge_dof=len(free))
    K = (G.T @ M @ G).tocsc()
    with threadpool_limits(limits=2):
        factor = sla.splu(K)
    bnd_edges = edges[np.setdiff1d(np.arange(len(edges)), free)]
    graph = sp.csr_matrix((np.ones(2 * len(bnd_edges)), (bnd_edges.T.ravel(), bnd_edges[:, ::-1].T.ravel())), shape=(len(p), len(p)))[bn][:, bn]
    count, labels = connected_components(graph)
    assert count == 2
    inner = int(np.argmin([np.ptp(p[bn[labels == i]], axis=0).max() for i in range(count)]))
    ph = np.zeros(len(p))
    ph[bn[labels == inner]] = 1
    H = ga @ ph
    H -= G @ factor.solve(G.T @ (M @ H))
    H /= np.sqrt(H @ (M @ H))
    H = H[:, None]

    def project(a):
        z = a - G @ factor.solve(G.T @ (M @ a))
        return z - H @ (H.T @ (M @ z))

    def reverse(a):
        z = a - M @ (G @ factor.solve(G.T @ a))
        return z - M @ (H @ (H.T @ z))
    yy, J, det, check = mapping.map(cp.asarray(p))
    q = (yy - 0.5) / c.outer_half_extent
    out = 1 - cp.sum(q ** c.outer_superellipsoid_power, 1)
    d = yy[:, None] - geo[None, :, :3]
    dist = cp.linalg.norm(d, axis=2) - geo[None, :, 3]
    m = dist.min(1)
    innerlevel = m - c.metaball_softmin_tau * cp.log(cp.exp(-(dist - m[:, None]) / c.metaball_softmin_tau).sum(1))
    envelope = out / (c.outer_envelope_transition + out) * innerlevel / (c.obstacle_envelope_transition + innerlevel)
    rows = []
    for frequency in (1, 4, 12, 32):
        psi = cp.asnumpy(envelope ** 2 * cp.sin(2 * np.pi * frequency * (yy @ cp.asarray([1.0, 2.0, 3.0]))))
        boundary_max = float(np.max(np.abs(psi[bn])))
        psi[bn] = 0
        a = (ga @ psi)[:, None]
        z = project(a)
        mass = float(np.sum(z * (M @ z)) / np.sum(a * (M @ a)))
        rows.append(dict(frequency=frequency, discrete_gradient_residual_mass=mass, analytic_endpoint_commuting=True, boundary_scalar_max_before_roundoff_zeroing=boundary_max))
        assert mass < 1e-16, rows[-1]
    rng = np.random.default_rng(61006)
    a = rng.normal(size=(len(free), 4))
    cot = rng.normal(size=a.shape)
    z = project(a)
    back = reverse(cot)
    adj = abs(np.sum(z * cot) - np.sum(a * back)) / (np.linalg.norm(z) * np.linalg.norm(cot))
    curlrel = np.linalg.norm(C @ (z - a)) / max(np.linalg.norm(C @ a), 1e-300)
    idem = np.linalg.norm(project(z) - z) / np.linalg.norm(z)
    gauge = np.linalg.norm(G.T @ (M @ z)) / np.linalg.norm(M @ z)
    harmonic = np.linalg.norm(H.T @ (M @ z)) / np.linalg.norm(M @ z)
    kernel = float(sla.norm(S @ G) / sla.norm(S))
    hkernel = float(np.linalg.norm(S @ H) / np.linalg.norm(H))
    assert max(adj, curlrel, idem, gauge, harmonic, kernel) < 1e-09
    np.savez(data_dir / 'complex.npz', p=p, t=t, free=free, vn=vn, bn=bn, H=H, points=points, weights=weights)
    return dict(status='complete', qualified=True, scope='Compatible discrete gauge only; continuous ANN representation and training not qualified', same_original_mesh_cells=len(t), vertices=len(p), relative_edges=len(free), scalar_dof=len(vn), boundary_components=count, harmonic_dimension=1, physical_metric='Analytic radial map to original smooth walls, covariant potential/contravariant curl Piola', quadrature_level=LEVEL, quadrature_points_per_cell=len(points), CG_exact_integer_zero=cg.nnz == 0, rows=rows, adjoint_relative=adj, curl_invariance_relative=curlrel, idempotence_relative=idem, gauge_relative=gauge, harmonic_relative=harmonic, stiffness_gradient_kernel_relative=kernel, stiffness_harmonic_kernel_relative=hkernel, data_dir=str(data_dir), neural_checkpoint_evaluated=False, continuous_manufactured_field_approximation_not_claimed=True, training_not_released=True)
