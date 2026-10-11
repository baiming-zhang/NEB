class FittedComplex:

    def __init__(self, mesh):
        self.mesh = mesh
        self.basis = Basis(mesh, ElementTetN1(), intorder=4)
        boundary = self.basis.get_dofs().all()
        self.free = np.setdiff1d(np.arange(self.basis.N), boundary)
        self.M = asm(mass, self.basis).tocsr()[self.free][:, self.free]
        self.S = asm(stiffness, self.basis).tocsr()[self.free][:, self.free]
        edge = mesh.edges[:, self.free]
        ga = sp.csr_matrix((np.tile([-1.0, 1.0], len(self.free)), (np.repeat(np.arange(len(self.free)), 2), edge.T.reshape(-1))), shape=(len(self.free), mesh.p.shape[1]))
        bn = mesh.boundary_nodes()
        vn = np.setdiff1d(np.arange(mesh.p.shape[1]), bn)
        self.G = ga[:, vn]
        self.poisson = sla.splu((self.G.T @ self.M @ self.G).tocsc())
        bedges = mesh.edges[:, boundary]
        graph = sp.csr_matrix((np.ones(2 * len(boundary)), (np.r_[bedges[0], bedges[1]], np.r_[bedges[1], bedges[0]])), shape=(mesh.p.shape[1],) * 2)[bn][:, bn]
        count, labels = connected_components(graph)
        extents = [np.ptp(mesh.p[:, bn[labels == i]], axis=1).max() for i in range(count)]
        outer = int(np.argmax(extents))
        harmonics = []
        for comp in range(count):
            if comp == outer:
                continue
            phi = np.zeros(mesh.p.shape[1])
            phi[bn[labels == comp]] = 1
            h = ga @ phi
            h -= self.G @ self.poisson.solve(self.G.T @ self.M @ h)
            for old in harmonics:
                h -= old * (old @ self.M @ h)
            h /= np.sqrt(h @ self.M @ h)
            harmonics.append(h)
        self.H = np.column_stack(harmonics) if harmonics else np.empty((len(self.free), 0))
        interior = mesh.f2t[:, mesh.f2t[1] >= 0]
        cell_graph = sp.csr_matrix((np.ones(interior.shape[1] * 2), (np.r_[interior[0], interior[1]], np.r_[interior[1], interior[0]])), shape=(mesh.t.shape[1],) * 2)
        b0 = int(connected_components(cell_graph)[0])
        node_labels = np.full(mesh.p.shape[1], -1, dtype=int)
        node_labels[bn] = labels
        boundary_facets = mesh.facets[:, mesh.boundary_facets()]
        surface_checks = []
        for comp in range(count):
            nodes = bn[labels == comp]
            faces = boundary_facets[:, node_labels[boundary_facets[0]] == comp]
            all_pairs = np.sort(np.concatenate([faces[[0, 1]].T, faces[[1, 2]].T, faces[[0, 2]].T]), axis=1)
            unique, incidence = np.unique(all_pairs, axis=0, return_counts=True)
            euler = len(nodes) - len(unique) + faces.shape[1]
            surface_checks.append(dict(component=comp, vertices=len(nodes), edges=len(unique), faces=faces.shape[1], euler=euler, genus=(2 - euler) / 2, closed_edge_manifold=bool(np.all(incidence == 2))))
        euler_absolute = mesh.p.shape[1] - mesh.edges.shape[1] + mesh.facets.shape[1] - mesh.t.shape[1]
        b2 = count - b0
        absolute_b1 = b0 + b2 - euler_absolute
        canonical = len(self.free) - len(vn) - self.H.shape[1]
        b1 = mesh.facets.shape[1] - len(mesh.boundary_facets()) - mesh.t.shape[1] + b0 - canonical
        self.metadata = dict(vertices=mesh.p.shape[1], tetrahedra=mesh.t.shape[1], relative_edges=len(self.free), boundary_components=count, b0=b0, b1=int(b1), b2=b2, dim_H_A_1=int(b1), dim_H_R_1=self.H.shape[1], absolute_euler=euler_absolute, absolute_b1=int(absolute_b1), boundary_surface_checks=surface_checks, canonical_dimension=canonical, gradient_kernel_relative=float(sla.norm(self.S @ self.G) / sla.norm(self.S)), harmonic_kernel_relative=float(np.linalg.norm(self.S @ self.H) / max(np.linalg.norm(self.H), 1e-30)))
        if b0 != 1 or b1 != absolute_b1 or b1 != 0 or (count != 2) or any((not s['closed_edge_manifold'] or s['genus'] != 0 for s in surface_checks)):
            raise ValueError(f'Candidate mesh topology out of scope: {self.metadata}')
        rng = np.random.default_rng(7001)
        x = rng.normal(size=(len(self.free), 2))
        y = rng.normal(size=x.shape)
        px = self.project(x)
        py = self.project(y)
        self.metadata['projector_idempotence'] = float(np.linalg.norm(self.project(px) - px) / np.linalg.norm(px))
        self.metadata['projector_mass_self_adjointness'] = float(np.linalg.norm(x.T @ self.M @ py - px.T @ self.M @ y) / np.linalg.norm(x.T @ self.M @ py))
        self.metadata['projector_euclidean_adjoint_check'] = float(np.linalg.norm(x.T @ py - self.project_transpose(x).T @ y) / np.linalg.norm(x.T @ py))
        perturb = self.G @ rng.normal(size=(self.G.shape[1], 2)) + self.H @ rng.normal(size=(self.H.shape[1], 2))
        self.metadata['gauge_invariance'] = float(np.linalg.norm(self.project(x + perturb) - px) / np.linalg.norm(px))
        self.metadata['curl_kernel_energy_relative'] = float(abs(np.sum(perturb * (self.S @ perturb))) / np.sum(x * (self.S @ x)))

    def project(self, x):
        result = x - self.G @ self.poisson.solve(np.asarray(self.G.T @ (self.M @ x)))
        return result - self.H @ (self.H.T @ (self.M @ result))

    def project_transpose(self, x):
        result = x - self.M @ (self.G @ self.poisson.solve(np.asarray(self.G.T @ x)))
        return result - self.M @ (self.H @ (self.H.T @ result))

    def spectrum(self, k):
        factor = sla.splu((self.S + self.M).tocsc())
        op = sla.LinearOperator(self.S.shape, matvec=lambda x: self.project(factor.solve(self.project_transpose(x))), dtype=np.float64)
        lam, a = sla.eigsh(self.S, k=k + 8, M=self.M, sigma=-1, which='LM', OPinv=op, tol=1e-09, v0=self.project(np.random.default_rng(20260907).normal(size=len(self.free))))
        a = self.project(a)
        gram = a.T @ self.M @ a
        w, v = np.linalg.eigh(gram)
        a = a @ (v / np.sqrt(w)) @ v.T
        lam, rot = np.linalg.eigh(a.T @ self.S @ a)
        a = a @ rot[:, :k]
        lam = lam[:k]
        residual = np.linalg.norm(self.S @ a - self.M @ a * lam, axis=0) / np.linalg.norm(self.M @ a * lam, axis=0)
        return (lam, a, residual)
