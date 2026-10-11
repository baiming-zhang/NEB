
import importlib.util
import sys
from types import SimpleNamespace
import numpy as np
from common import CAMPAIGN, WORK, read
from core import features


def topology_module():
    path = CAMPAIGN/'cases/topology3/baseline_fresh_v1/src/geometry.py'
    spec = importlib.util.spec_from_file_location('nkb_original_topology_geometry', path)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


class Domain:
    def __init__(self, name, xp):
        self.name, self.xp = name, xp
        self.periodic = name == 'periodic'
        self.dim = 3 if name == 'metaball' else 2
        self.descriptor = None
        if name == 'metaball':
            self.base_config = read(CAMPAIGN/'cases/metaball/results/ablation_failure_v1/lr_training32_v1/config.json')
            with np.load(CAMPAIGN/'cases/metaball/results/velocity_k1024_full2000_v1/checkpoint_comparison20/case_01/samples.npz') as z:
                self.geo = z['geometry'].copy()
            self.descriptor = self.geo.ravel()
            self.lo, self.hi, self.bandwidth = np.zeros(3), np.ones(3), .05
        elif self.periodic:
            self.lo, self.hi, self.bandwidth = np.zeros(2), np.full(2, 2*np.pi), 0.
        else:
            self.topology = topology_module()
            self.topology.get_obstacles(name)
            self.lo, self.hi, self.bandwidth = np.array([1., -4.]), np.array([9., 4.]), .4

    def encode(self, points):
        if self.periodic: return features(points, self.xp, True)
        lo, hi = self.xp.asarray(self.lo, dtype=points.dtype), self.xp.asarray(self.hi, dtype=points.dtype)
        f, j = features((points-lo)/(hi-lo), self.xp, False, self.descriptor)
        j /= (hi-lo)[None, :, None]
        return f, j

    def weights(self, points, geometry=None):
        xp = self.xp
        if self.periodic:
            return xp.ones(len(points), dtype=points.dtype), xp.zeros(len(points), dtype=points.dtype), xp.zeros_like(points)
        if self.name == 'metaball':
            geo = xp.asarray(self.geo if geometry is None else geometry, dtype=points.dtype)
            cfg = self.base_config
            q = (points-.5)/cfg['outer_half_extent']
            level = 1-xp.sum(q**cfg['outer_superellipsoid_power'], axis=1)
            gradient = -cfg['outer_superellipsoid_power']/cfg['outer_half_extent']*q**(cfg['outer_superellipsoid_power']-1)
            delta = points[:, None, :]-geo[None, :, :3]
            length = xp.sqrt(xp.sum(delta*delta, axis=2)); length = xp.maximum(length, 1e-12)
            sphere = length-geo[None, :, 3]
            smallest = sphere.min(axis=1); tau = cfg['metaball_softmin_tau']
            ex = xp.exp(-(sphere-smallest[:, None])/tau)
            inner = smallest-tau*xp.log(ex.sum(axis=1))
            ig = xp.sum((ex/ex.sum(axis=1)[:, None])[:, :, None]*delta/length[:, :, None], axis=1)
            levels = xp.stack((level, inner), axis=1)
            grads = xp.stack((gradient, ig), axis=1)
        else:

            host = np.asarray(points) if xp is np else xp.asnumpy(points)
            g = self.topology
            levels = xp.asarray(np.stack([-g.outer_implicit(host, 0.)] + [g.obstacle_implicit(host, s) for s in g.get_obstacles(self.name)], axis=1), dtype=points.dtype)
            grads = xp.asarray(np.stack([-g.outer_gradient(host, 0.)] + [g.obstacle_gradient(host, s) for s in g.get_obstacles(self.name)], axis=1), dtype=points.dtype)
        lengths = xp.maximum(xp.sqrt(xp.sum(grads*grads, axis=2)), 1e-12)
        signed = levels/lengths
        index = xp.argmin(xp.abs(signed), axis=1)
        distance = signed[xp.arange(len(points)), index]
        normal = grads[xp.arange(len(points)), index]/lengths[xp.arange(len(points)), index, None]
        inside = xp.all(levels >= 0, axis=1)
        z = xp.minimum(xp.abs(distance)/self.bandwidth, 1)
        band = xp.maximum(2*z**3-3*z*z+1, 0)**4
        weight = xp.where(inside, 1, band)
        return weight, band, normal

    def sample(self, count, seed, geometry=None):
        rng = np.random.default_rng(seed)
        margin = self.bandwidth
        host = rng.uniform(self.lo-margin, self.hi+margin, (count, self.dim)).astype('float32')
        p = self.xp.asarray(host)
        f, j = self.encode(p)
        if geometry is not None and self.name == 'metaball':
            f[:, 3:] = self.xp.asarray(geometry, dtype=p.dtype).ravel()
        return p, f, j, self.weights(p, geometry)

    def validation_geometries(self):
        if self.name != 'metaball': return [None]

        import ast, math
        path = WORK/'3d_metaball_reproduction_latest/cases/3D_Metaball/scripts/train_validate.py'
        names = {'geometry_from_token', 'overlap_graph_connected', 'sphere_directions', 'spheres_inside_rounded_cube'}
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names], type_ignores=[])
        ns = dict(np=np, math=math, Config=object)
        exec(compile(module, str(path), 'exec'), ns)
        ns['SPHERE_DIRECTIONS'] = ns['sphere_directions']()
        cfg = SimpleNamespace(**self.base_config)
        return [ns['geometry_from_token'](cfg, 10000000+i) for i in range(8)]

    def contract(self):
        data = dict(name=self.name, dim=self.dim, bbox=[self.lo.tolist(), self.hi.tolist()],
                    boundary_bandwidth=self.bandwidth, periodic_sincos_embedding=self.periodic,
                    geometry_training='fixed original assessment domain', derivative_coordinates='physical',
                    hard_boundary_envelope=False, snapshot_supervision=False)
        if self.name == 'metaball': data.update(geometry=self.geo.tolist(), base_config=self.base_config)
        elif not self.periodic:
            data.update(outer_tau=0., obstacles=[s.as_dict() for s in self.topology.get_obstacles(self.name)])
        return data
