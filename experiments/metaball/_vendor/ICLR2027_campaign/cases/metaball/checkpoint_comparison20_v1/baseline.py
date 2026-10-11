





from _neb_paths import Path
from types import SimpleNamespace
import ast
import json
import sys
import numpy as np

CASE = Path(__file__).resolve().parent.parent
OLD = CASE / 'laplacian1024_full_ns_reconstruction_v1'
RESULT = CASE / 'results/laplacian1024_full_ns_reconstruction_v1'


def geometry_qualification(geometry):

    cfg = json.loads((RESULT/'geometry_config.json').read_text())
    c = SimpleNamespace(**cfg['legacy_config'])
    geo = np.asarray(geometry,dtype='f8')
    if geo.shape != (3,4) or not np.isfinite(geo).all(): raise ValueError('geometry must be finite (3,4)')
    center = geo[1,:3]; distance = np.linalg.norm(geo[:,:3]-center,axis=1)
    margin = geo[1,3]**2+geo[:,3]**2-distance**2
    bounds = distance+geo[:,3]; top = bounds.max()
    inner_hi = top+c.metaball_softmin_tau*np.log(np.exp((bounds-top)/c.metaball_softmin_tau).sum())
    outer_lo = c.outer_half_extent-np.linalg.norm(center-.5)
    original = np.array_equal(geo,np.asarray(cfg['geometry'],dtype='f8'))
    failed = []
    if np.any(margin<=1e-8): failed.append('inner_uniqueness_ball_margin')
    if outer_lo<=inner_hi+1e-6: failed.append('conservative_true_shell_separation')
    return dict(original_geometry=bool(original), center=center.tolist(),
        inner_uniqueness_ball_margins=margin.tolist(), true_shell_separation_lower=float(outer_lo-inner_hi),
        failed_necessary_checks=failed, baseline_available=bool(original),
        qualification='case1 saved Piola-mapped numerical Whitney curls; not exact eigenfields of curved geometry' if original else
        'incomplete: source mesh/eigenproblem is another geometry; target-specific radial mapping unverified. Passing these inequalities alone does not certify a map or eigenproblem.')


def _load_class(path, name, namespace):


    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name]
    if len(nodes)!=1: raise ValueError('saved source API changed')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]


class Baseline:
    def __init__(self):
        self.reference = None
        self.mapping = None
        self.qualification = None

    def _initialize(self, geometry):
        qualification = geometry_qualification(geometry)
        if not qualification['baseline_available']:
            raise NotImplementedError(json.dumps(qualification))
        if self.reference is not None: return
        import cupy as cp
        vendor = str(CASE/'vendor')
        if vendor not in sys.path: sys.path.insert(0,vendor)
        from skfem import MeshTet
        ns = dict(cp=cp,np=np,MeshTet=MeshTet,PAIRS=[(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)])
        Reference = _load_class(OLD/'reference.py','Reference',ns)
        RadialMap = _load_class(OLD/'piola_radial_geometry.py','RadialMap',dict(cp=cp,np=np))
        plan = json.loads((RESULT/'plan.json').read_text())
        cfg = json.loads((RESULT/'geometry_config.json').read_text())
        self.reference = Reference(Path(plan['modes']),lambda *args,**kwargs:None)
        self.mapping = RadialMap(self.reference.mesh,np.asarray(geometry,dtype='f8'),SimpleNamespace(**cfg['legacy_config']))
        self.qualification = qualification

    def matrix(self, points, geometry):





        self._initialize(geometry)
        import cupy as cp
        x = cp.asarray(points,dtype='f8')
        if x.ndim!=2 or x.shape[1]!=3: raise ValueError('points must have shape (N,3)')
        if len(x)==0: return cp.empty((0,1024),dtype='f8')
        fo,fi,_,_ = self.mapping.level_grad(x)
        if not bool(cp.all((fo>0)&(fi>0))): raise ValueError('non-fluid evaluation point')
        xp,J,det,check = self.mapping.map(x,inverse=True)
        ids = self.reference.locate(xp)
        if bool(cp.any(ids<0)): raise ValueError('point outside mapped source mesh')
        if float(det.min())<=0 or check['inverse_point_max']>=1e-10 or check['inverse_J_max']>=1e-8:
            raise ValueError('mapping inverse/determinant check failed')
        u = cp.einsum('nij,njk->nik',J/det[:,None,None],self.reference.velocity(ids))
        return u.reshape(-1,1024)


_default = None


def matrix(points, geometry):
    global _default
    if _default is None: _default = Baseline()
    return _default.matrix(points,geometry)
