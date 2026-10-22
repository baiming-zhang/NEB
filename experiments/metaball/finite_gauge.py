"""Finite-gauge potential objective with analytic native CuPy fields."""
from pathlib import Path
import ast
FG_SOURCE=Path(__file__).resolve().parent/"_backend/exports/potential_rayleigh/train_potential.py"
KS=(32,64,128,256)
FIXED = dict(seed=20260823, epochs=2000, points_per_step=6144,
             validation_points=4096, validation_geometries=8, validate_every=25,
             hidden=128, hidden_layers=5, siren_first_omega0=40.,
             siren_hidden_omega0=36., learning_rate=2.8e-5,
             min_learning_rate=1e-7, grad_clip=8., dtype='float32',
             mode_batch=16, point_chunk=1024)

def validate_configuration(c, K):
    if K not in KS or c.modes != K:
        raise ValueError('Explicit same-K matched FG32/64/128/256 required')
    for name, value in FIXED.items():
        if getattr(c, name, None) != value:
            raise ValueError('Shared velocity configuration differs: ' + name)
    return True

def original_objective(cp):
    """Compile the exact original function AST, without importing CUDA sources."""
    tree = ast.parse(FG_SOURCE.read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'bounded_potential_objective')
    ns = {'cp': cp}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(FG_SOURCE), 'exec'), ns)
    return ns[node.name]

class NativeFiniteGaugePipeline:
    """Original finite441 physics with caller's unchanged shared ANN functions.

    finite_factory must be original small.Small; geometry must expose original
    first_geometry.fields/gradient. Neither factory is imported here. These
    source identities must be pinned by the future scientific binding.
    """
    def __init__(self, cm, c, geo, source_points, finite_factory, geometry,
                 resource_callback, *, fixture=False):
        self.cm, self.cp, self.c, self.geo = cm, cm.cp, c, geo
        self.resource_callback = resource_callback
        if not fixture:
            validate_configuration(c, c.modes)
            if len(source_points) != 103244:
                raise ValueError('Complete original source points required')
        self.fixture = fixture
        self.resource_callback()
        self.x = self.cp.asarray(source_points, dtype='f8')
        self.fit = cm.rt.NeuralPoints(source_points, geo, c, 'f4')
        self.scalar = finite_factory(c, geo, self.x)
        if not fixture and (self.scalar.n != 441 or self.scalar.info['constraint_rank'] != 10):
            raise ValueError('Original finite441/10-constraint scalar method required')
        self.geometry = geometry
        self.objective = original_objective(self.cp)
        _, _, _, dlift = geometry.fields(self.x, geo, c)
        lf = dlift.reshape(-1, 1)
        self.hco = self.solve(lf, 'setup_harmonic')
        self.hf = lf - self.scalar.fit_field(self.hco)
        self.hm = self.cp.sum(self.hf * self.hf)
        if not float(self.hm) > 0:
            raise FloatingPointError('Harmonic mass is not positive')
        self.Gfh = self.scalar.fit_transpose(self.hf)
        self.resource_callback()

    def solve(self, a, label, transpose=False):
        self.resource_callback()
        value, record = (self.scalar.transpose(a, label, 180) if transpose
                         else self.scalar.solve(a, label, 180))
        if not record['converged'] or not bool(self.cp.all(self.cp.isfinite(value))):
            raise FloatingPointError('Original finite scalar solve failed')
        return value

    def compute(self, model, points, backward=True, dtype='f4'):
        if not self.fixture:
            validate_configuration(self.c, model.K)
            if len(points) not in (6144, 4096, 16384):
                raise ValueError('Original whole query/validation/independent scope required')
        self.resource_callback()
        cp, cm = self.cp, self.cm
        self.scalar.records = []
        data = cm.rt.NeuralPoints(points, self.geo, self.c, dtype)
        x = cp.asarray(points, dtype='f8')
        G, dlift = self.geometry.gradient(self.scalar.base, x, self.geo, self.c)
        h = dlift - G @ self.hco
        fields = cm.rt.grouped8.network_fields(model, data, 'au')
        af = cp.asarray(cm.rt.grouped8.network_fields(model, self.fit, 'a')['a'], dtype='f8')
        co = self.solve(af, 'forward')
        pf = af - self.scalar.fit_field(co)
        amp = self.hf.T @ pf / self.hm
        p = cp.asarray(fields['a'], dtype='f8') - G @ co - h @ amp
        obj = self.objective(p, cp.asarray(fields['u'], dtype='f8'), data.n)
        result = dict(obj=obj, co=co, amp=amp)
        if backward:
            beta = h.T @ obj['GA'] / self.hm
            rhs = -G.T @ obj['GA'] + self.Gfh @ beta
            adj = self.solve(rhs, 'transpose', True) - self.hf @ beta
            grads = cm.rt.grouped8.network_backward(model, data, value=obj['GA'], curl=obj['GU'])
            grads = cm.rt.grouped8.network_backward(model, self.fit, value=adj, grads=grads)
            result.update(grads=grads)
        result['solvers'] = list(self.scalar.records)
        self.resource_callback()
        return result

def full_epoch(pipe, model, sampling_geo, epoch):
    """One fresh sequential original sampler/Adam call; no AL term or dual."""
    import math
    c, cm = pipe.c, pipe.cm
    validate_configuration(c, model.K)
    if model.epoch != epoch - 1 or not 1 <= epoch <= 2000:
        raise ValueError('Fresh absolute sequential epoch required')
    points = cm.rt.fluid_points(c, sampling_geo, 6144, c.seed + 100000 + epoch)
    state = pipe.compute(model, points, True)
    lr = c.min_learning_rate + (c.learning_rate-c.min_learning_rate)*.5*(1+math.cos(math.pi*(epoch-1)/1999))
    norms = model.adam(*state['grads'], lr, c.grad_clip)
    if model.epoch != epoch:
        raise RuntimeError('Original Adam commit is not the absolute epoch')
    return state, norms, dict(epoch=epoch, sample_seed=c.seed+100000+epoch,
                             learning_rate=lr, dual_updates=0, multipliers='N/A',
                             rho='N/A', previous_constraint_residual='N/A')
