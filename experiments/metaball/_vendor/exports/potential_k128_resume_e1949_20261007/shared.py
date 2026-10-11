




import ast
import hashlib
import importlib.util
import json
import math
import os
from _neb_paths import Path
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASE = ROOT / 'ICLR2027_campaign/cases/metaball'
OLD = ROOT / 'exports/coulomb_multiplier_20261006'
IMPROVEMENT = ROOT / 'exports/coulomb_multiplier_improvement_20261006'
RUNTIME = HERE / 'runtime_vel_shared.py'
SAMPLER = ROOT / '3d_metaball_reproduction_latest/cases/3D_Metaball/scripts/train_validate.py'
SOURCE = CASE / 'results/canonical_poisson_residual_fit_v1/attempt_001/samples.npz'
KS = (32, 64, 128)
FIXED = dict(seed=20260823, epochs=2000, points_per_step=6144,
             validation_points=4096, validation_geometries=8, validate_every=25,
             checkpoint_every=1, input_features=15, hidden=128, hidden_layers=5,
             siren_first_omega0=40.0, siren_hidden_omega0=36.0,
             learning_rate=2.8e-5, min_learning_rate=1e-7, lr_scheduler='cosine',
             grad_clip=8.0, dtype='float32', mode_batch=16, point_chunk=1024,
             sample_seed_rule='seed+100000+epoch',
             validation_token_rule='10000000+index',
             validation_seed_rule='seed+9000000+index')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def configuration(K):

    if K not in KS:
        raise ValueError('Explicit supported K required: 32, 64, or 128')
    protocol = read(HERE / 'protocol.json')
    if protocol.get('Ks') != list(KS):
        raise ValueError('The declared three-run scope changed')
    setting = protocol['run_settings'][str(K)]
    config = dict(setting['vel_config'])
    if config.get('modes') != K:
        raise ValueError('Run configuration mode count must be explicit')
    for name, value in FIXED.items():
        if config.get(name) != value:
            raise ValueError('Historical velocity configuration mismatch: ' + name)
    historical = read(setting['vel_config_source'])
    for name, value in historical.items():
        expected = K if name == 'modes' else value
        if config.get(name) != expected:
            raise ValueError('Unmatched historical velocity configuration: ' + name)
    return protocol, config, setting


def initial_arrays(K):

    import numpy as np
    _, config, setting = configuration(K)
    generators = [np.random.Generator(np.random.PCG64(config['seed'] + 7919 * (i + 1)))
                  for i in range(K)]
    shapes = [15] + [config['hidden']] * config['hidden_layers'] + [3]
    weights, biases = [], []
    checkpoint = Path(setting['initial_checkpoint'])
    with np.load(checkpoint, allow_pickle=False) as z:
        if int(z['epoch']) != 0:
            raise ValueError('Historical initialization is not epoch zero')
        for j, (inp, out) in enumerate(zip(shapes[:-1], shapes[1:])):
            bound = 1 / inp if j == 0 else math.sqrt(6 / inp) / config['siren_hidden_omega0']
            rebuilt_w = np.stack([r.uniform(-bound, bound, (out, inp)).astype('f4') for r in generators])
            rebuilt_b = np.stack([r.uniform(-bound, bound, out).astype('f4') for r in generators])
            w, b = z[f'W_{j}'][:K].copy(), z[f'b_{j}'][:K].copy()
            if w.shape != rebuilt_w.shape or b.shape != rebuilt_b.shape:
                raise ValueError('Incomplete historical initialization prefix')
            if not np.array_equal(w, rebuilt_w) or not np.array_equal(b, rebuilt_b):
                raise ValueError('Historical initialization differs from original PCG64 reconstruction')
            for name in ('mW', 'vW', 'mb', 'vb'):
                if np.any(z[f'{name}_{j}'][:K] != 0):
                    raise ValueError('Historical epoch-zero optimizer state is not zero')
            weights.append(w)
            biases.append(b)
    digest = hashlib.sha256(b''.join(v.tobytes() for v in weights + biases)).hexdigest()
    return weights, biases, dict(source=str(checkpoint), source_sha256=sha(checkpoint),
        prefix_modes=K, epoch=0, seed=config['seed'], parameter_sha256=digest,
        independent_PCG64_reconstruction_bitwise=True, all_six_layers=True,
        optimizer_moments_zero=True)


def geometry_sampler():

    import numpy as np
    names = {'geometry_from_token', 'overlap_graph_connected', 'sphere_directions',
             'spheres_inside_rounded_cube'}
    tree = ast.parse(SAMPLER.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError('Historical geometry sampler interface changed')
    namespace = dict(np=np, math=math, Config=object)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SAMPLER), 'exec'), namespace)
    namespace['SPHERE_DIRECTIONS'] = namespace['sphere_directions']()
    return namespace['geometry_from_token']


def install_velocity_geometry(cm):






    cp = cm.cp
    namespace = cm.train.Model.forward.__globals__
    geometry, index = namespace['jet_geometry'], namespace['INDEX']
    Jet1, features1 = namespace['Jet1'], namespace['features1']
    first_indices = [index[tuple(int(k == j) for k in range(3))] for j in range(3)]
    original = cm.rt.NeuralPoints

    class MatchedNeuralPoints:
        def __init__(self, points, geo, c, dtype='f4'):
            self.x = cp.asarray(points, dtype=dtype)
            self.n = len(self.x)
            feature_value=cp.empty((self.n,15),dtype=dtype)
            feature_gradient=cp.empty((self.n,15,3),dtype=dtype)
            value = cp.empty((self.n, 1), dtype=dtype)
            gradient = cp.empty((self.n, 1, 3), dtype=dtype)
            physical_geometry = cp.asarray(geo, dtype='f8')
            for start in range(0, self.n, 1024):
                stop = min(start + 1024, self.n)
                features=features1(self.x[start:stop],cp.asarray(geo,dtype=dtype),c)
                feature_value[start:stop]=features.value
                feature_gradient[start:stop]=features.grad
                jets = geometry(self.x[start:stop].astype('f8'), physical_geometry, c)
                coefficients = jets[1]
                value[start:stop] = coefficients[:, 0, None].astype(dtype)
                gradient[start:stop] = cp.stack([coefficients[:, j] for j in first_indices], -1)[:, None, :].astype(dtype)
                del jets, coefficients, features
            self.features=Jet1(feature_value,feature_gradient)
            self.envelope = Jet1(value, gradient)

        def part(self, start, stop):
            return types.SimpleNamespace(n=stop-start, x=self.x[start:stop],
                features=Jet1(self.features.value[start:stop], self.features.grad[start:stop]),
                envelope=Jet1(self.envelope.value[start:stop], self.envelope.grad[start:stop]))

    cm.rt.NeuralPoints = MatchedNeuralPoints
    cm._matched_geometry_restore = lambda: setattr(cm.rt, 'NeuralPoints', original)


def load_shared(stage, K, runtime_module=None):

    _, config, setting = configuration(K)
    run = HERE / f'k{K}'
    os.environ['CUPY_CACHE_DIR'] = str(run / 'cupy_cache')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    source = (OLD / 'run.py').read_text(encoding='utf-8')
    replacements = {
        'HERE=Path(__file__).resolve().parent': 'HERE=Path(' + repr(str(run)) + ')',
        "DATA=Path('_data/metaball/coulomb_multiplier_20261006')": "DATA=HERE/'data'",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('Audited output-location interface changed')
        source = source.replace(old, new)

    run.mkdir(parents=True, exist_ok=True)
    argv = sys.argv[:]
    try:
        sys.argv = [str(OLD / 'run.py'), 'qualification']
        cm = types.ModuleType(f'velocity_matched_potential_k{K}')
        cm.__file__ = str(OLD / 'run.py')
        exec(compile(source, str(OLD / 'run.py'), 'exec'), cm.__dict__)
    finally:
        sys.argv = argv
    cm.STAGE, cm.DEST = stage, run / stage
    cm.DEST.mkdir(parents=True, exist_ok=True)
    cm._matched_K, cm._matched_config = K, config
    cm._matched_setting = setting
    cm.geometry_from_token = geometry_sampler()
    install_velocity_geometry(cm)
    original_Potential_objective=cm.train.bounded_potential_objective
    def physical_Potential_objective(a,u,n):


        return original_Potential_objective(cm.cp.asarray(a,dtype='f8'),cm.cp.asarray(u,dtype='f8'),n)
    cm.train.bounded_potential_objective=physical_Potential_objective


    tree=ast.parse(source)
    metrics_node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='metrics')
    conversion=ast.parse("f={key:cp.asarray(value,dtype='f8') for key,value in f.items()}").body[0]
    metrics_node.body.insert(1,conversion)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[metrics_node],type_ignores=[])),str(OLD/'run.py'),'exec'),cm.__dict__)

    def inputs():
        c = types.SimpleNamespace(**config)
        geo = cm.cp.asarray(config['fixed_training_geometry'], 'f8')
        with cm.np.load(SOURCE, allow_pickle=False) as z:
            x = z['fit'].copy()
        q = cm.rt.fluid_points(c, geo, 16384, 919000)
        if len(x) != 103244 or len(q) != 16384:
            raise ValueError('The original complete constraint/independent point scope changed')
        return c, geo, x, q

    cm.inputs = inputs
    implementation = dict(name='New velocity-matched nativeCuPy Jet1 literal', point_chunk=1024,
        mode_group=16, historical_config_point_chunk=config['point_chunk'],
        historical_config_mode_batch=config['mode_batch'], ANN='FP32',
        contractions_and_objective='FP64', execution_batch_differs_from_vel=False,
        geometry_envelope='Same native FP64 Taylor value/gradient as velocity FieldData, cast to ANN dtype',
        parameter_gradient_accumulation='FP32 sequential1024-point chunks')


    chosen = Path(runtime_module) if runtime_module else RUNTIME
    spec = importlib.util.spec_from_file_location('matched_runtime', chosen)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    implementation.update(module.install(cm) if runtime_module else module.install(cm, variant='literal'))
    implementation.update(equivalence='Requires this new K-specific actual preflight; earlier seed0 qualification is not reused',
        qualification_receipt=str(run / 'preflight/complete.json'),
        current_configuration_qualified=False)
    spec = importlib.util.spec_from_file_location('matched_span_penalty', HERE / 'span_added_only.py')
    span = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(span)
    return cm, span, implementation
