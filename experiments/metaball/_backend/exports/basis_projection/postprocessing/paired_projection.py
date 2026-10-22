from _neb_paths import Path
import argparse, contextlib, csv, gc, importlib.util, json, os, sys, tempfile, time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
CAM = HERE.parent
ROOT = CAM.parents[1]
POINTS = 1599754
WEIGHT = 1 / 128 ** 3
BLOCK = 2048

def read(p):
    return json.loads(Path(p).read_text('utf-8-sig'))

def guard():
    r = psutil.Process().memory_info().rss
    h = psutil.virtual_memory().available
    if r > 8 * 2 ** 30 or h < 9 * 2 ** 30:
        raise RuntimeError(f'CPU resource limit RSS={r},hostfree={h}')
    return dict(unix=time.time(), RSS=r, hostfree=h)

def write(p, j):
    q = Path(p).with_name(Path(p).name + '.tmp')
    with q.open('x', encoding='utf-8') as f:
        json.dump(j, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    deadline = time.monotonic() + 10.0
    while True:
        try:
            os.replace(q, p)
            return
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)

def open_npy(p, shape=None, dtype=None):
    f = Path(p).open('rb')
    try:
        version = np.lib.format.read_magic(f)
        actual, order, dt = np.lib.format._read_array_header(f, version)
        if order or dt.hasobject:
            raise ValueError('Only original C-order numeric NPY accepted')
        if shape is not None and tuple(actual) != tuple(shape):
            raise ValueError('Full NPY dimensions differ')
        if dtype is not None and dt != np.dtype(dtype):
            raise ValueError('Original NPY precision differs')
        if Path(p).stat().st_size != f.tell() + int(np.prod(actual)) * dt.itemsize:
            raise ValueError('Incomplete or appended NPY')
        return (f, actual, dt)
    except BaseException:
        f.close()
        raise

def targets(paths, points=POINTS, block=BLOCK):
    with contextlib.ExitStack() as stack:
        streams = [stack.enter_context(open_npy(p, (points, 3), 'f8')[0]) for p in paths]
        for lo in range(0, points, block):
            count = min(block, points - lo)
            values = []
            for f in streams:
                a = np.fromfile(f, dtype='f8', count=count * 3)
                if a.size != count * 3 or not np.isfinite(a).all():
                    raise ValueError('Invalid complete target block')
                values.append(a)
            yield (lo, np.stack(values, axis=1))

def basis(spec, K, points=POINTS, block=BLOCK):
    kind = spec['kind']
    if kind == 'saved_npy':
        stream, shape, dtype = open_npy(spec['path'], (points, 3, K), spec['dtype'])
        with stream:
            for lo in range(0, points, block):
                count = min(block, points - lo)
                v = np.fromfile(stream, dtype=dtype, count=count * 3 * K)
                if v.size != count * 3 * K:
                    raise ValueError('Incomplete saved original basis')
                yield (lo, v.reshape(count * 3, K).astype('f8', copy=False))
    elif kind == 'mapped_Laplacian_cells':
        ids = np.load(spec['cell_ids'], allow_pickle=False)
        P = np.load(spec['Piola'], allow_pickle=False)
        if ids.shape != (points,) or P.shape != (points, 3, 3):
            raise ValueError('Original allpoint Piola dimensions required')
        for lo in range(0, points, block):
            count = min(block, points - lo)
            out = np.empty((count, 3, K), dtype='f8')
            next_column = 0
            for item in spec['blocks']:
                first, stop = (item['first'], min(item['stop'], K))
                if first >= K:
                    break
                if first != next_column:
                    raise ValueError('Numerical positive spectrum must retain contiguous original order')
                mapped = np.load(item['path'], mmap_mode='r', allow_pickle=False)
                try:
                    if mapped.dtype != np.dtype('f8') or mapped.shape[1:] != (3, item['stop'] - item['first']):
                        raise ValueError('Original full cell block schema differs')
                    selected = np.array(mapped[ids[lo:lo + count], :, :stop - first], copy=True)
                finally:
                    mapped._mmap.close()
                    del mapped
                out[:, :, first:stop] = np.einsum('nij,njk->nik', P[lo:lo + count], selected)
                del selected
                next_column = stop
            if next_column != K:
                raise ValueError('Missing sameK numerical modes')
            yield (lo, out.reshape(count * 3, K))
    elif kind == 'original_NKB_numpy':
        original = ROOT / 'flow/nkb'
        sys.path.insert(0, str(original))
        from core import Model
        from domains import Domain
        run = Path(spec['run'])
        cfg = read(run / 'config.json')
        training = read(run / 'training_complete.json')
        if cfg['K'] != K or training['epoch'] != 2000 or (not training['all_scheduled_validations']):
            raise ValueError('Independent sameK frozen full2000 NKB required')
        model = Model(np, cfg['input_dim'], cfg['dim'], K, cfg['seed'], cfg['width'], cfg['hidden_layers'], dtype='float64')
        model.load(run / 'checkpoint_e2000.npz')
        if model.step != 2000:
            raise ValueError('NKB checkpoint not actual full2000')
        domain = Domain('metaball', np)
        flat = np.load(spec['flat'], allow_pickle=False)
        coordinate = np.load(spec['coordinates'], allow_pickle=False)
        if flat.shape != (points,):
            raise ValueError('Original fullfluid order required')
        for lo in range(0, points, block):
            ii = flat[lo:lo + block]
            p = np.stack((coordinate[ii % 128], coordinate[ii // 128 % 128], coordinate[ii // 16384]), axis=1).astype('f8')
            f, _ = domain.encode(p)
            v, _, _ = model.forward(f)
            yield (lo, np.ascontiguousarray(v.transpose(0, 2, 1)).reshape(-1, K))
        del model
        gc.collect()
    else:
        raise ValueError('Unknown original basis provider')

def project(mass, spec, paths, K, points=POINTS, weight=WEIGHT, block=BLOCK, observe=lambda phase, lo: None):
    if mass.shape != (K, K) or mass.dtype != np.dtype('f8') or (not np.isfinite(mass).all()):
        raise ValueError('Original complete FP64 sameK mass required')
    n = len(paths)
    loads = np.zeros((K, n), dtype='f8')
    norms = np.zeros(n, dtype='f8')
    covered = 0
    for (lo, B), (other, target) in zip(basis(spec, K, points, block), targets(paths, points, block), strict=True):
        if lo != other or lo != covered or B.shape != (target.shape[0], K) or (not np.isfinite(B).all()):
            raise ValueError('Full basis and paired targets disagree')
        loads += weight * (B.T @ target)
        norms += weight * np.sum(target * target, axis=0)
        covered += len(B) // 3
        observe('all20_RHS', covered)
    if covered != points or not np.all(norms > 0):
        raise ValueError('Incomplete full domain or zero target norm')
    coefficients = np.linalg.solve(mass, loads)
    normal = np.linalg.norm(mass @ coefficients - loads, axis=0) / np.linalg.norm(loads, axis=0)
    if not np.isfinite(coefficients).all() or not np.all(normal < 1e-08):
        raise ValueError('Original fullmass normal-residual gate failed')
    gram_square = norms - 2 * np.sum(coefficients * loads, axis=0) + np.sum(coefficients * (mass @ coefficients), axis=0)
    direct = np.zeros(n, dtype='f8')
    covered = 0
    for (lo, B), (other, target) in zip(basis(spec, K, points, block), targets(paths, points, block), strict=True):
        if lo != other or lo != covered or (not np.isfinite(B).all()):
            raise ValueError('Direct pass domain mismatch')
        difference = B @ coefficients - target
        direct += weight * np.sum(difference * difference, axis=0)
        covered += len(B) // 3
        observe('all20_direct_residual_second_pass', covered)
    if covered != points or np.any(gram_square < 0) or (not np.isfinite(direct).all()):
        raise ValueError('Invalid fullpoint residual')
    errors = np.sqrt(direct / norms)
    identity = np.sqrt(gram_square / norms)
    if not np.all(np.abs(errors - identity) < 1e-08):
        raise ValueError('Original Gram/direct identity tolerance failed')
    return dict(loads=loads, target_norms=norms, coefficients=coefficients, normal_residuals=normal, Gram_identity_errors=identity, direct_relative_L2=errors, direct_squared_norms=direct)

def cpu_check():
    rng = np.random.default_rng(8675309)
    N = 23
    K = 5
    weight = 0.125
    B = rng.normal(size=(N, 3, K)).astype('f4')
    M = B.reshape(-1, K).astype('f8').T @ B.reshape(-1, K).astype('f8') * weight
    with tempfile.TemporaryDirectory(prefix='metaball20_projection_interface_') as tmp:
        path = Path(tmp)
        np.save(path / 'basis.npy', B)
        targets0 = rng.normal(size=(N, 3, 20))
        paths = []
        for i in range(20):
            p = path / f'target{i}.npy'
            np.save(p, targets0[:, :, i])
            paths.append(p)
        result = project(M, dict(kind='saved_npy', path=str(path / 'basis.npy'), dtype='f4'), paths, K, N, weight, 7)
        dense = B.reshape(-1, K).astype('f8')
        Y = targets0.reshape(-1, 20)
        C = np.linalg.solve(M, weight * dense.T @ Y)
        exact = np.sqrt(np.sum((dense @ C - Y) ** 2, axis=0) / np.sum(Y * Y, axis=0))
        assert np.allclose(result['direct_relative_L2'], exact, rtol=1e-13, atol=1e-13)
        try:
            project(M * 1.5, dict(kind='saved_npy', path=str(path / 'basis.npy'), dtype='f4'), paths, K, N, weight, 7)
        except ValueError:
            pass
        else:
            raise AssertionError('Inconsistent saved mass accepted')
    assert 'cupy' not in sys.modules
    return dict(status='actual_CPU_twenty_paired_fullmass_two_pass_interface_fixture_passed', qualified_CPU_interface_only=True, dense_fullmass_equivalent_for_nonorthogonal_FP32_basis_all20=True, mismatched_mass_negative_rejected_by_actual_second_pass=True, original_fp64_weighted_fullmass_no_ridge_or_truncation=True, sample_statistics='20 initialfields; mean and sampleSD ddof1', GPU_calls=0)

def actual(args):
    entered = time.perf_counter()
    guard()
    inventory = read(args.assets)
    review = read(args.interface_review)
    case_review = read(args.case_review)
    if review.get('qualified_CPU_interface_only') is not True:
        raise ValueError('Exact independent CPU source and actual asset inventory review required')
    if case_review.get('qualified_case_assets_only') is not True or case_review['n'] != 20 or case_review.get('qualified_method_error_science') is not False:
        raise ValueError('Actual independent twenty target review required')
    cases = case_review['complete']
    manifest = read(cases['path'])
    if manifest['n'] != 20:
        raise ValueError('Published20targets changed')
    paths = [r['path'] for r in manifest['records']]
    for record in manifest['records']:
        pass
    entry = next((r for r in inventory['entries'] if r['method'] == args.method and r['K'] == args.K))
    destination = Path(entry['output']).resolve()
    if args.output is not None and args.output.resolve() != destination:
        raise ValueError('Exact new reviewed output required')
    destination.mkdir(parents=True, exist_ok=False)
    resources = []

    def observe(phase, points):
        r = guard()
        resources.append(r)
        with (destination / 'actual_CPU_resources.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(r) + '\n')
        write(destination / 'progress.json', dict(phase=phase, points=points, total_points=POINTS, K=args.K, method=args.method, actual_seconds=time.perf_counter() - entered))
    try:
        observe('original_fullmass_load', 0)
        source = entry['mass']
        M = np.load(source['path'], allow_pickle=False)
        if isinstance(M, np.lib.npyio.NpzFile):
            with M:
                mass = np.array(M[source['key']], dtype='f8')
        else:
            mass = np.array(M, dtype='f8')
        if mass.shape != (args.K, args.K):
            if args.method != 'Laplacian' or not source.get('original_ordered_spectrum_prefix'):
                raise ValueError('Only original ordered numerical spectrum permits sameK prefix')
            mass = mass[:args.K, :args.K].copy()
        result = project(mass, entry['basis'], paths, args.K, observe=observe)
        archive = destination / 'projection_all20.npz'
        with archive.open('xb') as f:
            np.savez(f, **result, mass=mass)
            f.flush()
            os.fsync(f.fileno())
        csvpath = destination / 'per_case_errors.csv'
        with csvpath.open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=('method', 'K', 'case', 'relative_L2'))
            writer.writeheader()
            for i, error in enumerate(result['direct_relative_L2']):
                writer.writerow(dict(method=args.method, K=args.K, case=i, relative_L2=float(error)))
            f.flush()
            os.fsync(f.fileno())
        observe('complete_direct_residual', POINTS)
        complete = dict(status='actual_CPU_twenty_paired_fullmass_direct_projection_complete', qualified_pending_independent_numerical_review=True, method=args.method, K=args.K, n=20, all_points=POINTS, all_modes=args.K, full_mass_original_source=source, same20case_complete=cases, frozen_basis_not_case_target_trained=True, basis_attribution=entry['basis_attribution'], basis_provider=entry['basis'], two_complete_passes_and_direct_residual_required=True, relative_L2_mean=float(np.mean(result['direct_relative_L2'])), relative_L2_sampleSD=float(np.std(result['direct_relative_L2'], ddof=1)), actual_CPU_seconds=time.perf_counter() - entered, resources=dict(RSS_max=max((r['RSS'] for r in resources)), hostfree_min=min((r['hostfree'] for r in resources))), GPU_calls=0)
        write(destination / 'complete.json', complete)
        return complete
    except BaseException:
        import traceback
        write(destination / 'failure.json', dict(error=traceback.format_exc(), actual_CPU_seconds=time.perf_counter() - entered))
        raise
