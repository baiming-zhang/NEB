"""Check portable case inputs and saved numerical state without file hashes.

Run from a source checkout. --models opens every downloaded model and Adam
array; --native evaluates representative saved models on four points using
their production backends. Neither option trains models or replaces results.
"""
from pathlib import Path
import argparse
from contextlib import contextmanager
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def numerical_arrays(path):
    # Load each member once. NpzFile's linear filename search is expensive for
    # the per-mode 2D optimizer archives, which contain about 40,000 members.
    source = io.BytesIO(path.read_bytes()) if path.stat().st_size < 128 * 2**20 else path
    class Arrays(dict):
        @property
        def files(self):
            return self.keys()
    with zipfile.ZipFile(source) as archive:
        result = Arrays()
        for name in archive.namelist():
            if name.endswith('.npy'):
                with archive.open(name) as member:
                    result[name[:-4]] = np.lib.format.read_array(member, allow_pickle=False)
        yield result


def state(entry):
    path = ROOT / entry['path']
    if not path.is_file():
        raise FileNotFoundError(f"Download {entry['id']} with neb checkpoints --download --only {entry['id']}")
    with numerical_arrays(path) as archive:
        keys = set(archive.files)
        if 'step' in keys:  # Direct-velocity NKB, including its Adam moments.
            epoch = int(archive['step'])
            layers = len([k for k in keys if re.fullmatch(r'W_\d+', k)])
            for i in range(layers):
                for kind in ('W', 'b'):
                    index = i if kind == 'W' else layers + i
                    value = archive[f'{kind}_{i}']
                    assert archive[f'first_{index}'].shape == value.shape
                    assert archive[f'second_{index}'].shape == value.shape
            assert archive[f'b_{layers-1}'].shape == (entry['K'] * (3 if entry['case'] == 'metaball' else 2),)
            assert epoch == entry['epoch']
        elif 'epoch' in keys:  # Mode-stacked 3D SIREN and complete Adam state.
            assert int(archive['epoch']) == entry['epoch']
            layers = len([k for k in keys if re.fullmatch(r'W_\d+', k)])
            for i in range(layers):
                for kind, moments in (('W', ('mW', 'vW')), ('b', ('mb', 'vb'))):
                    value = archive[f'{kind}_{i}']
                    assert value.shape[0] == entry['K']
                    for moment in moments:
                        assert archive[f'{moment}_{i}'].shape == value.shape
        else:  # Independent 2D SIRENs, each with its own optimizer counter.
            cfg = read(ROOT / entry['config'])
            layers = cfg['hidden_layers'] + 1
            for mode in range(entry['K']):
                prefix = f'm{mode:03d}.'
                for layer in range(layers):
                    label = f'l{layer}' if layer < layers - 1 else 'output'
                    for offset, kind in enumerate(('weight', 'bias')):
                        value = archive[f'{prefix}{label}.{kind}']
                        adam = f'{prefix}adam.p{2*layer+offset:02d}'
                        assert archive[adam + '.exp_avg'].shape == value.shape
                        assert archive[adam + '.exp_avg_sq'].shape == value.shape
                        assert int(archive[adam + '.step']) == entry['epoch']
        for key in archive.files:
            value = archive[key]
            if value.dtype.kind in 'fci':
                assert np.isfinite(value).all(), (entry['id'], key)
    for auxiliary in entry.get('auxiliary', []):
        with np.load(ROOT / auxiliary['path'], allow_pickle=False) as archive:
            assert archive.files
            for key in archive.files:
                value = archive[key]
                if value.dtype.kind in 'fci':
                    assert np.isfinite(value).all(), (entry['id'], key)


def native(entry, output):
    import cupy as cp
    cp.get_default_memory_pool().set_limit(size=1024 * 2**20)
    path = ROOT / entry['path']
    cfg = read(ROOT / entry['config'])
    c = SimpleNamespace(**cfg)
    if entry['case'] == 'metaball':
        base = ROOT / 'experiments/metaball'
        sys.path[:0] = [str(base), str(base / '_backend')]
        os.environ['NEB_DATA_DIR'] = str(output / 'data')
        train = load('train', base / 'train.py')
        cfg = train.runtime_configuration(cfg, entry['K'])
        c = SimpleNamespace(**cfg)
        if entry['method'] in ('NKB', 'Velocity'):
            evaluate = load('checkpoint_evaluate', base / 'evaluate.py')
            args = SimpleNamespace(method=entry['method'], K=entry['K'], checkpoint=str(path))
            fields, close = evaluate.field_provider(args, c, output)
            try:
                values = fields(np.array([[.5,.5,.5], [.3,.4,.5], [.6,.4,.3], [.7,.6,.5]]))
                assert values.shape == (4, 3, entry['K'])
                assert np.isfinite(values).all()
            finally:
                close()
        else:
            _, cm, _ = train.native_context(cfg, entry['K'], output / 'runtime', output / 'unused_samples.npz')
            model = cm.train.Model(c, entry['K'], c.seed)
            model.load(path)
            assert model.epoch == entry['epoch']
            points = np.array([[.5,.5,.5], [.3,.4,.5], [.6,.4,.3], [.7,.6,.5]])
            data = cm.rt.NeuralPoints(points, cp.asarray(c.fixed_training_geometry, 'f8'), c, 'f4')
            result = cm.rt.grouped8.network_fields(model, data, 'u')['u']
            assert result.shape == (12, entry['K'])
            assert bool(cp.isfinite(result).all())
            block = cm.train.block_model(model, 0, 8)
            source = cm.train.source_values(block, data)
            full = cm.rt.grouped8.network_fields(model, data, 'a')['a'][:, :8]
            cp.testing.assert_allclose(source, full, rtol=3e-5, atol=3e-6)
    elif entry['method'] == 'NKB':
        base = ROOT / 'experiments/2d/nkb'
        sys.path.insert(0, str(base))
        core = load('core', base / 'core.py')
        domains = load('domains', base / 'domains.py')
        domain = domains.Domain(entry['case'], cp)
        model = core.Model(cp, cfg['input_dim'], cfg['dim'], cfg['K'], cfg['seed'], cfg['width'], cfg['hidden_layers'])
        model.load(path)
        f, j = domain.encode(cp.asarray([[2., .2], [3., .3], [4., .4], [5., .5]], dtype='f4'))
        value, jac, _ = model.forward(f, j)
        assert model.step == entry['epoch'] and value.shape == (4, entry['K'], 2)
        assert bool(cp.isfinite(value).all()) and bool(cp.isfinite(jac).all())
    else:
        base = ROOT / 'experiments/2d/_backend'
        if entry['case'] == 'periodic':
            source = base / 'periodic_core/vortex_merger/src'
            sys.path.insert(0, str(source))
            from eigenfluid.gen.vortex_merger2d import hardcoded_backend as be
        else:
            top = base / 'flow/cases/topology3/learned_basis'
            sys.path[:0] = [str(top / 'positive/src'), str(top / 'harmonic/scripts')]
            from neb2d.basisnet.metaball2d import hardcoded_backend as be
        c = be.cfg_namespace(cfg)
        with numerical_arrays(path) as archive:
            model = be.HardcodedModel(c, {key: archive[key] for key in archive.files})
        assert model.modes[0].weights[0].dtype == np.dtype(cfg.get('master_weight_dtype', 'float32'))
        points = np.asarray([[2., .2], [3., .3], [4., .4], [5., .5]], dtype='f4')
        if entry['method'] == 'Harmonic':
            from harmonic_common import learned_harmonic_fields
            fields = learned_harmonic_fields(model, c, points, False)
        else:
            fields = model.forward_stream_jet1(c, points, 0., 0., False)
        modes = 0
        for _, value, _ in fields:
            assert bool(cp.isfinite(value.value).all()) and bool(cp.isfinite(value.grad).all())
            modes += 1
        assert modes == entry['K']
    cp.cuda.Stream.null.synchronize()
    print(f"{entry['id']}: saved model executed on four points", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--models', action='store_true', help='open all downloaded model/optimizer arrays')
    parser.add_argument('--native', action='store_true', help='execute one saved model per case/method/lift using native backends')
    parser.add_argument('--all-models', action='store_true', help='with --native, execute every selected checkpoint')
    parser.add_argument('--output', type=Path, default=Path('outputs/reproduction_check'))
    parser.add_argument('--only', nargs='+', help='restrict the check to these checkpoint IDs')
    parser.add_argument('--execute-model', help=argparse.SUPPRESS)
    args = parser.parse_args()
    entries = read(ROOT / 'checkpoints/index.json')['checkpoints']
    if args.only:
        unknown = set(args.only) - {item['id'] for item in entries}
        if unknown:
            parser.error(f'Unknown checkpoint IDs: {sorted(unknown)}')
        entries = [item for item in entries if item['id'] in args.only]
    if args.execute_model:
        entry = next(x for x in entries if x['id'] == args.execute_model)
        args.output.mkdir(parents=True, exist_ok=True)
        native(entry, args.output.resolve())
        return
    for item in entries:
        for key in ('config', 'metadata'):
            path = ROOT / item[key]
            assert path.resolve().is_relative_to(ROOT) and path.is_file(), path
        meta, cfg = read(ROOT / item['metadata']), read(ROOT / item['config'])
        assert meta['config'] == cfg and meta['epoch'] == item['epoch']
        if args.models:
            state(item)
            print(f"{item['id']}: model and optimizer state available", flush=True)
    print(f"Configurations and metadata available for {len(entries)} saved models.", flush=True)
    if args.native:
        selected = {}
        for item in sorted(entries, key=lambda x: (x['K'], x['id'])):
            lift = 'scaled' if item['id'].endswith('-scaled') else 'raw'
            selected.setdefault((item['case'], item['method'], lift), item)
        for item in entries if args.all_models else selected.values():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--execute-model', item['id'],
                            '--output', str(args.output / item['id'])], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
