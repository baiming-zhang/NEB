"""Portable input closure and complete checkpoints without checksum dependencies."""
from pathlib import Path
import ast
import importlib.util
import json
import re
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'experiments/metaball/_backend'


def checkpoint_module():
    sys.path.insert(0, str(BACKEND))
    path = BACKEND / 'exports/canonical_potential/support/atomic_checkpoint.py'
    spec = importlib.util.spec_from_file_location('portable_checkpoint', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('epoch,validations', [(2000, 80), (25, 0)])
def test_complete_restart_bundle_preserves_model_optimizer_and_sampler(tmp_path, epoch, validations):
    module = checkpoint_module()
    arrays = {'epoch': np.asarray(epoch)}
    for layer in range(2):
        for key in ('W', 'mW', 'vW'):
            arrays[f'{key}_{layer}'] = np.full((2, 3, 3), .1, 'f4')
        for key in ('b', 'mb', 'vb'):
            arrays[f'{key}_{layer}'] = np.full((2, 3), .2, 'f4')
    result = module.publish_bundle(
        tmp_path / 'restart', epoch=epoch, K=2, layers=2,
        model_writer=lambda path: np.savez(path, **arrays),
        sampler_state={'next_epoch': epoch + 1, 'seed': 17},
        counts={'optimizer_steps': epoch, 'validations': validations, 'dual_updates': 0},
        timing={'normal_training_seconds': 1., 'state_recovery_seconds': 0.,
                'discarded_attempt_seconds': 0.},
        bindings={'scheduled_validations': 0 if epoch == 25 else 80,
                  'config': {'modes': 2, 'seed': 17}},
        method_state={'method': 'Potential'}, auxiliary_na_reason='Rayleigh objective')
    state = module.verify_bundle(result['pointer'], expected_epoch=epoch)
    assert state['counts']['validations'] == validations
    assert state['sampler_state']['next_epoch'] == epoch + 1
    assert len(state['model']['arrays']) == 13
    model = Path(result['pointer']['bundle_directory']) / 'model.npz'
    with np.load(model) as saved:
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    # Removing optimizer state must fail for a functional reason, without a digest.
    arrays.pop('vW_1')
    np.savez(model, **arrays)
    with pytest.raises(ValueError, match='missing full parameter or Adam'):
        module.verify_bundle(result['pointer'])


def test_configuration_and_model_paths_are_readable_and_complete():
    manifest = json.loads((ROOT / 'checkpoints/index.json').read_text())
    entries = manifest['checkpoints']
    assert len(entries) == 89
    for item in entries:
        for key in ('path', 'metadata', 'config'):
            name = item[key]
            assert not re.search(r'20\d{6}|(?:^|[/_])v\d+(?:[/_.]|$)', name), name
        assert (ROOT / item['config']).is_file()
        assert (ROOT / item['metadata']).is_file()
        metadata = json.loads((ROOT / item['metadata']).read_text())
        assert metadata['config']['seed'] == json.loads((ROOT / item['config']).read_text())['seed']
        assert item['optimizer_state'] is True


def test_nkb_reads_its_saved_geometry_without_external_run_files():
    path = ROOT / 'experiments/metaball/train.py'
    spec = importlib.util.spec_from_file_location('portable_nkb_configuration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((ROOT / 'experiments/metaball/configs/metaball-nkb-k32-seed0.json').read_text())
    base, geometry = module.nkb_geometry(cfg)
    assert geometry.shape == (3, 4)
    assert base['outer_half_extent'] == .49
    flat = dict(base, fixed_training_geometry=geometry.tolist())
    _, alternative = module.nkb_geometry(flat)
    np.testing.assert_array_equal(geometry, alternative)


def test_finite_gauge_protocol_keeps_its_seed_frequencies_and_schedule():
    path = ROOT / 'experiments/metaball/train.py'
    spec = importlib.util.spec_from_file_location('portable_finite_configuration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for K in (32, 128, 256):
        source = ROOT / f'experiments/metaball/configs/metaball-potential-fg-k{K}-seed0-scalar-gauge.json'
        original = json.loads(source.read_text())
        expanded = module.runtime_configuration(original, K)
        assert expanded['seed'] == 0 and expanded['modes'] == K
        assert expanded['siren_first_omega0'] == 160
        assert expanded['points_per_step'] == 6144
        assert expanded['learning_rate'] == original['lr_initial']
        np.testing.assert_array_equal(expanded['fixed_training_geometry'], original['geometry'])
        assert module.runtime_configuration(expanded, K) == expanded


def test_all_positive_mode_configurations_pass_the_training_interface():
    # Compile the actual CPU-only policy function without initializing CUDA.
    source = ROOT / 'experiments/2d/_backend/flow/cases/topology3/learned_basis/positive/scripts/train_hardcoded.py'
    constants = {'SUPPORTED_OBJECTIVES', 'FORBIDDEN_CONFIG_KEYS',
                 'FP32_CONFIG_KEYS', 'JET_ORDER_BY_OBJECTIVE'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if
                (isinstance(node, ast.FunctionDef) and node.name == 'validate_config') or
                (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets))]
    namespace = {'Any': object}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    precisions = set()
    for path in (ROOT / 'experiments/multihole/configs').glob('*.json'):
        config = json.loads(path.read_text())
        if config.get('basis_objective') in ('streamfunction_dirichlet_rayleigh', 'velocity_rayleigh'):
            namespace['validate_config'](config)
            precisions.add(config['master_weight_dtype'])
    assert precisions == {'float32', 'float64'}
