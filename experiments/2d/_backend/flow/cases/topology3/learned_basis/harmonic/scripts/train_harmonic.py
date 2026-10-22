"""Train hard-Dirichlet neural harmonic extensions with native CuPy Jet1."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
import cupy as cp
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
METABALL = ROOT.parent
POSITIVE = METABALL / 'positive'
sys.path.insert(0, str(POSITIVE / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cupy_ascii_include import install_ascii_include_alias
install_ascii_include_alias(cp)
from neb2d.basisnet.metaball2d.geometry import get_obstacles
from neb2d.basisnet.metaball2d.hardcoded_backend import HardcodedModel, Jet1, cfg_namespace, fluid_points
from harmonic_common import learned_harmonic_fields

def initialize_arrays(cfg) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(int(cfg.seed)))
    arrays: dict[str, np.ndarray] = {}
    for mode in range(int(cfg.modes)):
        prefix = f'm{mode:03d}'
        names: list[str] = []
        fan_in = int(cfg.input_features)
        for layer in range(int(cfg.hidden_layers)):
            bound = 1.0 / fan_in if layer == 0 else math.sqrt(6.0 / fan_in) / float(cfg.siren_hidden_omega0)
            bias_bound = 1.0 / math.sqrt(fan_in)
            wn, bn = (f'{prefix}.l{layer}.weight', f'{prefix}.l{layer}.bias')
            arrays[wn] = rng.uniform(-bound, bound, (int(cfg.hidden), fan_in)).astype(np.float32)
            arrays[bn] = rng.uniform(-bias_bound, bias_bound, int(cfg.hidden)).astype(np.float32)
            names.extend((wn, bn))
            fan_in = int(cfg.hidden)
        bound = math.sqrt(6.0 / fan_in) / float(cfg.siren_hidden_omega0)
        wn, bn = (f'{prefix}.output.weight', f'{prefix}.output.bias')
        arrays[wn] = rng.uniform(-bound, bound, (1, fan_in)).astype(np.float32)
        arrays[bn] = rng.uniform(-1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in), 1).astype(np.float32)
        names.extend((wn, bn))
        for index, name in enumerate(names):
            stem = f'{prefix}.adam.p{index:02d}'
            arrays[f'{stem}.exp_avg'] = np.zeros_like(arrays[name])
            arrays[f'{stem}.exp_avg_sq'] = np.zeros_like(arrays[name])
            arrays[f'{stem}.step'] = np.asarray(0, dtype=np.int64)
    return arrays

def save_checkpoint(path: Path, model: HardcodedModel, config: dict, epoch: int, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **model.arrays())
    os.replace(temporary, path)
    metadata = {'format': 'metaball2d_learned_harmonic_dirichlet_jet1_v1', 'epoch': int(epoch), 'config': config, 'training_uses_pytorch': False, 'automatic_differentiation_used': False, 'finite_difference_used': False, 'spatial_derivatives': 'analytic Jet1 value-gradient in physical x,y', 'parameter_backward': 'handwritten reverse Jet1', 'objective': 'mean(|grad(ell_a + E_g f_theta,a)|^2), no denominator', 'latest_validation': history[-1] if history else None}
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

def load_checkpoint(path: Path):
    metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return (metadata, arrays)

def evaluate_energy(model: HardcodedModel, cfg, points) -> tuple[np.ndarray, float]:
    fields = learned_harmonic_fields(model, cfg, points, False)
    energies = cp.stack([cp.mean(cp.sum(field.grad[:, 0] ** 2, axis=1)) for _, field, _ in fields])
    finite = bool(cp.all(cp.isfinite(energies)).item())
    if not finite:
        raise FloatingPointError('non-finite validation Dirichlet energy')
    values = cp.asnumpy(energies)
    return (values, float(np.mean(values)))

def validate_config(config: dict) -> None:
    if config.get('basis_objective') != 'harmonic_dirichlet_energy':
        raise ValueError('this trainer accepts only the harmonic Dirichlet energy')
    if config.get('geometry_mode') != 'fixed_metaball_topology':
        raise ValueError('only the fixed topology-family geometry is supported')
    if float(config.get('outer_tau', -1)) != 0.0:
        raise ValueError('the matched K1024 study fixes outer_tau=0')
    if config.get('envelope_mode') != 'all_boundary_simple_zero':
        raise ValueError('the correction needs one simple zero on every boundary')
    if config.get('gauge_fixing') != 'outer_trace_zero':
        raise ValueError('the outer trace must fix the additive constant')
    expected = len(get_obstacles(config['subcase']))
    if int(config['modes']) != expected or int(config['harmonic_dimension']) != expected:
        raise ValueError('one learned harmonic output is required per obstacle')
    if int(config.get('checkpoint_every', 0)) != 200:
        raise ValueError('checkpoints must be archived every 200 epochs')
    if int(config.get('spatial_derivative_order', -1)) != 1:
        raise ValueError('Dirichlet training must use the minimal analytic Jet1 path')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    if 'torch' in sys.modules:
        raise RuntimeError('PyTorch is forbidden in the native-CuPy trainer')
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('exactly one visible GPU is required')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    validate_config(config)
    if args.epochs is not None:
        config['epochs'] = int(args.epochs)
    cfg = cfg_namespace(config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'checkpoints').mkdir(exist_ok=True)
    (args.run_dir / 'data').mkdir(exist_ok=True)
    if args.resume:
        metadata, arrays = load_checkpoint(args.resume)
        start_epoch = int(metadata['epoch']) + 1
    else:
        arrays = initialize_arrays(cfg)
        start_epoch = 1
        initial = args.run_dir / 'checkpoints' / 'harmonic_epoch0000.npz'
        model0 = HardcodedModel(cfg, arrays)
        save_checkpoint(initial, model0, config, 0, [])
        del model0
    model = HardcodedModel(cfg, arrays)
    validation_points = fluid_points(cfg, int(cfg.validation_points), 0.0, 0.0, int(cfg.seed) + 900000)
    history_path = args.run_dir / 'data' / 'training_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    started = time.perf_counter()
    final_epoch = int(cfg.epochs)
    for epoch in range(start_epoch, final_epoch + 1):
        progress = (epoch - 1) / max(final_epoch - 1, 1)
        cfg.learning_rate = float(cfg.min_learning_rate) + 0.5 * (float(config['learning_rate']) - float(cfg.min_learning_rate)) * (1.0 + math.cos(math.pi * progress))
        points = fluid_points(cfg, int(cfg.points_per_epoch), 0.0, 0.0, int(cfg.seed) + 1009 * epoch)
        fields = learned_harmonic_fields(model, cfg, points, True)
        modes = [item[0] for item in fields]
        caches = [item[2] for item in fields]
        energies_device = cp.stack([cp.mean(cp.sum(item[1].grad[:, 0] ** 2, axis=1)) for item in fields])
        adjoints = [Jet1(cp.zeros_like(item[1].value), 2.0 * item[1].grad / len(points)) for item in fields]
        model.backward_block_jet1(cfg, modes, caches, adjoints)
        if epoch == 1 or epoch % int(cfg.validate_every) == 0:
            train_energy = cp.asnumpy(energies_device)
            validation_energy, validation_mean = evaluate_energy(model, cfg, validation_points)
            row = {'epoch': epoch, 'elapsed_seconds': time.perf_counter() - started, 'learning_rate': float(cfg.learning_rate), 'train_energy_density': train_energy.tolist(), 'validation_energy_density': validation_energy.tolist(), 'validation_mean_energy_density': validation_mean}
            history.append(row)
            history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
        if epoch % int(cfg.checkpoint_every) == 0:
            archive = args.run_dir / 'checkpoints' / f'harmonic_epoch{epoch:04d}.npz'
            save_checkpoint(archive, model, config, epoch, history)
            save_checkpoint(args.run_dir / 'checkpoints' / 'harmonic_latest.npz', model, config, epoch, history)
    with (args.run_dir / 'data' / 'training_history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', 'elapsed_seconds', 'learning_rate', 'validation_mean_energy_density'])
        writer.writeheader()
        for row in history:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    manifest = {'status': 'complete', 'subcase': cfg.subcase, 'epochs': final_epoch, 'checkpoint': str((args.run_dir / 'checkpoints' / ('harmonic_latest.npz' if final_epoch % int(cfg.checkpoint_every) == 0 else f'harmonic_epoch{final_epoch:04d}_not_archived')).resolve()), 'config': str(args.config.resolve()), 'elapsed_seconds_this_invocation': time.perf_counter() - started, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(errors='replace')}
    (args.run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))
if __name__ == '__main__':
    main()
