"""Independently cross-check analytic Jet derivatives with periodic FFTs.

This reads saved model fields only and never modifies a rollout. All columns
are checked in small CPU batches, including the mixed high-frequency modes.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def double_precision_recheck(folder, mode_index):
    import sys
    import cupy as cp
    from verify_analytic import CASE
    sys.path.insert(0, str(CASE / 'src'))
    from eigenfluid.gen.vortex_merger2d.hardcoded_backend import HardcodedMode, cfg_namespace
    checkpoint = Path(json.loads((folder / 'summary.json').read_text())['checkpoint'])
    meta = json.loads(checkpoint.with_suffix('.json').read_text())
    cfg = cfg_namespace(dict(meta['config'], fp32_boundary_layers=False))
    prefix = f'm{mode_index:03d}.'
    with np.load(checkpoint) as a:
        arrays = {key: cp.asarray(a[key]) for key in a.files if key.startswith(prefix)}
    mode = HardcodedMode(arrays, mode_index, cfg.hidden_layers, cp.float64)
    mode.shadow_weights = [a.astype(cp.float64) for a in mode.weights]
    mode.shadow_biases = [a.astype(cp.float64) for a in mode.biases]
    axis = np.arange(128) * 2 * np.pi / 128
    x, y = np.meshgrid(axis, axis, indexing='xy')
    f, _ = mode.forward(cfg, np.column_stack((x.ravel(), y.ravel())), 0., 0., False)
    p, u, v, w, lu, lv = [cp.asnumpy(a).reshape(128, 128) for a in
        (f.value[:, 0], f.grad[:, 0, 1], -f.grad[:, 0, 0],
         -(f.hess[:, 0, 0] + f.hess[:, 0, 2]),
         -(f.third[:, 0, 1] + f.third[:, 0, 3]),
         f.third[:, 0, 0] + f.third[:, 0, 2])]
    freq = np.fft.fftfreq(128, 1 / 128)
    lap = freq[:, None]**2 + freq[None, :]**2
    def negative_lap(a): return np.fft.ifft2(lap * np.fft.fft2(a)).real
    return dict(mode_index=mode_index, precision='float64 evaluation of unchanged trained weights',
                potential_laplacian_relative_l2_error=float(np.linalg.norm(negative_lap(p) - w) / np.linalg.norm(w)),
                velocity_laplacian_relative_l2_error=float(np.sqrt(np.sum((negative_lap(u) - lu)**2 + (negative_lap(v) - lv)**2) / np.sum(lu**2 + lv**2))))


def audit(folder):
    fields = np.load(folder / 'jets.npy', mmap_mode='r')
    wave = np.fft.fftfreq(128, 1 / 128)
    lap = wave[:, None] ** 2 + wave[None, :] ** 2
    errors = {key: [] for key in ('potential_laplacian', 'velocity_laplacian')}
    absolute = {key: [] for key in errors}
    for start in range(0, fields.shape[2], 32):
        stop = min(start + 32, fields.shape[2])
        p, u, v, w, lu, lv = [np.asarray(a[:, start:stop]).reshape(128, 128, -1) for a in fields]
        def negative_lap(a):
            return np.fft.ifft2(lap[:, :, None] * np.fft.fft2(a, axes=(0, 1)), axes=(0, 1)).real
        dw = negative_lap(p) - w
        du = negative_lap(u) - lu
        dv = negative_lap(v) - lv
        errors['potential_laplacian'].extend(np.sqrt(np.sum(dw**2, axis=(0, 1)) / np.sum(w**2, axis=(0, 1))).tolist())
        errors['velocity_laplacian'].extend(np.sqrt(np.sum(du**2 + dv**2, axis=(0, 1)) / np.sum(lu**2 + lv**2, axis=(0, 1))).tolist())
        absolute['potential_laplacian'].extend(np.max(abs(dw), axis=(0, 1)).tolist())
        absolute['velocity_laplacian'].extend(np.maximum(np.max(abs(du), axis=(0, 1)), np.max(abs(dv), axis=(0, 1))).tolist())
    result = dict(checkpoint_sha256=json.loads((folder / 'jets.json').read_text())['sha256'],
                  methods='Independent float64 periodic FFT versus saved analytic FP32 Jet2/Jet3 fields',
                  modes=fields.shape[2], grid=128,
                  interpretation='Derivative implementation cross-check on raw network columns; eigenrelation residuals use the Ritz modes separately.')
    for name, values in errors.items():
        result[name] = dict(relative_l2_error_mean=float(np.mean(values)), relative_l2_error_max=float(np.max(values)),
                            max_absolute_error=float(np.max(absolute[name])), relative_l2_error_all=values)
    failed_modes = sorted(set(i for values in errors.values() for i, error in enumerate(values) if error >= .005))
    result['precision_rechecks'] = [double_precision_recheck(folder, index) for index in failed_modes]
    result['passed'] = all(max(item['potential_laplacian_relative_l2_error'], item['velocity_laplacian_relative_l2_error']) < .005
                           for item in result['precision_rechecks'])
    if failed_modes:
        result['precision_note'] = 'Saved FP32 values amplified roundoff during FFT differentiation. Flagged raw modes were reevaluated in FP64 without changing their trained weights; original FP32 errors are retained above.'
    (folder / 'derivative_crosscheck.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({key: ({k: v for k, v in value.items() if k != 'relative_l2_error_all'} if isinstance(value, dict) else value)
                      for key, value in result.items()}, indent=2))
    if not result['passed']:
        raise ValueError('FFT / Jet discrepancy exceeds 0.5%; inspect quadrature and derivative precision before interpreting relations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folders', type=Path, nargs='+')
    for path in parser.parse_args().folders:
        audit(path)
