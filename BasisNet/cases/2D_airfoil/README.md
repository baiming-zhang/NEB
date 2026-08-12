# 2D airfoil

## Configurations

All three FP16 K1024 configurations use the same script, backend, seed, network, precision, sampling, optimizer, scheduler, quadrature, validation pack schema, and checkpoint format.

| Configuration | Objective weights `(velocity, vorticity)` | Recorded checkpoint |
|---|---:|---:|
| [`velocity100`](configs/k1024_c2_sobol_1000_fp16_velocity_only.json) | `(1, 0)` | epoch 1000 |
| [`hybrid50`](configs/k1024_c2_sobol_10000_fp16_hybrid50.json) | `(0.5, 0.5)` | epoch 650 |
| [`vorticity100`](configs/k1024_c2_sobol_10000_fp16_vorticity100.json) | `(0, 1)` | epoch 5000 |

Run-only differences: `epochs`, `early_stopping_patience_epochs`, `result_root`. The config-equivalence test is in [`test_smoke.py`](tests/test_smoke.py).

The tables below retain the measured historical checkpoints; they were not retrained after the config-schema cleanup.

## Three-frame reconstruction

Mean relative L2 over `(beta, alpha) = (0.50,+10 deg), (0.85,0 deg), (0.50,-10 deg)`.

| Model | Velocity full | Velocity wake | Vorticity full | Vorticity wake |
|---|---:|---:|---:|---:|
| FP16 velocity 100% | 1.89% | 3.32% | 25.58% | 36.05% |
| FP16 hybrid 50/50 | 1.95% | 3.51% | 27.15% | 39.78% |
| FP64 velocity 100% | 1.82% | 3.44% | 23.65% | 36.80% |
| **FP16 vorticity 100%** | **1.71%** | **3.11%** | **20.75%** | **32.68%** |

![Three-frame comparison](comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_four_model_three_frame_l2_comparison_preview.png)

## Rollout30

| Method | Velocity full | Velocity wake | Vorticity full | Vorticity wake |
|---|---:|---:|---:|---:|
| **EigenFluid K1024** | **7.39% / 9.35%** | **16.14% / 21.89%** | **54.17% / 64.51%** | **78.58% / 98.25%** |
| Laplacian K1024 | 17.53% / 20.17% | 26.76% / 30.15% | 91.85% / 97.38% | 100.76% / 107.55% |
| NKB Eq. (8), K10 | 84.07% / 181.61% | 58.93% / 90.03% | 120.92% / 151.95% | 111.89% / 134.05% |

[PDF](comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases.pdf) · [MP4](comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases_10fps.mp4) · [full bundle](comparisons/rollout30_pure_vorticity_epoch5000/README.md)

![Rollout30](comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases_preview.png)

## Train

```powershell
$case = "BasisNet/cases/2D_airfoil"
python "$case/scripts/prepare_hardcoded_run.py" --config "$case/configs/k1024_c2_sobol_10000_fp16_vorticity100.json" --output-dir "$case/results/vorticity100/reference_inputs"
python -u "$case/scripts/train_hardcoded.py" --resume "$case/results/vorticity100/reference_inputs/initial_epoch0000.npz" --run-dir "$case/results/vorticity100/run_001" --epochs 5000 --validation-pack "$case/results/vorticity100/reference_inputs/fixed_validation_scipy_sobol_v1.npz" --archive-checkpoint-every 500 --early-stopping-patience-epochs 5000
```

```bash
python -m pytest -q BasisNet/cases/2D_airfoil/tests
```
