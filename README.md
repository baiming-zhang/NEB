# EigenFluid-SSS

## Results

| Case | Modes | Evaluation | Relative L2 |
|---|---:|---|---:|
| [2D Karman](BasisNet/cases/2D_karman/README.md) | 96 | velocity, 17 frames | 5.81% |
| [2D airfoil](BasisNet/cases/2D_airfoil/README.md) | 1024 | velocity, 3 frames | 1.71% |
| [3D periodic](BasisNet/cases/3D_periodic_free/README.md) | 256 | velocity, 50 frames | 0.25% |
| [3D wake](BasisNet/cases/3D_parametric_wake/README.md) | 256 | velocity, 30 frames | 17.11% |

### 2D airfoil: three-frame reconstruction

| Model | Epoch | Time | Velocity full | Velocity wake | Vorticity full | Vorticity wake |
|---|---:|---:|---:|---:|---:|---:|
| FP16 velocity 100% | 1000 | 9.43 h | 1.89% | 3.32% | 25.58% | 36.05% |
| FP16 velocity 50% + vorticity 50% | 650 | 3.90 h | 1.95% | 3.51% | 27.15% | 39.78% |
| FP64 velocity 100% | 5000 | 115.82 h | 1.82% | 3.44% | 23.65% | 36.80% |
| **FP16 vorticity 100%** | **5000** | **29.43 h** | **1.71%** | **3.11%** | **20.75%** | **32.68%** |

[PDF](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_four_model_three_frame_l2_comparison.pdf) · [first six modes](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_four_method_lowest6_modes_three_frames.pdf)

![Three-frame comparison](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_four_model_three_frame_l2_comparison_preview.png)

### 2D airfoil: rollout30

Mean / final relative L2.

| Method | Velocity full | Velocity wake | Vorticity full | Vorticity wake |
|---|---:|---:|---:|---:|
| **EigenFluid K1024** | **7.39% / 9.35%** | **16.14% / 21.89%** | **54.17% / 64.51%** | **78.58% / 98.25%** |
| Laplacian K1024 | 17.53% / 20.17% | 26.76% / 30.15% | 91.85% / 97.38% | 100.76% / 107.55% |
| Neural Kinematic Bases Eq. (8), K10 | 84.07% / 181.61% | 58.93% / 90.03% | 120.92% / 151.95% | 111.89% / 134.05% |

[PDF](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases.pdf) · [MP4](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases_10fps.mp4) · [metrics](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/metrics)

![Rollout30 comparison](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/artifacts/airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases_preview.png)

## Run

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Model checkpoints and full field archives are not stored in Git. Reproduction commands are in the [airfoil result bundle](BasisNet/cases/2D_airfoil/comparisons/rollout30_pure_vorticity_epoch5000/README.md).
