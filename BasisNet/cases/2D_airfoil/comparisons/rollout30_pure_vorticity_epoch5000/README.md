# 2-D airfoil results: pure-vorticity K1024 and rollout30

This directory publishes the reproducible result bundle for the 2-D airfoil
case. It includes the epoch-5000 pure-vorticity configuration,
machine-readable evaluation summaries, static four-model comparisons, the
first six modes, and a 30-frame comparison against five reduced-order
baselines.

The constructed GT is a D2Q9 TRT-LBM Navier--Stokes field at `Re=1500`,
`beta=0.50`, and `alpha=+10 deg` on a `1001 x 801` grid. The airfoil has a
no-slip boundary, the top and bottom use uniform free-stream conditions, and
the outlet uses a zero-gradient condition. Each forecast frame corresponds to
500 LBM steps. Every method is evaluated from the same reconstructed velocity
field against the same independent 30-frame trajectory.

## Published artifacts

- [Four-model three-frame L2 comparison](artifacts/airfoil_four_model_three_frame_l2_comparison.pdf)
- [Four-model three-frame preview](artifacts/airfoil_four_model_three_frame_l2_comparison_preview.png)
- [First six modes for four methods over three frames](artifacts/airfoil_four_method_lowest6_modes_three_frames.pdf)
- [GT and six-method rollout30 report](artifacts/airfoil_rollout30_literature_rom_comparison.pdf)
- [Rollout30 preview](artifacts/airfoil_rollout30_literature_rom_comparison_preview.png)
- [Rollout30 MP4, 10 fps](artifacts/airfoil_rollout30_literature_rom_comparison_10fps.mp4)
- [Aggregate metrics](metrics/rollout30_literature_rom_aggregate.csv) and individual JSON summaries in [`metrics/`](metrics/)

## Rollout30 result

Numbers are mean / final relative L2 error over 30 forecast frames.

| Method | Velocity, full | Velocity, wake | Vorticity, full | Vorticity, wake |
|---|---:|---:|---:|---:|
| EigenFluid, FP16 pure vorticity, K1024, epoch 5000 | 7.39% / 9.35% | 16.14% / 21.89% | 54.17% / 64.51% | 78.58% / 98.25% |
| Rectangular Laplacian, K1024 | 17.53% / 20.17% | 26.76% / 30.15% | 91.85% / 97.38% | 100.76% / 107.55% |
| Neural Kinematic Bases Eq. (8), K10 reproduction | 84.07% / 181.61% | 58.93% / 90.03% | 120.92% / 151.95% | 111.89% / 134.05% |
| POD--Galerkin, float64-resolved K40 | 5.76% / 5.95% | 12.66% / 13.24% | 44.51% / 44.30% | 78.27% / 76.56% |
| Exact projected DMD, float64-resolved K40 | 5.73% / 6.35% | 13.22% / 13.38% | 39.91% / 36.83% | 70.62% / 61.95% |
| Regularized quadratic Operator Inference, K7 | 9.62% / 8.33% | 22.86% / 18.53% | 53.92% / 39.57% | 96.65% / 68.16% |

The table is the comparison outcome; no training or validation objective is
used as a substitute for rollout error. On this trajectory, exact DMD and
POD--Galerkin have lower mean full-domain velocity error than EigenFluid.
Exact DMD also has the lowest mean vorticity errors. EigenFluid remains better
than the rectangular Laplacian and the K10 NKB reproduction.

## Baseline protocols

- **POD--Galerkin** uses the standard POD expansion and incompressible
  Navier--Stokes projection, including the outlet total-pressure term. The
  snapshot rank is 40 under a float64 spectral backward-error bound; the
  mass-orthogonality defect is `7.07e-14`. Reference:
  [Noack et al.](https://berndnoack.com/publications/Noack2012ceat.PDF).
- **Exact DMD** fits the exact projected one-step map in the same resolved
  K40 POD coordinates. Its fitted spectral radius is `1.0000000000000056`.
  Reference: [Schmid 2010](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/volume/79D53179B2E0CE770D2C3296E2565833).
- **Regularized Operator Inference** uses the maximum identifiable quadratic
  order, K7 (`36` independent feature columns from `39` derivative samples).
  The regularization is `1.0`; it minimizes integrated training-trajectory
  error subject to the published state bound and does not inspect the
  evaluation trajectory. Reference:
  [McQuarrie, Huang & Willcox 2021](https://kiwi.oden.utexas.edu/papers/nonlinear-non-intrusive-model-reduction-combustion-McQuarrie-Huang-Willcox.pdf).
- **Neural Kinematic Bases (NKB)** follows the SIGGRAPH Asia 2025
  camera-ready K10 architecture and its six unsupervised losses. Rollout uses
  camera-ready Eq. (8): semi-Lagrangian backward tracing, Euclidean projection
  to the closest domain point, source-velocity copy, and least-squares
  projection. The body/world rotation is an exact inverse pair and the ellipse
  projection solves the Euclidean closest-point problem rather than applying
  radial clipping. The method was reproduced at equation level because no
  official implementation or checkpoint was located for this study. NKB
  targets closed slip-boundary inviscid domains, so this fixed-airfoil result
  is an adaptation without external inlet forcing or viscosity, not an
  author-checkpoint result. Reference:
  [Liu et al. 2025](https://arxiv.org/abs/2504.15657).

POD--Galerkin, DMD, and Operator Inference use 41 constructed GT snapshots at
500-step spacing. The 30-frame evaluation trajectory is independent from that
training sequence.

## Reproduce the epoch-5000 model

The checkpoint is intentionally absent from Git. Its size is 188,118,934
bytes (179.4 MiB), above GitHub's normal 100 MiB object limit. Reproduce it
from the committed config and deterministic initializer:

```powershell
$case = "BasisNet/cases/2D_airfoil"
$config = "$case/configs/k1024_c2_sobol_10000_fp16_vorticity100.json"
$run = "$case/results/k1024_c2_sobol_10000_fp16_vorticity100/run_001"

python "$case/scripts/prepare_hardcoded_run.py" `
  --config $config `
  --output-dir "$run/reference_inputs"

python -u "$case/scripts/train_hardcoded.py" `
  --resume "$run/reference_inputs/initial_epoch0000.npz" `
  --run-dir $run `
  --epochs 5000 `
  --validation-pack "$run/reference_inputs/fixed_validation_scipy_sobol_v1.npz" `
  --archive-checkpoint-every 500 `
  --early-stopping-patience-epochs 5000
```

Expected deterministic identities:

| File | SHA-256 |
|---|---|
| epoch-0 checkpoint | `b1eb27bbc8f1c6f7dc5ece82ac852dd3f2842952c1dcd48d6acdb0e0b68f69a5` |
| fixed validation pack | `fa1817e4a83bbe93846bba84cb2af050d7a9b67ed37f9db0930ac1b56bbc11df` |
| epoch-5000 checkpoint | `f1ac263de87dd6f1a0370741f76c1fce309073b8561be5a314d89d144c0b981d` |

Training uses seed `20260801`, 1024 modes, 4096 mixed Sobol points per epoch,
512 vorticity anchors, FP16 hidden shadows, FP32 boundary/master/Adam and
analytic backward passes, and FP64 Gram/Ritz/reorthogonalization. The loss
weights are velocity `0` and vorticity `1`; all auxiliary vorticity or
smoothing regularizers are zero. Named checkpoints are archived every 500
epochs.

## Reproduce the GT and rollout

Construct the GT from the committed solver configuration:

```powershell
python "$case/scripts/solve_ns.py" `
  --config $config `
  --solver-config "$case/configs/gt_re1500_separated.json" `
  --beta 0.5 `
  --alpha-deg 10 `
  --output "$run/gt_beta050_alpha_p10.npz"
```

Create the initial reconstruction with `evaluate_single_gt.py`, phase `0.5`,
Ritz quadrature `4096`, and field batch `32768`. Run EigenFluid with
`rollout_eigenfluid_rom.py`; the measured operator uses all 797,716 fluid
nodes, retains rank 1024, and has mass-orthogonality defect `1.343155e-6`.

Train and run the three literature ROMs against the independent reference:

```powershell
python "$case/scripts/run_literature_rom_baselines.py" `
  --training-gt "$run/gt_beta050_alpha_p10.npz" `
  --initial-sample "$run/evaluation/reconstruction_sample.npz" `
  --reference-fields "$run/rollout30/eigenfluid/ns_vs_eigenfluid_rollout_fields.npz" `
  --airfoil-config $config `
  --output-dir "$run/rollout30/literature" `
  --frames 30 `
  --steps-per-frame 500 `
  --pod-substeps-per-frame 10 `
  --opinf-substeps-per-frame 50
```

Train NKB and apply its camera-ready Eq. (8) integrator:

```powershell
python "$case/scripts/train_neural_kinematic_baseline.py" `
  --output-dir "$run/nkb_camera_ready" `
  --epochs 10 --samples-per-epoch 1000000 --batch-size 10000 `
  --seed 20260812 --beta 0.5 --alpha-deg 10

python "$case/scripts/rollout_neural_kinematic_baseline.py" `
  --checkpoint "$run/nkb_camera_ready/latest.pt" `
  --sample "$run/evaluation/reconstruction_sample.npz" `
  --reference-fields "$run/rollout30/eigenfluid/ns_vs_eigenfluid_rollout_fields.npz" `
  --airfoil-config $config `
  --output-dir "$run/rollout30/nkb" `
  --frames 30 --steps-per-frame 500 --substeps-per-frame 10 `
  --integration-points 250000 --field-batch 32768
```

The Laplacian baseline and report builder are likewise committed in
[`scripts/`](../../scripts/). Their exact diagnostics and all 30 per-frame
errors are retained in the JSON summaries.

## Integrity and data policy

Reports, video, summaries, tests, and reproduction scripts are committed.
Model checkpoints and full rollout field archives are not; each field NPZ is
about 500 MiB and can be regenerated. [`checksums.sha256`](checksums.sha256)
records every published artifact and metric file.
