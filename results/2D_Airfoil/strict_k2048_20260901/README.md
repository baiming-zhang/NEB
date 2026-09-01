# 2D Airfoil strict K=2048 release (2026-09-01)

This is the reproducible Airfoil snapshot completed by 10:00 (Asia/Shanghai)
on 2026-09-01. The only later result admitted by request is the Potential ROM
time-step audit at `internal_dt=0.1`. Cubature experiments, scripts, logs, and
outputs are intentionally absent.

## Fair training protocol

Potential-, Velocity-, and Vorticity-Rayleigh use the same `K=2048`, seed
`20260823`, epoch-zero model/Adam state, 2000 geometries, and 4096 NumPy
PCG64 rejection-sampled fluid points per epoch. The common network is four
layers wide 32 with SIREN frequencies `(8,30)`, Adam with a cosine schedule
from `2e-4` to `1e-6`, gradient clipping at 5, FP32 compute, GPU-resident sample
packs, GPU stable causal sorting, and two-pass causal MGS. Only

`F = A` (Jet1), `F = u` (Jet2), or `F = omega` (Jet3)

changes in `R(F)=||grad F||_h^2/||F||_h^2`. All three finish at rank 2048.
Potential takes 8.14 h, Velocity 8.07 h, and Vorticity 11.99 h. See
[`training/training_summary.csv`](training/training_summary.csv), the archived
protocols, full epoch CSV histories, and the Velocity/Vorticity frozen
checkpoints. Potential reuses the K=2048 reference checkpoint already tracked
under `cases/2D_Airfoil/results/`.

## Reconstruction

Each objective/geometry produces one geometry-specific K=2048 transform on
32768 shared peeling points and then slices strict nested prefixes
`K={64,128,256,512,1024,2048}`. All methods use the same 65536 velocity
projection points, grid, boundary, GT mean, and target frame. The complete 54
rows and six relative-L2 metrics are in
[`reconstruction/strict_prefix_reconstruction_metrics.csv`](reconstruction/strict_prefix_reconstruction_metrics.csv).

At K=2048, total velocity error on `(up,middle,down)` is:

| Rayleigh field | up | middle | down |
| --- | ---: | ---: | ---: |
| Potential | 1.29% | 0.02% | 0.34% |
| Velocity | 1.26% | 0.02% | 0.34% |
| Vorticity | 1.30% | 0.02% | 0.37% |

## Temporal evaluation

The physical-viscosity rollout advances 30 output frames with the same
mimetic `grad-curl-Laplacian` chain, mean flow, boundary treatment,
frozen-base-subtracted closure, and implicit trapezoidal Newton-GMRES solver.
The complete per-frame metrics, energy budgets, summaries, timings, and the
five-method upper-geometry video are under `temporal/physical/`.

The upper-geometry mean total velocity errors are 6.62% (Potential), 6.68%
(Velocity), 6.60% (Vorticity), and 49.87% (matched mimetic FD-Laplacian).
All three learned spaces require about 137--138 s for coefficient rollout;
basis construction costs 53.01 s, 115.61 s, and 251.09 s, respectively.

The `dt=0.1` audit uses 50 internal steps per saved frame instead of 10. It
does not improve the `dt=0.5` Potential result: mean velocity error changes
from 6.62% to 6.64%, while coefficient rollout time rises from 138.47 s to
687.92 s (4.97x). See [`temporal/dt_sensitivity_up.csv`](temporal/dt_sensitivity_up.csv).

## Additional completed diagnostics

- `temporal/mu0_up/`: zero-viscosity diagnostic, not the physical baseline.
- `temporal/vorticity_init_up/`: vorticity-initialized upper rollout.
- `diagnostics/direct_projection_up/`: independent K=2048 Potential
  reconstruction of all 31 GT frames; mean velocity/vorticity errors are
  1.52%/16.78% and no temporal model is used.
- `diagnostics/dynamic_geometry/`: the pre-cutoff moving-geometry Potential
  ALE-ROM run; coefficients are transported between one inferred basis per
  saved geometry state before ROM advancement.

## Implementation map

- `generate_airfoil_strict_paired_pcg64_pack.py`: deterministic paired sample
  packs and source/hash audit.
- `run_airfoil_strict_rayleigh_k2048_paired_e2000.py`: paired training runner.
- `hardcoded_backend.py`: analytic Jet1/Jet2/Jet3 Rayleigh gradients, stable
  causal order, two-pass MGS, GPU-resident optimization, and transactional
  vorticity candidate rejection.
- `evaluate_strict_rayleigh_k2048_prefixes.py`: strict nested-prefix evaluator.
- `rollout_gt30_fd_vs_potential_k2048.py`: common mimetic ROM dynamics.
- `run_strict_rayleigh_k2048_rollout.py`: three-geometry/four-method runner.
- `reconstruct_gt_frames_potential_k2048.py`: independent per-frame projection.
- `rollout_dynamic_geometry_potential_k2048.py`: moving-geometry basis
  inference, coefficient transport, and ALE split.

Large intermediate grids and per-frame field archives are omitted. The exact
paired train pack and final Velocity/Vorticity checkpoints are stored with Git
LFS; compact metrics and media are normal Git objects.
