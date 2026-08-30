# Two-dimensional parametric airfoil

This case learns one geometry-conditioned divergence-free spatial basis over
airfoil thickness and angle of attack. Production training uses a case-local
CuPy/CUDA backend with explicit forward passes and handwritten reverse passes;
PyTorch autograd is not used in the production objective.

Two local research tracks are retained without duplicating the learned basis
mapping. [`hardcoded_backend.py`](src/eigenfluid_sss/basisnet/airfoil2d/hardcoded_backend.py)
is the strict Jet1/Jet2/Jet3 backend used by the latest fieldwise comparison;
it automatically delegates C2 geometry, hybrid, and certified-Stokes configs
to [`legacy_c2_backend.py`](src/eigenfluid_sss/basisnet/airfoil2d/legacy_c2_backend.py).
This preserves the main worktree's certified-Stokes progress while keeping the
paper-aligned strict protocol reproducible.

## Current fieldwise protocol

The latest paper comparison trains matched K=2048 bases with the same seed,
geometry schedule, samples, initialization, and optimizer state. The arms
differ only in the field used by the Rayleigh quotient:

| Arm | Objective | Potential jet order | Config |
| --- | --- | ---: | --- |
| Potential-Rayleigh | `||grad A||^2 / ||A||^2` | 1 | [`k2048_airfoil_strict_potential_rayleigh_2000.json`](configs/k2048_airfoil_strict_potential_rayleigh_2000.json) |
| Velocity-Rayleigh | `||grad u||^2 / ||u||^2` | 2 | [`k2048_airfoil_strict_velocity_rayleigh_2000.json`](configs/k2048_airfoil_strict_velocity_rayleigh_2000.json) |
| Vorticity-Rayleigh | `||grad omega||^2 / ||omega||^2` | 3 | [`k2048_airfoil_strict_vorticity_rayleigh_2000.json`](configs/k2048_airfoil_strict_vorticity_rayleigh_2000.json) |

All three use continuous PCG64 samples, exact analytic boundary envelopes,
stable raw-Rayleigh ordering, and causal modified Gram--Schmidt. The retained
K=2048 Potential-Rayleigh run reaches 2000 epochs. The paired-run audit is
read-only by default:

```bash
python cases/2D_airfoil/scripts/run_airfoil_strict_rayleigh_k2048_paired_e2000.py
```

Passing `--execute` starts or resumes the missing paired arms after validating
the fixed sample packs and reference hashes. The immutable K=2048 potential
reference used by that audit is stored under
`results/siren_original_jet1_gpu_k128_to_k2048_e2000_sequence/arms/k2048`
with Git LFS.

## Static reconstruction and temporal rollout

Three unseen Re=1500 geometries are evaluated at
`(beta, alpha)=(0.50,+10 deg),(0.85,0 deg),(0.50,-10 deg)`. The matched
finite-difference comparator uses the same masked domain, mode count, fitting
data, evaluation data, and discrete operators.

The paper's decisive result is the 30-frame K=2048 coefficient rollout. Mean
velocity error is `4.07%` for Potential-Rayleigh and `22.01%` for the matched
FD Laplacian. Final-frame velocity error is `5.46%` versus `46.56%`. Exact
per-geometry velocity/vorticity tables, a time-step audit, and PNG/PDF figures
are versioned in
[`../../results/airfoil_k2048_rollout30`](../../results/airfoil_k2048_rollout30/README.md).

The rollout implementation is
[`rollout_gt30_fd_vs_potential_k2048.py`](scripts/rollout_gt30_fd_vs_potential_k2048.py).
It uses a shared masked-Cartesian incidence complex, an implicit trapezoidal
coefficient update with exact viscous decay, and the frozen-base closure
`F(a)-F(0)` because the GT archive contains velocity but no pressure field.

## Additional local progress

- [`k1024_c2_stokes_certified_fp64_5000.json`](configs/k1024_c2_stokes_certified_fp64_5000.json)
  and its K=8 smoke config add the certified Stokes projection path. Their
  preserved entry points are
  [`prepare_c2_stokes_certified.py`](scripts/prepare_c2_stokes_certified.py)
  and [`train_c2_stokes_certified.py`](scripts/train_c2_stokes_certified.py).
- [`rollout_laplacian_rom.py`](scripts/rollout_laplacian_rom.py) provides the
  matched Laplacian ROM.
- [`train_neural_kinematic_baseline.py`](scripts/train_neural_kinematic_baseline.py)
  and [`rollout_neural_kinematic_baseline.py`](scripts/rollout_neural_kinematic_baseline.py)
  provide the neural-kinematic comparison without changing the EigenFluid
  basis contract.
- `mimetic_fd.py`, `strict_prefix.py`, and the Jet1/Jet2/Jet3 tests capture the
  latest discrete-operator and prefix-reproducibility audits.

## Verification

```bash
python -m pytest -q cases/2D_airfoil/tests
python -m ruff check cases/2D_airfoil
```

CUDA-specific tests skip when CuPy or a compatible GPU is unavailable. Most
raw GT archives, sample packs, checkpoints, training logs, and rendered videos
are excluded from Git; the hashed K=2048 Potential-Rayleigh reference files
required by the paired-protocol test are tracked through Git LFS.
