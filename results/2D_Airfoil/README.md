# 2D Airfoil

This bundle records the K=2048 thirty-frame ROM comparison between GEN's
Potential-Rayleigh space and a matched geometry-specific finite-difference
Laplacian space. Both methods start from projections of the same GT frame and
use the same operators, mean flow, viscosity, time step, and boundary
treatment.

Across three unseen geometries, mean total-field velocity error is `4.07%` for
Potential-Rayleigh and `22.01%` for FD Laplacian. Final-frame velocity error is
`5.46%` versus `46.56%`.

- `rollout_metrics.json` and `rollout_metrics.csv`: complete velocity and
  vorticity metrics.
- `figures/airfoil_temporal_error_all_geometries.{png,pdf}`: temporal-error
  comparison.
- `figures/*.pdf`: fieldwise objective and reconstruction figures.
- `MANIFEST.sha256`: bundle integrity hashes.
- [`strict_k2048_20260901/`](strict_k2048_20260901/README.md): strict paired
  Potential/Velocity/Vorticity K=2048 training, 54-row nested-prefix
  reconstruction evaluation, physical and diagnostic rollouts, frozen
  checkpoints, media, and the requested `dt=0.1` audit. Cubature is excluded.

The source rollout is
[`../../cases/2D_Airfoil/scripts/rollout_gt30_fd_vs_potential_k2048.py`](../../cases/2D_Airfoil/scripts/rollout_gt30_fd_vs_potential_k2048.py).
