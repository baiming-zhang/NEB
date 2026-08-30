# Airfoil K=2048 thirty-frame ROM rollout

This bundle records the latest paper comparison between the learned
Potential-Rayleigh space and a matched geometry-specific finite-difference
Laplacian space. Both start from independently projected versions of the same
GT frame 0 and advance 30 saved intervals with the same masked-Cartesian
operators, mean flow, viscosity, time step, and open-boundary treatment.

Mean total-field velocity error across the three unseen geometries is `4.07%`
for Potential-Rayleigh and `22.01%` for FD Laplacian. Final-frame velocity
error is `5.46%` versus `46.56%`. The complete velocity/vorticity table is in
`rollout_metrics.json` and `rollout_metrics.csv`.

`figures/airfoil_temporal_error_all_geometries.png` is the raster paper panel;
the matching vector PDF and the static objective/reconstruction PDFs are kept
beside it. The source rollout is
[`../../cases/2D_airfoil/scripts/rollout_gt30_fd_vs_potential_k2048.py`](../../cases/2D_airfoil/scripts/rollout_gt30_fd_vs_potential_k2048.py).

Raw Re=1500 HDF5 trajectories, K=2048 checkpoints, FD eigenvector caches,
peeled states, and full reconstructed frame volumes are excluded from Git.
