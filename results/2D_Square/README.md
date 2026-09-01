# 2D Square Taylor--Green result

This bundle records the latest 2026-09-01 13:17 Asia/Shanghai `(2,2)`
Taylor--Green result on `[0,pi]^2`.  It compares exact GT with independent
analytic Laplacian K=2048 and trained Potential K=2048 reconstructions and
viscous coefficient rollouts.

## Configuration and result

| Item | Value |
| --- | ---: |
| Grid / frames | `97x97` / `30` |
| Time range | `[0,80]` |
| Viscosity / initial amplitude | `1e-3` / `12` |
| Analytic / learned `(2,2)` eigenvalue | `8` / `7.9879621721` |
| Final / initial GT amplitude | `0.5272924` |
| Laplacian first / mean / final rL2 | `0 / 0 / 0` |
| Potential first / mean / final rL2 | `0.00379564 / 0.00383415 / 0.00391101` |

The earlier `(1,1)` visualization over `t in [0,4]` decayed by only `0.8%`.
This `(2,2)` result has decay rate `8 nu` and uses `t=80`, reducing the
amplitude from `12` to `6.32751` so the viscous decay is visually apparent.

## Files

- [`metrics/taylor_green_2x2_metrics.json`](metrics/taylor_green_2x2_metrics.json): full configuration, errors, eigenvalues, and video audit.
- [`metrics/taylor_green_2x2_frame_metrics.csv`](metrics/taylor_green_2x2_frame_metrics.csv): per-frame coefficients and relative errors.
- [`data/taylor_green_2x2_30frames.npz`](data/taylor_green_2x2_30frames.npz): GT streamfunction/vorticity/velocity and both reconstructed sequences.
- [`figures/taylor_green_2x2_first_frame.png`](figures/taylor_green_2x2_first_frame.png) and [PDF](figures/taylor_green_2x2_first_frame.pdf): first-frame reconstruction.
- [`figures/taylor_green_error_summary.png`](figures/taylor_green_error_summary.png) and [PDF](figures/taylor_green_error_summary.pdf): rollout error and modal decay.
- [`video/taylor_green_2x2_gt_vs_laplacian_vs_potential.mp4`](video/taylor_green_2x2_gt_vs_laplacian_vs_potential.mp4): 30-frame H.264 comparison video.
- [`MANIFEST.sha256`](MANIFEST.sha256): SHA-256 checksums for the bundle.

Implementation and reproduction instructions are in
[`cases/2D_Square/taylor_green`](../../cases/2D_Square/taylor_green/README.md).
