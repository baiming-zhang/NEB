# 2D Square Taylor--Green implementation

This directory contains the implementation used for the 2026-09-01 13:17
Asia/Shanghai result snapshot.  The default case is a four-cell `(2,2)`
Taylor--Green vortex on `[0, pi]^2`:

```text
omega = A exp(-8 nu t) sin(2x) sin(2y)
u_x   = A/4 exp(-8 nu t) sin(2x) cos(2y)
u_y   = -A/4 exp(-8 nu t) cos(2x) sin(2y)
```

The published configuration uses `A=12`, `nu=1e-3`, 30 frames on a `97x97`
grid, and `t in [0,80]`.  Its final-to-initial amplitude ratio is
`exp(-8*nu*80)=0.5272924`, so viscous decay is visible with a fixed color
scale.

## Method

1. Generate exact streamfunction, vorticity, and velocity GT.
2. Project only GT frame zero onto the analytic K=2048 square Laplacian basis.
3. Reconstruct frame zero independently with the analytic `(2,2)` mode and
   the aligned learned Potential K=2048 `(2,2)` mode.
4. Advance both active coefficients with their own exact viscous factors.
   The quadratic term is zero because an equal-wave-number mode has zero
   self-interaction.
5. Compare every predicted frame against exact GT with the FEM mass-matrix
   relative L2 norm, then render fixed-scale Nature red/blue figures and an
   H.264 comparison video.

The small [`potential_mode_2_2.npy`](potential_mode_2_2.npy) cache is the
aligned learned vorticity mode used for the published result.  It lets the
rollout reproduce without distributing the 94 MB training checkpoint.  Its
physical Ritz eigenvalue is `7.987962172069773`; the analytic value is `8`.

## Reproduce the published rollout

From the repository root:

```bash
python cases/2D_Square/taylor_green/run_taylor_green_validation.py
```

This writes to `output/taylor_green_2x2_validation`.  FFmpeg and FFprobe are
required for video creation and verification.

To re-export the learned mode from the original checkpoint instead of using
the cache, also provide its adjacent JSON metadata and the saved aligned Ritz
transform:

```bash
python cases/2D_Square/taylor_green/run_taylor_green_validation.py \
  --force-checkpoint-evaluation \
  --checkpoint /path/to/epoch_2000_neutral.npz \
  --transform /path/to/trained_k2048_basis_transform.npz
```

That path additionally requires the optional CUDA/CuPy dependency.

Published outputs are documented in
[`results/2D_Square`](../../../results/2D_Square/README.md).
