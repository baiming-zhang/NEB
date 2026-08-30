# Three-dimensional connected metaballs

This case covers a rounded unit cube containing a connected smooth obstacle
made from three pairwise-overlapping spheres. Sphere parameters are
canonically ordered and remain clear of the outer wall. Each learned mode is a
complete antisymmetric tensor potential with one external coefficient.

## Local reproducible workflow

The versioned PyTorch workflow trains K=64 potential-Rayleigh bases with
geometry-specific stable sorting and two MGS passes. The retained sweep winner
uses seed `20260823`, 2000 epochs, 6144 IID fluid points per epoch, eight fixed
validation geometries, SIREN frequencies `(6,36)`, and a squared boundary
envelope.

```bash
python cases/3D_metaball_connected/scripts/train_validate.py \
  --config cases/3D_metaball_connected/configs/k64_three_sphere_connected_siren_dirichlet_2000_sweep_winner.json
```

The run is resumable from its case-local checkpoint. The 128^3 finite-
difference comparison and reconstruction scripts are:

```bash
python cases/3D_metaball_connected/scripts/solve_unseen_laplacian_128.py
python cases/3D_metaball_connected/scripts/evaluate_sweep_winner_128.py
```

## Latest K=1024 publication comparison

The latest paper artifact scales the same connected three-sphere geometry to
1024 complete vector modes and compares five constructions:

| Method | Velocity rL2 | Vorticity rL2 |
| --- | ---: | ---: |
| Strict FD-Lap RR | **0.15965** | **0.57047** |
| Potential-Rayleigh | 0.17829 | 0.62882 |
| Velocity-Rayleigh | 0.28081 | 0.71109 |
| Vorticity-Rayleigh | 0.92911 | 1.18316 |
| Vector/Hodge-Lap RR | 0.56623 | 0.90298 |

All methods are evaluated on all `1,599,754` fluid nodes of a `128^3` grid
with the same second-order finite-difference curl and the same absolute
isovorticity level. Exact configs, timings, metrics, hashes, and the six-panel
PNG are synchronized in
[`../../results/3d_metaball_k1024`](../../results/3d_metaball_k1024/README.md).

The publication bundle intentionally omits raw geometry data, training
checkpoints, learned basis volumes, ground-truth volumes, caches, and logs.
Those heavy artifacts are not silently substituted by the older K=64 local
workflow.

## Verification

```bash
python -m pytest -q cases/3D_metaball_connected/tests
python -m ruff check cases/3D_metaball_connected
```
