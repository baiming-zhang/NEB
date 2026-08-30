# Reproducible cases

Each case contains its own configurations, training and evaluation entry
points, implementation modules, and tests. All cases use the same
snapshot-free geometry-conditioned basis idea; the directories separate
physical domains and audited numerical protocols, not different learned
mapping products.

## Active cases

| Case | Basis and backend | Independent evaluation | Current status |
| --- | --- | --- | --- |
| [2D_airfoil](2D_airfoil/README.md) | K=64--2048, CuPy/CUDA analytic Jet1/Jet2/Jet3 | Re=1500 D2Q9-TRT trajectories and matched FD Laplacian ROM | fieldwise Rayleigh comparison and 30-frame K=2048 rollout available |
| [2D_metaball](2D_metaball/README.md) | K=64--2048, SIREN with one- or two-pass causal MGS | three unseen geometries, random smooth targets, matched FD Dirichlet spectrum | K=1024 epoch 2000 and K=2048 epoch 1000 metrics available |
| [2D_square_analytic](2D_square_analytic/README.md) | K=8/K=64, generic PyTorch core | exact unit-square Dirichlet spectrum | sine/ELU and SIREN/WIRE diagnostics available |
| [3D_metaball_connected](3D_metaball_connected/README.md) | K=64 local PyTorch workflow; K=1024 publication configs | unseen connected three-sphere geometry and 128^3 FD comparators | K=1024 five-method publication bundle available under `results/` |
| [3D_parametric_fan](3D_parametric_fan/METHOD.md) | K=1024 optimized analytic backend | independent rotating-fan reference and Laplacian projection | complete evaluation bundle versioned |

## Paper-aligned publication artifacts

- [`../results/2d_metaball_k1024_k2048`](../results/2d_metaball_k1024_k2048/README.md)
  records the exact held-out K=1024/K=2048 comparison used by the latest
  manuscript revision.
- [`../results/3d_metaball_k1024`](../results/3d_metaball_k1024/README.md)
  records the K=1024 five-method 3D comparison synchronized from the paper
  repository.
- [`../results/airfoil_k2048_rollout30`](../results/airfoil_k2048_rollout30/README.md)
  records the complete 30-frame airfoil rollout table and paper figures.

## Verification

Run from the repository root:

```bash
python -m pytest -q tests
python -m pytest -q cases/2D_square_analytic/tests
python -m pytest -q cases/2D_metaball/tests
python -m pytest -q cases/3D_metaball_connected/tests
python -m pytest -q cases/2D_airfoil/tests
python -m pytest -q cases/3D_parametric_fan/tests
python -m ruff check basisnet.py tests cases
```

Tests that require CuPy or a CUDA device skip when that backend is not
available. Training campaigns are intentionally not started by the test
suite.

## Artifact policy

Version source code, exact JSON configs, fixed-input manifests, tests, compact
CSV/JSON metrics, and reviewed PNG/PDF figures. Keep raw HDF5/NPZ fields,
checkpoints, learned basis volumes, caches, logs, frames, and videos outside
Git unless a hashed artifact is required for an automated publication audit.
The K=2048 airfoil Potential-Rayleigh reference is that explicit exception and
is stored with Git LFS; other omitted heavy artifacts remain identified by
metadata and hashes.
