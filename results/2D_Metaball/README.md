# 2D Metaball

This bundle contains the K=1024 and K=2048 held-out comparison over three
unseen geometries on a `128 x 128` grid.

- Peeling sample: 16,384 accepted fluid points.
- Fit/test split: 70%/30% of valid nodes.
- K=1024 GEN checkpoint: epoch 2000.
- K=2048 GEN checkpoint: epoch 1000.
- Comparator: geometry-specific finite-difference Dirichlet Laplacian at the
  same K.

`mean_test_errors.json` and `mean_test_errors.csv` provide streamfunction,
velocity, and vorticity errors. The configurations and evaluation scripts are
in [`../../cases/2D_Metaball`](../../cases/2D_Metaball/README.md).
