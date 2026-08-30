# 2D metaball K=1024/K=2048 held-out comparison

The two machine-readable files in this directory are copied from the latest
completed local comparison. They aggregate three unseen geometry tokens
(`4000001`--`4000003`) using arithmetic means over disjoint held-out nodes.

- Grid: `128 x 128`.
- Peeling sample: 16,384 accepted fluid points.
- Fit/test split: 70%/30% of valid nodes.
- K=1024 learned checkpoint: final epoch 2000.
- K=2048 learned checkpoint: latest durable epoch 1000.
- Comparator: geometry-specific finite-difference Dirichlet Laplacian at the
  same K.

The source configs and evaluation scripts are versioned under
[`../../cases/2D_metaball`](../../cases/2D_metaball/README.md). Checkpoints,
sample packs, full training histories, and reconstruction fields are excluded
because they are large runtime artifacts.
