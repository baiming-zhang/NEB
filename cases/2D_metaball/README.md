# Two-dimensional connected metaballs

This case learns a geometry-conditioned streamfunction basis on a rounded
square containing a smooth connected union of three circles. Geometry is
encoded by the three circle centers and radii; every sampled training geometry
satisfies the connected-obstacle and wall-clearance contract.

## Latest scaling runs

The paper-aligned runs use the velocity-Rayleigh objective and exactly one
causal velocity-mass MGS pass:

| Modes | First SIREN frequency | Epoch used | Training status | Config |
| ---: | ---: | ---: | --- | --- |
| 1024 | 160 | 2000 | complete | [`k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json`](configs/k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json) |
| 2048 | 224 | 1000 | latest durable checkpoint | [`k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json`](configs/k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json) |

Both use 4096 continuous fluid samples per epoch, four fixed validation
geometries, batches of 32 modes, FP32 arithmetic, and no flow snapshots.

Launchers are resumable and keep their generated data under the ignored
case-local `results/` directory:

```bash
python cases/2D_metaball/scripts/run_metaball_velocity_rayleigh_omega160_k1024_e2000.py
python cases/2D_metaball/scripts/run_metaball_velocity_rayleigh_omega224_k2048_e2000.py
```

## Held-out comparison

Three unseen geometry tokens (`4000001`--`4000003`) are evaluated on a
`128 x 128` grid. Velocity coefficients are fit on 70% of valid nodes and all
relative L2 errors are measured on the disjoint 30%. Learned and FD bases use
the same K, domain mask, target, split, discrete velocity/vorticity operators,
and invertible velocity-mass whitening.

| Method | K | Streamfunction | Velocity | Vorticity |
| --- | ---: | ---: | ---: | ---: |
| FD Laplacian | 1024 | **0.02122** | **0.09423** | 0.62079 |
| Learned, epoch 2000 | 1024 | 0.05622 | 0.16255 | **0.58754** |
| FD Laplacian | 2048 | 0.01235 | **0.05674** | 0.51569 |
| Learned, epoch 1000 | 2048 | **0.01183** | 0.06998 | **0.36447** |

The complete machine-readable table is versioned in
[`../../results/2d_metaball_k1024_k2048`](../../results/2d_metaball_k1024_k2048/README.md).
At K=2048 the learned span improves streamfunction and vorticity error over
matched FD by 4.16% and 29.32%, respectively; FD retains the lower velocity
error.

## Evaluation and audits

- `evaluate_velocity_k1024_three_unseen.py` and
  `evaluate_velocity_checkpoint_three_unseen.py` reproduce the held-out
  learned/FD comparisons.
- `plot_fdlaplacian_k1024_k2048_reconstructions.py` composes the matched mode
  count comparison.
- `audit_k2048_rayleigh_sample_convergence.py` verifies that increasing fixed
  Rayleigh validation samples does not change the downstream reconstruction
  conclusion.
- `test_metaball_streamfunction_jet1.py` and
  `test_metaball_velocity_rayleigh.py` cover analytic spatial derivatives and
  the one-pass velocity-Rayleigh contract.

```bash
python -m pytest -q cases/2D_metaball/tests
python -m ruff check cases/2D_metaball
```

Checkpoints, fixed sample NPZs, complete training histories, generated fields,
and caches are intentionally excluded from Git.
