# 2D Metaball

This case trains a geometry-conditioned streamfunction basis in a rounded
square containing a smooth connected union of three circles. The three circle
centers and radii define the geometry input to GEN.

## Scaling runs

| Modes | First SIREN frequency | Epoch | Config |
| ---: | ---: | ---: | --- |
| 1024 | 160 | 2000 | [`k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json`](configs/k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json) |
| 2048 | 224 | 1000 | [`k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json`](configs/k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json) |

Both runs use 4096 continuous fluid samples per epoch, four fixed validation
geometries, batches of 32 modes, FP32 arithmetic, the velocity-Rayleigh
objective, and one causal velocity-mass MGS pass.

```bash
python cases/2D_Metaball/scripts/run_metaball_velocity_rayleigh_omega160_k1024_e2000.py
python cases/2D_Metaball/scripts/run_metaball_velocity_rayleigh_omega224_k2048_e2000.py
```

## Held-out comparison

Three unseen geometries are evaluated on a `128 x 128` grid. Velocity
coefficients are fit on 70% of fluid nodes and all errors are measured on the
disjoint 30%. GEN and FD share the same K, mask, targets, split, discrete
operators, and velocity-mass whitening.

| Method | K | Streamfunction | Velocity | Vorticity |
| --- | ---: | ---: | ---: | ---: |
| FD Laplacian | 1024 | **0.02122** | **0.09423** | 0.62079 |
| GEN, epoch 2000 | 1024 | 0.05622 | 0.16255 | **0.58754** |
| FD Laplacian | 2048 | 0.01235 | **0.05674** | 0.51569 |
| GEN, epoch 1000 | 2048 | **0.01183** | 0.06998 | **0.36447** |

At K=2048, GEN improves streamfunction error by `4.16%` and vorticity error
by `29.32%` over the matched FD basis. Machine-readable results are in
[`../../results/2D_Metaball`](../../results/2D_Metaball/README.md).

## Verification

```bash
python -m pytest -q cases/2D_Metaball/tests
python -m ruff check cases/2D_Metaball
```
