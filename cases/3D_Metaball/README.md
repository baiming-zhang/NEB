# 3D Metaball

This case trains GEN in a rounded cube containing a connected smooth obstacle
formed by three pairwise-overlapping spheres. Each mode is represented by a
complete antisymmetric tensor potential with one external coefficient.

## Training and evaluation

The local PyTorch workflow trains K=64 Potential-Rayleigh bases with stable
geometry-specific ordering and two MGS passes. The selected configuration uses
seed `20260823`, 2000 epochs, 6144 fluid points per epoch, eight validation
geometries, SIREN frequencies `(6,36)`, and a squared boundary envelope.

```bash
python cases/3D_Metaball/scripts/train_validate.py \
  --config cases/3D_Metaball/configs/k64_three_sphere_connected_siren_dirichlet_2000_sweep_winner.json
python cases/3D_Metaball/scripts/solve_unseen_laplacian_128.py
python cases/3D_Metaball/scripts/evaluate_sweep_winner_128.py
```

## K=1024 comparison

| Method | Velocity rL2 | Vorticity rL2 |
| --- | ---: | ---: |
| Strict FD-Lap RR | **0.15965** | **0.57047** |
| Potential-Rayleigh | 0.17829 | 0.62882 |
| Velocity-Rayleigh | 0.28081 | 0.71109 |
| Vorticity-Rayleigh | 0.92911 | 1.18316 |
| Vector/Hodge-Lap RR | 0.56623 | 0.90298 |

All methods are evaluated on `1,599,754` fluid nodes of a `128^3` grid with
the same second-order curl and isovorticity level. Exact configurations,
metrics, timings, figures, and hashes are available in
[`../../results/3D_Metaball`](../../results/3D_Metaball/README.md).

## Verification

```bash
python -m pytest -q cases/3D_Metaball/tests
python -m ruff check cases/3D_Metaball
```
