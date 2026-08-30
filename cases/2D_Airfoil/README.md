# 2D Airfoil

This case trains one GEN mapping over airfoil thickness and angle of attack.
The analytic CuPy/CUDA backend uses explicit forward and reverse passes to
construct divergence-free modes without flow snapshots.

## Fieldwise K=2048 protocol

The three matched runs share the seed, geometry schedule, samples,
initialization, and optimizer state. Only the Rayleigh field changes:

| Method | Objective | Jet order | Config |
| --- | --- | ---: | --- |
| Potential-Rayleigh | `||grad A||^2 / ||A||^2` | 1 | [`k2048_airfoil_strict_potential_rayleigh_2000.json`](configs/k2048_airfoil_strict_potential_rayleigh_2000.json) |
| Velocity-Rayleigh | `||grad u||^2 / ||u||^2` | 2 | [`k2048_airfoil_strict_velocity_rayleigh_2000.json`](configs/k2048_airfoil_strict_velocity_rayleigh_2000.json) |
| Vorticity-Rayleigh | `||grad omega||^2 / ||omega||^2` | 3 | [`k2048_airfoil_strict_vorticity_rayleigh_2000.json`](configs/k2048_airfoil_strict_vorticity_rayleigh_2000.json) |

All runs use continuous PCG64 samples, analytic boundary envelopes, stable
raw-Rayleigh ordering, and causal modified Gram--Schmidt.

```bash
python cases/2D_Airfoil/scripts/run_airfoil_strict_rayleigh_k2048_paired_e2000.py
```

## Reconstruction and rollout

Evaluation uses three unseen Re=1500 geometries at
`(beta, alpha)=(0.50,+10 deg),(0.85,0 deg),(0.50,-10 deg)`. GEN and the
finite-difference comparator share the domain mask, mode count, fitting data,
evaluation data, and discrete operators.

Across the 30-frame K=2048 rollout, Potential-Rayleigh reaches `4.07%` mean
velocity error versus `22.01%` for matched FD Laplacian. Final-frame errors
are `5.46%` and `46.56%`. The complete tables and figures are in
[`../../results/2D_Airfoil`](../../results/2D_Airfoil/README.md).

The rollout uses a shared masked-Cartesian incidence complex and an implicit
trapezoidal coefficient update with exact viscous decay:

```bash
python cases/2D_Airfoil/scripts/rollout_gt30_fd_vs_potential_k2048.py
```

The case also includes the C2/Stokes path (`c2_backend.py`), matched Laplacian
ROM, neural-kinematic comparison, and Jet1/Jet2/Jet3 derivative tests.

## Verification

```bash
python -m pytest -q cases/2D_Airfoil/tests
python -m ruff check cases/2D_Airfoil
```
