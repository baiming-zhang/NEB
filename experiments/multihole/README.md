# Multiply connected domains

`one_circle`, `circle_square` and `circle_square_triangle` have one, two and
three holes. The original native analytic-Jet positive modes and learned
harmonic extensions remain separate. Full PH allocation uses the actual final
causal order when selecting positive columns.

```sh
python experiments/multihole/run.py fields --case one_circle --output outputs/multihole/one_circle/fields
python experiments/multihole/run.py prepare --config experiments/multihole/configs/one_circle_p1023_h1_e2000.json --output outputs/multihole/one_circle/init
python experiments/multihole/run.py train --checkpoint outputs/multihole/one_circle/init/initial_epoch0000.npz --output outputs/multihole/one_circle/train
python experiments/multihole/run.py harmonic-train --config experiments/multihole/configs/one_circle_harmonic_seed1.json --output outputs/multihole/one_circle/harmonic
python -m experiments.multihole.reference.run --help
python -m experiments.multihole.reference.run --mode mesh --case one_circle --output outputs/multihole/one_circle/mesh
python -m experiments.multihole.reference.run --mode operators --case one_circle --mesh outputs/multihole/one_circle/mesh/mesh.npz --output outputs/multihole/one_circle/operators
python experiments/multihole/evaluate.py --checkpoint outputs/multihole/one_circle/train/checkpoints/hardcoded_epoch2000.npz --harmonic-checkpoint outputs/multihole/one_circle/harmonic/checkpoints/harmonic_epoch2000.npz --fields outputs/multihole/one_circle/fields --quadrature outputs/multihole/one_circle/operators/quadrature.npz --output outputs/multihole/one_circle/projection
```

The field generator uses the original quintic spline law with seed hierarchy
`[20260928, topology_index, field_index]`, twenty independent fields per
topology and one global maximum-speed scale per field. These are evaluation
fields. Training uses the original geometry-only objective. The reference subdirectory builds
the conforming mesh and original FEM operators/time evolution from geometry.

Evaluation accepts `quadrature.npz` with `points[N,2]` and positive `weights[N]`,
the generated fields, an epoch-2000 checkpoint and optional harmonic checkpoint.
It streams analytic gradients, applies one physical curl convention, solves
the full mass system and makes a second pass for direct physical residuals.
It saves twenty relative L2 errors and sample SD (`ddof=1`). The full raw-span mass system preserves all retained coordinates. Run the
supplied reference refinements to assess quadrature and trajectory convergence.

For the separate random large-vortex ensemble, use the original reference
initializer exposed by `reference` for its compact-Wendland vortex ensemble.

The learned P/PH autonomous operator and trajectory recipe is in
[`learned_rom/README.md`](learned_rom/README.md).
