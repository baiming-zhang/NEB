# Learned P/PH dynamics

The original analytic Jet2 kernels generate every retained positive and harmonic coordinate. Complete physical quadrature defines the global mass and curl stiffness matrices. The original all-point weak-Hodge tensor factorization advances twenty independent coefficient states with full-mass IMEX midpoint: Re = 750, dt = 0.001, T = 1, snapshots every 0.02. Gradient and vorticity arrays use disk-backed FP64 storage, and native contractions stream 1024-point blocks. Checkpoint parameters use the original FP64 execution policy.

From the repository root, after generating fields, reference quadrature, and the positive/harmonic models:

```sh
python -m experiments.multihole.learned_rom.run --mode operators --checkpoint outputs/multihole/one_circle/train/checkpoints/hardcoded_epoch2000.npz --harmonic-checkpoint outputs/multihole/one_circle/harmonic/checkpoints/harmonic_epoch2000.npz --quadrature outputs/multihole/one_circle/operators/quadrature.npz --fields outputs/multihole/one_circle/fields --output outputs/multihole/one_circle/learned_operators
python -m experiments.multihole.learned_rom.run --mode dynamics --operators outputs/multihole/one_circle/learned_operators --output outputs/multihole/one_circle/learned_dynamics
python -m experiments.multihole.reference.run --mode trajectory --case one_circle --mesh outputs/multihole/one_circle/mesh/mesh.npz --fields outputs/multihole/one_circle/fields --sample 0 --output outputs/multihole/one_circle/reference0
python -m experiments.multihole.learned_rom.compare --operators outputs/multihole/one_circle/learned_operators --trajectory outputs/multihole/one_circle/learned_dynamics/trajectory.npz --reference outputs/multihole/one_circle/reference0/trajectory.npz --reference-stiffness outputs/multihole/one_circle/reference0/stiffness.npz --sample 0 --output outputs/multihole/one_circle/comparison0
```

The comparator reconstructs analytic potentials on the reference nodes and reports their consistent reference-stiffness velocity energy error. It saves PNG and vector PDF. Repeat samples 0 through 19 for paired trajectory statistics. Temporal and spatial reference refinements retain their separate measurements. Positive-only P uses the positive checkpoint; PH adds the harmonic checkpoint. `--total-K` together with the actual `--causal-order` receipt selects P(K-h)H(h) from a larger trained positive model.

```sh
python -B -m experiments.multihole.learned_rom.check_cpu
```

The small CPU oracle compares the complete analytic tensor with its all-point factorization and the original full-mass integrator, then runs the public dynamics parser/read-write return using an explicitly declared NumPy API double. Native model evaluation and full-size trajectories run with the CUDA dependency.

## Compact-vortex and circulation controls

The large-vortex ensemble uses the original compact Wendland C4 vorticity,
PCG64 sample hierarchy, harmonic energy ratio, and peak-speed normalization.
Generate the twenty P1 reference initial states first:

```sh
python -m experiments.multihole.reference.run --mode vortex-trajectory --case one_circle --mesh outputs/multihole/one_circle/mesh/mesh.npz --sample 0 --output outputs/multihole/one_circle/vortex_gt/sample00
```

Repeat the sample index/output suffix for 1 through 19. The reference saves
`initial_state.npz`, `initial_specification.json`, and the complete trajectory.
Use its same mesh to generate quadrature. The learned operator generator reads
those original P1 cell velocities in the exact quadrature cell order:

```sh
python -m experiments.multihole.learned_rom.run --mode operators --field-ensemble compact-vortex --checkpoint outputs/multihole/one_circle/train/checkpoints/hardcoded_epoch2000.npz --harmonic-checkpoint checkpoints/2d/one_circle_harmonic_seed1/harmonic_epoch2000.npz --quadrature outputs/multihole/one_circle/operators/quadrature.npz --fields outputs/multihole/one_circle/vortex_gt --output outputs/multihole/one_circle/vortex_learned_operators
python -m experiments.multihole.learned_rom.run --mode dynamics --operators outputs/multihole/one_circle/vortex_learned_operators --output outputs/multihole/one_circle/vortex_learned_dynamics
```

For the scaled-lift control choose the corresponding scaled harmonic checkpoint
and repeat the operator/dynamics commands in a separate output directory.
The comparator uses each sample's reference `stiffness.npz` and trajectory.

```sh
python -B -m experiments.multihole.learned_rom.check_compact_cpu
```
