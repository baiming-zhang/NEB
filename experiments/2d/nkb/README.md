# Neural Kinematic Bases: 2D adaptation

The original geometry-only NKB loss combines smoothness, divergence, global
off-diagonal Gram orthogonality, boundary, mean length and small-field penalties.
The adaptation uses seven hidden layers of width 256, analytic coordinate
Jacobians, handwritten parameter reverse, Adam, 2000 updates and validation
every 25 updates. Saved FP32 parameters are evaluated in FP64 for physical
operators. The protocol and target mean length are written to each run config.

```sh
python experiments/2d/nkb/run.py prepare --case periodic --K 1024 --seed 0 --output outputs/nkb
python experiments/2d/nkb/run.py train --case periodic --K 1024 --seed 0 --output outputs/nkb
python experiments/periodic/run.py evaluate --method NKB --checkpoint checkpoints/2d/NKB_periodic_k1024_seed0/checkpoint_e2000.npz --gt outputs/periodic/gt --output outputs/periodic/nkb
```

For each multihole case, use the same `prepare` and `train` commands with its
case name. Then project the common twenty fields and construct the original
curl-form operators from the reference-generated physical quadrature:

```sh
python experiments/multihole/evaluate.py --basis-family NKB --checkpoint outputs/nkb/one_circle_k1024_seed1/checkpoint_e2000.npz --fields outputs/multihole/one_circle/fields --quadrature outputs/multihole/one_circle/operators/quadrature.npz --output outputs/multihole/one_circle/nkb_projection
python experiments/2d/nkb/run.py operators --case one_circle --checkpoint outputs/nkb/one_circle_k1024_seed1/checkpoint_e2000.npz --fields outputs/multihole/one_circle/fields --quadrature outputs/multihole/one_circle/operators/quadrature.npz --output outputs/multihole/one_circle/nkb_operators
python experiments/2d/nkb/run.py dynamics --operators outputs/multihole/one_circle/nkb_operators --dt .0025 --output outputs/multihole/one_circle/nkb_dynamics
```

Operator generation retains every mode and quadrature point and writes
symmetric-input-pair quadratic coefficients in bounded output slabs.
Dynamics uses the original implicit-viscosity midpoint method with Re=750,
51 snapshots through T=1, and full mass solves. Repeat at dt=0.005 for the
temporal check. CPU checks exercise analytic derivatives and all 48 state
arrays without importing CuPy.
