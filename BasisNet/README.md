# BasisNet

BasisNet learns parameter-dependent spatial basis functions for fluid
problems. The generic implementation supports fixed or dynamically sorted
Sequential Spectral Subspace peeling:

1. one neural field and one optimizer are assigned to each mode;
2. each new candidate is peeled against all earlier modes;
3. earlier modes are detached during later-mode optimization;
4. the velocity, streamfunction, and velocity gradient are peeled with the
   same least-squares coefficients;
5. the normalized Rayleigh energy is minimized.

## Generic algorithm core

[`basisnet.py`](basisnet.py) is the standalone, case-independent reference
implementation. It contains the shared algorithm rather than a Taylor--Green,
ABC, cylinder, geometry, or boundary-specific trainer. Its public components
include:

- independent Fourier/SIREN potential networks and Adam optimizers;
- 2D streamfunction and 3D antisymmetric-tensor hard-divergence-free fields;
- commuting centered differences for velocity and velocity gradients;
- fixed or dynamic pre-projection Rayleigh-energy ordering;
- least-squares residual peeling with gradient-causal stop gradients;
- orthogonality, effective-rank, projection, energy, and divergence metrics;
- one-shot frozen-basis inference and weighted velocity projection/reconstruction;
- validation-driven learning-rate reduction and material-improvement early
  stopping.

Cases provide only continuous collocation points, physical parameters,
quadrature weights, and an optional analytic boundary envelope:

```python
import torch

from BasisNet.basisnet import BasisNet, BasisNetConfig, make_optimizers, sss_step

config = BasisNetConfig(
    spatial_dim=3,
    parameter_dim=1,
    modes=256,
    ordering="dynamic",
)
model = BasisNet(config).cuda()
optimizers = make_optimizers(model, learning_rate=1.0e-4)

loss, metrics, basis = sss_step(
    model,
    points=continuous_points,
    parameters=torch.tensor([alpha], device="cuda"),
    weights=quadrature_weights,
    optimizers=optimizers,
    envelope=analytic_boundary_envelope,
)
```

Run the standalone verification and unit tests from the repository root:

```powershell
python BasisNet/basisnet.py
python -m pytest -q BasisNet/tests/test_basisnet.py
```

The `2D_airfoil` case also supplies a case-local, explicitly written CuPy
backend. It uses a fixed computation graph, fused value/gradient/Hessian jets,
FP16 shadow weights for MLP matrix products, and FP32 master weights and Adam
state. This changes the execution backend rather than the mathematical SSS
protocol: fixed peeling, Rayleigh minimization, divergence-free construction,
and Ritz postprocessing retain their documented semantics.

## Cases

- [2D_karman](cases/2D_karman/README.md): a translating cylinder in a
  two-dimensional channel, K=96 learned modes, and a completed two-location
  Re=100 wake benchmark with `6.13641%` mean velocity error.
- [2D_airfoil](cases/2D_airfoil/README.md): a jointly thickness/angle
  parameterized analytic airfoil, K=1024 shared modes, a velocity-only
  Rayleigh objective, and a completed three-case reconstruction benchmark
  against independent D2Q9 Navier--Stokes GT.
- [3D_periodic_free](cases/3D_periodic_free/README.md): K=256
  hard-discrete-divergence-free modes on a periodic cube without obstacles,
  with independent forced Taylor--Green and oscillatory ABC reconstruction
  benchmarks (`1.09998%` and `0.246989%` mean velocity error).
- [3D_parametric_wake](cases/3D_parametric_wake/README.md): K=256
  conditional hard-discrete-divergence-free modes for a continuously
  parameterized sphere-to-cube obstacle; the best completed Re=500 wake
  benchmark has `17.1100%` output-grid mean velocity error.

Case-specific configurations, scripts, tests, and documentation are kept
inside each case directory so that experiments remain reproducible and
self-contained. See the [case index](cases/README.md) for status, validated
reference results, and artifact policy.

Run case-level verification from the repository root:

```powershell
python -m pytest -q BasisNet/cases/2D_karman/tests
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python -m pytest -q BasisNet/cases/3D_periodic_free/tests
python -m pytest -q BasisNet/cases/3D_parametric_wake/tests
python -m ruff check BasisNet/cases
```
