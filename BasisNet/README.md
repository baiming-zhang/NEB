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

## Cases

- [2D_karman](cases/2D_karman/README.md): a translating cylinder in a
  two-dimensional channel, K=96 learned modes, and Re=100 wake
  reconstruction.
- [3D_periodic_free](cases/3D_periodic_free/README.md): K=256
  hard-discrete-divergence-free modes on a periodic cube without obstacles,
  with independent forced Taylor--Green and oscillatory ABC reconstruction
  benchmarks.

Case-specific configurations, scripts, tests, and documentation are kept
inside each case directory so that experiments remain reproducible and
self-contained. See the [case index](cases/README.md) for status, validated
reference results, and artifact policy.
