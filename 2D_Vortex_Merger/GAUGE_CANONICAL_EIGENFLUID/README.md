# GAUGE-CANONICAL EIGENFLUID

The [completed 64/1024-mode video comparison](outputs/comparison_64_1024/REPORT.md)
contains two-row GT/canonical/no-gauge/Laplacian videos, all six final MSE
results, and direct potential/velocity Laplacian eigenrelation checks.

Latest result: [canonical versus no-gauge paired training](outputs/gauge_ablation/REPORT.md).
Both arms use identical epoch-0 weights, periodic architecture, quadrature,
optimizer and a 2000-epoch budget at K=64. The only mathematical switch is
potential mean projection and its backward adjoint.

For seed 20260907, canonical reduces mean rollout velocity MSE by 2.18%,
final velocity MSE by 5.19%, and mean vorticity MSE by 5.06%. Intermediate
checkpoints are not uniformly better; this is one matched seed, not a
multi-seed significance claim. Full metrics and artifacts are in the report.

The earlier [analytic Laplacian verification](REPORT.md) establishes gauge
invariance and compares canonical neural modes with exact Fourier modes.
That earlier comparison alone is not a gauge ablation.
