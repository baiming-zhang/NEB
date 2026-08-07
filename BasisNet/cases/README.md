# BasisNet cases

This directory contains the reproducible BasisNet benchmarks. Each case keeps
its configuration, training and evaluation entry points, implementation,
tests, and detailed protocol together. Generated datasets, checkpoints,
frames, and videos are intentionally excluded from Git.

## Available cases

| Case | Domain and target | Learned basis | Independent evaluation | Status |
| --- | --- | ---: | --- | --- |
| [2D_karman](2D_karman/README.md) | Translating cylinder in a 2D channel | K=96 | Re=100 Karman-wake HDF5 data | Completed K96 velocity-fit benchmark |
| [2D_airfoil](2D_airfoil/README.md) | Analytic thickness and angle jointly parameterized in a 2D far field | K=1024 | Independent Re=1500 D2Q9-TRT NS trajectories | Completed K1024 velocity-only benchmark |
| [3D_periodic_free](3D_periodic_free/README.md) | Triply periodic cube without obstacles | K=256 | Exact forced Taylor--Green and oscillatory ABC GT | Completed K256 benchmark |
| [3D_parametric_wake](3D_parametric_wake/README.md) | Analytic sphere-to-cube obstacle in a driven 3D wake | K=256 | Independent Re=500 NS trajectory | Completed K256/Re500 fixed-order benchmark; newer dynamic target pending |

Basis optimization is unsupervised: independent Navier--Stokes or analytic GT
fields are used only after the learned basis has been frozen. See each case
README for the exact training, checkpoint-selection, projection, evaluation,
and rendering commands.

## Validated training runs

Generated checkpoints are deliberately excluded from Git. For each published
case, the source, configuration, compact result record, curated figures, and
tests required to audit the selected result are versioned.

| Case | Modes | Epoch budget | Observed epoch | Selected epoch | Selected fixed-validation objective | Selected rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D_karman velocity-fit | 96 | 5,000 | 5,000 | 4,950 | 0.305099 | 96 |
| 2D_airfoil velocity-only | 1024 | 1,000 | 1,000 | 1,000 | 1.378959 | 1024 |
| 3D_periodic_free | 256 | 10,000 | 10,000 | 10,000 | 0.113886 | 256 |
| 3D_parametric_wake fixed-order | 256 | 5,000 | 5,000 | 5,000 | 1.902128 | 256 |

The airfoil fixed-validation value is `R_u / 80`; it is not an independent
Navier--Stokes reconstruction error. No vorticity term contributes to training.

## Validated reference results

Only results with a documented independent GT generator or data source and a
reproducible evaluation path are listed here. Errors below use the raw learned
basis without spectral projection or per-snapshot fitting beyond the stated
linear basis projection.

| Case and benchmark | Checkpoint | Mean velocity relative error | Additional validated result |
| --- | --- | ---: | --- |
| 2D_karman, two Re=100 locations | Velocity-fit, epoch 4950 | 6.13641% | Best displayed location `alpha=0.40`: 5.80808%; mean vorticity error 20.0683% |
| 2D_airfoil, three Re=1500 cases | Velocity-only, epoch 1000 | 1.88709% | Mean wake velocity error 3.31803%; all coefficient fits use velocity L2 only |
| 3D_periodic_free, forced Taylor--Green, Re=500 | Epoch 10000 | 1.09998% | Hard discrete divergence 0; spectrum error 0.02478%; 4096-point inference 0.499 s |
| 3D_periodic_free, oscillatory ABC k=1, Re=500, 50 frames | Epoch 10000 | 0.246989% | Maximum frame error 0.267732% |
| 3D_parametric_wake, `alpha=0.1`, Re=500, 30 frames | Fixed-order, epoch 5000 | 17.1100% | Projection mean 16.5121%; maximum frame error 17.1647% |

The epoch-10000 periodic checkpoint is the formal result because it has the
lowest independent Taylor--Green error among completed checkpoints; the ABC
benchmark uses that same frozen basis. The published wake result uses the
exact fixed-order localized-channel snapshot in
`3D_parametric_wake/configs/k256_re500_published.json`. The newer dynamic
periodic-obstacle target in `k256_final.json` has no completed benchmark and
is not mixed with the published metrics. Ablations and incomplete runs are
omitted.

## Verification

Run the complete case test suite and static checks from the repository root:

```powershell
python -m pytest -q BasisNet/cases/2D_karman/tests
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python -m pytest -q BasisNet/cases/3D_periodic_free/tests
python -m pytest -q BasisNet/cases/3D_parametric_wake/tests
python -m ruff check BasisNet/cases
```

Case-specific smoke tests and end-to-end commands are documented in the
linked READMEs.

## Artifact policy

Commit source code, small configurations, tests, and documentation. Do not
commit raw third-party data, generated GT HDF5 files, checkpoints, training
logs, PID files, caches, temporary frames, or large videos. Recreate these
artifacts with the documented commands, or distribute large trained weights
through a release, object storage, or Git LFS together with the exact config
and training history.
