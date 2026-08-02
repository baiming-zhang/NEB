# BasisNet cases

This directory contains the reproducible BasisNet benchmarks. Each case keeps
its configuration, training and evaluation entry points, implementation,
tests, and detailed protocol together. Generated datasets, checkpoints,
frames, and videos are intentionally excluded from Git.

## Available cases

| Case | Domain and target | Learned basis | Independent evaluation | Status |
| --- | --- | ---: | --- | --- |
| [2D_karman](2D_karman/README.md) | Translating cylinder in a 2D channel | K=96 | Re=100 Karman-wake HDF5 data | Reproducible workflow; no version-controlled reference metric yet |
| [2D_airfoil](2D_airfoil/README.md) | Analytic thickness and angle jointly parameterized in a 2D far field | K=1024 | Independent Re=1500 D2Q9-TRT NS trajectories | Preliminary K1024 hybrid benchmark and matched velocity-only ablation |
| [3D_periodic_free](3D_periodic_free/README.md) | Triply periodic cube without obstacles | K=256 | Exact forced Taylor--Green and oscillatory ABC GT | Completed and benchmarked |
| [3D_parametric_wake](3D_parametric_wake/README.md) | Continuously morphed sphere-to-cube obstacle in a driven periodic 3D flow | K=256 | Independent Re=500 pseudo-spectral NS trajectory | Reproducible workflow; final 10000-epoch benchmark to be generated |

Basis optimization is unsupervised: independent Navier--Stokes or analytic GT
fields are used only after the learned basis has been frozen. See each case
README for the exact training, checkpoint-selection, projection, evaluation,
and rendering commands.

## Validated training runs

The current airfoil experiment uses a 10,000-epoch hybrid budget and a matched
1,000-epoch velocity-only ablation. Generated checkpoints are deliberately
excluded from Git; the source, configurations, fixed validation protocol,
compact result record, curated figures, and tests required to reproduce the
comparison are versioned.

| Case | Modes | Epoch budget | Observed epoch | Selected epoch | Selected fixed-validation objective | Selected rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2D_airfoil hybrid | 1024 | 10,000 | 4,300 | 650 | 0.528030 | 1024 |
| 2D_airfoil velocity-only ablation | 1024 | 1,000 | 200 snapshot | 200 snapshot | 1.699339 | 1024 |

The hybrid fixed-validation value is
`0.5 (R_u / 80) + 0.5 (R_omega / 232)`, while the velocity-only value is
`R_u / 80`; neither is an independent Navier--Stokes reconstruction error.
The ablation row is explicitly an in-progress epoch-200 snapshot.

## Validated reference results

Only results with a documented independent GT generator or data source and a
reproducible evaluation path are listed here. Errors below use the raw learned
basis without spectral projection or per-snapshot fitting beyond the stated
linear basis projection.

| Case and benchmark | Checkpoint | Mean velocity relative error | Additional validated result |
| --- | --- | ---: | --- |
| 2D_airfoil, three Re=1500 cases | Hybrid best, epoch 650 | 1.94527% | Rectangular Laplacian K1024: 2.98104%; learned basis is 34.74% lower |
| 3D_periodic_free, forced Taylor--Green, Re=500 | Latest, epoch 10000 | 1.09998% | Hard discrete divergence 0; spectrum error 0.02478%; 4096-point inference 0.499 s |
| 3D_periodic_free, forced Taylor--Green, Re=500 | Validation best, epoch 9800 | 1.21760% | Hard discrete divergence 0; spectrum error 0.03042%; 4096-point inference 0.501 s |
| 3D_periodic_free, oscillatory ABC k=1, Re=500, 50 frames | Latest, epoch 10000 | about 0.247% | Maximum frame error about 0.268% |

The latest 3D checkpoint is the formal Taylor--Green result because it
reconstructs the independent GT more accurately than the checkpoint selected
by validation Rayleigh loss. The oscillatory ABC benchmark uses the same
frozen epoch-10000 basis and introduces no retraining.

The 2D_karman case is included as a complete reproducible workflow, but its
README does not currently record a final reference-error table. It is
therefore not assigned an undocumented score here. Experimental flows with
poor reconstruction or without an independent GT are likewise omitted.

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
