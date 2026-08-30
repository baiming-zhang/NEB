# EigenFluid-SSS

EigenFluid-SSS learns one geometry-conditioned spatial basis mapping and uses
that basis for reconstruction and reduced fluid dynamics. The repository is
now flat: the generic implementation is [`basisnet.py`](basisnet.py), all
reproducible experiments live in [`cases/`](cases/README.md), and reviewed
paper artifacts live in [`results/`](results/README.md). The former
`BasisNet/` wrapper has been removed.

The basis is trained without flow snapshots. Continuous potential fields,
analytic boundary envelopes, causal peeling, and field-specific Rayleigh
objectives produce divergence-free modes that are evaluated only afterward
against independent analytic, finite-difference, or Navier--Stokes targets.

## Latest paper-aligned results

| Benchmark | Latest evaluated basis | Headline result |
| --- | ---: | --- |
| [2D connected metaball](results/2d_metaball_k1024_k2048/README.md) | K=2048, epoch 1000 | streamfunction/velocity/vorticity held-out rL2 = `0.01183 / 0.06998 / 0.36447`; matched FD = `0.01235 / 0.05674 / 0.51569` |
| [3D connected metaball](results/3d_metaball_k1024/README.md) | K=1024 | five-method comparison on all `1,599,754` fluid nodes of a `128^3` grid; strict FD velocity rL2 `0.15965`, Potential-Rayleigh `0.17829` |
| [2D airfoil ROM](results/airfoil_k2048_rollout30/README.md) | K=2048, 30 predicted frames | mean velocity error `4.07%` for Potential-Rayleigh versus `22.01%` for matched FD Laplacian |
| [3D parametric fan](cases/3D_parametric_fan/METHOD.md) | K=1024 | complete evaluation bundle, parameter audit, convergence curves, and PNG/PDF field slices are versioned with the case |

The 2D and 3D metaball publication bundles are synchronized from the latest
local experiments and the paper repository's artifact commit `ecff940`,
respectively. The current paper `master` (`d9e68c7`) subsequently removed the
3D bundle from the writing-only repository; its identical paper figure now
lives on the manuscript branch at `b11d90a` of
[`Zhecheng-Wang/neural-eigenfluid-paper`](https://github.com/Zhecheng-Wang/neural-eigenfluid-paper),
while the complete reproducibility bundle remains here. The airfoil figures
and metrics are synchronized from the paper's
`codex/anti-defensive-writing-revision` branch.

## Repository layout

```text
eigenfluid-sss/
|-- basisnet.py                  # generic SSS basis implementation
|-- cases/
|   |-- 2D_airfoil/              # CUDA Jet1/Jet2/Jet3 training and ROMs
|   |-- 2D_metaball/             # K=64--2048 connected-obstacle studies
|   |-- 2D_square_analytic/      # exact-spectrum activation checks
|   |-- 3D_metaball_connected/   # connected-sphere training/evaluation
|   `-- 3D_parametric_fan/       # optimized 3D fan benchmark
|-- results/                     # compact reviewed publication artifacts
|-- tests/                       # generic basis tests
|-- pyproject.toml
`-- README.md
```

Most large checkpoints, raw GT fields, sampled basis volumes, logs, caches,
and videos remain local. The exact K=2048 airfoil Potential-Rayleigh reference
needed by the paired-protocol audit is the deliberate exception: its initial
state, frozen checkpoint, and fixed validation pack are versioned with Git LFS.
Small configs, metrics, manifests, PNGs, and vector PDFs needed to audit the
reported claims are versioned directly in Git.

## Installation

Python 3.11 or newer is required. CUDA training uses an NVIDIA GPU and the
CuPy wheel matching the installed CUDA toolkit; CUDA 12 users can install the
provided extra.

```bash
git clone https://github.com/baiming-zhang/eigenfluid-sss.git
cd eigenfluid-sss
git lfs install
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cuda12]"
```

CPU-only analytic and generic tests do not require the `cuda12` extra.

## Verification

```bash
python -m pytest -q tests
python -m pytest -q cases/2D_square_analytic/tests
python -m pytest -q cases/2D_metaball/tests
python -m pytest -q cases/3D_metaball_connected/tests
python -m pytest -q cases/2D_airfoil/tests
python -m pytest -q cases/3D_parametric_fan/tests
python -m ruff check basisnet.py tests cases
```

GPU-only tests skip automatically when their backend is unavailable. Exact
training and evaluation entry points are documented in each case README.

## Artifact integrity

Each publication result directory contains machine-readable metrics and a
SHA-256 manifest. Recompute the hashes from the repository root with:

```powershell
Get-ChildItem results -Recurse -File |
  Get-FileHash -Algorithm SHA256
```
