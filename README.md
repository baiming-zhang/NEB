# EigenFluid-SSS

EigenFluid-SSS is a research codebase for learning parametric spatial
subspaces and, in future releases, reduced temporal dynamics for fluid flows.
The name combines **EigenFluid** with the **Sequential Spectral Subspace
(SSS)** construction used by the spatial model.

## Modules

| Module | Purpose | Current status |
| --- | --- | --- |
| [BasisNet](BasisNet/README.md) | Learns parameter-dependent spatial bases with fixed-order SSS peeling. | Implemented |
| [TimeNet](TimeNet/README.md) | Reserved for temporal coefficient models and reduced-order rollout. | Placeholder |

## Included cases

| Module | Case | Description |
| --- | --- | --- |
| BasisNet | [2D_karman](BasisNet/cases/2D_karman/README.md) | K=96 basis learning for a translating cylinder and reconstruction of a Re=100 Karman wake. |
| BasisNet | [3D_periodic_free](BasisNet/cases/3D_periodic_free/README.md) | K=256 hard-discrete-divergence-free basis learning on a periodic cube without obstacles, with forced Taylor--Green and oscillatory ABC evaluation. |
| BasisNet | [2D_airfoil](BasisNet/cases/2D_airfoil/README.md) | K=256 fixed-order SSS basis jointly conditioned on analytic airfoil thickness and angle of attack, with a manual fixed-graph trainer and independent D2Q9 Navier--Stokes evaluation. |

See the [BasisNet case index](BasisNet/cases/README.md) for benchmark status
and validated reference results.

## Repository layout

```text
EigenFluid-SSS/
|-- BasisNet/
|   |-- README.md
|   `-- cases/
|       |-- README.md
|       |-- 2D_karman/
|       |-- 2D_airfoil/
|       `-- 3D_periodic_free/
|-- TimeNet/
|   `-- README.md
|-- pyproject.toml
`-- README.md
```

Generated results, checkpoints, raw datasets, caches, frames, and videos are
excluded from version control.

## Installation

Python 3.11 or newer is required. A CUDA GPU is recommended for the full
K=256 airfoil training run; the smaller reference cases provide lighter-weight
verification paths.

```bash
git clone https://github.com/baiming-zhang/eigenfluid-sss.git
cd eigenfluid-sss
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest -q
python BasisNet/cases/2D_karman/scripts/train.py --self-test
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python BasisNet/cases/3D_periodic_free/scripts/train.py --self-test
```

See the [BasisNet case index](BasisNet/cases/README.md) for complete training,
restart, inference, evaluation, and video-generation workflows.

Only source code, canonical configurations, tests, and documentation belong in
Git. Generated checkpoints, GT fields, reconstructions, figures, videos, and
all `results/` directories remain local artifacts.
