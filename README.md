# Eigenfluid

Eigenfluid introduces the **Geometry-conditioned Eigenfluid Net (GEN)**: one
neural mapping from geometry parameters and spatial coordinates to a
divergence-free spectral basis. GEN is trained without flow snapshots and
supports fast reconstruction and reduced-order dynamics across changing
geometries.

The method combines continuous potential fields, analytic boundary envelopes,
causal mode peeling, and field-specific Rayleigh objectives. Independent
analytic, finite-difference, and Navier--Stokes targets are used only for
evaluation.

## Results

| Case | Basis | Main result |
| --- | ---: | --- |
| [2D Metaball](results/2D_Metaball/README.md) | K=2048, epoch 1000 | held-out streamfunction/velocity/vorticity rL2 `0.01183 / 0.06998 / 0.36447`; matched FD `0.01235 / 0.05674 / 0.51569` |
| [3D Metaball](results/3D_Metaball/README.md) | K=1024 | five-method comparison on all `1,599,754` fluid nodes of a `128^3` grid |
| [2D Airfoil](results/2D_Airfoil/README.md) | K=2048, strict paired objectives | complete Potential/Velocity/Vorticity training, nested prefixes 64--2048, and 30-frame common-dynamics evaluation |

## Repository layout

```text
eigenfluid-sss/
|-- gen.py             # generic GEN implementation
|-- cases/
|   |-- 2D_Airfoil/
|   |-- 2D_Metaball/
|   |-- 2D_Square/
|   `-- 3D_Metaball/
|-- results/           # publication metrics and figures
|-- tests/
|-- pyproject.toml
`-- README.md
```

## Installation

Python 3.11 or newer is required. CUDA training uses an NVIDIA GPU and the
CuPy wheel matching the installed CUDA toolkit.

```bash
git clone https://github.com/baiming-zhang/eigenfluid-sss.git
cd eigenfluid-sss
git lfs install
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,cuda12]"
```

CPU workflows can omit the `cuda12` extra.

## Verification

```bash
python -m pytest -q tests
python -m pytest -q cases/2D_Square/tests
python -m pytest -q cases/2D_Metaball/tests
python -m pytest -q cases/3D_Metaball/tests
python -m pytest -q cases/2D_Airfoil/tests
python -m ruff check gen.py tests cases
```

Each publication bundle under [`results/`](results/README.md) includes
machine-readable metrics, figures, and a SHA-256 manifest. The K=2048 Airfoil
reference inputs and checkpoint are tracked with Git LFS.
