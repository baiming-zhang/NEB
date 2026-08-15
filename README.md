# EigenFluid-SSS

This repository currently publishes the cleaned CUDA implementation of the
two-dimensional parametric airfoil case.

## Published case

| Case | Backend | Basis | Results |
| --- | --- | ---: | --- |
| [2D airfoil](BasisNet/cases/2D_airfoil/README.md) | CuPy/CUDA, explicit forward and handwritten reverse pass | K=1024 | [step 5000](BasisNet/cases/2D_airfoil/publication/step5000_cuda/README.md) |
| 2D ten circles | CUDA | - | Planned |
| 3D ten balls | CUDA | - | Planned |

The airfoil trainer uses analytic compact third-order spatial jets, a
pure-vorticity Rayleigh objective, deterministic Sobol sampling, and no
PyTorch autograd in the production training path.

## Install

Python 3.11 or newer and an NVIDIA CUDA GPU are required for training. Install
the CuPy wheel that matches the local CUDA toolkit; CUDA 12 users can use the
provided extra:

```bash
python -m pip install -e ".[dev,cuda12]"
```

## Verify

```bash
python -m pytest -q
python -m ruff check BasisNet/cases/2D_airfoil
```

Curated metrics, figures, and reports are versioned. Generated samples,
checkpoints, logs, frames, and videos remain excluded from Git.
