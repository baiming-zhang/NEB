# BasisNet

| Case | Modes | Result |
|---|---:|---:|
| [2D Karman](cases/2D_karman/README.md) | 96 | 5.81% mean velocity L2 |
| [2D airfoil](cases/2D_airfoil/README.md) | 1024 | 1.71% mean three-frame velocity L2 |
| [3D periodic](cases/3D_periodic_free/README.md) | 256 | 0.25% mean velocity L2 |
| [3D wake](cases/3D_parametric_wake/README.md) | 256 | 17.11% mean velocity L2 |

Shared implementation: [`basisnet.py`](basisnet.py).

```bash
python BasisNet/basisnet.py
python -m pytest -q BasisNet/tests/test_basisnet.py
```
