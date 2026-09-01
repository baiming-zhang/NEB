# Eigenfluid cases

Each case provides configurations, training and evaluation entry points,
implementation modules, and tests for Geometry-conditioned Eigenfluid Net
(GEN).

| Case | GEN configuration | Evaluation | Result |
| --- | --- | --- | --- |
| [2D Airfoil](2D_Airfoil/README.md) | K=64--2048, analytic CUDA Jet1/Jet2/Jet3 | Re=1500 trajectories and matched FD Laplacian ROM | [30-frame K=2048 rollout](../results/2D_Airfoil/README.md) |
| [2D Metaball](2D_Metaball/README.md) | K=64--2048, SIREN with causal MGS | unseen geometries and matched FD spectrum | [K=1024/K=2048 comparison](../results/2D_Metaball/README.md) |
| [2D Square](2D_Square/README.md) | K=8/K=64, generic PyTorch GEN | exact unit-square Dirichlet spectrum | sine/ELU and SIREN/WIRE ablations |
| [3D Metaball](3D_Metaball/README.md) | K=64 local workflow; K=1024/K=2048 studies | reconstruction and physical NS/de Witt trajectories | [K=1024 and K=2048 results](../results/3D_Metaball/README.md) |

## Verification

Run from the repository root:

```bash
python -m pytest -q tests
python -m pytest -q cases/2D_Square/tests
python -m pytest -q cases/2D_Metaball/tests
python -m pytest -q cases/3D_Metaball/tests
python -m pytest -q cases/2D_Airfoil/tests
python -m ruff check gen.py tests cases
```
