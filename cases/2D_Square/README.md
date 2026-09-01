# 2D Square

This case isolates the hidden activation on the unit square. GEN uses the
same unsupervised normalized Dirichlet/Rayleigh loss and causal peeling as the
other cases; exact eigenfunctions are used only after training for evaluation.

For streamfunction Dirichlet data,

`psi_mn = sin(m pi x) sin(n pi y)` and `lambda_mn = pi^2 (m^2 + n^2)`.

The velocity is `u = (partial_y psi, -partial_x psi)`. The envelope
`x(1-x)y(1-y)` enforces the wall condition exactly. Degenerate eigenspaces are
aligned by orthogonal Procrustes matching.

```bash
python cases/2D_Square/scripts/run_activation_ablation.py
```

## Matched-seed K=8 result

The 3,000-step seed-17 comparison fixes initialization, Fourier features,
network size, collocation stream, optimizer, and loss. Only the hidden
activation changes.

| Metric | Sine | ELU |
| --- | ---: | ---: |
| Effective raw rank | 8 | 8 |
| Minimum raw velocity mass | 3.074e-3 | 2.663e-3 |
| Mean eigenvalue relative error | **0.844%** | 0.926% |
| Mean eigenspace-aligned velocity rL2 | **4.394%** | 5.621% |
| Mean 36-mode spectral capture | **99.9821%** | 99.9788% |
| Highest learned mode aligned rL2 | **4.910%** | 20.920% |

Sine achieves the lowest aligned errors while both activations retain full
rank and near-complete analytical spectral capture.

## Verification

```bash
python -m pytest -q cases/2D_Square/tests
python -m ruff check cases/2D_Square
```

## Taylor--Green rollout

The reproducible `(2,2)` four-cell Taylor--Green implementation and its
published learned-mode cache are in
[`taylor_green/`](taylor_green/README.md).  The corresponding data, figures,
metrics, and video are under [`results/2D_Square`](../../results/2D_Square/README.md).
