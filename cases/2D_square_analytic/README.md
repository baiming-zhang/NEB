# 2D analytic square activation ablation

This case isolates the hidden activation on the classical unit square. The
training objective is the same unsupervised normalized Dirichlet/Rayleigh loss
and causal peeling used by the generic BasisNet. Ground-truth eigenfunctions
are used only after the model is frozen.

For streamfunction Dirichlet data,

`psi_mn = sin(m pi x) sin(n pi y)` and `lambda_mn = pi^2 (m^2 + n^2)`.

The velocity is `u = (partial_y psi, -partial_x psi)`. Multiplying the learned
streamfunction by `x(1-x)y(1-y)` enforces the square-wall condition exactly.
Degenerate analytical eigenspaces, such as `(1,2)/(2,1)`, are compared by an
orthogonal Procrustes alignment rather than by arbitrary individual signs or
rotations.

Run the default matched sine/ELU ablation from the repository root:

```powershell
python cases/2D_square_analytic/scripts/run_activation_ablation.py
```

The run writes checkpoints, CSV/JSON metrics, and Nature-style red-blue PNG
and vector PDF figures under `results/run_<timestamp>/`.

## Matched-seed K=8 result

The validated 3,000-step seed-17 ablation holds the initialization, Fourier
features, width, depth, collocation stream, optimizer, and loss fixed. Only the
hidden activation changes. Analytical matching searches the first 36 square
modes and aligns rotations inside degenerate eigenspaces.

| Metric | Sine | ELU |
| --- | ---: | ---: |
| Effective raw rank | 8 | 8 |
| Minimum raw velocity mass | 3.074e-3 | 2.663e-3 |
| Mean matched eigenvalue relative error | 0.844% | 0.926% |
| Mean eigenspace-aligned velocity relative L2 | 4.394% | 5.621% |
| Mean 36-mode analytical spectral capture | 99.9821% | 99.9788% |
| Highest learned mode aligned relative L2 | 4.910% | 20.920% |

Both activations learn the same eight analytical eigenspaces: `(1,1)`, the
`(1,2)/(2,1)` pair, the `(1,3)/(3,1)` pair, the `(1,4)/(4,1)` pair, and
`(1,5)`. Thus this finite-budget causal run skips some lower eigenvalues; mode
index must not be treated as the analytical index.

ELU does not collapse: it remains full-rank with raw masses many orders of
magnitude above the numerical floor and nearly exact analytical spectral
capture. It is modestly less sharp overall, and the highest learned mode shows
a clear localized deformation. The evidence supports "nontrivial but somewhat
blurrier," especially at higher frequency, not a broad loss of modal structure.
