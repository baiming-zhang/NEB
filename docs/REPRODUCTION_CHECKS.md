# Reproduction checks

The repository retains 89 trained model endpoints and three Potential-AL
multiplier archives. Every endpoint has its configuration and adjacent model
metadata. The saved weights, Adam moments and counters are retained; model
payloads live in the `paper` GitHub release.

## Executed checks

| Area | Check |
|---|---|
| All saved state | Opened all 92 NPZ files; checked finite parameters/moments, dimensions and optimizer endpoints |
| Every model | Loaded all 89 endpoints through the production model/field backends and evaluated every mode on four query points |
| Precision | Exercised FP32 and FP64 initialization, analytic spatial derivatives and Adam updates with small numerical fixtures |
| Numerical interfaces | 21 CPU tests, including full-mass projection, reference operators, Galerkin contractions, case/configuration coverage and complete restart state |
| CUDA interfaces | Five network, precision and gauge-extension checks |
| Data generation | Generated the complete 103,244 source points and 16,384 independent points; exercised fitted mesh/spectral generation on a small mesh |
| Download | Downloaded a release checkpoint through the public CLI and loaded its saved numerical state |
| Figures | Generated 13 matching PNG/PDF plot pairs from the supplied measurements |
| Manuscript | Compiled the supplied LaTeX source three times; all 53 pages matched the published PDF's text and word positions |

The checks ran in a separate source tree with an editable installation in a
new Python 3.13 environment on Windows. It reused installed NumPy/SciPy/CuPy
and CUDA libraries, and installed the missing FEM dependencies. The pinned
dependency set resolves through the documented installation commands.

## Runtime and naming fixes

- Backend paths describe their purpose, such as `periodic_core`,
  `learned_basis`, `reference_solver`, `shared_runtime` and
  `canonical_potential`.
- Metaball model names use a replicate label instead of a date-like RNG seed.
  The actual numerical seeds are unchanged in the configurations. The
  `scalar-gauge` suffix distinguishes the separate seed-zero finite-gauge
  setting from the matched setting.
- Saved FP64 obstacle models retain FP64 parameters and arithmetic when loaded.
  Their initialization and training interfaces accept the recorded policy.
- NKB and finite-gauge protocol files expand their nested geometry/network
  settings explicitly. Shared runtime imports use packaged sources, and the
  scalar finite-gauge path retains its original envelope and execution policy.
- File hashes and source-signature gates are removed. Model shapes, complete
  optimizer state, finite values and the computations' numerical residual
  checks remain.

These are state, source-closure and functional execution checks. Full training,
full-size canonical solves, DNS, projection campaigns and trajectories were
not repeated. Their original results are retained, and their regeneration
commands are in the [reproduction guide](REPRODUCIBILITY.md) and linked case
guides. The saved K1024 Potential endpoint remains a 25-update timing run.
