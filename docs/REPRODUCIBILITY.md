# Reproducing Neural Eigenfluid Bases

Run commands from the repository root. `requirements-repro.txt` pins the Python dependencies. Install `.[test,fem]` for the CPU checks and measurement plots, and `.[cuda,torch,fem,test]` for native neural calculations. CuPy uses CUDA 12. Set `NEB_MKL_RUNTIME` to the directory containing your installed `mkl_rt.dll` or `libmkl_rt.so` for the canonical direct solver; alternatively, set `NEB_CUDSS_LIBRARY` to your installed cuDSS 0.8 library.

## Checkpoints and resources

`neb checkpoints` lists models; `neb checkpoints --download --only MODEL_ID` downloads a model and its associated multiplier state. The model index supplies configuration paths, capacities, and training endpoints. Adjacent `model.json` files retain model metadata. The configuration's `seed` is the actual generator seed; `seed_index` identifies the reported replicate.

For total K=1024, a one-hole completed basis combines 1023 positive coordinates and one harmonic coordinate. The other topologies use 1022+2 and 1021+3. Select the corresponding checkpoints and raw/scaled lift configurations.

Models retain numerical parameters, Adam moments, and optimizer counters. Potential-AL additionally supplies multipliers, rho, and the preceding constraint residual. Use one GPU experiment at a time. Metaball runners log memory use and preserve host/GPU memory reserves. Full K=1024 physical basis storage requires about 39.3 GB, with additional space for references and operators. CPU contractions stream the complete arrays in bounded blocks.

## Periodic vortex merger

Generate DNS with `experiments/periodic/run.py dns` and the K=1024 Fourier control with `run.py laplacian`. Evaluate Potential, Velocity, and NKB on the common DNS with `run.py evaluate`; repeat seeds 0, 1, 2. Fresh training uses:

```bash
python experiments/periodic/run.py prepare \
  --config experiments/periodic/configs/periodic-potential-k1024-seed0.json \
  --output runs/periodic/train
python experiments/periodic/run.py train \
  --checkpoint runs/periodic/train/initial_epoch0000.npz \
  --output runs/periodic/train
```

The protocol is d=1.35, T=18π, viscosity 0.0010742913669168762, K=1024, 256² quadrature, 128² observations, and 241 saved states. The evaluator runs 0.005 and 0.0025 time steps. [Full periodic recipe](../experiments/periodic/README.md).

## Multiply connected domains

Follow the [projection/training recipe](../experiments/multihole/README.md) for `one_circle`, `circle_square`, and `circle_square_triangle`. Generate the 20 quintic fields, mesh, quadrature, and numerical operators, then evaluate learned positive and harmonic models at fixed total dimension.

For the compact-vortex ensemble, run `experiments.multihole.reference.run --mode vortex-trajectory` for samples 0–19. The [reference guide](../experiments/multihole/reference/USAGE.md) gives initial-field generation, numerical spectral controls, trajectories, and refinements.

The [learned ROM recipe](../experiments/multihole/learned_rom/README.md) assembles full-quadrature mass, stiffness, and nonlinear operators, then advances autonomous coefficients. Its `--field-ensemble` flag selects quintic or compact-vortex inputs. Repeat raw/scaled harmonic models for the circulation controls. The [NKB recipe](../experiments/2d/nkb/README.md) supplies direct-velocity comparisons.

## Three-dimensional Metaball

Follow the [Metaball guide](../experiments/metaball/USAGE.md) in order:

1. Generate geometry, source samples, the fitted physical complex, source-curl map, and physical volume.
2. Generate the common 128³ numerical reference and the 20 independent large-eddy targets.
3. Download the selected model or train from its exact configuration.
4. Evaluate every mode on the complete ordered fluid grid, producing FP64 `(1599754,3,K)` fields and the full mass matrix.
5. Run `experiments/metaball/run.py paired20` with all 20 target paths for two-pass full-mass projection.
6. Derive and assemble Galerkin operators with `operators.py`, then run `run.py rom` for autonomous coefficient dynamics.

Use the paper method name with the matching configuration. Potential requires the full physical complex and source map. Matched Potential-FG uses the Velocity architecture, initialization, optimizer, sampler, and validation settings. `--finite-gauge-setting scalar-gauge` selects the supplied seed-0 finite-space setting for transfer/control comparisons.

`configs/large_eddy20.json` supplies the 20 independent Philox keys and the 16 Fourier wavevectors. `configs/initial_condition.json` specifies the original random-field conditioning, RMS normalization, and four warm-up substeps. `configs/dns.json` specifies the 30-frame trajectory reference. All are under `experiments/metaball/`.

The [gauge guide](../experiments/gauge/USAGE.md) covers finite, spline, FMM, and Fourier controls. `heldout_samples.py` regenerates each transfer domain and its disjoint 16,384-point fitting/assessment sets. The [fourth-domain guide](../experiments/metaball/GEOMETRY4.md) builds the quadratic physical chart and its 256-mode reference.

For the K1024 Potential preparation-timing experiment, select its timing configuration and add `--timing-only` to the Potential training command. This runs 25 optimizer updates with the original 2000-update learning-rate horizon, saves the full endpoint, and performs zero scheduled validations. Its corresponding saved checkpoint is `metaball-potential-k1024-timing25`.

## Data, figures, and manuscript

```bash
neb figures --output outputs/paper_figures
python tools/redraw_combined_metaball.py --repo . --output outputs/metaball_paper
```

Both commands export PNG/vector PDF. The main Metaball figure uses 400 measurements through K=512; its K=1024 appendix uses 60. Sample deviations use `ddof=1`. Preparation-cost estimates, fit coefficients, and assumptions are supplied in `data/paper_results/preparation_cost_estimates.json`; measured full-basis times are in `primary_basis_runtime.csv`.

All paper artwork is included. [Visualization guidance](../experiments/VISUALIZATIONS.md) describes field plates and mode atlases. Build from `paper/` with `pdflatex main.tex` three times; `main.bbl` is supplied. Method and airfoil schematic panels are included as artwork.

`python -m pytest tests -q` checks analytic derivatives, full-mass projection, finite-element controls, Galerkin contractions, published statistics, checkpoint coverage, and CPU-only command help. Native full-size experiment commands use the CUDA dependencies and the case-specific reference/time-step checks.
