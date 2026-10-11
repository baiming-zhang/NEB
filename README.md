# NEB: Neural Eigenfluid Bases

**Continuous spectral coordinates for incompressible flow.**

Code for **Neural Eigenfluid Bases: Spectral Objectives, Gauge, and Topology** by Baiming Zhang, Shiying Xiong, and Zhecheng Wang.

Neural eigenfluids learn spatial coordinates that satisfy incompressibility through analytic differential lifts. Geometry-local spectral orthogonalization turns these continuous fields into reusable bases for full-mass projection and autonomous Galerkin dynamics. Learning in potential space provides a lower-derivative spectral formulation, while harmonic coordinates encode the independent circulations of multiply connected domains.

The experiments connect three aspects of reduced fluid simulation: **spectral learning**, **topological completion**, and **reusable coefficient dynamics**. At K=1024, the periodic vortex-merger experiment achieves sub-percent mean velocity error with the streamfunction formulation. The multiply connected experiments examine positive-mode and circulation allocation on the same physical initial fields. The 3D experiments compare representation capacity and autonomous evolution, then reuse assembled operators for rapid coefficient updates.

![Paired initial-field representation in the 3D Metaball domain](assets/figures/metaball20.png)

## Methods

| Name | Formulation |
|---|---|
| **SF-R** | Streamfunction Rayleigh learning in 2D |
| **VRQ** | Velocity Rayleigh Quotient learning |
| **CCQ** | Canonical Curl Quotient learning in 3D |
| **LEB** | Laplacian Eigenbasis at the same basis capacity |
| **NKB** | Neural kinematic basis controls |
| **GCP-AL / GCP-FG** | Gauge-constrained potential controls with the corresponding augmented-Lagrangian or finite-gauge objective |

Spatial derivatives are evaluated analytically. Projection uses the full mass matrix, and the reduced dynamics evolve coefficients with the assembled Galerkin operators. Experiment configurations specify geometry, sampling, initialization, optimizer, precision, capacity, and validation cadence.

## Installation

Use Python 3.11 or newer in a virtual environment. From the repository root:

```bash
python -m pip install -e ".[test,fem]" -c requirements-repro.txt
python -m pytest tests -q
neb figures --output outputs/paper_figures
```

The first workflow runs on CPU and redraws the supplied measurements as PNG and vector PDF. For native training and reference assembly, install the experiment dependencies:

```bash
python -m pip install -e ".[cuda,torch,fem]" -c requirements-repro.txt
```

The native spectral trainers use CuPy with CUDA 12. The 3D canonical solvers use an explicit oneMKL LP64 runtime or cuDSS 0.8 library; the [Metaball guide](experiments/metaball/USAGE.md) specifies the solver selection and installation paths. Each experiment directory documents its inputs and commands.

## Reference checkpoints

The [reference-checkpoint release](https://github.com/baiming-zhang/NEB/releases/tag/paper-v1.0) contains selected trained models. The source repository includes the corresponding configurations and a SHA256 manifest.

With the GitHub CLI installed and authenticated:

```bash
neb checkpoints
neb checkpoints --download
```

Select individual models with `--only CHECKPOINT_ID`. Files are verified after download and placed under `checkpoints/2d/` or `checkpoints/3d/`. These models provide a direct starting point for basis evaluation, full-mass projection, and coefficient evolution; the training commands reproduce their configurations from fresh initialization.

| Experiment | Reference models | Capacity |
|---|---|---:|
| Periodic vortex merger | SF-R, VRQ, NKB | 1024 |
| Two- and three-hole domains | VRQ | 32 |
| Harmonic circulation | Raw and scaled lifts | 1 |
| 3D Metaball | CCQ, VRQ, GCP-AL, NKB | 32 |

All eleven reference models are saved at epoch 2000. The manifest lists their individual download IDs, sizes and hashes; adjacent JSON files retain the model configurations.

## Periodic vortex merger

The periodic case compares learned and Fourier bases at K=1024 over 18π time units. It uses a 256×256 quadrature grid, a 128×128 observation grid, and 241 saved states. Three learned initialization seeds provide the paper's mean and sample standard deviation.

Generate the reference flow and evaluate the supplied SF-R checkpoint:

```bash
python experiments/periodic/run.py dns --output runs/periodic/dns
python experiments/periodic/run.py evaluate --method Potential --seed 0 \
  --checkpoint checkpoints/2d/K1024_potential_rayleigh_seed0/hardcoded_epoch2000.npz \
  --gt runs/periodic/dns --output runs/periodic/sfr_evaluation
```

Train from the paper configuration:

```bash
python experiments/periodic/run.py prepare \
  --config experiments/periodic/configs/K1024_potential_rayleigh_seed0.json \
  --output runs/periodic/sfr
python experiments/periodic/run.py train \
  --checkpoint runs/periodic/sfr/initial_epoch0000.npz \
  --output runs/periodic/sfr
```

Use `K1024_velocity_rayleigh_seed0.json` and `--method Velocity` for VRQ. The method names in these commands match the original mathematical implementation; the paper labels are SF-R and VRQ.

## Multiply connected domains

The one-, two-, and three-hole cases study positive-spectrum coordinates and harmonic circulation completion. Twenty fixed quintic streamfunctions per topology support paired projection comparisons, with impermeability and boundary-vorticity conditions built into the field construction. The compact Wendland vortex ensemble provides the trajectory protocol for harmonic allocation and raw/scaled lift controls. Both ensembles have dedicated generators and preserve the same physical initial field across their basis comparisons.

```bash
python experiments/multihole/run.py fields --case circle_square \
  --output runs/multihole/circle_square/fields
python experiments/multihole/run.py prepare \
  --config experiments/multihole/configs/circle_square_velocity_k32_seed1.json \
  --output runs/multihole/circle_square/vrq
python experiments/multihole/run.py train \
  --checkpoint runs/multihole/circle_square/vrq/initial_epoch0000.npz \
  --output runs/multihole/circle_square/vrq
python experiments/multihole/run.py harmonic-train \
  --config experiments/multihole/configs/circle_square_harmonic_seed1.json \
  --output runs/multihole/circle_square/harmonic
python -m experiments.multihole.reference.run --mode mesh --case circle_square \
  --output runs/multihole/circle_square/mesh
python -m experiments.multihole.reference.run --mode operators --case circle_square \
  --mesh runs/multihole/circle_square/mesh/mesh.npz \
  --output runs/multihole/circle_square/operators
python experiments/multihole/evaluate.py \
  --checkpoint checkpoints/2d/circle_square_velocity_k32_seed1/epoch2000.npz \
  --fields runs/multihole/circle_square/fields \
  --quadrature runs/multihole/circle_square/operators/quadrature.npz \
  --output runs/multihole/circle_square/vrq_projection
```

The configuration collection includes positive and harmonic training, raw and scaled lift controls, and fixed-capacity allocations. The [multihole guide](experiments/multihole/README.md) gives the P/PH projection commands. The [learned dynamics guide](experiments/multihole/learned_rom/README.md) constructs the full-mass Galerkin operators, advances autonomous trajectories and compares them with the physical reference. The [reference guide](experiments/multihole/reference/USAGE.md) provides adaptive meshes, P3 reference trajectories, compact-vortex ensembles and full-space LEB controls. The [NKB guide](experiments/2d/nkb/README.md) covers geometry-only training and its coefficient dynamics.

## 3D Metaball

The 3D experiments use the same physical target and same-capacity LEB controls for full-domain projection. They include the VRQ and CCQ objectives, gauge-constrained potential controls, NKB, gauge-solver ablations, held-out geometry, and coefficient-space dynamics. Capacities and method-specific configurations are supplied with the experiment sources.

Generate the analytic source samples and evaluate the reference VRQ model on the full physical grid:

```bash
python experiments/metaball/prepare.py source --output runs/metaball/source
python experiments/metaball/prepare.py dns --output runs/metaball/dns
python experiments/metaball/prepare.py random20 \
  --dns runs/metaball/dns --output runs/metaball/targets
python experiments/metaball/evaluate.py --method VRQ --K 32 \
  --config experiments/metaball/configs/vrq_ccq_k32.json \
  --checkpoint checkpoints/3d/vrq_k32_e2000.npz \
  --samples runs/metaball/source/samples.npz \
  --flat runs/metaball/targets/fluid_flat_indices.npy \
  --coordinates runs/metaball/dns/coordinates.npy \
  --output runs/metaball/vrq_basis
```

The [Metaball guide](experiments/metaball/USAGE.md) continues with CCQ physical operators, full-mass paired projection, VRQ capacity sweeps and assembled coefficient dynamics. The [gauge guide](experiments/gauge/USAGE.md) gives finite-space, spline, volume-FMM and Fourier controls. The [quadratic geometry guide](experiments/metaball/GEOMETRY4.md) reproduces the held-out Nédélec mesh, curved physical chart, 264-vector Ritz solve and 256-mode reconstruction.

The paired-field comparison in `data/paper/metaball20.csv` contains 20 initial fields for each of 16 method-capacity groups. Error bars show the sample standard deviation across fields, with `ddof=1`. The `metaball_capacity.json` measurements separate initial representation from autonomous evolution.

![Representation capacity and autonomous evolution](assets/figures/metaball_capacity.png)

## Repository layout

```text
experiments/periodic/     Periodic data generation, training, and full-mass rollout
experiments/multihole/    Random streamfunctions, positive and harmonic coordinates
experiments/metaball/     3D spectral learning, projection, and reduced dynamics
experiments/gauge/        Canonical gauge and solver experiments
src/neb/                 Checkpoint retrieval and paper-data plotting
checkpoints/             Reference configurations and download manifest
data/paper/              Numerical measurements used by the paper
assets/figures/           PNG and vector PDF comparisons
tests/                   CPU numerical and reproduction checks
```

Each case includes its numerical kernels and dependency closure. Case entry points take explicit configuration, checkpoint, data, and output paths. The [visualization guide](experiments/VISUALIZATIONS.md) covers the ten supplied measurement plots, field plates and basis atlases.

## Measurements and citation

Velocity errors use the stated full-domain or observation-grid relative L2 measure. Seed statistics and initial-field statistics are computed within their respective experiment protocols. Projection, training, reference assembly, and online coefficient evolution retain their separate timing boundaries. Figure data provenance is recorded in `data/paper/provenance.json`.

```bibtex
@misc{zhang_neural_eigenfluid_bases,
  title  = {Neural Eigenfluid Bases: Spectral Objectives, Gauge, and Topology},
  author = {Zhang, Baiming and Xiong, Shiying and Wang, Zhecheng},
  year   = {2026}
}
```

See `CITATION.cff` for a machine-readable citation.
