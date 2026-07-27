# BasisNet case: 2D_karman

This case learns a continuous K=96 spatial basis for a translating cylinder
in a two-dimensional channel and reconstructs an independent Re=100 Karman
wake after training.

The basis optimizer does not read Navier-Stokes snapshots, numerical
eigenfunctions, or supervised labels. The independent NS data are used only
after the model is frozen for projection, error evaluation, and rendering.

The jointly parameterized airfoil counterpart follows the same case-directory,
training-entry-point, fixed-peeling, and Ritz-postprocessing conventions; see
[`2D_airfoil`](../2D_airfoil/README.md).

## Final protocol

- Parameter range: `alpha in [0, 1]`
- Cylinder center: `x_c(alpha) = 1 + 8 alpha`, `y_c = 0`
- Channel: `[0, 10] x [-2, 2]`
- Cylinder radius: `0.45`
- Number of modes: `K = 96`
- One independent sinusoidal MLP and one Adam optimizer per mode
- Fixed mode identity and order; modes are never sorted or swapped
- Earlier modes are detached during later-mode optimization
- The velocity, streamfunction, and velocity gradient are peeled with the
  same least-squares coefficients
- Localized double-zero boundary envelope
- Mixed Sobol sampling: 50% global, 25% cylinder neighborhood, and 25% wake
- All Fourier features are active from epoch 1
- No frequency curriculum
- No projection penalty
- Constant learning rate: `8e-4`
- No learning-rate scheduler
- Training loss:

  ```text
  loss = mean_k[(||grad u_k||^2 / ||u_k||^2) / 80]
  ```

The raw projection ratio is recorded only as an overlap diagnostic. It is
not part of the loss and does not contribute gradients. The peeled output
basis is orthogonal under the discrete velocity inner product.

## Case layout

```text
2D_karman/
|-- configs/
|   `-- k96_final.json
|-- scripts/
|   |-- build_ns_cache.py
|   |-- plot_training.py
|   |-- predict_video.py
|   `-- train.py
|-- src/
|   `-- eigenfluid_sss/basisnet/karman2d/
|       |-- ns_reference.py
|       |-- postprocess.py
|       `-- training.py
|-- tests/
|   `-- test_smoke.py
`-- README.md
```

All commands below are run from the repository root.

## Environment and installation

Recommended environment:

- Windows 10/11 or Linux
- Python 3.11-3.13
- PyTorch 2.5 or newer
- CUDA GPU with at least 16 GB of memory for the full K=96 run
- `ffmpeg` available on `PATH` for MP4 generation

Install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest -q
python BasisNet/cases/2D_karman/scripts/train.py --self-test
```

The self-test uses three small modes to verify:

- later-mode losses do not leak gradients into earlier modes;
- each independent optimizer updates its own mode;
- the peeled basis has finite orthogonality error;
- one-shot parameter inference returns the expected shape.

## Training

The complete version-controlled configuration is
`BasisNet/cases/2D_karman/configs/k96_final.json`.

```powershell
$case = "BasisNet\cases\2D_karman"
$run = "$case\results\k96_final\run_001"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k96_final.json" `
  --run-dir $run
```

Command-line values override fields loaded from the JSON file. A short
workflow check can therefore be run without editing the final configuration:

```powershell
$case = "BasisNet\cases\2D_karman"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k96_final.json" `
  --modes 3 `
  --epochs 1 `
  --hidden 8 `
  --hidden-layers 2 `
  --frequencies 2 `
  --points-per-step 16 `
  --validation-points 16 `
  --validation-alphas 2 `
  --validate-every 1 `
  --checkpoint-every 1 `
  --probe-every 1 `
  --run-dir "$case\results\smoke"
```

### Per-epoch learning logic

1. A deterministic Sobol sequence provides one continuous `alpha`. Every
   twentieth epoch inserts an `alpha=0` or `alpha=1` endpoint.
2. The sampler creates 512 mixed points in the global domain, cylinder
   neighborhood, and wake region.
3. Modes are processed in fixed order from 0 through 95.
4. For mode `k`, automatic differentiation evaluates its streamfunction,
   divergence-free velocity, and velocity gradient.
5. The raw velocity is projected onto peeled modes `0` through `k-1`.
6. The same least-squares coefficients remove the earlier components from
   the streamfunction and velocity gradient.
7. The residual velocity is normalized.
8. Only mode `k` is updated by minimizing its normalized Rayleigh energy
   divided by 80. All earlier modes are detached.
9. The epoch records loss, mean and maximum energy, raw projection ratio,
   peeling orthogonality error, and effective raw rank.
10. Validation runs every 50 epochs at seven fixed parameter values. A
    deterministic probe runs every 100 epochs at `alpha=0.15`.
11. `checkpoints/best_model.pt` is updated when validation loss improves.

### Restarting training

```powershell
$case = "BasisNet\cases\2D_karman"
$run = "$case\results\k96_final\run_001"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k96_final.json" `
  --run-dir $run `
  --resume "$run\checkpoints\latest.pt"
```

Restarting restores the model, all 96 Adam optimizer states, the epoch,
training history, and best validation value.

### Training outputs

```text
config.json
protocol.json
train.log
data/training_history.json
data/training_history.csv
data/alpha_0p15_milestones.json
data/alpha_0p15_milestones.csv
checkpoints/latest.pt
checkpoints/best_model.pt
checkpoints/epoch_*.pt
checkpoints/sss_official_peeling_frozen.pt
```

A full run is complete only when:

1. `train.log` contains `training complete`; and
2. `checkpoints/sss_official_peeling_frozen.pt` exists.

## Training curves

```powershell
$case = "BasisNet\cases\2D_karman"
$run = "$case\results\k96_final\run_001"

python "$case\scripts\plot_training.py" `
  --run-dir $run
```

The script produces PNG, SVG, and PDF plots containing:

- train, validation, and `alpha=0.15` probe losses;
- learning rate;
- mean Rayleigh energy;
- raw projection ratio;
- peeling orthogonality error;
- effective raw basis rank.

## Independent NS reference cache

The reference data are Re=100 cylinder-flow HDF5 data from DOI
`10.5281/zenodo.18600528`. The cache builder expects:

```text
grid/x
grid/y
grid/t
fields/field_0
fields/field_1
fields/field_2
```

where `field_1` and `field_2` contain the velocity components.

```powershell
$case = "BasisNet\cases\2D_karman"

python "$case\scripts\build_ns_cache.py" `
  --input data\raw\re100_cylinder.h5 `
  --output "$case\data\cache\actual_canonical_cache.npz"
```

The resulting cache contains cylinder-relative coordinates, 17 periodic
phases of `u`, `v`, streamfunction, and vorticity, plus the temporal mean
lifting field. Basis training never opens this cache.

## Direct 101-parameter prediction and video

The released prediction path evaluates:

```python
alphas = np.linspace(0.0, 1.0, 101)
```

The parameter step is therefore 0.01. Every parameter value independently
performs:

1. one-shot neural-field evaluation;
2. first- and second-order derivatives on a 512-point symmetric quadrature;
3. construction of the raw 96-dimensional mass and stiffness matrices;
4. solution of the generalized symmetric eigenproblem to obtain a K=96 Ritz
   basis;
5. mean-flow lifting of the independent NS phase snapshot;
6. velocity-weighted least-squares projection onto the 96 modes;
7. reconstruction of velocity, streamfunction, and vorticity;
8. calculation of velocity and vorticity relative errors.

No low-resolution parameter anchors or interpolated fields are used.

```powershell
$case = "BasisNet\cases\2D_karman"
$run = "$case\results\k96_final\run_001"

python "$case\scripts\predict_video.py" `
  --checkpoint "$run\checkpoints\best_model.pt" `
  --source-cache "$case\data\cache\actual_canonical_cache.npz" `
  --output-dir "$run\prediction_alpha_0_to_1" `
  --alpha-samples 101 `
  --phase-cycles 2 `
  --fps 10
```

The video retains a single-column, three-row layout:

1. independent Re=100 NS reference;
2. K=96 PINN reconstruction;
3. absolute velocity error.

The 101 parameter values advance through two flow cycles. At 10 frames per
second, the resulting video lasts 10.1 seconds.

Prediction is restartable. Each completed Ritz basis and reconstructed sample
is stored under `basis/` and `samples/`; rerunning the same command reuses
completed files.

Main outputs:

```text
k96_ns_vs_pinn_continuous_alpha_flow.mp4
summary.json
alpha_metrics.csv
frame_metrics.csv
basis/
samples/
frames/
```

## Reproducibility

- All random sources use `seed=20260715`.
- Training and Ritz quadrature use deterministic scrambled Sobol sequences
  with fixed seeds.
- Each checkpoint stores the complete configuration, all optimizer states,
  and training history.
- `best_model.pt` is selected by validation loss, not NS reconstruction
  error.
- The training protocol uses a constant learning rate, no scheduler, no
  projection penalty, and no frequency curriculum.
- Different CUDA and PyTorch versions may produce small floating-point
  differences. Rank, curve trends, and reconstruction metrics should remain
  stable.
- Full K=96 training and 101-parameter second-derivative inference are
  computationally expensive. Verify the environment with the smoke tests
  before starting a full run.

## Data, checkpoints, and generated media

Third-party NS HDF5 data, checkpoints, caches, frames, and videos are not
stored in Git. Obtain the reference data through the DOI above or provide
data with the same schema. Large trained weights should be distributed
through a release, private object storage, or Git LFS together with
`config.json` and `training_history.json`.
