# BasisNet case: 3D_parametric_wake

This case learns a continuous `K=256` spatial basis for a three-dimensional
driven wake around a centered analytic obstacle. The obstacle changes
continuously from a sphere (`alpha=0`) to a cube (`alpha=1`) inside the
triply periodic cube `[-pi, pi]^3`.

The reference flow is an incompressible `Re=500` Navier--Stokes trajectory
driven by the constant body force `(1, 0, 0)`. It produces a persistent
streamwise wake analogous to a three-dimensional Karman-type shedding case.
The domain is periodic rather than an inlet/outlet channel: flow leaving one
side re-enters from the opposite side.

Basis training does not read Navier--Stokes snapshots, reference fields, or
supervised labels. Independent reference trajectories are generated only
after the basis is frozen and are used for projection, error evaluation, and
rendering.

## Final protocol

- Parameter range: `alpha in [0, 1]`
- Exact analytic geometry:

  ```text
  sdf(x, alpha) =
      (1 - alpha) sdf_sphere(x) + alpha sdf_box(x)
  ```

- `alpha=0` is exactly a sphere and `alpha=1` is exactly a cube
- No surface mesh, volume mesh, voxel SDF, or finite alpha table
- Complete dynamic Shape Space Spectra algorithm
- `K=256` jointly trained modes
- One continuous alpha and 4096 continuous Sobol fluid points per step
- Dynamic sorting by pre-projection normalized Dirichlet energy
- Detached gradient-causal filtering and least-squares residual peeling
- Energy-only Rayleigh loss, scaled by 80
- One sinusoidal MLP per mode, with four width-64 hidden layers
- Six Fourier-frequency bands active from the first epoch
- Antisymmetric tensor potential with a double-zero obstacle envelope
- Commuting centered differences with step `h=0.02`
- Hard zero discrete divergence under the training operator
- `ReduceLROnPlateau`: learning rate halved after 10 unimproved validations
- Minimum learning rate `2.5e-5`
- Early stopping after 24 validations without an improvement above `1e-4`
- Maximum 10000 epochs

With independent antisymmetric tensor entries `A12=a`, `A13=b`, and `A23=c`,
the velocity is

```text
u = Dy(a) + Dz(b)
v = -Dx(a) + Dz(c)
w = -Dx(b) - Dy(c)
```

The same centered-difference operator is used for every derivative, so the
discrete divergence cancels algebraically.

## Case layout

```text
3D_parametric_wake/
|-- configs/
|   `-- k256_final.json
|-- scripts/
|   |-- build_ns_cache.py
|   |-- plot_training.py
|   |-- predict_video.py
|   `-- train.py
|-- src/
|   `-- eigenfluid_sss/basisnet/parametric_wake3d/
|       |-- ns_reference.py
|       |-- postprocess.py
|       `-- training.py
|-- tests/
|   `-- test_smoke.py
`-- README.md
```

All commands below are run from the repository root.

## Environment and tests

Recommended environment:

- Windows 10/11 or Linux
- Python 3.11 or newer
- PyTorch 2.5 or newer
- CUDA GPU with at least 16 GB memory for the full `K=256` run
- `ffmpeg` on `PATH` for MP4 generation

Install and validate:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q BasisNet/cases/3D_parametric_wake/tests
python BasisNet/cases/3D_parametric_wake/scripts/train.py --self-test
python -m ruff check BasisNet/cases/3D_parametric_wake
```

The smoke tests verify:

- exact sphere and cube SDF endpoints;
- deterministic continuous-alpha sampling;
- fluid-point rejection inside the analytic obstacle;
- full raw rank for a reduced test model;
- exactly zero hard discrete divergence.

## Training

The complete configuration is
`BasisNet/cases/3D_parametric_wake/configs/k256_final.json`.

```powershell
$case = "BasisNet\cases\3D_parametric_wake"
$run = "$case\results\k256_final\run_001"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_final.json" `
  --run-dir $run
```

Command-line values override JSON fields. A minimal workflow check can be run
without editing the final configuration:

```powershell
$case = "BasisNet\cases\3D_parametric_wake"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_final.json" `
  --modes 3 --epochs 1 --hidden 8 --hidden-layers 2 --frequencies 2 `
  --points-per-step 32 --validation-points 32 --validation-alphas 2 `
  --validate-every 1 --checkpoint-every 1 --probe-every 1 `
  --run-dir "$case\results\smoke"
```

### Per-epoch learning logic

1. A deterministic scrambled Sobol sequence supplies one continuous alpha.
   Periodic endpoint insertion guarantees validation of the exact sphere and
   cube without replacing continuous training.
2. The sampler produces 4096 continuous fluid points, split equally between
   global and obstacle-neighborhood samples.
3. Each network evaluates an antisymmetric tensor potential. The analytic
   obstacle envelope and commuting centered differences yield no-slip basis
   traces and hard discrete divergence freedom.
4. Raw modes are dynamically ordered by normalized Dirichlet energy.
5. Gradient-causal least-squares peeling removes earlier modes from the
   velocity, potential, and gradient using shared coefficients.
6. The optimizer minimizes the normalized Rayleigh energy of the peeled
   basis. GT fields never enter the loss.
7. Validation runs at seven fixed alpha values every 50 epochs.
8. The scheduler and early-stopping state are updated only from validation
   loss.

### Restarting training

```powershell
python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_final.json" `
  --run-dir $run `
  --resume "$run\checkpoints\latest.pt"
```

Restarting restores the model, optimizer, scheduler, epoch, training history,
best validation value, and early-stopping counters.

Main outputs:

```text
config.json
protocol.json
train.log
data/training_history.json
data/training_history.csv
data/sampled_alpha_by_epoch.csv
checkpoints/latest.pt
checkpoints/best_model.pt
checkpoints/sss_dynamic_causal_frozen.pt
```

## Training curves

```powershell
python "$case\scripts\plot_training.py" --run-dir $run
```

The command writes PNG, SVG, and PDF figures for the training and validation
losses, Rayleigh energy, projection diagnostic, orthogonality error, hard
discrete divergence, effective rank, and learning rate.

## Independent driven-wake reference

The reference generator advances a dealiased pseudo-spectral incompressible
solver with smooth Brinkman penalization of the analytic obstacle. It does
not load the trained model.

```powershell
$cache = "$case\data\cache\alpha_0p30_re500.npz"

python "$case\scripts\build_ns_cache.py" `
  --config "$case\configs\k256_final.json" `
  --output $cache `
  --alpha 0.30 --resolution 48 --reynolds 500 `
  --final-time 6 --snapshots 31
```

The cache stores velocity snapshots, time, the spatial axis, the exact
analytic SDF, and the reference solver's divergence diagnostic.

## One-shot basis reconstruction and video

For a fixed cache alpha, the network is evaluated once to generate all 256
spatial modes. Every reference snapshot is projected onto that frozen basis.
There is no per-alpha fitting, coefficient optimization, or interpolation
between stored bases.

```powershell
$eval = "$run\evaluation_alpha_0p30"

python "$case\scripts\predict_video.py" `
  --checkpoint "$run\checkpoints\sss_dynamic_causal_frozen.pt" `
  --source-cache $cache `
  --output-dir $eval `
  --basis-chunk 512 --fps 8
```

The three-row video shows:

1. the independent reference central vorticity slice;
2. the learned-basis reconstruction;
3. the absolute vorticity error.

`summary.json` records the velocity relative `L2` error for every snapshot,
reference divergence, checkpoint epoch, and one-shot basis inference time.

Repeat the command with `best_model.pt` to compare the validation-selected
checkpoint with the final frozen basis.

## Reproducibility

- All training random sources use `seed=20260722`.
- Alpha and spatial sampling use deterministic scrambled Sobol sequences.
- Each checkpoint stores the complete configuration and optimizer,
  scheduler, history, and early-stopping states.
- `best_model.pt` is selected by validation Rayleigh loss, not by GT
  reconstruction error.
- The reference solver is independent of the learned basis.
- Different CUDA, PyTorch, and FFT versions can cause small floating-point
  differences.

## Data, checkpoints, and generated media

Reference caches, checkpoints, training logs, PID files, temporary frames,
plots, and videos are intentionally excluded from Git. Generate them with the
commands above. Large trained weights should be distributed through a release,
object storage, or Git LFS together with the exact configuration and training
history.
