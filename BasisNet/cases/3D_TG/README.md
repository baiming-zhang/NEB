# BasisNet case: 3D_TG

This case learns a 256-dimensional divergence-free spatial basis on the
triply periodic cube `[-pi, pi]^3`, then projects an independent forced
Taylor--Green Navier--Stokes truth onto the frozen learned basis.

Basis training never reads GT snapshots or supervised labels. Training uses
continuous Sobol/Monte Carlo points only; there is no training mesh. A regular
64^3 array is used solely to export and render the independent evaluation.

## Final protocol

- Complete dynamic Shape Space Spectra algorithm
- `K=256` jointly trained modes
- 4096 continuous points per training step and validation
- Dynamic sorting by pre-projection normalized Dirichlet energy
- Detached gradient-causal filtering and least-squares residual peeling
- Energy-only Rayleigh loss, scaled by 80
- One sinusoidal MLP per mode, each with 4 hidden layers of width 64
- Six Fourier-frequency bands active from the first epoch
- Antisymmetric tensor output with a commuting centered-difference divergence
- Exactly zero discrete divergence under the same `h=0.02` operator
- `ReduceLROnPlateau`: factor 0.5 after 10 unimproved validations
- Minimum learning rate `2.5e-5`
- Early stopping after 24 validations without improvement above `1e-4`
- Maximum 10000 epochs

With independent tensor entries `A12=a`, `A13=b`, and `A23=c`, velocity is

```text
u = Dy(a) + Dz(b)
v = -Dx(a) + Dz(c)
w = -Dx(b) - Dy(c)
```

Because every `D` is the same centered-difference operator, the discrete
divergence cancels algebraically.

## Layout

```text
3D_TG/
|-- configs/k256_dynamic_sss.json
|-- scripts/
|   |-- train.py
|   |-- generate_gt.py
|   |-- reconstruct.py
|   |-- evaluate.py
|   |-- render.py
|   `-- plot_training.py
|-- src/eigenfluid_sss/basisnet/tg3d/
|   |-- training.py
|   `-- postprocess.py
|-- tests/test_smoke.py
`-- README.md
```

All commands below are run from the repository root.

Install the project development dependencies and the marching-cubes renderer:

```powershell
python -m pip install -e ".[dev]"
python -m pip install scikit-image
```

## Tests

```powershell
python -m pytest -q BasisNet/cases/3D_TG/tests
python BasisNet/cases/3D_TG/scripts/train.py --self-test
python -m ruff check BasisNet/cases/3D_TG
```

## Training

```powershell
$case = "BasisNet\cases\3D_TG"
$run = "$case\results\k256_dynamic_sss\run_001"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_dynamic_sss.json" `
  --run-dir $run
```

Resume all networks, optimizer states, scheduler state, and early-stopping
state from `latest.pt`:

```powershell
python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_dynamic_sss.json" `
  --run-dir $run `
  --resume "$run\checkpoints\latest.pt"
```

Use `checkpoints/sss_dynamic_causal_frozen.pt` for the final epoch and
`checkpoints/best_model.pt` for the validation-selected checkpoint.

## Independent Taylor--Green truth

The GT is an exact forced, time-periodic incompressible Navier--Stokes
solution at `Re=500`:

```text
u(x,y,z,t) = a(t) (sin(x)cos(y), -cos(x)sin(y), 0)
a(t) = cos(2*pi*t/T),  T=3
nu = 1/Re
f = (a'(t) + 2*nu*a(t)) (sin(x)cos(y), -cos(x)sin(y), 0)
```

It is independent of the learned basis and periodic in space and time.

```powershell
$case = "BasisNet\cases\3D_TG"

python "$case\scripts\generate_gt.py" `
  --output "$case\data\tg_re500.h5" `
  --resolution 64 --reynolds 500 --frames 30 `
  --frame-seconds 0.1 --period 3
```

## Raw learned-basis reconstruction

The formal command below uses the network basis directly. Do not add
`--spectral-project-basis` for the formal result, because that option is only
an explicit post-processing control.

```powershell
$case = "BasisNet\cases\3D_TG"
$run = "$case\results\k256_dynamic_sss\run_001"
$eval = "$run\evaluation_latest_raw"

python "$case\scripts\reconstruct.py" `
  --checkpoint "$run\checkpoints\sss_dynamic_causal_frozen.pt" `
  --source "$case\data\tg_re500.h5" `
  --output "$eval\periodic_tg_k256.h5" `
  --projection-stride 1 --basis-batch 512 --x-block 4

python "$case\scripts\evaluate.py" `
  --checkpoint "$run\checkpoints\sss_dynamic_causal_frozen.pt" `
  --source "$case\data\tg_re500.h5" `
  --reconstruction "$eval\periodic_tg_k256.h5" `
  --output "$eval\final_metrics.json" `
  --points 4096 --batch 256

python "$case\scripts\render.py" `
  --source "$case\data\tg_re500.h5" `
  --reconstruction "$eval\periodic_tg_k256.h5" `
  --output-dir "$eval\visualization" `
  --fps 10 --movie-name "tg-k256-gt-vs-reconstruction.mp4"
```

Repeat the same commands with `best_model.pt` for the validation-best
comparison. The final/latest and validation-best checkpoints are deliberately
reported separately: validation uses Rayleigh energy, not GT reconstruction
error.

## Reference result

The completed 10000-epoch run produced the following raw-network metrics:

| Metric | Latest, epoch 10000 | Best, epoch 9800 |
|---|---:|---:|
| Mean velocity relative error | 1.09998% | 1.21760% |
| Hard discrete divergence RMS / max | 0 / 0 | 0 / 0 |
| Peeled orthogonality max error | 1.78e-15 | 1.55e-15 |
| Energy-spectrum relative L2 error | 0.02478% | 0.03042% |
| 4096-point one-shot basis inference | 0.499 s | 0.501 s |

The latest checkpoint is the formal result because it reconstructs the
independent GT more accurately than the validation-best checkpoint.

## Data and generated artifacts

GT HDF5 files, checkpoints, training logs, PID files, temporary frames, and
videos are intentionally excluded from Git. Generate them with the commands
above. Large weights should be distributed through a release, object storage,
or Git LFS together with the exact config and training history.
