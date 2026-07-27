# EigenFluid-SSS

EigenFluid-SSS contains BasisNet for parameter-dependent spatial bases and a
TimeNet placeholder for temporal coefficients.

## Modules

- [BasisNet](BasisNet/README.md): Sequential Spectral Subspace basis training
  and linear reconstruction.
- [TimeNet](TimeNet/README.md): no implemented training or inference path.

## Recorded runs

All errors below are relative L2 errors. Percentages are computed from the
stored JSON summaries. Training time is taken from timestamped logs; when a run
was resumed more than once, the accumulated active time stored in
`training_history.json` is also listed.

- Workstation: NVIDIA GeForce RTX 4080 SUPER, `16,376 MiB` VRAM;
  Intel Core i9-14900K, `24` cores, `32` logical processors.

- [2D_karman](BasisNet/cases/2D_karman/README.md)

  - Training: 96 modes, 512 collocation points/epoch, 5,000 epochs.
  - Training time: `14:00:00` wall time
    (`2026-07-16 17:06:24` to `2026-07-17 07:06:24`).
  - Selected checkpoint: epoch 4,950; validation loss `0.3050989267`;
    train loss `0.3264800681`; validation `max |G-I| =
    2.184016856e-15`; rank `96/96`.
  - Velocity-fit reconstruction at `alpha=0.40`, 17 Re=100 GT frames:
    mean velocity error `5.808077%`; mean vorticity error `18.569974%`;
    total relative divergence `1.762272%`; modal relative divergence `0`;
    effective mass rank `96`.
  - Vorticity-fit reconstruction at `alpha=0.40`, 17 Re=100 GT frames:
    mean velocity error `9.552876%`; mean vorticity error `16.517892%`;
    total relative divergence `1.772122%`; modal relative divergence `0`;
    effective mass rank `96`.

- [2D_airfoil](BasisNet/cases/2D_airfoil/README.md)

  - Training: 256 modes, 4,096 Sobol points/epoch, epoch budget 10,000,
    early stop at epoch 6,800.
  - Training time: `08:37:53` wall span
    (`2026-07-27 01:29:52` to `2026-07-27 10:07:45`);
    accumulated active time `07:02:43` across epochs 1-6,800.
  - Early-stopping reference: epoch 5,800; validation mean `0.3375403533`;
    validation P95 `0.3457105660`; validation maximum `0.3460695776`;
    train loss `0.3456664867`; validation `max |G-I| =
    1.332267630e-15`; rank `256/256`.
  - Smallest recorded validation mean: `0.3375304713` at epoch 6,700.
    It did not replace epoch 5,800 under the configured material-improvement
    threshold.
  - Frozen epoch-6,800 checkpoint, in-domain 21-frame path
    (`beta=0.15..0.85`, `alpha=-10..10 degrees`), velocity-fit:
    mean/P95/maximum velocity error
    `3.045560% / 4.151656% / 4.232482%`; mean vorticity error
    `30.082296%`; mean fluctuation-velocity error `53.531760%`; mean
    fluctuation-vorticity error `85.777799%`; mean wake-velocity error
    `6.803666%`; mean wake-vorticity error `51.585655%`; Ritz rank
    `256/256`.
  - The same 21 frames with vorticity-fit coefficients: mean velocity error
    `4.012830%`; mean vorticity error `29.465638%`; mean
    fluctuation-vorticity error `84.187929%`; vorticity projection rank
    `256/256`.
  - Outside-training-domain test at `beta=3`, `alpha=30 degrees`, Re=100:
    velocity-fit velocity/vorticity errors
    `2.002120% / 23.584463%`; fluctuation velocity/vorticity errors
    `19.458154% / 105.584121%`; wake velocity/vorticity errors
    `2.040363% / 26.215795%`.
  - Outside-training-domain test at `beta=3`, `alpha=30 degrees`, Re=1,500:
    velocity-fit velocity/vorticity errors
    `12.020214% / 67.008829%`; fluctuation velocity/vorticity errors
    `34.149164% / 93.678719%`; wake velocity/vorticity errors
    `16.335338% / 74.660629%`.

- [3D_periodic_free](BasisNet/cases/3D_periodic_free/README.md)

  - Training: 256 modes, 4,096 collocation points/epoch, 10,000 epochs.
  - Training time: accumulated active time `54:44:12` across six recorded
    training segments.
  - Lowest validation value: epoch 9,800; validation loss
    `0.1127998991`; train loss `0.1128278521`; validation `max |G-I| =
    1.443289932e-15`; maximum discrete divergence `0`; rank `256/256`.
  - Epoch-10,000 checkpoint, oscillatory ABC, `k=1`, Re=500, 50 frames:
    projection mean velocity
    error `0.235647%`; output-grid mean/maximum velocity error
    `0.246989% / 0.267732%`; reconstruction time `60.859079 s`.
  - Epoch-10,000 checkpoint, forced Taylor--Green, Re=500: mean/maximum
    velocity error
    `1.099978% / 1.099978%`; energy-spectrum relative L2 error
    `0.024780%`; kinetic-energy relative error `0.012101%`.
  - Basis evaluation on 4,096 points: `0.499109 s`,
    `8,206.63 points/s`; raw rank `256/256`; peeled `max |G-I| =
    1.776356839e-15`; maximum and RMS discrete divergence `0 / 0`.

- [3D_parametric_wake](BasisNet/cases/3D_parametric_wake/README.md)

  - Training: 256 modes, 512 collocation points/epoch, 5,000 epochs.
  - Training time: `01:07:59` wall time
    (`2026-07-18 21:06:19` to `2026-07-18 22:14:18`).
  - Lowest validation value: epoch 4,700; validation loss
    `1.898210114`; train loss `2.110306882`; validation `max |G-I| =
    8.881784197e-16`; validation maximum discrete divergence
    `8.881784197e-16`; rank `256/256`.
  - Evaluated checkpoint: epoch 5,000, `alpha=0.1`, Re=500, 30 frames,
    17,919 projection points.
  - Projection mean velocity error `16.512132%`; output-grid
    mean/maximum velocity error `17.110033% / 17.164747%`;
    reconstruction time `49.879134 s`; basis rank `256/256`.

## Repository layout

```text
EigenFluid-SSS/
|-- BasisNet/
|   |-- README.md
|   `-- cases/
|       |-- README.md
|       |-- 2D_karman/
|       |-- 2D_airfoil/
|       |-- 3D_periodic_free/
|       `-- 3D_parametric_wake/
|-- TimeNet/
|   `-- README.md
|-- pyproject.toml
`-- README.md
```

## Installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/baiming-zhang/eigenfluid-sss.git
cd eigenfluid-sss
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Verification

```bash
python -m pytest -q
python BasisNet/cases/2D_karman/scripts/train.py --self-test
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python BasisNet/cases/3D_periodic_free/scripts/train.py --self-test
python BasisNet/cases/3D_parametric_wake/scripts/train.py --self-test
```

Git tracks source code, configurations, tests, and documentation. The ignore
rules exclude `results/`, checkpoints, GT fields, reconstructions, figures,
videos, and training logs.
