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

  - Current training: 1,024 shared modes, 4,096 Sobol points/epoch,
    FP16 hidden-layer shadows with FP32 boundary/master/Adam arithmetic.
  - Hybrid objective:
    `0.5 (velocity Rayleigh / 80) + 0.5 (vorticity Rayleigh / 232)`;
    selected epoch 650 with validation `0.5280304703`, rank `1024/1024`,
    orthogonality error `2.61e-12`, and 3:53:51 accumulated active time.
  - Three Re=1500 velocity-L2 reconstructions at
    `(beta, alpha)=(0.50,+10 deg),(0.85,0 deg),(0.50,-10 deg)` have mean
    total/wake/fluctuation/wake-fluctuation velocity errors
    `1.945275% / 3.511677% / 27.684598% / 20.135484%`.
  - A matched rectangular Dirichlet Laplacian K1024 basis gives
    `2.981040% / 5.478333% / 42.469079% / 31.854173%` on the same fields;
    its errors are 53--58% higher depending on the metric.
  - A from-scratch K1024 velocity-only ablation uses identical epoch-0
    weights and validation points. The versioned preliminary snapshot is
    epoch 200 with validation `1.6993393228`, rank `1024/1024`; the final
    1,000-epoch comparison is pending.
  - The earlier K256 result remains documented as a legacy baseline inside
    the case README; it is no longer the primary result.

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
