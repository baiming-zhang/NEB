# BasisNet case: 2D_airfoil

This case learns a continuous `K=256` spatial basis for left-to-right flow
past a two-dimensional analytic airfoil. Two parameters are trained jointly:

- `beta in [0, 1]` controls the thickness ratio,
  `t/c = 0.08 + 0.10 beta`;
- `alpha in [-12, 12] degrees` is the angle of attack about the fixed
  quarter-chord point.

The basis optimizer does not read Navier--Stokes snapshots, numerical
eigenfunctions, or supervised labels. Independent D2Q9 Navier--Stokes data
are used only after the model is frozen for projection, error evaluation,
and rendering.

The case follows the same repository format and fixed-order SSS protocol as
[`2D_karman`](../2D_karman/README.md). Airfoil-specific additions are the
joint `(beta, alpha)` parameterization, the analytic obstacle, the
fixed-graph manual derivative backend, and the independent GT solver.

## Final protocol

- Domain: `[0, 10] x [-4, 4]`
- Uniform free stream from left to right
- Analytic `C-infinity` elliptical airfoil with chord `c=2`
- Fixed quarter-chord point: `(3, 0)`
- Airfoil thickness ratio: `0.08..0.18`
- Angle of attack: `-12..12 degrees`
- Number of modes: `K=256`
- Four hidden sine layers of width 64 with six Fourier frequencies
- One independent MLP and one Adam state per mode
- Fixed mode identity and order; modes are never sorted or swapped
- Earlier modes are frozen during later-mode optimization
- Streamfunction, velocity, and spatial derivatives share the same peeling
  coefficients
- Divergence-free velocity is obtained analytically from the streamfunction
- Double-zero perturbation envelope at the inlet, airfoil, top, and bottom
- The reconstructed total field still contains the uniform free stream, so
  the top and bottom are far-field boundaries, not no-slip walls
- Mixed scrambled-Sobol sampling: 50% global, 25% airfoil neighborhood,
  and 25% wake
- 4096 fresh mesh-free spatial samples per epoch
- 4096 fixed validation points per joint parameter case
- MLP matrix products use FP16 shadow weights
- FP32 master weights, Adam state, reductions, peeling, and Ritz algebra
- Explicit Jet2 value/gradient/Hessian propagation; no PyTorch autograd or
  dynamic training graph
- Raw-mode forward batch size: 32
- Base learning rate: `8e-4`
- Reduce-on-plateau patience: 200 epochs
- Early-stopping patience: 1000 epochs
- Maximum training length: 10000 epochs
- Training loss:

  ```text
  loss = mean_k[(||grad u_k||^2 / ||u_k||^2) / 80]
  ```

The final run stopped at epoch 6800. Its best fixed-validation result was at
epoch 5800 with `val_mean=0.337540`; all 256 modes remained full rank.
Checkpoints, training logs, GT caches, images, and videos are not stored in
Git.

### Validated final-run summary

| Quantity | Result |
| --- | ---: |
| Mode count | 256 |
| Maximum epoch budget | 10,000 |
| Early-stop epoch | 6,800 |
| Best validation epoch | 5,800 |
| Best fixed-validation mean | 0.337540 |
| Final Ritz rank | 256 |
| Frozen checkpoint basename | `hardcoded_sss_frozen.npz` |

The validation mean is the fixed training-selection objective evaluated on the
immutable validation pack. It is not a velocity or vorticity error against the
independent Navier--Stokes GT.

### Relationship to 2D_karman

The directory and command surface intentionally match
[`2D_karman`](../2D_karman/README.md): both cases keep `configs/`, `scripts/`,
`tests/`, a canonical training entry point, a prediction/video entry point,
fixed-identity peeling, and Ritz postprocessing inside the case directory.

| Concern | `2D_karman` | `2D_airfoil` |
| --- | --- | --- |
| Geometry | Translating cylinder | C2-smooth analytic airfoil |
| Parameters | Cylinder position parameter | Thickness `beta` and angle `alpha` |
| Canonical modes | 96 | 256 |
| SSS order | Fixed identity peeling | Fixed identity peeling |
| Training backend | Case reference backend | Hand-written fixed-graph CuPy backend |
| MLP arithmetic | Case reference precision | FP16 shadow forward, FP32 master/Adam |
| Independent GT | Re=100 Karman wake | D2Q9 TRT Navier--Stokes trajectories |
| Reconstruction | Ritz projection | Velocity- and vorticity-optimal Ritz fits |

The matching format is structural and semantic; it does not imply identical
geometry, boundary data, Reynolds number, or GT fields.

## Case layout

```text
2D_airfoil/
|-- configs/
|   |-- k256_final.json
|   |-- gt_re1500_separated.json
|   `-- continuous_gt_video_k256_fp16_farfield.json
|-- scripts/
|   |-- train.py
|   |-- plot_training.py
|   |-- solve_ns.py
|   |-- render_ns.py
|   |-- predict_video.py
|   |-- evaluate_single_gt.py
|   |-- train_hardcoded.py
|   `-- run_continuous_postprocess.py
|-- src/
|   `-- eigenfluid_sss/basisnet/airfoil2d/
|       |-- hardcoded_backend.py
|       |-- hardcoded_postprocess.py
|       |-- ns_reference.py
|       |-- postprocess.py
|       `-- training.py
|-- tests/
|   `-- test_smoke.py
`-- README.md
```

`train.py` and `predict_video.py` are the canonical public entry points,
matching the `2D_karman` layout. The longer implementation files are kept
separate so the explicit fixed-graph derivative and postprocessing passes
remain inspectable.

All commands below are run from the repository root.

## Environment and installation

Recommended environment:

- Windows 10/11 or Linux
- Python 3.11-3.13
- CUDA GPU with at least 16 GB of memory
- CuPy matching the installed CUDA runtime
- PyTorch 2.5 or newer for the independent LBM reference path
- `ffmpeg` available on `PATH` for MP4 generation

Install the project and run the case checks:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q BasisNet/cases/2D_airfoil/tests
python -m ruff check BasisNet/cases/2D_airfoil
```

The tests cover analytic geometry, mesh-free Sobol sampling, joint parameter
conditioning, free-stream boundaries, the generalized eigenproblem, and a
small independent NS solve.

## Training

The complete canonical configuration is
`BasisNet/cases/2D_airfoil/configs/k256_final.json`.

```powershell
$case = "BasisNet\cases\2D_airfoil"
$run = "$case\results\k256_final\run_001"

python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_final.json" `
  --run-dir $run
```

For a new run, the public entry point first creates:

```text
reference_inputs/initial_epoch0000.npz
reference_inputs/initial_epoch0000.json
reference_inputs/fixed_validation_scipy_sobol_v1.npz
reference_inputs/manifest.json
```

The initialization and immutable validation pack use NumPy/SciPy only.
Training then runs the explicit CuPy forward/reverse pass.

Resume from a complete checkpoint:

```powershell
python -u "$case\scripts\train.py" `
  --config "$case\configs\k256_final.json" `
  --run-dir $run `
  --resume "$run\checkpoints\hardcoded_epoch_5000.npz"
```

The checkpoint stores FP32 master parameters, Adam state, epoch, learning
rate scheduler state, early-stopping state, and training history. FP16
shadow weights are synchronized from the masters.

### Per-epoch learning logic

1. A deterministic two-dimensional Sobol sample provides one `(beta, alpha)`
   pair. Every twentieth epoch inserts a joint-domain corner.
2. The sampler creates 4096 continuous fluid points without a training mesh.
3. Jet2 propagates value, spatial gradient, and spatial Hessian through each
   sine MLP using fused batched matrix products.
4. Streamfunction derivatives produce an exactly divergence-free velocity.
5. Fixed-order peeling removes components in all earlier modes.
6. Only the active mode receives the explicit reverse-pass gradient.
7. FP32 Adam updates the master parameters and synchronizes FP16 shadows.
8. Fixed validation runs every 50 epochs; checkpoints are written every 100
   epochs.
9. The learning rate is reduced after 200 epochs without sufficient
   validation improvement.
10. Training stops after 1000 epochs without a new best validation result.

### Training outputs

```text
hardcoded_train.log
data/hardcoded_training_history.json
data/hardcoded_training_history.csv
checkpoints/hardcoded_latest.npz
checkpoints/hardcoded_latest.json
checkpoints/hardcoded_epoch_*.npz
checkpoints/hardcoded_sss_frozen.npz
checkpoints/hardcoded_sss_frozen.json
```

Plot the training curves:

```powershell
python "$case\scripts\plot_training.py" --run-dir $run
```

## Independent Navier--Stokes reference

The formal GT solver is independent of basis training:

```powershell
$beta = 0.5
$alpha = 10.0
$cache = "$run\gt_beta_0p5_alpha_p10.npz"

python "$case\scripts\solve_ns.py" `
  --config "$case\configs\k256_final.json" `
  --solver-config "$case\configs\gt_re1500_separated.json" `
  --beta $beta `
  --alpha-deg $alpha `
  --output $cache
```

The formal configuration uses a `1001 x 801` isotropic lattice, TRT-D2Q9
collision, `Re_c=1500`, 60000 steps, a 40000-step warm-up, and 41
post-warm-up snapshots. The airfoil is no-slip, the left boundary is the
uniform inlet, top and bottom are uniform far field, and the right boundary
is open zero-gradient.

Each GT cache records near-wall vorticity, wake vorticity RMS, reverse-flow
fraction, minimum streamwise wake velocity, coherent-vortex counts, and
temporal wake-vorticity RMS. High-angle GT is rejected unless the separated
wake passes the configured quality gate.

Render one GT frame:

```powershell
python "$case\scripts\render_ns.py" `
  --input $cache `
  --output "$run\gt_beta_0p5_alpha_p10.png"
```

## Ritz reconstruction and videos

The frozen checkpoint is postprocessed with the same Ritz convention used
by `2D_karman`. No GT field participates in training or basis extraction.

```powershell
python "$case\scripts\predict_video.py" `
  --checkpoint "$run\checkpoints\hardcoded_sss_frozen.npz" `
  --run-dir "$run\evaluation" `
  --config "$case\configs\continuous_gt_video_k256_fp16_farfield.json"
```

For every joint parameter sample, postprocessing:

1. evaluates the raw 256-mode neural field;
2. computes velocity and vorticity from explicit spatial derivatives;
3. constructs mass and stiffness matrices on independent Sobol quadrature;
4. solves the generalized symmetric Ritz eigenproblem;
5. projects the held-out GT onto the frozen basis;
6. produces both velocity-`L2`-optimal and fluctuation-vorticity-`L2`-optimal
   reconstructions;
7. reports total-field, fluctuation, and wake errors;
8. renders the GT, reconstruction, and absolute-error panels.

Main generated outputs:

```text
postprocess/continuous_alpha_beta_gt_pinn_error.mp4
postprocess/continuous_alpha_beta_gt_pinn_vorticity_optimal_error.mp4
postprocess/continuous_pipeline_complete.json
postprocess/basis_lowest6/contact_sheet.png
postprocess/samples/
postprocess/frames/
postprocess/frames_vorticity_optimal/
```

The velocity-optimal video selects coefficients by velocity L2 minimization.
The vorticity-optimal video selects coefficients by fluctuation-vorticity L2
minimization. Each frame places GT on the top row, the basis reconstruction in
the middle row, and absolute error on the bottom row; velocity magnitude and
streamlines are shown on the left, and vorticity is shown on the right. The
workflow also exports the six lowest-frequency Ritz modes individually and as
`basis_lowest6/contact_sheet.png`.

Evaluate one existing GT cache:

```powershell
python "$case\scripts\evaluate_single_gt.py" `
  --checkpoint "$run\checkpoints\hardcoded_sss_frozen.npz" `
  --gt $cache `
  --output-dir "$run\single_gt_evaluation" `
  --phase 0.5 `
  --ritz-quadrature 4096
```

## Reproducibility

- Training seed: `20260727`.
- Spatial and joint-parameter samples use deterministic scrambled Sobol
  sequences.
- Validation uses an immutable independently seeded point pack.
- The analytic airfoil participates directly in sampling and derivatives;
  no body-fitted or Cartesian training mesh is used.
- Peeling order is fixed and no causal sorting is performed.
- Results can vary slightly with CUDA, CuPy, BLAS, and GPU architecture.
- Full K=256 training and 1001-by-801 GT generation are computationally
  expensive; use the tests before launching the complete workflow.

## Data, checkpoints, and generated media

The repository ignores every `results/` directory as well as checkpoints,
NPZ caches, logs, frames, plots, and MP4 files. Publish large trained weights
through a release, object storage, or Git LFS if needed; do not commit them
to the source tree. The GitHub publication contains only source code,
canonical configurations, tests, and documentation.
