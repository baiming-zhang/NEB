# Three-dimensional Metaball experiments

Run commands from the repository root. Install the GPU and FEM extras for native CuPy training and fitted tetrahedral assembly. NumPy/SciPy provide the full-mass contraction, NKB, reconstruction and ROM tools. Training saves complete optimizer-state bundles every 100 updates and at update 2000; validation runs every 25 updates. Analytic ANN fields retain the configured FP32 network arithmetic and promote complete physical fields and contractions to FP64.

## Geometry, sampling and physical operators

```sh
python experiments/metaball/prepare.py mesh --config experiments/metaball/configs/spectral_k32.json --K 1024 --output runs/mesh
python experiments/metaball/prepare.py source --config experiments/metaball/configs/spectral_k32.json --output runs/source
python experiments/metaball/prepare.py physical --mesh runs/mesh/modes.npz --output runs/physical
python experiments/metaball/prepare.py source-map --physical runs/physical --samples runs/source/samples.npz --output runs/source-map
python experiments/metaball/prepare.py volume --physical runs/physical --output runs/volume
python experiments/metaball/prepare.py dns --output runs/dns
python experiments/metaball/prepare.py random20 --dns runs/dns --output runs/targets20
```

The source-map command writes the full ordered analytic curl operator and inverse-map inputs. Pass its generated filenames to the corresponding training options; the physical-volume value comes from runs/volume/complete.json. The physical mesh uses the positive 16-point q1 cell quadrature, full relative edge space and scalar/harmonic kernels.

## Training and full-grid representation

```sh
python experiments/metaball/train.py --method Potential --K 32 --config experiments/metaball/configs/spectral_k32.json --samples runs/source/samples.npz --physical runs/physical --source-operator runs/source-map/B_source_curl.npz --source-map runs/source-map/source_inverse_map.npz --output runs/ccq32
python experiments/metaball/train.py --method Velocity --K 32 --config experiments/metaball/configs/spectral_k32.json --output runs/vrq32
python experiments/metaball/train.py --method Potential-AL --K 32 --config experiments/metaball/configs/metaball-potential-al-k32-seed0.json --samples runs/source/samples.npz --output runs/gcp-al32
python experiments/metaball/train.py --method Potential-FG --finite-gauge-setting scalar-gauge --K 256 --config experiments/metaball/configs/metaball-potential-fg-k256-seed0-scalar-gauge.json --samples runs/source/samples.npz --output runs/gcp-fg256
python experiments/metaball/train.py --method NKB --K 32 --config experiments/metaball/configs/spectral_k32.json --output runs/nkb32
python experiments/metaball/capacity.py --K 128 --output runs/vrq128
```

Potential is the canonical curl quotient; Velocity uses the componentwise-Dirichlet velocity Rayleigh quotient. Potential-AL retains its augmented-Lagrangian gauge objective. Potential-FG uses a finite scalar projector; the seed-0 control has its own frequency configuration. Each family has its own configuration and objective. Capacity configs select the actual per-K velocity protocol.

`evaluate.py` reconstructs every requested mode in 2048-point blocks. Use `--flat runs/targets20/fluid_flat_indices.npy --coordinates runs/dns/coordinates.npy` for the full fluid grid, or `--points points.npy` for an ordered assessment set. Supply the same method, config and complete checkpoint; Potential also takes the physical/source inputs above, and Laplacian takes `--mesh runs/mesh/modes.npz`. The generated basis.npy is FP64 `(N,3,K)` and its mass.npz contains the complete nonorthogonal mass matrix.

```sh
python experiments/metaball/run.py paired20 --help
python experiments/metaball/operators.py derive --basis runs/basis/basis.npy --flat runs/targets20/fluid_flat_indices.npy --normal runs/dns/boundary_normal_xyz.npy --solid runs/dns/solid_fraction.npy --output runs/derived
python experiments/metaball/operators.py assemble --basis runs/basis/basis.npy --derived runs/derived/derived.npy --output runs/operators
python experiments/metaball/run.py rom --operators runs/operators/operators.npz --quadratic runs/operators/quadratic.npy --initial initial_coefficients.npy --output runs/rom
```

Paired evaluation uses all 20 saved initial fields, the full Gram solve and a second pass for direct reconstruction errors. The ROM uses full M/A and packed quadratic interactions from the original 128-cubed spectral operators.

## Held-out geometry and solver variants

```sh
python experiments/metaball/heldout_samples.py --case 1 --output runs/heldout1
python experiments/metaball/evaluate.py --help
python experiments/metaball/heldout.py --fit-basis fit_basis.npy --assessment-basis assessment_basis.npy --samples runs/heldout1/samples.npz --output runs/reconstruction1
```

For a held-out learned basis, put samples.npz fit_points and assessment_points into separate NPY files, set fixed_training_geometry to its geometry in a copy of the model configuration, and run evaluate.py twice with the same complete 256-mode checkpoint and those point files. For Potential and target-specific Laplacian, regenerate the geometry's fitted mesh and physical/source-map inputs using prepare.py. GEOMETRY4.md gives the curved-chart finite-element pipeline for the fourth geometry.

The default canonical backend preserves the original oneMKL/PARDISO factorization and projected solves. Set `NEB_MKL_RUNTIME` to the directory containing `libmkl_rt.so` on Linux or `mkl_rt.dll` on Windows (for example, `export NEB_MKL_RUNTIME=/opt/intel/oneapi/mkl/latest/lib`); the loader searches that directory recursively. To select the cuDSS 0.8 variant, set NEB_CUDSS_LIBRARY to the installed library and pass `--canonical-module experiments/metaball/cudss.py --canonical-class TreeCanonicalLift`; its true N/T solves use the original unshifted matrix. Resource journals enforce aggregate RAM/GPU limits and available-memory reserves with a dynamic pool budget.

For the matched finite-gauge comparison, use `--method Potential-FG --K 32 --config experiments/metaball/configs/metaball-potential-fg-k32-seed0.json` with the source samples. Repeat the K64/K128/K256 configurations from the checkpoint index. This entry uses the original finite441 projector and its true transpose, the shared Velocity sampler and Adam, and all eight scheduled geometry diagnostics.
