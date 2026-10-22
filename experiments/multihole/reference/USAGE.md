# Multihole FEM references

Run commands from the repository root. Mesh and full-mass operator construction use NumPy, SciPy and MeshPy. Native reference integration and Laplacian controls additionally use CuPy with a compatible CUDA installation. CPU checks use NumPy and SciPy.

```bash
python -m experiments.multihole.reference.run --mode mesh --case one_circle --output runs/reference/one_circle/mesh
python -m experiments.multihole.reference.run --mode operators --case one_circle --mesh runs/reference/one_circle/mesh/mesh.npz --output runs/reference/one_circle/operators
python -m experiments.multihole.reference.run --mode adaptive-mesh --case one_circle --fields runs/fields/one_circle --sample 0 --output runs/reference/one_circle/refinement
python -m experiments.multihole.reference.run --mode trajectory --case one_circle --fields runs/fields/one_circle --sample 0 --mesh runs/reference/one_circle/refinement/mesh4.npz --dt 0.001 --final-time 1 --snapshot-dt 0.02 --output runs/reference/one_circle/sample00_dt001
```

The refinement directory records each mesh and its interpolation error. Choose its final published mesh from `mesh_refinement.json`; `mesh4.npz` above illustrates that choice. The same field specification is read without resampling or renormalization. Cases are `one_circle`, `circle_square`, and `circle_square_triangle`. Field files are `sample00.npz` through `sample19.npz` from the multihole fields command. Each includes the full lattice, trace constants, normalization and seed hierarchy.

`operators/quadrature.npz` contains `points[N,2]` and physical `weights[N]` in cell-major, quadrature-major order, ready for `experiments.multihole.evaluate`. Weights integrate physical volume without mean normalization. `mass.npz` and `stiffness.npz` contain full consistent P2/P3 FEM matrices; `space.npz` holds all high-order nodes and cells. Degree defaults to P3, product-Gauss order to five (exact through polynomial degree eight).

The native trajectory uses the original full-mass vorticity/circulation system, analytical initial vorticity projection, harmonic boundary traces, IMEX midpoint/Crank–Nicolson integration and CG true-residual correction. It writes every requested full nodal frame and harmonic reference. Running separately at `--dt 0.002` and `--dt 0.001` enables temporal comparison; spatial comparison uses separate refinement meshes. Compare the paired trajectories and refinement levels using the same field specification and physical observation times.

## Numerical Laplacian P/PH controls

```bash
python -m experiments.multihole.reference.sparse_controls --case one_circle --mesh runs/reference/one_circle/mesh/mesh.npz --modes 1024 --iterations 800 --output runs/Laplacian/one_circle/PH1024
python -m experiments.multihole.reference.sparse_controls --case one_circle --mesh runs/reference/one_circle/mesh/mesh.npz --modes 1024 --positive-only --output runs/Laplacian/one_circle/P1024
```

These commands retain the entire constrained P1 space and consistent mass matrix. Positive modes solve `S x = lambda M x` with the original true-residual LOBPCG source, diagonal stiffness preconditioner and tolerance `1e-12`. The full free-node count must exceed five times the positive-mode count. P uses K positive modes; PH uses K−h positive modes plus the h harmonic boundary-trace functions, normalized by their full stiffness Gram matrix. Final relative true residuals and energy Gram errors must each be below `1e-8`. Assembly uses the full consistent mass matrix, all specified K modes and global orthogonalization.

`--initial` accepts a complete same-mesh archive with `free_vectors` and positive `eigenvalues`, using the original warm-start expression. Without it, a reproducible PCG64 full-space initial block is generated (`--initial-seed 20260928`). `controls.npz` exports all nodes, triangles, free nodes, potentials, rates and residual history. Dense small-mesh controls use `reference.run --mode controls --modes 32` and the original Cholesky generalized eigensolve.

## Compact-vortex initial ensemble

```bash
python -m experiments.multihole.reference.run --mode vortex-trajectory --case one_circle --mesh runs/reference/one_circle/mesh/mesh.npz --sample 0 --dt 0.002 --snapshot-dt 0.25 --output runs/reference/one_circle/vortex_sample00
```

This separate P1 reference uses the original compact Wendland C4 law: 3–7 physically sampled vortices, support radii 0.65–0.95, harmonic-to-vortical energy ratio sampled uniformly on [0,0.30] and peak speed 0.5. The three supplied protocol files retain the original per-topology master seeds 20261001/2/3, `SeedSequence.spawn(20)` sample seeds and independent ratio RNG (`master_seed+17`). `--sample` selects the exact original ensemble member; `--vortex-config` accepts the corresponding protocol. The full sampled specification is saved with the trajectory. `initial_state.npz` stores the original FEM streamfunction and cell velocity; `mass.npz` and `stiffness.npz` provide the same-mesh consistent operators for full-mass projection and trajectory assessment. Generate samples 0–19 into `sample00`–`sample19` directories for the compact-vortex learned-ROM target provider. The compact-vortex and quintic streamfunction field ensembles retain their separate sampling laws.

## CPU checks

```bash
python -m experiments.multihole.reference.check_cpu
python -m experiments.multihole.reference.check_sparse_cpu
```

These exercise public mesh/operator output, P2/P3 polynomial and derivative exactness, full-mass area/positivity, original field-format round trip and the unchanged sparse eigensolver against a dense generalized-eigenvalue oracle. They use temporary directories and leave existing run directories intact.
