# Fourth-domain physical quadratic reference

Generate the original two disjoint 16,384-point target sets:

```bash
python experiments/metaball/heldout_samples.py --case 4 --output runs/heldout4
```

The reference uses Gmsh 4.15.2 (including its Netgen optimizer), scikit-fem 12.0.2, meshio, scikit-image, NumPy and SciPy. Install the FEM dependencies and provide the path to the oneMKL LP64 runtime for the shift-invert solve.

Run the stages in this order, retaining the same samples, configuration and output directory:

```bash
python -m experiments.metaball.heldout_geometry4 --stage mesh --samples runs/heldout4/samples.npz --output runs/heldout4/quadratic
```

Repeat this command with `--stage quality`, `chord`, `repair`, `chart`, `manifold`, `boundary`, and `assemble`. Then solve and evaluate:

```bash
python -m experiments.metaball.heldout_geometry4 --stage solve --samples runs/heldout4/samples.npz --output runs/heldout4/quadratic --mkl-library /path/to/mkl_rt.dll
python -m experiments.metaball.heldout_geometry4 --stage evaluate --samples runs/heldout4/samples.npz --output runs/heldout4/quadratic
```

The mesh parameters are `h=0.025` and surface resolution 80. Five Netgen optimization iterations preserve the boundary. The quadratic chart projects boundary vertices and edge midpoints, retains straight interior edges, and applies the original local interior repair when required. The manifold, determinant Bernstein bounds and entire-boundary homotopy stages establish the physical chart before operator assembly.

The consistent FP64 physical N茅d茅lec mass and curl-stiffness matrices use positive Gauss鈥揇uffy quadrature with per-cell convergence tolerance 1e-6. Canonical gradient and harmonic deflation precede the 264-vector shift-invert solve and Rayleigh鈥揜itz selection of 256 modes. The spectral seed is 20260907. Evaluation solves the full fitted mass system and reconstructs the independent 16,384-point target set; `numerical/result.json` and `reconstruction_arrays.npz` contain the resulting errors and fields.

The default CPU limit is 62.5 GB with a 9 GiB free-memory reserve, two compute threads and a 7200-second stage deadline. `--RAM-limit` and `--timeout-seconds` make these execution parameters explicit. Each stage records elapsed time and the exact samples and configuration in its run directory.
