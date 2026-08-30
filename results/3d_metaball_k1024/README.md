# 3D metaball reconstruction benchmark (K=1024)

This directory contains the lightweight, publication-facing artifacts for the
three-metaball reconstruction comparison.  It intentionally excludes training
data, checkpoints, learned basis volumes, ground-truth volumes, caches, logs,
or any auxiliary diagnostic cases.

## Common protocol

- Geometry: rounded cube containing three pairwise-overlapping Wyvill spheres.
- Learned geometry input: canonically ordered sphere `xyzr` parameters.
- Training schedule: 2,000 changing geometries and 8 disjoint validation geometries.
- Modes: 1,024 complete antisymmetric tensor-potential modes.
- Degrees of freedom: 1,024 external coefficients, one coefficient per complete mode.
- Boundary envelope: squared envelope (`e^2`), giving a double-zero potential.
- Training samples: 9,192 continuous points per epoch.
- Validation samples: 4,096 continuous points per validation geometry.
- Final evaluation: all 1,599,754 fluid nodes on a `128 x 128 x 128` grid.
- Vorticity metric: common second-order finite-difference curl on the final grid.
- Arithmetic: native CUDA IEEE FP32 with TF32 disabled.

The three learned variants use the same seed (`20260823`), the same geometry
schedule, the same sampling counts, stable raw-Rayleigh sorting, and two-pass
mass-inner-product MGS.  They differ only in the Rayleigh objective and the
analytic derivative order required by that objective.

## Files

- `configs/`: exact publication configurations for the five methods.
- `metrics/k1024_comparison.csv`: compact numerical comparison.
- `metrics/k1024_comparison.json`: metrics plus shared evaluation metadata.
- `figures/k1024_gt_plus_five_methods_3d.png`: the only included result figure;
  GT plus five methods at one common absolute isovorticity level.

No PDF is included; the publication-facing artifact is the single six-panel PNG.

The bundle is byte-identical to paper artifact commit `ecff940`. The paper's
current `master` (`d9e68c7`) intentionally removes this result directory from
the writing repository, while manuscript branch `b11d90a` keeps the identical
PNG under `assets/figures/`.
