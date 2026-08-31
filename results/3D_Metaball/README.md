# 3D Metaball

This bundle contains the K=1024 reconstruction comparison for a rounded cube
with three connected spheres.

## Common protocol

- Geometry input: canonically ordered sphere `xyzr` parameters.
- Schedule: 2,000 training geometries and 8 validation geometries.
- Representation: 1,024 complete antisymmetric tensor-potential modes.
- Boundary condition: squared analytic envelope.
- Samples: 9,192 training points per epoch and 4,096 validation points per
  geometry.
- Evaluation: all 1,599,754 fluid nodes of a `128 x 128 x 128` grid.
- Vorticity: common second-order finite-difference curl.
- Arithmetic: CUDA IEEE FP32 with TF32 disabled.

The three GEN variants share the seed (`20260823`), geometry schedule, sample
counts, stable raw-Rayleigh sorting, and two-pass mass-inner-product MGS. They
differ only in the Rayleigh objective and analytic derivative order.

## Files

- `configs/`: exact configurations for the five compared methods.
- `metrics/k1024_comparison.{csv,json}`: numerical comparison and evaluation
  metadata.
- `figures/k1024_gt_plus_five_methods_3d.png`: GT and five methods at a common
  absolute isovorticity level.
- `MANIFEST.sha256`: bundle integrity hashes.
