# 3D Metaball K=2048 physical NS and ROM trajectories

This publication bundle contains a `128^3`, 30-frame incompressible
Navier--Stokes trajectory and four K=2048 reduced-order trajectories for the
canonical rounded-cube/three-metaball geometry.

## Methodological status

The initial condition is method-informed: it is constructed in a high-overlap
common subspace of the Strict FD-Lap and Potential K=2048 spaces, then passed
through an incompressible spectral projection before Navier--Stokes time
integration. It is therefore useful for studying representability and ROM
time integration, but it is **not an independent benchmark** for comparing
Strict FD-Lap against Potential. This provenance is part of the experimental
definition and must accompany reported errors.

## Protocol

- Grid: `128 x 128 x 128`; `1,599,754` fluid nodes.
- Modes: 2,048 independent coefficients per ROM.
- Saved frames: 30 at `frame_dt = 0.01`.
- Ground truth: Fourier pseudo-spectral incompressible Navier--Stokes,
  two-thirds de-aliasing, Leray projection at every stage, smooth Brinkman
  wall penalization, viscosity `2.5e-4`, and ten `dt = 0.001` substeps per
  saved frame.
- ROM initialization: each method projects GT frame 0 once on all fluid nodes.
- ROM evolution: exact viscous half steps around implicit trapezoidal de Witt
  vorticity--Galerkin advection with Newton--GMRES.
- Quadrature: 32,768 fixed points.
- Vorticity metric: common second-order finite-difference curl on the full
  grid with zero solid values.

Frame-0 velocity relative L2 errors are `0.9088%` (FD-Laplacian), `0.9797%`
(Potential), `16.5200%` (Velocity), and `83.2994%` (Vorticity). Full
per-frame velocity and vorticity errors are stored under [`metrics/`](metrics/).

## Contents

- `metrics/`: GT diagnostics, ROM substep timings, per-frame errors, and
  method summaries.
- `figures/`: first/final 3-D isovorticity and three-slice comparisons in PNG
  and PDF.
- `videos/`: 30-frame five-column 3-D and annotated three-slice MP4 videos.
- [`protocol.json`](protocol.json): machine-readable numerical protocol and
  benchmark status.
- `MANIFEST.sha256`: SHA-256 hashes for every file in this bundle.

Binary volume data, checkpoints, basis caches, and coefficient arrays are
omitted from this compact paper-results bundle. The supplied drivers reproduce
the GT, ROM trajectories, metrics, and renderings when used with the trained
K=2048 basis assets described in the parent 3D Metaball case.

## Code

The exact experiment drivers used for construction, DNS, projection, implicit
ROM rollout, and rendering are mirrored in
[`cases/3D_Metaball/k2048_joint_ns`](../../../cases/3D_Metaball/k2048_joint_ns/README.md).
