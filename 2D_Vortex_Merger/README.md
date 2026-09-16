## Latest published implementation

See [the periodic gauge-canonical / no-gauge K64 and K1024 comparison](GAUGE_CANONICAL_EIGENFLUID/outputs/comparison_64_1024/REPORT.md) and [publication contents and reproduction notes](PUBLICATION.md). The free-boundary experiments described below are a separate experiment.

# 2D vortex merger Navier--Stokes ground truth

This is a standalone case folder at the repository root.  Its DNS, learned
free-boundary learned eigenbasis, tests, configurations, and outputs live entirely below
`2D_Vortex_Merger/`.

This case generates a direct numerical solution of the two-dimensional,
incompressible Navier--Stokes equations in vorticity form,

```text
partial_t omega + u dot grad(omega) = nu laplacian(omega),
u = (partial_y psi, -partial_x psi),  -laplacian(psi) = omega.
```

The setup follows the public vortex-pair-merging benchmark of Wang et al.
(2025): a doubly periodic `[0, 2*pi)^2` box, two equal Gaussian vortices with
`sigma=0.1*pi`, circulation `Gamma=0.2*pi`, separation `l=0.59`, and
`Re=350`.  The centers are `(pi-l/2, pi)` and `(pi+l/2, pi)`.  The spatially
constant vorticity component is removed because the curl of a periodic
velocity field must have zero spatial mean.

Spatial derivatives and the Poisson solve are Fourier spectral.  The
quadratic advection term is evaluated pseudo-spectrally with the Orszag 2/3
rule, and viscosity is integrated exactly by the fourth-order ETDRK4 scheme
of Kassam and Trefethen.  This is an unforced Navier--Stokes DNS, not a
reduced model, learned surrogate, or semi-Lagrangian rollout.

## Generate the 128 x 128 GT

From the repository root:

```powershell
python -u 2D_Vortex_Merger/scripts/generate_gt.py
```

The default run writes 61 uniformly spaced frames from `t=0` through
`t=6*pi`, including the literature comparison times `0`, `2*pi`, `4*pi`,
and `6*pi`.  The output HDF5 file contains `time`, `x`, `y`, `vorticity`,
`streamfunction`, `velocity_u`, and `velocity_v`, all in double precision.
It also writes JSON diagnostics and red/blue PNG and vector PDF previews.

For a temporal refinement check:

```powershell
python -u 2D_Vortex_Merger/scripts/generate_gt.py --validate-time
```

Render every stored GT frame as a 1920 x 1080 H.264 video:

```powershell
python -u 2D_Vortex_Merger/scripts/render_gt_video.py
```

## Learn the free-boundary raw-xy K=1024 eigenbasis

The DNS above remains the standard doubly periodic vortex-merger benchmark.
For the requested learned basis experiment, however, the neural streamfunction
uses only the physical `x,y` coordinates: no Fourier/positional encoding, no
boundary envelope, and no periodic seam constraint.  All four sides are free.
The spatially constant streamfunction gauge is projected out before the
Potential-Rayleigh causal sort and two-pass peeling.

The sweep fixes five affine layers (four hidden plus one output), trains 1024
modes for 2000 epochs, and evaluates widths 32, 64, 128, 256, and 512.  Each
width is visualized immediately after training as red/blue analytic-vorticity
modes on a 128 x 128 grid including all four free boundaries:

```powershell
python -u 2D_Vortex_Merger/scripts/run_free_rawxy_k1024_depth5_width_sweep_e2000.py
```

After each width reaches epoch 2000, the runner evaluates the analytic Jet2
velocity and vorticity fields, projects GT frame 0 into the learned span, and
advances all 61 GT output times with the de Witt--Lessig--Fiume
vorticity--Galerkin ROM.  Evaluation reports plain elementwise MSE (not a
relative or normalized error): first-frame, final-frame, and mean over all GT
frames.  It writes PNG/PDF error curves, PNG/PDF first/final velocity
streamline comparisons, the full field archive, a per-frame CSV, and an H.264
streamline video before the next width begins.

The wider-separation variant requested for a visually explicit merger uses
`l=0.90` (`l/sigma=2.86`) and is generated with:

```powershell
python -u 2D_Vortex_Merger/scripts/generate_gt.py `
  --separation 0.9 `
  --output 2D_Vortex_Merger/outputs/gt_128_re350_l0p90_t6pi `
  --validate-time
```

## References

- B. Wang, Z. Meng, Y. Zhao, et al., *Quantum lattice Boltzmann method for
  simulating nonlinear fluid dynamics*, npj Quantum Information 11, 196
  (2025), DOI: 10.1038/s41534-025-01142-6.
- A.-K. Kassam and L. N. Trefethen, *Fourth-order time-stepping for stiff
  PDEs*, SIAM Journal on Scientific Computing 26(4), 1214--1233 (2005),
  DOI: 10.1137/S1064827502410633.
- S. A. Orszag, *Comparison of pseudospectral and spectral approximation*,
  Studies in Applied Mathematics 51(3), 253--259 (1972),
  DOI: 10.1002/sapm1972513253.
