# 3D parametric fan: complete reconstruction evaluation

## Checkpoint and reference

- Best checkpoint epoch: 1980
- Best validation loss: 5.2405016
- Independent GT: 2/3-dealiased GPU pseudo-spectral projection, Re=500, grid 96^3
- GT phase-periodic residual: 0.00146535
- GT grid-independent at 5%: False

## All-1024-mode projection

- Fit disturbance relative L2: 0.674749
- Validation disturbance relative L2: 0.680409
- Validation full-velocity relative L2: 0.388565
- Near-rotor disturbance relative L2: 0.819497
- Near-wake disturbance relative L2: 0.574193
- Far-wake disturbance relative L2: 0.384671

## Physical diagnostics

- Divergence L2 / gradient L2: 5.43821e-07
- GT C4 symmetry defect: 5.47432e-05
- Reconstruction C4 symmetry defect: 0.607304
- Inlet disturbance RMS: 0
- Lateral far-field disturbance RMS: 7.39124e-08

## Interpretation

The trained span is divergence-free and satisfies the homogeneous disturbance
boundary construction, but the numerical GT uses a genuinely moving no-slip
rotor.  Only the uniform inlet was lifted before projection; the non-homogeneous
rotor-wall velocity was not lifted.  Therefore the near-rotor mismatch is a
boundary-data incompatibility as well as a finite-dimensional approximation
error.  In addition, the 96^3 GT is phase-periodic but not grid-independent, so
the reported projection errors should not be interpreted below the documented
GT discretization uncertainty.
