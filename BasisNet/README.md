# BasisNet

BasisNet learns parameter-dependent spatial basis functions for fluid
problems. The current implementation uses fixed-order Sequential Spectral
Subspace peeling:

1. one neural field and one optimizer are assigned to each mode;
2. each new candidate is peeled against all earlier modes;
3. earlier modes are detached during later-mode optimization;
4. the velocity, streamfunction, and velocity gradient are peeled with the
   same least-squares coefficients;
5. the normalized Rayleigh energy is minimized.

## Cases

- [2D_karman](cases/2D_karman/README.md): a translating cylinder in a
  two-dimensional channel, K=96 learned modes, and Re=100 wake
  reconstruction.
- [3D_TG](cases/3D_TG/README.md): K=256 hard-discrete-divergence-free modes
  on a periodic cube, with independent forced Taylor--Green and oscillatory
  ABC reconstruction benchmarks.

Case-specific configurations, scripts, tests, and documentation are kept
inside each case directory so that experiments remain reproducible and
self-contained. See the [case index](cases/README.md) for status, validated
reference results, and artifact policy.
