# Connected 3D three-sphere metaball

This case trains a geometry-conditioned K=64 Dirichlet basis on a rounded
unit cube containing one connected smooth obstacle made from three spheres.
It uses the fixed-three-component Victoria/arXiv-v1 distribution:

- each center coordinate is IID uniform on `[0.25, 0.75]`;
- each radius is IID uniform on `[0.1, 0.3]`;
- samples are rejection-conditioned so that the sphere-overlap graph is
  connected and the obstacle remains separated from the outer wall;
- the sphere distances are composed with a soft minimum at `tau=0.02`.

Every basis mode is an independent SIREN returning the three independent
entries of an antisymmetric 3D potential.  The potential is multiplied by a
hard single-zero envelope on both boundaries.  Training uses the Dirichlet
Rayleigh quotient of this three-component potential, geometry-specific
Rayleigh sorting, and two modified Gram--Schmidt passes.  Only the peeled
basis is diagnosed or reported.

```powershell
python scripts/train_validate.py --config configs/k64_three_sphere_connected_siren_dirichlet_500.json
```

The command is resumable from `checkpoints/latest.pt`.  To run only the frozen
held-out validation and regenerate the PNG/PDF report:

```powershell
python scripts/train_validate.py --config configs/k64_three_sphere_connected_siren_dirichlet_500.json `
  --validate-only --resume results/k64_siren_dirichlet_e500/run_001/checkpoints/final.pt
```

