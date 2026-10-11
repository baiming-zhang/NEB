# Measurement plots and field plates

From the repository root, redraw the supplied measurements:

```bash
neb figures --output outputs/paper_figures
```

This writes PNG and vector PDF versions of paired Metaball errors, capacity, retained-field energy and final errors, harmonic-span fractions, periodic seed errors, NKB comparisons, gauge-solver residuals and coefficient-stage costs. Periodic shaded bands summarize training seeds; retained-topology error bars summarize two fixed physical initial fields per topology. Both use sample standard deviation with `ddof=1`.

The field plates use the physical fields produced by the experiment recipes. Periodic evaluation saves `fields_dt*.npz` with `u`, `v`, `omega`, coefficients and time. Multiply connected operator outputs retain physical quadrature points and weights, while trajectories retain coefficient states. Evaluate the checkpoint and its harmonic modes analytically on the retained points, then contract the resulting spatial basis with each coefficient vector. Metaball evaluation writes the ordered full-domain basis, and the ROM produces coefficient states on the same basis. Preserve mesh coordinates, masks, point order, time and the common color range when comparing methods.

For planar plates, display signed vorticity on the physical domain with the red/blue diverging palette and draw the same boundary geometry in every panel. For volumetric plates, import the ordered grid velocity/vorticity and solid mask into Houdini, convert the grid to volume fields, and render the shared camera, lighting and geometry with a common physical color scale. Export rendered panels to PNG before composing the final plate. The method diagram and airfoil topology illustration are schematic assets in the manuscript.

For a basis atlas, evaluate the chosen checkpoint analytically, select the first eight ordered modes from its spatial basis and use one common physical scale within the comparison. SF-R uses the streamfunction differential lift, CCQ uses analytic curl, VRQ uses the velocity output, and the LEB columns follow the numerical eigenvalue order.
