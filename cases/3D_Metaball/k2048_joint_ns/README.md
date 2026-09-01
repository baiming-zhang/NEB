# K=2048 physical NS / de Witt ROM drivers

This directory mirrors the exact Python drivers used for the K=2048
method-informed GT, the 30-frame physical NS trajectory, four frame-0 ROM
projections and implicit rollouts, and both publication videos.

The drivers expect the trained K=2048 basis volumes and Gram/Ritz caches from
the full 3D Metaball experiment workspace. Publication metrics, figures, and
videos are stored under
[`results/3D_Metaball/k2048_joint_ns`](../../../results/3D_Metaball/k2048_joint_ns/README.md).

Main entry points:

- `scripts/construct_joint_fd_potential_ns_gt_k2048.py`: common-subspace
  initial condition, incompressible projection, NS trajectory, and frame-0
  acceptance audit.
- `scripts/run_joint_gt30_four_method_k2048_3d.py`: frame-0 projection and
  independent implicit de Witt rollout for the four K=2048 ROMs.
- `scripts/render_joint_gt30_four_method_k2048_three_slices_with_errors.py`:
  annotated `3 x 5` three-slice video.
- `scripts/build_strict_global_ritz_A2048_k2048.py`: strict global Ritz basis
  construction from 6,144 candidates.

The benchmark status and full numerical protocol are recorded in the result
bundle's `protocol.json`.
