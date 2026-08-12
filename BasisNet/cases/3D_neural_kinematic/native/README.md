# Native CUDA EigenFluid-SSS

This directory contains the self-contained CUDA/C++ implementation of the
formal three-dimensional EigenFluid-SSS trainer. It does not depend on Python,
PyTorch, ATen, LibTorch, or an automatic-differentiation framework.

The implementation is intentionally narrow: it trains `K=1024` continuous
antisymmetric-tensor modes with one fixed objective and one fixed optimizer
schedule. Historical velocity-supervised and task-aware experiments are not
part of this backend or its release artifacts.

## Mathematical contract

For each raw neural antisymmetric tensor

\[
A_\theta(x;\mu)=
\begin{pmatrix}
0&A_{12}&A_{13}\\
-A_{12}&0&A_{23}\\
-A_{13}&-A_{23}&0
\end{pmatrix},
\]

the velocity, vorticity, and vorticity gradient are evaluated continuously:

\[
u_i=\sum_j\partial_j A_{ij},\qquad
\omega=\nabla\times u,\qquad
\nabla\omega=\nabla(\nabla\times u).
\]

The first-, second-, and third-order spatial derivatives are propagated by
hand-written analytic Taylor jets. There is no finite-difference step and no
training grid. Random continuous collocation points provide Monte Carlo
quadrature only.

After full causal least-squares SSS peeling against every earlier normalized
vorticity mode, the only per-mode objective is the unscaled quotient

\[
R_\omega=
\frac{\int_\Omega |\nabla\omega|^2\,dx}
     {\int_\Omega |\omega|^2\,dx}.
\]

The global optimizer objective is the direct sum of the 1024 independent
unscaled quotients. There is no `/80`, `/232`, or other spectral scaling.
There is no velocity loss, task loss, ground-truth supervision,
orthogonality penalty, divergence penalty, boundary penalty, parameter
regularizer, gradient clipping, or weight decay. Orthogonality is constructed
only by causal peeling; incompressibility follows from the antisymmetric
tensor; the analytic double-zero envelope imposes the wall condition.

The closed contracts are defined in
`include/eigenfluid_native/objective_contract.hpp` and
`include/eigenfluid_native/training_contract.hpp`.

## Implementation

- `K=1024` modes and `10000` fixed Adam steps.
- Independent Fourier/SIREN potential networks with three hidden layers of
  width 32 and three outputs `(A12,A13,A23)`.
- FP16 neural weights and primal activations with FP32 GEMM accumulation.
- FP32 coordinates, geometry, envelope, analytic derivatives, physical
  fields, peeling, Rayleigh integrals, parameter gradients, master weights,
  and Adam state.
- Matrix-form full causal peeling through Cholesky-QR and triangular solves;
  this is the batched algebraic form of sequential least-squares peeling.
- Hand-written reverse propagation for the physical operator, analytic sine
  jets, linear layers, causal peeling, and Rayleigh quotient.
- Atomic binary checkpoints containing FP32 master weights and Adam moments.
  A readable `latest.bin` is resumed automatically.
- No validation metric or ground truth can modify the training schedule or
  create a manually tuned stage.

## Requirements

- Windows x64
- Visual Studio 2019 with the x64 C++ toolchain
- CUDA Toolkit 11.8
- NVIDIA GPU supported by CUDA 11.8; the default target is `sm_89`

## Build and verify

From the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  BasisNet\cases\3D_neural_kinematic\native\build.ps1 -RunSelfTest
```

The self-test checks the CUDA runtime, a custom kernel, cuBLAS, analytic
third-order derivatives against a closed-form reference, the exact objective
contract, and the immutable mixed-precision training contract.

Run the allocation-and-touch dry run before training:

```powershell
BasisNet\cases\3D_neural_kinematic\native\build\eigenfluid_native.exe `
  --memory-dry-run --device 0 --seed 250415657
```

It tests mode blocks `1024,512,256,128,64` in order and selects the largest
divisor that respects the 23 GiB device cap and 768 MiB free-memory reserve.
On the reference RTX 4090, all 1024 modes fit in one block.

## Train

```powershell
BasisNet\cases\3D_neural_kinematic\native\build\eigenfluid_native.exe `
  --train --device 0 --seed 250415657 `
  --run-dir BasisNet\cases\3D_neural_kinematic\results\native_exact_k1024
```

The run directory contains:

- `protocol.json`: immutable mathematical and precision contract;
- `history.jsonl`: step, unscaled Rayleigh sum and range, and elapsed time;
- `latest.bin`: atomic resumable checkpoint;
- `summary.json`: final completion record;
- `failure.txt`: implementation/runtime error, when present.

For runs longer than two hours, the repository monitor checks the process,
history freshness, GPU status, checkpoint availability, and completion every
30 minutes:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  BasisNet\cases\3D_neural_kinematic\native\monitor_native_training.ps1
```

The monitor may restart a failed process from `latest.bin`, but it never
changes the objective, mode count, optimizer schedule, precision contract, or
training hyperparameters.
