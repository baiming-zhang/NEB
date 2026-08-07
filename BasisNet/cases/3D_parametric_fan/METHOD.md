# 3D parametric four-blade fan: numerical contract

This case follows the same gradient-causal spectral-subspace (SSS) construction
as the other EigenFluid cases, with a framework-free implementation.

## Parameter and rotation split

The learned map has exactly three geometry inputs:

\[
\mu=(\beta_R,c_R,t/c).
\]

They generate the radial laws

\[
\beta(r)=\beta_0+(\beta_R-\beta_0)s^{0.82},\qquad
c(r)=\left[c_0+(c_R-c_0)(3s^2-2s^3)\right](1-0.2s^{10}),
\]

where \(s=(r-R_h)/(R-R_h)\). The thickness ratio \(t/c\) multiplies the
NACA-style thickness law. Sweep, rake, camber, hub geometry, and four-fold
blade count are fixed industrial-design constants rather than learned inputs.

The known state \(\theta\) is excluded from every network feature. Inference
uses the exact group action

\[
u_\theta(x;\mu)=R_x(\theta)\,
u_0(R_x(-\theta)x;\mu).
\]

Consequently, no approximate angular interpolation and no learned fourth
parameter are introduced.

## Domain and boundary construction

The training domain is the cylinder

\[
x\in[-\pi,\pi],\qquad y^2+z^2 < (2R)^2.
\]

The tensor potential is multiplied by a double-zero envelope at the inlet,
lateral far-field cylinder, and rotor. The outlet remains natural/open. The
quadrature is a mesh-free, rotationally symmetric, uniform-volume Sobol rule.

## Solenoidal representation

Each independent network predicts the three entries
\((A_{12},A_{13},A_{23})\) of an antisymmetric tensor potential. Velocity is
formed as

\[
u_i=\partial_j A_{ij},
\]

using commuting centered differences. The 19-point stencil supplies both first
and mixed second derivatives. Its assembled matrix satisfies
\(\nabla_h\cdot u_h=0\) to floating-point roundoff by construction.

## SSS update

For every geometry sample:

1. evaluate all raw modes;
2. sort by raw Rayleigh quotient in stable ascending order;
3. apply least-squares causal peeling under the cylindrical volume inner
   product;
4. detach normalized earlier modes from later losses;
5. differentiate the Rayleigh quotient by an explicit algebraic adjoint;
6. apply the exact transpose of the 19-point operator;
7. recompute each network and run the handwritten reverse pass;
8. update one independent handwritten Adam state per mode.

No automatic differentiation or neural-network framework is imported by the
training or inference process. CuPy is used only for explicit CUDA array and
matrix operations.

The browser's “dominant mode” selector ranks the already peeled orthonormal
basis columns by their final Rayleigh quotient in stable ascending order. This
display reordering does not change the causal order used during training.

## Pilot limits

The production continuation uses 1024 modes, 512 volume points, and a maximum
of 10,000 epochs, resumed from the framework-free epoch-100 checkpoint. The
learning rate is halved after 200 consecutive non-improving validations down to
the configured floor, and training stops after 1,000 consecutive non-improving
validations. Since
\(3N=1536>1024\), the sampled velocity Gram matrix can be full rank; its
effective rank and orthogonality error are recorded every epoch. This is a
maximum training budget; early stopping selects the actual terminal epoch.
