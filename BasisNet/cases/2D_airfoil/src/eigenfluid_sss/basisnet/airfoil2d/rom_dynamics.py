"""Matrix-free variational Galerkin dynamics for 2-D Eigenfluids.

The contraction implemented here is the collocation equivalent of the
energy-conserving tensor from Cui, Sen, and Kim (2018),

    C[g, h, i] = integral curl(Phi_i) dot (Phi_g cross Phi_h) dOmega.

For two-dimensional velocity fields the full third-order tensor is never
formed.  Contracting it with a coefficient vector costs O(QK), where Q is
the quadrature size and K is the number of modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenBoundaryTerms:
    """Discrete data for the total-pressure term on an open boundary.

    The rotational momentum equation contains ``-grad(p + |u|^2 / 2)``.
    Divergence-free test functions eliminate its volume contribution, but not
    the boundary integral when their normal component is non-zero.  This
    structure stores the values needed to retain that outlet contribution.
    """

    basis_u: Any
    basis_v: Any
    basis_normal: Any
    weights: Any
    static_pressure: Any = 0.0
    mean_u: Any | None = None
    mean_v: Any | None = None


def _open_boundary_rhs(
    boundary: OpenBoundaryTerms,
    coefficients: Any,
) -> Any:
    u = boundary.basis_u @ coefficients
    v = boundary.basis_v @ coefficients
    if boundary.mean_u is not None:
        u = u + boundary.mean_u
    if boundary.mean_v is not None:
        v = v + boundary.mean_v
    total_pressure = boundary.static_pressure + 0.5 * (u * u + v * v)
    return -(boundary.basis_normal.T @ (boundary.weights * total_pressure))


def _open_boundary_jvp(
    boundary: OpenBoundaryTerms,
    coefficients: Any,
    direction: Any,
) -> Any:
    u = boundary.basis_u @ coefficients
    v = boundary.basis_v @ coefficients
    if boundary.mean_u is not None:
        u = u + boundary.mean_u
    if boundary.mean_v is not None:
        v = v + boundary.mean_v
    du = boundary.basis_u @ direction
    dv = boundary.basis_v @ direction
    return -(boundary.basis_normal.T @ (boundary.weights * (u * du + v * dv)))


def variational_advection_rhs(
    xp: Any,
    basis_u: Any,
    basis_v: Any,
    basis_omega: Any,
    coefficients: Any,
    quadrature_weight: Any,
    mean_u: Any | None = None,
    mean_v: Any | None = None,
    mean_omega: Any | None = None,
    open_boundary: OpenBoundaryTerms | None = None,
) -> Any:
    """Evaluate the variational Eigenfluid advection term.

    ``xp`` may be NumPy or CuPy.  The optional mean fields define an affine
    velocity space, which is needed here because the learned perturbation
    modes vanish at the far-field boundary while the airfoil flow has a
    non-zero free-stream base flow.
    """

    u = basis_u @ coefficients
    v = basis_v @ coefficients
    omega = basis_omega @ coefficients
    if mean_u is not None:
        u = u + mean_u
    if mean_v is not None:
        v = v + mean_v
    if mean_omega is not None:
        omega = omega + mean_omega
    result = basis_u.T @ (quadrature_weight * omega * v) - basis_v.T @ (
        quadrature_weight * omega * u
    )
    if open_boundary is not None:
        result = result + _open_boundary_rhs(open_boundary, coefficients)
    return result


def variational_advection_jvp(
    xp: Any,
    basis_u: Any,
    basis_v: Any,
    basis_omega: Any,
    coefficients: Any,
    direction: Any,
    quadrature_weight: Any,
    mean_u: Any | None = None,
    mean_v: Any | None = None,
    mean_omega: Any | None = None,
    open_boundary: OpenBoundaryTerms | None = None,
) -> Any:
    """Jacobian-vector product of :func:`variational_advection_rhs`."""

    u = basis_u @ coefficients
    v = basis_v @ coefficients
    omega = basis_omega @ coefficients
    if mean_u is not None:
        u = u + mean_u
    if mean_v is not None:
        v = v + mean_v
    if mean_omega is not None:
        omega = omega + mean_omega

    du = basis_u @ direction
    dv = basis_v @ direction
    domega = basis_omega @ direction
    result = basis_u.T @ (quadrature_weight * (domega * v + omega * dv)) - basis_v.T @ (
        quadrature_weight * (domega * u + omega * du)
    )
    if open_boundary is not None:
        result = result + _open_boundary_jvp(open_boundary, coefficients, direction)
    return result
