











from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenBoundaryTerms:








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


def frozen_base_rhs_offset(
    xp: Any,
    basis_u: Any,
    basis_v: Any,
    basis_omega: Any,
    quadrature_weight: Any,
    mean_u: Any,
    mean_v: Any,
    mean_omega: Any,
    open_boundary: OpenBoundaryTerms | None = None,
) -> Any:











    zeros = xp.zeros(basis_u.shape[1], dtype=basis_u.dtype)
    return -variational_advection_rhs(
        xp,
        basis_u,
        basis_v,
        basis_omega,
        zeros,
        quadrature_weight,
        mean_u,
        mean_v,
        mean_omega,
        open_boundary,
    )


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
