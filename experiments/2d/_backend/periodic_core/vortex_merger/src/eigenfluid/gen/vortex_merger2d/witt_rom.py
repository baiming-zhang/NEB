"""Matrix-free de Witt vorticity--Galerkin dynamics for 2-D velocity bases."""

from __future__ import annotations

from typing import Any

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, gmres
import numpy as np


def generalized_symmetric_eig(
    stiffness: np.ndarray,
    mass: np.ndarray,
    modes: int,
    *,
    sample_rows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve ``S c = lambda M c`` on the backward-resolved mass range."""

    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    values, vectors = np.linalg.eigh(mass)
    largest = max(float(values[-1]), 0.0)
    floor = np.finfo(np.float64).eps * max(sample_rows, len(values)) * largest
    keep = values > floor
    rank = int(np.sum(keep))
    if rank < modes:
        raise np.linalg.LinAlgError(
            f"velocity mass resolves only {rank} directions for K={modes}"
        )
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    reduced = whitener.T @ stiffness @ whitener
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    order = np.argsort(eigenvalues)[:modes]
    return (
        np.maximum(eigenvalues[order], 0.0),
        whitener @ eigenvectors[:, order],
        rank,
    )


def advection_rhs(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    coefficients: cp.ndarray,
    weight: cp.ndarray,
) -> cp.ndarray:
    """Contract the de Witt rotational-form advection operator in O(QK)."""

    u = basis_u @ coefficients
    v = basis_v @ coefficients
    omega = basis_omega @ coefficients
    return basis_u.T @ (weight * omega * v) - basis_v.T @ (weight * omega * u)


def advection_jvp(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    coefficients: cp.ndarray,
    direction: cp.ndarray,
    weight: cp.ndarray,
) -> cp.ndarray:
    """Jacobian-vector product of :func:`advection_rhs`."""

    u = basis_u @ coefficients
    v = basis_v @ coefficients
    omega = basis_omega @ coefficients
    du = basis_u @ direction
    dv = basis_v @ direction
    domega = basis_omega @ direction
    return basis_u.T @ (weight * (domega * v + omega * dv)) - basis_v.T @ (
        weight * (domega * u + omega * du)
    )


class ImplicitTrapezoidalWittROM:
    """Implicit advection with Strang-split exact modal viscous decay."""

    def __init__(
        self,
        basis_u: cp.ndarray,
        basis_v: cp.ndarray,
        basis_omega: cp.ndarray,
        eigenvalues: cp.ndarray,
        quadrature_weight: float,
        viscosity: float,
        *,
        newton_tolerance: float = 2.0e-6,
        gmres_tolerance: float = 2.0e-5,
    ) -> None:
        self.u = basis_u
        self.v = basis_v
        self.omega = basis_omega
        self.eigenvalues = eigenvalues
        self.weight = cp.asarray(quadrature_weight, dtype=cp.float32)
        self.viscosity = float(viscosity)
        self.newton_tolerance = float(newton_tolerance)
        self.gmres_tolerance = float(gmres_tolerance)
        self.linear_iterations = 0
        self.newton_iterations = 0
        self.step_splits = 0

    def rhs(self, coefficients: cp.ndarray) -> cp.ndarray:
        return advection_rhs(
            self.u, self.v, self.omega, coefficients, self.weight
        )

    def jvp(self, coefficients: cp.ndarray, direction: cp.ndarray) -> cp.ndarray:
        return advection_jvp(
            self.u,
            self.v,
            self.omega,
            coefficients,
            direction,
            self.weight,
        )

    def trapezoidal_advection(
        self, start: cp.ndarray, dt: float
    ) -> cp.ndarray:
        rhs_start = self.rhs(start)
        candidate = start + cp.float32(dt) * rhs_start
        dimension = len(start)
        for _ in range(6):
            rhs_candidate = self.rhs(candidate)
            residual = candidate - start - cp.float32(0.5 * dt) * (
                rhs_start + rhs_candidate
            )
            residual_norm = float(cp.linalg.norm(residual).get())
            scale = 1.0 + float(cp.linalg.norm(candidate).get())
            if residual_norm <= self.newton_tolerance * scale:
                return candidate

            def matvec(direction: cp.ndarray) -> cp.ndarray:
                return direction - cp.float32(0.5 * dt) * self.jvp(
                    candidate, direction
                )

            operator = LinearOperator(
                (dimension, dimension), matvec=matvec, dtype=candidate.dtype
            )
            iterations = [0]

            def callback(_value: Any) -> None:
                iterations[0] += 1

            correction, info = gmres(
                operator,
                -residual,
                rtol=self.gmres_tolerance,
                atol=0.0,
                restart=20,
                maxiter=8,
                callback=callback,
                callback_type="pr_norm",
            )
            self.linear_iterations += iterations[0]
            self.newton_iterations += 1
            if info != 0 or not bool(cp.isfinite(correction).all()):
                raise RuntimeError(f"GMRES failed with info={info}")
            candidate += correction
        raise RuntimeError("implicit trapezoidal Newton iteration did not converge")

    def step(
        self, coefficients: cp.ndarray, dt: float, depth: int = 0
    ) -> cp.ndarray:
        half_decay = cp.exp(
            -cp.float32(0.5 * self.viscosity * dt) * self.eigenvalues
        )
        try:
            value = coefficients * half_decay
            value = self.trapezoidal_advection(value, dt)
            value *= half_decay
            if not bool(cp.isfinite(value).all()):
                raise RuntimeError("ROM coefficients became non-finite")
            return value
        except RuntimeError:
            if depth >= 5:
                raise
            self.step_splits += 1
            midpoint = self.step(coefficients, 0.5 * dt, depth + 1)
            return self.step(midpoint, 0.5 * dt, depth + 1)
