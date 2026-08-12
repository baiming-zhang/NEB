from __future__ import annotations

import numpy as np

from eigenfluid_sss.basisnet.airfoil2d.rom_dynamics import (
    OpenBoundaryTerms,
    variational_advection_jvp,
    variational_advection_rhs,
)


def test_variational_term_preserves_coefficient_energy_without_mean() -> None:
    rng = np.random.default_rng(20260802)
    q, k = 97, 13
    basis_u = rng.normal(size=(q, k))
    basis_v = rng.normal(size=(q, k))
    basis_omega = rng.normal(size=(q, k))
    coefficients = rng.normal(size=k)
    rhs = variational_advection_rhs(
        np,
        basis_u,
        basis_v,
        basis_omega,
        coefficients,
        0.03125,
    )
    assert abs(float(coefficients @ rhs)) < 1.0e-11


def test_variational_jvp_matches_centered_difference() -> None:
    rng = np.random.default_rng(20260803)
    q, k = 83, 11
    basis_u = rng.normal(size=(q, k))
    basis_v = rng.normal(size=(q, k))
    basis_omega = rng.normal(size=(q, k))
    coefficients = rng.normal(size=k)
    direction = rng.normal(size=k)
    mean_u = rng.normal(size=q)
    mean_v = rng.normal(size=q)
    mean_omega = rng.normal(size=q)
    epsilon = 1.0e-6

    def rhs(value: np.ndarray) -> np.ndarray:
        return variational_advection_rhs(
            np,
            basis_u,
            basis_v,
            basis_omega,
            value,
            0.017,
            mean_u,
            mean_v,
            mean_omega,
        )

    finite_difference = (
        rhs(coefficients + epsilon * direction)
        - rhs(coefficients - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = variational_advection_jvp(
        np,
        basis_u,
        basis_v,
        basis_omega,
        coefficients,
        direction,
        0.017,
        mean_u,
        mean_v,
        mean_omega,
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-8, atol=2.0e-8)


def test_vector_quadrature_weights_match_explicit_sum() -> None:
    rng = np.random.default_rng(20260804)
    q, k = 19, 7
    basis_u = rng.normal(size=(q, k))
    basis_v = rng.normal(size=(q, k))
    basis_omega = rng.normal(size=(q, k))
    coefficients = rng.normal(size=k)
    weights = rng.uniform(0.01, 0.04, size=q)
    u = basis_u @ coefficients
    v = basis_v @ coefficients
    omega = basis_omega @ coefficients
    expected = np.sum(
        weights[:, None]
        * (basis_u * (omega * v)[:, None] - basis_v * (omega * u)[:, None]),
        axis=0,
    )
    actual = variational_advection_rhs(
        np,
        basis_u,
        basis_v,
        basis_omega,
        coefficients,
        weights,
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


def test_open_boundary_adds_total_pressure_energy_flux() -> None:
    rng = np.random.default_rng(20260805)
    q, qb, k = 31, 9, 8
    basis_u = rng.normal(size=(q, k))
    basis_v = rng.normal(size=(q, k))
    basis_omega = rng.normal(size=(q, k))
    coefficients = rng.normal(size=k)
    outlet_u = rng.normal(size=(qb, k))
    outlet_v = rng.normal(size=(qb, k))
    outlet_normal = rng.normal(size=(qb, k))
    weights = rng.uniform(0.01, 0.05, size=qb)
    pressure = 0.17
    boundary = OpenBoundaryTerms(
        basis_u=outlet_u,
        basis_v=outlet_v,
        basis_normal=outlet_normal,
        weights=weights,
        static_pressure=pressure,
    )
    rhs = variational_advection_rhs(
        np,
        basis_u,
        basis_v,
        basis_omega,
        coefficients,
        0.023,
        open_boundary=boundary,
    )
    outlet_velocity_u = outlet_u @ coefficients
    outlet_velocity_v = outlet_v @ coefficients
    test_normal_velocity = outlet_normal @ coefficients
    expected_energy_rate = -np.sum(
        weights
        * (pressure + 0.5 * (outlet_velocity_u**2 + outlet_velocity_v**2))
        * test_normal_velocity
    )
    np.testing.assert_allclose(
        coefficients @ rhs, expected_energy_rate, rtol=2.0e-13, atol=2.0e-13
    )


def test_open_boundary_jvp_matches_centered_difference() -> None:
    rng = np.random.default_rng(20260806)
    q, qb, k = 29, 7, 6
    basis_u = rng.normal(size=(q, k))
    basis_v = rng.normal(size=(q, k))
    basis_omega = rng.normal(size=(q, k))
    coefficients = rng.normal(size=k)
    direction = rng.normal(size=k)
    mean_u = rng.normal(size=q)
    mean_v = rng.normal(size=q)
    mean_omega = rng.normal(size=q)
    boundary = OpenBoundaryTerms(
        basis_u=rng.normal(size=(qb, k)),
        basis_v=rng.normal(size=(qb, k)),
        basis_normal=rng.normal(size=(qb, k)),
        weights=rng.uniform(0.01, 0.05, size=qb),
        static_pressure=0.11,
        mean_u=rng.normal(size=qb),
        mean_v=rng.normal(size=qb),
    )
    volume_weights = rng.uniform(0.01, 0.03, size=q)
    epsilon = 1.0e-6

    def rhs(value: np.ndarray) -> np.ndarray:
        return variational_advection_rhs(
            np,
            basis_u,
            basis_v,
            basis_omega,
            value,
            volume_weights,
            mean_u,
            mean_v,
            mean_omega,
            boundary,
        )

    finite_difference = (
        rhs(coefficients + epsilon * direction)
        - rhs(coefficients - epsilon * direction)
    ) / (2.0 * epsilon)
    analytic = variational_advection_jvp(
        np,
        basis_u,
        basis_v,
        basis_omega,
        coefficients,
        direction,
        volume_weights,
        mean_u,
        mean_v,
        mean_omega,
        boundary,
    )
    np.testing.assert_allclose(analytic, finite_difference, rtol=3.0e-8, atol=3.0e-8)
