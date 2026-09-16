from __future__ import annotations

import sys
from pathlib import Path

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.vortex_merger2d.witt_rom import (  # noqa: E402
    advection_rhs,
    generalized_symmetric_eig,
)
from rollout_free_rawxy_witt import aggregate_mse, mse_rows  # noqa: E402


def test_de_witt_advection_is_energy_orthogonal() -> None:
    rng = np.random.default_rng(4)
    u = cp.asarray(rng.normal(size=(31, 7)).astype(np.float32))
    v = cp.asarray(rng.normal(size=(31, 7)).astype(np.float32))
    omega = cp.asarray(rng.normal(size=(31, 7)).astype(np.float32))
    coefficients = cp.asarray(rng.normal(size=7).astype(np.float32))
    rhs = advection_rhs(u, v, omega, coefficients, cp.float32(0.13))
    power = float((coefficients @ rhs).get())
    scale = float((cp.linalg.norm(coefficients) * cp.linalg.norm(rhs)).get())
    assert abs(power) <= 2.0e-6 * max(scale, 1.0)


def test_generalized_ritz_basis_is_mass_orthonormal() -> None:
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(19, 6))
    mass = matrix.T @ matrix
    stiffness_seed = rng.normal(size=(6, 6))
    stiffness = stiffness_seed.T @ stiffness_seed
    eigenvalues, rotation, rank = generalized_symmetric_eig(
        stiffness, mass, 6, sample_rows=19
    )
    assert rank == 6
    np.testing.assert_allclose(rotation.T @ mass @ rotation, np.eye(6), atol=2e-12)
    assert np.all(np.diff(eigenvalues) >= 0.0)


def test_reported_errors_are_plain_elementwise_mse() -> None:
    zeros = np.zeros((2, 2, 2), dtype=np.float32)
    predicted_u = np.ones_like(zeros)
    predicted_v = np.full_like(zeros, 3.0)
    predicted_omega = np.full_like(zeros, 2.0)
    rows = mse_rows(
        np.asarray([0.0, 1.0]),
        zeros,
        zeros,
        zeros,
        predicted_u,
        predicted_v,
        predicted_omega,
    )
    assert rows[0]["velocity_mse"] == 5.0
    assert rows[0]["vorticity_mse"] == 4.0
    summary = aggregate_mse(rows)
    assert summary["velocity_mse"] == {
        "first_frame": 5.0,
        "last_frame": 5.0,
        "mean_all_frames": 5.0,
    }
