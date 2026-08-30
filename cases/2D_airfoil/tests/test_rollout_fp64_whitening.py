"""Tests for realized-basis FP64 whitening in the temporal rollout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


CASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CASE_ROOT / "scripts" / "rollout_gt30_fd_vs_potential_k2048.py"


def _module():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("FP64 whitening test requires CUDA")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - host dependent
        pytest.skip(f"FP64 whitening test requires CUDA: {error}")
    spec = importlib.util.spec_from_file_location("airfoil_temporal_rollout", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cp, module


def _fields(cp, rows: int = 192, modes: int = 12):
    rng = np.random.default_rng(20260830)
    weight = np.float32(1.0 / rows)
    velocity = rng.normal(size=(2 * rows, modes))
    velocity, _ = np.linalg.qr(velocity)
    velocity /= np.sqrt(float(weight))
    omega = rng.normal(size=(rows, modes))
    # A non-orthogonal realized coefficient transform emulates the digits lost
    # when a poorly conditioned FP64 Ritz rotation is applied in FP32.
    distortion = np.eye(modes)
    distortion[0, 0] = 1.012
    distortion[1, 1] = 0.991
    distortion[2, 3] = 0.018
    velocity = velocity @ distortion
    omega = omega @ distortion
    return (
        cp.asarray(velocity[:rows].astype(np.float32)),
        cp.asarray(velocity[rows:].astype(np.float32)),
        cp.asarray(omega.astype(np.float32)),
        weight,
    )


def test_fp64_refinement_restores_mass_and_rediagonalizes_stiffness() -> None:
    cp, module = _module()
    basis_u, basis_v, basis_omega, weight = _fields(cp)
    modes = basis_u.shape[1]
    result = module._refine_realized_ritz_basis_fp64(
        basis_u,
        basis_v,
        basis_omega,
        weight,
        np.arange(1, modes + 1, dtype=np.float64),
        np.eye(modes, dtype=np.float64),
        modes,
        "synthetic",
    )
    refined_u, refined_v, refined_omega, values, rotation, diagnostic = result
    mass = module._realized_mass_gram(refined_u, refined_v, weight)
    mass_error = float(
        cp.max(cp.abs(mass - cp.eye(modes, dtype=cp.float32))).get()
    )
    stiffness = module._realized_stiffness_gram(refined_omega, weight)
    stiffness_host = cp.asnumpy(stiffness).astype(np.float64)
    offdiag = stiffness_host - np.diag(np.diag(stiffness_host))

    assert diagnostic["fp64_whitening_applied"] is True
    assert diagnostic["fp64_whitening_iterations"] >= 1
    assert diagnostic["mass_orthogonality_pre_refinement_max_abs_fp32"] > 5.0e-3
    assert mass_error < 5.0e-4
    assert diagnostic["mass_orthogonality_post_refinement_max_abs_fp32"] < 5.0e-4
    assert np.max(np.abs(offdiag)) < 2.0e-3 * max(
        float(np.max(np.abs(np.diag(stiffness_host)))), 1.0
    )
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) >= -1.0e-10)
    assert rotation.shape == (modes, modes)
    assert np.all(np.isfinite(rotation))


def test_well_orthogonalized_basis_is_left_unchanged() -> None:
    cp, module = _module()
    basis_u, basis_v, basis_omega, weight = _fields(cp)
    # Undo the synthetic distortion with an FP64 solve, then represent the
    # well-conditioned basis in FP32 as it appears in production.
    mass = cp.asnumpy(module._realized_mass_gram(basis_u, basis_v, weight)).astype(
        np.float64
    )
    values, vectors = np.linalg.eigh(0.5 * (mass + mass.T))
    correction = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    correction_gpu = cp.asarray(correction.astype(np.float32))
    basis_u = basis_u @ correction_gpu
    basis_v = basis_v @ correction_gpu
    basis_omega = basis_omega @ correction_gpu
    original_ids = (id(basis_u), id(basis_v), id(basis_omega))

    result = module._refine_realized_ritz_basis_fp64(
        basis_u,
        basis_v,
        basis_omega,
        weight,
        np.arange(1, basis_u.shape[1] + 1, dtype=np.float64),
        np.eye(basis_u.shape[1], dtype=np.float64),
        basis_u.shape[1],
        "well-conditioned",
    )
    assert result[-1]["fp64_whitening_applied"] is False
    assert (id(result[0]), id(result[1]), id(result[2])) == original_ids
