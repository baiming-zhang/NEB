"""Numerical contracts for the framework-free 3D fan backend."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (  # noqa: E402
    HardcodedFanModel,
    STENCIL_OFFSETS,
    fields_from_stencil,
    initialize_arrays,
    peel_and_adjoint,
    rotor_distance,
    stencil_adjoint_matrix,
)


def tiny_config(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "seed": 17,
        "modes": 3,
        "hidden": 7,
        "hidden_layers": 2,
        "frequencies": 2,
        "mlp_compute_dtype": "float32",
        "difference_step": 0.03,
        "mode_forward_batch": 2,
        "grad_clip": 1.0e9,
        "mass_floor": 1.0e-12,
        "rayleigh_scale": 80.0,
        "xmin": -np.pi,
        "xmax": np.pi,
        "farfield_radius": 2.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_stencil_operator_matches_forward_and_divergence_is_exact() -> None:
    cfg = tiny_config(modes=2)
    rng = np.random.default_rng(2)
    count = 5
    evaluated = cp.asarray(
        rng.standard_normal((cfg.modes, len(STENCIL_OFFSETS) * count, 3)),
        dtype=cp.float64,
    )
    _, velocity, gradient = fields_from_stencil(cfg, evaluated, count)
    operator = stencil_adjoint_matrix(cfg)
    samples = evaluated.reshape(cfg.modes, len(STENCIL_OFFSETS), count, 3)
    samples = samples.transpose(0, 2, 1, 3).reshape(cfg.modes, count, -1)
    direct = samples @ operator.T
    expected = cp.concatenate(
        (velocity, gradient.reshape(cfg.modes, count, 9)), axis=2
    )
    np.testing.assert_allclose(
        cp.asnumpy(direct), cp.asnumpy(expected), rtol=0.0, atol=2.0e-12
    )
    divergence = gradient[:, :, 0, 0] + gradient[:, :, 1, 1] + gradient[:, :, 2, 2]
    assert float(cp.max(cp.abs(divergence)).get()) < 2.0e-12


def test_manual_mlp_reverse_matches_central_difference() -> None:
    cfg = tiny_config(modes=1)
    model = HardcodedFanModel(cfg, initialize_arrays(cfg))
    rng = np.random.default_rng(4)
    inputs = cp.asarray(
        rng.standard_normal((11, 6 * cfg.frequencies + 6)), dtype=cp.float32
    )
    envelope = cp.asarray(rng.uniform(0.2, 0.9, 11), dtype=cp.float32)
    adjoint = cp.asarray(rng.standard_normal((11, 3)), dtype=cp.float32)
    field, cache = model.forward_single(0, inputs, envelope, cache=True)
    assert cache is not None
    gradients = model.backward_single(0, cache, adjoint)
    analytic = float(gradients[0][2, 3].get())
    original = float(model.weights[0][0, 2, 3].get())
    epsilon = 2.0e-4

    def objective(value: float) -> float:
        model.weights[0][0, 2, 3] = value
        model.shadow_weights[0][0, 2, 3] = value
        output, _ = model.forward_single(0, inputs, envelope, cache=False)
        return float(cp.sum(output * adjoint).get())

    finite = (objective(original + epsilon) - objective(original - epsilon)) / (
        2.0 * epsilon
    )
    objective(original)
    np.testing.assert_allclose(analytic, finite, rtol=3.0e-3, atol=3.0e-3)


def test_last_causal_mode_adjoint_matches_finite_difference() -> None:
    cfg = tiny_config()
    rng = np.random.default_rng(9)
    count = 7
    raw = {
        "potential": cp.asarray(rng.standard_normal((3 * count, cfg.modes))),
        "velocity": cp.asarray(rng.standard_normal((3 * count, cfg.modes))),
        "gradient": cp.asarray(rng.standard_normal((9 * count, cfg.modes))),
    }
    base = peel_and_adjoint(
        cfg, raw, count, need_adjoint=True, need_coefficients=False
    )
    last_mode = int(base["mode_order"][-1])
    dv = cp.asarray(rng.standard_normal(3 * count))
    dg = cp.asarray(rng.standard_normal(9 * count))
    analytic = float(
        cp.dot(base["adjoint_velocity"][:, last_mode], dv).get()
        + cp.dot(base["adjoint_gradient"][:, last_mode], dg).get()
    )
    epsilon = 1.0e-6

    def final_mode_loss(sign: float) -> float:
        perturbed = {key: value.copy() for key, value in raw.items()}
        perturbed["velocity"][:, last_mode] += sign * epsilon * dv
        perturbed["gradient"][:, last_mode] += sign * epsilon * dg
        result = peel_and_adjoint(
            cfg, perturbed, count, need_adjoint=False, need_coefficients=False
        )
        assert int(result["mode_order"][-1]) == last_mode
        return float(result["mode_energies"][-1] / cfg.rayleigh_scale)

    finite = (final_mode_loss(1.0) - final_mode_loss(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite, rtol=2.0e-6, atol=2.0e-7)


def test_four_blade_geometry_is_quarter_turn_invariant() -> None:
    cfg = tiny_config(
        blade_count=4,
        tip_radius=1.0,
        hub_radius=0.27,
        hub_length=0.58,
        beta_root_deg=54.0,
        beta_tip_min_deg=8.0,
        beta_tip_max_deg=38.0,
        chord_root=0.49,
        chord_tip_min=0.16,
        chord_tip_max=0.40,
        thickness_min=0.08,
        thickness_max=0.20,
        camber=0.045,
        camber_position=0.42,
        stacking_fraction=0.36,
    )
    rng = np.random.default_rng(12)
    points = rng.uniform((-0.7, -1.1, -1.1), (0.7, 1.1, 1.1), (100, 3))
    quarter_turn = points.copy()
    quarter_turn[:, 1] = -points[:, 2]
    quarter_turn[:, 2] = points[:, 1]
    parameters = np.asarray((0.37, 0.61, 0.44))
    np.testing.assert_allclose(
        rotor_distance(cfg, points, parameters),
        rotor_distance(cfg, quarter_turn, parameters),
        rtol=0.0,
        atol=2.0e-15,
    )
