"""Numerical contracts for the analytic blocked 3D fan backend."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.fan3d.hardcoded_backend import (  # noqa: E402
    initialize_arrays,
    peel_and_adjoint as reference_peel,
)
from eigenfluid.gen.fan3d.optimized_backend import (  # noqa: E402
    OptimizedFanModel,
    _envelope_product,
    _envelope_product_reverse,
    _network_potential_jet,
    _network_potential_jet_reverse,
    boundary_envelope_jet,
    c4_potential_jet,
    c4_potential_jet_reverse,
    moving_wall_lifting_fields,
    peel_and_adjoint as blocked_peel,
    prepared_analytic_inputs,
    raw_fields,
)


def tiny_config(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "seed": 17,
        "modes": 3,
        "hidden": 7,
        "hidden_layers": 2,
        "frequencies": 2,
        "mlp_compute_dtype": "float32",
        "mode_forward_batch": 3,
        "mode_backward_batch": 2,
        "sss_block_size": 2,
        "grad_clip": 1.0e9,
        "mass_floor": 1.0e-12,
        "rayleigh_scale": 80.0,
        "xmin": -np.pi,
        "xmax": np.pi,
        "farfield_radius": 2.0,
        "tip_radius": 1.0,
        "hub_radius": 0.27,
        "hub_length": 0.58,
        "blade_count": 4,
        "beta_root_deg": 54.0,
        "beta_tip_min_deg": 8.0,
        "beta_tip_max_deg": 38.0,
        "chord_root": 0.49,
        "chord_tip_min": 0.16,
        "chord_tip_max": 0.40,
        "thickness_min": 0.08,
        "thickness_max": 0.20,
        "camber": 0.045,
        "camber_position": 0.42,
        "stacking_fraction": 0.36,
        "inflow_velocity": 1.0,
        "rotation_rate": 2.0943951023931953,
        "moving_wall_lifting": False,
        "moving_lift_axial_core": 0.70,
        "moving_lift_axial_outer": 1.55,
        "moving_lift_radial_core": 1.05,
        "moving_lift_radial_outer": 1.75,
        "c4_equivariant_network": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_blocked_sss_matches_scalar_reference() -> None:
    cfg = tiny_config(modes=8, sss_block_size=3)
    rng = np.random.default_rng(42)
    point_count = 12
    host = {
        "potential": rng.normal(size=(3 * point_count, cfg.modes)),
        "velocity": rng.normal(size=(3 * point_count, cfg.modes)),
        "gradient": rng.normal(size=(9 * point_count, cfg.modes)),
    }
    reference = reference_peel(
        cfg,
        {key: cp.asarray(value) for key, value in host.items()},
        point_count,
        need_adjoint=True,
        need_coefficients=True,
    )
    blocked = blocked_peel(
        cfg,
        {key: cp.asarray(value) for key, value in host.items()},
        point_count,
        need_adjoint=True,
        need_coefficients=True,
        diagnostics=True,
    )
    np.testing.assert_array_equal(reference["mode_order"], blocked["mode_order"])
    np.testing.assert_allclose(
        reference["mode_energies"], blocked["mode_energies"], rtol=0.0, atol=1e-13
    )
    for key in ("q_velocity", "adjoint_velocity", "adjoint_gradient", "coefficient_matrix"):
        np.testing.assert_allclose(
            cp.asnumpy(reference[key]), cp.asnumpy(blocked[key]), rtol=0.0, atol=1e-13
        )


def test_analytic_jet_reverse_matches_directional_difference() -> None:
    cfg = tiny_config(modes=2, hidden=6)
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    parameters = np.asarray((0.4, 0.6, 0.5), dtype=np.float32)
    points = np.asarray(
        ((0.8, 1.25, 0.2), (1.2, -1.1, 0.55), (-1.0, 0.35, 1.35)),
        dtype=np.float32,
    )
    input_jet, envelope = prepared_analytic_inputs(cfg, points, parameters)
    network, cache = model.forward_jet_block(input_jet, 0, 2, cache=True)
    assert cache is not None
    rng = np.random.default_rng(123)
    output_adjoint = cp.asarray(rng.normal(size=network.shape), dtype=cp.float32)
    network_adjoint = _envelope_product_reverse(network, envelope, output_adjoint)
    gradients = model.backward_jet_block(0, 2, cache, network_adjoint)
    directions = [
        cp.asarray(rng.normal(size=gradient.shape), dtype=cp.float32)
        for gradient in gradients
    ]
    analytic = sum(
        float(cp.sum(gradient * direction).get())
        for gradient, direction in zip(gradients, directions)
    )
    original_weights = [value.copy() for value in model.weights]
    original_biases = [value.copy() for value in model.biases]

    def objective(sign: float, epsilon: float) -> float:
        index = 0
        for layer in range(len(model.weights)):
            model.weights[layer][...] = (
                original_weights[layer] + sign * epsilon * directions[index]
            )
            index += 1
            model.biases[layer][...] = (
                original_biases[layer] + sign * epsilon * directions[index]
            )
            index += 1
            model.shadow_weights[layer][...] = model.weights[layer]
            model.shadow_biases[layer][...] = model.biases[layer]
        output, _ = model.forward_jet_block(input_jet, 0, 2, cache=False)
        potential = _envelope_product(output, envelope)
        return float(cp.sum(potential * output_adjoint).get())

    epsilon = 1.0e-3
    numeric = (objective(1.0, epsilon) - objective(-1.0, epsilon)) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(analytic, numeric, rtol=5.0e-4, atol=5.0e-4)


def test_boundary_envelope_jet_is_finite_and_symmetric() -> None:
    cfg = tiny_config()
    points = cp.asarray(
        ((0.8, 1.25, 0.2), (1.2, -1.1, 0.55), (-1.0, 0.35, 1.35)),
        dtype=cp.float32,
    )
    parameters = cp.asarray((0.4, 0.6, 0.5), dtype=cp.float32)
    jet = boundary_envelope_jet(cfg, points, parameters)
    assert bool(cp.all(cp.isfinite(jet.value)).get())
    assert bool(cp.all(cp.isfinite(jet.gradient)).get())
    assert bool(cp.all(cp.isfinite(jet.hessian)).get())
    assert float(cp.max(cp.abs(jet.hessian - jet.hessian.swapaxes(1, 2))).get()) < 1e-6


def test_c4_potential_reverse_matches_directional_difference() -> None:
    rng = np.random.default_rng(83)
    points = cp.asarray(
        ((0.2, 0.7, 0.3), (-0.5, -0.8, 0.45), (1.1, 0.4, -1.2)),
        dtype=cp.float32,
    )
    scalar = cp.asarray(rng.normal(size=(2, 3, 3, 13)), dtype=cp.float32)
    direction = cp.asarray(rng.normal(size=scalar.shape), dtype=cp.float32)
    output_adjoint = cp.asarray(
        rng.normal(size=(2, 3, 3, 13)), dtype=cp.float32
    )
    _transformed = c4_potential_jet(scalar, points)
    analytic_adjoint = c4_potential_jet_reverse(output_adjoint, points)
    analytic = float(cp.sum(analytic_adjoint * direction).get())
    epsilon = 2.0e-3
    upper = float(
        cp.sum(c4_potential_jet(scalar + epsilon * direction, points) * output_adjoint).get()
    )
    lower = float(
        cp.sum(c4_potential_jet(scalar - epsilon * direction, points) * output_adjoint).get()
    )
    numeric = (upper - lower) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, rtol=2.0e-4, atol=2.0e-3)


def test_c4_end_to_end_analytic_reverse_matches_directional_difference() -> None:
    cfg = tiny_config(
        modes=2,
        hidden=6,
        c4_equivariant_network=True,
        grad_clip=1.0e9,
    )
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    parameters = np.asarray((0.4, 0.6, 0.5), dtype=np.float32)
    points = np.asarray(
        ((0.8, 1.25, 0.2), (1.2, -1.1, 0.55), (-1.0, 0.35, 1.35)),
        dtype=np.float32,
    )
    point_device = cp.asarray(points)
    input_jet, envelope = prepared_analytic_inputs(cfg, points, parameters)
    network, cache = model.forward_jet_block(input_jet, 0, 2, cache=True)
    assert cache is not None
    equivariant = _network_potential_jet(cfg, network, point_device)
    rng = np.random.default_rng(901)
    output_adjoint = cp.asarray(
        rng.normal(size=equivariant.shape), dtype=cp.float32
    )
    potential_adjoint = _envelope_product_reverse(
        equivariant, envelope, output_adjoint
    )
    network_adjoint = _network_potential_jet_reverse(
        cfg, potential_adjoint, point_device
    )
    gradients = model.backward_jet_block(0, 2, cache, network_adjoint)
    directions = [
        cp.asarray(rng.normal(size=gradient.shape), dtype=cp.float32)
        for gradient in gradients
    ]
    analytic = sum(
        float(cp.sum(gradient * direction).get())
        for gradient, direction in zip(gradients, directions)
    )
    original_weights = [value.copy() for value in model.weights]
    original_biases = [value.copy() for value in model.biases]

    def objective(sign: float, epsilon: float) -> float:
        index = 0
        for layer in range(len(model.weights)):
            model.weights[layer][...] = (
                original_weights[layer] + sign * epsilon * directions[index]
            )
            index += 1
            model.biases[layer][...] = (
                original_biases[layer] + sign * epsilon * directions[index]
            )
            index += 1
            model.shadow_weights[layer][...] = model.weights[layer]
            model.shadow_biases[layer][...] = model.biases[layer]
        output, _ = model.forward_jet_block(input_jet, 0, 2, cache=False)
        potential = _network_potential_jet(cfg, output, point_device)
        potential = _envelope_product(potential, envelope)
        return float(cp.sum(potential * output_adjoint).get())

    epsilon = 1.0e-3
    numeric = (objective(1.0, epsilon) - objective(-1.0, epsilon)) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(analytic, numeric, rtol=8.0e-4, atol=8.0e-4)


def test_c4_raw_velocity_is_architecturally_equivariant() -> None:
    cfg = tiny_config(modes=4, hidden=8, c4_equivariant_network=True)
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    parameters = np.asarray((0.35, 0.6, 0.45), dtype=np.float32)
    points = np.asarray(
        ((0.9, 1.25, 0.2), (1.3, -1.1, 0.5), (-1.0, 0.4, 1.3)),
        dtype=np.float32,
    )
    rotation = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )
    rotated_points = points @ rotation.T
    velocity = cp.asnumpy(raw_fields(cfg, model, points, parameters)["velocity"])
    rotated_velocity = cp.asnumpy(
        raw_fields(cfg, model, rotated_points, parameters)["velocity"]
    )
    velocity = velocity.reshape(3, len(points), cfg.modes).transpose(1, 2, 0)
    rotated_velocity = rotated_velocity.reshape(
        3, len(points), cfg.modes
    ).transpose(1, 2, 0)
    expected = np.einsum("ij,nmj->nmi", rotation, velocity)
    np.testing.assert_allclose(rotated_velocity, expected, rtol=2.0e-4, atol=2.0e-4)


def test_c4_network_is_finite_on_fluid_axis() -> None:
    cfg = tiny_config(modes=4, hidden=8, c4_equivariant_network=True)
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    # These axial points lie outside the finite hub and therefore belong to
    # the fluid domain; a cylindrical 1/r representation would fail here.
    points = np.asarray(
        ((-2.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.6, 1.0e-7, -1.0e-7)),
        dtype=np.float32,
    )
    fields = raw_fields(cfg, model, points, parameters)
    for key in ("potential", "velocity", "gradient"):
        assert bool(cp.all(cp.isfinite(fields[key])).get())


def test_moving_wall_lifting_is_exact_and_solenoidal() -> None:
    cfg = tiny_config(moving_wall_lifting=True)
    points = np.asarray(
        (
            (0.0, 0.35, 0.0),
            (0.2, 0.4, 0.3),
            (-np.pi, 0.2, 0.1),
            (0.0, 1.9, 0.0),
        ),
        dtype=np.float32,
    )
    fields = moving_wall_lifting_fields(cfg, points)
    velocity = cp.asnumpy(fields["velocity"])
    gradient = cp.asnumpy(fields["gradient"])
    omega = cfg.rotation_rate
    expected = np.asarray(
        (
            (0.0, 0.0, omega * 0.35),
            (0.0, -omega * 0.3, omega * 0.4),
            (cfg.inflow_velocity, 0.0, 0.0),
            (cfg.inflow_velocity, 0.0, 0.0),
        )
    )
    np.testing.assert_allclose(velocity, expected, rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(
        np.trace(gradient, axis1=1, axis2=2), 0.0, atol=2.0e-6
    )
