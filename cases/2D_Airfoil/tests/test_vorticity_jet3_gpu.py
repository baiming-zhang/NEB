"""Regression tests for the synchronized-free pure-vorticity Jet3 path."""

from __future__ import annotations

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def _backend():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Jet3 GPU tests require a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - depends on test host
        pytest.skip(f"Jet3 GPU tests require working CUDA: {error}")
    module = importlib.import_module(
        "eigenfluid.gen.airfoil2d.hardcoded_backend"
    )
    return cp, module


def _config(modes: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        modes=modes,
        hidden_layers=2,
        hidden=5,
        input_features=6,
        mlp_compute_dtype="float32",
        fp32_boundary_layers=False,
        canonical_siren=True,
        siren_first_omega0=8.0,
        siren_hidden_omega0=30.0,
        input_encoding="normalized_raw_xy_plus_geometry",
        xmin=0.0,
        xmax=10.0,
        ymin=-4.0,
        ymax=4.0,
        chord=2.0,
        quarter_chord_x=3.0,
        quarter_chord_y=0.0,
        thickness_min=0.08,
        thickness_max=0.18,
        alpha_min_deg=-12.0,
        alpha_max_deg=12.0,
        geometry_mode="cinf_ellipse",
        envelope_mode="cinf_farfield_freestream",
        basis_objective="pure_vorticity_rayleigh",
        spatial_derivative_order=3,
        sampling_mode="uniform_random",
        causal_sorting="geometry_rayleigh",
        reorthogonalization_passes=2,
        freeze_prefix_modes=0,
        mode_forward_batch=modes,
        mode_backward_batch=modes,
        mass_floor=1.0e-20,
        learning_rate=2.0e-5,
        grad_clip=100.0,
    )


def _arrays(cfg: SimpleNamespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260830)
    arrays: dict[str, np.ndarray] = {}
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameters: list[str] = []
        fan_in = cfg.input_features
        for layer in range(cfg.hidden_layers):
            weight = f"{prefix}.l{layer}.weight"
            bias = f"{prefix}.l{layer}.bias"
            arrays[weight] = rng.normal(0.0, 0.06, (cfg.hidden, fan_in)).astype(
                np.float32
            )
            arrays[bias] = rng.normal(0.0, 0.03, cfg.hidden).astype(np.float32)
            parameters.extend((weight, bias))
            fan_in = cfg.hidden
        weight = f"{prefix}.output.weight"
        bias = f"{prefix}.output.bias"
        arrays[weight] = rng.normal(0.0, 0.04, (1, fan_in)).astype(np.float32)
        arrays[bias] = rng.normal(0.0, 0.02, 1).astype(np.float32)
        parameters.extend((weight, bias))
        for index, name in enumerate(parameters):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def _points() -> np.ndarray:
    rng = np.random.default_rng(19)
    points = np.column_stack(
        (rng.uniform(0.2, 9.8, 40), rng.uniform(-3.7, 3.7, 40))
    ).astype(np.float32)
    return points


def _legacy_reference_step(cp, backend, cfg, model, points):
    """Small-K transcription of the former host-synchronized Jet3 step."""
    n, modes = len(points), cfg.modes
    area = np.float32(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - backend.airfoil_area(cfg, 0.37)
    )
    weight = cp.asarray(area / np.float32(n), dtype=backend.F32)
    raw_rayleigh = []
    for _mode, field, _cache in model.forward_stream(
        cfg, points, 0.37, 4.25, False
    ):
        omega, gradient = backend.vorticity_and_gradient(field)
        omega = omega[:, 0].astype(backend.F32, copy=False)
        gradient = gradient[:, 0].reshape(-1).astype(backend.F32, copy=False)
        mass = weight * cp.dot(omega, omega)
        stiffness = weight * cp.dot(gradient, gradient)
        raw_rayleigh.append(float((stiffness / mass).get()))
    order = np.argsort(np.asarray(raw_rayleigh, dtype=np.float32), kind="stable")

    basis_omega = cp.empty((n, modes), dtype=backend.F32)
    basis_gradient = cp.empty((2 * n, modes), dtype=backend.F32)
    rayleighs = []
    projections = []
    orth = []
    backward_modes = []
    backward_caches = []
    backward_adjoints = []
    stream = model.forward_stream(
        cfg, points, 0.37, 4.25, True, mode_indices=order.tolist()
    )
    for causal_rank, (mode, field, cache) in enumerate(stream):
        omega_field, gradient_field = backend.vorticity_and_gradient(field)
        omega = omega_field[:, 0].astype(backend.F32, copy=False)
        gradient = gradient_field[:, 0].reshape(-1).astype(
            backend.F32, copy=False
        )
        residual_omega = omega.copy()
        residual_gradient = gradient.copy()
        coefficients = cp.zeros(causal_rank, dtype=backend.F32)
        if causal_rank:
            previous_omega = basis_omega[:, :causal_rank]
            previous_gradient = basis_gradient[:, :causal_rank]
            for _ in range(2):
                increment = weight * (previous_omega.T @ residual_omega)
                coefficients += increment
                residual_omega -= previous_omega @ increment
                residual_gradient -= previous_gradient @ increment
        else:
            previous_omega = previous_gradient = None
        mass = weight * cp.dot(residual_omega, residual_omega)
        stiffness = weight * cp.dot(residual_gradient, residual_gradient)
        rayleigh = stiffness / mass
        adjoint_omega = (
            -2.0 * stiffness * weight / (mass * mass)
        ) * residual_omega
        adjoint_gradient = (2.0 * weight / mass) * residual_gradient
        if causal_rank:
            for _ in range(2):
                coefficient_adjoint = -(
                    previous_omega.T @ adjoint_omega
                    + previous_gradient.T @ adjoint_gradient
                )
                adjoint_omega += weight * (
                    previous_omega @ coefficient_adjoint
                )
        adjoint = backend.Jet3(
            cp.zeros_like(field.value, dtype=backend.F32),
            cp.zeros_like(field.grad, dtype=backend.F32),
            cp.zeros_like(field.hess, dtype=backend.F32),
            cp.zeros_like(field.third, dtype=backend.F32),
        )
        adjoint.hess[:, 0, 0] = -adjoint_omega
        adjoint.hess[:, 0, 2] = -adjoint_omega
        gradient_adjoint = adjoint_gradient.reshape(n, 2)
        adjoint.third[:, 0, 0] = -gradient_adjoint[:, 0]
        adjoint.third[:, 0, 2] = -gradient_adjoint[:, 0]
        adjoint.third[:, 0, 1] = -gradient_adjoint[:, 1]
        adjoint.third[:, 0, 3] = -gradient_adjoint[:, 1]
        backward_modes.append(mode)
        backward_caches.append(cache)
        backward_adjoints.append(adjoint)

        normalizer = cp.reciprocal(cp.sqrt(mass))
        normalized_omega = residual_omega * normalizer
        normalized_gradient = residual_gradient * normalizer
        basis_omega[:, causal_rank] = normalized_omega
        basis_gradient[:, causal_rank] = normalized_gradient
        normalized_mass = weight * cp.dot(normalized_omega, normalized_omega)
        orth_value = cp.abs(normalized_mass - backend.F32(1.0))
        if causal_rank:
            overlap = weight * (previous_omega.T @ normalized_omega)
            orth_value = cp.maximum(orth_value, cp.max(cp.abs(overlap)))
        orth.append(float(orth_value.get()))
        projections.append(
            float((cp.dot(coefficients, coefficients) / mass).get())
            if causal_rank
            else 0.0
        )
        rayleighs.append(float(rayleigh.get()))
    model.backward_block(
        cfg, backward_modes, backward_caches, backward_adjoints
    )
    return {
        "order": order,
        "raw": np.asarray(raw_rayleigh, dtype=np.float32),
        "rayleigh": np.asarray(rayleighs, dtype=np.float32),
        "projection": np.asarray(projections, dtype=np.float32),
        "orth": np.asarray(orth, dtype=np.float32),
        "basis_omega": cp.asnumpy(basis_omega),
    }


def test_vorticity_gpu_step_matches_legacy_math_and_parameter_update() -> None:
    cp, backend = _backend()
    cfg = _config()
    arrays = _arrays(cfg)
    production = backend.HardcodedModel(
        cfg, {name: value.copy() for name, value in arrays.items()}
    )
    reference = backend.HardcodedModel(
        cfg, {name: value.copy() for name, value in arrays.items()}
    )
    points = _points()

    diagnostic = backend.peel_step(
        cfg, production, points, 0.37, 4.25, training=True
    )
    legacy = _legacy_reference_step(cp, backend, cfg, reference, points)

    np.testing.assert_array_equal(diagnostic["causal_order"], legacy["order"])
    np.testing.assert_allclose(
        diagnostic["raw_mode_energies"], legacy["raw"], rtol=2.0e-5, atol=2.0e-5
    )
    np.testing.assert_allclose(
        diagnostic["mode_energies"], legacy["rayleigh"], rtol=2.0e-5, atol=2.0e-5
    )
    assert diagnostic["mean_projection_ratio"] == pytest.approx(
        float(np.mean(legacy["projection"])), rel=2.0e-5, abs=2.0e-5
    )
    assert diagnostic["orth_error"] == pytest.approx(
        float(np.max(legacy["orth"])), rel=2.0e-5, abs=2.0e-6
    )
    assert np.all(np.isfinite(legacy["basis_omega"]))

    for actual, expected in zip(
        production.master_weight_stacks + production.master_bias_stacks,
        reference.master_weight_stacks + reference.master_bias_stacks,
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual), cp.asnumpy(expected), rtol=5.0e-5, atol=2.0e-7
        )


def test_vorticity_step_audits_minimal_jet3_gpu_backend() -> None:
    cp, backend = _backend()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    diagnostic = backend.peel_step(
        cfg, model, _points(), 0.37, 4.25, training=True
    )
    assert diagnostic["spatial_jet_order_used"] == 3
    assert diagnostic["third_derivative_channels_allocated"] == 4
    assert diagnostic["causal_sort_device"] == "gpu"
    assert diagnostic["optimizer_device"] == "gpu_batched_all_modes"
    assert diagnostic["diagnostic_host_transfers_per_step"] == 1
    assert diagnostic["optimizer_host_transfers_per_step"] == 0
    assert diagnostic["input_host_to_device_transfers_per_step"] == 1
    assert diagnostic["peeled_rank"] == cfg.modes
    assert sorted(diagnostic["causal_order"]) == list(range(cfg.modes))
    assert np.isfinite(diagnostic["orth_error"])

    resident = backend.peel_step(
        cfg,
        backend.HardcodedModel(cfg, _arrays(cfg)),
        cp.asarray(_points()),
        0.37,
        4.25,
        training=True,
    )
    assert resident["sample_pack_device"] == "gpu_resident"
    assert resident["input_host_to_device_transfers_per_step"] == 0
    assert resident["diagnostic_host_transfers_per_step"] == 1
    assert resident["optimizer_host_transfers_per_step"] == 0
    np.testing.assert_array_equal(
        resident["causal_order"], diagnostic["causal_order"]
    )
    np.testing.assert_array_equal(
        resident["mode_energies"], diagnostic["mode_energies"]
    )

    source = inspect.getsource(backend._pure_vorticity_rayleigh_peel_step)
    assert ".get(" not in source
    assert source.count("cp.asnumpy(") == 1
    assert "cp.argsort" in source
    assert "adam_step_jet1_device" in source
    forward_source = inspect.getsource(backend.HardcodedModel.forward_stream)
    assert forward_source.count("xy_jets(points") == 1
    assert forward_source.count("envelope_jet(") == 1
    assert forward_source.count("encoding_jet(") == 1


def test_vorticity_requires_exactly_two_mgs_passes() -> None:
    _cp, backend = _backend()
    cfg = _config()
    cfg.reorthogonalization_passes = 1
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    with pytest.raises(ValueError, match="exactly two MGS passes"):
        backend.peel_step(cfg, model, _points(), 0.37, 4.25, training=False)
