"""Regression tests for the compact Jet2 velocity-Rayleigh path."""

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
            pytest.skip("Jet2 tests require a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - depends on test host
        pytest.skip(f"Jet2 tests require working CUDA: {error}")
    module = importlib.import_module(
        "eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend"
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
        basis_objective="velocity_rayleigh",
        spatial_derivative_order=2,
        causal_sorting="geometry_rayleigh",
        reorthogonalization_passes=2,
        peeling_relative_mass_floor=1.0e-8,
        peeling_overlap_tolerance=1.0e-2,
        freeze_prefix_modes=0,
        mode_forward_batch=modes,
        mode_backward_batch=modes,
        mass_floor=1.0e-12,
        learning_rate=2.0e-4,
        grad_clip=5.0,
    )


def _arrays(cfg: SimpleNamespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260830)
    arrays: dict[str, np.ndarray] = {}
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = cfg.input_features
        for layer in range(cfg.hidden_layers):
            weight = f"{prefix}.l{layer}.weight"
            bias = f"{prefix}.l{layer}.bias"
            arrays[weight] = rng.normal(
                0.0, 0.08, (cfg.hidden, fan_in)
            ).astype(np.float32)
            arrays[bias] = rng.normal(0.0, 0.04, cfg.hidden).astype(np.float32)
            parameter_names.extend((weight, bias))
            fan_in = cfg.hidden
        weight = f"{prefix}.output.weight"
        bias = f"{prefix}.output.bias"
        arrays[weight] = rng.normal(0.0, 0.05, (1, fan_in)).astype(np.float32)
        arrays[bias] = rng.normal(0.0, 0.03, 1).astype(np.float32)
        parameter_names.extend((weight, bias))
        for index, name in enumerate(parameter_names):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def _points() -> np.ndarray:
    return np.asarray(
        [
            [0.2, -3.1],
            [1.7, 0.8],
            [4.4, -1.2],
            [8.6, 2.3],
            [9.2, -2.6],
            [5.7, 3.0],
            [0.8, 2.1],
            [7.3, -0.7],
        ],
        dtype=np.float32,
    )


def test_jet2_value_gradient_hessian_match_jet3_without_third_storage() -> None:
    cp, backend = _backend()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    jet2 = list(model.forward_stream_jet2(cfg, _points(), 0.37, 4.25, False))
    jet3 = list(model.forward_stream(cfg, _points(), 0.37, 4.25, False))

    for second, third in zip(jet2, jet3):
        field2, field3 = second[1], third[1]
        assert field2.hess.shape[-1] == 3
        assert not hasattr(field2, "third")
        for actual, expected in (
            (field2.value, field3.value),
            (field2.grad, field3.grad),
            (field2.hess, field3.hess),
        ):
            np.testing.assert_allclose(
                cp.asnumpy(actual),
                cp.asnumpy(expected),
                rtol=2.0e-4,
                atol=3.0e-6,
            )


def test_jet2_reverse_matches_jet3_for_second_order_functional() -> None:
    cp, backend = _backend()
    cfg = _config()
    arrays = _arrays(cfg)
    model2 = backend.HardcodedModel(cfg, arrays)
    model3 = backend.HardcodedModel(
        cfg, {name: value.copy() for name, value in arrays.items()}
    )
    outputs2 = list(
        model2.forward_stream_jet2(cfg, _points(), 0.37, 4.25, True)
    )
    outputs3 = list(model3.forward_stream(cfg, _points(), 0.37, 4.25, True))
    rng = np.random.default_rng(17)
    adjoints2, adjoints3 = [], []
    for (_, field2, _), (_, field3, _) in zip(outputs2, outputs3):
        value = cp.asarray(rng.normal(size=field2.value.shape).astype(np.float32))
        gradient = cp.asarray(
            rng.normal(size=field2.grad.shape).astype(np.float32)
        )
        hessian = cp.asarray(
            rng.normal(size=field2.hess.shape).astype(np.float32)
        )
        adjoints2.append(backend.Jet2(value, gradient, hessian))
        adjoints3.append(
            backend.Jet3(
                value,
                gradient,
                hessian,
                cp.zeros_like(field3.third, dtype=backend.F32),
            )
        )
    weights2, biases2 = model2.backward_gradients_block_jet2(
        cfg,
        [item[0] for item in outputs2],
        [item[2] for item in outputs2],
        adjoints2,
    )
    weights3, biases3 = model3.backward_gradients_block(
        cfg,
        [item[0] for item in outputs3],
        [item[2] for item in outputs3],
        adjoints3,
    )
    for actual, expected in zip((*weights2, *biases2), (*weights3, *biases3)):
        np.testing.assert_allclose(
            cp.asnumpy(actual),
            cp.asnumpy(expected),
            rtol=3.0e-5,
            atol=2.0e-6,
        )


def test_velocity_step_audits_true_jet2_and_never_calls_jet3() -> None:
    cp, backend = _backend()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    diagnostic = backend.peel_step(
        cfg, model, _points(), 0.37, 4.25, training=True
    )
    assert diagnostic["spatial_jet_order_used"] == 2
    assert diagnostic["third_derivative_channels_allocated"] == 0
    assert diagnostic["causal_sort_device"] == "gpu"
    assert diagnostic["optimizer_device"] == "gpu_batched_all_modes"
    assert diagnostic["diagnostic_host_transfers_per_step"] == 1
    assert diagnostic["optimizer_host_transfers_per_step"] == 0
    assert diagnostic["input_host_to_device_transfers_per_step"] == 1
    assert diagnostic["peeled_rank"] == cfg.modes
    assert diagnostic["peeling_passes_used"] == 2
    assert diagnostic["shared_triangular_transform_across_derivatives"] is True
    assert diagnostic["max_post_peel_overlap"] <= (
        diagnostic["max_post_first_pass_overlap"] + 5.0e-7
    )

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

    source = inspect.getsource(backend._velocity_rayleigh_peel_step)
    assert "forward_stream_jet2" in source
    assert "backward_gradients_block_jet2" in source
    assert "field.third" not in source
    assert "Jet3(" not in source
    assert source.count("cp.asnumpy(") == 1
    assert ".get(" not in source
    assert "physical_valid_device" in source
    assert "adam_step_jet1_device" in source
    assert "for pass_index in range(passes)" in source
    assert source.count("for _ in range(passes)") == 1
    compact_sources = (
        inspect.getsource(backend.HardcodedModel.forward_stream_jet2),
        inspect.getsource(backend.HardcodedModel.backward_gradients_block_jet2),
        inspect.getsource(backend.HardcodedModel.adam_step_jet1_device),
    )
    assert all(".get(" not in item for item in compact_sources)
    assert all("cp.asnumpy(" not in item for item in compact_sources)
    assert all(".third" not in item for item in compact_sources[:2])


def test_velocity_device_gate_rejects_duplicate_without_parameter_update() -> None:
    _cp, backend = _backend()
    cfg = _config(modes=2)
    arrays = _arrays(cfg)
    for name, value in list(arrays.items()):
        if name.startswith("m000"):
            arrays["m001" + name[4:]] = value.copy()
    model = backend.HardcodedModel(cfg, arrays)
    before = model.arrays()

    with pytest.raises(FloatingPointError, match="rejected after the two-pass peel"):
        backend.peel_step(cfg, model, _points(), 0.37, 4.25, training=True)

    after = model.arrays()
    assert after.keys() == before.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
