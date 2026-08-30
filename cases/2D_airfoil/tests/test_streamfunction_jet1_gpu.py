"""Regression tests for the synchronized-free Jet1 Lq training path."""

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
            pytest.skip("Jet1 tests require a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - depends on test host
        pytest.skip(f"Jet1 tests require working CUDA: {error}")
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
        basis_objective="streamfunction_dirichlet_rayleigh",
        spatial_derivative_order=1,
        causal_sorting="geometry_rayleigh",
        reorthogonalization_passes=2,
        freeze_prefix_modes=0,
        mode_forward_batch=modes,
        mode_backward_batch=modes,
        mass_floor=1.0e-12,
        learning_rate=2.0e-4,
        grad_clip=5.0,
    )


def _arrays(cfg: SimpleNamespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260827)
    arrays: dict[str, np.ndarray] = {}
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = cfg.input_features
        for layer in range(cfg.hidden_layers):
            weight = f"{prefix}.l{layer}.weight"
            bias = f"{prefix}.l{layer}.bias"
            arrays[weight] = rng.normal(0.0, 0.08, (cfg.hidden, fan_in)).astype(
                np.float32
            )
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


def test_jet1_network_value_and_gradient_match_jet3_exactly() -> None:
    cp, backend = _backend()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    points = np.asarray(
        [[0.2, -3.1], [1.7, 0.8], [4.4, -1.2], [8.6, 2.3]],
        dtype=np.float32,
    )
    jet1 = list(model.forward_stream_jet1(cfg, points, 0.37, 4.25, False))
    jet3 = list(model.forward_stream(cfg, points, 0.37, 4.25, False))

    for first, third in zip(jet1, jet3):
        np.testing.assert_allclose(
            cp.asnumpy(first[1].value),
            cp.asnumpy(third[1].value),
            rtol=1.0e-4,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            cp.asnumpy(first[1].grad),
            cp.asnumpy(third[1].grad),
            rtol=1.0e-4,
            atol=2.0e-7,
        )


def test_jet1_reverse_matches_jet3_for_value_gradient_functional() -> None:
    cp, backend = _backend()
    cfg = _config()
    arrays = _arrays(cfg)
    model1 = backend.HardcodedModel(cfg, arrays)
    model3 = backend.HardcodedModel(
        cfg, {name: value.copy() for name, value in arrays.items()}
    )
    points = np.asarray(
        [[0.2, -3.1], [1.7, 0.8], [4.4, -1.2], [8.6, 2.3]],
        dtype=np.float32,
    )
    mode1, field1, cache1 = next(
        model1.forward_stream_jet1(cfg, points, 0.37, 4.25, True)
    )
    mode3, field3, cache3 = next(
        model3.forward_stream(cfg, points, 0.37, 4.25, True)
    )
    rng = np.random.default_rng(7)
    value = cp.asarray(rng.normal(size=field1.value.shape).astype(np.float32))
    gradient = cp.asarray(rng.normal(size=field1.grad.shape).astype(np.float32))
    adjoint1 = backend.Jet1(value, gradient)
    adjoint3 = backend.Jet3(
        value,
        gradient,
        cp.zeros((*value.shape, 3), dtype=backend.F32),
        cp.zeros((*value.shape, 4), dtype=backend.F32),
    )
    model1.backward_block_jet1(cfg, [mode1], [cache1], [adjoint1])
    model3.backward_block(cfg, [mode3], [cache3], [adjoint3])

    for actual, expected in zip(
        model1.modes[0].weights + model1.modes[0].biases,
        model3.modes[0].weights + model3.modes[0].biases,
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual), cp.asnumpy(expected), rtol=2.0e-6, atol=1.0e-8
        )


def test_lq_step_uses_gpu_sort_batched_optimizer_and_one_transfer() -> None:
    cp, backend = _backend()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    points = np.asarray(
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
    diagnostic = backend.peel_step(
        cfg, model, points, 0.37, 4.25, training=True
    )
    assert diagnostic["spatial_jet_order_used"] == 1
    assert diagnostic["causal_sort_device"] == "gpu"
    assert diagnostic["optimizer_device"] == "gpu_batched_all_modes"
    assert diagnostic["diagnostic_host_transfers_per_step"] == 1
    assert diagnostic["optimizer_host_transfers_per_step"] == 0
    assert diagnostic["input_host_to_device_transfers_per_step"] == 1
    assert diagnostic["peeled_rank"] == cfg.modes
    assert sorted(diagnostic["causal_order"]) == list(range(cfg.modes))

    resident = backend.peel_step(
        cfg,
        backend.HardcodedModel(cfg, _arrays(cfg)),
        cp.asarray(points),
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

    source = inspect.getsource(
        backend._streamfunction_dirichlet_rayleigh_peel_step
    )
    synchronized_free_sources = (
        source,
        inspect.getsource(backend.HardcodedModel.forward_stream_jet1),
        inspect.getsource(backend.HardcodedModel.backward_gradients_block_jet1),
        inspect.getsource(backend.HardcodedModel.adam_step_jet1_device),
    )
    assert all(".get(" not in item for item in synchronized_free_sources)
    assert source.count("cp.asnumpy(") == 1
    assert "cp.argsort" in source
    assert "adam_step_jet1_device" in source
    forward_source = inspect.getsource(backend.HardcodedModel.forward_stream_jet1)
    assert forward_source.count("xy_jets1(points") == 1
    assert forward_source.count("envelope_jet1(") == 1
    assert forward_source.count("encoding_jet1(") == 1


def test_batched_checkpoint_export_is_bitwise_exact() -> None:
    _cp, backend = _backend()
    cfg = _config()
    expected = _arrays(cfg)
    model = backend.HardcodedModel(
        cfg, {name: value.copy() for name, value in expected.items()}
    )
    actual = model.arrays()
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        np.testing.assert_array_equal(actual[name], reference)
