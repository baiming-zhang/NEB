"""Metaball-specific regression tests for the value/gradient-only Jet1 path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


CASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = CASE_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from metaball_runtime import backend, install_backend  # noqa: E402


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Jet1 tests require a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - host dependent
        pytest.skip(f"Jet1 tests require working CUDA: {error}")
    return cp


def _config(modes: int = 3):
    path = (
        CASE_ROOT
        / "configs/k1024_metaball_connected_siren_original_jet1_2000.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        modes=modes,
        hidden=7,
        hidden_layers=2,
        mode_forward_batch=modes,
        mode_backward_batch=modes,
        points_per_step=64,
        validation_points=64,
    )
    install_backend(config)
    return backend.cfg_namespace(config)


def _arrays(cfg) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260828)
    arrays: dict[str, np.ndarray] = {}
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        names: list[str] = []
        fan_in = cfg.input_features
        for layer in range(cfg.hidden_layers):
            weight = f"{prefix}.l{layer}.weight"
            bias = f"{prefix}.l{layer}.bias"
            arrays[weight] = rng.normal(0.0, 0.04, (cfg.hidden, fan_in)).astype(
                np.float32
            )
            arrays[bias] = rng.normal(0.0, 0.02, cfg.hidden).astype(np.float32)
            names.extend((weight, bias))
            fan_in = cfg.hidden
        weight = f"{prefix}.output.weight"
        bias = f"{prefix}.output.bias"
        arrays[weight] = rng.normal(0.0, 0.03, (1, fan_in)).astype(np.float32)
        arrays[bias] = rng.normal(0.0, 0.01, 1).astype(np.float32)
        names.extend((weight, bias))
        for index, name in enumerate(names):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def test_metaball_jet1_matches_jet3_value_and_gradient() -> None:
    cp = _cuda()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    points = np.asarray(
        [[0.08, 0.12], [0.17, 0.82], [0.46, 0.17], [0.88, 0.73]],
        dtype=np.float32,
    )
    first = list(model.forward_stream_jet1(cfg, points, 4000001.0, 0.0, False))
    third = list(model.forward_stream(cfg, points, 4000001.0, 0.0, False))
    for jet1, jet3 in zip(first, third):
        assert isinstance(jet1[1], backend.Jet1)
        assert not hasattr(jet1[1], "hess")
        assert not hasattr(jet1[1], "third")
        np.testing.assert_allclose(
            cp.asnumpy(jet1[1].value),
            cp.asnumpy(jet3[1].value),
            rtol=2.0e-5,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            cp.asnumpy(jet1[1].grad),
            cp.asnumpy(jet3[1].grad),
            rtol=3.0e-5,
            atol=4.0e-7,
        )


def test_metaball_rayleigh_step_reports_true_jet1() -> None:
    _cuda()
    cfg = _config()
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    rng = np.random.default_rng(33)
    points = rng.uniform(0.02, 0.98, (96, 2)).astype(np.float32)
    diagnostic = backend.peel_step(
        cfg, model, points, 4000001.0, 0.0, training=False
    )
    assert diagnostic["spatial_jet_order_used"] == 1
    assert diagnostic["optimizer_device"] == "gpu_batched_all_modes"
    assert diagnostic["input_host_to_device_transfers_per_step"] == 1
    assert diagnostic["peeled_rank"] == cfg.modes
