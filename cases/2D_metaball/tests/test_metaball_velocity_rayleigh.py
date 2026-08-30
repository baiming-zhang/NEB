"""Regression tests for strict one-pass metaball velocity-Rayleigh training."""

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

from metaball_runtime import (  # noqa: E402
    backend,
    install_backend,
    metaball_fluid_points,
)


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("velocity-Rayleigh tests require CUDA")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - host dependent
        pytest.skip(f"velocity-Rayleigh tests require working CUDA: {error}")
    return cp


def _config(modes: int):
    path = CASE_ROOT / (
        "configs/"
        "k1024_metaball_connected_siren_velocity_rayleigh_omega128_onepass_2000.json"
    )
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        modes=modes,
        hidden=7,
        hidden_layers=2,
        mode_forward_batch=modes,
        mode_backward_batch=modes,
        points_per_step=128,
        validation_points=128,
    )
    install_backend(config)
    return backend.cfg_namespace(config)


def _arrays(cfg) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260829)
    arrays: dict[str, np.ndarray] = {}
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        names: list[str] = []
        fan_in = cfg.input_features
        for layer in range(cfg.hidden_layers):
            weight = f"{prefix}.l{layer}.weight"
            bias = f"{prefix}.l{layer}.bias"
            bound = 1.0 / fan_in if layer == 0 else np.sqrt(6.0 / fan_in) / 30.0
            arrays[weight] = rng.uniform(
                -bound, bound, (cfg.hidden, fan_in)
            ).astype(np.float32)
            arrays[bias] = rng.uniform(
                -1.0 / np.sqrt(fan_in), 1.0 / np.sqrt(fan_in), cfg.hidden
            ).astype(np.float32)
            names.extend((weight, bias))
            fan_in = cfg.hidden
        weight = f"{prefix}.output.weight"
        bias = f"{prefix}.output.bias"
        bound = np.sqrt(6.0 / fan_in) / 30.0
        arrays[weight] = rng.uniform(-bound, bound, (1, fan_in)).astype(np.float32)
        arrays[bias] = rng.uniform(
            -1.0 / np.sqrt(fan_in), 1.0 / np.sqrt(fan_in), 1
        ).astype(np.float32)
        names.extend((weight, bias))
        for index, name in enumerate(names):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def test_one_mode_velocity_rayleigh_matches_direct_jet3_forms() -> None:
    cp = _cuda()
    cfg = _config(1)
    model = backend.HardcodedModel(cfg, _arrays(cfg))
    points = metaball_fluid_points(cfg, 128, 4000001.0, 0.0, cfg.seed + 17)
    _mode, field, _cache = next(
        model.forward_stream(cfg, points, 4000001.0, 0.0, False)
    )
    gradient = field.grad[:, 0]
    hessian = field.hess[:, 0]
    direct = cp.sum(
        hessian[:, 0] ** 2
        + 2.0 * hessian[:, 1] ** 2
        + hessian[:, 2] ** 2
    ) / cp.sum(gradient * gradient)
    diagnostic = backend.peel_step(
        cfg, model, points, 4000001.0, 0.0, training=False
    )
    np.testing.assert_allclose(
        diagnostic["velocity_rayleigh"], float(direct.get()), rtol=2.0e-5
    )
    assert diagnostic["peeling_passes_used"] == 1
    assert diagnostic["peeled_rank"] == 1
    assert diagnostic["rejected_modes"] == 0


def test_collapsed_candidate_is_rejected_before_any_parameter_update() -> None:
    _cuda()
    cfg = _config(2)
    arrays = _arrays(cfg)
    for name in list(arrays):
        if name.startswith("m001"):
            arrays[name] = arrays[name.replace("m001", "m000")].copy()
    model = backend.HardcodedModel(cfg, arrays)
    before = model.arrays()
    points = metaball_fluid_points(cfg, 128, 4000001.0, 0.0, cfg.seed + 19)
    with pytest.raises(FloatingPointError, match="no parameters updated"):
        backend.peel_step(
            cfg, model, points, 4000001.0, 0.0, training=True
        )
    after = model.arrays()
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(after[name], before[name])
