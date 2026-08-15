"""Regression tests for the continuous pure-vorticity training path.

These tests intentionally never use a finite-difference oracle.  Spatial
derivatives are compared with closed-form derivatives, and the handwritten
parameter reverse pass is checked with complex-step differentiation.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import tokenize
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


CASE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = (
    CASE_ROOT
    / "src"
    / "eigenfluid_sss"
    / "basisnet"
    / "airfoil2d"
    / "hardcoded_backend.py"
)
TRAINER_PATH = CASE_ROOT / "scripts" / "train_hardcoded.py"
PURE_CONFIG_PATH = (
    CASE_ROOT
    / "configs"
    / "k1024_cinf_sobol_5000_fp16_pure_vorticity_rayleigh.json"
)


def _cuda_backend():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("continuous Jet3 tests require a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - depends on test host
        pytest.skip(f"continuous Jet3 tests require working CUDA: {error}")
    backend = importlib.import_module(
        "eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend"
    )
    return cp, backend


def _numpy(cp, value) -> np.ndarray:
    return cp.asnumpy(value)


def test_compact_jet3_polynomial_derivatives_are_exact() -> None:
    cp, backend = _cuda_backend()
    points = np.asarray(
        [[0.2, -0.3], [-0.7, 0.4], [1.1, -0.8]], dtype=np.float32
    )
    x, y = backend.xy_jets(points, backend.F32)
    field = x.pow(4) + y.pow(4)

    px, py = points[:, 0:1], points[:, 1:2]
    expected_value = px**4 + py**4
    expected_grad = np.stack((4.0 * px**3, 4.0 * py**3), axis=-1)
    expected_hess = np.stack(
        (12.0 * px**2, np.zeros_like(px), 12.0 * py**2), axis=-1
    )
    expected_third = np.stack(
        (24.0 * px, np.zeros_like(px), np.zeros_like(px), 24.0 * py),
        axis=-1,
    )

    # Compact symmetric layout: [xx, xy, yy] and [xxx, xxy, xyy, yyy].
    assert field.hess.shape[-1] == 3
    assert field.third.shape[-1] == 4
    np.testing.assert_allclose(_numpy(cp, field.value), expected_value, atol=2e-6)
    np.testing.assert_allclose(_numpy(cp, field.grad), expected_grad, atol=2e-6)
    np.testing.assert_allclose(_numpy(cp, field.hess), expected_hess, atol=3e-6)
    np.testing.assert_allclose(_numpy(cp, field.third), expected_third, atol=4e-6)


def test_compact_jet3_sine_derivatives_are_exact() -> None:
    cp, backend = _cuda_backend()
    points = np.asarray(
        [[-0.8, 0.1], [-0.1, -0.4], [0.6, 0.9]], dtype=np.float32
    )
    ax, ay, offset = 1.7, -0.6, 0.23
    x, y = backend.xy_jets(points, backend.F32)
    phase = ax * x + ay * y + offset
    field = backend.sine_forward(phase)

    scalar_phase = ax * points[:, 0:1] + ay * points[:, 1:2] + offset
    sine, cosine = np.sin(scalar_phase), np.cos(scalar_phase)
    expected_grad = cosine[..., None] * np.asarray([ax, ay])
    expected_hess = -sine[..., None] * np.asarray(
        [ax * ax, ax * ay, ay * ay]
    )
    expected_third = -cosine[..., None] * np.asarray(
        [ax**3, ax * ax * ay, ax * ay * ay, ay**3]
    )

    np.testing.assert_allclose(_numpy(cp, field.value), sine, atol=2e-6)
    np.testing.assert_allclose(_numpy(cp, field.grad), expected_grad, atol=2e-6)
    np.testing.assert_allclose(_numpy(cp, field.hess), expected_hess, atol=2e-6)
    np.testing.assert_allclose(_numpy(cp, field.third), expected_third, atol=3e-6)


def test_positional_encoding_channels_carry_analytic_third_derivatives() -> None:
    cp, backend = _cuda_backend()
    points = np.asarray([[-0.8, 0.1], [0.6, -0.3]], dtype=np.float32)
    cfg = SimpleNamespace(
        xmin=-1.0,
        xmax=3.0,
        ymin=-2.0,
        ymax=2.0,
        frequencies=2,
        alpha_min_deg=-12.0,
        alpha_max_deg=12.0,
        quarter_chord_x=0.0,
        quarter_chord_y=0.0,
        chord=2.0,
    )
    x, y = backend.xy_jets(points, backend.F32)
    encoded = backend.encoding_jet(cfg, x, y, beta=0.4, alpha_deg=3.0)

    # Channel one is sin(2*x_normalized).  Its third physical x derivative
    # is -cos(2*x_normalized) * (4/(xmax-xmin))^3, with no stencil step.
    normalized_x = 2.0 * (points[:, 0:1] - cfg.xmin) / (
        cfg.xmax - cfg.xmin
    ) - 1.0
    phase = 2.0 * normalized_x
    physical_frequency = 4.0 / (cfg.xmax - cfg.xmin)
    expected_xxx = -np.cos(phase) * physical_frequency**3
    expected = np.zeros((len(points), 4), dtype=np.float32)
    expected[:, 0:1] = expected_xxx
    np.testing.assert_allclose(
        _numpy(cp, encoded.third[:, 1]), expected, rtol=2e-6, atol=2e-6
    )


def test_pure_vorticity_rayleigh_matches_known_polynomial() -> None:
    cp, backend = _cuda_backend()
    nodes = np.asarray(
        [-0.86113631, -0.33998104, 0.33998104, 0.86113631],
        dtype=np.float32,
    )
    one_dimensional_weights = np.asarray(
        [0.34785485, 0.65214515, 0.65214515, 0.34785485],
        dtype=np.float32,
    )
    gx, gy = np.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = np.meshgrid(
        one_dimensional_weights, one_dimensional_weights, indexing="ij"
    )
    points = np.column_stack((gx.ravel(), gy.ravel()))
    weights = cp.asarray((wx * wy).ravel(), dtype=backend.F32)

    x, y = backend.xy_jets(points, backend.F32)
    # psi=x^4+y^4 gives omega=-12(x^2+y^2) on [-1,1]^2.
    field = x.pow(4) + y.pow(4)
    omega, omega_gradient = backend.vorticity_and_gradient(field)
    omega = omega[:, 0]
    omega_gradient = omega_gradient[:, 0]
    mass = cp.sum(weights * omega * omega)
    stiffness = cp.sum(weights[:, None] * omega_gradient * omega_gradient)
    rayleigh = float((stiffness / mass).get())

    # Integral calculation is exact under the tensor Gauss rule:
    # int |grad omega|^2 / int omega^2 = 1536 / (1792/5) = 30/7.
    assert rayleigh == pytest.approx(30.0 / 7.0, rel=2e-6, abs=2e-6)


def test_handwritten_jet3_reverse_matches_complex_step_parameter_gradient() -> None:
    cp, backend = _cuda_backend()
    points = np.asarray(
        [[-0.7, 0.2], [-0.15, -0.8], [0.5, 0.4], [0.9, -0.3]],
        dtype=np.float32,
    )
    spatial_coefficients = np.asarray([1.2, -0.7], dtype=np.float32)
    parameters = np.asarray([0.83, -0.17, 1.31, 0.29], dtype=np.float32)
    w1_value, b1_value, w2_value, b2_value = parameters

    rng = np.random.default_rng(20260812)
    coefficient_value = rng.normal(size=(len(points), 1)).astype(np.float32)
    coefficient_grad = rng.normal(size=(len(points), 1, 2)).astype(np.float32)
    coefficient_hess = rng.normal(size=(len(points), 1, 3)).astype(np.float32)
    coefficient_third = rng.normal(size=(len(points), 1, 4)).astype(np.float32)

    x, y = backend.xy_jets(points, backend.F32)
    source = spatial_coefficients[0] * x + spatial_coefficients[1] * y
    weight1 = cp.asarray([[w1_value]], dtype=backend.F32)
    bias1 = cp.asarray([b1_value], dtype=backend.F32)
    weight2 = cp.asarray([[w2_value]], dtype=backend.F32)
    preactivation = backend.linear_forward(
        source, weight1, bias1, backend.F32
    )
    hidden = backend.sine_forward(preactivation)
    output_adjoint = backend.Jet3(
        cp.asarray(coefficient_value),
        cp.asarray(coefficient_grad),
        cp.asarray(coefficient_hess),
        cp.asarray(coefficient_third),
    )

    hidden_adjoint, grad_weight2, grad_bias2 = backend.linear_backward(
        hidden, output_adjoint, weight2, backend.F32
    )
    preactivation_adjoint = backend.sine_backward(
        preactivation, hidden_adjoint
    )
    _, grad_weight1, grad_bias1 = backend.linear_backward(
        source, preactivation_adjoint, weight1, backend.F32
    )
    handwritten = np.asarray(
        [
            float(grad_weight1[0, 0].get()),
            float(grad_bias1[0].get()),
            float(grad_weight2[0, 0].get()),
            float(grad_bias2[0].get()),
        ]
    )

    q = (
        spatial_coefficients[0] * points[:, 0:1]
        + spatial_coefficients[1] * points[:, 1:2]
    )
    first_products = spatial_coefficients
    second_products = np.asarray(
        [
            spatial_coefficients[0] ** 2,
            spatial_coefficients[0] * spatial_coefficients[1],
            spatial_coefficients[1] ** 2,
        ]
    )
    third_products = np.asarray(
        [
            spatial_coefficients[0] ** 3,
            spatial_coefficients[0] ** 2 * spatial_coefficients[1],
            spatial_coefficients[0] * spatial_coefficients[1] ** 2,
            spatial_coefficients[1] ** 3,
        ]
    )

    def analytic_linear_functional(candidate: np.ndarray) -> complex:
        w1, b1, w2, b2 = candidate
        phase = w1 * q + b1
        sine, cosine = np.sin(phase), np.cos(phase)
        value = w2 * sine + b2
        grad = w2 * cosine[..., None] * w1 * first_products
        hess = -w2 * sine[..., None] * w1**2 * second_products
        third = -w2 * cosine[..., None] * w1**3 * third_products
        return (
            np.sum(coefficient_value * value)
            + np.sum(coefficient_grad * grad)
            + np.sum(coefficient_hess * hess)
            + np.sum(coefficient_third * third)
        )

    complex_step = np.empty_like(parameters)
    complex_epsilon = np.float32(1.0e-4)
    for index in range(len(parameters)):
        candidate = parameters.astype(np.complex64)
        candidate[index] += 1j * complex_epsilon
        complex_step[index] = (
            analytic_linear_functional(candidate).imag / complex_epsilon
        )

    np.testing.assert_allclose(
        handwritten, complex_step, rtol=2e-5, atol=2e-5
    )


def test_batched_linear_reverse_matches_independent_modes() -> None:
    cp, backend = _cuda_backend()
    rng = np.random.default_rng(20260812)
    point_count, feature_count, output_count, mode_count = 5, 3, 2, 2

    def random_jet(shape: tuple[int, ...]):
        def array(*suffix: int):
            return cp.asarray(
                rng.normal(size=(*shape, *suffix)).astype(np.float32)
            )

        return backend.Jet3(array(), array(2), array(3), array(4))

    shared_source = random_jet((point_count, feature_count))
    batched_source = random_jet((mode_count, point_count, feature_count))
    adjoint = random_jet((mode_count, point_count, output_count))
    weights = cp.asarray(
        rng.normal(size=(mode_count, output_count, feature_count)).astype(
            np.float32
        )
    )

    for source in (shared_source, batched_source):
        batch_source_adjoint, batch_weight, batch_bias = (
            backend.linear_backward_batch(source, adjoint, weights)
        )
        for index in range(mode_count):
            source_for_mode = (
                source if source.value.ndim == 2 else backend._slice_jet(source, index)
            )
            independent = backend.linear_backward(
                source_for_mode,
                backend._slice_jet(adjoint, index),
                weights[index],
                backend.F32,
            )
            source_adjoint, grad_weight, grad_bias = independent
            for channel in ("value", "grad", "hess", "third"):
                np.testing.assert_allclose(
                    _numpy(cp, getattr(batch_source_adjoint, channel)[index]),
                    _numpy(cp, getattr(source_adjoint, channel)),
                    rtol=2e-6,
                    atol=2e-6,
                )
            np.testing.assert_allclose(
                _numpy(cp, batch_weight[index]),
                _numpy(cp, grad_weight),
                rtol=2e-6,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                _numpy(cp, batch_bias[index]),
                _numpy(cp, grad_bias),
                rtol=2e-6,
                atol=2e-6,
            )


def _python_identifiers(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    return {
        token.string
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type == tokenize.NAME
    }


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_production_training_path_has_no_discrete_spatial_derivative_or_torch() -> None:
    for path in (BACKEND_PATH, TRAINER_PATH):
        identifiers = _python_identifiers(path)
        forbidden_discrete_names = sorted(
            name
            for name in identifiers
            if name.lower() != "difference"
            and re.search(r"(?:neighbou?r|difference)", name, flags=re.IGNORECASE)
        )
        assert not forbidden_discrete_names, (
            f"{path.name} retains discrete spatial-derivative identifiers: "
            f"{forbidden_discrete_names}"
        )
        assert "torch" not in _import_roots(path)
        forbidden_precision_names = sorted({"F64", "float64"}.intersection(identifiers))
        assert not forbidden_precision_names, (
            f"{path.name} retains non-FP32 numerical paths: "
            f"{forbidden_precision_names}"
        )

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden_gradient_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (
                node.func.attr == "gradient"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"np", "numpy", "cp", "cupy"}
            ):
                forbidden_gradient_calls.append(node.lineno)
        assert not forbidden_gradient_calls, (
            f"{path.name} calls a grid-gradient operator at lines "
            f"{forbidden_gradient_calls}"
        )


def test_reverse_guard_allows_rayleigh_invariant_zero_parameter_tensor() -> None:
    source = BACKEND_PATH.read_text(encoding="utf-8")
    assert "zero hardcoded gradient tensor" not in source
    assert "zero or non-finite total hardcoded gradient" in source


def test_full_k_optimizer_step_is_not_a_cache_minibatch() -> None:
    source = BACKEND_PATH.read_text(encoding="utf-8")
    trainer = TRAINER_PATH.read_text(encoding="utf-8")
    assert "complete causal least-squares peeling" in source
    assert "model.forward_mode_block(" in source
    assert "complete_gradients: list[cp.ndarray] | None = None" in source
    assert "model.adam_step_all(complete_gradients, cfg)" in source
    assert "for point_start in range(0, n, point_block)" in source
    assert "cache_block_is_optimizer_minibatch" in trainer
    assert "point_stream_is_mode_minibatch" in trainer
    assert "checkpoint_due = (" in trainer
    assert "epoch % cfg.checkpoint_every == 0" in trainer
    assert "epoch == stop_epoch" in trainer


def test_pure_vorticity_config_has_one_unscaled_unregularized_objective() -> None:
    config_path = (
        PURE_CONFIG_PATH.parent
        / "k1024_ncl_cinf_sobol_10000_fp16_pure_vorticity_rayleigh.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    objective_keys = {key for key in config if "objective" in key.lower()}
    assert objective_keys == {"basis_objective"}
    assert config["basis_objective"] == "pure_vorticity_rayleigh"
    assert config["spatial_derivative_order"] == 3
    assert config["epochs"] == 10000
    assert config["validate_every"] == 500
    assert config["probe_every"] == 0
    assert config["checkpoint_every"] == 256
    assert config["archive_checkpoint_every"] == 256
    assert config["field_representation"] == "ncl_antisymmetric_tensor_2d"
    assert config["modes_per_optimizer_step"] == config["modes"] == 1024
    assert config["mode_forward_batch"] == config["modes"]
    assert config["mode_backward_batch"] == config["modes"]
    assert config["point_stream_batch"] > 0

    forbidden_keys = sorted(
        key
        for key in config
        if key.lower().endswith("_weight")
        or "regularization" in key.lower()
        or key.lower().endswith("_scale")
        or "hybrid" in key.lower()
        or "loss" in key.lower()
    )
    assert not forbidden_keys, (
        "pure vorticity configuration must not contain objective weights, "
        f"regularizers, or loss scaling: {forbidden_keys}"
    )

    assert config["mlp_compute_dtype"] == "float16"
    for key in (
        "mlp_accumulation_dtype",
        "jet_dtype",
        "parameter_backward_dtype",
        "quadrature_dtype",
        "ritz_dtype",
        "master_weight_dtype",
        "optimizer_dtype",
    ):
        assert config[key] == "float32", key
    assert config["fp32_boundary_layers"] is False
