"""Handwritten CuPy backend for the 3D parametric fan SSS case.

There is deliberately no PyTorch, TensorFlow, JAX, autograd, or ML framework in
this module.  The implementation writes out the MLP forward pass, reverse pass,
Adam update, centered-difference tensor-potential operator, dynamic Rayleigh
ordering, and gradient-causal SSS peeling explicitly.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np
from scipy.stats import qmc


F16 = cp.float16
F32 = cp.float32
F64 = cp.float64


def cfg_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def _compute_dtype(cfg: SimpleNamespace) -> type[cp.generic]:
    name = str(getattr(cfg, "mlp_compute_dtype", "float32")).lower()
    if name == "float16":
        return F16
    if name == "float32":
        return F32
    raise ValueError(f"unsupported MLP compute dtype: {name}")


def _matmul(
    left: cp.ndarray,
    right: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> cp.ndarray:
    if compute_dtype is F16:
        return cp.matmul(
            left.astype(F16, copy=False),
            right.astype(F16, copy=False),
            dtype=F32,
        )
    return left @ right


def _batched_matmul(
    left: cp.ndarray,
    right: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> cp.ndarray:
    if compute_dtype is F16:
        return cp.matmul(
            left.astype(F16, copy=False),
            right.astype(F16, copy=False),
            dtype=F32,
        )
    return cp.matmul(left, right)


def parameter_values(
    cfg: SimpleNamespace, parameters: np.ndarray | cp.ndarray
) -> tuple[Any, Any, Any]:
    beta = cfg.beta_tip_min_deg + parameters[..., 0] * (
        cfg.beta_tip_max_deg - cfg.beta_tip_min_deg
    )
    chord = cfg.chord_tip_min + parameters[..., 1] * (
        cfg.chord_tip_max - cfg.chord_tip_min
    )
    thickness = cfg.thickness_min + parameters[..., 2] * (
        cfg.thickness_max - cfg.thickness_min
    )
    return beta, chord, thickness


def _camber(
    cfg: SimpleNamespace, xi: np.ndarray | cp.ndarray
) -> np.ndarray | cp.ndarray:
    module = cp.get_array_module(xi)
    m, p = cfg.camber, cfg.camber_position
    fore = m / (p * p) * (2.0 * p * xi - xi * xi)
    aft = m / ((1.0 - p) ** 2) * (
        (1.0 - 2.0 * p) + 2.0 * p * xi - xi * xi
    )
    return module.where(xi < p, fore, aft)


def _half_thickness(
    xi: np.ndarray | cp.ndarray,
    thickness: float | np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    module = cp.get_array_module(xi)
    clipped = module.clip(xi, 0.0, 1.0)
    return 5.0 * thickness * (
        0.2969 * module.sqrt(clipped)
        - 0.1260 * clipped
        - 0.3516 * clipped**2
        + 0.2843 * clipped**3
        - 0.1036 * clipped**4
    )


def rotor_distance(
    cfg: SimpleNamespace,
    points: np.ndarray | cp.ndarray,
    parameters: np.ndarray | cp.ndarray,
) -> np.ndarray | cp.ndarray:
    """Signed-distance proxy for the four lofted blades and capped hub."""
    module = cp.get_array_module(points)
    parameters = module.asarray(parameters, dtype=points.dtype)
    beta_tip, chord_tip, thickness_ratio = parameter_values(cfg, parameters)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    radius = module.sqrt(y * y + z * z + 1.0e-12)
    azimuth = module.arctan2(z, y)
    sector = 2.0 * math.pi / cfg.blade_count
    phase = module.remainder(azimuth + 0.5 * sector, sector) - 0.5 * sector
    tangential = radius * phase
    span = module.clip(
        (radius - cfg.hub_radius)
        / (cfg.tip_radius - cfg.hub_radius),
        0.0,
        1.0,
    )
    smooth_span = span * span * (3.0 - 2.0 * span)
    beta = module.deg2rad(
        cfg.beta_root_deg
        + (beta_tip - cfg.beta_root_deg) * span**0.82
    )
    chord = cfg.chord_root + (chord_tip - cfg.chord_root) * smooth_span
    chord = chord * (1.0 - 0.20 * span**10)
    sweep = 0.18 * span * span - 0.035 * span
    rake = -0.08 * span * span
    xlocal = x - rake
    tlocal = tangential - sweep
    chord_coordinate = xlocal * module.sin(beta) + tlocal * module.cos(beta)
    normal_coordinate = xlocal * module.cos(beta) - tlocal * module.sin(beta)
    xi = chord_coordinate / module.maximum(chord, 1.0e-5) + cfg.stacking_fraction
    camber = chord * _camber(cfg, module.clip(xi, 0.0, 1.0))
    thickness = chord * _half_thickness(xi, thickness_ratio)
    blade_distance = module.maximum(
        module.maximum(
            module.abs(normal_coordinate - camber) - thickness,
            -xi * chord,
        ),
        module.maximum(
            (xi - 1.0) * chord,
            module.maximum(
                cfg.hub_radius - radius,
                radius - cfg.tip_radius,
            ),
        ),
    )
    hub_distance = module.maximum(
        module.abs(x) - 0.5 * cfg.hub_length,
        radius - cfg.hub_radius,
    )
    return module.minimum(blade_distance, hub_distance)


def inside_rotor(
    cfg: SimpleNamespace, points: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    return np.asarray(rotor_distance(cfg, points, parameters)) <= 0.0


def boundary_envelope(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> cp.ndarray:
    """Double-zero perturbation envelope at inlet, cylinder, and rotor."""
    radius = cp.sqrt(points[:, 1] ** 2 + points[:, 2] ** 2 + 1.0e-12)
    inlet = cp.maximum((points[:, 0] - cfg.xmin) / cfg.tip_radius, 0.0)
    cylinder = cp.maximum(
        (cfg.farfield_radius - radius) / cfg.tip_radius,
        0.0,
    )
    obstacle = cp.maximum(
        rotor_distance(cfg, points, parameters) / cfg.tip_radius,
        0.0,
    )

    def saturate(value: cp.ndarray) -> cp.ndarray:
        return value / (1.0 + value)

    return (saturate(inlet) * saturate(cylinder) * saturate(obstacle)) ** 2


def encoded_inputs(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> cp.ndarray:
    """Match the open-domain 3D Fourier/SIREN feature convention."""
    bounds_low = cp.asarray(
        (cfg.xmin, -cfg.farfield_radius, -cfg.farfield_radius),
        dtype=F32,
    )
    bounds_high = cp.asarray(
        (cfg.xmax, cfg.farfield_radius, cfg.farfield_radius),
        dtype=F32,
    )
    normal = (points - bounds_low) / (bounds_high - bounds_low)
    frequencies = 2.0 ** cp.arange(cfg.frequencies, dtype=F32)
    phase = 2.0 * math.pi * normal[:, :, None] * frequencies[None, None, :]
    parameter_rows = cp.broadcast_to(
        parameters.reshape(1, 3), (len(points), 3)
    )
    return cp.concatenate(
        (
            cp.sin(phase).reshape(len(points), -1),
            cp.cos(phase).reshape(len(points), -1),
            2.0 * normal - 1.0,
            parameter_rows,
        ),
        axis=1,
    ).astype(F32)


def stencil_offsets() -> list[tuple[int, int, int]]:
    offsets: set[tuple[int, int, int]] = {(0, 0, 0)}
    for axis in range(3):
        for sign in (-1, 1):
            value = [0, 0, 0]
            value[axis] = sign
            offsets.add(tuple(value))
    for first in range(3):
        for second in range(first + 1, 3):
            for sign_first in (-1, 1):
                for sign_second in (-1, 1):
                    value = [0, 0, 0]
                    value[first], value[second] = sign_first, sign_second
                    offsets.add(tuple(value))
    return sorted(offsets)


STENCIL_OFFSETS = stencil_offsets()
ORIGIN_INDEX = STENCIL_OFFSETS.index((0, 0, 0))


def shifted_points(cfg: SimpleNamespace, points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            points
            + np.asarray(offset, dtype=np.float32)[None, :]
            * cfg.difference_step
            for offset in STENCIL_OFFSETS
        ],
        axis=0,
    ).astype(np.float32)


def _first(
    values: dict[tuple[int, int, int], cp.ndarray],
    axis: int,
    h: float,
) -> cp.ndarray:
    plus, minus = [0, 0, 0], [0, 0, 0]
    plus[axis], minus[axis] = 1, -1
    return (values[tuple(plus)] - values[tuple(minus)]) / (2.0 * h)


def _hessian(
    values: dict[tuple[int, int, int], cp.ndarray],
    origin: cp.ndarray,
    first_axis: int,
    second_axis: int,
    h: float,
) -> cp.ndarray:
    if first_axis == second_axis:
        plus, minus = [0, 0, 0], [0, 0, 0]
        plus[first_axis], minus[first_axis] = 1, -1
        return (
            values[tuple(plus)] - 2.0 * origin + values[tuple(minus)]
        ) / (h * h)
    pp, pm, mp, mm = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
    pp[first_axis], pp[second_axis] = 1, 1
    pm[first_axis], pm[second_axis] = 1, -1
    mp[first_axis], mp[second_axis] = -1, 1
    mm[first_axis], mm[second_axis] = -1, -1
    return (
        values[tuple(pp)]
        - values[tuple(pm)]
        - values[tuple(mp)]
        + values[tuple(mm)]
    ) / (4.0 * h * h)


def fields_from_stencil(
    cfg: SimpleNamespace,
    evaluated: cp.ndarray,
    point_count: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Map [mode, stencil*points, A12/A13/A23] to p, velocity, gradient."""
    mode_count = evaluated.shape[0]
    chunks = evaluated.reshape(
        mode_count, len(STENCIL_OFFSETS), point_count, 3
    )
    values = {
        offset: chunks[:, index]
        for index, offset in enumerate(STENCIL_OFFSETS)
    }
    origin = values[(0, 0, 0)]
    first = cp.empty((mode_count, point_count, 3, 3), dtype=F64)
    for axis in range(3):
        first[:, :, :, axis] = _first(
            values, axis, cfg.difference_step
        ).astype(F64)
    a12, a13, a23 = 0, 1, 2
    velocity = cp.stack(
        (
            first[:, :, a12, 1] + first[:, :, a13, 2],
            -first[:, :, a12, 0] + first[:, :, a23, 2],
            -first[:, :, a13, 0] - first[:, :, a23, 1],
        ),
        axis=2,
    )
    hessian = cp.empty(
        (mode_count, point_count, 3, 3, 3), dtype=F64
    )
    for first_axis in range(3):
        for second_axis in range(first_axis, 3):
            mixed = _hessian(
                values,
                origin,
                first_axis,
                second_axis,
                cfg.difference_step,
            ).astype(F64)
            hessian[:, :, :, first_axis, second_axis] = mixed
            hessian[:, :, :, second_axis, first_axis] = mixed
    gradient = cp.empty(
        (mode_count, point_count, 3, 3), dtype=F64
    )
    for derivative in range(3):
        gradient[:, :, 0, derivative] = (
            hessian[:, :, a12, 1, derivative]
            + hessian[:, :, a13, 2, derivative]
        )
        gradient[:, :, 1, derivative] = (
            -hessian[:, :, a12, 0, derivative]
            + hessian[:, :, a23, 2, derivative]
        )
        gradient[:, :, 2, derivative] = (
            -hessian[:, :, a13, 0, derivative]
            - hessian[:, :, a23, 1, derivative]
        )
    return (
        origin.astype(F64),
        velocity,
        gradient,
    )


def stencil_adjoint_matrix(cfg: SimpleNamespace) -> cp.ndarray:
    """Exact transpose map from velocity/gradient adjoints to A samples."""
    sample_count = len(STENCIL_OFFSETS) * 3
    matrix = np.zeros((12, sample_count), dtype=np.float64)
    index = {
        (offset, potential): offset_index * 3 + potential
        for offset_index, offset in enumerate(STENCIL_OFFSETS)
        for potential in range(3)
    }
    h = cfg.difference_step

    def add_first(row: int, potential: int, axis: int, scale: float) -> None:
        plus, minus = [0, 0, 0], [0, 0, 0]
        plus[axis], minus[axis] = 1, -1
        matrix[row, index[(tuple(plus), potential)]] += scale / (2.0 * h)
        matrix[row, index[(tuple(minus), potential)]] -= scale / (2.0 * h)

    def add_hessian(
        row: int,
        potential: int,
        first_axis: int,
        second_axis: int,
        scale: float,
    ) -> None:
        if first_axis == second_axis:
            plus, minus = [0, 0, 0], [0, 0, 0]
            plus[first_axis], minus[first_axis] = 1, -1
            factor = scale / (h * h)
            matrix[row, index[(tuple(plus), potential)]] += factor
            matrix[row, index[(tuple(minus), potential)]] += factor
            matrix[row, index[((0, 0, 0), potential)]] -= 2.0 * factor
            return
        pp, pm, mp, mm = (
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        )
        pp[first_axis], pp[second_axis] = 1, 1
        pm[first_axis], pm[second_axis] = 1, -1
        mp[first_axis], mp[second_axis] = -1, 1
        mm[first_axis], mm[second_axis] = -1, -1
        factor = scale / (4.0 * h * h)
        matrix[row, index[(tuple(pp), potential)]] += factor
        matrix[row, index[(tuple(pm), potential)]] -= factor
        matrix[row, index[(tuple(mp), potential)]] -= factor
        matrix[row, index[(tuple(mm), potential)]] += factor

    # velocity rows
    add_first(0, 0, 1, 1.0)
    add_first(0, 1, 2, 1.0)
    add_first(1, 0, 0, -1.0)
    add_first(1, 2, 2, 1.0)
    add_first(2, 1, 0, -1.0)
    add_first(2, 2, 1, -1.0)
    # gradient rows are ordered component-major, derivative-minor.
    for derivative in range(3):
        row_u = 3 + derivative
        row_v = 3 + 3 + derivative
        row_w = 3 + 6 + derivative
        add_hessian(row_u, 0, 1, derivative, 1.0)
        add_hessian(row_u, 1, 2, derivative, 1.0)
        add_hessian(row_v, 0, 0, derivative, -1.0)
        add_hessian(row_v, 2, 2, derivative, 1.0)
        add_hessian(row_w, 1, 0, derivative, -1.0)
        add_hessian(row_w, 2, 1, derivative, -1.0)
    divergence_row = matrix[3] + matrix[7] + matrix[11]
    if np.count_nonzero(divergence_row) != 0:
        raise RuntimeError("centered-difference div(curl) operator is not exact")
    return cp.asarray(matrix, dtype=F64)


@dataclass
class ForwardCache:
    inputs: list[cp.ndarray]
    preactivations: list[cp.ndarray]
    output_input: cp.ndarray
    envelope: cp.ndarray


class HardcodedFanModel:
    """Stacked independent SIREN modes with explicit reverse and Adam."""

    def __init__(
        self, cfg: SimpleNamespace, arrays: dict[str, np.ndarray]
    ) -> None:
        self.cfg = cfg
        self.compute_dtype = _compute_dtype(cfg)
        self.weights = [
            cp.asarray(arrays[f"l{layer}.weight"], dtype=F32)
            for layer in range(cfg.hidden_layers)
        ]
        self.biases = [
            cp.asarray(arrays[f"l{layer}.bias"], dtype=F32)
            for layer in range(cfg.hidden_layers)
        ]
        self.weights.append(cp.asarray(arrays["output.weight"], dtype=F32))
        self.biases.append(cp.asarray(arrays["output.bias"], dtype=F32))
        self.exp_avg = [
            cp.asarray(arrays[f"adam.p{index:02d}.exp_avg"], dtype=F32)
            for index in range(2 * len(self.weights))
        ]
        self.exp_avg_sq = [
            cp.asarray(
                arrays[f"adam.p{index:02d}.exp_avg_sq"], dtype=F32
            )
            for index in range(2 * len(self.weights))
        ]
        self.steps = np.asarray(arrays["adam.steps"], dtype=np.int64).copy()
        shadow_dtype = F16 if self.compute_dtype is F16 else F32
        self.shadow_weights = [
            value.astype(shadow_dtype) for value in self.weights
        ]
        self.shadow_biases = [
            value.astype(shadow_dtype) for value in self.biases
        ]

    def forward_block(
        self,
        inputs: cp.ndarray,
        envelope: cp.ndarray,
        start: int,
        stop: int,
    ) -> cp.ndarray:
        current = cp.broadcast_to(
            inputs[None, :, :], (stop - start, *inputs.shape)
        )
        for layer in range(len(self.weights) - 1):
            current = cp.sin(
                _batched_matmul(
                    current,
                    self.shadow_weights[layer][start:stop].transpose(0, 2, 1),
                    self.compute_dtype,
                )
                + self.shadow_biases[layer][start:stop, None, :]
            )
        raw = _batched_matmul(
            current,
            self.shadow_weights[-1][start:stop].transpose(0, 2, 1),
            self.compute_dtype,
        ) + self.shadow_biases[-1][start:stop, None, :]
        return raw * envelope[None, :, None]

    def forward_single(
        self,
        mode: int,
        inputs: cp.ndarray,
        envelope: cp.ndarray,
        *,
        cache: bool,
    ) -> tuple[cp.ndarray, ForwardCache | None]:
        current = inputs
        layer_inputs: list[cp.ndarray] = []
        preactivations: list[cp.ndarray] = []
        for layer in range(len(self.weights) - 1):
            if cache:
                layer_inputs.append(current)
            preactivation = (
                _matmul(
                    current,
                    self.shadow_weights[layer][mode].T,
                    self.compute_dtype,
                )
                + self.shadow_biases[layer][mode]
            )
            if cache:
                preactivations.append(preactivation)
            current = cp.sin(preactivation)
        output_input = current
        raw = (
            _matmul(
                current,
                self.shadow_weights[-1][mode].T,
                self.compute_dtype,
            )
            + self.shadow_biases[-1][mode]
        )
        field = raw * envelope[:, None]
        if not cache:
            return field, None
        return field, ForwardCache(
            layer_inputs,
            preactivations,
            output_input,
            envelope,
        )

    def backward_single(
        self,
        mode: int,
        cache: ForwardCache,
        field_adjoint: cp.ndarray,
    ) -> list[cp.ndarray]:
        raw_adjoint = field_adjoint.astype(F32) * cache.envelope[:, None]
        output_weight = raw_adjoint.T @ cache.output_input
        output_bias = raw_adjoint.sum(axis=0)
        current = raw_adjoint @ self.weights[-1][mode]
        weight_gradients = [output_weight]
        bias_gradients = [output_bias]
        for layer in range(len(self.weights) - 2, -1, -1):
            current = current * cp.cos(cache.preactivations[layer])
            weight_gradients.append(
                current.T @ cache.inputs[layer]
            )
            bias_gradients.append(current.sum(axis=0))
            current = current @ self.weights[layer][mode]
        weight_gradients.reverse()
        bias_gradients.reverse()
        gradients: list[cp.ndarray] = []
        for weight, bias in zip(weight_gradients, bias_gradients):
            gradients.extend((weight, bias))
        return gradients

    def adam_step(
        self,
        mode: int,
        gradients: list[cp.ndarray],
        learning_rate: float,
    ) -> None:
        if len(gradients) != 2 * len(self.weights):
            raise RuntimeError("manual reverse returned wrong gradient count")
        total = sum(cp.sum(value.astype(F64) ** 2) for value in gradients)
        norm = float(cp.sqrt(total).get())
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError(
                f"mode {mode} has invalid manual gradient norm {norm}"
            )
        scale = min(1.0, self.cfg.grad_clip / max(norm, 1.0e-30))
        beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
        parameter_index = 0
        for layer in range(len(self.weights)):
            for parameter, gradient in (
                (self.weights[layer], gradients[2 * layer]),
                (self.biases[layer], gradients[2 * layer + 1]),
            ):
                gradient = gradient.astype(F32) * scale
                self.steps[mode, parameter_index] += 1
                step = int(self.steps[mode, parameter_index])
                average = self.exp_avg[parameter_index]
                square = self.exp_avg_sq[parameter_index]
                average[mode] *= beta1
                average[mode] += (1.0 - beta1) * gradient
                square[mode] *= beta2
                square[mode] += (1.0 - beta2) * gradient * gradient
                correction1 = 1.0 - beta1**step
                correction2 = 1.0 - beta2**step
                denominator = cp.sqrt(square[mode] / correction2) + epsilon
                parameter[mode] -= (
                    learning_rate / correction1
                ) * average[mode] / denominator
                parameter_index += 1
        for layer in range(len(self.weights)):
            self.shadow_weights[layer][mode] = self.weights[layer][mode]
            self.shadow_biases[layer][mode] = self.biases[layer][mode]

    def arrays(self, *, include_adam: bool) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for layer in range(len(self.weights) - 1):
            result[f"l{layer}.weight"] = cp.asnumpy(self.weights[layer])
            result[f"l{layer}.bias"] = cp.asnumpy(self.biases[layer])
        result["output.weight"] = cp.asnumpy(self.weights[-1])
        result["output.bias"] = cp.asnumpy(self.biases[-1])
        if include_adam:
            for index in range(2 * len(self.weights)):
                result[f"adam.p{index:02d}.exp_avg"] = cp.asnumpy(
                    self.exp_avg[index]
                )
                result[f"adam.p{index:02d}.exp_avg_sq"] = cp.asnumpy(
                    self.exp_avg_sq[index]
                )
            result["adam.steps"] = self.steps
        return result


def initialize_arrays(cfg: SimpleNamespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    feature_count = 6 * cfg.frequencies + 6
    arrays: dict[str, np.ndarray] = {}
    fan_in = feature_count
    parameter_shapes: list[tuple[int, ...]] = []
    for layer in range(cfg.hidden_layers):
        bound = 1.0 / math.sqrt(fan_in)
        weight = rng.uniform(
            -bound,
            bound,
            (cfg.modes, cfg.hidden, fan_in),
        ).astype(np.float32)
        bias = rng.uniform(
            -bound,
            bound,
            (cfg.modes, cfg.hidden),
        ).astype(np.float32)
        arrays[f"l{layer}.weight"] = weight
        arrays[f"l{layer}.bias"] = bias
        parameter_shapes.extend((weight.shape, bias.shape))
        fan_in = cfg.hidden
    bound = 1.0 / math.sqrt(fan_in)
    output_weight = rng.uniform(
        -bound,
        bound,
        (cfg.modes, 3, fan_in),
    ).astype(np.float32)
    output_bias = rng.uniform(
        -bound,
        bound,
        (cfg.modes, 3),
    ).astype(np.float32)
    arrays["output.weight"] = output_weight
    arrays["output.bias"] = output_bias
    parameter_shapes.extend((output_weight.shape, output_bias.shape))
    for index, shape in enumerate(parameter_shapes):
        arrays[f"adam.p{index:02d}.exp_avg"] = np.zeros(shape, np.float32)
        arrays[f"adam.p{index:02d}.exp_avg_sq"] = np.zeros(shape, np.float32)
    arrays["adam.steps"] = np.zeros(
        (cfg.modes, len(parameter_shapes)), dtype=np.int64
    )
    return arrays


def _sobol_cylinder(
    cfg: SimpleNamespace,
    count: int,
    seed: int,
) -> np.ndarray:
    sampler = qmc.Sobol(3, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol",
            category=UserWarning,
        )
        sample = sampler.random(count)
    x = cfg.xmin + (cfg.xmax - cfg.xmin) * sample[:, 0]
    radius = cfg.farfield_radius * np.sqrt(sample[:, 1])
    angle = 2.0 * math.pi * sample[:, 2]
    return np.column_stack(
        (x, radius * np.cos(angle), radius * np.sin(angle))
    ).astype(np.float32)


def fluid_points(
    cfg: SimpleNamespace,
    count: int,
    parameters: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Mesh-free cylindrical Sobol quadrature outside the four-blade rotor."""
    pieces: list[np.ndarray] = []
    remaining = count
    attempt = 0
    while remaining:
        candidates = _sobol_cylinder(
            cfg,
            max(2 * remaining, 128),
            seed + attempt,
        )
        accepted = candidates[
            ~inside_rotor(cfg, candidates, parameters)
        ][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
        attempt += 1
        if attempt > 30:
            raise RuntimeError("failed to sample cylindrical fluid points")
    points = np.concatenate(pieces)
    return points[np.random.default_rng(seed).permutation(len(points))]


def continuous_parameters(
    cfg: SimpleNamespace, epoch: int
) -> np.ndarray:
    if cfg.endpoint_period > 0 and epoch % cfg.endpoint_period == 0:
        corner = (epoch // cfg.endpoint_period - 1) % 8
        return np.asarray(
            (
                float((corner >> 0) & 1),
                float((corner >> 1) & 1),
                float((corner >> 2) & 1),
            ),
            dtype=np.float32,
        )
    sampler = qmc.Sobol(
        3, scramble=True, seed=cfg.seed + 700_000 + epoch
    )
    return sampler.random(1)[0].astype(np.float32)


def approximate_fluid_volume(cfg: SimpleNamespace) -> float:
    cylinder = (
        math.pi
        * cfg.farfield_radius**2
        * (cfg.xmax - cfg.xmin)
    )
    return cylinder


def prepared_network_inputs(
    cfg: SimpleNamespace,
    points: np.ndarray,
    parameters: np.ndarray,
) -> tuple[np.ndarray, cp.ndarray, cp.ndarray]:
    shifted = shifted_points(cfg, points)
    shifted_device = cp.asarray(shifted, dtype=F32)
    parameter_device = cp.asarray(parameters, dtype=F32)
    return (
        shifted,
        encoded_inputs(cfg, shifted_device, parameter_device),
        boundary_envelope(cfg, shifted_device, parameter_device),
    )


def raw_fields(
    cfg: SimpleNamespace,
    model: HardcodedFanModel,
    points: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, cp.ndarray]:
    point_count = len(points)
    _, inputs, envelope = prepared_network_inputs(
        cfg, points, parameters
    )
    raw_p = cp.empty((3 * point_count, cfg.modes), dtype=F64)
    raw_v = cp.empty((3 * point_count, cfg.modes), dtype=F64)
    raw_g = cp.empty((9 * point_count, cfg.modes), dtype=F64)
    block = max(1, int(cfg.mode_forward_batch))
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        evaluated = model.forward_block(inputs, envelope, start, stop)
        potential, velocity, gradient = fields_from_stencil(
            cfg, evaluated, point_count
        )
        raw_p[:, start:stop] = potential.transpose(2, 1, 0).reshape(
            3 * point_count, stop - start
        )
        raw_v[:, start:stop] = velocity.transpose(2, 1, 0).reshape(
            3 * point_count, stop - start
        )
        raw_g[:, start:stop] = gradient.transpose(2, 3, 1, 0).reshape(
            9 * point_count, stop - start
        )
    return {
        "potential": raw_p,
        "velocity": raw_v,
        "gradient": raw_g,
        "inputs": inputs,
        "envelope": envelope,
    }


def peel_and_adjoint(
    cfg: SimpleNamespace,
    raw: dict[str, cp.ndarray],
    point_count: int,
    *,
    need_adjoint: bool,
    need_coefficients: bool = False,
) -> dict[str, Any]:
    weight = approximate_fluid_volume(cfg) / point_count
    raw_p = raw["potential"]
    raw_v = raw["velocity"]
    raw_g = raw["gradient"]
    mass = weight * cp.sum(raw_v * raw_v, axis=0)
    stiffness = weight * cp.sum(raw_g * raw_g, axis=0)
    raw_energy = stiffness / cp.maximum(mass, cfg.mass_floor)
    order = cp.argsort(raw_energy, kind="stable")

    qv = cp.empty_like(raw_v)
    qg = cp.empty_like(raw_g)
    qp = cp.empty_like(raw_p)
    adj_v = cp.zeros_like(raw_v) if need_adjoint else None
    adj_g = cp.zeros_like(raw_g) if need_adjoint else None
    coefficient_matrix = (
        cp.zeros((cfg.modes, cfg.modes), dtype=F64)
        if need_coefficients
        else None
    )
    energies = np.empty(cfg.modes, dtype=np.float64)
    projection_ratios = np.empty(cfg.modes, dtype=np.float64)
    energy_by_network = np.empty(cfg.modes, dtype=np.float64)

    for rank in range(cfg.modes):
        mode = int(order[rank].get())
        velocity = raw_v[:, mode]
        gradient = raw_g[:, mode]
        potential = raw_p[:, mode]
        if rank:
            coefficients = weight * (qv[:, :rank].T @ velocity)
            residual_v = velocity - qv[:, :rank] @ coefficients
            residual_g = gradient - qg[:, :rank] @ coefficients
            residual_p = potential - qp[:, :rank] @ coefficients
        else:
            coefficients = cp.empty(0, dtype=F64)
            residual_v = velocity
            residual_g = gradient
            residual_p = potential
        residual_mass = weight * cp.dot(residual_v, residual_v)
        residual_stiffness = weight * cp.dot(residual_g, residual_g)
        denominator = cp.maximum(residual_mass, cfg.mass_floor)
        rayleigh = residual_stiffness / denominator
        if need_adjoint:
            grad_residual_v = (
                -2.0
                * residual_stiffness
                * weight
                / (
                    cfg.rayleigh_scale
                    * denominator
                    * denominator
                )
            ) * residual_v
            grad_residual_g = (
                2.0
                * weight
                / (cfg.rayleigh_scale * denominator)
            ) * residual_g
            grad_velocity = grad_residual_v
            if rank:
                grad_coefficients = -(
                    qv[:, :rank].T @ grad_residual_v
                    + qg[:, :rank].T @ grad_residual_g
                )
                grad_velocity = (
                    grad_velocity
                    + weight * qv[:, :rank] @ grad_coefficients
                )
            adj_v[:, mode] = grad_velocity
            adj_g[:, mode] = grad_residual_g
        normalizer = 1.0 / cp.sqrt(denominator)
        qv[:, rank] = residual_v * normalizer
        qg[:, rank] = residual_g * normalizer
        qp[:, rank] = residual_p * normalizer
        if coefficient_matrix is not None:
            column = cp.zeros(cfg.modes, dtype=F64)
            column[mode] = 1.0
            if rank:
                column -= coefficient_matrix[:, :rank] @ coefficients
            coefficient_matrix[:, rank] = column * normalizer
        projection = velocity - residual_v
        projection_mass = weight * cp.dot(projection, projection)
        energy_value = float(rayleigh.get())
        energies[rank] = energy_value
        energy_by_network[mode] = energy_value
        projection_ratios[rank] = float(
            (projection_mass / denominator).get()
        )

    gram = weight * (qv.T @ qv)
    orth_error = float(
        cp.max(cp.abs(gram - cp.eye(cfg.modes, dtype=F64))).get()
    )
    raw_gram = weight * (raw_v.T @ raw_v)
    eigenvalues = cp.linalg.eigvalsh(
        0.5 * (raw_gram + raw_gram.T)
    )
    threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
    result: dict[str, Any] = {
        "loss": float(energies.mean() / cfg.rayleigh_scale),
        "mean_energy": float(energies.mean()),
        "max_energy": float(energies.max()),
        "min_energy": float(energies.min()),
        "mean_projection_ratio": float(projection_ratios.mean()),
        "orth_error": orth_error,
        "effective_raw_rank": int(cp.sum(eigenvalues > threshold).get()),
        "mode_order": cp.asnumpy(order).astype(np.int32),
        "mode_energies": energies,
        "mode_energies_by_network": energy_by_network,
        "q_velocity": qv,
    }
    if need_adjoint:
        result["adjoint_velocity"] = adj_v
        result["adjoint_gradient"] = adj_g
    if coefficient_matrix is not None:
        result["coefficient_matrix"] = coefficient_matrix
    return result


def train_step(
    cfg: SimpleNamespace,
    model: HardcodedFanModel,
    points: np.ndarray,
    parameters: np.ndarray,
    learning_rate: float,
) -> dict[str, Any]:
    raw = raw_fields(cfg, model, points, parameters)
    peeled = peel_and_adjoint(
        cfg,
        raw,
        len(points),
        need_adjoint=True,
    )
    operator = stencil_adjoint_matrix(cfg)
    inputs = raw["inputs"]
    envelope = raw["envelope"]
    for mode in range(cfg.modes):
        velocity_adjoint = peeled["adjoint_velocity"][:, mode].reshape(
            3, len(points)
        ).T
        gradient_adjoint = peeled["adjoint_gradient"][:, mode].reshape(
            3, 3, len(points)
        ).transpose(2, 0, 1).reshape(len(points), 9)
        output_adjoint = cp.concatenate(
            (velocity_adjoint, gradient_adjoint),
            axis=1,
        )
        stencil_adjoint = (output_adjoint @ operator).reshape(
            len(points), len(STENCIL_OFFSETS), 3
        ).transpose(1, 0, 2).reshape(-1, 3)
        _, cache = model.forward_single(
            mode,
            inputs,
            envelope,
            cache=True,
        )
        if cache is None:
            raise RuntimeError("manual forward cache missing")
        gradients = model.backward_single(
            mode, cache, stencil_adjoint
        )
        model.adam_step(mode, gradients, learning_rate)
    return {
        key: value
        for key, value in peeled.items()
        if key
        not in {
            "adjoint_velocity",
            "adjoint_gradient",
            "q_velocity",
        }
    }


def inference_basis_mode(
    cfg: SimpleNamespace,
    model: HardcodedFanModel,
    quadrature_points: np.ndarray,
    visual_points: np.ndarray,
    parameters: np.ndarray,
    rank: int,
) -> dict[str, np.ndarray | float | int]:
    if not 0 <= rank < cfg.modes:
        raise ValueError("rank outside trained mode range")
    quadrature_raw = raw_fields(
        cfg, model, quadrature_points, parameters
    )
    peeled = peel_and_adjoint(
        cfg,
        quadrature_raw,
        len(quadrature_points),
        need_adjoint=False,
        need_coefficients=True,
    )
    coefficients = peeled["coefficient_matrix"][:, rank]
    visual_raw = raw_velocity(
        cfg, model, visual_points, parameters
    )
    velocity = visual_raw @ coefficients
    return {
        "velocity": cp.asnumpy(velocity.reshape(3, len(visual_points)).T),
        "mode_network": int(peeled["mode_order"][rank]),
        "rayleigh_energy": float(peeled["mode_energies"][rank]),
        "orth_error": float(peeled["orth_error"]),
    }


def raw_velocity(
    cfg: SimpleNamespace,
    model: HardcodedFanModel,
    points: np.ndarray,
    parameters: np.ndarray,
) -> cp.ndarray:
    """Seven-point value-only velocity map for browser inference."""
    offsets = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    shifted = np.concatenate(
        [
            points
            + np.asarray(offset, dtype=np.float32)[None, :]
            * cfg.difference_step
            for offset in offsets
        ],
        axis=0,
    )
    shifted_device = cp.asarray(shifted, dtype=F32)
    parameter_device = cp.asarray(parameters, dtype=F32)
    inputs = encoded_inputs(cfg, shifted_device, parameter_device)
    envelope = boundary_envelope(
        cfg, shifted_device, parameter_device
    )
    velocity = cp.empty((3 * len(points), cfg.modes), dtype=F64)
    block = max(1, int(cfg.mode_forward_batch))
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        evaluated = model.forward_block(
            inputs, envelope, start, stop
        ).reshape(stop - start, len(offsets), len(points), 3)
        dx = (evaluated[:, 1] - evaluated[:, 0]) / (
            2.0 * cfg.difference_step
        )
        dy = (evaluated[:, 3] - evaluated[:, 2]) / (
            2.0 * cfg.difference_step
        )
        dz = (evaluated[:, 5] - evaluated[:, 4]) / (
            2.0 * cfg.difference_step
        )
        block_velocity = cp.stack(
            (
                dy[:, :, 0] + dz[:, :, 1],
                -dx[:, :, 0] + dz[:, :, 2],
                -dx[:, :, 1] - dy[:, :, 2],
            ),
            axis=2,
        ).astype(F64)
        velocity[:, start:stop] = block_velocity.transpose(
            2, 1, 0
        ).reshape(3 * len(points), stop - start)
    return velocity
