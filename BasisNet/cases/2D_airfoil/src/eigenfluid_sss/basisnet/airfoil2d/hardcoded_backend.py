"""Fixed-graph CuPy backend for the 2D airfoil SSS trainer.

This module deliberately has no PyTorch dependency and performs no automatic
differentiation.  Spatial jets through second order and the reverse pass with
respect to every MLP parameter are written out explicitly.
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


@dataclass
class Jet2:
    value: cp.ndarray
    grad: cp.ndarray
    hess: cp.ndarray

    @staticmethod
    def constant(
        value: cp.ndarray | float,
        shape: tuple[int, int],
        dtype: type[cp.generic] = F32,
    ) -> "Jet2":
        v = cp.broadcast_to(cp.asarray(value, dtype=dtype), shape)
        return Jet2(
            v,
            cp.zeros((*shape, 2), dtype=dtype),
            cp.zeros((*shape, 2, 2), dtype=dtype),
        )

    def __add__(self, other: "Jet2" | float) -> "Jet2":
        other = as_jet(other, self.value.shape, self.value.dtype.type)
        return Jet2(self.value + other.value, self.grad + other.grad, self.hess + other.hess)

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, -self.grad, -self.hess)

    def __sub__(self, other: "Jet2" | float) -> "Jet2":
        return self + (-as_jet(other, self.value.shape, self.value.dtype.type))

    def __rsub__(self, other: "Jet2" | float) -> "Jet2":
        return as_jet(other, self.value.shape, self.value.dtype.type) - self

    def __mul__(self, other: "Jet2" | float) -> "Jet2":
        other = as_jet(other, self.value.shape, self.value.dtype.type)
        av, bv = self.value, other.value
        grad = self.grad * bv[..., None] + other.grad * av[..., None]
        hess = (
            self.hess * bv[..., None, None]
            + other.hess * av[..., None, None]
            + self.grad[..., :, None] * other.grad[..., None, :]
            + other.grad[..., :, None] * self.grad[..., None, :]
        )
        return Jet2(av * bv, grad, hess)

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet2":
        inv = 1.0 / self.value
        first = -(inv * inv)
        second = 2.0 * inv * inv * inv
        hess = (
            first[..., None, None] * self.hess
            + second[..., None, None]
            * self.grad[..., :, None]
            * self.grad[..., None, :]
        )
        return Jet2(inv, first[..., None] * self.grad, hess)

    def __truediv__(self, other: "Jet2" | float) -> "Jet2":
        return self * as_jet(
            other, self.value.shape, self.value.dtype.type
        ).reciprocal()

    def square(self) -> "Jet2":
        return self * self

    def pow(self, exponent: int) -> "Jet2":
        result = as_jet(1.0, self.value.shape, self.value.dtype.type)
        for _ in range(exponent):
            result = result * self
        return result


def as_jet(
    value: Jet2 | float,
    shape: tuple[int, int],
    dtype: type[cp.generic] = F32,
) -> Jet2:
    return value if isinstance(value, Jet2) else Jet2.constant(value, shape, dtype)


def unary(jet: Jet2, value: cp.ndarray, first: cp.ndarray, second: cp.ndarray) -> Jet2:
    return Jet2(
        value,
        first[..., None] * jet.grad,
        first[..., None, None] * jet.hess
        + second[..., None, None] * jet.grad[..., :, None] * jet.grad[..., None, :],
    )


def jet_sqrt(jet: Jet2) -> Jet2:
    value = cp.sqrt(jet.value)
    first = 0.5 / value
    second = -0.25 / (value * value * value)
    return unary(jet, value, first, second)


def jet_abs(jet: Jet2) -> Jet2:
    sign = cp.sign(jet.value)
    return Jet2(cp.abs(jet.value), sign[..., None] * jet.grad, sign[..., None, None] * jet.hess)


def jet_select(mask: cp.ndarray, left: Jet2, right: Jet2) -> Jet2:
    return Jet2(
        cp.where(mask, left.value, right.value),
        cp.where(mask[..., None], left.grad, right.grad),
        cp.where(mask[..., None, None], left.hess, right.hess),
    )


def jet_maximum(left: Jet2, right: Jet2) -> Jet2:
    return jet_select(left.value >= right.value, left, right)


def jet_clamp(jet: Jet2, low: float, high: float | None = None) -> Jet2:
    inside = jet.value >= low
    if high is not None:
        inside &= jet.value <= high
    value = cp.maximum(jet.value, low)
    if high is not None:
        value = cp.minimum(value, high)
    return Jet2(
        value,
        cp.where(inside[..., None], jet.grad, 0.0),
        cp.where(inside[..., None, None], jet.hess, 0.0),
    )


def xy_jets(
    points: np.ndarray, dtype: type[cp.generic] = F32
) -> tuple[Jet2, Jet2]:
    xy = cp.asarray(points, dtype=dtype)
    count = len(points)
    zero_hess = cp.zeros((count, 1, 2, 2), dtype=dtype)
    gx = cp.zeros((count, 1, 2), dtype=dtype)
    gy = cp.zeros_like(gx)
    gx[:, 0, 0] = 1.0
    gy[:, 0, 1] = 1.0
    return (
        Jet2(xy[:, 0:1], gx, zero_hess.copy()),
        Jet2(xy[:, 1:2], gy, zero_hess),
    )


def body_jets(cfg: SimpleNamespace, x: Jet2, y: Jet2, alpha_deg: float) -> tuple[Jet2, Jet2]:
    angle = math.radians(-alpha_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = x - cfg.quarter_chord_x
    dy = y - cfg.quarter_chord_y
    return (
        cosine * dx + sine * dy + 0.25 * cfg.chord,
        -sine * dx + cosine * dy,
    )


def naca_half_thickness_jet(cfg: SimpleNamespace, xb: Jet2, beta: float) -> Jet2:
    s = jet_clamp(xb / cfg.chord, 0.0, 1.0)
    epsilon = 1.0e-6
    root = (jet_sqrt(s + epsilon) - math.sqrt(epsilon)) / (
        math.sqrt(1.0 + epsilon) - math.sqrt(epsilon)
    )
    polynomial = (
        0.2969 * root
        - 0.1260 * s
        - 0.3516 * s.square()
        + 0.2843 * s.pow(3)
        - 0.1036 * s.pow(4)
    )
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    return 5.0 * thickness * cfg.chord * polynomial


def c2_airfoil_implicit_jet(
    cfg: SimpleNamespace, xb: Jet2, yb: Jet2, beta: float
) -> Jet2:
    """C-infinity ellipse implicit function; zero is the airfoil boundary."""
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )


def c2_positive_envelope(level: Jet2) -> Jet2:
    """q_+^3/(1+q_+^3) is C2 across q=0 and saturates away from it."""
    zero = Jet2.constant(0.0, level.value.shape, level.value.dtype.type)
    positive = jet_maximum(level, zero)
    cubic = positive.pow(3)
    return cubic / (1.0 + cubic)


def envelope_jet(
    cfg: SimpleNamespace, x: Jet2, y: Jet2, beta: float, alpha_deg: float
) -> Jet2:
    scale = 0.25 * cfg.chord
    inlet = c2_positive_envelope((x - cfg.xmin) / scale)
    xb, yb = body_jets(cfg, x, y, alpha_deg)
    level = c2_airfoil_implicit_jet(cfg, xb, yb, beta)
    airfoil = c2_positive_envelope(level)
    envelope = inlet * airfoil
    if cfg.envelope_mode in ("farfield_freestream", "closed_channel"):
        yn = (
            2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin)
            - 1.0
        )
        # This is a double-zero envelope for the *perturbation* streamfunction.
        # The reconstructed total field still contains the uniform free stream,
        # so y=ymin/ymax are far-field free-stream boundaries, not no-slip walls.
        farfield = c2_positive_envelope((1.0 - yn.pow(2)) / 0.25)
        envelope = envelope * farfield
    elif cfg.envelope_mode != "localized":
        raise ValueError(f"Unknown envelope mode: {cfg.envelope_mode}")
    return envelope


def encoding_jet(
    cfg: SimpleNamespace, x: Jet2, y: Jet2, beta: float, alpha_deg: float
) -> Jet2:
    xn = 2.0 * (x - cfg.xmin) / (cfg.xmax - cfg.xmin) - 1.0
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    features: list[Jet2] = []
    for spatial in (xn, yn):
        for frequency in 2.0 ** np.arange(cfg.frequencies):
            phase = spatial * float(frequency)
            features.append(
                unary(phase, cp.sin(phase.value), cp.cos(phase.value), -cp.sin(phase.value))
            )
    for spatial in (xn, yn):
        for frequency in 2.0 ** np.arange(cfg.frequencies):
            phase = spatial * float(frequency)
            features.append(
                unary(phase, cp.cos(phase.value), -cp.sin(phase.value), -cp.cos(phase.value))
            )
    shape = x.value.shape
    features.extend(
        [
            Jet2.constant(2.0 * beta - 1.0, shape, x.value.dtype.type),
            Jet2.constant(
                2.0 * (alpha_deg - cfg.alpha_min_deg)
                / (cfg.alpha_max_deg - cfg.alpha_min_deg)
                - 1.0,
                shape,
                x.value.dtype.type,
            ),
        ]
    )
    xb, yb = body_jets(cfg, x, y, alpha_deg)
    features.extend([2.0 * xb / cfg.chord - 1.0, 2.0 * yb / cfg.chord])
    return Jet2(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
        cp.concatenate([item.hess for item in features], axis=1),
    )


def _compute_dtype(cfg: SimpleNamespace) -> type[cp.generic]:
    name = str(getattr(cfg, "mlp_compute_dtype", "float32")).lower()
    if name == "float16":
        return F16
    if name == "float32":
        return F32
    raise ValueError(f"Unsupported MLP compute dtype: {name}")


def _matmul(
    left: cp.ndarray, right: cp.ndarray, compute_dtype: type[cp.generic]
) -> cp.ndarray:
    if compute_dtype is F16:
        return cp.matmul(
            left.astype(F16, copy=False),
            right.astype(F16, copy=False),
            dtype=F32,
        )
    return left @ right


def _einsum(
    expression: str,
    *operands: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> cp.ndarray:
    if compute_dtype is F16:
        return cp.einsum(
            expression,
            *(operand.astype(F16, copy=False) for operand in operands),
            dtype=F32,
        )
    return cp.einsum(expression, *operands)


def linear_forward(
    source: Jet2,
    weight: cp.ndarray,
    bias: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet2:
    count = source.value.shape[0]
    packed = cp.concatenate(
        (
            source.value,
            source.grad[..., 0],
            source.grad[..., 1],
            source.hess[..., 0, 0],
            source.hess[..., 0, 1],
            source.hess[..., 1, 0],
            source.hess[..., 1, 1],
        ),
        axis=0,
    )
    projected = _matmul(packed, weight.T, compute_dtype).reshape(
        7, count, weight.shape[0]
    )
    return Jet2(
        projected[0] + bias,
        cp.stack((projected[1], projected[2]), axis=-1),
        cp.stack(
            (
                cp.stack((projected[3], projected[4]), axis=-1),
                cp.stack((projected[5], projected[6]), axis=-1),
            ),
            axis=-2,
        ),
    )


def _slice_jet(source: Jet2, index: int) -> Jet2:
    return Jet2(source.value[index], source.grad[index], source.hess[index])


def _expand_mode_axis(source: Jet2) -> Jet2:
    return Jet2(source.value[None], source.grad[None], source.hess[None])


def linear_forward_batch(
    source: Jet2,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet2:
    """Project seven Jet2 channels for several modes in one batched GEMM."""
    batched_source = source.value.ndim == 3
    point_axis = 1 if batched_source else 0
    count = source.value.shape[point_axis]
    packed = cp.concatenate(
        (
            source.value,
            source.grad[..., 0],
            source.grad[..., 1],
            source.hess[..., 0, 0],
            source.hess[..., 0, 1],
            source.hess[..., 1, 0],
            source.hess[..., 1, 1],
        ),
        axis=point_axis,
    ).astype(compute_dtype, copy=False)
    projected = cp.matmul(
        packed,
        weights.transpose(0, 2, 1).astype(compute_dtype, copy=False),
        dtype=F32,
    )
    projected = projected.reshape(
        weights.shape[0], 7, count, weights.shape[1]
    )
    return Jet2(
        projected[:, 0] + biases[:, None, :],
        cp.stack((projected[:, 1], projected[:, 2]), axis=-1),
        cp.stack(
            (
                cp.stack((projected[:, 3], projected[:, 4]), axis=-1),
                cp.stack((projected[:, 5], projected[:, 6]), axis=-1),
            ),
            axis=-2,
        ),
    )


def sine_forward(source: Jet2) -> Jet2:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    return Jet2(
        sine,
        cosine[..., None] * source.grad,
        cosine[..., None, None] * source.hess
        - sine[..., None, None]
        * source.grad[..., :, None]
        * source.grad[..., None, :],
    )


def linear_backward(
    source: Jet2,
    adjoint: Jet2,
    weight: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> tuple[Jet2, cp.ndarray, cp.ndarray]:
    grad_weight = _matmul(adjoint.value.T, source.value, compute_dtype)
    grad_weight += _einsum(
        "noi,nfi->of", adjoint.grad, source.grad, compute_dtype=compute_dtype
    )
    grad_weight += _einsum(
        "noij,nfij->of", adjoint.hess, source.hess, compute_dtype=compute_dtype
    )
    grad_bias = adjoint.value.sum(axis=0)
    return (
        Jet2(
            _matmul(adjoint.value, weight, compute_dtype),
            _einsum(
                "noi,of->nfi", adjoint.grad, weight, compute_dtype=compute_dtype
            ),
            _einsum(
                "noij,of->nfij", adjoint.hess, weight, compute_dtype=compute_dtype
            ),
        ),
        grad_weight,
        grad_bias,
    )


def sine_backward(source: Jet2, adjoint: Jet2) -> Jet2:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    value = adjoint.value * cosine
    value += cp.sum(adjoint.grad * (-sine[..., None] * source.grad), axis=-1)
    value += cp.sum(
        adjoint.hess
        * (
            -sine[..., None, None] * source.hess
            - cosine[..., None, None]
            * source.grad[..., :, None]
            * source.grad[..., None, :]
        ),
        axis=(-2, -1),
    )
    grad = adjoint.grad * cosine[..., None]
    grad += cp.einsum("nfij,nfj->nfi", adjoint.hess, -sine[..., None] * source.grad)
    grad += cp.einsum("nfji,nfj->nfi", adjoint.hess, -sine[..., None] * source.grad)
    hess = adjoint.hess * cosine[..., None, None]
    return Jet2(value, grad, hess)


def envelope_product_backward(envelope: Jet2, adjoint: Jet2) -> Jet2:
    value = adjoint.value * envelope.value
    value += cp.sum(adjoint.grad * envelope.grad, axis=-1)
    value += cp.sum(adjoint.hess * envelope.hess, axis=(-2, -1))
    grad = adjoint.grad * envelope.value[..., None]
    grad += cp.einsum("nfij,nfi->nfj", adjoint.hess, envelope.grad)
    grad += cp.einsum("nfij,nfj->nfi", adjoint.hess, envelope.grad)
    hess = adjoint.hess * envelope.value[..., None, None]
    return Jet2(value, grad, hess)


class HardcodedMode:
    def __init__(
        self,
        arrays: dict[str, cp.ndarray],
        mode: int,
        layers: int,
        compute_dtype: type[cp.generic],
    ):
        prefix = f"m{mode:03d}"
        self.compute_dtype = compute_dtype
        self.weights = [arrays[f"{prefix}.l{index}.weight"] for index in range(layers)]
        self.biases = [arrays[f"{prefix}.l{index}.bias"] for index in range(layers)]
        self.weights.append(arrays[f"{prefix}.output.weight"])
        self.biases.append(arrays[f"{prefix}.output.bias"])
        self.shadow_weights = self.weights
        self.shadow_biases = self.biases
        parameter_count = 2 * (layers + 1)
        self.exp_avg = [
            arrays[f"{prefix}.adam.p{index:02d}.exp_avg"]
            for index in range(parameter_count)
        ]
        self.exp_avg_sq = [
            arrays[f"{prefix}.adam.p{index:02d}.exp_avg_sq"]
            for index in range(parameter_count)
        ]
        self.steps = [
            int(arrays[f"{prefix}.adam.p{index:02d}.step"])
            for index in range(parameter_count)
        ]

    def forward(
        self, cfg: SimpleNamespace, points: np.ndarray, beta: float, alpha: float, cache: bool
    ) -> tuple[Jet2, Any]:
        x, y = xy_jets(points)
        envelope = envelope_jet(cfg, x, y, beta, alpha)
        current = encoding_jet(cfg, x, y, beta, alpha)
        layer_cache: list[tuple[Jet2, Jet2]] = []
        for layer_index in range(len(self.weights) - 1):
            before = current
            preactivation = linear_forward(
                before,
                self.shadow_weights[layer_index],
                self.shadow_biases[layer_index],
                self.compute_dtype,
            )
            current = sine_forward(preactivation)
            if cache:
                layer_cache.append((before, preactivation))
        output_input = current
        raw = linear_forward(
            output_input,
            self.shadow_weights[-1],
            self.shadow_biases[-1],
            self.compute_dtype,
        )
        field = envelope * raw
        return field, (envelope, output_input, layer_cache) if cache else None

    def backward(self, cache: Any, field_adjoint: Jet2) -> list[cp.ndarray]:
        envelope, output_input, layer_cache = cache
        raw_adjoint = envelope_product_backward(envelope, field_adjoint)
        backward_dtype = F32
        current, output_weight, output_bias = linear_backward(
            output_input,
            raw_adjoint,
            self.weights[-1],
            backward_dtype,
        )
        weight_grads = [output_weight]
        bias_grads = [output_bias]
        for index in range(len(layer_cache) - 1, -1, -1):
            source, preactivation = layer_cache[index]
            current = sine_backward(preactivation, current)
            current, grad_weight, grad_bias = linear_backward(
                source,
                current,
                self.weights[index],
                backward_dtype,
            )
            weight_grads.append(grad_weight)
            bias_grads.append(grad_bias)
        weight_grads.reverse()
        bias_grads.reverse()
        result: list[cp.ndarray] = []
        for weight, bias in zip(weight_grads, bias_grads):
            result.extend([weight, bias])
        return result

    def adam_step(self, gradients: list[cp.ndarray], cfg: SimpleNamespace) -> None:
        if len(gradients) != 2 * len(self.weights):
            raise RuntimeError("hardcoded reverse pass returned the wrong gradient count")
        for index, gradient in enumerate(gradients):
            if not bool(cp.all(cp.isfinite(gradient)).get()):
                raise FloatingPointError(f"non-finite hardcoded gradient tensor {index}")
            if float(cp.max(cp.abs(gradient)).get()) == 0.0:
                raise FloatingPointError(f"zero hardcoded gradient tensor {index}")
        total = sum(cp.sum(item.astype(F64) ** 2) for item in gradients)
        norm = float(cp.sqrt(total).get())
        scale = min(1.0, cfg.grad_clip / max(norm, 1.0e-30))
        parameters: list[cp.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            parameters.extend([weight, bias])
        beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            gradient = gradient.astype(F32) * scale
            self.steps[index] += 1
            self.exp_avg[index] *= beta1
            self.exp_avg[index] += (1.0 - beta1) * gradient
            self.exp_avg_sq[index] *= beta2
            self.exp_avg_sq[index] += (1.0 - beta2) * gradient * gradient
            correction1 = 1.0 - beta1 ** self.steps[index]
            correction2 = 1.0 - beta2 ** self.steps[index]
            denominator = cp.sqrt(self.exp_avg_sq[index] / correction2) + epsilon
            parameter -= (cfg.learning_rate / correction1) * self.exp_avg[index] / denominator
        if self.compute_dtype is F16:
            for shadow, master in zip(self.shadow_weights, self.weights):
                shadow[...] = master
            for shadow, master in zip(self.shadow_biases, self.biases):
                shadow[...] = master


class HardcodedModel:
    def __init__(self, cfg: SimpleNamespace, arrays: dict[str, np.ndarray]):
        device_arrays = {name: cp.asarray(value) for name, value in arrays.items()}
        compute_dtype = _compute_dtype(cfg)
        self.modes = [
            HardcodedMode(
                device_arrays, mode, cfg.hidden_layers, compute_dtype
            )
            for mode in range(cfg.modes)
        ]
        layer_count = cfg.hidden_layers + 1
        shadow_dtype = F16 if compute_dtype is F16 else F32
        self.shadow_weight_stacks = [
            cp.stack([mode.weights[layer] for mode in self.modes]).astype(
                shadow_dtype
            )
            for layer in range(layer_count)
        ]
        self.shadow_bias_stacks = [
            cp.stack([mode.biases[layer] for mode in self.modes]).astype(
                shadow_dtype
            )
            for layer in range(layer_count)
        ]
        for mode_index, mode in enumerate(self.modes):
            mode.shadow_weights = [
                stack[mode_index] for stack in self.shadow_weight_stacks
            ]
            mode.shadow_biases = [
                stack[mode_index] for stack in self.shadow_bias_stacks
            ]

    def forward_stream(
        self,
        cfg: SimpleNamespace,
        points: np.ndarray,
        beta: float,
        alpha: float,
        cache: bool,
    ):
        block_size = max(1, int(getattr(cfg, "mode_forward_batch", 1)))
        for start in range(0, len(self.modes), block_size):
            stop = min(start + block_size, len(self.modes))
            x, y = xy_jets(points)
            envelope = envelope_jet(cfg, x, y, beta, alpha)
            current = encoding_jet(cfg, x, y, beta, alpha)
            layer_cache: list[tuple[Jet2, Jet2]] = []
            for layer in range(cfg.hidden_layers):
                before = current
                preactivation = linear_forward_batch(
                    before,
                    self.shadow_weight_stacks[layer][start:stop],
                    self.shadow_bias_stacks[layer][start:stop],
                    self.modes[start].compute_dtype,
                )
                current = sine_forward(preactivation)
                if cache:
                    layer_cache.append((before, preactivation))
            output_input = current
            raw = linear_forward_batch(
                output_input,
                self.shadow_weight_stacks[-1][start:stop],
                self.shadow_bias_stacks[-1][start:stop],
                self.modes[start].compute_dtype,
            )
            field = _expand_mode_axis(envelope) * raw
            for local, mode_index in enumerate(range(start, stop)):
                if cache:
                    individual_layers = []
                    for source, preactivation in layer_cache:
                        individual_source = (
                            source
                            if source.value.ndim == 2
                            else _slice_jet(source, local)
                        )
                        individual_layers.append(
                            (individual_source, _slice_jet(preactivation, local))
                        )
                    individual_cache = (
                        envelope,
                        _slice_jet(output_input, local),
                        individual_layers,
                    )
                else:
                    individual_cache = None
                yield (
                    self.modes[mode_index],
                    _slice_jet(field, local),
                    individual_cache,
                )

    def arrays(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for mode_index, mode in enumerate(self.modes):
            prefix = f"m{mode_index:03d}"
            for index in range(len(mode.weights) - 1):
                result[f"{prefix}.l{index}.weight"] = cp.asnumpy(mode.weights[index])
                result[f"{prefix}.l{index}.bias"] = cp.asnumpy(mode.biases[index])
            result[f"{prefix}.output.weight"] = cp.asnumpy(mode.weights[-1])
            result[f"{prefix}.output.bias"] = cp.asnumpy(mode.biases[-1])
            for state_index in range(2 * len(mode.weights)):
                stem = f"{prefix}.adam.p{state_index:02d}"
                result[f"{stem}.exp_avg"] = cp.asnumpy(mode.exp_avg[state_index])
                result[f"{stem}.exp_avg_sq"] = cp.asnumpy(mode.exp_avg_sq[state_index])
                result[f"{stem}.step"] = np.asarray(mode.steps[state_index], dtype=np.int64)
        return result


def cfg_namespace(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def naca_half_thickness_numpy(cfg: SimpleNamespace, s: np.ndarray, beta: float) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    epsilon = 1.0e-6
    root = (np.sqrt(s + epsilon) - math.sqrt(epsilon)) / (
        math.sqrt(1.0 + epsilon) - math.sqrt(epsilon)
    )
    polynomial = 0.2969 * root - 0.1260 * s - 0.3516 * s**2 + 0.2843 * s**3 - 0.1036 * s**4
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    return 5.0 * thickness * cfg.chord * polynomial


def c2_airfoil_implicit_numpy(
    cfg: SimpleNamespace, points: np.ndarray, beta: float, alpha: float
) -> np.ndarray:
    xb, yb = body_numpy(cfg, points, alpha)
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord) ** 2
        + (yb / semi_thickness) ** 2
        - 1.0
    )


def body_numpy(cfg: SimpleNamespace, points: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(-alpha)
    c, s = math.cos(angle), math.sin(angle)
    dx = points[:, 0] - cfg.quarter_chord_x
    dy = points[:, 1] - cfg.quarter_chord_y
    return c * dx + s * dy + 0.25 * cfg.chord, -s * dx + c * dy


def inside_airfoil(cfg: SimpleNamespace, points: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    return c2_airfoil_implicit_numpy(cfg, points, beta, alpha) <= 0.0


def _sobol_box(
    cfg: SimpleNamespace, count: int, bounds: tuple[float, float, float, float],
    seed: int, beta: float, alpha: float
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    remaining = count
    sampler = qmc.Sobol(2, scramble=True, seed=seed)
    while remaining:
        draw_count = max(3 * remaining, 128)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol",
                category=UserWarning,
            )
            sample = sampler.random(draw_count)
        xmin, xmax, ymin, ymax = bounds
        candidates = np.column_stack(
            (xmin + sample[:, 0] * (xmax - xmin), ymin + sample[:, 1] * (ymax - ymin))
        )
        accepted = candidates[~inside_airfoil(cfg, candidates, beta, alpha)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces)


def fluid_points(
    cfg: SimpleNamespace, count: int, beta: float, alpha: float, seed: int
) -> np.ndarray:
    fractions = np.asarray([cfg.global_fraction, cfg.airfoil_fraction, cfg.wake_fraction])
    raw = fractions / fractions.sum() * count
    counts = np.floor(raw).astype(int)
    counts[np.argsort(-(raw - counts))[: count - counts.sum()]] += 1
    half_x = 0.5 * cfg.chord + cfg.airfoil_band_chords * cfg.chord
    half_y = 0.5 * cfg.chord
    center_x = cfg.quarter_chord_x + 0.25 * cfg.chord
    angle = math.radians(-alpha)
    tx = cfg.quarter_chord_x + math.cos(angle) * 0.75 * cfg.chord
    ty = cfg.quarter_chord_y + math.sin(angle) * 0.75 * cfg.chord
    boxes = [
        (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
        (max(cfg.xmin, center_x - half_x), min(cfg.xmax, center_x + half_x),
         max(cfg.ymin, -half_y), min(cfg.ymax, half_y)),
        (min(cfg.xmax - 1.0e-6, tx + 0.02 * cfg.chord),
         min(cfg.xmax, tx + cfg.wake_length_chords * cfg.chord),
         max(cfg.ymin, ty - cfg.wake_half_height_chords * cfg.chord),
         min(cfg.ymax, ty + cfg.wake_half_height_chords * cfg.chord)),
    ]
    pieces = [
        _sobol_box(cfg, int(n), box, seed + index, beta, alpha)
        for index, (n, box) in enumerate(zip(counts, boxes))
    ]
    points = np.concatenate(pieces)
    return points[np.random.default_rng(seed).permutation(len(points))]


def continuous_parameters(cfg: SimpleNamespace, epoch: int) -> tuple[float, float]:
    if cfg.endpoint_period > 0 and epoch % cfg.endpoint_period == 0:
        corners = (
            (0.0, cfg.alpha_min_deg), (0.0, cfg.alpha_max_deg),
            (1.0, cfg.alpha_min_deg), (1.0, cfg.alpha_max_deg),
        )
        return corners[(epoch // cfg.endpoint_period - 1) % len(corners)]
    sample = qmc.Sobol(2, scramble=True, seed=cfg.seed + 700_000 + epoch).random(1)[0]
    return float(sample[0]), float(
        cfg.alpha_min_deg + sample[1] * (cfg.alpha_max_deg - cfg.alpha_min_deg)
    )


def airfoil_area(cfg: SimpleNamespace, beta: float) -> float:
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    return float(math.pi * 0.5 * cfg.chord * 0.5 * thickness * cfg.chord)


def peel_step(
    cfg: SimpleNamespace, model: HardcodedModel, points: np.ndarray,
    beta: float, alpha: float, training: bool
) -> dict[str, Any]:
    weight = ((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(cfg, beta)) / len(points)
    previous_v: list[cp.ndarray] = []
    previous_g: list[cp.ndarray] = []
    raw_velocities: list[cp.ndarray] = []
    energies: list[float] = []
    projections: list[float] = []
    for mode, field, cache in model.forward_stream(
        cfg, points, beta, alpha, training
    ):
        psi_x, psi_y = field.grad[:, 0, 0], field.grad[:, 0, 1]
        velocity = cp.concatenate((psi_y, -psi_x)).astype(F64)
        raw_velocities.append(velocity)
        h = field.hess[:, 0]
        gradient = cp.concatenate((h[:, 1, 0], h[:, 1, 1], -h[:, 0, 0], -h[:, 0, 1])).astype(F64)
        if previous_v:
            u = cp.stack(previous_v, axis=1)
            g = cp.stack(previous_g, axis=1)
            lhs = weight * (u.T @ u)
            rhs = weight * (u.T @ velocity)
            coefficients = cp.linalg.solve(lhs, rhs)
            reconstructed_v = u @ coefficients
            reconstructed_g = g @ coefficients
        else:
            u = g = None
            lhs = coefficients = None
            reconstructed_v = cp.zeros_like(velocity)
            reconstructed_g = cp.zeros_like(gradient)
        residual_v = velocity - reconstructed_v
        residual_g = gradient - reconstructed_g
        mass = weight * cp.dot(residual_v, residual_v)
        stiffness = weight * cp.dot(residual_g, residual_g)
        denominator = cp.maximum(mass, cfg.mass_floor)
        rayleigh = stiffness / denominator
        projection_mass = weight * cp.dot(reconstructed_v, reconstructed_v)
        if training:
            grad_residual_v = (
                -2.0 * stiffness * weight / (cfg.rayleigh_scale * denominator * denominator)
            ) * residual_v
            grad_residual_g = (
                2.0 * weight / (cfg.rayleigh_scale * denominator)
            ) * residual_g
            grad_velocity = grad_residual_v
            if previous_v:
                grad_q = -(u.T @ grad_residual_v) - (g.T @ grad_residual_g)
                grad_velocity = grad_velocity + weight * u @ cp.linalg.solve(lhs.T, grad_q)
            n = len(points)
            adjoint_dtype = (
                F16
                if bool(getattr(cfg, "all_network_fp16", False))
                else F32
            )
            if adjoint_dtype is F16:
                configured_scale = float(getattr(cfg, "fp16_loss_scale", 128.0))
                raw_adjoint_max = max(
                    float(cp.max(cp.abs(grad_velocity)).get()),
                    float(cp.max(cp.abs(grad_residual_g)).get()),
                    1.0e-30,
                )
                loss_scale = min(configured_scale, 1.0e-4 / raw_adjoint_max)
            else:
                loss_scale = 1.0
            adjoint = Jet2(
                cp.zeros((n, 1), dtype=adjoint_dtype),
                cp.zeros((n, 1, 2), dtype=adjoint_dtype),
                cp.zeros((n, 1, 2, 2), dtype=adjoint_dtype),
            )
            adjoint.grad[:, 0, 1] = (
                grad_velocity[:n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.grad[:, 0, 0] = (
                -grad_velocity[n:] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:, 0, 1, 0] = (
                grad_residual_g[:n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:, 0, 1, 1] = (
                grad_residual_g[n : 2 * n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:, 0, 0, 0] = (
                -grad_residual_g[2 * n : 3 * n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:, 0, 0, 1] = (
                -grad_residual_g[3 * n :] * loss_scale
            ).astype(adjoint_dtype)
            gradients = mode.backward(cache, adjoint)
            if (
                loss_scale != 1.0
                and not bool(getattr(cfg, "all_network_fp16", False))
            ):
                gradients = [gradient / loss_scale for gradient in gradients]
            mode.adam_step(gradients, cfg)
        normalizer = 1.0 / cp.sqrt(denominator)
        previous_v.append(residual_v * normalizer)
        previous_g.append(residual_g * normalizer)
        energies.append(float(rayleigh.get()))
        projections.append(float((projection_mass / denominator).get()))
    matrix = cp.stack(previous_v, axis=1)
    gram = weight * matrix.T @ matrix
    orth_error = float(cp.max(cp.abs(gram - cp.eye(cfg.modes, dtype=F64))).get())
    raw_matrix = cp.stack(raw_velocities, axis=1)
    raw_gram = weight * raw_matrix.T @ raw_matrix
    eigenvalues = cp.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
    effective_rank = int(cp.sum(eigenvalues > threshold).get())
    return {
        "loss": float(np.mean(energies) / cfg.rayleigh_scale),
        "mean_energy": float(np.mean(energies)),
        "max_energy": float(np.max(energies)),
        "min_energy": float(np.min(energies)),
        "mean_projection_ratio": float(np.mean(projections)),
        "orth_error": orth_error,
        "effective_raw_rank": effective_rank,
        "mode_energies": energies,
    }
