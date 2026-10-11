"""Fixed-graph CuPy backend for the 2D analytic eigenbasis trainer.

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
    if name == "float64":
        return F64
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
    accumulation_dtype = F32 if compute_dtype is F16 else compute_dtype
    projected = cp.matmul(
        packed,
        weights.transpose(0, 2, 1).astype(compute_dtype, copy=False),
        dtype=accumulation_dtype,
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
        # The published FP16 protocol keeps coordinate/boundary jets in FP32
        # and uses FP16 only for hidden-layer shadows.  Casting coordinates
        # themselves to FP16 can overflow second derivatives on the 10x8 box
        # and makes otherwise finite archived checkpoints produce NaNs.
        coordinate_dtype = F32 if self.compute_dtype is F16 else self.compute_dtype
        x, y = xy_jets(points, coordinate_dtype)
        envelope = envelope_jet(cfg, x, y, beta, alpha)
        current = encoding_jet(cfg, x, y, beta, alpha)
        layer_cache: list[tuple[Jet2, Jet2]] = []
        for layer_index in range(len(self.weights) - 1):
            before = current
            preactivation = linear_forward(
                before,
                self.shadow_weights[layer_index],
                self.shadow_biases[layer_index],
                (
                    F32
                    if bool(getattr(cfg, "fp32_boundary_layers", False))
                    and layer_index == 0
                    else self.compute_dtype
                ),
            )
            current = sine_forward(preactivation)
            if cache:
                layer_cache.append((before, preactivation))
        output_input = current
        raw = linear_forward(
            output_input,
            self.shadow_weights[-1],
            self.shadow_biases[-1],
            (
                F32
                if bool(getattr(cfg, "fp32_boundary_layers", False))
                else self.compute_dtype
            ),
        )
        field = envelope * raw
        return field, (envelope, output_input, layer_cache) if cache else None

    def backward(self, cache: Any, field_adjoint: Jet2) -> list[cp.ndarray]:
        envelope, output_input, layer_cache = cache
        raw_adjoint = envelope_product_backward(envelope, field_adjoint)
        backward_dtype = F64 if self.compute_dtype is F64 else F32
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
        allow_zero_tensors = bool(
            getattr(cfg, "allow_zero_gradient_tensors", False)
        )
        for index, gradient in enumerate(gradients):
            if not bool(cp.all(cp.isfinite(gradient)).get()):
                raise FloatingPointError(f"non-finite hardcoded gradient tensor {index}")
            if (
                not allow_zero_tensors
                and float(cp.max(cp.abs(gradient)).get()) == 0.0
            ):
                raise FloatingPointError(f"zero hardcoded gradient tensor {index}")
        total = sum(cp.sum(item.astype(F64) ** 2) for item in gradients)
        norm = float(cp.sqrt(total).get())
        if not math.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("zero or non-finite total hardcoded gradient")
        scale = min(1.0, cfg.grad_clip / max(norm, 1.0e-30))
        parameters: list[cp.ndarray] = []
        for weight, bias in zip(self.weights, self.biases):
            parameters.extend([weight, bias])
        beta1, beta2, epsilon = 0.9, 0.999, 1.0e-8
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            gradient = gradient.astype(parameter.dtype, copy=False) * scale
            self.steps[index] += 1
            self.exp_avg[index] *= beta1
            self.exp_avg[index] += (1.0 - beta1) * gradient
            self.exp_avg_sq[index] *= beta2
            self.exp_avg_sq[index] += (1.0 - beta2) * gradient * gradient
            correction1 = 1.0 - beta1 ** self.steps[index]
            correction2 = 1.0 - beta2 ** self.steps[index]
            denominator = cp.sqrt(self.exp_avg_sq[index] / correction2) + epsilon
            parameter -= (cfg.learning_rate / correction1) * self.exp_avg[index] / denominator
        # ``cp.stack`` owns separate storage from the per-mode master arrays.
        # Keep the batched forward weights synchronized after every optimizer
        # step for every compute dtype, not only for the FP16-shadow path.
        for shadow, master in zip(self.shadow_weights, self.weights):
            shadow[...] = master
        for shadow, master in zip(self.shadow_biases, self.biases):
            shadow[...] = master


class HardcodedModel:
    def __init__(self, cfg: SimpleNamespace, arrays: dict[str, np.ndarray]):
        compute_dtype = _compute_dtype(cfg)
        device_arrays = {
            name: cp.asarray(value).astype(F64, copy=False)
            if compute_dtype is F64 and np.issubdtype(value.dtype, np.floating)
            else cp.asarray(value)
            for name, value in arrays.items()
        }
        self.modes = [
            HardcodedMode(
                device_arrays, mode, cfg.hidden_layers, compute_dtype
            )
            for mode in range(cfg.modes)
        ]
        layer_count = cfg.hidden_layers + 1
        fp32_boundary_layers = bool(getattr(cfg, "fp32_boundary_layers", False))
        self.shadow_weight_stacks = [
            cp.stack([mode.weights[layer] for mode in self.modes]).astype(
                F32
                if fp32_boundary_layers and layer in (0, layer_count - 1)
                else F16
                if compute_dtype is F16
                else F64
                if compute_dtype is F64
                else F32
            )
            for layer in range(layer_count)
        ]
        self.shadow_bias_stacks = [
            cp.stack([mode.biases[layer] for mode in self.modes]).astype(
                F32
                if fp32_boundary_layers and layer in (0, layer_count - 1)
                else F16
                if compute_dtype is F16
                else F64
                if compute_dtype is F64
                else F32
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
            coordinate_dtype = (
                F32
                if self.modes[start].compute_dtype is F16
                else self.modes[start].compute_dtype
            )
            x, y = xy_jets(points, coordinate_dtype)
            envelope = envelope_jet(cfg, x, y, beta, alpha)
            current = encoding_jet(cfg, x, y, beta, alpha)
            layer_cache: list[tuple[Jet2, Jet2]] = []
            for layer in range(cfg.hidden_layers):
                before = current
                preactivation = linear_forward_batch(
                    before,
                    self.shadow_weight_stacks[layer][start:stop],
                    self.shadow_bias_stacks[layer][start:stop],
                    (
                        F32
                        if self.shadow_weight_stacks[layer].dtype == F32
                        else self.modes[start].compute_dtype
                    ),
                )
                current = sine_forward(preactivation)
                if cache:
                    layer_cache.append((before, preactivation))
            output_input = current
            raw = linear_forward_batch(
                output_input,
                self.shadow_weight_stacks[-1][start:stop],
                self.shadow_bias_stacks[-1][start:stop],
                (
                    F32
                    if self.shadow_weight_stacks[-1].dtype == F32
                    else self.modes[start].compute_dtype
                ),
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


def uniform_fluid_points(
    cfg: SimpleNamespace,
    count: int,
    beta: float,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Return a nested, uniform Sobol quadrature of the fluid domain.

    The legacy mixed sampler deliberately over-samples the wall and wake but
    assigns every accepted point the same area weight.  That changes the
    measure in both the mass and stiffness forms.  A variational Stokes
    problem needs the physical area measure, so certified runs use a single
    uniform proposal over the bounding box and reject solid points.  For a
    fixed seed, the first ``count`` accepted points are a prefix of every
    finer call, which gives the nested quadrature sequence used by the
    a-posteriori certificate.
    """

    if count <= 0:
        raise ValueError("uniform fluid quadrature count must be positive")
    return _sobol_box(
        cfg,
        int(count),
        (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
        int(seed),
        beta,
        alpha,
    )


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


def _small_vorticity_stencil(
    cfg: SimpleNamespace, points: np.ndarray, beta: float, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return globally distributed anchors, four neighbours, and normalized weights."""
    requested = max(
        1, min(len(points), int(getattr(cfg, "small_vorticity_anchor_points", 512)))
    )
    anchor_indices = np.linspace(0, len(points) - 1, requested, dtype=np.int64)
    anchors = np.asarray(points[anchor_indices], dtype=np.float64)
    distance = float(
        getattr(cfg, "small_vorticity_scale_chords", 0.05)
    ) * float(cfg.chord)
    offsets = distance * np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        dtype=np.float64,
    )
    neighbours = anchors[:, None, :] + offsets[None, :, :]
    flat = neighbours.reshape(-1, 2)
    valid = (
        (flat[:, 0] >= cfg.xmin)
        & (flat[:, 0] <= cfg.xmax)
        & (flat[:, 1] >= cfg.ymin)
        & (flat[:, 1] <= cfg.ymax)
        & ~inside_airfoil(cfg, flat, beta, alpha)
    ).reshape(requested, 4)
    neighbours = np.where(valid[:, :, None], neighbours, anchors[:, None, :])
    counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    weights = valid.astype(np.float64) / counts
    return anchor_indices, neighbours.reshape(-1, 2), weights


def _velocity_peel_step(
    cfg: SimpleNamespace, model: HardcodedModel, points: np.ndarray,
    beta: float, alpha: float, training: bool,
    reference_model: HardcodedModel | None = None,
    full_diagnostics: bool = True,
) -> dict[str, Any]:
    certified_stokes = (
        str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
        == "stokes_certified"
    )
    if certified_stokes and any(
        float(getattr(cfg, name, 0.0)) != 0.0
        for name in (
            "vorticity_regularization_weight",
            "small_vorticity_regularization_weight",
            "large_vorticity_preservation_weight",
            "peeled_subspace_vorticity_regularization_weight",
        )
    ):
        raise ValueError(
            "stokes_certified forbids empirical vorticity/filter penalties"
        )
    weight = ((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(cfg, beta)) / len(points)
    regularization_weight = float(
        getattr(cfg, "vorticity_regularization_weight", 0.0)
    )
    small_weight = float(
        getattr(cfg, "small_vorticity_regularization_weight", 0.0)
    )
    preservation_weight = float(
        getattr(cfg, "large_vorticity_preservation_weight", 0.0)
    )
    peeled_subspace_weight = float(
        getattr(cfg, "peeled_subspace_vorticity_regularization_weight", 0.0)
    )
    raw_scale_selective = small_weight > 0.0 or preservation_weight > 0.0
    scale_selective = raw_scale_selective or peeled_subspace_weight > 0.0
    if raw_scale_selective and reference_model is None:
        raise ValueError(
            "scale-selective vorticity regularization requires a frozen reference model"
        )
    n = len(points)
    if scale_selective:
        anchor_indices, neighbour_points, stencil_weights_np = (
            _small_vorticity_stencil(cfg, points, beta, alpha)
        )
        anchor_count = len(anchor_indices)
        current_points = np.concatenate((points, neighbour_points), axis=0)
        reference_points = np.concatenate(
            (points[anchor_indices], neighbour_points), axis=0
        )
        stencil_weights = cp.asarray(stencil_weights_np, dtype=F64)
        reference_stream = (
            reference_model.forward_stream(
                cfg, reference_points, beta, alpha, False
            )
            if raw_scale_selective
            else None
        )
        valid_neighbour_fraction = float(np.mean(stencil_weights_np > 0.0))
    else:
        anchor_indices = np.empty(0, dtype=np.int64)
        anchor_count = 0
        current_points = points
        reference_stream = None
        stencil_weights = None
        valid_neighbour_fraction = 1.0
    # Allocate the construction bases once.  Repeated ``cp.stack`` of an
    # ever-growing Python list makes CuPy's memory pool retain many obsolete
    # matrix sizes and can consume the full GPU before a single epoch ends.
    # These column views are algebraically identical and keep peak memory
    # proportional to the current quadrature rather than to its history.
    basis_v = cp.empty((2 * n, cfg.modes), dtype=F64)
    basis_g = cp.empty((4 * n, cfg.modes), dtype=F64)
    previous_omega_augmented: list[cp.ndarray] = []
    raw_matrix = cp.empty((2 * n, cfg.modes), dtype=F64)
    energies: list[float] = []
    projections: list[float] = []
    enstrophy_ratios: list[float] = []
    regularization_losses: list[float] = []
    small_ratios: list[float] = []
    preservation_ratios: list[float] = []
    small_losses: list[float] = []
    preservation_losses: list[float] = []
    peeled_subspace_ratios: list[float] = []
    peeled_subspace_losses: list[float] = []
    freeze_prefix_modes = int(getattr(cfg, "freeze_prefix_modes", 0))
    fast_orthonormal_projection = bool(
        getattr(cfg, "orthonormal_peeling_fast_path", False)
    ) or certified_stokes
    for mode_index, (mode, field, cache) in enumerate(model.forward_stream(
        cfg, current_points, beta, alpha, training
    )):
        requires_grad = training and mode_index >= freeze_prefix_modes
        center_field = Jet2(
            field.value[:n], field.grad[:n], field.hess[:n]
        )
        psi_x = center_field.grad[:, 0, 0]
        psi_y = center_field.grad[:, 0, 1]
        velocity = cp.concatenate((psi_y, -psi_x)).astype(F64)
        raw_matrix[:, mode_index] = velocity
        h = center_field.hess[:, 0]
        gradient = cp.concatenate((h[:, 1, 0], h[:, 1, 1], -h[:, 0, 0], -h[:, 0, 1])).astype(F64)
        if mode_index:
            u = basis_v[:, :mode_index]
            g = basis_g[:, :mode_index]
            lhs = (
                None
                if fast_orthonormal_projection
                else weight * (u.T @ u)
            )
            rhs = weight * (u.T @ velocity)
            if fast_orthonormal_projection:
                coefficients = rhs
            elif certified_stokes:
                symmetric_lhs = 0.5 * (lhs + lhs.T)
                lhs_values, lhs_vectors = cp.linalg.eigh(symmetric_lhs)
                largest_lhs = cp.maximum(lhs_values[-1], 0.0)
                lhs_tolerance = (
                    np.finfo(np.float64).eps
                    * max(2 * n, mode_index)
                    * largest_lhs
                )
                keep_lhs = lhs_values > lhs_tolerance
                if int(cp.count_nonzero(keep_lhs).get()) != mode_index:
                    raise np.linalg.LinAlgError(
                        "construction quadrature does not resolve the previous "
                        f"Stokes mass space at mode {mode_index}"
                    )
                kept_vectors = lhs_vectors[:, keep_lhs]
                coefficients = kept_vectors @ (
                    (kept_vectors.T @ rhs) / lhs_values[keep_lhs]
                )
            else:
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
        if certified_stokes:
            raw_mass = weight * cp.dot(velocity, velocity)
            mass_tolerance = (
                np.finfo(np.float64).eps
                * max(2 * n, mode_index + 1)
                * raw_mass
            )
            if not bool((mass > mass_tolerance).get()):
                raise np.linalg.LinAlgError(
                    "construction quadrature cannot resolve Stokes mode "
                    f"{mode_index}: residual mass={float(mass.get()):.6e}, "
                    f"backward-error bound={float(mass_tolerance.get()):.6e}"
                )
            denominator = mass
        else:
            denominator = cp.maximum(mass, cfg.mass_floor)
        rayleigh = stiffness / denominator
        residual_omega = (
            residual_g[2 * n : 3 * n] - residual_g[n : 2 * n]
        )
        enstrophy = weight * cp.dot(residual_omega, residual_omega)
        enstrophy_ratio = enstrophy / denominator
        global_regularization_loss = regularization_weight * enstrophy_ratio
        if raw_scale_selective:
            _, reference_field, _ = next(reference_stream)
            raw_omega = -(
                field.hess[:, 0, 0, 0] + field.hess[:, 0, 1, 1]
            ).astype(F64)
            reference_omega = -(
                reference_field.hess[:, 0, 0, 0]
                + reference_field.hess[:, 0, 1, 1]
            ).astype(F64)
            anchor_omega = raw_omega[cp.asarray(anchor_indices)]
            neighbour_omega = raw_omega[n:].reshape(anchor_count, 4)
            reference_anchor_omega = reference_omega[:anchor_count]
            reference_neighbour_omega = reference_omega[anchor_count:].reshape(
                anchor_count, 4
            )
            averaged_omega = cp.sum(
                stencil_weights * neighbour_omega, axis=1
            )
            reference_averaged_omega = cp.sum(
                stencil_weights * reference_neighbour_omega, axis=1
            )
            highpass = anchor_omega - averaged_omega
            lowpass = 0.5 * (anchor_omega + averaged_omega)
            reference_lowpass = 0.5 * (
                reference_anchor_omega + reference_averaged_omega
            )
            reference_scale = cp.maximum(
                cp.mean(reference_anchor_omega * reference_anchor_omega),
                cfg.mass_floor,
            )
            small_ratio = cp.mean(highpass * highpass) / reference_scale
            preservation_ratio = cp.mean(
                (lowpass - reference_lowpass) ** 2
            ) / reference_scale
            small_loss = small_weight * small_ratio
            preservation_loss = preservation_weight * preservation_ratio
        else:
            highpass = lowpass = reference_lowpass = reference_scale = None
            small_ratio = preservation_ratio = cp.asarray(0.0, dtype=F64)
            small_loss = preservation_loss = cp.asarray(0.0, dtype=F64)
        if scale_selective:
            raw_omega_augmented = -(
                field.hess[:, 0, 0, 0] + field.hess[:, 0, 1, 1]
            ).astype(F64)
        else:
            raw_omega_augmented = None
        if peeled_subspace_weight > 0.0:
            if previous_omega_augmented:
                omega_basis = cp.stack(previous_omega_augmented, axis=1)
                reconstructed_omega_augmented = omega_basis @ coefficients
            else:
                omega_basis = None
                reconstructed_omega_augmented = cp.zeros_like(
                    raw_omega_augmented
                )
            residual_omega_augmented = (
                raw_omega_augmented - reconstructed_omega_augmented
            )
            normalized_omega_augmented = (
                residual_omega_augmented / cp.sqrt(denominator)
            )
            anchor_normalized_omega = normalized_omega_augmented[
                cp.asarray(anchor_indices)
            ]
            neighbour_normalized_omega = normalized_omega_augmented[
                n:
            ].reshape(anchor_count, 4)
            peeled_highpass = anchor_normalized_omega - cp.sum(
                stencil_weights * neighbour_normalized_omega, axis=1
            )
            peeled_subspace_ratio = cp.mean(peeled_highpass ** 2)
            peeled_subspace_loss = (
                peeled_subspace_weight * peeled_subspace_ratio
            )
        else:
            omega_basis = residual_omega_augmented = None
            normalized_omega_augmented = peeled_highpass = None
            peeled_subspace_ratio = cp.asarray(0.0, dtype=F64)
            peeled_subspace_loss = cp.asarray(0.0, dtype=F64)
        regularization_loss = (
            global_regularization_loss + small_loss + preservation_loss
            + peeled_subspace_loss
        )
        projection_mass = weight * cp.dot(reconstructed_v, reconstructed_v)
        if requires_grad:
            grad_residual_v = (
                -2.0 * stiffness * weight / (cfg.rayleigh_scale * denominator * denominator)
            ) * residual_v
            grad_residual_g = (
                2.0 * weight / (cfg.rayleigh_scale * denominator)
            ) * residual_g
            if regularization_weight > 0.0:
                grad_residual_v += (
                    -2.0
                    * regularization_weight
                    * enstrophy
                    * weight
                    / (denominator * denominator)
                ) * residual_v
                grad_omega = (
                    2.0
                    * regularization_weight
                    * weight
                    / denominator
                ) * residual_omega
                grad_residual_g[n : 2 * n] -= grad_omega
                grad_residual_g[2 * n : 3 * n] += grad_omega
            scale_adjoint = cp.zeros(len(current_points), dtype=F64)
            projection_coefficient_adjoint = None
            if raw_scale_selective:
                grad_highpass = (
                    2.0 * small_weight * highpass
                    / (anchor_count * reference_scale)
                )
                grad_lowpass = (
                    2.0
                    * preservation_weight
                    * (lowpass - reference_lowpass)
                    / (anchor_count * reference_scale)
                )
                scale_adjoint[cp.asarray(anchor_indices)] += (
                    grad_highpass + 0.5 * grad_lowpass
                )
                neighbour_adjoint = (
                    -stencil_weights * grad_highpass[:, None]
                    + 0.5 * stencil_weights * grad_lowpass[:, None]
                )
                scale_adjoint[n:] = neighbour_adjoint.reshape(-1)
            if peeled_subspace_weight > 0.0:
                grad_peeled_highpass = (
                    2.0
                    * peeled_subspace_weight
                    * peeled_highpass
                    / anchor_count
                )
                grad_normalized_omega = cp.zeros(
                    len(current_points), dtype=F64
                )
                grad_normalized_omega[cp.asarray(anchor_indices)] += (
                    grad_peeled_highpass
                )
                grad_normalized_omega[n:] = (
                    -stencil_weights * grad_peeled_highpass[:, None]
                ).reshape(-1)
                grad_residual_omega_augmented = (
                    grad_normalized_omega / cp.sqrt(denominator)
                )
                scale_adjoint += grad_residual_omega_augmented
                mass_adjoint = (
                    -0.5
                    * cp.dot(
                        grad_normalized_omega,
                        normalized_omega_augmented,
                    )
                    / denominator
                )
                grad_residual_v += (
                    2.0 * weight * mass_adjoint
                ) * residual_v
                if previous_omega_augmented:
                    projection_coefficient_adjoint = -(
                        omega_basis.T @ grad_residual_omega_augmented
                    )
            grad_velocity = grad_residual_v
            if mode_index:
                grad_q = -(u.T @ grad_residual_v) - (g.T @ grad_residual_g)
                if projection_coefficient_adjoint is not None:
                    grad_q += projection_coefficient_adjoint
                if fast_orthonormal_projection:
                    projection_gradient = grad_q
                elif certified_stokes:
                    projection_gradient = kept_vectors @ (
                        (kept_vectors.T @ grad_q) / lhs_values[keep_lhs]
                    )
                else:
                    projection_gradient = cp.linalg.solve(lhs.T, grad_q)
                grad_velocity = grad_velocity + weight * u @ projection_gradient
            network_dtype = _compute_dtype(cfg)
            adjoint_dtype = (
                F16
                if bool(getattr(cfg, "all_network_fp16", False))
                else F64
                if network_dtype is F64
                else F32
            )
            if adjoint_dtype is F16:
                configured_scale = float(getattr(cfg, "fp16_loss_scale", 128.0))
                raw_adjoint_max = max(
                    float(cp.max(cp.abs(grad_velocity)).get()),
                    float(cp.max(cp.abs(grad_residual_g)).get()),
                    float(cp.max(cp.abs(scale_adjoint)).get()),
                    1.0e-30,
                )
                loss_scale = min(configured_scale, 1.0e-4 / raw_adjoint_max)
            else:
                loss_scale = 1.0
            adjoint = Jet2(
                cp.zeros((len(current_points), 1), dtype=adjoint_dtype),
                cp.zeros((len(current_points), 1, 2), dtype=adjoint_dtype),
                cp.zeros((len(current_points), 1, 2, 2), dtype=adjoint_dtype),
            )
            adjoint.grad[:n, 0, 1] = (
                grad_velocity[:n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.grad[:n, 0, 0] = (
                -grad_velocity[n:] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:n, 0, 1, 0] = (
                grad_residual_g[:n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:n, 0, 1, 1] = (
                grad_residual_g[n : 2 * n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:n, 0, 0, 0] = (
                -grad_residual_g[2 * n : 3 * n] * loss_scale
            ).astype(adjoint_dtype)
            adjoint.hess[:n, 0, 0, 1] = (
                -grad_residual_g[3 * n :] * loss_scale
            ).astype(adjoint_dtype)
            scale_hessian_adjoint = (-scale_adjoint * loss_scale).astype(
                adjoint_dtype
            )
            adjoint.hess[:, 0, 0, 0] += scale_hessian_adjoint
            adjoint.hess[:, 0, 1, 1] += scale_hessian_adjoint
            gradients = mode.backward(cache, adjoint)
            if (
                loss_scale != 1.0
                and not bool(getattr(cfg, "all_network_fp16", False))
            ):
                gradients = [gradient / loss_scale for gradient in gradients]
            mode.adam_step(gradients, cfg)
        normalizer = 1.0 / cp.sqrt(denominator)
        basis_v[:, mode_index] = residual_v * normalizer
        basis_g[:, mode_index] = residual_g * normalizer
        if peeled_subspace_weight > 0.0:
            previous_omega_augmented.append(normalized_omega_augmented)
        energies.append(float(rayleigh.get()))
        projections.append(float((projection_mass / denominator).get()))
        enstrophy_ratios.append(float(enstrophy_ratio.get()))
        regularization_losses.append(float(regularization_loss.get()))
        small_ratios.append(float(small_ratio.get()))
        preservation_ratios.append(float(preservation_ratio.get()))
        small_losses.append(float(small_loss.get()))
        preservation_losses.append(float(preservation_loss.get()))
        peeled_subspace_ratios.append(float(peeled_subspace_ratio.get()))
        peeled_subspace_losses.append(float(peeled_subspace_loss.get()))
        trim_every = int(getattr(cfg, "cuda_memory_pool_trim_modes", 0))
        if certified_stokes and trim_every > 0 and (
            (mode_index + 1) % trim_every == 0
        ):
            # CuPy caches the many Jet2 reverse-pass temporaries even though
            # their shapes repeat.  Releasing only currently free pool blocks
            # is numerically inert: live model/Adam/basis arrays keep their
            # allocations, while adaptive quadrature retains usable headroom.
            cp.get_default_memory_pool().free_all_blocks()
    gram = weight * basis_v.T @ basis_v
    orth_error = float(cp.max(cp.abs(gram - cp.eye(cfg.modes, dtype=F64))).get())
    raw_gram = weight * raw_matrix.T @ raw_matrix
    eigenvalues = cp.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    if certified_stokes:
        threshold = (
            np.finfo(np.float64).eps
            * max(2 * n, cfg.modes)
            * cp.maximum(eigenvalues[-1], 0.0)
        )
    else:
        threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
    effective_rank = int(cp.sum(eigenvalues > threshold).get())
    return {
        "loss": float(
            np.mean(energies) / cfg.rayleigh_scale
            + np.mean(regularization_losses)
        ),
        "base_loss": float(np.mean(energies) / cfg.rayleigh_scale),
        "vorticity_regularization_loss": float(
            np.mean(regularization_losses)
        ),
        "enstrophy_ratio": float(
            np.mean(enstrophy_ratios)
        ),
        "small_vorticity_regularization_loss": float(np.mean(small_losses)),
        "large_vorticity_preservation_loss": float(
            np.mean(preservation_losses)
        ),
        "small_vorticity_highpass_ratio": float(np.mean(small_ratios)),
        "large_vorticity_drift_ratio": float(
            np.mean(preservation_ratios)
        ),
        "regularization_valid_neighbor_fraction": valid_neighbour_fraction,
        "peeled_subspace_vorticity_regularization_loss": float(
            np.mean(peeled_subspace_losses)
        ),
        "peeled_subspace_vorticity_highpass_trace": float(
            np.mean(peeled_subspace_ratios)
        ),
        "mean_energy": float(np.mean(energies)),
        "max_energy": float(np.max(energies)),
        "min_energy": float(np.min(energies)),
        "mean_projection_ratio": float(np.mean(projections)),
        "orth_error": orth_error,
        "effective_raw_rank": effective_rank,
        "mode_energies": energies,
    }


def _hybrid_velocity_vorticity_peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    full_diagnostics: bool,
) -> dict[str, Any]:
    """Train one shared basis in an equal velocity/vorticity metric.

    The velocity term is the existing vector Rayleigh quotient.  The
    vorticity term is a graph-consistent Rayleigh quotient: vorticity itself
    is evaluated exactly from the Jet2 Hessian, while its spatial Dirichlet
    energy is evaluated with symmetric four-neighbour differences.  This
    keeps the fixed graph at seven Jet2 channels instead of increasing every
    MLP activation to the fifteen channels required by Jet3.
    """

    velocity_weight = float(getattr(cfg, "hybrid_velocity_weight", 0.5))
    vorticity_weight = float(getattr(cfg, "hybrid_vorticity_weight", 0.5))
    total_objective_weight = velocity_weight + vorticity_weight
    if total_objective_weight <= 0.0:
        raise ValueError("hybrid velocity/vorticity weights must sum to a positive value")
    velocity_weight /= total_objective_weight
    vorticity_weight /= total_objective_weight
    length_scale = float(getattr(
        cfg, "hybrid_vorticity_mass_length_scale_chords", 1.0
    )) * float(cfg.chord)
    distance = float(getattr(
        cfg, "hybrid_vorticity_difference_chords", 0.05
    )) * float(cfg.chord)
    if distance <= 0.0:
        raise ValueError("hybrid vorticity difference distance must be positive")
    reorthogonalization_passes = max(
        1, int(getattr(cfg, "hybrid_reorthogonalization_passes", 2))
    )
    velocity_rayleigh_scale = float(getattr(
        cfg, "hybrid_velocity_rayleigh_scale", cfg.rayleigh_scale
    ))
    vorticity_rayleigh_scale = float(getattr(
        cfg, "hybrid_vorticity_rayleigh_scale", cfg.rayleigh_scale
    ))
    if velocity_rayleigh_scale <= 0.0 or vorticity_rayleigh_scale <= 0.0:
        raise ValueError("hybrid Rayleigh normalization scales must be positive")
    n = len(points)
    area = (
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = area / n

    requested = max(
        1,
        min(
            n,
            int(getattr(cfg, "hybrid_vorticity_anchor_points", 512)),
        ),
    )
    anchor_indices = np.linspace(0, n - 1, requested, dtype=np.int64)
    anchors = np.asarray(points[anchor_indices], dtype=np.float64)
    offsets = distance * np.asarray(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
        dtype=np.float64,
    )
    neighbours = anchors[:, None, :] + offsets[None, :, :]
    flat_neighbours = neighbours.reshape(-1, 2)
    valid = (
        (flat_neighbours[:, 0] >= cfg.xmin)
        & (flat_neighbours[:, 0] <= cfg.xmax)
        & (flat_neighbours[:, 1] >= cfg.ymin)
        & (flat_neighbours[:, 1] <= cfg.ymax)
        & ~inside_airfoil(cfg, flat_neighbours, beta, alpha)
    ).reshape(requested, 4)
    neighbours = np.where(valid[:, :, None], neighbours, anchors[:, None, :])
    valid_counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    stencil_weights_np = valid.astype(np.float64) / valid_counts
    stencil_weights = cp.asarray(stencil_weights_np, dtype=F64)
    current_points = np.concatenate((points, neighbours.reshape(-1, 2)), axis=0)
    anchor_indices_device = cp.asarray(anchor_indices)
    anchor_weight = area / requested
    direction_factor = 2.0

    modes = int(cfg.modes)
    basis_v = cp.empty((2 * n, modes), dtype=F64)
    basis_g = cp.empty((4 * n, modes), dtype=F64)
    basis_omega = cp.empty((n, modes), dtype=F64)
    basis_omega_difference = cp.empty((4 * requested, modes), dtype=F64)
    raw_velocities: list[cp.ndarray] = []
    raw_omegas: list[cp.ndarray] = []
    hybrid_energies: list[float] = []
    hybrid_losses: list[float] = []
    velocity_rayleighs: list[float] = []
    vorticity_rayleighs: list[float] = []
    enstrophy_ratios: list[float] = []
    projection_ratios: list[float] = []
    orth_error = 0.0
    mass_length_squared = length_scale * length_scale

    for mode_index, (mode, field, cache) in enumerate(model.forward_stream(
        cfg, current_points, beta, alpha, training
    )):
        center = Jet2(field.value[:n], field.grad[:n], field.hess[:n])
        psi_x = center.grad[:, 0, 0]
        psi_y = center.grad[:, 0, 1]
        velocity = cp.concatenate((psi_y, -psi_x)).astype(F64)
        hessian = center.hess[:, 0]
        velocity_gradient = cp.concatenate(
            (
                hessian[:, 1, 0],
                hessian[:, 1, 1],
                -hessian[:, 0, 0],
                -hessian[:, 0, 1],
            )
        ).astype(F64)
        omega_all = -(
            field.hess[:, 0, 0, 0] + field.hess[:, 0, 1, 1]
        ).astype(F64)
        omega = omega_all[:n]
        omega_anchor = omega[anchor_indices_device]
        omega_neighbours = omega_all[n:].reshape(requested, 4)
        omega_difference = (omega_neighbours - omega_anchor[:, None]) / distance

        if full_diagnostics:
            raw_velocities.append(velocity.copy())
            raw_omegas.append(omega.copy())

        residual_v = velocity.copy()
        residual_g = velocity_gradient.copy()
        residual_omega = omega.copy()
        residual_omega_difference = omega_difference.copy()
        coefficients = cp.zeros(mode_index, dtype=F64)
        if mode_index:
            previous_v = basis_v[:, :mode_index]
            previous_g = basis_g[:, :mode_index]
            previous_omega = basis_omega[:, :mode_index]
            previous_omega_difference = basis_omega_difference[:, :mode_index]
            for _ in range(reorthogonalization_passes):
                increment = (
                    velocity_weight * weight * (previous_v.T @ residual_v)
                    + vorticity_weight
                    * mass_length_squared
                    * weight
                    * (previous_omega.T @ residual_omega)
                )
                coefficients += increment
                residual_v -= previous_v @ increment
                residual_g -= previous_g @ increment
                residual_omega -= previous_omega @ increment
                residual_omega_difference -= (
                    previous_omega_difference @ increment
                ).reshape(requested, 4)
        else:
            previous_v = previous_g = previous_omega = None
            previous_omega_difference = None

        velocity_mass = weight * cp.dot(residual_v, residual_v)
        velocity_stiffness = weight * cp.dot(residual_g, residual_g)
        vorticity_mass = weight * cp.dot(residual_omega, residual_omega)
        vorticity_stiffness = (
            direction_factor
            * anchor_weight
            * cp.sum(
                stencil_weights * residual_omega_difference**2,
                dtype=F64,
            )
        )
        velocity_denominator = cp.maximum(velocity_mass, cfg.mass_floor)
        vorticity_denominator = cp.maximum(vorticity_mass, cfg.mass_floor)
        velocity_rayleigh = velocity_stiffness / velocity_denominator
        vorticity_rayleigh = vorticity_stiffness / vorticity_denominator
        hybrid_energy = (
            velocity_weight * velocity_rayleigh
            + vorticity_weight * vorticity_rayleigh
        )
        hybrid_loss = (
            velocity_weight * velocity_rayleigh / velocity_rayleigh_scale
            + vorticity_weight * vorticity_rayleigh / vorticity_rayleigh_scale
        )
        mixed_mass = (
            velocity_weight * velocity_mass
            + vorticity_weight * mass_length_squared * vorticity_mass
        )
        mixed_denominator = cp.maximum(mixed_mass, cfg.mass_floor)

        requires_grad = training and mode_index >= int(
            getattr(cfg, "freeze_prefix_modes", 0)
        )
        if requires_grad:
            grad_residual_v = (
                -2.0
                * velocity_weight
                * velocity_stiffness
                * weight
                / (velocity_denominator * velocity_denominator)
                / velocity_rayleigh_scale
            ) * residual_v
            grad_residual_g = (
                2.0
                * velocity_weight
                * weight
                / velocity_denominator
                / velocity_rayleigh_scale
            ) * residual_g
            grad_residual_omega = (
                -2.0
                * vorticity_weight
                * vorticity_stiffness
                * weight
                / (vorticity_denominator * vorticity_denominator)
                / vorticity_rayleigh_scale
            ) * residual_omega
            grad_residual_omega_difference = (
                2.0
                * vorticity_weight
                * direction_factor
                * anchor_weight
                * stencil_weights
                / vorticity_denominator
                / vorticity_rayleigh_scale
            ) * residual_omega_difference

            if mode_index:
                for _ in range(reorthogonalization_passes):
                    coefficient_adjoint = -(
                        previous_v.T @ grad_residual_v
                        + previous_g.T @ grad_residual_g
                        + previous_omega.T @ grad_residual_omega
                        + previous_omega_difference.T
                        @ grad_residual_omega_difference.reshape(-1)
                    )
                    grad_residual_v += (
                        velocity_weight
                        * weight
                        * (previous_v @ coefficient_adjoint)
                    )
                    grad_residual_omega += (
                        vorticity_weight
                        * mass_length_squared
                        * weight
                        * (previous_omega @ coefficient_adjoint)
                    )

            omega_adjoint = cp.zeros(len(current_points), dtype=F64)
            omega_adjoint[:n] = grad_residual_omega
            neighbour_adjoint = grad_residual_omega_difference / distance
            omega_adjoint[anchor_indices_device] -= cp.sum(
                neighbour_adjoint, axis=1
            )
            omega_adjoint[n:] = neighbour_adjoint.reshape(-1)

            adjoint = Jet2(
                cp.zeros((len(current_points), 1), dtype=F32),
                cp.zeros((len(current_points), 1, 2), dtype=F32),
                cp.zeros((len(current_points), 1, 2, 2), dtype=F32),
            )
            adjoint.grad[:n, 0, 1] = grad_residual_v[:n].astype(F32)
            adjoint.grad[:n, 0, 0] = (-grad_residual_v[n:]).astype(F32)
            adjoint.hess[:n, 0, 1, 0] = grad_residual_g[:n].astype(F32)
            adjoint.hess[:n, 0, 1, 1] = grad_residual_g[n : 2 * n].astype(F32)
            adjoint.hess[:n, 0, 0, 0] = (-grad_residual_g[2 * n : 3 * n]).astype(F32)
            adjoint.hess[:n, 0, 0, 1] = (-grad_residual_g[3 * n :]).astype(F32)
            omega_hessian_adjoint = (-omega_adjoint).astype(F32)
            adjoint.hess[:, 0, 0, 0] += omega_hessian_adjoint
            adjoint.hess[:, 0, 1, 1] += omega_hessian_adjoint
            gradients = mode.backward(cache, adjoint)
            mode.adam_step(gradients, cfg)

        normalizer = 1.0 / cp.sqrt(mixed_denominator)
        normalized_v = residual_v * normalizer
        normalized_g = residual_g * normalizer
        normalized_omega = residual_omega * normalizer
        normalized_omega_difference = residual_omega_difference * normalizer
        orth_error = max(
            orth_error,
            abs(float((mixed_mass / mixed_denominator).get()) - 1.0),
        )
        if mode_index:
            final_overlap = (
                velocity_weight * weight * (previous_v.T @ normalized_v)
                + vorticity_weight
                * mass_length_squared
                * weight
                * (previous_omega.T @ normalized_omega)
            )
            orth_error = max(
                orth_error, float(cp.max(cp.abs(final_overlap)).get())
            )
        basis_v[:, mode_index] = normalized_v
        basis_g[:, mode_index] = normalized_g
        basis_omega[:, mode_index] = normalized_omega
        basis_omega_difference[:, mode_index] = normalized_omega_difference.reshape(-1)

        hybrid_energies.append(float(hybrid_energy.get()))
        hybrid_losses.append(float(hybrid_loss.get()))
        velocity_rayleighs.append(float(velocity_rayleigh.get()))
        vorticity_rayleighs.append(float(vorticity_rayleigh.get()))
        enstrophy_ratios.append(float((vorticity_mass / velocity_denominator).get()))
        projection_ratios.append(
            float((cp.dot(coefficients, coefficients) / mixed_denominator).get())
            if mode_index
            else 0.0
        )

    if full_diagnostics:
        raw_velocity_matrix = cp.stack(raw_velocities, axis=1)
        raw_omega_matrix = cp.stack(raw_omegas, axis=1)
        raw_gram = (
            velocity_weight * weight * (raw_velocity_matrix.T @ raw_velocity_matrix)
            + vorticity_weight
            * mass_length_squared
            * weight
            * (raw_omega_matrix.T @ raw_omega_matrix)
        )
        raw_gram = 0.5 * (raw_gram + raw_gram.T)
        eigenvalues = cp.linalg.eigvalsh(raw_gram)
        threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
        effective_rank = int(cp.sum(eigenvalues > threshold).get())
    else:
        effective_rank = -1

    zeros = 0.0
    return {
        "loss": float(np.mean(hybrid_losses)),
        "base_loss": float(np.mean(hybrid_losses)),
        "velocity_rayleigh": float(np.mean(velocity_rayleighs)),
        "vorticity_rayleigh": float(np.mean(vorticity_rayleighs)),
        "hybrid_velocity_weight": velocity_weight,
        "hybrid_vorticity_weight": vorticity_weight,
        "hybrid_velocity_rayleigh_scale": velocity_rayleigh_scale,
        "hybrid_vorticity_rayleigh_scale": vorticity_rayleigh_scale,
        "vorticity_regularization_loss": zeros,
        "enstrophy_ratio": float(np.mean(enstrophy_ratios)),
        "small_vorticity_regularization_loss": zeros,
        "large_vorticity_preservation_loss": zeros,
        "small_vorticity_highpass_ratio": zeros,
        "large_vorticity_drift_ratio": zeros,
        "regularization_valid_neighbor_fraction": float(np.mean(valid)),
        "peeled_subspace_vorticity_regularization_loss": zeros,
        "peeled_subspace_vorticity_highpass_trace": zeros,
        "mean_energy": float(np.mean(hybrid_energies)),
        "max_energy": float(np.max(hybrid_energies)),
        "min_energy": float(np.min(hybrid_energies)),
        "mean_projection_ratio": float(np.mean(projection_ratios)),
        "orth_error": orth_error,
        "effective_raw_rank": effective_rank,
        "mode_energies": hybrid_energies,
    }


def peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    reference_model: HardcodedModel | None = None,
    full_diagnostics: bool = True,
) -> dict[str, Any]:
    objective = str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
    if objective == "velocity_vorticity_hybrid":
        if reference_model is not None:
            raise ValueError("hybrid basis training does not use a reference model")
        return _hybrid_velocity_vorticity_peel_step(
            cfg,
            model,
            points,
            beta,
            alpha,
            training,
            full_diagnostics,
        )
    if objective not in ("velocity_rayleigh", "stokes_certified"):
        raise ValueError(f"unknown basis objective: {objective}")
    return _velocity_peel_step(
        cfg,
        model,
        points,
        beta,
        alpha,
        training,
        reference_model=reference_model,
        full_diagnostics=full_diagnostics,
    )
