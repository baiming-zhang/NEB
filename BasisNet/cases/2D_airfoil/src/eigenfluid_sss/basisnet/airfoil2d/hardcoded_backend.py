"""Fixed-graph CuPy backend for the 2D airfoil SSS trainer.

This module deliberately has no PyTorch dependency and performs no automatic
differentiation.  A compact symmetric third-order spatial jet and the reverse
pass with respect to every MLP parameter are written out explicitly.  The ten
jet channels are ``value, x, y, xx, xy, yy, xxx, xxy, xyy, yyy``.
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


@dataclass
class Jet3:
    value: cp.ndarray
    grad: cp.ndarray
    hess: cp.ndarray
    third: cp.ndarray

    @staticmethod
    def constant(
        value: cp.ndarray | float,
        shape: tuple[int, int],
        dtype: type[cp.generic] = F32,
    ) -> "Jet3":
        v = cp.broadcast_to(cp.asarray(value, dtype=dtype), shape)
        return Jet3(
            v,
            cp.zeros((*shape, 2), dtype=dtype),
            cp.zeros((*shape, 3), dtype=dtype),
            cp.zeros((*shape, 4), dtype=dtype),
        )

    def __add__(self, other: "Jet3" | float) -> "Jet3":
        other = as_jet(other, self.value.shape, self.value.dtype.type)
        return Jet3(
            self.value + other.value,
            self.grad + other.grad,
            self.hess + other.hess,
            self.third + other.third,
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet3":
        return Jet3(-self.value, -self.grad, -self.hess, -self.third)

    def __sub__(self, other: "Jet3" | float) -> "Jet3":
        return self + (-as_jet(other, self.value.shape, self.value.dtype.type))

    def __rsub__(self, other: "Jet3" | float) -> "Jet3":
        return as_jet(other, self.value.shape, self.value.dtype.type) - self

    def __mul__(self, other: "Jet3" | float) -> "Jet3":
        other = as_jet(other, self.value.shape, self.value.dtype.type)
        av, bv = self.value, other.value
        ax, ay = self.grad[..., 0], self.grad[..., 1]
        bx, by = other.grad[..., 0], other.grad[..., 1]
        axx, axy, ayy = (self.hess[..., i] for i in range(3))
        bxx, bxy, byy = (other.hess[..., i] for i in range(3))
        axxx, axxy, axyy, ayyy = (self.third[..., i] for i in range(4))
        bxxx, bxxy, bxyy, byyy = (other.third[..., i] for i in range(4))
        grad = cp.stack((ax * bv + av * bx, ay * bv + av * by), axis=-1)
        hess = cp.stack(
            (
                axx * bv + 2.0 * ax * bx + av * bxx,
                axy * bv + ax * by + ay * bx + av * bxy,
                ayy * bv + 2.0 * ay * by + av * byy,
            ),
            axis=-1,
        )
        third = cp.stack(
            (
                axxx * bv + 3.0 * axx * bx + 3.0 * ax * bxx + av * bxxx,
                axxy * bv + axx * by + 2.0 * axy * bx
                + 2.0 * ax * bxy + ay * bxx + av * bxxy,
                axyy * bv + ayy * bx + 2.0 * axy * by
                + 2.0 * ay * bxy + ax * byy + av * bxyy,
                ayyy * bv + 3.0 * ayy * by + 3.0 * ay * byy + av * byyy,
            ),
            axis=-1,
        )
        return Jet3(av * bv, grad, hess, third)

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet3":
        inv = 1.0 / self.value
        first = -(inv * inv)
        second = 2.0 * inv * inv * inv
        third = -6.0 * inv**4
        return unary(self, inv, first, second, third)

    def __truediv__(self, other: "Jet3" | float) -> "Jet3":
        return self * as_jet(
            other, self.value.shape, self.value.dtype.type
        ).reciprocal()

    def square(self) -> "Jet3":
        return self * self

    def pow(self, exponent: int) -> "Jet3":
        result = as_jet(1.0, self.value.shape, self.value.dtype.type)
        for _ in range(exponent):
            result = result * self
        return result


Jet2 = Jet3  # checkpoint/evaluation import compatibility; storage is Jet3.


def as_jet(
    value: Jet3 | float,
    shape: tuple[int, int],
    dtype: type[cp.generic] = F32,
) -> Jet3:
    return value if isinstance(value, Jet3) else Jet3.constant(value, shape, dtype)


def unary(
    jet: Jet3,
    value: cp.ndarray,
    first: cp.ndarray,
    second: cp.ndarray,
    third: cp.ndarray,
) -> Jet3:
    gx, gy = jet.grad[..., 0], jet.grad[..., 1]
    hxx, hxy, hyy = (jet.hess[..., i] for i in range(3))
    txxx, txxy, txyy, tyyy = (jet.third[..., i] for i in range(4))
    grad = first[..., None] * jet.grad
    hess = cp.stack(
        (
            first * hxx + second * gx * gx,
            first * hxy + second * gx * gy,
            first * hyy + second * gy * gy,
        ),
        axis=-1,
    )
    third_jet = cp.stack(
        (
            first * txxx + 3.0 * second * gx * hxx + third * gx**3,
            first * txxy + second * (gy * hxx + 2.0 * gx * hxy)
            + third * gx * gx * gy,
            first * txyy + second * (gx * hyy + 2.0 * gy * hxy)
            + third * gx * gy * gy,
            first * tyyy + 3.0 * second * gy * hyy + third * gy**3,
        ),
        axis=-1,
    )
    return Jet3(value, grad, hess, third_jet)


def xy_jets(
    points: np.ndarray, dtype: type[cp.generic] = F32
) -> tuple[Jet3, Jet3]:
    xy = cp.asarray(points, dtype=dtype)
    count = len(points)
    zero_hess = cp.zeros((count, 1, 3), dtype=dtype)
    zero_third = cp.zeros((count, 1, 4), dtype=dtype)
    gx = cp.zeros((count, 1, 2), dtype=dtype)
    gy = cp.zeros_like(gx)
    gx[:, 0, 0] = 1.0
    gy[:, 0, 1] = 1.0
    return (
        Jet3(xy[:, 0:1], gx, zero_hess.copy(), zero_third.copy()),
        Jet3(xy[:, 1:2], gy, zero_hess, zero_third),
    )


def body_jets(cfg: SimpleNamespace, x: Jet3, y: Jet3, alpha_deg: float) -> tuple[Jet3, Jet3]:
    angle = math.radians(-alpha_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = x - cfg.quarter_chord_x
    dy = y - cfg.quarter_chord_y
    return (
        cosine * dx + sine * dy + 0.25 * cfg.chord,
        -sine * dx + cosine * dy,
    )


def ellipse_implicit_jet(
    cfg: SimpleNamespace, xb: Jet3, yb: Jet3, beta: float
) -> Jet3:
    """C-infinity ellipse implicit function; zero is the airfoil boundary."""
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )


def cinf_double_zero_envelope(level: Jet3) -> Jet3:
    """Smooth double-zero factor on the closure of the sampled fluid domain.

    Fluid samples satisfy ``level >= 0``.  No ``max`` or branch is applied to
    the differentiable expression: ``q**2 / (1 + q**2)`` is C-infinity and
    enforces both streamfunction and its normal derivative at ``q=0`` while
    retaining a nonzero second derivative (and hence wall vorticity).
    """
    squared = level.square()
    return squared / (1.0 + squared)


def envelope_jet(
    cfg: SimpleNamespace, x: Jet3, y: Jet3, beta: float, alpha_deg: float
) -> Jet3:
    if not str(getattr(cfg, "envelope_mode", "")).startswith("cinf_"):
        raise ValueError("the Jet3 backend only supports smooth-closure envelopes")
    scale = 0.25 * cfg.chord
    positive_envelope = cinf_double_zero_envelope
    inlet = positive_envelope((x - cfg.xmin) / scale)
    xb, yb = body_jets(cfg, x, y, alpha_deg)
    level = ellipse_implicit_jet(cfg, xb, yb, beta)
    airfoil = positive_envelope(level)
    envelope = inlet * airfoil
    if cfg.envelope_mode in (
        "cinf_farfield_freestream",
    ):
        yn = (
            2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin)
            - 1.0
        )
        # This is a double-zero envelope for the *perturbation* streamfunction.
        # The reconstructed total field still contains the uniform free stream,
        # so y=ymin/ymax are far-field free-stream boundaries, not no-slip walls.
        farfield = positive_envelope((1.0 - yn.pow(2)) / 0.25)
        envelope = envelope * farfield
    elif cfg.envelope_mode != "cinf_localized":
        raise ValueError(f"Unknown envelope mode: {cfg.envelope_mode}")
    return envelope


def encoding_jet(
    cfg: SimpleNamespace, x: Jet3, y: Jet3, beta: float, alpha_deg: float
) -> Jet3:
    xn = 2.0 * (x - cfg.xmin) / (cfg.xmax - cfg.xmin) - 1.0
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    features: list[Jet3] = []
    for spatial in (xn, yn):
        for frequency in 2.0 ** np.arange(cfg.frequencies):
            phase = spatial * float(frequency)
            features.append(
                unary(
                    phase, cp.sin(phase.value), cp.cos(phase.value),
                    -cp.sin(phase.value), -cp.cos(phase.value),
                )
            )
    for spatial in (xn, yn):
        for frequency in 2.0 ** np.arange(cfg.frequencies):
            phase = spatial * float(frequency)
            features.append(
                unary(
                    phase, cp.cos(phase.value), -cp.sin(phase.value),
                    -cp.cos(phase.value), cp.sin(phase.value),
                )
            )
    shape = x.value.shape
    features.extend(
        [
            Jet3.constant(2.0 * beta - 1.0, shape, x.value.dtype.type),
            Jet3.constant(
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
    return Jet3(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
        cp.concatenate([item.hess for item in features], axis=1),
        cp.concatenate([item.third for item in features], axis=1),
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
    source: Jet3,
    weight: cp.ndarray,
    bias: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet3:
    count = source.value.shape[0]
    packed = cp.concatenate(
        (
            source.value,
            source.grad[..., 0],
            source.grad[..., 1],
            source.hess[..., 0],
            source.hess[..., 1],
            source.hess[..., 2],
            source.third[..., 0],
            source.third[..., 1],
            source.third[..., 2],
            source.third[..., 3],
        ),
        axis=0,
    )
    projected = _matmul(packed, weight.T, compute_dtype).reshape(
        10, count, weight.shape[0]
    )
    return Jet3(
        projected[0] + bias,
        cp.stack((projected[1], projected[2]), axis=-1),
        cp.stack((projected[3], projected[4], projected[5]), axis=-1),
        cp.stack(
            (projected[6], projected[7], projected[8], projected[9]), axis=-1
        ),
    )


def _slice_jet(source: Jet3, index: int) -> Jet3:
    return Jet3(
        source.value[index], source.grad[index], source.hess[index],
        source.third[index],
    )


def _expand_mode_axis(source: Jet3) -> Jet3:
    return Jet3(
        source.value[None], source.grad[None], source.hess[None],
        source.third[None],
    )


def linear_forward_batch(
    source: Jet3,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet3:
    """Project ten compact Jet3 channels for several modes in one GEMM."""
    batched_source = source.value.ndim == 3
    point_axis = 1 if batched_source else 0
    count = source.value.shape[point_axis]
    packed = cp.concatenate(
        (
            source.value,
            source.grad[..., 0],
            source.grad[..., 1],
            source.hess[..., 0],
            source.hess[..., 1],
            source.hess[..., 2],
            source.third[..., 0],
            source.third[..., 1],
            source.third[..., 2],
            source.third[..., 3],
        ),
        axis=point_axis,
    ).astype(compute_dtype, copy=False)
    projected = cp.matmul(
        packed,
        weights.transpose(0, 2, 1).astype(compute_dtype, copy=False),
        dtype=F32,
    )
    projected = projected.reshape(
        weights.shape[0], 10, count, weights.shape[1]
    )
    return Jet3(
        projected[:, 0] + biases[:, None, :],
        cp.stack((projected[:, 1], projected[:, 2]), axis=-1),
        cp.stack((projected[:, 3], projected[:, 4], projected[:, 5]), axis=-1),
        cp.stack(
            (projected[:, 6], projected[:, 7], projected[:, 8], projected[:, 9]),
            axis=-1,
        ),
    )


def sine_forward(source: Jet3) -> Jet3:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    return unary(source, sine, cosine, -sine, -cosine)


def linear_backward(
    source: Jet3,
    adjoint: Jet3,
    weight: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> tuple[Jet3, cp.ndarray, cp.ndarray]:
    grad_weight = _matmul(adjoint.value.T, source.value, compute_dtype)
    grad_weight += _einsum(
        "noi,nfi->of", adjoint.grad, source.grad, compute_dtype=compute_dtype
    )
    grad_weight += _einsum(
        "noi,nfi->of", adjoint.hess, source.hess, compute_dtype=compute_dtype
    )
    grad_weight += _einsum(
        "noi,nfi->of", adjoint.third, source.third, compute_dtype=compute_dtype
    )
    grad_bias = adjoint.value.sum(axis=0)
    return (
        Jet3(
            _matmul(adjoint.value, weight, compute_dtype),
            _einsum(
                "noi,of->nfi", adjoint.grad, weight, compute_dtype=compute_dtype
            ),
            _einsum(
                "noi,of->nfi", adjoint.hess, weight, compute_dtype=compute_dtype
            ),
            _einsum(
                "noi,of->nfi", adjoint.third, weight, compute_dtype=compute_dtype
            ),
        ),
        grad_weight,
        grad_bias,
    )


def unary_backward(
    source: Jet3,
    adjoint: Jet3,
    first: cp.ndarray,
    second: cp.ndarray,
    third: cp.ndarray,
    fourth: cp.ndarray,
) -> Jet3:
    """Transpose of the compact third-order unary chain rule."""
    gx, gy = source.grad[..., 0], source.grad[..., 1]
    hxx, hxy, hyy = (source.hess[..., i] for i in range(3))
    txxx, txxy, txyy, tyyy = (source.third[..., i] for i in range(4))
    agx, agy = adjoint.grad[..., 0], adjoint.grad[..., 1]
    ahxx, ahxy, ahyy = (adjoint.hess[..., i] for i in range(3))
    atxxx, atxxy, atxyy, atyyy = (adjoint.third[..., i] for i in range(4))

    value = adjoint.value * first
    value += agx * second * gx + agy * second * gy
    value += ahxx * (second * hxx + third * gx * gx)
    value += ahxy * (second * hxy + third * gx * gy)
    value += ahyy * (second * hyy + third * gy * gy)
    value += atxxx * (
        second * txxx + 3.0 * third * gx * hxx + fourth * gx**3
    )
    value += atxxy * (
        second * txxy + third * (gy * hxx + 2.0 * gx * hxy)
        + fourth * gx * gx * gy
    )
    value += atxyy * (
        second * txyy + third * (gx * hyy + 2.0 * gy * hxy)
        + fourth * gx * gy * gy
    )
    value += atyyy * (
        second * tyyy + 3.0 * third * gy * hyy + fourth * gy**3
    )

    grad_x = agx * first
    grad_x += 2.0 * ahxx * second * gx + ahxy * second * gy
    grad_x += atxxx * (3.0 * second * hxx + 3.0 * third * gx * gx)
    grad_x += atxxy * (2.0 * second * hxy + 2.0 * third * gx * gy)
    grad_x += atxyy * (second * hyy + third * gy * gy)
    grad_y = agy * first
    grad_y += ahxy * second * gx + 2.0 * ahyy * second * gy
    grad_y += atxxy * (second * hxx + third * gx * gx)
    grad_y += atxyy * (2.0 * second * hxy + 2.0 * third * gx * gy)
    grad_y += atyyy * (3.0 * second * hyy + 3.0 * third * gy * gy)

    adj_hxx = ahxx * first + atxxx * 3.0 * second * gx + atxxy * second * gy
    adj_hxy = ahxy * first + atxxy * 2.0 * second * gx + atxyy * 2.0 * second * gy
    adj_hyy = ahyy * first + atxyy * second * gx + atyyy * 3.0 * second * gy
    return Jet3(
        value,
        cp.stack((grad_x, grad_y), axis=-1),
        cp.stack((adj_hxx, adj_hxy, adj_hyy), axis=-1),
        adjoint.third * first[..., None],
    )


def sine_backward(source: Jet3, adjoint: Jet3) -> Jet3:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    return unary_backward(source, adjoint, cosine, -sine, -cosine, sine)


def envelope_product_backward(envelope: Jet3, adjoint: Jet3) -> Jet3:
    """Transpose of ``field = envelope * raw`` with fixed envelope."""
    ex, ey = envelope.grad[..., 0], envelope.grad[..., 1]
    exx, exy, eyy = (envelope.hess[..., i] for i in range(3))
    value = adjoint.value * envelope.value
    value += cp.sum(adjoint.grad * envelope.grad, axis=-1)
    value += cp.sum(adjoint.hess * envelope.hess, axis=-1)
    value += cp.sum(adjoint.third * envelope.third, axis=-1)
    grad_x = adjoint.grad[..., 0] * envelope.value
    grad_x += 2.0 * adjoint.hess[..., 0] * ex + adjoint.hess[..., 1] * ey
    grad_x += 3.0 * adjoint.third[..., 0] * exx
    grad_x += 2.0 * adjoint.third[..., 1] * exy
    grad_x += adjoint.third[..., 2] * eyy
    grad_y = adjoint.grad[..., 1] * envelope.value
    grad_y += adjoint.hess[..., 1] * ex + 2.0 * adjoint.hess[..., 2] * ey
    grad_y += adjoint.third[..., 1] * exx
    grad_y += 2.0 * adjoint.third[..., 2] * exy
    grad_y += 3.0 * adjoint.third[..., 3] * eyy
    hxx = adjoint.hess[..., 0] * envelope.value
    hxx += 3.0 * adjoint.third[..., 0] * ex + adjoint.third[..., 1] * ey
    hxy = adjoint.hess[..., 1] * envelope.value
    hxy += 2.0 * adjoint.third[..., 1] * ex + 2.0 * adjoint.third[..., 2] * ey
    hyy = adjoint.hess[..., 2] * envelope.value
    hyy += adjoint.third[..., 2] * ex + 3.0 * adjoint.third[..., 3] * ey
    return Jet3(
        value,
        cp.stack((grad_x, grad_y), axis=-1),
        cp.stack((hxx, hxy, hyy), axis=-1),
        adjoint.third * envelope.value[..., None],
    )


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
        self.weights = [
            arrays[f"{prefix}.l{index}.weight"].astype(F32, copy=False)
            for index in range(layers)
        ]
        self.biases = [
            arrays[f"{prefix}.l{index}.bias"].astype(F32, copy=False)
            for index in range(layers)
        ]
        self.weights.append(
            arrays[f"{prefix}.output.weight"].astype(F32, copy=False)
        )
        self.biases.append(
            arrays[f"{prefix}.output.bias"].astype(F32, copy=False)
        )
        self.shadow_weights = self.weights
        self.shadow_biases = self.biases
        parameter_count = 2 * (layers + 1)
        self.exp_avg = [
            arrays[f"{prefix}.adam.p{index:02d}.exp_avg"].astype(F32, copy=False)
            for index in range(parameter_count)
        ]
        self.exp_avg_sq = [
            arrays[f"{prefix}.adam.p{index:02d}.exp_avg_sq"].astype(F32, copy=False)
            for index in range(parameter_count)
        ]
        self.steps = [
            int(arrays[f"{prefix}.adam.p{index:02d}.step"])
            for index in range(parameter_count)
        ]

    def forward(
        self, cfg: SimpleNamespace, points: np.ndarray, beta: float, alpha: float, cache: bool
    ) -> tuple[Jet3, Any]:
        # FP16 is confined to MLP GEMM operands.  Coordinates, envelope and
        # persistent Jet3 channels remain FP32.
        coordinate_dtype = F32 if self.compute_dtype is F16 else self.compute_dtype
        x, y = xy_jets(points, coordinate_dtype)
        envelope = envelope_jet(cfg, x, y, beta, alpha)
        current = encoding_jet(cfg, x, y, beta, alpha)
        layer_cache: list[tuple[Jet3, Jet3]] = []
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

    def backward(self, cache: Any, field_adjoint: Jet3) -> list[cp.ndarray]:
        envelope, output_input, layer_cache = cache
        raw_adjoint = envelope_product_backward(envelope, field_adjoint)
        backward_dtype = F32
        current, output_weight, output_bias = linear_backward(
            output_input,
            raw_adjoint,
            self.shadow_weights[-1],
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
                self.shadow_weights[index],
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
        # Individual tensors may be exactly zero along a Rayleigh quotient's
        # scale/gauge-invariant directions.  Reject only an all-zero reverse
        # pass below; Adam already leaves an individual zero-gradient tensor
        # unchanged without introducing a new objective or heuristic.
        total = cp.asarray(0.0, dtype=F32)
        for item in gradients:
            total += cp.sum(item.astype(F32, copy=False) ** 2, dtype=F32)
        norm = float(cp.sqrt(total).get())
        if not math.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("zero or non-finite total hardcoded gradient")
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
        fp32_boundary_layers = bool(getattr(cfg, "fp32_boundary_layers", False))
        self.shadow_weight_stacks = [
            cp.stack([mode.weights[layer] for mode in self.modes]).astype(
                F32
                if fp32_boundary_layers and layer in (0, layer_count - 1)
                else F16
                if compute_dtype is F16
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
            layer_cache: list[tuple[Jet3, Jet3]] = []
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


def ellipse_implicit_numpy(
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
    return ellipse_implicit_numpy(cfg, points, beta, alpha) <= 0.0


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
    sampling_mode = str(getattr(cfg, "sampling_mode", ""))
    if sampling_mode in (
        "uniform_area", "uniform_physical_area_sobol", "uniform_fluid_area_sobol"
    ):
        return _sobol_box(
            cfg,
            count,
            (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
            seed,
            beta,
            alpha,
        ).astype(np.float32, copy=False)
    if sampling_mode:
        raise ValueError(
            "unsupported sampling_mode for the analytic pure-vorticity "
            f"backend: {sampling_mode!r}"
        )
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


def vorticity_and_gradient(field: Jet3) -> tuple[cp.ndarray, cp.ndarray]:
    """Return analytic vorticity and its analytic spatial gradient.

    For the divergence-free velocity `u=(psi_y,-psi_x)`,
    `omega=-Delta psi`.  No positional or physical coordinate is shifted:
    every derivative comes from the compact continuous Jet3.
    """
    omega = -(field.hess[..., 0] + field.hess[..., 2])
    grad_omega = cp.stack(
        (
            -(field.third[..., 0] + field.third[..., 2]),
            -(field.third[..., 1] + field.third[..., 3]),
        ),
        axis=-1,
    )
    return omega, grad_omega


def _pure_vorticity_rayleigh_peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    full_diagnostics: bool,
) -> dict[str, Any]:
    """One residual-peeling step for the sole objective K_omega/M_omega.

    M_omega = integral omega**2 and K_omega = integral |grad omega|**2
    are assembled in FP32 on equal-area physical-fluid Sobol samples.  The
    mass projection, reorthogonalization, Rayleigh derivative and rank
    diagnostic are likewise entirely FP32.  There are no neighbour samples,
    finite differences, velocity terms, normalizing scales or regularizers.
    """
    if str(getattr(cfg, "geometry_mode", "")) != "cinf_ellipse":
        raise ValueError("pure vorticity Jet3 requires geometry_mode='cinf_ellipse'")
    if str(getattr(cfg, "envelope_mode", "")).startswith("cinf_") is False:
        raise ValueError("pure vorticity Jet3 requires a smooth-closure envelope")
    if int(getattr(cfg, "spatial_derivative_order", 0)) != 3:
        raise ValueError("pure vorticity Rayleigh requires spatial_derivative_order=3")
    sampling_mode = str(getattr(cfg, "sampling_mode", ""))
    if sampling_mode not in (
        "uniform_area",
        "uniform_physical_area_sobol",
        "uniform_fluid_area_sobol",
    ):
        raise ValueError("pure vorticity Rayleigh requires uniform fluid-area Sobol samples")

    n = len(points)
    if n == 0:
        raise ValueError("pure vorticity Rayleigh received no quadrature points")
    modes = int(cfg.modes)
    if modes != len(model.modes):
        raise ValueError("configuration/model mode count mismatch")
    area = np.float32(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = cp.asarray(area / np.float32(n), dtype=F32)
    mass_floor = cp.asarray(cfg.mass_floor, dtype=F32)
    reorthogonalization_passes = max(
        1, int(getattr(cfg, "reorthogonalization_passes", 2))
    )

    basis_omega = cp.empty((n, modes), dtype=F32)
    basis_grad_omega = cp.empty((2 * n, modes), dtype=F32)
    raw_omega_matrix = (
        cp.empty((n, modes), dtype=F32) if full_diagnostics else None
    )
    rayleighs: list[float] = []
    projection_ratios: list[float] = []
    orth_error = 0.0

    for mode_index, (mode, field, cache) in enumerate(
        model.forward_stream(cfg, points, beta, alpha, training)
    ):
        omega_field, grad_omega_field = vorticity_and_gradient(field)
        omega = omega_field[:, 0].astype(F32, copy=False)
        grad_omega = grad_omega_field[:, 0].astype(F32, copy=False)
        grad_omega_flat = grad_omega.reshape(-1)
        if raw_omega_matrix is not None:
            raw_omega_matrix[:, mode_index] = omega

        residual_omega = omega.copy()
        residual_grad_omega = grad_omega_flat.copy()
        coefficients = cp.zeros((mode_index,), dtype=F32)
        if mode_index:
            previous_omega = basis_omega[:, :mode_index]
            previous_grad_omega = basis_grad_omega[:, :mode_index]
            for _ in range(reorthogonalization_passes):
                increment = weight * (previous_omega.T @ residual_omega)
                coefficients += increment
                residual_omega -= previous_omega @ increment
                residual_grad_omega -= previous_grad_omega @ increment
        else:
            previous_omega = previous_grad_omega = None

        mass = weight * cp.dot(residual_omega, residual_omega)
        stiffness = weight * cp.dot(
            residual_grad_omega, residual_grad_omega
        )
        if not bool(cp.isfinite(mass).get()) or not bool(
            cp.isfinite(stiffness).get()
        ):
            raise FloatingPointError(
                f"mode {mode_index} has non-finite analytic vorticity forms"
            )
        if float(mass.get()) <= float(mass_floor.get()):
            raise FloatingPointError(
                f"mode {mode_index} is unresolved in the vorticity mass metric: "
                f"M={float(mass.get()):.8e}"
            )
        denominator = mass
        rayleigh = stiffness / denominator

        requires_grad = training and mode_index >= int(
            getattr(cfg, "freeze_prefix_modes", 0)
        )
        if requires_grad:
            adjoint_omega = (
                -2.0
                * stiffness
                * weight
                / (denominator * denominator)
            ) * residual_omega
            adjoint_grad_omega = (
                2.0 * weight / denominator
            ) * residual_grad_omega

            if mode_index:
                # Exact transpose of each modified Gram-Schmidt pass.
                for _ in range(reorthogonalization_passes):
                    coefficient_adjoint = -(
                        previous_omega.T @ adjoint_omega
                        + previous_grad_omega.T @ adjoint_grad_omega
                    )
                    adjoint_omega += (
                        weight * (previous_omega @ coefficient_adjoint)
                    )

            adjoint = Jet3(
                cp.zeros_like(field.value, dtype=F32),
                cp.zeros_like(field.grad, dtype=F32),
                cp.zeros_like(field.hess, dtype=F32),
                cp.zeros_like(field.third, dtype=F32),
            )
            # omega = -(psi_xx + psi_yy)
            adjoint.hess[:, 0, 0] = -adjoint_omega
            adjoint.hess[:, 0, 2] = -adjoint_omega
            grad_adjoint = adjoint_grad_omega.reshape(n, 2)
            # omega_x = -(psi_xxx + psi_xyy)
            adjoint.third[:, 0, 0] = -grad_adjoint[:, 0]
            adjoint.third[:, 0, 2] = -grad_adjoint[:, 0]
            # omega_y = -(psi_xxy + psi_yyy)
            adjoint.third[:, 0, 1] = -grad_adjoint[:, 1]
            adjoint.third[:, 0, 3] = -grad_adjoint[:, 1]
            gradients = mode.backward(cache, adjoint)
            mode.adam_step(gradients, cfg)

        normalizer = cp.reciprocal(cp.sqrt(denominator))
        normalized_omega = residual_omega * normalizer
        normalized_grad_omega = residual_grad_omega * normalizer
        basis_omega[:, mode_index] = normalized_omega
        basis_grad_omega[:, mode_index] = normalized_grad_omega

        normalized_mass = mass / denominator
        orth_error = max(
            orth_error, abs(float(normalized_mass.get()) - 1.0)
        )
        if mode_index:
            overlap = weight * (previous_omega.T @ normalized_omega)
            orth_error = max(
                orth_error, float(cp.max(cp.abs(overlap)).get())
            )
        projection_ratios.append(
            float((cp.dot(coefficients, coefficients) / denominator).get())
            if mode_index else 0.0
        )
        rayleighs.append(float(rayleigh.get()))

    if raw_omega_matrix is not None:
        raw_gram = weight * (raw_omega_matrix.T @ raw_omega_matrix)
        raw_gram = np.float32(0.5) * (raw_gram + raw_gram.T)
        eigenvalues = cp.linalg.eigvalsh(raw_gram)
        largest = cp.maximum(cp.abs(eigenvalues[-1]), mass_floor)
        backward_error = cp.asarray(
            np.finfo(np.float32).eps * max(n, modes), dtype=F32
        ) * largest
        effective_rank = int(cp.sum(eigenvalues > backward_error).get())
    else:
        effective_rank = -1

    mean_rayleigh = float(np.mean(np.asarray(rayleighs, dtype=np.float32)))
    return {
        "loss": mean_rayleigh,
        "base_loss": mean_rayleigh,
        "vorticity_rayleigh": mean_rayleigh,
        "mean_energy": mean_rayleigh,
        "max_energy": float(np.max(np.asarray(rayleighs, dtype=np.float32))),
        "min_energy": float(np.min(np.asarray(rayleighs, dtype=np.float32))),
        "mean_projection_ratio": float(
            np.mean(np.asarray(projection_ratios, dtype=np.float32))
        ),
        "orth_error": orth_error,
        "effective_raw_rank": effective_rank,
        "requested_modes": modes,
        "mode_energies": rayleighs,
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
    """Dispatch the one supported continuous pure-vorticity objective."""
    objective = str(getattr(cfg, "basis_objective", ""))
    if objective != "pure_vorticity_rayleigh":
        raise ValueError(
            "this backend only implements basis_objective="
            "'pure_vorticity_rayleigh'"
        )
    if reference_model is not None:
        raise ValueError("pure-vorticity Rayleigh training uses no reference model")
    return _pure_vorticity_rayleigh_peel_step(
        cfg, model, points, beta, alpha, training, full_diagnostics
    )
