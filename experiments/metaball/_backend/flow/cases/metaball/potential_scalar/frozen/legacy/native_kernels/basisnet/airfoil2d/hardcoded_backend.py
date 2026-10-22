










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


def _device_points(
    points: np.ndarray | cp.ndarray,
    dtype: type[cp.generic] = F32,
) -> cp.ndarray:

    if isinstance(points, cp.ndarray):
        if points.dtype != dtype:
            raise TypeError(
                "device-resident quadrature points must already use "
                f"{dtype.__name__}, received {points.dtype}"
            )
        return points
    return cp.asarray(points, dtype=dtype)


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


@dataclass
class Jet2:







    value: cp.ndarray
    grad: cp.ndarray
    hess: cp.ndarray

    @staticmethod
    def constant(
        value: cp.ndarray | float,
        shape: tuple[int, ...],
        dtype: type[cp.generic] = F32,
    ) -> "Jet2":
        result = cp.broadcast_to(cp.asarray(value, dtype=dtype), shape)
        return Jet2(
            result,
            cp.zeros((*shape, 2), dtype=dtype),
            cp.zeros((*shape, 3), dtype=dtype),
        )

    def __add__(self, other: "Jet2" | float) -> "Jet2":
        other = as_jet2(other, self.value.shape, self.value.dtype.type)
        return Jet2(
            self.value + other.value,
            self.grad + other.grad,
            self.hess + other.hess,
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, -self.grad, -self.hess)

    def __sub__(self, other: "Jet2" | float) -> "Jet2":
        return self + (-as_jet2(other, self.value.shape, self.value.dtype.type))

    def __rsub__(self, other: "Jet2" | float) -> "Jet2":
        return as_jet2(other, self.value.shape, self.value.dtype.type) - self

    def __mul__(self, other: "Jet2" | float) -> "Jet2":
        other = as_jet2(other, self.value.shape, self.value.dtype.type)
        av, bv = self.value, other.value
        ax, ay = self.grad[..., 0], self.grad[..., 1]
        bx, by = other.grad[..., 0], other.grad[..., 1]
        axx, axy, ayy = (self.hess[..., index] for index in range(3))
        bxx, bxy, byy = (other.hess[..., index] for index in range(3))
        return Jet2(
            av * bv,
            cp.stack((ax * bv + av * bx, ay * bv + av * by), axis=-1),
            cp.stack(
                (
                    axx * bv + 2.0 * ax * bx + av * bxx,
                    axy * bv + ax * by + ay * bx + av * bxy,
                    ayy * bv + 2.0 * ay * by + av * byy,
                ),
                axis=-1,
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet2":
        inverse = 1.0 / self.value
        return unary2(
            self,
            inverse,
            -(inverse * inverse),
            2.0 * inverse * inverse * inverse,
        )

    def __truediv__(self, other: "Jet2" | float) -> "Jet2":
        return self * as_jet2(
            other, self.value.shape, self.value.dtype.type
        ).reciprocal()

    def square(self) -> "Jet2":
        return self * self

    def pow(self, exponent: int) -> "Jet2":
        result = as_jet2(1.0, self.value.shape, self.value.dtype.type)
        for _ in range(exponent):
            result = result * self
        return result


def as_jet2(
    value: Jet2 | float,
    shape: tuple[int, ...],
    dtype: type[cp.generic] = F32,
) -> Jet2:
    return value if isinstance(value, Jet2) else Jet2.constant(value, shape, dtype)


def unary2(
    jet: Jet2,
    value: cp.ndarray,
    first: cp.ndarray,
    second: cp.ndarray,
) -> Jet2:
    gx, gy = jet.grad[..., 0], jet.grad[..., 1]
    hxx, hxy, hyy = (jet.hess[..., index] for index in range(3))
    return Jet2(
        value,
        first[..., None] * jet.grad,
        cp.stack(
            (
                first * hxx + second * gx * gx,
                first * hxy + second * gx * gy,
                first * hyy + second * gy * gy,
            ),
            axis=-1,
        ),
    )


@dataclass
class Jet1:








    value: cp.ndarray
    grad: cp.ndarray

    @staticmethod
    def constant(
        value: cp.ndarray | float,
        shape: tuple[int, ...],
        dtype: type[cp.generic] = F32,
    ) -> "Jet1":
        result = cp.broadcast_to(cp.asarray(value, dtype=dtype), shape)
        return Jet1(result, cp.zeros((*shape, 2), dtype=dtype))

    def __add__(self, other: "Jet1" | float) -> "Jet1":
        other = as_jet1(other, self.value.shape, self.value.dtype.type)
        return Jet1(self.value + other.value, self.grad + other.grad)

    __radd__ = __add__

    def __neg__(self) -> "Jet1":
        return Jet1(-self.value, -self.grad)

    def __sub__(self, other: "Jet1" | float) -> "Jet1":
        return self + (-as_jet1(other, self.value.shape, self.value.dtype.type))

    def __rsub__(self, other: "Jet1" | float) -> "Jet1":
        return as_jet1(other, self.value.shape, self.value.dtype.type) - self

    def __mul__(self, other: "Jet1" | float) -> "Jet1":
        other = as_jet1(other, self.value.shape, self.value.dtype.type)
        return Jet1(
            self.value * other.value,
            self.grad * other.value[..., None]
            + self.value[..., None] * other.grad,
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet1":
        inverse = 1.0 / self.value
        return Jet1(inverse, -(inverse * inverse)[..., None] * self.grad)

    def __truediv__(self, other: "Jet1" | float) -> "Jet1":
        return self * as_jet1(
            other, self.value.shape, self.value.dtype.type
        ).reciprocal()

    def square(self) -> "Jet1":
        return self * self

    def pow(self, exponent: int) -> "Jet1":
        result = as_jet1(1.0, self.value.shape, self.value.dtype.type)
        for _ in range(exponent):
            result = result * self
        return result


def as_jet1(
    value: Jet1 | float,
    shape: tuple[int, ...],
    dtype: type[cp.generic] = F32,
) -> Jet1:
    return value if isinstance(value, Jet1) else Jet1.constant(value, shape, dtype)


def unary1(jet: Jet1, value: cp.ndarray, first: cp.ndarray) -> Jet1:
    return Jet1(value, first[..., None] * jet.grad)


def xy_jets1(
    points: np.ndarray | cp.ndarray, dtype: type[cp.generic] = F32
) -> tuple[Jet1, Jet1]:
    xy = _device_points(points, dtype)
    count = len(points)
    gx = cp.zeros((count, 1, 2), dtype=dtype)
    gy = cp.zeros_like(gx)
    gx[:, 0, 0] = 1.0
    gy[:, 0, 1] = 1.0
    return Jet1(xy[:, 0:1], gx), Jet1(xy[:, 1:2], gy)


def body_jets1(
    cfg: SimpleNamespace, x: Jet1, y: Jet1, alpha_deg: float
) -> tuple[Jet1, Jet1]:
    angle = math.radians(-alpha_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = x - cfg.quarter_chord_x
    dy = y - cfg.quarter_chord_y
    return (
        cosine * dx + sine * dy + 0.25 * cfg.chord,
        -sine * dx + cosine * dy,
    )


def ellipse_implicit_jet1(
    cfg: SimpleNamespace, xb: Jet1, yb: Jet1, beta: float
) -> Jet1:
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )


def cinf_double_zero_envelope1(level: Jet1) -> Jet1:
    squared = level.square()
    return squared / (1.0 + squared)


def envelope_jet1(
    cfg: SimpleNamespace, x: Jet1, y: Jet1, beta: float, alpha_deg: float
) -> Jet1:
    if not str(getattr(cfg, "envelope_mode", "")).startswith("cinf_"):
        raise ValueError("the Jet1 backend only supports smooth-closure envelopes")
    scale = 0.25 * cfg.chord
    inlet = cinf_double_zero_envelope1((x - cfg.xmin) / scale)
    xb, yb = body_jets1(cfg, x, y, alpha_deg)
    airfoil = cinf_double_zero_envelope1(
        ellipse_implicit_jet1(cfg, xb, yb, beta)
    )
    envelope = inlet * airfoil
    if cfg.envelope_mode == "cinf_farfield_freestream":
        yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
        farfield = cinf_double_zero_envelope1((1.0 - yn.pow(2)) / 0.25)
        envelope = envelope * farfield
    elif cfg.envelope_mode != "cinf_localized":
        raise ValueError(f"Unknown envelope mode: {cfg.envelope_mode}")
    return envelope


def encoding_jet1(
    cfg: SimpleNamespace, x: Jet1, y: Jet1, beta: float, alpha_deg: float
) -> Jet1:
    xn = 2.0 * (x - cfg.xmin) / (cfg.xmax - cfg.xmin) - 1.0
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    input_encoding = str(
        getattr(cfg, "input_encoding", "airfoil_fourier_body_relative")
    )
    if input_encoding == "normalized_raw_xy_plus_geometry":
        beta_feature = Jet1.constant(
            2.0 * beta - 1.0, x.value.shape, x.value.dtype.type
        )
        alpha_feature = Jet1.constant(
            2.0 * (alpha_deg - cfg.alpha_min_deg)
            / (cfg.alpha_max_deg - cfg.alpha_min_deg)
            - 1.0,
            x.value.shape,
            x.value.dtype.type,
        )
        xb, yb = body_jets1(cfg, x, y, alpha_deg)
        features = [
            xn,
            yn,
            beta_feature,
            alpha_feature,
            2.0 * xb / cfg.chord - 1.0,
            2.0 * yb / cfg.chord,
        ]
    elif input_encoding == "airfoil_fourier_body_relative":
        features: list[Jet1] = []
        for spatial in (xn, yn):
            for frequency in 2.0 ** np.arange(cfg.frequencies):
                phase = spatial * float(frequency)
                features.append(
                    unary1(phase, cp.sin(phase.value), cp.cos(phase.value))
                )
        for spatial in (xn, yn):
            for frequency in 2.0 ** np.arange(cfg.frequencies):
                phase = spatial * float(frequency)
                features.append(
                    unary1(phase, cp.cos(phase.value), -cp.sin(phase.value))
                )
        shape = x.value.shape
        features.extend(
            [
                Jet1.constant(2.0 * beta - 1.0, shape, x.value.dtype.type),
                Jet1.constant(
                    2.0 * (alpha_deg - cfg.alpha_min_deg)
                    / (cfg.alpha_max_deg - cfg.alpha_min_deg)
                    - 1.0,
                    shape,
                    x.value.dtype.type,
                ),
            ]
        )
        xb, yb = body_jets1(cfg, x, y, alpha_deg)
        features.extend([2.0 * xb / cfg.chord - 1.0, 2.0 * yb / cfg.chord])
    else:
        raise ValueError(f"unsupported input_encoding={input_encoding!r}")
    return Jet1(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
    )


def xy_jets2(
    points: np.ndarray | cp.ndarray, dtype: type[cp.generic] = F32
) -> tuple[Jet2, Jet2]:
    xy = _device_points(points, dtype)
    count = len(points)
    zero_hess = cp.zeros((count, 1, 3), dtype=dtype)
    gx = cp.zeros((count, 1, 2), dtype=dtype)
    gy = cp.zeros_like(gx)
    gx[:, 0, 0] = 1.0
    gy[:, 0, 1] = 1.0
    return (
        Jet2(xy[:, 0:1], gx, zero_hess.copy()),
        Jet2(xy[:, 1:2], gy, zero_hess),
    )


def body_jets2(
    cfg: SimpleNamespace, x: Jet2, y: Jet2, alpha_deg: float
) -> tuple[Jet2, Jet2]:
    angle = math.radians(-alpha_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = x - cfg.quarter_chord_x
    dy = y - cfg.quarter_chord_y
    return (
        cosine * dx + sine * dy + 0.25 * cfg.chord,
        -sine * dx + cosine * dy,
    )


def ellipse_implicit_jet2(
    cfg: SimpleNamespace, xb: Jet2, yb: Jet2, beta: float
) -> Jet2:
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )


def cinf_double_zero_envelope2(level: Jet2) -> Jet2:
    squared = level.square()
    return squared / (1.0 + squared)


def envelope_jet2(
    cfg: SimpleNamespace, x: Jet2, y: Jet2, beta: float, alpha_deg: float
) -> Jet2:
    if not str(getattr(cfg, "envelope_mode", "")).startswith("cinf_"):
        raise ValueError("the Jet2 backend only supports smooth-closure envelopes")
    scale = 0.25 * cfg.chord
    inlet = cinf_double_zero_envelope2((x - cfg.xmin) / scale)
    xb, yb = body_jets2(cfg, x, y, alpha_deg)
    airfoil = cinf_double_zero_envelope2(
        ellipse_implicit_jet2(cfg, xb, yb, beta)
    )
    envelope = inlet * airfoil
    if cfg.envelope_mode == "cinf_farfield_freestream":
        yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
        farfield = cinf_double_zero_envelope2((1.0 - yn.pow(2)) / 0.25)
        envelope = envelope * farfield
    elif cfg.envelope_mode != "cinf_localized":
        raise ValueError(f"Unknown envelope mode: {cfg.envelope_mode}")
    return envelope


def encoding_jet2(
    cfg: SimpleNamespace, x: Jet2, y: Jet2, beta: float, alpha_deg: float
) -> Jet2:
    xn = 2.0 * (x - cfg.xmin) / (cfg.xmax - cfg.xmin) - 1.0
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    input_encoding = str(
        getattr(cfg, "input_encoding", "airfoil_fourier_body_relative")
    )
    if input_encoding == "normalized_raw_xy_plus_geometry":
        beta_feature = Jet2.constant(
            2.0 * beta - 1.0, x.value.shape, x.value.dtype.type
        )
        alpha_feature = Jet2.constant(
            2.0 * (alpha_deg - cfg.alpha_min_deg)
            / (cfg.alpha_max_deg - cfg.alpha_min_deg)
            - 1.0,
            x.value.shape,
            x.value.dtype.type,
        )
        xb, yb = body_jets2(cfg, x, y, alpha_deg)
        features = [
            xn,
            yn,
            beta_feature,
            alpha_feature,
            2.0 * xb / cfg.chord - 1.0,
            2.0 * yb / cfg.chord,
        ]
    elif input_encoding == "airfoil_fourier_body_relative":
        features: list[Jet2] = []
        for spatial in (xn, yn):
            for frequency in 2.0 ** np.arange(cfg.frequencies):
                phase = spatial * float(frequency)
                sine = cp.sin(phase.value)
                features.append(
                    unary2(phase, sine, cp.cos(phase.value), -sine)
                )
        for spatial in (xn, yn):
            for frequency in 2.0 ** np.arange(cfg.frequencies):
                phase = spatial * float(frequency)
                cosine = cp.cos(phase.value)
                features.append(
                    unary2(phase, cosine, -cp.sin(phase.value), -cosine)
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
        xb, yb = body_jets2(cfg, x, y, alpha_deg)
        features.extend([2.0 * xb / cfg.chord - 1.0, 2.0 * yb / cfg.chord])
    else:
        raise ValueError(f"unsupported input_encoding={input_encoding!r}")
    return Jet2(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
        cp.concatenate([item.hess for item in features], axis=1),
    )


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
    points: np.ndarray | cp.ndarray, dtype: type[cp.generic] = F32
) -> tuple[Jet3, Jet3]:
    xy = _device_points(points, dtype)
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

    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )


def cinf_double_zero_envelope(level: Jet3) -> Jet3:







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
    input_encoding = str(
        getattr(cfg, "input_encoding", "airfoil_fourier_body_relative")
    )
    if input_encoding == "normalized_raw_xy_plus_geometry":
        beta_feature = Jet3.constant(
            2.0 * beta - 1.0, x.value.shape, x.value.dtype.type
        )
        alpha_feature = Jet3.constant(
            2.0 * (alpha_deg - cfg.alpha_min_deg)
            / (cfg.alpha_max_deg - cfg.alpha_min_deg)
            - 1.0,
            x.value.shape,
            x.value.dtype.type,
        )
        xb, yb = body_jets(cfg, x, y, alpha_deg)
        features = [
            xn,
            yn,
            beta_feature,
            alpha_feature,
            2.0 * xb / cfg.chord - 1.0,
            2.0 * yb / cfg.chord,
        ]
        return Jet3(
            cp.concatenate([item.value for item in features], axis=1),
            cp.concatenate([item.grad for item in features], axis=1),
            cp.concatenate([item.hess for item in features], axis=1),
            cp.concatenate([item.third for item in features], axis=1),
        )
    if input_encoding != "airfoil_fourier_body_relative":
        raise ValueError(f"unsupported input_encoding={input_encoding!r}")
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


def _stack_jets(items: list[Jet3]) -> Jet3:

    return Jet3(
        cp.stack([item.value for item in items]),
        cp.stack([item.grad for item in items]),
        cp.stack([item.hess for item in items]),
        cp.stack([item.third for item in items]),
    )


def _slice_jet1(source: Jet1, index: int) -> Jet1:
    return Jet1(source.value[index], source.grad[index])


def _expand_mode_axis_jet1(source: Jet1) -> Jet1:
    return Jet1(source.value[None], source.grad[None])


def _stack_jets1(items: list[Jet1]) -> Jet1:
    return Jet1(
        cp.stack([item.value for item in items]),
        cp.stack([item.grad for item in items]),
    )


def _slice_jet2(source: Jet2, index: int) -> Jet2:
    return Jet2(source.value[index], source.grad[index], source.hess[index])


def _expand_mode_axis_jet2(source: Jet2) -> Jet2:
    return Jet2(source.value[None], source.grad[None], source.hess[None])


def _stack_jets2(items: list[Jet2]) -> Jet2:
    return Jet2(
        cp.stack([item.value for item in items]),
        cp.stack([item.grad for item in items]),
        cp.stack([item.hess for item in items]),
    )


def linear_forward_batch_jet2(
    source: Jet2,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet2:

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
        ),
        axis=point_axis,
    ).astype(compute_dtype, copy=False)
    projected = cp.matmul(
        packed,
        weights.transpose(0, 2, 1).astype(compute_dtype, copy=False),
        dtype=F32,
    ).reshape(weights.shape[0], 6, count, weights.shape[1])
    return Jet2(
        projected[:, 0] + biases[:, None, :],
        cp.stack((projected[:, 1], projected[:, 2]), axis=-1),
        cp.stack((projected[:, 3], projected[:, 4], projected[:, 5]), axis=-1),
    )


def sine_forward_jet2(source: Jet2) -> Jet2:
    sine = cp.sin(source.value)
    return unary2(source, sine, cp.cos(source.value), -sine)


def linear_forward_batch_jet1(
    source: Jet1,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet1:

    batched_source = source.value.ndim == 3
    point_axis = 1 if batched_source else 0
    count = source.value.shape[point_axis]
    packed = cp.concatenate(
        (source.value, source.grad[..., 0], source.grad[..., 1]),
        axis=point_axis,
    ).astype(compute_dtype, copy=False)
    projected = cp.matmul(
        packed,
        weights.transpose(0, 2, 1).astype(compute_dtype, copy=False),
        dtype=F32,
    ).reshape(weights.shape[0], 3, count, weights.shape[1])
    return Jet1(
        projected[:, 0] + biases[:, None, :],
        cp.stack((projected[:, 1], projected[:, 2]), axis=-1),
    )


def sine_forward_jet1(source: Jet1) -> Jet1:
    sine = cp.sin(source.value)
    return Jet1(sine, cp.cos(source.value)[..., None] * source.grad)


def linear_backward_batch_jet1(
    source: Jet1,
    adjoint: Jet1,
    weights: cp.ndarray,
) -> tuple[Jet1, cp.ndarray, cp.ndarray]:

    shared_source = source.value.ndim == 2
    value_expression = "bno,nf->bof" if shared_source else "bno,bnf->bof"
    channel_expression = (
        "bnoi,nfi->bof" if shared_source else "bnoi,bnfi->bof"
    )
    grad_weight = cp.einsum(
        value_expression, adjoint.value, source.value, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.grad, source.grad, dtype=F32
    )
    grad_bias = cp.sum(adjoint.value, axis=1, dtype=F32)
    return (
        Jet1(
            cp.matmul(adjoint.value, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.grad, weights, dtype=F32),
        ),
        grad_weight,
        grad_bias,
    )


def sine_backward_jet1(source: Jet1, adjoint: Jet1) -> Jet1:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    value = adjoint.value * cosine
    value -= sine * cp.sum(adjoint.grad * source.grad, axis=-1)
    return Jet1(value, adjoint.grad * cosine[..., None])


def envelope_product_backward_jet1(envelope: Jet1, adjoint: Jet1) -> Jet1:
    value = adjoint.value * envelope.value
    value += cp.sum(adjoint.grad * envelope.grad, axis=-1)
    return Jet1(value, adjoint.grad * envelope.value[..., None])


def linear_backward_batch_jet2(
    source: Jet2,
    adjoint: Jet2,
    weights: cp.ndarray,
) -> tuple[Jet2, cp.ndarray, cp.ndarray]:

    shared_source = source.value.ndim == 2
    value_expression = "bno,nf->bof" if shared_source else "bno,bnf->bof"
    channel_expression = (
        "bnoi,nfi->bof" if shared_source else "bnoi,bnfi->bof"
    )
    grad_weight = cp.einsum(
        value_expression, adjoint.value, source.value, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.grad, source.grad, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.hess, source.hess, dtype=F32
    )
    grad_bias = cp.sum(adjoint.value, axis=1, dtype=F32)
    return (
        Jet2(
            cp.matmul(adjoint.value, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.grad, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.hess, weights, dtype=F32),
        ),
        grad_weight,
        grad_bias,
    )


def unary_backward_jet2(
    source: Jet2,
    adjoint: Jet2,
    first: cp.ndarray,
    second: cp.ndarray,
    third: cp.ndarray,
) -> Jet2:

    gx, gy = source.grad[..., 0], source.grad[..., 1]
    hxx, hxy, hyy = (source.hess[..., index] for index in range(3))
    agx, agy = adjoint.grad[..., 0], adjoint.grad[..., 1]
    ahxx, ahxy, ahyy = (adjoint.hess[..., index] for index in range(3))

    value = adjoint.value * first
    value += agx * second * gx + agy * second * gy
    value += ahxx * (second * hxx + third * gx * gx)
    value += ahxy * (second * hxy + third * gx * gy)
    value += ahyy * (second * hyy + third * gy * gy)
    grad_x = agx * first
    grad_x += 2.0 * ahxx * second * gx + ahxy * second * gy
    grad_y = agy * first
    grad_y += ahxy * second * gx + 2.0 * ahyy * second * gy
    return Jet2(
        value,
        cp.stack((grad_x, grad_y), axis=-1),
        adjoint.hess * first[..., None],
    )


def sine_backward_jet2(source: Jet2, adjoint: Jet2) -> Jet2:
    sine = cp.sin(source.value)
    cosine = cp.cos(source.value)
    return unary_backward_jet2(source, adjoint, cosine, -sine, -cosine)


def envelope_product_backward_jet2(envelope: Jet2, adjoint: Jet2) -> Jet2:

    ex, ey = envelope.grad[..., 0], envelope.grad[..., 1]
    exx, exy, eyy = (envelope.hess[..., index] for index in range(3))
    value = adjoint.value * envelope.value
    value += cp.sum(adjoint.grad * envelope.grad, axis=-1)
    value += cp.sum(adjoint.hess * envelope.hess, axis=-1)
    grad_x = adjoint.grad[..., 0] * envelope.value
    grad_x += 2.0 * adjoint.hess[..., 0] * ex + adjoint.hess[..., 1] * ey
    grad_y = adjoint.grad[..., 1] * envelope.value
    grad_y += adjoint.hess[..., 1] * ex + 2.0 * adjoint.hess[..., 2] * ey
    return Jet2(
        value,
        cp.stack((grad_x, grad_y), axis=-1),
        adjoint.hess * envelope.value[..., None],
    )


def linear_forward_batch(
    source: Jet3,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> Jet3:

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


def siren_layer_omega(cfg: SimpleNamespace, layer_index: int) -> float:

    if not bool(getattr(cfg, "canonical_siren", False)):
        return 1.0
    return float(
        getattr(
            cfg,
            "siren_first_omega0" if layer_index == 0 else "siren_hidden_omega0",
            30.0,
        )
    )


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


def linear_backward_batch(
    source: Jet3,
    adjoint: Jet3,
    weights: cp.ndarray,
) -> tuple[Jet3, cp.ndarray, cp.ndarray]:






    shared_source = source.value.ndim == 2
    value_expression = "bno,nf->bof" if shared_source else "bno,bnf->bof"
    channel_expression = (
        "bnoi,nfi->bof" if shared_source else "bnoi,bnfi->bof"
    )
    grad_weight = cp.einsum(
        value_expression, adjoint.value, source.value, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.grad, source.grad, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.hess, source.hess, dtype=F32
    )
    grad_weight += cp.einsum(
        channel_expression, adjoint.third, source.third, dtype=F32
    )
    grad_bias = cp.sum(adjoint.value, axis=1, dtype=F32)
    return (
        Jet3(
            cp.matmul(adjoint.value, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.grad, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.hess, weights, dtype=F32),
            cp.einsum("bnoi,bof->bnfi", adjoint.third, weights, dtype=F32),
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
        self.mode_index = mode
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


        coordinate_dtype = F32 if self.compute_dtype is F16 else self.compute_dtype
        x, y = xy_jets(points, coordinate_dtype)
        envelope = envelope_jet(cfg, x, y, beta, alpha)
        current = encoding_jet(cfg, x, y, beta, alpha)
        layer_cache: list[tuple[Jet3, Jet3, float]] = []
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
            omega = siren_layer_omega(cfg, layer_index)
            scaled_preactivation = omega * preactivation
            current = sine_forward(scaled_preactivation)
            if cache:
                layer_cache.append((before, scaled_preactivation, omega))
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
            source, scaled_preactivation, omega = layer_cache[index]
            current = omega * sine_backward(scaled_preactivation, current)
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




        self.master_weight_stacks = [
            cp.stack([mode.weights[layer] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        self.master_bias_stacks = [
            cp.stack([mode.biases[layer] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        self.exp_avg_weight_stacks = [
            cp.stack([mode.exp_avg[2 * layer] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        self.exp_avg_bias_stacks = [
            cp.stack([mode.exp_avg[2 * layer + 1] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        self.exp_avg_sq_weight_stacks = [
            cp.stack([mode.exp_avg_sq[2 * layer] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        self.exp_avg_sq_bias_stacks = [
            cp.stack([mode.exp_avg_sq[2 * layer + 1] for mode in self.modes]).astype(F32)
            for layer in range(layer_count)
        ]
        for mode_index, mode in enumerate(self.modes):
            mode.weights = [
                stack[mode_index] for stack in self.master_weight_stacks
            ]
            mode.biases = [
                stack[mode_index] for stack in self.master_bias_stacks
            ]
            mode.exp_avg = [
                item
                for layer in range(layer_count)
                for item in (
                    self.exp_avg_weight_stacks[layer][mode_index],
                    self.exp_avg_bias_stacks[layer][mode_index],
                )
            ]
            mode.exp_avg_sq = [
                item
                for layer in range(layer_count)
                for item in (
                    self.exp_avg_sq_weight_stacks[layer][mode_index],
                    self.exp_avg_sq_bias_stacks[layer][mode_index],
                )
            ]
        fp32_boundary_layers = bool(getattr(cfg, "fp32_boundary_layers", False))
        self.shadow_weight_stacks = [
            self.master_weight_stacks[layer].astype(
                F32
                if fp32_boundary_layers and layer in (0, layer_count - 1)
                else F16
                if compute_dtype is F16
                else F32
            )
            for layer in range(layer_count)
        ]
        self.shadow_bias_stacks = [
            self.master_bias_stacks[layer].astype(
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

    def forward_stream_jet1(
        self,
        cfg: SimpleNamespace,
        points: np.ndarray,
        beta: float,
        alpha: float,
        cache: bool,
    ):







        block_size = max(1, int(getattr(cfg, "mode_forward_batch", 1)))
        coordinate_dtype = (
            F32
            if self.modes[0].compute_dtype is F16
            else self.modes[0].compute_dtype
        )
        x, y = xy_jets1(points, coordinate_dtype)
        envelope = envelope_jet1(cfg, x, y, beta, alpha)
        encoded = encoding_jet1(cfg, x, y, beta, alpha)
        for start in range(0, len(self.modes), block_size):
            stop = min(start + block_size, len(self.modes))
            current = encoded
            layer_cache: list[tuple[Jet1, Jet1, float]] = []
            for layer in range(cfg.hidden_layers):
                before = current
                preactivation = linear_forward_batch_jet1(
                    before,
                    self.shadow_weight_stacks[layer][start:stop],
                    self.shadow_bias_stacks[layer][start:stop],
                    (
                        F32
                        if self.shadow_weight_stacks[layer].dtype == F32
                        else self.modes[start].compute_dtype
                    ),
                )
                omega = siren_layer_omega(cfg, layer)
                scaled_preactivation = omega * preactivation
                current = sine_forward_jet1(scaled_preactivation)
                if cache:
                    layer_cache.append((before, scaled_preactivation, omega))
            output_input = current
            raw = linear_forward_batch_jet1(
                output_input,
                self.shadow_weight_stacks[-1][start:stop],
                self.shadow_bias_stacks[-1][start:stop],
                (
                    F32
                    if self.shadow_weight_stacks[-1].dtype == F32
                    else self.modes[start].compute_dtype
                ),
            )
            field = _expand_mode_axis_jet1(envelope) * raw
            for local, mode_index in enumerate(range(start, stop)):
                if cache:
                    individual_layers = []
                    for source, scaled_preactivation, omega in layer_cache:
                        individual_source = (
                            source
                            if source.value.ndim == 2
                            else _slice_jet1(source, local)
                        )
                        individual_layers.append(
                            (
                                individual_source,
                                _slice_jet1(scaled_preactivation, local),
                                omega,
                            )
                        )
                    individual_cache = (
                        envelope,
                        _slice_jet1(output_input, local),
                        individual_layers,
                    )
                else:
                    individual_cache = None
                yield (
                    self.modes[mode_index],
                    _slice_jet1(field, local),
                    individual_cache,
                )

    def forward_stream_jet2(
        self,
        cfg: SimpleNamespace,
        points: np.ndarray,
        beta: float,
        alpha: float,
        cache: bool,
    ):

        block_size = max(1, int(getattr(cfg, "mode_forward_batch", 1)))
        coordinate_dtype = (
            F32
            if self.modes[0].compute_dtype is F16
            else self.modes[0].compute_dtype
        )
        x, y = xy_jets2(points, coordinate_dtype)
        envelope = envelope_jet2(cfg, x, y, beta, alpha)
        encoded = encoding_jet2(cfg, x, y, beta, alpha)
        for start in range(0, len(self.modes), block_size):
            stop = min(start + block_size, len(self.modes))
            current = encoded
            layer_cache: list[tuple[Jet2, Jet2, float]] = []
            for layer in range(cfg.hidden_layers):
                before = current
                preactivation = linear_forward_batch_jet2(
                    before,
                    self.shadow_weight_stacks[layer][start:stop],
                    self.shadow_bias_stacks[layer][start:stop],
                    (
                        F32
                        if self.shadow_weight_stacks[layer].dtype == F32
                        else self.modes[start].compute_dtype
                    ),
                )
                omega = siren_layer_omega(cfg, layer)
                scaled_preactivation = omega * preactivation
                current = sine_forward_jet2(scaled_preactivation)
                if cache:
                    layer_cache.append((before, scaled_preactivation, omega))
            output_input = current
            raw = linear_forward_batch_jet2(
                output_input,
                self.shadow_weight_stacks[-1][start:stop],
                self.shadow_bias_stacks[-1][start:stop],
                (
                    F32
                    if self.shadow_weight_stacks[-1].dtype == F32
                    else self.modes[start].compute_dtype
                ),
            )
            field = _expand_mode_axis_jet2(envelope) * raw
            for local, mode_index in enumerate(range(start, stop)):
                if cache:
                    individual_layers = []
                    for source, scaled_preactivation, omega in layer_cache:
                        individual_source = (
                            source
                            if source.value.ndim == 2
                            else _slice_jet2(source, local)
                        )
                        individual_layers.append(
                            (
                                individual_source,
                                _slice_jet2(scaled_preactivation, local),
                                omega,
                            )
                        )
                    individual_cache = (
                        envelope,
                        _slice_jet2(output_input, local),
                        individual_layers,
                    )
                else:
                    individual_cache = None
                yield (
                    self.modes[mode_index],
                    _slice_jet2(field, local),
                    individual_cache,
                )

    def forward_stream(
        self,
        cfg: SimpleNamespace,
        points: np.ndarray,
        beta: float,
        alpha: float,
        cache: bool,
        mode_indices: list[int] | None = None,
    ):
        order = (
            list(range(len(self.modes)))
            if mode_indices is None
            else [int(index) for index in mode_indices]
        )
        if (
            len(order) != len(self.modes)
            or len(set(order)) != len(order)
            or any(index < 0 or index >= len(self.modes) for index in order)
        ):
            raise ValueError("mode_indices must be a permutation of all modes")
        block_size = max(1, int(getattr(cfg, "mode_forward_batch", 1)))





        coordinate_dtype = (
            F32
            if self.modes[order[0]].compute_dtype is F16
            else self.modes[order[0]].compute_dtype
        )
        x, y = xy_jets(points, coordinate_dtype)
        envelope = envelope_jet(cfg, x, y, beta, alpha)
        encoded = encoding_jet(cfg, x, y, beta, alpha)
        for start in range(0, len(order), block_size):
            stop = min(start + block_size, len(order))
            block_indices = order[start:stop]
            contiguous = block_indices == list(
                range(block_indices[0], block_indices[0] + len(block_indices))
            )
            selector: slice | cp.ndarray = (
                slice(block_indices[0], block_indices[0] + len(block_indices))
                if contiguous
                else cp.asarray(block_indices, dtype=cp.int64)
            )
            current = encoded
            layer_cache: list[tuple[Jet3, Jet3, float]] = []
            for layer in range(cfg.hidden_layers):
                before = current
                preactivation = linear_forward_batch(
                    before,
                    self.shadow_weight_stacks[layer][selector],
                    self.shadow_bias_stacks[layer][selector],
                    (
                        F32
                        if self.shadow_weight_stacks[layer].dtype == F32
                        else self.modes[block_indices[0]].compute_dtype
                    ),
                )
                omega = siren_layer_omega(cfg, layer)
                scaled_preactivation = omega * preactivation
                current = sine_forward(scaled_preactivation)
                if cache:
                    layer_cache.append((before, scaled_preactivation, omega))
            output_input = current
            raw = linear_forward_batch(
                output_input,
                self.shadow_weight_stacks[-1][selector],
                self.shadow_bias_stacks[-1][selector],
                (
                    F32
                    if self.shadow_weight_stacks[-1].dtype == F32
                    else self.modes[block_indices[0]].compute_dtype
                ),
            )
            field = _expand_mode_axis(envelope) * raw
            for local, mode_index in enumerate(block_indices):
                if cache:
                    individual_layers = []
                    for source, scaled_preactivation, omega in layer_cache:
                        individual_source = (
                            source
                            if source.value.ndim == 2
                            else _slice_jet(source, local)
                        )
                        individual_layers.append(
                            (
                                individual_source,
                                _slice_jet(scaled_preactivation, local),
                                omega,
                            )
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
        weight_host = [cp.asnumpy(item) for item in self.master_weight_stacks]
        bias_host = [cp.asnumpy(item) for item in self.master_bias_stacks]
        avg_weight_host = [
            cp.asnumpy(item) for item in self.exp_avg_weight_stacks
        ]
        avg_bias_host = [cp.asnumpy(item) for item in self.exp_avg_bias_stacks]
        avg_sq_weight_host = [
            cp.asnumpy(item) for item in self.exp_avg_sq_weight_stacks
        ]
        avg_sq_bias_host = [
            cp.asnumpy(item) for item in self.exp_avg_sq_bias_stacks
        ]
        for mode_index, mode in enumerate(self.modes):
            prefix = f"m{mode_index:03d}"
            for index in range(len(mode.weights) - 1):
                result[f"{prefix}.l{index}.weight"] = weight_host[index][mode_index]
                result[f"{prefix}.l{index}.bias"] = bias_host[index][mode_index]
            result[f"{prefix}.output.weight"] = weight_host[-1][mode_index]
            result[f"{prefix}.output.bias"] = bias_host[-1][mode_index]
            for state_index in range(2 * len(mode.weights)):
                stem = f"{prefix}.adam.p{state_index:02d}"
                layer = state_index // 2
                is_bias = state_index % 2 == 1
                result[f"{stem}.exp_avg"] = (
                    avg_bias_host[layer][mode_index]
                    if is_bias
                    else avg_weight_host[layer][mode_index]
                )
                result[f"{stem}.exp_avg_sq"] = (
                    avg_sq_bias_host[layer][mode_index]
                    if is_bias
                    else avg_sq_weight_host[layer][mode_index]
                )
                result[f"{stem}.step"] = np.asarray(mode.steps[state_index], dtype=np.int64)
        return result

    def backward_gradients_block_jet1(
        self,
        cfg: SimpleNamespace,
        modes: list[HardcodedMode],
        caches: list[Any],
        field_adjoints: list[Jet1],
    ) -> tuple[list[cp.ndarray], list[cp.ndarray]]:

        if not modes:
            return [], []
        mode_indices = [mode.mode_index for mode in modes]
        start = mode_indices[0]
        stop = start + len(mode_indices)
        if mode_indices != list(range(start, stop)):
            raise RuntimeError("Jet1 backward mode block must be contiguous")

        envelope = caches[0][0]
        output_input = _stack_jets1([cache[1] for cache in caches])
        layer_count = len(caches[0][2])
        layers: list[tuple[Jet1, Jet1, float]] = []
        for layer_index in range(layer_count):
            sources = [cache[2][layer_index][0] for cache in caches]
            source = sources[0] if layer_index == 0 else _stack_jets1(sources)
            scaled_preactivation = _stack_jets1(
                [cache[2][layer_index][1] for cache in caches]
            )
            omega = float(caches[0][2][layer_index][2])
            layers.append((source, scaled_preactivation, omega))

        current = envelope_product_backward_jet1(
            envelope, _stack_jets1(field_adjoints)
        )
        current, output_weight, output_bias = linear_backward_batch_jet1(
            output_input,
            current,
            self.shadow_weight_stacks[-1][start:stop].astype(F32),
        )
        weight_grads = [output_weight]
        bias_grads = [output_bias]
        for layer_index in range(layer_count - 1, -1, -1):
            source, scaled_preactivation, omega = layers[layer_index]
            current = omega * sine_backward_jet1(scaled_preactivation, current)
            current, grad_weight, grad_bias = linear_backward_batch_jet1(
                source,
                current,
                self.shadow_weight_stacks[layer_index][start:stop].astype(F32),
            )
            weight_grads.append(grad_weight)
            bias_grads.append(grad_bias)
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads

    def backward_gradients_block_jet2(
        self,
        cfg: SimpleNamespace,
        modes: list[HardcodedMode],
        caches: list[Any],
        field_adjoints: list[Jet2],
    ) -> tuple[list[cp.ndarray], list[cp.ndarray]]:

        if not modes:
            return [], []
        mode_indices = [mode.mode_index for mode in modes]
        start = mode_indices[0]
        stop = start + len(mode_indices)
        if mode_indices != list(range(start, stop)):
            raise RuntimeError("Jet2 backward mode block must be contiguous")

        envelope = caches[0][0]
        output_input = _stack_jets2([cache[1] for cache in caches])
        layer_count = len(caches[0][2])
        layers: list[tuple[Jet2, Jet2, float]] = []
        for layer_index in range(layer_count):
            sources = [cache[2][layer_index][0] for cache in caches]
            source = sources[0] if layer_index == 0 else _stack_jets2(sources)
            scaled_preactivation = _stack_jets2(
                [cache[2][layer_index][1] for cache in caches]
            )
            omega = float(caches[0][2][layer_index][2])
            layers.append((source, scaled_preactivation, omega))

        current = envelope_product_backward_jet2(
            envelope, _stack_jets2(field_adjoints)
        )
        current, output_weight, output_bias = linear_backward_batch_jet2(
            output_input,
            current,
            self.shadow_weight_stacks[-1][start:stop].astype(F32),
        )
        weight_grads = [output_weight]
        bias_grads = [output_bias]
        for layer_index in range(layer_count - 1, -1, -1):
            source, scaled_preactivation, omega = layers[layer_index]
            current = omega * sine_backward_jet2(scaled_preactivation, current)
            current, grad_weight, grad_bias = linear_backward_batch_jet2(
                source,
                current,
                self.shadow_weight_stacks[layer_index][start:stop].astype(F32),
            )
            weight_grads.append(grad_weight)
            bias_grads.append(grad_bias)
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads

    def backward_block_jet1(
        self,
        cfg: SimpleNamespace,
        modes: list[HardcodedMode],
        caches: list[Any],
        field_adjoints: list[Jet1],
    ) -> None:







        weight_grads, bias_grads = self.backward_gradients_block_jet1(
            cfg, modes, caches, field_adjoints
        )
        if not modes:
            return
        full_weight_grads = [cp.zeros_like(stack) for stack in self.master_weight_stacks]
        full_bias_grads = [cp.zeros_like(stack) for stack in self.master_bias_stacks]
        start = modes[0].mode_index
        stop = start + len(modes)
        for target, source in zip(full_weight_grads, weight_grads):
            target[start:stop] = source
        for target, source in zip(full_bias_grads, bias_grads):
            target[start:stop] = source
        active = cp.zeros(len(self.modes), dtype=cp.bool_)
        active[start:stop] = True
        norms, committed = self.adam_step_jet1_device(
            cfg,
            full_weight_grads,
            full_bias_grads,
            active,
            cp.asarray(True),
        )
        check = cp.asnumpy(
            cp.concatenate((norms, committed.astype(F32)[None]))
        )
        if not bool(check[-1]):
            raise FloatingPointError("zero or non-finite total hardcoded gradient")
        for mode in modes:
            mode.steps = [step + 1 for step in mode.steps]

    def adam_step_jet1_device(
        self,
        cfg: SimpleNamespace,
        weight_grads: list[cp.ndarray],
        bias_grads: list[cp.ndarray],
        active_modes: cp.ndarray,
        physical_valid: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:







        if len(weight_grads) != len(self.master_weight_stacks) or len(
            bias_grads
        ) != len(self.master_bias_stacks):
            raise RuntimeError("Jet1 reverse pass returned the wrong gradient count")
        modes = len(self.modes)
        norm_squared = cp.zeros(modes, dtype=F32)
        finite = cp.ones(modes, dtype=cp.bool_)
        for gradient in (*weight_grads, *bias_grads):
            flattened = gradient.reshape(modes, -1).astype(F32, copy=False)
            finite &= cp.all(cp.isfinite(flattened), axis=1)
            norm_squared += cp.sum(flattened * flattened, axis=1, dtype=F32)
        norms = cp.sqrt(norm_squared)
        valid_active = finite & cp.isfinite(norms) & (norms > F32(0.0))
        all_gradients_valid = cp.all((~active_modes) | valid_active)
        committed = cp.asarray(physical_valid, dtype=cp.bool_) & all_gradients_valid
        safe_norms = cp.where(valid_active, norms, F32(1.0))
        scales = cp.minimum(
            F32(1.0), F32(cfg.grad_clip) / cp.maximum(safe_norms, F32(1.0e-30))
        )
        scales = cp.where(active_modes, scales, F32(0.0))
        beta1, beta2, epsilon = F32(0.9), F32(0.999), F32(1.0e-8)

        for layer, (grad_weight, grad_bias) in enumerate(
            zip(weight_grads, bias_grads)
        ):
            for parameter_index, (gradient, parameter, exp_avg, exp_avg_sq) in enumerate(
                (
                    (
                        grad_weight,
                        self.master_weight_stacks[layer],
                        self.exp_avg_weight_stacks[layer],
                        self.exp_avg_sq_weight_stacks[layer],
                    ),
                    (
                        grad_bias,
                        self.master_bias_stacks[layer],
                        self.exp_avg_bias_stacks[layer],
                        self.exp_avg_sq_bias_stacks[layer],
                    ),
                )
            ):
                parameter_slot = 2 * layer + parameter_index
                next_steps = cp.asarray(
                    [mode.steps[parameter_slot] + 1 for mode in self.modes],
                    dtype=F32,
                )
                shape = (modes,) + (1,) * (gradient.ndim - 1)
                scaled_gradient = gradient.astype(F32, copy=False) * scales.reshape(shape)
                candidate_avg = beta1 * exp_avg + (F32(1.0) - beta1) * scaled_gradient
                candidate_avg_sq = (
                    beta2 * exp_avg_sq
                    + (F32(1.0) - beta2) * scaled_gradient * scaled_gradient
                )
                correction1 = F32(1.0) - beta1**next_steps
                correction2 = F32(1.0) - beta2**next_steps
                denominator = (
                    cp.sqrt(candidate_avg_sq / correction2.reshape(shape)) + epsilon
                )
                candidate_parameter = parameter - (
                    F32(cfg.learning_rate) / correction1.reshape(shape)
                ) * candidate_avg / denominator
                update_mask = (committed & active_modes).reshape(shape)
                exp_avg[...] = cp.where(update_mask, candidate_avg, exp_avg)
                exp_avg_sq[...] = cp.where(
                    update_mask, candidate_avg_sq, exp_avg_sq
                )
                parameter[...] = cp.where(
                    update_mask, candidate_parameter, parameter
                )

        for shadow, master in zip(
            self.shadow_weight_stacks, self.master_weight_stacks
        ):
            shadow[...] = master.astype(shadow.dtype, copy=False)
        for shadow, master in zip(
            self.shadow_bias_stacks, self.master_bias_stacks
        ):
            shadow[...] = master.astype(shadow.dtype, copy=False)
        return norms, committed

    def backward_gradients_block(
        self,
        cfg: SimpleNamespace,
        modes: list[HardcodedMode],
        caches: list[Any],
        field_adjoints: list[Jet3],
    ) -> tuple[list[cp.ndarray], list[cp.ndarray]]:








        if not modes:
            return [], []
        mode_indices = [mode.mode_index for mode in modes]
        contiguous = mode_indices == list(
            range(mode_indices[0], mode_indices[0] + len(mode_indices))
        )
        selector: slice | cp.ndarray = (
            slice(mode_indices[0], mode_indices[0] + len(mode_indices))
            if contiguous
            else cp.asarray(mode_indices, dtype=cp.int64)
        )

        envelope = caches[0][0]
        output_input = _stack_jets([cache[1] for cache in caches])
        layer_count = len(caches[0][2])
        layers: list[tuple[Jet3, Jet3, float]] = []
        for layer_index in range(layer_count):
            sources = [cache[2][layer_index][0] for cache in caches]
            source = sources[0] if layer_index == 0 else _stack_jets(sources)
            scaled_preactivation = _stack_jets(
                [cache[2][layer_index][1] for cache in caches]
            )
            omega = float(caches[0][2][layer_index][2])
            layers.append((source, scaled_preactivation, omega))

        current = envelope_product_backward(
            envelope, _stack_jets(field_adjoints)
        )
        current, output_weight, output_bias = linear_backward_batch(
            output_input,
            current,
            self.shadow_weight_stacks[-1][selector].astype(F32),
        )
        weight_grads = [output_weight]
        bias_grads = [output_bias]
        for layer_index in range(layer_count - 1, -1, -1):
            source, scaled_preactivation, omega = layers[layer_index]
            current = omega * sine_backward(scaled_preactivation, current)
            current, grad_weight, grad_bias = linear_backward_batch(
                source,
                current,
                self.shadow_weight_stacks[layer_index][selector].astype(F32),
            )
            weight_grads.append(grad_weight)
            bias_grads.append(grad_bias)
        weight_grads.reverse()
        bias_grads.reverse()
        return weight_grads, bias_grads

    def backward_block(
        self,
        cfg: SimpleNamespace,
        modes: list[HardcodedMode],
        caches: list[Any],
        field_adjoints: list[Jet3],
    ) -> None:

        weight_grads, bias_grads = self.backward_gradients_block(
            cfg, modes, caches, field_adjoints
        )

        for local, mode in enumerate(modes):
            gradients: list[cp.ndarray] = []
            for grad_weight, grad_bias in zip(weight_grads, bias_grads):
                gradients.extend([grad_weight[local], grad_bias[local]])
            mode.adam_step(gradients, cfg)


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
    if sampling_mode == "uniform_random":
        rng = np.random.default_rng(seed)
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            candidates = np.column_stack(
                (
                    rng.uniform(cfg.xmin, cfg.xmax, max(2 * remaining, 128)),
                    rng.uniform(cfg.ymin, cfg.ymax, max(2 * remaining, 128)),
                )
            )
            accepted = candidates[
                ~inside_airfoil(cfg, candidates, beta, alpha)
            ][:remaining]
            pieces.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(pieces).astype(np.float32, copy=False)
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










    del full_diagnostics
    if str(getattr(cfg, "geometry_mode", "")) != "cinf_ellipse":
        raise ValueError("pure vorticity Jet3 requires geometry_mode='cinf_ellipse'")
    if str(getattr(cfg, "envelope_mode", "")).startswith("cinf_") is False:
        raise ValueError("pure vorticity Jet3 requires a smooth-closure envelope")
    if int(getattr(cfg, "spatial_derivative_order", 0)) != 3:
        raise ValueError("pure vorticity Rayleigh requires spatial_derivative_order=3")
    sampling_mode = str(getattr(cfg, "sampling_mode", ""))
    if sampling_mode not in (
        "uniform_random",
        "uniform_area",
        "uniform_physical_area_sobol",
        "uniform_fluid_area_sobol",
    ):
        raise ValueError(
            "pure vorticity Rayleigh requires uniform fluid-area samples"
        )

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
    mass_floor = float(cfg.mass_floor)
    relative_mass_floor = float(
        getattr(cfg, "peeling_relative_mass_floor", 1.0e-4)
    )
    overlap_tolerance = float(
        getattr(cfg, "peeling_overlap_tolerance", 1.0e-3)
    )
    if not 0.0 < relative_mass_floor < 1.0:
        raise ValueError("peeling_relative_mass_floor must be in (0,1)")
    if not 0.0 < overlap_tolerance < 1.0:
        raise ValueError("peeling_overlap_tolerance must be in (0,1)")
    passes = int(getattr(cfg, "reorthogonalization_passes", 2))
    if passes != 2:
        raise ValueError(
            "pure vorticity Rayleigh requires exactly two MGS passes"
        )
    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    if causal_sorting not in {"fixed_order", "geometry_rayleigh"}:
        raise ValueError(f"unsupported causal_sorting {causal_sorting!r}")
    points_device = _device_points(points, F32)
    input_host_to_device_transfers = int(not isinstance(points, cp.ndarray))




    raw_omega = cp.empty((n, modes), dtype=F32)
    raw_gradient = cp.empty((2 * n, modes), dtype=F32)
    for mode, field, _cache in model.forward_stream(
        cfg, points_device, beta, alpha, False
    ):
        network = int(mode.mode_index)
        omega, gradient = vorticity_and_gradient(field)
        raw_omega[:, network] = omega[:, 0].astype(F32, copy=False)
        raw_gradient[:, network] = gradient[:, 0].reshape(-1).astype(
            F32, copy=False
        )
    raw_mass = weight * cp.sum(raw_omega * raw_omega, axis=0, dtype=F32)
    raw_stiffness = weight * cp.sum(
        raw_gradient * raw_gradient, axis=0, dtype=F32
    )
    sorting_rayleigh = raw_stiffness / raw_mass
    causal_order_device = (
        cp.argsort(sorting_rayleigh, kind="stable")
        if causal_sorting == "geometry_rayleigh"
        else cp.arange(modes, dtype=cp.int64)
    )
    sorted_omega = raw_omega[:, causal_order_device]
    sorted_gradient = raw_gradient[:, causal_order_device]




    basis_omega = cp.empty((n, modes), dtype=F32)
    basis_gradient = cp.empty((2 * n, modes), dtype=F32)
    mass_values = cp.empty(modes, dtype=F32)
    stiffness_values = cp.empty(modes, dtype=F32)
    rayleigh_values = cp.empty(modes, dtype=F32)
    projection_values = cp.empty(modes, dtype=F32)
    relative_mass_values = cp.empty(modes, dtype=F32)
    orth_values = cp.empty(modes, dtype=F32)
    first_pass_overlap_values = cp.zeros(modes, dtype=F32)
    adjoint_raw_omega = cp.zeros_like(raw_omega) if training else None
    adjoint_raw_gradient = cp.zeros_like(raw_gradient) if training else None
    freeze_prefix = int(getattr(cfg, "freeze_prefix_modes", 0))
    for causal_rank in range(modes):
        residual_omega = sorted_omega[:, causal_rank].copy()
        residual_gradient = sorted_gradient[:, causal_rank].copy()
        coefficients = cp.zeros((causal_rank,), dtype=F32)
        if causal_rank:
            previous_omega = basis_omega[:, :causal_rank]
            previous_gradient = basis_gradient[:, :causal_rank]
            for pass_index in range(passes):
                increment = weight * (previous_omega.T @ residual_omega)
                if pass_index == 1:
                    first_pass_mass = weight * cp.dot(
                        residual_omega, residual_omega
                    )
                    first_pass_overlap_values[causal_rank] = cp.max(
                        cp.abs(increment)
                    ) / cp.sqrt(cp.maximum(first_pass_mass, F32(mass_floor)))
                coefficients += increment
                residual_omega -= previous_omega @ increment
                residual_gradient -= previous_gradient @ increment
        else:
            previous_omega = previous_gradient = None

        mass = weight * cp.dot(residual_omega, residual_omega)
        stiffness = weight * cp.dot(residual_gradient, residual_gradient)
        network = causal_order_device[causal_rank]
        relative_mass = mass / raw_mass[network]
        safe_mass = cp.maximum(mass, F32(mass_floor))
        rayleigh = stiffness / safe_mass
        mass_values[causal_rank] = mass
        stiffness_values[causal_rank] = stiffness
        rayleigh_values[causal_rank] = rayleigh
        relative_mass_values[causal_rank] = relative_mass

        if training and causal_rank >= freeze_prefix:
            adjoint_omega = (
                -2.0 * stiffness * weight / (safe_mass * safe_mass)
            ) * residual_omega
            adjoint_gradient = (2.0 * weight / safe_mass) * residual_gradient
            if causal_rank:


                for _ in range(passes):
                    coefficient_adjoint = -(
                        previous_omega.T @ adjoint_omega
                        + previous_gradient.T @ adjoint_gradient
                    )
                    adjoint_omega += weight * (
                        previous_omega @ coefficient_adjoint
                    )
            if adjoint_raw_omega is None or adjoint_raw_gradient is None:
                raise RuntimeError("Jet3 adjoint buffers were not allocated")
            adjoint_raw_omega[:, network] = adjoint_omega
            adjoint_raw_gradient[:, network] = adjoint_gradient

        normalizer = cp.reciprocal(cp.sqrt(safe_mass))
        normalized_omega = residual_omega * normalizer
        normalized_gradient = residual_gradient * normalizer
        basis_omega[:, causal_rank] = normalized_omega
        basis_gradient[:, causal_rank] = normalized_gradient
        normalized_mass = weight * cp.dot(normalized_omega, normalized_omega)
        orth_value = cp.abs(normalized_mass - F32(1.0))
        if causal_rank:
            overlap = weight * (previous_omega.T @ normalized_omega)
            orth_value = cp.maximum(orth_value, cp.max(cp.abs(overlap)))
        orth_values[causal_rank] = orth_value
        projection_values[causal_rank] = (
            cp.dot(coefficients, coefficients) / safe_mass
            if causal_rank
            else F32(0.0)
        )

    peeled_by_network_device = cp.empty(modes, dtype=F32)
    peeled_by_network_device[causal_order_device] = rayleigh_values
    physical_valid_device = cp.all(
        cp.isfinite(raw_mass)
        & cp.isfinite(raw_stiffness)
        & cp.isfinite(sorting_rayleigh)
        & (raw_mass > F32(mass_floor))
    ) & cp.all(
        cp.isfinite(mass_values)
        & cp.isfinite(stiffness_values)
        & cp.isfinite(rayleigh_values)
        & cp.isfinite(projection_values)
        & cp.isfinite(relative_mass_values)
        & cp.isfinite(orth_values)
        & cp.isfinite(first_pass_overlap_values)
        & (mass_values > F32(mass_floor))
        & (relative_mass_values >= F32(relative_mass_floor))
        & (orth_values <= F32(overlap_tolerance))
    )




    gradient_norms_device = cp.zeros(modes, dtype=F32)
    optimizer_committed_device = physical_valid_device
    active_modes_device = cp.zeros(modes, dtype=cp.bool_)
    active_modes_device[causal_order_device[freeze_prefix:]] = True
    if training:
        if adjoint_raw_omega is None or adjoint_raw_gradient is None:
            raise RuntimeError("Jet3 training adjoints are unavailable")
        backward_batch = max(1, int(getattr(cfg, "mode_backward_batch", 1)))
        all_weight_grads = [
            cp.zeros_like(stack) for stack in model.master_weight_stacks
        ]
        all_bias_grads = [
            cp.zeros_like(stack) for stack in model.master_bias_stacks
        ]
        backward_modes: list[HardcodedMode] = []
        backward_caches: list[Any] = []
        backward_adjoints: list[Jet3] = []

        def flush_backward_block() -> None:
            if backward_modes:
                block_weight_grads, block_bias_grads = (
                    model.backward_gradients_block(
                        cfg, backward_modes, backward_caches, backward_adjoints
                    )
                )
                start = backward_modes[0].mode_index
                stop = start + len(backward_modes)
                for target, source in zip(all_weight_grads, block_weight_grads):
                    target[start:stop] = source
                for target, source in zip(all_bias_grads, block_bias_grads):
                    target[start:stop] = source
                backward_modes.clear()
                backward_caches.clear()
                backward_adjoints.clear()

        for mode, field, cache in model.forward_stream(
            cfg, points_device, beta, alpha, True
        ):
            network = int(mode.mode_index)
            adjoint = Jet3(
                cp.zeros_like(field.value, dtype=F32),
                cp.zeros_like(field.grad, dtype=F32),
                cp.zeros_like(field.hess, dtype=F32),
                cp.zeros_like(field.third, dtype=F32),
            )
            omega_adjoint = adjoint_raw_omega[:, network]
            gradient_adjoint = adjoint_raw_gradient[:, network].reshape(n, 2)

            adjoint.hess[:, 0, 0] = -omega_adjoint
            adjoint.hess[:, 0, 2] = -omega_adjoint

            adjoint.third[:, 0, 0] = -gradient_adjoint[:, 0]
            adjoint.third[:, 0, 2] = -gradient_adjoint[:, 0]
            adjoint.third[:, 0, 1] = -gradient_adjoint[:, 1]
            adjoint.third[:, 0, 3] = -gradient_adjoint[:, 1]
            backward_modes.append(mode)
            backward_caches.append(cache)
            backward_adjoints.append(adjoint)
            if len(backward_modes) == backward_batch:
                flush_backward_block()
        flush_backward_block()
        gradient_norms_device, optimizer_committed_device = (
            model.adam_step_jet1_device(
                cfg,
                all_weight_grads,
                all_bias_grads,
                active_modes_device,
                physical_valid_device,
            )
        )



    diagnostic_rows = cp.asnumpy(
        cp.stack(
            (
                raw_mass,
                raw_stiffness,
                sorting_rayleigh,
                mass_values,
                stiffness_values,
                rayleigh_values,
                projection_values,
                relative_mass_values,
                orth_values,
                first_pass_overlap_values,
                peeled_by_network_device,
                causal_order_device.astype(F32),
                gradient_norms_device,
                cp.broadcast_to(
                    optimizer_committed_device.astype(F32), (modes,)
                ),
            )
        )
    )
    (
        raw_mass_host,
        raw_stiffness_host,
        sorting_rayleigh_host,
        mass_host,
        stiffness_host,
        rayleigh_host,
        projection_host,
        relative_mass_host,
        orth_host,
        first_pass_overlap_host,
        peeled_by_network_host,
        causal_order_float,
        gradient_norms_host,
        optimizer_committed_host,
    ) = diagnostic_rows
    causal_order = causal_order_float.astype(np.int64)
    invalid_raw = ~(
        np.isfinite(raw_mass_host)
        & np.isfinite(raw_stiffness_host)
        & np.isfinite(sorting_rayleigh_host)
    )
    unresolved_raw = raw_mass_host <= mass_floor
    if np.any(invalid_raw | unresolved_raw):
        network = int(np.flatnonzero(invalid_raw | unresolved_raw)[0])
        raise FloatingPointError(
            "causal sorting received an invalid vorticity candidate "
            f"{network + 1}: M={raw_mass_host[network]:.8e}, "
            f"K={raw_stiffness_host[network]:.8e}"
        )
    invalid_peeled = ~(
        np.isfinite(mass_host)
        & np.isfinite(stiffness_host)
        & np.isfinite(rayleigh_host)
        & np.isfinite(projection_host)
        & np.isfinite(relative_mass_host)
        & np.isfinite(orth_host)
        & np.isfinite(first_pass_overlap_host)
    )
    collapsed = (mass_host <= mass_floor) | (
        relative_mass_host < relative_mass_floor
    )
    nonorthogonal = orth_host > overlap_tolerance
    rejected = invalid_peeled | collapsed | nonorthogonal
    if np.any(rejected):
        causal_rank = int(np.flatnonzero(rejected)[0])
        network = int(causal_order[causal_rank])
        raise FloatingPointError(
            f"vorticity candidate {network + 1} (causal rank "
            f"{causal_rank + 1}) rejected after the two-pass peel: "
            f"relative_mass={relative_mass_host[causal_rank]:.8e} "
            f"(floor={relative_mass_floor:.8e}), "
            f"overlap={orth_host[causal_rank]:.8e} "
            f"(tolerance={overlap_tolerance:.8e}); no parameters updated"
        )
    if training and not bool(optimizer_committed_host[0]):
        active_host = np.zeros(modes, dtype=bool)
        active_host[causal_order[freeze_prefix:]] = True
        invalid_gradient = active_host & (
            ~np.isfinite(gradient_norms_host) | (gradient_norms_host <= 0.0)
        )
        network = (
            int(np.flatnonzero(invalid_gradient)[0])
            if np.any(invalid_gradient)
            else -1
        )
        raise FloatingPointError(
            "zero or non-finite total hardcoded gradient"
            + (f" for network mode {network + 1}" if network >= 0 else "")
        )
    if training:
        active_networks = set(causal_order[freeze_prefix:].astype(int).tolist())
        for network, mode in enumerate(model.modes):
            if network in active_networks:
                mode.steps = [step + 1 for step in mode.steps]

    rayleighs = rayleigh_host.astype(np.float32, copy=False).tolist()
    sorting_by_network = sorting_rayleigh_host.astype(
        np.float32, copy=False
    ).tolist()
    peeled_by_network = peeled_by_network_host.astype(
        np.float32, copy=False
    ).tolist()
    mean_rayleigh = float(np.mean(rayleigh_host, dtype=np.float32))
    return {
        "loss": mean_rayleigh,
        "base_loss": mean_rayleigh,
        "vorticity_rayleigh": mean_rayleigh,
        "mean_energy": mean_rayleigh,
        "max_energy": float(np.max(rayleigh_host)),
        "min_energy": float(np.min(rayleigh_host)),
        "mean_projection_ratio": float(np.mean(projection_host, dtype=np.float32)),
        "orth_error": float(np.max(orth_host)),
        "peeled_rank": modes,
        "requested_modes": modes,
        "mode_energies": rayleighs,
        "mode_energies_by_network": peeled_by_network,
        "causal_sorting": causal_sorting,
        "causal_order": causal_order.astype(int).tolist(),
        "causal_order_1based": (causal_order + 1).astype(int).tolist(),
        "raw_mode_energies": sorting_by_network,
        "sorting_rayleigh_by_network": sorting_by_network,
        "spatial_jet_order_used": 3,
        "third_derivative_channels_allocated": 4,
        "peeling_passes_used": passes,
        "shared_triangular_transform_across_derivatives": True,
        "peeling_relative_mass_floor": relative_mass_floor,
        "peeling_overlap_tolerance": overlap_tolerance,
        "min_relative_residual_mass": float(np.min(relative_mass_host)),
        "max_post_first_pass_overlap": float(
            np.max(first_pass_overlap_host)
        ),
        "max_post_peel_overlap": float(np.max(orth_host)),
        "rejected_modes": 0,
        "causal_sort_device": "gpu",
        "diagnostic_host_transfers_per_step": 1,
        "optimizer_device": "gpu_batched_all_modes",
        "optimizer_host_transfers_per_step": 0,
        "input_host_to_device_transfers_per_step": (
            input_host_to_device_transfers
        ),
        "sample_pack_device": (
            "gpu_resident" if input_host_to_device_transfers == 0
            else "host_per_step"
        ),
        "gradient_norm_min": (
            float(np.min(gradient_norms_host)) if training else float("nan")
        ),
        "gradient_norm_max": (
            float(np.max(gradient_norms_host)) if training else float("nan")
        ),
    }


def _velocity_rayleigh_peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    full_diagnostics: bool,
) -> dict[str, Any]:
















    del full_diagnostics
    if str(getattr(cfg, "geometry_mode", "")) != "cinf_ellipse":
        raise ValueError(
            "velocity Rayleigh Jet training requires a smooth hard-boundary "
            "geometry hook"
        )
    if str(getattr(cfg, "envelope_mode", "")).startswith("cinf_") is False:
        raise ValueError("velocity Rayleigh training requires a smooth envelope")
    if int(getattr(cfg, "spatial_derivative_order", 0)) < 2:
        raise ValueError("velocity Rayleigh requires analytic second derivatives")
    passes = int(getattr(cfg, "reorthogonalization_passes", 2))
    if passes != 2:
        raise ValueError(
            "velocity Rayleigh requires exactly two causal MGS passes"
        )
    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    if causal_sorting not in {"fixed_order", "geometry_rayleigh"}:
        raise ValueError(f"unsupported causal_sorting {causal_sorting!r}")

    n = len(points)
    if n == 0:
        raise ValueError("velocity Rayleigh received no quadrature points")
    modes = int(cfg.modes)
    if modes != len(model.modes):
        raise ValueError("configuration/model mode count mismatch")
    area = np.float32(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = cp.asarray(area / np.float32(n), dtype=F32)
    mass_floor = float(cfg.mass_floor)
    relative_mass_floor = float(
        getattr(cfg, "peeling_relative_mass_floor", 1.0e-4)
    )
    overlap_tolerance = float(
        getattr(cfg, "peeling_overlap_tolerance", 1.0e-3)
    )
    if not 0.0 < relative_mass_floor < 1.0:
        raise ValueError("peeling_relative_mass_floor must be in (0,1)")
    if not 0.0 < overlap_tolerance < 1.0:
        raise ValueError("peeling_overlap_tolerance must be in (0,1)")
    points_device = _device_points(points, F32)
    input_host_to_device_transfers = int(not isinstance(points, cp.ndarray))




    raw_gradient = cp.empty((2 * n, modes), dtype=F32)
    raw_hessian = cp.empty((3 * n, modes), dtype=F32)
    for mode, field, _cache in model.forward_stream_jet2(
        cfg, points_device, beta, alpha, False
    ):
        network = int(mode.mode_index)
        raw_gradient[:, network] = field.grad[:, 0].reshape(-1).astype(
            F32, copy=False
        )
        raw_hessian[:, network] = field.hess[:, 0].reshape(-1).astype(
            F32, copy=False
        )
    hessian_metric = cp.tile(cp.asarray((1.0, 2.0, 1.0), dtype=F32), n)
    raw_mass = weight * cp.sum(raw_gradient * raw_gradient, axis=0, dtype=F32)
    raw_stiffness = weight * cp.sum(
        hessian_metric[:, None] * raw_hessian * raw_hessian,
        axis=0,
        dtype=F32,
    )
    sorting_rayleigh = raw_stiffness / raw_mass
    causal_order_device = (
        cp.argsort(sorting_rayleigh, kind="stable")
        if causal_sorting == "geometry_rayleigh"
        else cp.arange(modes, dtype=cp.int64)
    )
    sorted_gradient = raw_gradient[:, causal_order_device]
    sorted_hessian = raw_hessian[:, causal_order_device]




    basis_gradient = cp.empty((2 * n, modes), dtype=F32)
    basis_hessian = cp.empty((3 * n, modes), dtype=F32)
    mass_values = cp.empty(modes, dtype=F32)
    stiffness_values = cp.empty(modes, dtype=F32)
    rayleigh_values = cp.empty(modes, dtype=F32)
    projection_values = cp.empty(modes, dtype=F32)
    relative_mass_values = cp.empty(modes, dtype=F32)
    orth_values = cp.empty(modes, dtype=F32)
    first_pass_overlap_values = cp.zeros(modes, dtype=F32)
    adjoint_raw_gradient = cp.zeros_like(raw_gradient) if training else None
    adjoint_raw_hessian = cp.zeros_like(raw_hessian) if training else None
    freeze_prefix = int(getattr(cfg, "freeze_prefix_modes", 0))

    for causal_rank in range(modes):
        residual_gradient = sorted_gradient[:, causal_rank].copy()
        residual_hessian = sorted_hessian[:, causal_rank].copy()
        coefficients = cp.zeros((causal_rank,), dtype=F32)
        if causal_rank:
            previous_gradient = basis_gradient[:, :causal_rank]
            previous_hessian = basis_hessian[:, :causal_rank]
            for pass_index in range(passes):
                increment = weight * (
                    previous_gradient.T @ residual_gradient
                )
                if pass_index == 1:
                    first_pass_mass = weight * cp.dot(
                        residual_gradient, residual_gradient
                    )
                    first_pass_overlap_values[causal_rank] = cp.max(
                        cp.abs(increment)
                    ) / cp.sqrt(cp.maximum(first_pass_mass, F32(mass_floor)))
                coefficients += increment
                residual_gradient -= previous_gradient @ increment


                residual_hessian -= previous_hessian @ increment
        else:
            previous_gradient = previous_hessian = None

        mass = weight * cp.dot(residual_gradient, residual_gradient)
        stiffness = weight * cp.dot(
            hessian_metric * residual_hessian, residual_hessian
        )
        network = causal_order_device[causal_rank]
        relative_mass = mass / raw_mass[network]
        safe_mass = cp.maximum(mass, F32(mass_floor))
        rayleigh = stiffness / safe_mass
        mass_values[causal_rank] = mass
        stiffness_values[causal_rank] = stiffness
        rayleigh_values[causal_rank] = rayleigh
        relative_mass_values[causal_rank] = relative_mass

        requires_grad = training and causal_rank >= freeze_prefix
        if requires_grad:
            adjoint_gradient = (
                -2.0 * stiffness * weight / (safe_mass * safe_mass)
            ) * residual_gradient
            adjoint_hessian = (
                2.0 * weight / safe_mass
            ) * hessian_metric * residual_hessian
            if causal_rank:





                for _ in range(passes):
                    coefficient_adjoint = -(
                        previous_gradient.T @ adjoint_gradient
                        + previous_hessian.T @ adjoint_hessian
                    )
                    adjoint_gradient += weight * (
                        previous_gradient @ coefficient_adjoint
                    )
            if adjoint_raw_gradient is None or adjoint_raw_hessian is None:
                raise RuntimeError("velocity adjoint buffers were not allocated")
            adjoint_raw_gradient[:, network] = adjoint_gradient
            adjoint_raw_hessian[:, network] = adjoint_hessian

        normalizer = cp.reciprocal(cp.sqrt(safe_mass))
        normalized_gradient = residual_gradient * normalizer
        normalized_hessian = residual_hessian * normalizer
        basis_gradient[:, causal_rank] = normalized_gradient
        basis_hessian[:, causal_rank] = normalized_hessian
        normalized_mass = weight * cp.dot(
            normalized_gradient, normalized_gradient
        )
        orth_value = cp.abs(normalized_mass - F32(1.0))
        if causal_rank:
            overlap = weight * (previous_gradient.T @ normalized_gradient)
            orth_value = cp.maximum(orth_value, cp.max(cp.abs(overlap)))
        orth_values[causal_rank] = orth_value
        projection_values[causal_rank] = (
            cp.dot(coefficients, coefficients) / safe_mass
            if causal_rank
            else F32(0.0)
        )




    raw_valid_device = cp.all(
        cp.isfinite(raw_mass)
        & cp.isfinite(raw_stiffness)
        & cp.isfinite(sorting_rayleigh)
        & (raw_mass > F32(mass_floor))
    )
    peeled_valid_device = cp.all(
        cp.isfinite(mass_values)
        & cp.isfinite(stiffness_values)
        & cp.isfinite(rayleigh_values)
        & cp.isfinite(projection_values)
        & cp.isfinite(relative_mass_values)
        & cp.isfinite(orth_values)
        & cp.isfinite(first_pass_overlap_values)
        & (mass_values > F32(mass_floor))
        & (relative_mass_values >= F32(relative_mass_floor))
        & (orth_values <= F32(overlap_tolerance))
    )
    physical_valid_device = raw_valid_device & peeled_valid_device




    gradient_norms_device = cp.zeros(modes, dtype=F32)
    optimizer_committed_device = physical_valid_device
    active_modes_device = cp.zeros(modes, dtype=cp.bool_)
    active_modes_device[causal_order_device[freeze_prefix:]] = True
    if training:
        if adjoint_raw_gradient is None or adjoint_raw_hessian is None:
            raise RuntimeError("velocity training adjoints are unavailable")
        all_weight_grads = [
            cp.zeros_like(stack) for stack in model.master_weight_stacks
        ]
        all_bias_grads = [
            cp.zeros_like(stack) for stack in model.master_bias_stacks
        ]
        backward_batch = max(1, int(getattr(cfg, "mode_backward_batch", 1)))
        backward_modes: list[HardcodedMode] = []
        backward_caches: list[Any] = []
        backward_adjoints: list[Jet2] = []

        def flush_backward_block() -> None:
            if not backward_modes:
                return
            block_weight_grads, block_bias_grads = (
                model.backward_gradients_block_jet2(
                    cfg, backward_modes, backward_caches, backward_adjoints
                )
            )
            start = backward_modes[0].mode_index
            stop = start + len(backward_modes)
            for target, source in zip(all_weight_grads, block_weight_grads):
                target[start:stop] = source
            for target, source in zip(all_bias_grads, block_bias_grads):
                target[start:stop] = source
            backward_modes.clear()
            backward_caches.clear()
            backward_adjoints.clear()

        for mode, field, cache in model.forward_stream_jet2(
            cfg, points_device, beta, alpha, True
        ):
            network = int(mode.mode_index)
            adjoint = Jet2(
                cp.zeros_like(field.value, dtype=F32),
                cp.zeros_like(field.grad, dtype=F32),
                cp.zeros_like(field.hess, dtype=F32),
            )
            adjoint.grad[:, 0] = adjoint_raw_gradient[:, network].reshape(n, 2)
            adjoint.hess[:, 0] = adjoint_raw_hessian[:, network].reshape(n, 3)
            backward_modes.append(mode)
            backward_caches.append(cache)
            backward_adjoints.append(adjoint)
            if len(backward_modes) == backward_batch:
                flush_backward_block()
        flush_backward_block()
        gradient_norms_device, optimizer_committed_device = (
            model.adam_step_jet1_device(
                cfg,
                all_weight_grads,
                all_bias_grads,
                active_modes_device,
                physical_valid_device,
            )
        )

    peeled_by_network_device = cp.empty(modes, dtype=F32)
    peeled_by_network_device[causal_order_device] = rayleigh_values
    diagnostic_rows = cp.asnumpy(
        cp.stack(
            (
                raw_mass,
                raw_stiffness,
                sorting_rayleigh,
                mass_values,
                stiffness_values,
                rayleigh_values,
                projection_values,
                relative_mass_values,
                orth_values,
                first_pass_overlap_values,
                causal_order_device.astype(F32),
                peeled_by_network_device,
                gradient_norms_device,
                cp.broadcast_to(
                    optimizer_committed_device.astype(F32), (modes,)
                ),
            )
        )
    )
    (
        raw_mass_host,
        raw_stiffness_host,
        sorting_rayleigh_host,
        mass_host,
        stiffness_host,
        rayleigh_host,
        projection_host,
        relative_mass_host,
        orth_host,
        first_pass_overlap_host,
        causal_order_float,
        peeled_by_network_host,
        gradient_norms_host,
        committed_host,
    ) = diagnostic_rows
    causal_order = causal_order_float.astype(np.int64)
    invalid_raw = ~(
        np.isfinite(raw_mass_host)
        & np.isfinite(raw_stiffness_host)
        & np.isfinite(sorting_rayleigh_host)
        & (raw_mass_host > mass_floor)
    )
    if np.any(invalid_raw):
        network = int(np.flatnonzero(invalid_raw)[0])
        raise FloatingPointError(
            "causal sorting received an invalid velocity candidate "
            f"{network + 1}: M={raw_mass_host[network]:.8e}, "
            f"K={raw_stiffness_host[network]:.8e}"
        )
    invalid_peeled = ~(
        np.isfinite(mass_host)
        & np.isfinite(stiffness_host)
        & np.isfinite(rayleigh_host)
        & np.isfinite(projection_host)
        & np.isfinite(relative_mass_host)
        & np.isfinite(orth_host)
        & np.isfinite(first_pass_overlap_host)
    )
    collapsed = (mass_host <= mass_floor) | (
        relative_mass_host < relative_mass_floor
    )
    nonorthogonal = orth_host > overlap_tolerance
    rejected = invalid_peeled | collapsed | nonorthogonal
    if np.any(rejected):
        causal_rank = int(np.flatnonzero(rejected)[0])
        network = int(causal_order[causal_rank])
        raise FloatingPointError(
            f"velocity candidate {network + 1} (causal rank "
            f"{causal_rank + 1}) rejected after the two-pass peel: "
            f"relative_mass={relative_mass_host[causal_rank]:.8e} "
            f"(floor={relative_mass_floor:.8e}), "
            f"overlap={orth_host[causal_rank]:.8e} "
            f"(tolerance={overlap_tolerance:.8e}); no parameters updated"
        )
    if training and not bool(committed_host[0]):
        invalid_gradient = (
            ~np.isfinite(gradient_norms_host) | (gradient_norms_host <= 0.0)
        )
        network = (
            int(np.flatnonzero(invalid_gradient)[0])
            if np.any(invalid_gradient)
            else -1
        )
        raise FloatingPointError(
            "zero or non-finite total hardcoded velocity gradient"
            + (f" for network mode {network + 1}" if network >= 0 else "")
        )
    if training:
        active_networks = set(causal_order[freeze_prefix:].astype(int).tolist())
        for network, mode in enumerate(model.modes):
            if network in active_networks:
                mode.steps = [step + 1 for step in mode.steps]

    mean_rayleigh = float(np.mean(rayleigh_host, dtype=np.float32))
    return {
        "loss": mean_rayleigh,
        "base_loss": mean_rayleigh,
        "velocity_rayleigh": mean_rayleigh,
        "mean_energy": mean_rayleigh,
        "max_energy": float(np.max(rayleigh_host)),
        "min_energy": float(np.min(rayleigh_host)),
        "mean_projection_ratio": float(
            np.mean(projection_host, dtype=np.float32)
        ),
        "orth_error": float(np.max(orth_host)),
        "peeled_rank": modes,
        "requested_modes": modes,
        "effective_velocity_rank": modes,
        "mode_energies": rayleigh_host.astype(np.float32, copy=False).tolist(),
        "mode_energies_by_network": peeled_by_network_host.astype(
            np.float32, copy=False
        ).tolist(),
        "causal_sorting": causal_sorting,
        "causal_order": causal_order.astype(int).tolist(),
        "causal_order_1based": (causal_order + 1).astype(int).tolist(),
        "sorting_rayleigh_by_network": sorting_rayleigh_host.astype(
            np.float32, copy=False
        ).tolist(),
        "spatial_jet_order_used": 2,
        "third_derivative_channels_allocated": 0,
        "peeling_passes_used": passes,
        "shared_triangular_transform_across_derivatives": True,
        "peeling_relative_mass_floor": relative_mass_floor,
        "peeling_overlap_tolerance": overlap_tolerance,
        "min_relative_residual_mass": float(np.min(relative_mass_host)),
        "max_post_first_pass_overlap": float(
            np.max(first_pass_overlap_host)
        ),
        "max_post_peel_overlap": float(np.max(orth_host)),
        "rejected_modes": 0,
        "causal_sort_device": "gpu",
        "diagnostic_host_transfers_per_step": 1,
        "optimizer_device": "gpu_batched_all_modes",
        "optimizer_host_transfers_per_step": 0,
        "input_host_to_device_transfers_per_step": (
            input_host_to_device_transfers
        ),
        "sample_pack_device": (
            "gpu_resident" if input_host_to_device_transfers == 0
            else "host_per_step"
        ),
        "gradient_norm_min": (
            float(np.min(gradient_norms_host)) if training else float("nan")
        ),
        "gradient_norm_max": (
            float(np.max(gradient_norms_host)) if training else float("nan")
        ),
    }


def _streamfunction_dirichlet_rayleigh_peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    full_diagnostics: bool,
) -> dict[str, Any]:








    del full_diagnostics
    if str(getattr(cfg, "geometry_mode", "")) != "cinf_ellipse":
        raise ValueError(
            "Dirichlet streamfunction Jet training requires a smooth "
            "hard-boundary geometry hook"
        )
    if str(getattr(cfg, "envelope_mode", "")).startswith("cinf_") is False:
        raise ValueError(
            "Dirichlet streamfunction training requires a smooth envelope"
        )
    if int(getattr(cfg, "spatial_derivative_order", 0)) < 1:
        raise ValueError(
            "Dirichlet streamfunction Rayleigh requires spatial gradients"
        )
    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    if causal_sorting not in {"fixed_order", "geometry_rayleigh"}:
        raise ValueError(f"unsupported causal_sorting {causal_sorting!r}")

    n = len(points)
    if n == 0:
        raise ValueError("Dirichlet streamfunction Rayleigh received no points")
    modes = int(cfg.modes)
    if modes != len(model.modes):
        raise ValueError("configuration/model mode count mismatch")
    area = np.float32(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = cp.asarray(area / np.float32(n), dtype=F32)
    mass_floor = float(cfg.mass_floor)
    passes = max(1, int(getattr(cfg, "reorthogonalization_passes", 2)))




    points_device = _device_points(points, F32)
    input_host_to_device_transfers = int(not isinstance(points, cp.ndarray))



    raw_value = cp.empty((n, modes), dtype=F32)
    raw_gradient = cp.empty((2 * n, modes), dtype=F32)
    for mode, field, _cache in model.forward_stream_jet1(
        cfg, points_device, beta, alpha, False
    ):
        network = int(mode.mode_index)
        raw_value[:, network] = field.value[:, 0].astype(F32, copy=False)
        raw_gradient[:, network] = field.grad[:, 0].reshape(-1).astype(
            F32, copy=False
        )
    raw_mass = weight * cp.sum(raw_value * raw_value, axis=0, dtype=F32)
    raw_stiffness = weight * cp.sum(
        raw_gradient * raw_gradient, axis=0, dtype=F32
    )
    sorting_rayleigh = raw_stiffness / raw_mass
    causal_order_device = (
        cp.argsort(sorting_rayleigh, kind="stable")
        if causal_sorting == "geometry_rayleigh"
        else cp.arange(modes, dtype=cp.int64)
    )
    sorted_value = raw_value[:, causal_order_device]
    sorted_gradient = raw_gradient[:, causal_order_device]



    basis_value = cp.empty((n, modes), dtype=F32)
    basis_gradient = cp.empty((2 * n, modes), dtype=F32)
    mass_values = cp.empty(modes, dtype=F32)
    stiffness_values = cp.empty(modes, dtype=F32)
    rayleigh_values = cp.empty(modes, dtype=F32)
    projection_values = cp.empty(modes, dtype=F32)
    orth_values = cp.empty(modes, dtype=F32)
    adjoint_raw_value = cp.zeros_like(raw_value) if training else None
    adjoint_raw_gradient = cp.zeros_like(raw_gradient) if training else None
    freeze_prefix = int(getattr(cfg, "freeze_prefix_modes", 0))
    for causal_rank in range(modes):
        residual_value = sorted_value[:, causal_rank].copy()
        residual_gradient = sorted_gradient[:, causal_rank].copy()
        coefficients = cp.zeros((causal_rank,), dtype=F32)
        if causal_rank:
            previous_value = basis_value[:, :causal_rank]
            previous_gradient = basis_gradient[:, :causal_rank]
            for _ in range(passes):
                increment = weight * (previous_value.T @ residual_value)
                coefficients += increment
                residual_value -= previous_value @ increment
                residual_gradient -= previous_gradient @ increment
        else:
            previous_value = previous_gradient = None

        mass = weight * cp.dot(residual_value, residual_value)
        stiffness = weight * cp.dot(residual_gradient, residual_gradient)
        rayleigh = stiffness / mass
        mass_values[causal_rank] = mass
        stiffness_values[causal_rank] = stiffness
        rayleigh_values[causal_rank] = rayleigh

        requires_grad = training and causal_rank >= freeze_prefix
        if requires_grad:
            adjoint_value = (
                -2.0 * stiffness * weight / (mass * mass)
            ) * residual_value
            adjoint_gradient = (
                2.0 * weight / mass
            ) * residual_gradient
            if causal_rank:
                for _ in range(passes):
                    coefficient_adjoint = -(
                        previous_value.T @ adjoint_value
                        + previous_gradient.T @ adjoint_gradient
                    )
                    adjoint_value += weight * (
                        previous_value @ coefficient_adjoint
                    )
            if adjoint_raw_value is None or adjoint_raw_gradient is None:
                raise RuntimeError("Jet1 adjoint buffers were not allocated")
            network = causal_order_device[causal_rank]
            adjoint_raw_value[:, network] = adjoint_value
            adjoint_raw_gradient[:, network] = adjoint_gradient

        normalizer = cp.reciprocal(cp.sqrt(mass))
        normalized_value = residual_value * normalizer
        normalized_gradient = residual_gradient * normalizer
        basis_value[:, causal_rank] = normalized_value
        basis_gradient[:, causal_rank] = normalized_gradient
        normalized_mass = weight * cp.dot(normalized_value, normalized_value)
        orth_value = cp.abs(normalized_mass - F32(1.0))
        if causal_rank:
            overlap = weight * (previous_value.T @ normalized_value)
            orth_value = cp.maximum(orth_value, cp.max(cp.abs(overlap)))
        orth_values[causal_rank] = orth_value
        projection_values[causal_rank] = (
            cp.dot(coefficients, coefficients) / mass
            if causal_rank
            else F32(0.0)
        )

    peeled_rayleigh_by_network_device = cp.empty(modes, dtype=F32)
    peeled_rayleigh_by_network_device[causal_order_device] = rayleigh_values
    physical_valid_device = cp.all(
        cp.isfinite(raw_mass)
        & cp.isfinite(raw_stiffness)
        & cp.isfinite(sorting_rayleigh)
        & (raw_mass > F32(mass_floor))
    ) & cp.all(
        cp.isfinite(mass_values)
        & cp.isfinite(stiffness_values)
        & cp.isfinite(rayleigh_values)
        & cp.isfinite(projection_values)
        & cp.isfinite(orth_values)
        & (mass_values > F32(mass_floor))
    )




    gradient_norms_device = cp.zeros(modes, dtype=F32)
    optimizer_committed_device = physical_valid_device
    active_modes_device = cp.zeros(modes, dtype=cp.bool_)
    active_modes_device[causal_order_device[freeze_prefix:]] = True
    if training:
        if adjoint_raw_value is None or adjoint_raw_gradient is None:
            raise RuntimeError("Jet1 training adjoints are unavailable")
        backward_batch = max(1, int(getattr(cfg, "mode_backward_batch", 1)))
        all_weight_grads = [
            cp.zeros_like(stack) for stack in model.master_weight_stacks
        ]
        all_bias_grads = [
            cp.zeros_like(stack) for stack in model.master_bias_stacks
        ]
        backward_modes: list[HardcodedMode] = []
        backward_caches: list[Any] = []
        backward_adjoints: list[Jet1] = []

        def flush_backward_block() -> None:
            if backward_modes:
                block_weight_grads, block_bias_grads = (
                    model.backward_gradients_block_jet1(
                    cfg, backward_modes, backward_caches, backward_adjoints
                    )
                )
                start = backward_modes[0].mode_index
                stop = start + len(backward_modes)
                for target, source in zip(
                    all_weight_grads, block_weight_grads
                ):
                    target[start:stop] = source
                for target, source in zip(all_bias_grads, block_bias_grads):
                    target[start:stop] = source
                backward_modes.clear()
                backward_caches.clear()
                backward_adjoints.clear()

        for mode, field, cache in model.forward_stream_jet1(
            cfg, points_device, beta, alpha, True
        ):
            network = int(mode.mode_index)
            adjoint = Jet1(
                cp.zeros_like(field.value, dtype=F32),
                cp.zeros_like(field.grad, dtype=F32),
            )
            adjoint.value[:, 0] = adjoint_raw_value[:, network]
            adjoint.grad[:, 0] = adjoint_raw_gradient[:, network].reshape(n, 2)
            backward_modes.append(mode)
            backward_caches.append(cache)
            backward_adjoints.append(adjoint)
            if len(backward_modes) == backward_batch:
                flush_backward_block()
        flush_backward_block()
        gradient_norms_device, optimizer_committed_device = (
            model.adam_step_jet1_device(
                cfg,
                all_weight_grads,
                all_bias_grads,
                active_modes_device,
                physical_valid_device,
            )
        )



    diagnostic_rows = cp.asnumpy(
        cp.stack(
            (
                raw_mass,
                raw_stiffness,
                sorting_rayleigh,
                mass_values,
                stiffness_values,
                rayleigh_values,
                projection_values,
                orth_values,
                peeled_rayleigh_by_network_device,
                causal_order_device.astype(F32),
                gradient_norms_device,
                cp.broadcast_to(
                    optimizer_committed_device.astype(F32), (modes,)
                ),
            )
        )
    )
    (
        raw_mass_host,
        raw_stiffness_host,
        sorting_rayleigh_host,
        mass_host,
        stiffness_host,
        rayleigh_host,
        projection_host,
        orth_host,
        peeled_by_network_host,
        causal_order_float,
        gradient_norms_host,
        optimizer_committed_host,
    ) = diagnostic_rows
    causal_order = causal_order_float.astype(np.int64)
    invalid_raw = ~(
        np.isfinite(raw_mass_host)
        & np.isfinite(raw_stiffness_host)
        & np.isfinite(sorting_rayleigh_host)
    )
    unresolved_raw = raw_mass_host <= mass_floor
    if np.any(invalid_raw | unresolved_raw):
        network = int(np.flatnonzero(invalid_raw | unresolved_raw)[0])
        raise FloatingPointError(
            "causal sorting received an invalid streamfunction candidate "
            f"{network + 1}: M={raw_mass_host[network]:.8e}, "
            f"K={raw_stiffness_host[network]:.8e}"
        )
    invalid_peeled = ~(
        np.isfinite(mass_host)
        & np.isfinite(stiffness_host)
        & np.isfinite(rayleigh_host)
        & np.isfinite(projection_host)
        & np.isfinite(orth_host)
    )
    unresolved_peeled = mass_host <= mass_floor
    if np.any(invalid_peeled | unresolved_peeled):
        causal_rank = int(np.flatnonzero(invalid_peeled | unresolved_peeled)[0])
        network = int(causal_order[causal_rank])
        raise FloatingPointError(
            f"streamfunction candidate {network + 1} "
            f"(causal rank {causal_rank + 1}) is invalid: "
            f"M={mass_host[causal_rank]:.8e}, "
            f"K={stiffness_host[causal_rank]:.8e}"
        )
    if training and not bool(optimizer_committed_host[0]):
        active_host = np.zeros(modes, dtype=bool)
        active_host[causal_order[freeze_prefix:]] = True
        invalid_gradient = active_host & (
            ~np.isfinite(gradient_norms_host) | (gradient_norms_host <= 0.0)
        )
        network = (
            int(np.flatnonzero(invalid_gradient)[0])
            if np.any(invalid_gradient)
            else -1
        )
        raise FloatingPointError(
            "zero or non-finite total hardcoded gradient"
            + (f" for network mode {network + 1}" if network >= 0 else "")
        )
    if training:
        active_networks = set(causal_order[freeze_prefix:].astype(int).tolist())
        for network, mode in enumerate(model.modes):
            if network in active_networks:
                mode.steps = [step + 1 for step in mode.steps]

    rayleighs = rayleigh_host.astype(np.float32, copy=False).tolist()
    sorting_rayleigh_by_network = sorting_rayleigh_host.astype(
        np.float32, copy=False
    ).tolist()
    peeled_rayleigh_by_network = peeled_by_network_host.astype(
        np.float32, copy=False
    ).tolist()
    mean_rayleigh = float(np.mean(rayleigh_host, dtype=np.float32))
    return {
        "loss": mean_rayleigh,
        "base_loss": mean_rayleigh,
        "streamfunction_rayleigh": mean_rayleigh,
        "mean_energy": mean_rayleigh,
        "max_energy": float(np.max(rayleigh_host)),
        "min_energy": float(np.min(rayleigh_host)),
        "mean_projection_ratio": float(np.mean(projection_host, dtype=np.float32)),
        "orth_error": float(np.max(orth_host)),
        "peeled_rank": modes,
        "requested_modes": modes,
        "mode_energies": rayleighs,
        "mode_energies_by_network": peeled_rayleigh_by_network,
        "causal_sorting": causal_sorting,
        "causal_order": causal_order.astype(int).tolist(),
        "causal_order_1based": (causal_order + 1).astype(int).tolist(),
        "sorting_rayleigh_by_network": sorting_rayleigh_by_network,
        "spatial_jet_order_used": 1,
        "causal_sort_device": "gpu",
        "diagnostic_host_transfers_per_step": 1,
        "optimizer_device": "gpu_batched_all_modes",
        "optimizer_host_transfers_per_step": 0,
        "input_host_to_device_transfers_per_step": (
            input_host_to_device_transfers
        ),
        "sample_pack_device": (
            "gpu_resident" if input_host_to_device_transfers == 0
            else "host_per_step"
        ),
        "gradient_norm_min": (
            float(np.min(gradient_norms_host)) if training else float("nan")
        ),
        "gradient_norm_max": (
            float(np.max(gradient_norms_host)) if training else float("nan")
        ),
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

    objective = str(getattr(cfg, "basis_objective", ""))
    if reference_model is not None:
        raise ValueError("Rayleigh training uses no reference model")
    if objective == "pure_vorticity_rayleigh":
        return _pure_vorticity_rayleigh_peel_step(
            cfg, model, points, beta, alpha, training, full_diagnostics
        )
    if objective == "streamfunction_dirichlet_rayleigh":
        return _streamfunction_dirichlet_rayleigh_peel_step(
            cfg, model, points, beta, alpha, training, full_diagnostics
        )
    if objective == "velocity_rayleigh":
        return _velocity_rayleigh_peel_step(
            cfg, model, points, beta, alpha, training, full_diagnostics
        )
    raise ValueError(f"unsupported basis_objective={objective!r}")
