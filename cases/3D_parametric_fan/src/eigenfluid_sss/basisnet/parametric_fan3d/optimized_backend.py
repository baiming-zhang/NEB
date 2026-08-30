"""Optimized handwritten backend for the 3D parametric fan benchmark.

The network, second-order spatial jets, reverse pass, Adam update, tensor-
potential operator, and SSS peeling are all explicit CuPy/NumPy code.  No
automatic differentiation or neural-network framework is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np

from .hardcoded_backend import (
    F16,
    F32,
    F64,
    _compute_dtype,
    approximate_fluid_volume,
    continuous_parameters,
    fluid_points,
    initialize_arrays,
    parameter_values,
)


JET_SIZE = 13
GRADIENT_SLICE = slice(1, 4)
HESSIAN_SLICE = slice(4, 13)


@dataclass
class Jet2:
    value: cp.ndarray
    gradient: cp.ndarray
    hessian: cp.ndarray

    def __add__(self, other: Any) -> "Jet2":
        right = _as_jet(other, self)
        return Jet2(
            self.value + right.value,
            self.gradient + right.gradient,
            self.hessian + right.hessian,
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, -self.gradient, -self.hessian)

    def __sub__(self, other: Any) -> "Jet2":
        return self + (-_as_jet(other, self))

    def __rsub__(self, other: Any) -> "Jet2":
        return _as_jet(other, self) - self

    def __mul__(self, other: Any) -> "Jet2":
        right = _as_jet(other, self)
        gradient = (
            self.gradient * right.value[:, None]
            + right.gradient * self.value[:, None]
        )
        hessian = (
            self.hessian * right.value[:, None, None]
            + right.hessian * self.value[:, None, None]
            + self.gradient[:, :, None] * right.gradient[:, None, :]
            + right.gradient[:, :, None] * self.gradient[:, None, :]
        )
        return Jet2(self.value * right.value, gradient, hessian)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Jet2":
        return self * _jreciprocal(_as_jet(other, self))

    def __rtruediv__(self, other: Any) -> "Jet2":
        return _as_jet(other, self) / self

    def __pow__(self, exponent: float) -> "Jet2":
        return _jpower_nonnegative(self, float(exponent))


def _jconstant(value: Any, reference: Jet2) -> Jet2:
    scalar = cp.asarray(value, dtype=F32)
    values = cp.broadcast_to(scalar, reference.value.shape)
    return Jet2(
        values,
        cp.zeros_like(reference.gradient),
        cp.zeros_like(reference.hessian),
    )


def _as_jet(value: Any, reference: Jet2) -> Jet2:
    return value if isinstance(value, Jet2) else _jconstant(value, reference)


def _junary(
    argument: Jet2,
    value: cp.ndarray,
    first: cp.ndarray,
    second: cp.ndarray,
) -> Jet2:
    gradient = first[:, None] * argument.gradient
    hessian = (
        first[:, None, None] * argument.hessian
        + second[:, None, None]
        * argument.gradient[:, :, None]
        * argument.gradient[:, None, :]
    )
    return Jet2(value, gradient, hessian)


def _jsin(argument: Jet2) -> Jet2:
    return _junary(
        argument,
        cp.sin(argument.value),
        cp.cos(argument.value),
        -cp.sin(argument.value),
    )


def _jcos(argument: Jet2) -> Jet2:
    return _junary(
        argument,
        cp.cos(argument.value),
        -cp.sin(argument.value),
        -cp.cos(argument.value),
    )


def _jsqrt(argument: Jet2) -> Jet2:
    positive = cp.maximum(argument.value, F32(1.0e-20))
    root = cp.sqrt(positive)
    first = 0.5 / root
    second = -0.25 / (positive * root)
    return _junary(argument, root, first, second)


def _jabs(argument: Jet2) -> Jet2:
    return _junary(
        argument,
        cp.abs(argument.value),
        cp.sign(argument.value),
        cp.zeros_like(argument.value),
    )


def _jreciprocal(argument: Jet2) -> Jet2:
    inverse = 1.0 / argument.value
    return _junary(
        argument,
        inverse,
        -(inverse * inverse),
        2.0 * inverse * inverse * inverse,
    )


def _jpower_nonnegative(argument: Jet2, exponent: float) -> Jet2:
    active = argument.value > 0.0
    safe = cp.maximum(argument.value, F32(1.0e-12))
    value = cp.where(active, safe**exponent, 0.0).astype(F32)
    first = cp.where(active, exponent * safe ** (exponent - 1.0), 0.0)
    second = cp.where(
        active,
        exponent * (exponent - 1.0) * safe ** (exponent - 2.0),
        0.0,
    )
    return _junary(argument, value, first.astype(F32), second.astype(F32))


def _jwhere(condition: cp.ndarray, left: Jet2, right: Jet2) -> Jet2:
    return Jet2(
        cp.where(condition, left.value, right.value),
        cp.where(condition[:, None], left.gradient, right.gradient),
        cp.where(condition[:, None, None], left.hessian, right.hessian),
    )


def _jmaximum(left: Jet2, right: Any) -> Jet2:
    other = _as_jet(right, left)
    return _jwhere(left.value >= other.value, left, other)


def _jminimum(left: Jet2, right: Any) -> Jet2:
    other = _as_jet(right, left)
    return _jwhere(left.value <= other.value, left, other)


def _jclip(argument: Jet2, lower: float, upper: float) -> Jet2:
    low = _jconstant(lower, argument)
    high = _jconstant(upper, argument)
    return _jwhere(
        argument.value <= lower,
        low,
        _jwhere(argument.value >= upper, high, argument),
    )


def _jatan2(numerator: Jet2, denominator: Jet2) -> Jet2:
    z, y = numerator, denominator
    # A float32 radius^4 built from 1e-20 underflows to zero at the axis and
    # turns the nominal 0/0 second derivative into NaN.  The fan blade branch
    # only uses azimuth outside the finite hub, so this numerical floor has no
    # effect on physical blade geometry.
    radius2 = cp.maximum(y.value * y.value + z.value * z.value, 1.0e-12)
    radius4 = radius2 * radius2
    fy = -z.value / radius2
    fz = y.value / radius2
    fyy = 2.0 * y.value * z.value / radius4
    fzz = -fyy
    fyz = (z.value * z.value - y.value * y.value) / radius4
    gradient = fy[:, None] * y.gradient + fz[:, None] * z.gradient
    hessian = (
        fy[:, None, None] * y.hessian
        + fz[:, None, None] * z.hessian
        + fyy[:, None, None]
        * y.gradient[:, :, None]
        * y.gradient[:, None, :]
        + fzz[:, None, None]
        * z.gradient[:, :, None]
        * z.gradient[:, None, :]
        + fyz[:, None, None]
        * (
            y.gradient[:, :, None] * z.gradient[:, None, :]
            + z.gradient[:, :, None] * y.gradient[:, None, :]
        )
    )
    return Jet2(cp.arctan2(z.value, y.value), gradient, hessian)


def _jremainder(argument: Jet2, period: float, shift: float) -> Jet2:
    value = cp.remainder(argument.value + shift, period) - shift
    return Jet2(value, argument.gradient, argument.hessian)


def _coordinate_jet(points: cp.ndarray, axis: int) -> Jet2:
    gradient = cp.zeros((len(points), 3), dtype=F32)
    gradient[:, axis] = 1.0
    return Jet2(
        points[:, axis],
        gradient,
        cp.zeros((len(points), 3, 3), dtype=F32),
    )


def _camber_jet(cfg: SimpleNamespace, xi: Jet2) -> Jet2:
    m, p = cfg.camber, cfg.camber_position
    fore = m / (p * p) * (2.0 * p * xi - xi * xi)
    aft = m / ((1.0 - p) ** 2) * (
        (1.0 - 2.0 * p) + 2.0 * p * xi - xi * xi
    )
    return _jwhere(xi.value < p, fore, aft)


def _half_thickness_jet(xi: Jet2, thickness: Any) -> Jet2:
    clipped = _jclip(xi, 0.0, 1.0)
    root = _jsqrt(clipped)
    square = clipped * clipped
    cube = square * clipped
    fourth = square * square
    return (
        0.2969 * root
        - 0.1260 * clipped
        - 0.3516 * square
        + 0.2843 * cube
        - 0.1036 * fourth
    ) * thickness * 5.0


def rotor_distance_jet(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> Jet2:
    """Piecewise analytic second-order jet of the four-blade distance proxy."""
    x = _coordinate_jet(points, 0)
    y = _coordinate_jet(points, 1)
    z = _coordinate_jet(points, 2)
    radius = _jsqrt(y * y + z * z + 1.0e-12)
    raw_azimuth = _jatan2(z, y)
    azimuth = _jwhere(
        radius.value >= 0.5 * cfg.hub_radius,
        raw_azimuth,
        _jconstant(0.0, raw_azimuth),
    )
    sector = 2.0 * math.pi / cfg.blade_count
    phase = _jremainder(azimuth, sector, 0.5 * sector)
    tangential = radius * phase
    span = _jclip(
        (radius - cfg.hub_radius) / (cfg.tip_radius - cfg.hub_radius),
        0.0,
        1.0,
    )
    smooth_span = span * span * (3.0 - 2.0 * span)
    beta_tip, chord_tip, thickness_ratio = parameter_values(cfg, parameters)
    beta = (
        cfg.beta_root_deg
        + (span**0.82) * (beta_tip - cfg.beta_root_deg)
    ) * (math.pi / 180.0)
    chord = cfg.chord_root + smooth_span * (chord_tip - cfg.chord_root)
    chord = chord * (1.0 - 0.20 * (span**10.0))
    sweep = 0.18 * span * span - 0.035 * span
    rake = -0.08 * span * span
    xlocal = x - rake
    tlocal = tangential - sweep
    chord_coordinate = xlocal * _jsin(beta) + tlocal * _jcos(beta)
    normal_coordinate = xlocal * _jcos(beta) - tlocal * _jsin(beta)
    xi = chord_coordinate / chord + cfg.stacking_fraction
    camber = chord * _camber_jet(cfg, _jclip(xi, 0.0, 1.0))
    thickness = chord * _half_thickness_jet(xi, thickness_ratio)
    blade_distance = _jmaximum(
        _jmaximum(
            _jabs(normal_coordinate - camber) - thickness,
            -xi * chord,
        ),
        _jmaximum(
            (xi - 1.0) * chord,
            _jmaximum(
                cfg.hub_radius - radius,
                radius - cfg.tip_radius,
            ),
        ),
    )
    hub_distance = _jmaximum(
        _jabs(x) - 0.5 * cfg.hub_length,
        radius - cfg.hub_radius,
    )
    return _jminimum(blade_distance, hub_distance)


def boundary_envelope_jet(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> Jet2:
    x = _coordinate_jet(points, 0)
    y = _coordinate_jet(points, 1)
    z = _coordinate_jet(points, 2)
    radius = _jsqrt(y * y + z * z + 1.0e-12)
    inlet = _jmaximum((x - cfg.xmin) / cfg.tip_radius, 0.0)
    cylinder = _jmaximum(
        (cfg.farfield_radius - radius) / cfg.tip_radius,
        0.0,
    )
    obstacle = _jmaximum(
        rotor_distance_jet(cfg, points, parameters) / cfg.tip_radius,
        0.0,
    )

    def saturate(value: Jet2) -> Jet2:
        return value / (1.0 + value)

    product = saturate(inlet) * saturate(cylinder) * saturate(obstacle)
    return product * product


def encoded_input_jet(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> cp.ndarray:
    low = cp.asarray(
        (cfg.xmin, -cfg.farfield_radius, -cfg.farfield_radius), dtype=F32
    )
    high = cp.asarray(
        (cfg.xmax, cfg.farfield_radius, cfg.farfield_radius), dtype=F32
    )
    scale = 1.0 / (high - low)
    normal = (points - low) * scale
    frequencies = 2.0 ** cp.arange(cfg.frequencies, dtype=F32)
    phase = 2.0 * math.pi * normal[:, :, None] * frequencies[None, None, :]
    sine = cp.sin(phase).reshape(len(points), -1)
    cosine = cp.cos(phase).reshape(len(points), -1)
    parameter_rows = cp.broadcast_to(parameters[None, :], (len(points), 3))
    value = cp.concatenate(
        (sine, cosine, 2.0 * normal - 1.0, parameter_rows), axis=1
    ).astype(F32)
    gradient = cp.zeros((len(points), value.shape[1], 3), dtype=F32)
    hessian = cp.zeros((len(points), value.shape[1], 3, 3), dtype=F32)
    feature = 0
    for family in ("sin", "cos"):
        for axis in range(3):
            for frequency_index in range(cfg.frequencies):
                wave = (
                    2.0
                    * math.pi
                    * frequencies[frequency_index]
                    * scale[axis]
                )
                angle = phase[:, axis, frequency_index]
                if family == "sin":
                    gradient[:, feature, axis] = cp.cos(angle) * wave
                    hessian[:, feature, axis, axis] = -cp.sin(angle) * wave * wave
                else:
                    gradient[:, feature, axis] = -cp.sin(angle) * wave
                    hessian[:, feature, axis, axis] = -cp.cos(angle) * wave * wave
                feature += 1
    for axis in range(3):
        gradient[:, feature + axis, axis] = 2.0 * scale[axis]
    return cp.concatenate(
        (
            value[..., None],
            gradient,
            hessian.reshape(len(points), value.shape[1], 9),
        ),
        axis=2,
    )


def _stack_scalar_jets(features: list[Jet2]) -> cp.ndarray:
    """Pack scalar second-order jets as a network feature tensor."""
    return _pack_jet(
        cp.stack([feature.value for feature in features], axis=1),
        cp.stack([feature.gradient for feature in features], axis=1),
        cp.stack([feature.hessian for feature in features], axis=1),
    )


def c4_encoded_input_jet(
    cfg: SimpleNamespace,
    points: cp.ndarray,
    parameters: cp.ndarray,
) -> cp.ndarray:
    """C4-invariant analytic features for the cylindrical fan domain.

    The three scalar network outputs are interpreted later as axial, radial,
    and azimuthal coefficients of a physical vector potential.  Dependence on
    azimuth is represented by smooth fourth-order Cartesian invariants, so the
    resulting field is exactly C4 equivariant without an axis singularity.
    """
    x = _coordinate_jet(points, 0)
    y = _coordinate_jet(points, 1)
    z = _coordinate_jet(points, 2)
    radius_squared = y * y + z * z
    axial_normal = (x - cfg.xmin) / (cfg.xmax - cfg.xmin)
    radial_squared_normal = radius_squared / (cfg.farfield_radius**2)
    y2 = y * y
    z2 = z * z
    c4_cos = (y2 * y2 - 6.0 * y2 * z2 + z2 * z2) / (
        cfg.farfield_radius**4
    )
    c4_sin = 4.0 * y * z * (y2 - z2) / (cfg.farfield_radius**4)
    c4_cos_normal = 0.5 * (c4_cos + 1.0)
    c4_sin_normal = 0.5 * (c4_sin + 1.0)
    features: list[Jet2] = []
    for family in ("sin", "cos"):
        for coordinate in ("axial", "c4_cos", "c4_sin"):
            for frequency_index in range(cfg.frequencies):
                if coordinate == "axial":
                    phase = (
                        2.0
                        * math.pi
                        * (2.0**frequency_index)
                        * axial_normal
                    )
                elif coordinate == "c4_cos":
                    phase = (
                        2.0
                        * math.pi
                        * (2.0**frequency_index)
                        * c4_cos_normal
                    )
                else:
                    phase = (
                        2.0
                        * math.pi
                        * (2.0**frequency_index)
                        * c4_sin_normal
                    )
                features.append(_jsin(phase) if family == "sin" else _jcos(phase))
    features.extend(
        (
            2.0 * axial_normal - 1.0,
            2.0 * radial_squared_normal - 1.0,
            2.0 * radial_squared_normal * radial_squared_normal - 1.0,
        )
    )
    features.extend(_jconstant(parameters[index], x) for index in range(3))
    return _stack_scalar_jets(features)


def _fixed_jet_product(
    scalar_jet: cp.ndarray,
    coefficient: Jet2,
) -> cp.ndarray:
    """Multiply batched scalar jets by a fixed spatial scalar jet."""
    value, gradient, hessian = _unpack_jet(scalar_jet)
    output_value = value * coefficient.value[None, :]
    output_gradient = (
        gradient * coefficient.value[None, :, None]
        + value[..., None] * coefficient.gradient[None, :, :]
    )
    output_hessian = (
        hessian * coefficient.value[None, :, None, None]
        + value[..., None, None] * coefficient.hessian[None, :, :, :]
        + coefficient.gradient[None, :, :, None] * gradient[..., None, :]
        + gradient[..., :, None] * coefficient.gradient[None, :, None, :]
    )
    return _pack_jet(output_value, output_gradient, output_hessian)


def _fixed_jet_product_reverse(
    output_adjoint: cp.ndarray,
    coefficient: Jet2,
) -> cp.ndarray:
    """Adjoint of :func:`_fixed_jet_product` with respect to the scalar jet."""
    value_adjoint, gradient_adjoint, hessian_adjoint = _unpack_jet(
        output_adjoint
    )
    scalar_value_adjoint = (
        value_adjoint * coefficient.value[None, :]
        + cp.sum(
            gradient_adjoint * coefficient.gradient[None, :, :], axis=-1
        )
        + cp.sum(
            hessian_adjoint * coefficient.hessian[None, :, :, :],
            axis=(-2, -1),
        )
    )
    scalar_gradient_adjoint = (
        gradient_adjoint * coefficient.value[None, :, None]
        + cp.einsum(
            "bnij,ni->bnj", hessian_adjoint, coefficient.gradient
        )
        + cp.einsum(
            "bnij,nj->bni", hessian_adjoint, coefficient.gradient
        )
    )
    scalar_hessian_adjoint = (
        hessian_adjoint * coefficient.value[None, :, None, None]
    )
    return _pack_jet(
        scalar_value_adjoint,
        scalar_gradient_adjoint,
        scalar_hessian_adjoint,
    )


def _c4_potential_coefficients(points: cp.ndarray) -> tuple[Jet2, Jet2]:
    """Smooth transverse equivariant generators; finite on the fan axis."""
    y = _coordinate_jet(points, 1)
    z = _coordinate_jet(points, 2)
    return y, z


def c4_potential_jet(
    scalar_jet: cp.ndarray,
    points: cp.ndarray,
) -> cp.ndarray:
    """Map invariant axial/radial/azimuthal coefficients to A components.

    The backend's antisymmetric tensor potential corresponds to the physical
    vector potential ``B=(A2,-A1,A0)``.  Smooth Cartesian radial/tangential
    generators make C4 equivariance exact while retaining arbitrary fourfold-
    periodic structure and avoiding the singular unit frame at r=0.
    """
    radial_y, radial_z = _c4_potential_coefficients(points)
    axial = scalar_jet[:, :, 0, :]
    radial = scalar_jet[:, :, 1, :]
    azimuthal = scalar_jet[:, :, 2, :]
    a0 = _fixed_jet_product(radial, radial_z) + _fixed_jet_product(
        azimuthal, radial_y
    )
    a1 = -_fixed_jet_product(radial, radial_y) + _fixed_jet_product(
        azimuthal, radial_z
    )
    return cp.stack((a0, a1, axial), axis=2)


def c4_potential_jet_reverse(
    potential_adjoint: cp.ndarray,
    points: cp.ndarray,
) -> cp.ndarray:
    """Adjoint of :func:`c4_potential_jet` for manual backpropagation."""
    radial_y, radial_z = _c4_potential_coefficients(points)
    a0_adjoint = potential_adjoint[:, :, 0, :]
    a1_adjoint = potential_adjoint[:, :, 1, :]
    axial_adjoint = potential_adjoint[:, :, 2, :]
    radial_adjoint = _fixed_jet_product_reverse(
        a0_adjoint, radial_z
    ) - _fixed_jet_product_reverse(a1_adjoint, radial_y)
    azimuthal_adjoint = _fixed_jet_product_reverse(
        a0_adjoint, radial_y
    ) + _fixed_jet_product_reverse(a1_adjoint, radial_z)
    return cp.stack(
        (axial_adjoint, radial_adjoint, azimuthal_adjoint), axis=2
    )


def _quintic_smoothstep(argument: Jet2) -> Jet2:
    clipped = _jclip(argument, 0.0, 1.0)
    return clipped**3 * (10.0 - 15.0 * clipped + 6.0 * clipped * clipped)


def moving_wall_lifting_jet(
    cfg: SimpleNamespace,
    points: cp.ndarray,
) -> cp.ndarray:
    """Divergence-free affine lifting for inlet and moving rotor data.

    A compact C2 plateau is one on a cylinder containing every admissible
    blade and the hub, and zero before the inlet/outlet and lateral far field.
    Blending is performed in vector-potential space, so the transition remains
    exactly solenoidal.  On the rotor plateau its curl is exactly
    ``Omega cross r``; outside the support its curl is the uniform inlet flow.
    """
    x = _coordinate_jet(points, 0)
    y = _coordinate_jet(points, 1)
    z = _coordinate_jet(points, 2)
    radius = _jsqrt(y * y + z * z + 1.0e-12)
    axial_fraction = (
        (_jabs(x) - cfg.moving_lift_axial_core)
        / (cfg.moving_lift_axial_outer - cfg.moving_lift_axial_core)
    )
    radial_fraction = (
        (radius - cfg.moving_lift_radial_core)
        / (cfg.moving_lift_radial_outer - cfg.moving_lift_radial_core)
    )
    plateau = (1.0 - _quintic_smoothstep(axial_fraction)) * (
        1.0 - _quintic_smoothstep(radial_fraction)
    )
    ambient = 1.0 - plateau
    inflow = float(cfg.inflow_velocity)
    rotation = float(cfg.rotation_rate)
    # B=(A2,-A1,A0).  curl(0,-Uy/2? see A below) gives U e_x;
    # curl(-Omega*r^2/2,0,0) gives Omega cross r.
    a0 = 0.5 * inflow * ambient * y
    a1 = 0.5 * inflow * ambient * z
    a2 = -0.5 * rotation * plateau * (y * y + z * z)
    return _stack_scalar_jets([a0, a1, a2])


def moving_wall_lifting_fields(
    cfg: SimpleNamespace,
    points: np.ndarray | cp.ndarray,
) -> dict[str, cp.ndarray]:
    """Return lifting potential, velocity, and velocity gradient on the GPU."""
    point_device = cp.asarray(points, dtype=F32)
    potential_jet = moving_wall_lifting_jet(cfg, point_device)[None, ...]
    potential, velocity, gradient = _fields_from_potential_jet(potential_jet)
    return {
        "potential": potential[0],
        "velocity": velocity[0],
        "gradient": gradient[0],
    }


def _pack_jet(value: cp.ndarray, gradient: cp.ndarray, hessian: cp.ndarray) -> cp.ndarray:
    return cp.concatenate(
        (value[..., None], gradient, hessian.reshape(*value.shape, 9)), axis=-1
    )


def _unpack_jet(jet: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    return (
        jet[..., 0],
        jet[..., GRADIENT_SLICE],
        jet[..., HESSIAN_SLICE].reshape(*jet.shape[:-1], 3, 3),
    )


def _jet_affine(
    inputs: cp.ndarray,
    weights: cp.ndarray,
    biases: cp.ndarray,
    compute_dtype: type[cp.generic],
) -> cp.ndarray:
    left = inputs.transpose(0, 3, 1, 2)
    right = weights.transpose(0, 2, 1)[:, None, :, :]
    if compute_dtype is F16:
        output = cp.matmul(
            left.astype(F16, copy=False),
            right.astype(F16, copy=False),
            dtype=F32,
        )
    else:
        output = cp.matmul(left, right)
    output = output.transpose(0, 2, 3, 1)
    output[..., 0] += biases[:, None, :]
    return output


def _jet_sine(inputs: cp.ndarray) -> cp.ndarray:
    value, gradient, hessian = _unpack_jet(inputs)
    sine = cp.sin(value)
    cosine = cp.cos(value)
    output_gradient = cosine[..., None] * gradient
    output_hessian = (
        cosine[..., None, None] * hessian
        - sine[..., None, None]
        * gradient[..., :, None]
        * gradient[..., None, :]
    )
    return _pack_jet(sine, output_gradient, output_hessian)


def _jet_sine_reverse(inputs: cp.ndarray, adjoint: cp.ndarray) -> cp.ndarray:
    value, gradient, hessian = _unpack_jet(inputs)
    adjoint_value, adjoint_gradient, adjoint_hessian = _unpack_jet(adjoint)
    sine = cp.sin(value)
    cosine = cp.cos(value)
    outer = gradient[..., :, None] * gradient[..., None, :]
    input_value = (
        adjoint_value * cosine
        - sine * cp.sum(adjoint_gradient * gradient, axis=-1)
        + cp.sum(
            adjoint_hessian
            * (-sine[..., None, None] * hessian - cosine[..., None, None] * outer),
            axis=(-2, -1),
        )
    )
    symmetric_adjoint = adjoint_hessian + adjoint_hessian.swapaxes(-1, -2)
    input_gradient = (
        adjoint_gradient * cosine[..., None]
        - sine[..., None]
        * cp.einsum("...ij,...j->...i", symmetric_adjoint, gradient)
    )
    input_hessian = adjoint_hessian * cosine[..., None, None]
    return _pack_jet(input_value, input_gradient, input_hessian)


@dataclass
class JetForwardCache:
    inputs: list[cp.ndarray]
    preactivations: list[cp.ndarray]
    output_input: cp.ndarray


class OptimizedFanModel:
    """Independent SIREN modes with blocked second-order reverse and Adam."""

    def __init__(self, cfg: SimpleNamespace, arrays: dict[str, np.ndarray]) -> None:
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
        parameter_count = 2 * len(self.weights)
        self.exp_avg = [
            cp.asarray(arrays[f"adam.p{index:02d}.exp_avg"], dtype=F32)
            for index in range(parameter_count)
        ]
        self.exp_avg_sq = [
            cp.asarray(arrays[f"adam.p{index:02d}.exp_avg_sq"], dtype=F32)
            for index in range(parameter_count)
        ]
        self.steps = cp.asarray(arrays["adam.steps"], dtype=cp.int64)
        shadow_dtype = F16 if self.compute_dtype is F16 else F32
        self.shadow_weights = [value.astype(shadow_dtype) for value in self.weights]
        self.shadow_biases = [value.astype(shadow_dtype) for value in self.biases]

    def forward_jet_block(
        self,
        input_jet: cp.ndarray,
        start: int,
        stop: int,
        *,
        cache: bool,
    ) -> tuple[cp.ndarray, JetForwardCache | None]:
        current = cp.broadcast_to(
            input_jet[None, :, :, :], (stop - start, *input_jet.shape)
        )
        layer_inputs: list[cp.ndarray] = []
        preactivations: list[cp.ndarray] = []
        for layer in range(len(self.weights) - 1):
            if cache:
                layer_inputs.append(current)
            preactivation = _jet_affine(
                current,
                self.shadow_weights[layer][start:stop],
                self.shadow_biases[layer][start:stop],
                self.compute_dtype,
            )
            if cache:
                preactivations.append(preactivation)
            current = _jet_sine(preactivation)
        output_input = current
        output = _jet_affine(
            current,
            self.shadow_weights[-1][start:stop],
            self.shadow_biases[-1][start:stop],
            self.compute_dtype,
        )
        if not cache:
            return output, None
        return output, JetForwardCache(layer_inputs, preactivations, output_input)

    def _affine_backward(
        self,
        inputs: cp.ndarray,
        weights: cp.ndarray,
        adjoint: cp.ndarray,
        *,
        need_input: bool,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
        block, point_count, _, jet_size = inputs.shape
        output_width = adjoint.shape[2]
        input_width = inputs.shape[2]
        left = adjoint.transpose(0, 2, 1, 3).reshape(
            block, output_width, point_count * jet_size
        )
        right = inputs.transpose(0, 1, 3, 2).reshape(
            block, point_count * jet_size, input_width
        )
        if self.compute_dtype is F16:
            # Keep the forward MLP on tensor-core FP16, but accumulate the
            # analytic reverse in FP32.  SSS adjoints can contain factors of
            # 1 / mass**2; casting those adjoints back to FP16 can overflow
            # before the FP32 GEMM accumulator ever sees them.
            weight_gradient = cp.matmul(
                left.astype(F32, copy=False),
                right.astype(F32, copy=False),
            )
        else:
            weight_gradient = cp.matmul(left, right)
        bias_gradient = adjoint[..., 0].sum(axis=1)
        if not need_input:
            return weight_gradient, bias_gradient, None
        adjoint_rows = adjoint.transpose(0, 3, 1, 2)
        weight_rows = weights[:, None, :, :]
        if self.compute_dtype is F16:
            input_adjoint = cp.matmul(
                adjoint_rows.astype(F32, copy=False),
                weight_rows.astype(F32, copy=False),
            )
        else:
            input_adjoint = cp.matmul(adjoint_rows, weight_rows)
        return (
            weight_gradient,
            bias_gradient,
            input_adjoint.transpose(0, 2, 3, 1),
        )

    def backward_jet_block(
        self,
        start: int,
        stop: int,
        cache: JetForwardCache,
        output_adjoint: cp.ndarray,
    ) -> list[cp.ndarray]:
        output_weight, output_bias, current = self._affine_backward(
            cache.output_input,
            self.weights[-1][start:stop],
            output_adjoint,
            need_input=True,
        )
        if current is None:
            raise RuntimeError("output reverse did not return a hidden adjoint")
        reversed_gradients: list[tuple[cp.ndarray, cp.ndarray]] = []
        for layer in range(len(self.weights) - 2, -1, -1):
            preactivation_adjoint = _jet_sine_reverse(
                cache.preactivations[layer], current
            )
            weight_gradient, bias_gradient, next_adjoint = self._affine_backward(
                cache.inputs[layer],
                self.weights[layer][start:stop],
                preactivation_adjoint,
                need_input=layer > 0,
            )
            reversed_gradients.append((weight_gradient, bias_gradient))
            if layer > 0:
                if next_adjoint is None:
                    raise RuntimeError("hidden reverse lost its input adjoint")
                current = next_adjoint
        reversed_gradients.reverse()
        gradients: list[cp.ndarray] = []
        for weight_gradient, bias_gradient in reversed_gradients:
            gradients.extend((weight_gradient, bias_gradient))
        gradients.extend((output_weight, output_bias))
        return gradients

    def adam_step_block(
        self,
        start: int,
        stop: int,
        gradients: list[cp.ndarray],
        learning_rate: float,
    ) -> cp.ndarray:
        if len(gradients) != 2 * len(self.weights):
            raise RuntimeError("blocked reverse returned wrong gradient count")
        total = cp.zeros(stop - start, dtype=F64)
        for gradient in gradients:
            total += cp.sum(
                gradient.astype(F64) * gradient.astype(F64),
                axis=tuple(range(1, gradient.ndim)),
            )
        norms = cp.sqrt(total)
        valid_by_mode = cp.isfinite(norms) & (norms > 0.0)
        valid = cp.all(valid_by_mode)
        safe_norms = cp.where(valid_by_mode, norms, 1.0)
        scale = cp.minimum(
            1.0,
            self.cfg.grad_clip / cp.maximum(safe_norms, 1.0e-30),
        ).astype(F32)
        beta1, beta2, epsilon = F32(0.9), F32(0.999), F32(1.0e-8)
        parameter_index = 0
        for layer in range(len(self.weights)):
            for parameter, gradient in (
                (self.weights[layer], gradients[2 * layer]),
                (self.biases[layer], gradients[2 * layer + 1]),
            ):
                shape = (stop - start,) + (1,) * (gradient.ndim - 1)
                valid_mask = valid_by_mode.reshape(shape)
                gradient = cp.where(
                    valid_mask, gradient.astype(F32), F32(0.0)
                ) * scale.reshape(shape)
                self.steps[start:stop, parameter_index] += 1
                steps = self.steps[start:stop, parameter_index].astype(F32)
                average = self.exp_avg[parameter_index][start:stop]
                square = self.exp_avg_sq[parameter_index][start:stop]
                average *= beta1
                average += (1.0 - beta1) * gradient
                square *= beta2
                square += (1.0 - beta2) * gradient * gradient
                correction1 = 1.0 - cp.power(beta1, steps)
                correction2 = 1.0 - cp.power(beta2, steps)
                correction_shape = (stop - start,) + (1,) * (gradient.ndim - 1)
                denominator = cp.sqrt(
                    square / correction2.reshape(correction_shape)
                ) + epsilon
                parameter[start:stop] -= (
                    learning_rate
                    / correction1.reshape(correction_shape)
                ) * average / denominator
                parameter_index += 1
        for layer in range(len(self.weights)):
            self.shadow_weights[layer][start:stop] = self.weights[layer][
                start:stop
            ]
            self.shadow_biases[layer][start:stop] = self.biases[layer][
                start:stop
            ]
        return valid

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
            result["adam.steps"] = cp.asnumpy(self.steps)
        return result


def _envelope_product(network_jet: cp.ndarray, envelope: Jet2) -> cp.ndarray:
    value, gradient, hessian = _unpack_jet(network_jet)
    output_value = value * envelope.value[None, :, None]
    output_gradient = (
        gradient * envelope.value[None, :, None, None]
        + value[..., None] * envelope.gradient[None, :, None, :]
    )
    output_hessian = (
        hessian * envelope.value[None, :, None, None, None]
        + value[..., None, None] * envelope.hessian[None, :, None, :, :]
        + envelope.gradient[None, :, None, :, None]
        * gradient[..., None, :]
        + gradient[..., :, None]
        * envelope.gradient[None, :, None, None, :]
    )
    return _pack_jet(output_value, output_gradient, output_hessian)


def _envelope_product_reverse(
    network_jet: cp.ndarray,
    envelope: Jet2,
    output_adjoint: cp.ndarray,
) -> cp.ndarray:
    _, _, _ = _unpack_jet(network_jet)
    adjoint_value, adjoint_gradient, adjoint_hessian = _unpack_jet(
        output_adjoint
    )
    network_value_adjoint = (
        adjoint_value * envelope.value[None, :, None]
        + cp.sum(
            adjoint_gradient * envelope.gradient[None, :, None, :], axis=-1
        )
        + cp.sum(
            adjoint_hessian * envelope.hessian[None, :, None, :, :],
            axis=(-2, -1),
        )
    )
    network_gradient_adjoint = (
        adjoint_gradient * envelope.value[None, :, None, None]
        + cp.einsum(
            "bncdk,nd->bnck", adjoint_hessian, envelope.gradient
        )
        + cp.einsum(
            "bncke,ne->bnck", adjoint_hessian, envelope.gradient
        )
    )
    network_hessian_adjoint = (
        adjoint_hessian * envelope.value[None, :, None, None, None]
    )
    return _pack_jet(
        network_value_adjoint,
        network_gradient_adjoint,
        network_hessian_adjoint,
    )


def _fields_from_potential_jet(
    potential_jet: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    potential, first, second = _unpack_jet(potential_jet)
    velocity = cp.stack(
        (
            first[:, :, 0, 1] + first[:, :, 1, 2],
            -first[:, :, 0, 0] + first[:, :, 2, 2],
            -first[:, :, 1, 0] - first[:, :, 2, 1],
        ),
        axis=2,
    )
    gradient = cp.empty((*velocity.shape, 3), dtype=F32)
    for derivative in range(3):
        gradient[:, :, 0, derivative] = (
            second[:, :, 0, 1, derivative]
            + second[:, :, 1, 2, derivative]
        )
        gradient[:, :, 1, derivative] = (
            -second[:, :, 0, 0, derivative]
            + second[:, :, 2, 2, derivative]
        )
        gradient[:, :, 2, derivative] = (
            -second[:, :, 1, 0, derivative]
            - second[:, :, 2, 1, derivative]
        )
    return potential, velocity, gradient


def _field_adjoint_to_potential_jet(
    velocity_adjoint: cp.ndarray,
    gradient_adjoint: cp.ndarray,
) -> cp.ndarray:
    block, points, _ = velocity_adjoint.shape
    value = cp.zeros((block, points, 3), dtype=F32)
    first = cp.zeros((block, points, 3, 3), dtype=F32)
    second = cp.zeros((block, points, 3, 3, 3), dtype=F32)
    first[:, :, 0, 1] += velocity_adjoint[:, :, 0]
    first[:, :, 1, 2] += velocity_adjoint[:, :, 0]
    first[:, :, 0, 0] -= velocity_adjoint[:, :, 1]
    first[:, :, 2, 2] += velocity_adjoint[:, :, 1]
    first[:, :, 1, 0] -= velocity_adjoint[:, :, 2]
    first[:, :, 2, 1] -= velocity_adjoint[:, :, 2]
    for derivative in range(3):
        second[:, :, 0, 1, derivative] += gradient_adjoint[
            :, :, 0, derivative
        ]
        second[:, :, 1, 2, derivative] += gradient_adjoint[
            :, :, 0, derivative
        ]
        second[:, :, 0, 0, derivative] -= gradient_adjoint[
            :, :, 1, derivative
        ]
        second[:, :, 2, 2, derivative] += gradient_adjoint[
            :, :, 1, derivative
        ]
        second[:, :, 1, 0, derivative] -= gradient_adjoint[
            :, :, 2, derivative
        ]
        second[:, :, 2, 1, derivative] -= gradient_adjoint[
            :, :, 2, derivative
        ]
    return _pack_jet(value, first, second)


def prepared_analytic_inputs(
    cfg: SimpleNamespace,
    points: np.ndarray,
    parameters: np.ndarray,
) -> tuple[cp.ndarray, Jet2]:
    point_device = cp.asarray(points, dtype=F32)
    parameter_device = cp.asarray(parameters, dtype=F32)
    encoder = (
        c4_encoded_input_jet
        if bool(getattr(cfg, "c4_equivariant_network", False))
        else encoded_input_jet
    )
    return (
        encoder(cfg, point_device, parameter_device),
        boundary_envelope_jet(cfg, point_device, parameter_device),
    )


def _network_potential_jet(
    cfg: SimpleNamespace,
    network_jet: cp.ndarray,
    points: cp.ndarray,
) -> cp.ndarray:
    if bool(getattr(cfg, "c4_equivariant_network", False)):
        return c4_potential_jet(network_jet, points)
    return network_jet


def _network_potential_jet_reverse(
    cfg: SimpleNamespace,
    potential_adjoint: cp.ndarray,
    points: cp.ndarray,
) -> cp.ndarray:
    if bool(getattr(cfg, "c4_equivariant_network", False)):
        return c4_potential_jet_reverse(potential_adjoint, points)
    return potential_adjoint


def raw_fields(
    cfg: SimpleNamespace,
    model: OptimizedFanModel,
    points: np.ndarray,
    parameters: np.ndarray,
) -> dict[str, Any]:
    point_count = len(points)
    point_device = cp.asarray(points, dtype=F32)
    input_jet, envelope = prepared_analytic_inputs(cfg, points, parameters)
    raw_p = cp.empty((3 * point_count, cfg.modes), dtype=F64)
    raw_v = cp.empty((3 * point_count, cfg.modes), dtype=F64)
    raw_g = cp.empty((9 * point_count, cfg.modes), dtype=F64)
    block = max(1, int(cfg.mode_forward_batch))
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        network_jet, _ = model.forward_jet_block(
            input_jet, start, stop, cache=False
        )
        equivariant_jet = _network_potential_jet(
            cfg, network_jet, point_device
        )
        potential_jet = _envelope_product(equivariant_jet, envelope)
        potential, velocity, gradient = _fields_from_potential_jet(
            potential_jet
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
        "input_jet": input_jet,
        "envelope_jet": envelope,
        "point_device": point_device,
    }


def peel_and_adjoint(
    cfg: SimpleNamespace,
    raw: dict[str, Any],
    point_count: int,
    *,
    need_adjoint: bool,
    need_coefficients: bool = False,
    diagnostics: bool = True,
) -> dict[str, Any]:
    """Blocked causal SSS with device-resident statistics and one final sync."""
    weight = approximate_fluid_volume(cfg) / point_count
    raw_p = raw["potential"]
    raw_v = raw["velocity"]
    raw_g = raw["gradient"]
    mass = weight * cp.sum(raw_v * raw_v, axis=0)
    stiffness = weight * cp.sum(raw_g * raw_g, axis=0)
    raw_energy = stiffness / cp.maximum(mass, cfg.mass_floor)
    order_device = cp.argsort(raw_energy, kind="stable")
    order = cp.asnumpy(order_device).astype(np.int32)
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
    energies_device = cp.empty(cfg.modes, dtype=F64)
    ratios_device = cp.empty(cfg.modes, dtype=F64)
    energy_by_network_device = cp.empty(cfg.modes, dtype=F64)
    block_size = max(1, int(getattr(cfg, "sss_block_size", 32)))
    for block_start in range(0, cfg.modes, block_size):
        block_stop = min(block_start + block_size, cfg.modes)
        indices = cp.asarray(order[block_start:block_stop], dtype=cp.int32)
        velocity_block = raw_v[:, indices]
        gradient_block = raw_g[:, indices]
        potential_block = raw_p[:, indices]
        if block_start:
            previous_coefficients = weight * (
                qv[:, :block_start].T @ velocity_block
            )
            projected_velocity = (
                velocity_block - qv[:, :block_start] @ previous_coefficients
            )
            projected_gradient = (
                gradient_block - qg[:, :block_start] @ previous_coefficients
            )
            projected_potential = (
                potential_block - qp[:, :block_start] @ previous_coefficients
            )
        else:
            previous_coefficients = cp.empty((0, block_stop), dtype=F64)
            projected_velocity = velocity_block
            projected_gradient = gradient_block
            projected_potential = potential_block
        block_width = block_stop - block_start
        residual_velocity = cp.empty_like(projected_velocity)
        residual_gradient = cp.empty_like(projected_gradient)
        gradient_residual_velocity = (
            cp.empty_like(projected_velocity) if need_adjoint else None
        )
        gradient_residual_gradient = (
            cp.empty_like(projected_gradient) if need_adjoint else None
        )
        internal_coefficients: list[cp.ndarray] = []
        for local_rank in range(block_width):
            rank = block_start + local_rank
            if local_rank:
                coefficients = weight * (
                    qv[:, block_start:rank].T
                    @ projected_velocity[:, local_rank]
                )
                velocity = (
                    projected_velocity[:, local_rank]
                    - qv[:, block_start:rank] @ coefficients
                )
                gradient = (
                    projected_gradient[:, local_rank]
                    - qg[:, block_start:rank] @ coefficients
                )
                potential = (
                    projected_potential[:, local_rank]
                    - qp[:, block_start:rank] @ coefficients
                )
            else:
                coefficients = cp.empty(0, dtype=F64)
                velocity = projected_velocity[:, local_rank]
                gradient = projected_gradient[:, local_rank]
                potential = projected_potential[:, local_rank]
            internal_coefficients.append(coefficients)
            residual_velocity[:, local_rank] = velocity
            residual_gradient[:, local_rank] = gradient
            residual_mass = weight * cp.dot(velocity, velocity)
            residual_stiffness = weight * cp.dot(gradient, gradient)
            denominator = cp.maximum(residual_mass, cfg.mass_floor)
            rayleigh = residual_stiffness / denominator
            if need_adjoint:
                gradient_residual_velocity[:, local_rank] = (
                    -2.0
                    * residual_stiffness
                    * weight
                    / (cfg.rayleigh_scale * denominator * denominator)
                ) * velocity
                gradient_residual_gradient[:, local_rank] = (
                    2.0 * weight / (cfg.rayleigh_scale * denominator)
                ) * gradient
            normalizer = 1.0 / cp.sqrt(denominator)
            qv[:, rank] = velocity * normalizer
            qg[:, rank] = gradient * normalizer
            qp[:, rank] = potential * normalizer
            if coefficient_matrix is not None:
                column = cp.zeros(cfg.modes, dtype=F64)
                column[indices[local_rank]] = 1.0
                if block_start:
                    column -= (
                        coefficient_matrix[:, :block_start]
                        @ previous_coefficients[:, local_rank]
                    )
                if local_rank:
                    column -= (
                        coefficient_matrix[:, block_start:rank] @ coefficients
                    )
                coefficient_matrix[:, rank] = column * normalizer
            projection = velocity_block[:, local_rank] - velocity
            projection_mass = weight * cp.dot(projection, projection)
            energies_device[rank] = rayleigh
            energy_by_network_device[indices[local_rank]] = rayleigh
            ratios_device[rank] = projection_mass / denominator
        if need_adjoint:
            if gradient_residual_velocity is None or gradient_residual_gradient is None:
                raise RuntimeError("SSS adjoint buffers were not allocated")
            velocity_adjoint = gradient_residual_velocity.copy()
            if block_start:
                coefficient_adjoint = -(
                    qv[:, :block_start].T @ gradient_residual_velocity
                    + qg[:, :block_start].T @ gradient_residual_gradient
                )
                velocity_adjoint += (
                    weight * qv[:, :block_start] @ coefficient_adjoint
                )
            for local_rank in range(block_width):
                rank = block_start + local_rank
                if local_rank:
                    coefficient_adjoint = -(
                        qv[:, block_start:rank].T
                        @ gradient_residual_velocity[:, local_rank]
                        + qg[:, block_start:rank].T
                        @ gradient_residual_gradient[:, local_rank]
                    )
                    velocity_adjoint[:, local_rank] += (
                        weight
                        * qv[:, block_start:rank]
                        @ coefficient_adjoint
                    )
            adj_v[:, indices] = velocity_adjoint
            adj_g[:, indices] = gradient_residual_gradient
    energies = cp.asnumpy(energies_device)
    ratios = cp.asnumpy(ratios_device)
    energy_by_network = cp.asnumpy(energy_by_network_device)
    orth_error = float("nan")
    effective_rank = -1
    if diagnostics:
        gram = weight * (qv.T @ qv)
        orth_error = float(
            cp.max(cp.abs(gram - cp.eye(cfg.modes, dtype=F64))).get()
        )
        raw_gram = weight * (raw_v.T @ raw_v)
        eigenvalues = cp.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
        threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
        effective_rank = int(cp.sum(eigenvalues > threshold).get())
    result: dict[str, Any] = {
        "loss": float(energies.mean() / cfg.rayleigh_scale),
        "mean_energy": float(energies.mean()),
        "max_energy": float(energies.max()),
        "min_energy": float(energies.min()),
        "mean_projection_ratio": float(ratios.mean()),
        "orth_error": orth_error,
        "effective_raw_rank": effective_rank,
        "mode_order": order,
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
    model: OptimizedFanModel,
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
        diagnostics=False,
    )
    input_jet = raw["input_jet"]
    envelope = raw["envelope_jet"]
    point_device = raw["point_device"]
    del raw["potential"], raw["velocity"], raw["gradient"]
    peeled.pop("q_velocity", None)
    block = max(1, int(getattr(cfg, "mode_backward_batch", 8)))
    valid = cp.asarray(True)
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        network_jet, cache = model.forward_jet_block(
            input_jet, start, stop, cache=True
        )
        if cache is None:
            raise RuntimeError("analytic forward cache missing")
        velocity_adjoint = peeled["adjoint_velocity"][:, start:stop].T.reshape(
            stop - start, 3, len(points)
        ).transpose(0, 2, 1)
        gradient_adjoint = peeled["adjoint_gradient"][:, start:stop].T.reshape(
            stop - start, 3, 3, len(points)
        ).transpose(0, 3, 1, 2)
        potential_adjoint = _field_adjoint_to_potential_jet(
            velocity_adjoint.astype(F32), gradient_adjoint.astype(F32)
        )
        equivariant_jet = _network_potential_jet(
            cfg, network_jet, point_device
        )
        equivariant_adjoint = _envelope_product_reverse(
            equivariant_jet, envelope, potential_adjoint
        )
        network_adjoint = _network_potential_jet_reverse(
            cfg, equivariant_adjoint, point_device
        )
        gradients = model.backward_jet_block(
            start, stop, cache, network_adjoint
        )
        valid &= model.adam_step_block(
            start, stop, gradients, learning_rate
        )
    if not bool(valid.get()):
        raise FloatingPointError("blocked analytic reverse produced invalid norms")
    return {
        key: value
        for key, value in peeled.items()
        if key not in {"adjoint_velocity", "adjoint_gradient", "q_velocity"}
    }


def raw_velocity(
    cfg: SimpleNamespace,
    model: OptimizedFanModel,
    points: np.ndarray,
    parameters: np.ndarray,
) -> cp.ndarray:
    point_device = cp.asarray(points, dtype=F32)
    input_jet, envelope = prepared_analytic_inputs(cfg, points, parameters)
    velocity = cp.empty((3 * len(points), cfg.modes), dtype=F64)
    block = max(1, int(cfg.mode_forward_batch))
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        network_jet, _ = model.forward_jet_block(
            input_jet, start, stop, cache=False
        )
        equivariant_jet = _network_potential_jet(
            cfg, network_jet, point_device
        )
        potential_jet = _envelope_product(equivariant_jet, envelope)
        _, block_velocity, _ = _fields_from_potential_jet(potential_jet)
        velocity[:, start:stop] = block_velocity.transpose(2, 1, 0).reshape(
            3 * len(points), stop - start
        )
    return velocity


def inference_basis_mode(
    cfg: SimpleNamespace,
    model: OptimizedFanModel,
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
        diagnostics=True,
    )
    coefficients = peeled["coefficient_matrix"][:, rank]
    visual_raw = raw_velocity(cfg, model, visual_points, parameters)
    selected = visual_raw @ coefficients
    return {
        "velocity": cp.asnumpy(selected.reshape(3, len(visual_points)).T),
        "mode_network": int(peeled["mode_order"][rank]),
        "rayleigh_energy": float(peeled["mode_energies"][rank]),
        "orth_error": float(peeled["orth_error"]),
    }


__all__ = [
    "OptimizedFanModel",
    "boundary_envelope_jet",
    "c4_encoded_input_jet",
    "c4_potential_jet",
    "c4_potential_jet_reverse",
    "continuous_parameters",
    "fluid_points",
    "inference_basis_mode",
    "initialize_arrays",
    "moving_wall_lifting_fields",
    "moving_wall_lifting_jet",
    "peel_and_adjoint",
    "raw_fields",
    "raw_velocity",
    "rotor_distance_jet",
    "train_step",
]
