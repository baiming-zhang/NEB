"""Victoria SA 2025 three-circle geometry hooks for native CUDA jets.

The numerical eigensolver is the production 2D-airfoil backend.  This module
only replaces the domain, the hard slip-boundary streamfunction envelope,
the raw geometry-conditioned encoding, and the geometry sampler.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np
from scipy.stats import qmc


CASE_ROOT = Path(__file__).resolve().parents[1]
AIRFOIL_ROOT = CASE_ROOT.parent / "2D_airfoil"
BACKEND_PATH = (
    AIRFOIL_ROOT
    / "src/eigenfluid_sss/basisnet/airfoil2d/hardcoded_backend.py"
)
sys.path.insert(0, str(AIRFOIL_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d import hardcoded_backend as backend  # noqa: E402


_PRODUCTION_PEEL_STEP = backend.peel_step
_PRODUCTION_ADAM_STEP = backend.HardcodedMode.adam_step
FP32_KEYS = (
    "mlp_compute_dtype",
    "mlp_accumulation_dtype",
    "jet_dtype",
    "parameter_backward_dtype",
    "quadrature_dtype",
    "ritz_dtype",
    "master_weight_dtype",
    "optimizer_dtype",
)
ZERO_GEOMETRY_TOKEN = -1.0


def is_zero_geometry_token(token: float) -> bool:
    """Return whether ``token`` denotes the explicit no-metaball mapping."""
    return abs(float(token) - ZERO_GEOMETRY_TOKEN) <= 1.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.lru_cache(maxsize=32768)
def _geometry_from_key(
    seed: int,
    token: int,
    center_min: float,
    center_max: float,
    radius_min: float,
    radius_max: float,
    connectivity: str,
    minimum_overlap: float,
    outer_half_extent: float,
    outer_superellipse_power: int,
    outer_clearance: float,
    maximum_attempts: int,
) -> tuple[float, ...]:
    """Draw one deterministic geometry, optionally conditioned on connectivity.

    ``independent`` reproduces the original Victoria-paper distribution.  The
    connected variant is rejection sampling from that same nine-dimensional
    uniform distribution, conditioned on (i) a connected circle-overlap graph
    and (ii) the smoothed obstacle remaining inside the rounded outer wall.
    It therefore introduces no reflection, rotation, or analytic mode prior.
    """
    mixed_seed = (int(seed) + 104729 * int(token)) % (2**63 - 1)
    rng = np.random.default_rng(mixed_seed)
    for _ in range(int(maximum_attempts)):
        sample = rng.random(9)
        geometry = np.empty(9, dtype=np.float64)
        for circle in range(3):
            base = 3 * circle
            geometry[base:base + 2] = center_min + sample[base:base + 2] * (
                center_max - center_min
            )
            geometry[base + 2] = radius_min + sample[base + 2] * (
                radius_max - radius_min
            )
        if connectivity == "independent":
            return tuple(float(value) for value in geometry)
        if connectivity != "connected_smooth_union":
            raise ValueError(f"unsupported metaball_connectivity={connectivity!r}")
        if not _circle_overlap_graph_connected(geometry, minimum_overlap):
            continue
        if not _circles_inside_rounded_square(
            geometry,
            outer_half_extent,
            outer_superellipse_power,
            outer_clearance,
        ):
            continue
        return tuple(float(value) for value in geometry)
    raise RuntimeError(
        "could not sample a connected interior three-circle metaball after "
        f"{maximum_attempts} attempts for token {token}"
    )


def _circle_overlap_graph_connected(
    geometry: np.ndarray, minimum_overlap: float
) -> bool:
    """Return whether all three discs form one overlap-connected component."""
    circles = geometry.reshape(3, 3)
    adjacency = [set() for _ in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            distance = float(np.linalg.norm(circles[left, :2] - circles[right, :2]))
            overlap = float(circles[left, 2] + circles[right, 2] - distance)
            if overlap >= minimum_overlap:
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == 3


def _circles_inside_rounded_square(
    geometry: np.ndarray,
    half_extent: float,
    superellipse_power: int,
    clearance: float,
) -> bool:
    """Conservatively keep every disc inside a shrunken rounded square."""
    effective_half = float(half_extent) - float(clearance)
    if effective_half <= 0.0:
        return False
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    for cx, cy, radius in geometry.reshape(3, 3):
        boundary = np.asarray([cx, cy]) + float(radius) * directions
        normalized = np.abs((boundary - 0.5) / effective_half)
        if np.any(np.sum(normalized**int(superellipse_power), axis=1) >= 1.0):
            return False
    return True


def geometry_from_token(cfg: SimpleNamespace, token: float) -> np.ndarray:
    if is_zero_geometry_token(token):
        return np.zeros(9, dtype=np.float32)
    rounded = int(round(float(token)))
    if abs(float(token) - rounded) > 1.0e-5:
        raise ValueError(f"geometry token must be integer-valued, got {token}")
    return np.asarray(
        _geometry_from_key(
            int(cfg.seed),
            rounded,
            float(cfg.circle_center_min),
            float(cfg.circle_center_max),
            float(cfg.circle_radius_min),
            float(cfg.circle_radius_max),
            str(getattr(cfg, "metaball_connectivity", "independent")),
            float(getattr(cfg, "metaball_minimum_overlap", 0.0)),
            float(cfg.outer_half_extent),
            int(cfg.outer_superellipse_power),
            float(getattr(cfg, "metaball_outer_clearance", 0.0)),
            int(getattr(cfg, "metaball_geometry_max_attempts", 10000)),
        ),
        dtype=np.float32,
    )


def _sqrt_jet(source: backend.Jet3) -> backend.Jet3:
    value = cp.sqrt(source.value)
    first = 0.5 / value
    second = -0.25 / (value**3)
    third = 0.375 / (value**5)
    return backend.unary(source, value, first, second, third)


def _exp_jet(source: backend.Jet3) -> backend.Jet3:
    value = cp.exp(source.value)
    return backend.unary(source, value, value, value, value)


def _log_jet(source: backend.Jet3) -> backend.Jet3:
    inverse = 1.0 / source.value
    return backend.unary(
        source,
        cp.log(source.value),
        inverse,
        -(inverse**2),
        2.0 * inverse**3,
    )


def _sqrt_jet1(source: backend.Jet1) -> backend.Jet1:
    """Square root carrying only the value and first spatial gradient."""
    value = cp.sqrt(source.value)
    return backend.unary1(source, value, 0.5 / value)


def _exp_jet1(source: backend.Jet1) -> backend.Jet1:
    """Exponential carrying only the value and first spatial gradient."""
    value = cp.exp(source.value)
    return backend.unary1(source, value, value)


def _log_jet1(source: backend.Jet1) -> backend.Jet1:
    """Logarithm carrying only the value and first spatial gradient."""
    inverse = 1.0 / source.value
    return backend.unary1(source, cp.log(source.value), inverse)


def _sqrt_jet2(source: backend.Jet2) -> backend.Jet2:
    """Square root carrying value, gradient, and compact Hessian."""
    value = cp.sqrt(source.value)
    return backend.unary2(
        source, value, 0.5 / value, -0.25 / (value**3)
    )


def _exp_jet2(source: backend.Jet2) -> backend.Jet2:
    value = cp.exp(source.value)
    return backend.unary2(source, value, value, value)


def _log_jet2(source: backend.Jet2) -> backend.Jet2:
    inverse = 1.0 / source.value
    return backend.unary2(
        source, cp.log(source.value), inverse, -(inverse**2)
    )


def _softmin_circle_distance_jet(
    cfg: SimpleNamespace,
    x: backend.Jet3,
    y: backend.Jet3,
    geometry: np.ndarray,
) -> backend.Jet3:
    tau = float(cfg.metaball_softmin_tau)
    distances: list[backend.Jet3] = []
    for cx, cy, radius in geometry.reshape(3, 3):
        squared = (x - float(cx)).square() + (y - float(cy)).square()
        distances.append(_sqrt_jet(squared) - float(radius))
    # Use an independent value-only shift for every quadrature point.  The
    # log-sum-exp identity is invariant to this constant, so its spatial
    # derivatives are unchanged, while the exponential sum remains in [1,3]
    # pointwise.  A single batch-wide shift can make the sum ~1e-30 at points
    # far from the closest circle; the explicit third derivative of log then
    # overflows in FP32 even though the composed soft minimum is finite.
    shift_value = cp.minimum(
        cp.minimum(distances[0].value, distances[1].value),
        distances[2].value,
    )
    shift = backend.Jet3.constant(
        shift_value, x.value.shape, x.value.dtype.type
    )
    total = _exp_jet(-(distances[0] - shift) / tau)
    for distance in distances[1:]:
        total = total + _exp_jet(-(distance - shift) / tau)
    return shift - tau * _log_jet(total)


def _softmin_circle_distance_jet1(
    cfg: SimpleNamespace,
    x: backend.Jet1,
    y: backend.Jet1,
    geometry: np.ndarray,
) -> backend.Jet1:
    """Jet1 soft minimum matching the production Jet3 value/gradient path."""
    tau = float(cfg.metaball_softmin_tau)
    distances: list[backend.Jet1] = []
    for cx, cy, radius in geometry.reshape(3, 3):
        squared = (x - float(cx)).square() + (y - float(cy)).square()
        distances.append(_sqrt_jet1(squared) - float(radius))
    shift_value = cp.minimum(
        cp.minimum(distances[0].value, distances[1].value),
        distances[2].value,
    )
    shift = backend.Jet1.constant(
        shift_value, x.value.shape, x.value.dtype.type
    )
    total = _exp_jet1(-(distances[0] - shift) / tau)
    for distance in distances[1:]:
        total = total + _exp_jet1(-(distance - shift) / tau)
    return shift - tau * _log_jet1(total)


def _softmin_circle_distance_jet2(
    cfg: SimpleNamespace,
    x: backend.Jet2,
    y: backend.Jet2,
    geometry: np.ndarray,
) -> backend.Jet2:
    """Compact-Hessian soft minimum for velocity-Rayleigh training."""
    tau = float(cfg.metaball_softmin_tau)
    distances: list[backend.Jet2] = []
    for cx, cy, radius in geometry.reshape(3, 3):
        squared = (x - float(cx)).square() + (y - float(cy)).square()
        distances.append(_sqrt_jet2(squared) - float(radius))
    shift_value = cp.minimum(
        cp.minimum(distances[0].value, distances[1].value),
        distances[2].value,
    )
    shift = backend.Jet2.constant(
        shift_value, x.value.shape, x.value.dtype.type
    )
    total = _exp_jet2(-(distances[0] - shift) / tau)
    for distance in distances[1:]:
        total = total + _exp_jet2(-(distance - shift) / tau)
    return shift - tau * _log_jet2(total)


def metaball_envelope_jet(
    cfg: SimpleNamespace,
    x: backend.Jet3,
    y: backend.Jet3,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet3:
    """C-infinity single-zero streamfunction envelope on every boundary."""
    del unused_alpha
    half = float(cfg.outer_half_extent)
    power = int(cfg.outer_superellipse_power)
    xn = (x - 0.5) / half
    yn = (y - 0.5) / half
    outer_level = 1.0 - xn.pow(power) - yn.pow(power)
    outer_scale = float(cfg.outer_envelope_transition)
    outer = outer_level / (outer_scale + outer_level)
    if is_zero_geometry_token(geometry_token):
        return outer
    geometry = geometry_from_token(cfg, geometry_token)
    obstacle_level = _softmin_circle_distance_jet(cfg, x, y, geometry)
    obstacle_scale = float(cfg.obstacle_envelope_transition)
    obstacle = obstacle_level / (obstacle_scale + obstacle_level)
    return outer * obstacle


def metaball_envelope_jet1(
    cfg: SimpleNamespace,
    x: backend.Jet1,
    y: backend.Jet1,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet1:
    """Jet1 C-infinity boundary envelope without Hessian/third storage."""
    del unused_alpha
    half = float(cfg.outer_half_extent)
    power = int(cfg.outer_superellipse_power)
    xn = (x - 0.5) / half
    yn = (y - 0.5) / half
    outer_level = 1.0 - xn.pow(power) - yn.pow(power)
    outer_scale = float(cfg.outer_envelope_transition)
    outer = outer_level / (outer_scale + outer_level)
    if is_zero_geometry_token(geometry_token):
        return outer
    geometry = geometry_from_token(cfg, geometry_token)
    obstacle_level = _softmin_circle_distance_jet1(cfg, x, y, geometry)
    obstacle_scale = float(cfg.obstacle_envelope_transition)
    obstacle = obstacle_level / (obstacle_scale + obstacle_level)
    return outer * obstacle


def metaball_envelope_jet2(
    cfg: SimpleNamespace,
    x: backend.Jet2,
    y: backend.Jet2,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet2:
    """Jet2 C-infinity boundary envelope for velocity Rayleigh."""
    del unused_alpha
    half = float(cfg.outer_half_extent)
    power = int(cfg.outer_superellipse_power)
    xn = (x - 0.5) / half
    yn = (y - 0.5) / half
    outer_level = 1.0 - xn.pow(power) - yn.pow(power)
    outer_scale = float(cfg.outer_envelope_transition)
    outer = outer_level / (outer_scale + outer_level)
    if is_zero_geometry_token(geometry_token):
        return outer
    geometry = geometry_from_token(cfg, geometry_token)
    obstacle_level = _softmin_circle_distance_jet2(cfg, x, y, geometry)
    obstacle_scale = float(cfg.obstacle_envelope_transition)
    obstacle = obstacle_level / (obstacle_scale + obstacle_level)
    return outer * obstacle


def _concatenate(features: list[backend.Jet3]) -> backend.Jet3:
    return backend.Jet3(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
        cp.concatenate([item.hess for item in features], axis=1),
        cp.concatenate([item.third for item in features], axis=1),
    )


def _concatenate_jet1(features: list[backend.Jet1]) -> backend.Jet1:
    return backend.Jet1(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
    )


def _concatenate_jet2(features: list[backend.Jet2]) -> backend.Jet2:
    return backend.Jet2(
        cp.concatenate([item.value for item in features], axis=1),
        cp.concatenate([item.grad for item in features], axis=1),
        cp.concatenate([item.hess for item in features], axis=1),
    )


def metaball_encoding_jet(
    cfg: SimpleNamespace,
    x: backend.Jet3,
    y: backend.Jet3,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet3:
    """Normalize raw spatial and geometry inputs with the configured convention."""
    del unused_alpha
    zero_geometry = is_zero_geometry_token(geometry_token)
    geometry = geometry_from_token(cfg, geometry_token)
    # A strict SIREN receives normalized raw coordinates.  Its omega factors
    # belong inside every sine layer, not in this encoder.  Keep the legacy
    # spatial-only scaling solely for old, explicitly non-canonical campaigns.
    normalization = str(
        getattr(cfg, "coordinate_normalization", "centered_unit")
    )
    if normalization == "unit_interval":
        xn, yn = x, y
    elif normalization == "centered_unit":
        xn, yn = 2.0 * x - 1.0, 2.0 * y - 1.0
    elif normalization == "standardized_uniform":
        scale = math.sqrt(12.0)
        xn, yn = scale * (x - 0.5), scale * (y - 0.5)
    else:
        raise ValueError(f"unsupported coordinate normalization {normalization!r}")
    if not bool(getattr(cfg, "canonical_siren", False)):
        spatial_scale = float(
            getattr(cfg, "siren_first_layer_spatial_omega0", 1.0)
        )
        xn, yn = spatial_scale * xn, spatial_scale * yn
    shape = x.value.shape
    constants: list[backend.Jet3] = []
    for circle in range(3):
        cx, cy, radius = geometry[3 * circle:3 * circle + 3]
        if zero_geometry:
            constants.extend(
                [
                    backend.Jet3.constant(0.0, shape, x.value.dtype.type),
                    backend.Jet3.constant(0.0, shape, x.value.dtype.type),
                    backend.Jet3.constant(0.0, shape, x.value.dtype.type),
                ]
            )
            continue
        center_u = (float(cx) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        radius_u = (float(radius) - float(cfg.circle_radius_min)) / (
            float(cfg.circle_radius_max) - float(cfg.circle_radius_min)
        )
        center_y_u = (float(cy) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        if normalization == "unit_interval":
            center_x, center_y = center_u, center_y_u
        else:
            factor = 2.0 if normalization == "centered_unit" else math.sqrt(12.0)
            center_x = factor * (center_u - 0.5)
            center_y = factor * (center_y_u - 0.5)
            radius_u = factor * (radius_u - 0.5)
        constants.extend(
            [
                backend.Jet3.constant(center_x, shape, x.value.dtype.type),
                backend.Jet3.constant(center_y, shape, x.value.dtype.type),
                backend.Jet3.constant(radius_u, shape, x.value.dtype.type),
            ]
        )
    encoded = _concatenate([xn, yn, *constants])
    if encoded.value.shape[1] != int(cfg.input_features):
        raise RuntimeError("metaball raw encoding width mismatch")
    return encoded


def metaball_encoding_jet1(
    cfg: SimpleNamespace,
    x: backend.Jet1,
    y: backend.Jet1,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet1:
    """Jet1 normalized raw xy + circle encoding with no higher derivatives."""
    del unused_alpha
    zero_geometry = is_zero_geometry_token(geometry_token)
    geometry = geometry_from_token(cfg, geometry_token)
    normalization = str(
        getattr(cfg, "coordinate_normalization", "centered_unit")
    )
    if normalization == "unit_interval":
        xn, yn = x, y
    elif normalization == "centered_unit":
        xn, yn = 2.0 * x - 1.0, 2.0 * y - 1.0
    elif normalization == "standardized_uniform":
        scale = math.sqrt(12.0)
        xn, yn = scale * (x - 0.5), scale * (y - 0.5)
    else:
        raise ValueError(f"unsupported coordinate normalization {normalization!r}")
    if not bool(getattr(cfg, "canonical_siren", False)):
        spatial_scale = float(
            getattr(cfg, "siren_first_layer_spatial_omega0", 1.0)
        )
        xn, yn = spatial_scale * xn, spatial_scale * yn
    shape = x.value.shape
    constants: list[backend.Jet1] = []
    for circle in range(3):
        cx, cy, radius = geometry[3 * circle:3 * circle + 3]
        if zero_geometry:
            constants.extend(
                [backend.Jet1.constant(0.0, shape, x.value.dtype.type)] * 3
            )
            continue
        center_u = (float(cx) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        center_y_u = (float(cy) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        radius_u = (float(radius) - float(cfg.circle_radius_min)) / (
            float(cfg.circle_radius_max) - float(cfg.circle_radius_min)
        )
        if normalization == "unit_interval":
            center_x, center_y = center_u, center_y_u
        else:
            factor = 2.0 if normalization == "centered_unit" else math.sqrt(12.0)
            center_x = factor * (center_u - 0.5)
            center_y = factor * (center_y_u - 0.5)
            radius_u = factor * (radius_u - 0.5)
        constants.extend(
            [
                backend.Jet1.constant(center_x, shape, x.value.dtype.type),
                backend.Jet1.constant(center_y, shape, x.value.dtype.type),
                backend.Jet1.constant(radius_u, shape, x.value.dtype.type),
            ]
        )
    encoded = _concatenate_jet1([xn, yn, *constants])
    if encoded.value.shape[1] != int(cfg.input_features):
        raise RuntimeError("metaball raw encoding width mismatch")
    return encoded


def metaball_encoding_jet2(
    cfg: SimpleNamespace,
    x: backend.Jet2,
    y: backend.Jet2,
    geometry_token: float,
    unused_alpha: float,
) -> backend.Jet2:
    """Jet2 normalized raw xy plus three-circle condition encoding."""
    del unused_alpha
    zero_geometry = is_zero_geometry_token(geometry_token)
    geometry = geometry_from_token(cfg, geometry_token)
    normalization = str(
        getattr(cfg, "coordinate_normalization", "centered_unit")
    )
    if normalization == "unit_interval":
        xn, yn = x, y
    elif normalization == "centered_unit":
        xn, yn = 2.0 * x - 1.0, 2.0 * y - 1.0
    elif normalization == "standardized_uniform":
        scale = math.sqrt(12.0)
        xn, yn = scale * (x - 0.5), scale * (y - 0.5)
    else:
        raise ValueError(
            f"unsupported coordinate normalization {normalization!r}"
        )
    if not bool(getattr(cfg, "canonical_siren", False)):
        spatial_scale = float(
            getattr(cfg, "siren_first_layer_spatial_omega0", 1.0)
        )
        xn, yn = spatial_scale * xn, spatial_scale * yn
    shape = x.value.shape
    constants: list[backend.Jet2] = []
    for circle in range(3):
        cx, cy, radius = geometry[3 * circle:3 * circle + 3]
        if zero_geometry:
            constants.extend(
                [backend.Jet2.constant(0.0, shape, x.value.dtype.type)] * 3
            )
            continue
        center_u = (float(cx) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        center_y_u = (float(cy) - float(cfg.circle_center_min)) / (
            float(cfg.circle_center_max) - float(cfg.circle_center_min)
        )
        radius_u = (float(radius) - float(cfg.circle_radius_min)) / (
            float(cfg.circle_radius_max) - float(cfg.circle_radius_min)
        )
        if normalization == "unit_interval":
            center_x, center_y = center_u, center_y_u
        else:
            factor = (
                2.0
                if normalization == "centered_unit"
                else math.sqrt(12.0)
            )
            center_x = factor * (center_u - 0.5)
            center_y = factor * (center_y_u - 0.5)
            radius_u = factor * (radius_u - 0.5)
        constants.extend(
            [
                backend.Jet2.constant(
                    center_x, shape, x.value.dtype.type
                ),
                backend.Jet2.constant(
                    center_y, shape, x.value.dtype.type
                ),
                backend.Jet2.constant(
                    radius_u, shape, x.value.dtype.type
                ),
            ]
        )
    encoded = _concatenate_jet2([xn, yn, *constants])
    if encoded.value.shape[1] != int(cfg.input_features):
        raise RuntimeError("metaball raw encoding width mismatch")
    return encoded


def outer_level_numpy(cfg: SimpleNamespace, points: np.ndarray) -> np.ndarray:
    half = float(cfg.outer_half_extent)
    power = int(cfg.outer_superellipse_power)
    return 1.0 - ((points[:, 0] - 0.5) / half) ** power - (
        (points[:, 1] - 0.5) / half
    ) ** power


def obstacle_level_numpy(
    cfg: SimpleNamespace, points: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    centers = geometry.reshape(3, 3)[:, :2]
    radii = geometry.reshape(3, 3)[:, 2]
    distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    distances -= radii[None, :]
    tau = float(cfg.metaball_softmin_tau)
    minimum = np.min(distances, axis=1, keepdims=True)
    return minimum[:, 0] - tau * np.log(
        np.sum(np.exp(-(distances - minimum) / tau), axis=1)
    )


def fluid_mask(
    cfg: SimpleNamespace, points: np.ndarray, geometry_token: float
) -> np.ndarray:
    if is_zero_geometry_token(geometry_token):
        return outer_level_numpy(cfg, points) > 0.0
    geometry = geometry_from_token(cfg, geometry_token)
    return (outer_level_numpy(cfg, points) > 0.0) & (
        obstacle_level_numpy(cfg, points, geometry) > 0.0
    )


def metaball_fluid_points(
    cfg: SimpleNamespace,
    count: int,
    geometry_token: float,
    unused_alpha: float,
    seed: int,
) -> np.ndarray:
    del unused_alpha
    sampling_mode = str(
        getattr(cfg, "sampling_mode", "uniform_fluid_area_sobol")
    )
    sampler = (
        qmc.Sobol(2, scramble=True, seed=seed)
        if sampling_mode == "uniform_fluid_area_sobol"
        else None
    )
    rng = np.random.default_rng(seed) if sampling_mode == "uniform_random" else None
    if sampler is None and rng is None:
        raise ValueError(f"unsupported metaball sampling_mode={sampling_mode!r}")
    pieces: list[np.ndarray] = []
    remaining = int(count)
    while remaining:
        draw_count = max(2 * remaining, 256)
        if rng is not None:
            sample = rng.random((draw_count, 2))
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The balance properties of Sobol",
                    category=UserWarning,
                )
                sample = sampler.random(draw_count)
        accepted = sample[fluid_mask(cfg, sample, geometry_token)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def metaball_continuous_parameters(
    cfg: SimpleNamespace, epoch: int
) -> tuple[float, float]:
    del cfg
    return float(epoch), 0.0


def raw_basis_field_matrices(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    geometry_token: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return raw network vorticity and velocity matrices."""
    count, modes = len(points), int(cfg.modes)
    omega = cp.empty((count, modes), dtype=backend.F32)
    velocity = cp.empty((count, 2, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, geometry_token, 0.0, False)
    ):
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        velocity[:, 0, index] = field.grad[:, 0, 1]
        velocity[:, 1, index] = -field.grad[:, 0, 0]
    return omega, velocity


def vorticity_peeling_transform(
    cfg: SimpleNamespace,
    model: Any,
    reference_points: np.ndarray,
    geometry_token: float,
) -> tuple[cp.ndarray, dict[str, Any]]:
    """Build the training-equivalent FP32 vorticity MGS transform.

    If ``W`` contains raw vorticity columns, the returned transform ``T``
    satisfies ``Q = W @ T``.  ``Q`` is the twice-
    reorthogonalized, unit-mass peeled basis on ``reference_points``.
    The same ``T`` must be applied to raw velocity fields.
    """
    raw_omega, _ = raw_basis_field_matrices(
        cfg, model, reference_points, geometry_token
    )
    count, modes = raw_omega.shape
    box_area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
    fluid_area = box_area - excluded_area(cfg, geometry_token)
    weight = cp.asarray(np.float32(fluid_area / count), dtype=backend.F32)
    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    raw_rayleigh_by_mode: list[float] = []
    if causal_sorting == "geometry_rayleigh":
        for _, field, _ in model.forward_stream(
            cfg, reference_points, geometry_token, 0.0, False
        ):
            omega, grad_omega = backend.vorticity_and_gradient(field)
            omega_vector = omega[:, 0].astype(backend.F32, copy=False)
            grad_vector = grad_omega[:, 0].reshape(-1).astype(
                backend.F32, copy=False
            )
            mass = weight * cp.dot(omega_vector, omega_vector)
            stiffness = weight * cp.dot(grad_vector, grad_vector)
            raw_rayleigh_by_mode.append(float((stiffness / mass).get()))
        causal_order = np.argsort(
            np.asarray(raw_rayleigh_by_mode, dtype=np.float64), kind="stable"
        ).astype(np.int64).tolist()
    elif causal_sorting == "fixed_order":
        causal_order = list(range(modes))
    else:
        raise ValueError(f"unsupported causal_sorting {causal_sorting!r}")
    basis = cp.empty_like(raw_omega)
    transform = cp.zeros((modes, modes), dtype=backend.F32)
    residual_masses: list[float] = []
    passes = max(1, int(getattr(cfg, "reorthogonalization_passes", 2)))
    for causal_rank, raw_mode in enumerate(causal_order):
        residual = raw_omega[:, raw_mode].copy()
        coefficients = cp.zeros((modes,), dtype=backend.F32)
        coefficients[raw_mode] = np.float32(1.0)
        if causal_rank:
            previous = basis[:, :causal_rank]
            previous_transform = transform[:, :causal_rank]
            for _ in range(passes):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= float(cfg.mass_floor):
            raise FloatingPointError(
                f"peeled causal mode {causal_rank + 1} (network "
                f"{raw_mode + 1}) is unresolved: M={mass_value:.8e}"
            )
        normalizer = cp.reciprocal(cp.sqrt(mass))
        basis[:, causal_rank] = residual * normalizer
        transform[:, causal_rank] = coefficients * normalizer
        residual_masses.append(mass_value)

    raw_gram = weight * (raw_omega.T @ raw_omega)
    raw_gram = np.float32(0.5) * (raw_gram + raw_gram.T)
    peeled_gram = weight * (basis.T @ basis)
    peeled_gram = np.float32(0.5) * (peeled_gram + peeled_gram.T)
    raw_eigenvalues = cp.linalg.eigvalsh(raw_gram)
    peeled_eigenvalues = cp.linalg.eigvalsh(peeled_gram)
    relative_threshold = np.finfo(np.float32).eps * max(count, modes)
    raw_largest = cp.maximum(cp.abs(raw_eigenvalues[-1]), cfg.mass_floor)
    raw_rank = int(
        cp.count_nonzero(raw_eigenvalues > relative_threshold * raw_largest).get()
    )
    peeled_largest = cp.maximum(cp.abs(peeled_eigenvalues[-1]), cfg.mass_floor)
    peeled_rank = int(
        cp.count_nonzero(
            peeled_eigenvalues > relative_threshold * peeled_largest
        ).get()
    )
    identity = cp.eye(modes, dtype=backend.F32)
    diagnostics = {
        "reference_points": int(count),
        "reorthogonalization_passes": passes,
        "causal_sorting": causal_sorting,
        "causal_order": [int(index) for index in causal_order],
        "causal_order_1based": [int(index) + 1 for index in causal_order],
        "raw_mode_energies": raw_rayleigh_by_mode,
        "fluid_area": float(fluid_area),
        "rank_relative_threshold": float(relative_threshold),
        "raw_effective_rank": raw_rank,
        "peeled_effective_rank": peeled_rank,
        "peeled_orthogonality_max_abs": float(
            cp.max(cp.abs(peeled_gram - identity)).get()
        ),
        "raw_gram_eigenvalue_min": float(raw_eigenvalues[0].get()),
        "raw_gram_eigenvalue_max": float(raw_eigenvalues[-1].get()),
        "peeled_gram_eigenvalue_min": float(peeled_eigenvalues[0].get()),
        "peeled_gram_eigenvalue_max": float(peeled_eigenvalues[-1].get()),
        "residual_mass_min": float(min(residual_masses)),
        "residual_mass_max": float(max(residual_masses)),
        "transform_frobenius": float(cp.linalg.norm(transform).get()),
        "transform_abs_max": float(cp.max(cp.abs(transform)).get()),
    }
    del raw_omega, basis, raw_gram, peeled_gram
    return transform, diagnostics


def apply_basis_transform(
    omega: cp.ndarray,
    velocity: cp.ndarray,
    transform: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Apply one mode-space transform to vorticity and velocity fields."""
    return omega @ transform, cp.einsum(
        "ncm,mk->nck", velocity, transform, optimize=True
    )


@functools.lru_cache(maxsize=32768)
def _fluid_area_cached(
    seed: int,
    token: int,
    half: float,
    power: int,
    tau: float,
    center_min: float,
    center_max: float,
    radius_min: float,
    radius_max: float,
    connectivity: str,
    minimum_overlap: float,
    outer_clearance: float,
    maximum_attempts: int,
    count: int,
) -> float:
    cfg = SimpleNamespace(
        seed=seed,
        outer_half_extent=half,
        outer_superellipse_power=power,
        metaball_softmin_tau=tau,
        circle_center_min=center_min,
        circle_center_max=center_max,
        circle_radius_min=radius_min,
        circle_radius_max=radius_max,
        metaball_connectivity=connectivity,
        metaball_minimum_overlap=minimum_overlap,
        metaball_outer_clearance=outer_clearance,
        metaball_geometry_max_attempts=maximum_attempts,
    )
    sampler = qmc.Sobol(2, scramble=True, seed=seed + 3000003 + token)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The balance properties of Sobol", category=UserWarning
        )
        points = sampler.random(count)
    return float(np.mean(fluid_mask(cfg, points, float(token)), dtype=np.float64))


def excluded_area(cfg: SimpleNamespace, geometry_token: float) -> float:
    token = int(round(float(geometry_token)))
    fluid_area = _fluid_area_cached(
        int(cfg.seed), token, float(cfg.outer_half_extent),
        int(cfg.outer_superellipse_power), float(cfg.metaball_softmin_tau),
        float(cfg.circle_center_min), float(cfg.circle_center_max),
        float(cfg.circle_radius_min), float(cfg.circle_radius_max),
        str(getattr(cfg, "metaball_connectivity", "independent")),
        float(getattr(cfg, "metaball_minimum_overlap", 0.0)),
        float(getattr(cfg, "metaball_outer_clearance", 0.0)),
        int(getattr(cfg, "metaball_geometry_max_attempts", 10000)),
        int(cfg.area_quadrature_points),
    )
    box_area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
    return float(box_area - fluid_area)


def metaball_peel_step(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    geometry_token: float,
    alpha: float,
    training: bool,
    reference_model: Any | None = None,
    full_diagnostics: bool = True,
) -> dict[str, Any]:
    geometry_mode = cfg.geometry_mode
    had_single_pass_flag = hasattr(cfg, "allow_single_mgs_pass")
    single_pass_flag = getattr(cfg, "allow_single_mgs_pass", None)
    try:
        cfg.geometry_mode = "cinf_ellipse"
        cfg.allow_single_mgs_pass = True
        return _PRODUCTION_PEEL_STEP(
            cfg, model, points, geometry_token, alpha, training,
            reference_model=reference_model,
            full_diagnostics=full_diagnostics,
        )
    finally:
        cfg.geometry_mode = geometry_mode
        if had_single_pass_flag:
            cfg.allow_single_mgs_pass = single_pass_flag
        else:
            delattr(cfg, "allow_single_mgs_pass")


def all_fp32_adam_step(
    self: Any, gradients: list[cp.ndarray], cfg: SimpleNamespace
) -> None:
    _PRODUCTION_ADAM_STEP(self, gradients, cfg)
    if self.compute_dtype is backend.F32:
        for shadow, master in zip(self.shadow_weights, self.weights):
            shadow[...] = master
        for shadow, master in zip(self.shadow_biases, self.biases):
            shadow[...] = master


def validate_metaball_config(
    config: dict[str, Any], production_validate: Any
) -> None:
    if config.get("case_domain") not in {
        "victoria_sa2025_three_circle_metaball",
        "victoria_sa2025_three_circle_connected_metaball",
    }:
        raise ValueError("unexpected metaball case_domain")
    if config.get("geometry_mode") not in {
        "metaball_three_circle_family",
        "metaball_three_circle_connected_family",
    }:
        raise ValueError("unexpected geometry_mode")
    if config.get("activation") != "siren":
        raise ValueError("the requested case must use SIREN")
    allowed_encodings = {
        "raw_xy_plus_circle_parameters",
        "normalized_raw_xy_plus_circle_parameters",
    }
    if config.get("input_encoding") not in allowed_encodings:
        raise ValueError("the requested case must use raw xy+geometry input")
    if int(config.get("input_features", -1)) != 11:
        raise ValueError("three-circle raw input must have 11 features")
    if config.get("causal_sorting", "fixed_order") not in {
        "fixed_order", "geometry_rayleigh"
    }:
        raise ValueError("unsupported metaball causal_sorting")
    if config.get("basis_objective") not in {
        "pure_vorticity_rayleigh", "streamfunction_dirichlet_rayleigh",
        "velocity_rayleigh",
    }:
        raise ValueError("unsupported metaball basis_objective")
    if config.get("sampling_mode") not in {
        "uniform_fluid_area_sobol", "uniform_random"
    }:
        raise ValueError("unsupported metaball sampling_mode")
    connectivity = str(config.get("metaball_connectivity", "independent"))
    if connectivity not in {"independent", "connected_smooth_union"}:
        raise ValueError("unsupported metaball_connectivity")
    if config.get("geometry_mode") == "metaball_three_circle_connected_family":
        if connectivity != "connected_smooth_union":
            raise ValueError("connected geometry mode requires connected_smooth_union")
        if float(config.get("metaball_minimum_overlap", -1.0)) < 0.0:
            raise ValueError("metaball_minimum_overlap must be non-negative")
        if float(config.get("metaball_outer_clearance", -1.0)) < 0.0:
            raise ValueError("metaball_outer_clearance must be non-negative")
    if float(config.get("siren_first_layer_spatial_omega0", 0.0)) <= 0.0:
        raise ValueError("SIREN spatial omega0 must be positive")
    campaign_size = (
        int(config.get("modes", -1)), int(config.get("epochs", -1))
    )
    allowed_campaign_sizes = {
        (64, 500),
        (64, 2000),
        (1024, 1000),
        (1024, 2000),
        (1024, 5000),
        (2048, 2000),
    }
    if campaign_size not in allowed_campaign_sizes:
        raise ValueError(
            "supported metaball campaigns are K=64/E=500, K=64/E=2000, "
            "K=1024/E=1000, K=1024/E=2000, K=1024/E=5000, "
            "or K=2048/E=2000"
        )
    if bool(config.get("canonical_siren", False)):
        if config.get("coordinate_normalization", "centered_unit") not in {
            "unit_interval", "centered_unit", "standardized_uniform"
        }:
            raise ValueError("unsupported canonical metaball coordinate normalization")
        if float(config.get("siren_first_omega0", 0.0)) <= 0.0:
            raise ValueError("canonical first-layer omega0 must be positive")
        if float(config.get("siren_hidden_omega0", 0.0)) <= 0.0:
            raise ValueError("canonical hidden-layer omega0 must be positive")
    wrong = {
        key: config.get(key) for key in FP32_KEYS
        if str(config.get(key, "")).lower() != "float32"
    }
    if wrong:
        raise ValueError(f"metaball campaign is all-FP32: {wrong}")
    shadow = dict(config)
    shadow["geometry_mode"] = "cinf_ellipse"
    shadow["envelope_mode"] = "cinf_farfield_freestream"
    shadow["sampling_mode"] = "uniform_fluid_area_sobol"
    shadow["mlp_compute_dtype"] = "float16"
    # The shared airfoil validator fixes its rolling-checkpoint cadence at 100
    # epochs.  Metaball campaigns may intentionally use a different cadence;
    # validate the shared assumptions with the airfoil value while preserving
    # the real metaball config consumed by the trainer.
    shadow["checkpoint_every"] = 100
    production_validate(shadow)


def install_backend(config: dict[str, Any]) -> ModuleType:
    backend.envelope_jet = metaball_envelope_jet
    backend.encoding_jet = metaball_encoding_jet
    backend.envelope_jet1 = metaball_envelope_jet1
    backend.encoding_jet1 = metaball_encoding_jet1
    backend.envelope_jet2 = metaball_envelope_jet2
    backend.encoding_jet2 = metaball_encoding_jet2
    backend.fluid_points = metaball_fluid_points
    backend.continuous_parameters = metaball_continuous_parameters
    backend.airfoil_area = excluded_area
    backend.peel_step = metaball_peel_step
    backend.HardcodedMode.adam_step = all_fp32_adam_step
    return backend


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    jet_order = int(config.get("spatial_derivative_order", 3))
    jet_label = f"Jet{jet_order}"
    return {
        "source_case": "Neural Kinematic Bases for Fluids, SIGGRAPH Asia 2025",
        "geometry_family": {
            "outer_domain": "unit p=8 superellipse (rounded square)",
            "holes": "soft-min union of three circles",
            "circle_centers": "independent uniform [0.25,0.75]^2",
            "circle_radii": "independent uniform [0.1,0.3]",
            "overlap": (
                "rejection-conditioned to one connected component"
                if config.get("metaball_connectivity") == "connected_smooth_union"
                else "allowed"
            ),
            "minimum_overlap": config.get("metaball_minimum_overlap", 0.0),
            "outer_clearance": config.get("metaball_outer_clearance", 0.0),
            "geometry_parameters": 9,
            "training_samples": "one deterministic PCG64 geometry per epoch token",
            "causal_sorting": config.get("causal_sorting", "fixed_order"),
            "zero_geometry_mapping": {
                "token": ZERO_GEOMETRY_TOKEN,
                "condition_parameters": [0.0] * 9,
                "obstacles": "none",
            },
        },
        "method": (
            f"native CUDA {jet_label} + strict canonical SIREN + normalized raw 11D"
            if bool(config.get("canonical_siren", False))
            else f"native CUDA {jet_label} + spatial-SIREN + raw xy and circle parameters"
        ),
        "siren_first_layer_spatial_omega0": config[
            "siren_first_layer_spatial_omega0"
        ],
        "canonical_siren": bool(config.get("canonical_siren", False)),
        "siren_first_omega0": config.get("siren_first_omega0"),
        "siren_hidden_omega0": config.get("siren_hidden_omega0"),
        "coordinate_normalization": config.get(
            "coordinate_normalization", "centered_unit"
        ),
        "production_backend": str(BACKEND_PATH.resolve()),
        "production_backend_sha256": sha256(BACKEND_PATH),
        "precision": {key: config[key] for key in FP32_KEYS},
        "case_hooks": [
            (
                "IID PCG64 rejection sampler over physical fluid area"
                if config.get("sampling_mode") == "uniform_random"
                else "uniform physical-fluid-area scrambled Sobol sampler"
            ),
            "C-infinity single-zero slip streamfunction envelope",
            "11D raw xy plus three-circle encoding",
            "FP32 master-to-batched-forward synchronization",
        ],
        "unchanged_production_paths": [
            (
                "native CUDA value-gradient Jet1 layout"
                if jet_order == 1
                else "compact symmetric native CUDA Jet3 layout"
            ),
            f"{config.get('causal_sorting', 'fixed_order')} residual peeling",
            str(config.get("basis_objective", "pure_vorticity_rayleigh")),
            f"handwritten {jet_label} parameter reverse",
            "Adam, scheduler, and checkpointing",
        ],
    }


def write_provenance(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance(config), indent=2), encoding="utf-8")
