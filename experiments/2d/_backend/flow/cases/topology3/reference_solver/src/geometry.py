"""Analytic circle/square/triangle obstacle family for topology ablations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

WINDOW = (0.0, 10.0, -4.0, 4.0)
OUTER_CENTER = np.array([5.0, 0.0], dtype=np.float64)
OUTER_HALF_SIZE = 4.0
OUTER_POWER_SQUARE = 12.0
OUTER_POWER_CIRCLE = 2.0


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    kind: str
    x: float
    y: float
    size: float
    rotation: float = 0.0
    power: int = 10
    smooth_k: float = 20.0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def anchor(self) -> np.ndarray:
        return self.center

    def as_dict(self) -> dict:
        value = asdict(self)
        for key in ("x", "y", "size", "rotation", "smooth_k"):
            value[key] = float(value[key])
        return value


CASE_REGISTRY: dict[str, tuple[ObstacleSpec, ...]] = {
    "one_circle": (
        ObstacleSpec("circle", "circle", 4.28, 0.62, 0.82),
    ),
    "circle_square": (
        ObstacleSpec("circle", "circle", 3.35, 0.45, 0.70),
        ObstacleSpec("square", "square", 6.35, -0.25, 0.72, rotation=0.12, power=10),
    ),
    "circle_square_triangle": (
        ObstacleSpec("circle", "circle", 2.85, 0.95, 0.64),
        ObstacleSpec("square", "square", 6.35, 0.85, 0.68, rotation=0.10, power=10),
        ObstacleSpec("triangle", "triangle", 4.65, -1.65, 0.48, rotation=-0.05, smooth_k=20.0),
    ),
}


def get_obstacles(case_name: str) -> tuple[ObstacleSpec, ...]:
    try:
        return CASE_REGISTRY[case_name]
    except KeyError as exc:
        raise ValueError(f"unknown subcase {case_name!r}; choose {sorted(CASE_REGISTRY)}") from exc


def outer_power(tau: float) -> float:
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0,1]")
    return OUTER_POWER_SQUARE + tau * (OUTER_POWER_CIRCLE - OUTER_POWER_SQUARE)


def outer_implicit(points: np.ndarray, tau: float = 0.0) -> np.ndarray:
    power = outer_power(tau)
    q = (np.asarray(points, dtype=np.float64) - OUTER_CENTER) / OUTER_HALF_SIZE
    return np.abs(q[..., 0])**power + np.abs(q[..., 1])**power - 1.0


def outer_gradient(points: np.ndarray, tau: float = 0.0) -> np.ndarray:
    power = outer_power(tau)
    q = (np.asarray(points, dtype=np.float64) - OUTER_CENTER) / OUTER_HALF_SIZE
    return (power / OUTER_HALF_SIZE) * np.sign(q) * np.abs(q)**(power - 1.0)


def outer_boundary(tau: float, count: int) -> np.ndarray:
    power = outer_power(tau)
    theta = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    c, s = np.cos(theta), np.sin(theta)
    radius = OUTER_HALF_SIZE * (np.abs(c)**power + np.abs(s)**power)**(-1.0 / power)
    return OUTER_CENTER + radius[:, None] * np.stack((c, s), axis=1)


def _to_local(points: np.ndarray, spec: ObstacleSpec) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - spec.center
    c, s = np.cos(spec.rotation), np.sin(spec.rotation)
    return np.stack((c * delta[..., 0] + s * delta[..., 1],
                     -s * delta[..., 0] + c * delta[..., 1]), axis=-1)


def _gradient_to_global(local_gradient: np.ndarray, spec: ObstacleSpec) -> np.ndarray:
    c, s = np.cos(spec.rotation), np.sin(spec.rotation)
    return np.stack((c * local_gradient[..., 0] - s * local_gradient[..., 1],
                     s * local_gradient[..., 0] + c * local_gradient[..., 1]), axis=-1)


def obstacle_implicit(points: np.ndarray, spec: ObstacleSpec) -> np.ndarray:
    """Dimensionless smooth level function: negative inside, positive in fluid."""
    q = _to_local(points, spec)
    if spec.kind == "circle":
        return np.sum(q * q, axis=-1) / spec.size**2 - 1.0
    if spec.kind == "square":
        return (q[..., 0] / spec.size)**spec.power + (q[..., 1] / spec.size)**spec.power - 1.0
    if spec.kind == "triangle":
        angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(3) / 3.0
        normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        halfplanes = np.einsum("...d,jd->...j", q, normals) / spec.size - 1.0
        return logsumexp(spec.smooth_k * halfplanes, axis=-1) / spec.smooth_k
    raise ValueError(f"unsupported obstacle kind {spec.kind!r}")


def obstacle_gradient(points: np.ndarray, spec: ObstacleSpec) -> np.ndarray:
    q = _to_local(points, spec)
    if spec.kind == "circle":
        local = 2.0 * q / spec.size**2
    elif spec.kind == "square":
        local = (spec.power / spec.size) * np.stack(
            ((q[..., 0] / spec.size)**(spec.power - 1),
             (q[..., 1] / spec.size)**(spec.power - 1)), axis=-1)
    elif spec.kind == "triangle":
        angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(3) / 3.0
        normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        halfplanes = np.einsum("...d,jd->...j", q, normals) / spec.size - 1.0
        logits = spec.smooth_k * halfplanes
        logits -= np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        local = np.einsum("...j,jd->...d", weights, normals) / spec.size
    else:
        raise ValueError(f"unsupported obstacle kind {spec.kind!r}")
    return _gradient_to_global(local, spec)


def obstacle_boundary(spec: ObstacleSpec, count: int) -> np.ndarray:
    theta = np.arange(count, dtype=np.float64) * (2.0 * np.pi / count)
    direction = np.stack((np.cos(theta), np.sin(theta)), axis=1)
    if spec.kind == "circle":
        radius = np.full(count, spec.size)
    elif spec.kind == "square":
        c, s = np.cos(spec.rotation), np.sin(spec.rotation)
        local_direction = np.stack((c * direction[:, 0] + s * direction[:, 1],
                                    -s * direction[:, 0] + c * direction[:, 1]), axis=1)
        radius = spec.size * np.sum(np.abs(local_direction)**spec.power, axis=1)**(-1.0 / spec.power)
    else:
        radius = np.empty(count, dtype=np.float64)
        for index, ray in enumerate(direction):
            residual = lambda value: float(obstacle_implicit(spec.center + value * ray, spec))
            radius[index] = brentq(residual, 0.0, 4.0 * spec.size, xtol=1e-13, rtol=1e-13)
    return spec.center + radius[:, None] * direction


def fluid_mask(points: np.ndarray, case_name: str, tau: float = 0.0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    mask = outer_implicit(points, tau) <= 5e-13
    for spec in get_obstacles(case_name):
        mask &= obstacle_implicit(points, spec) >= -5e-13
    return mask


def simple_zero_coordinates(points: np.ndarray, case_name: str, tau: float = 0.0) -> np.ndarray:
    values = [-outer_implicit(points, tau)]
    values.extend(obstacle_implicit(points, spec) for spec in get_obstacles(case_name))
    return np.stack(values, axis=-1)


def simple_zero_envelope(points: np.ndarray, case_name: str, tau: float = 0.0) -> np.ndarray:
    q = simple_zero_coordinates(points, case_name, tau)
    return np.prod(q / np.sqrt(1.0 + q * q), axis=-1)


def simple_zero_envelope_gradient(points: np.ndarray, case_name: str, tau: float = 0.0) -> np.ndarray:
    q = simple_zero_coordinates(points, case_name, tau)
    gradients = [-outer_gradient(points, tau)]
    gradients.extend(obstacle_gradient(points, spec) for spec in get_obstacles(case_name))
    grad_q = np.stack(gradients, axis=-2)
    s = q / np.sqrt(1.0 + q * q)
    ds = (1.0 + q * q)**(-1.5)
    result = np.zeros(np.asarray(points).shape, dtype=np.float64)
    for index in range(q.shape[-1]):
        others = np.prod(np.delete(s, index, axis=-1), axis=-1)
        result += (ds[..., index] * others)[..., None] * grad_q[..., index, :]
    return result


def sample_random_fluid_points(case_name: str, count: int,
                               rng: np.random.Generator, tau: float = 0.0) -> tuple[np.ndarray, int]:
    accepted: list[np.ndarray] = []
    draws = 0
    while sum(len(chunk) for chunk in accepted) < count:
        batch = max(4096, 2 * (count - sum(len(chunk) for chunk in accepted)))
        candidate = np.column_stack((rng.uniform(1.0, 9.0, batch), rng.uniform(-4.0, 4.0, batch)))
        accepted.append(candidate[fluid_mask(candidate, case_name, tau)])
        draws += batch
    return np.vstack(accepted)[:count], draws


def _signed_area(loop: np.ndarray) -> float:
    shifted = np.roll(loop, -1, axis=0)
    return 0.5 * float(np.sum(loop[:, 0] * shifted[:, 1] - shifted[:, 0] * loop[:, 1]))


def geometry_diagnostics(case_name: str, tau: float = 0.0) -> dict:
    specs = get_obstacles(case_name)
    outer = outer_boundary(tau, 720)
    loops = [obstacle_boundary(spec, 512) for spec in specs]
    residuals = [float(np.max(np.abs(obstacle_implicit(loop, spec)))) for loop, spec in zip(loops, specs)]
    gradient_min = [float(np.min(np.linalg.norm(obstacle_gradient(loop, spec), axis=1)))
                    for loop, spec in zip(loops, specs)]
    pair_clearances = [float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)))
                       for i, a in enumerate(loops) for b in loops[i + 1:]]
    outer_clearances = [float(np.min(np.linalg.norm(loop[:, None] - outer[None], axis=2))) for loop in loops]
    return {
        "case_name": case_name,
        "obstacles": [spec.as_dict() for spec in specs],
        "obstacle_count": len(specs),
        "boundary_components": len(specs) + 1,
        "first_betti_number": len(specs),
        "outer_tau": float(tau),
        "outer_formula": "|q_x|^p+|q_y|^p=1, p(tau)=12-10*tau",
        "outer_signed_area": _signed_area(outer),
        "inner_signed_areas": [_signed_area(loop) for loop in loops],
        "boundary_residual_max": max(residuals),
        "boundary_gradient_min": min(gradient_min),
        "minimum_pair_clearance": min(pair_clearances) if pair_clearances else None,
        "minimum_outer_clearance": min(outer_clearances),
        "simple_zero": True,
    }
