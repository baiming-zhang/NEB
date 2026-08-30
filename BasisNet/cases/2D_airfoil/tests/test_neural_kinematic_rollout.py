from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from rollout_neural_kinematic_baseline import project_to_domain  # noqa: E402


def _body_to_world(points: np.ndarray, alpha_deg: float) -> np.ndarray:
    angle = math.radians(-alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    shifted = points - np.array([0.5, 0.0])
    return np.column_stack(
        (
            c * shifted[:, 0] - s * shifted[:, 1] + 3.0,
            s * shifted[:, 0] + c * shifted[:, 1],
        )
    )


def _world_to_body(points: np.ndarray, alpha_deg: float) -> np.ndarray:
    angle = math.radians(-alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    shifted = points - np.array([3.0, 0.0])
    return np.column_stack(
        (
            c * shifted[:, 0] + s * shifted[:, 1] + 0.5,
            -s * shifted[:, 0] + c * shifted[:, 1],
        )
    )


def test_projection_maps_airfoil_interior_to_ellipse_boundary() -> None:
    rng = np.random.default_rng(7)
    angle = rng.uniform(0.0, 2.0 * np.pi, 1000)
    radius = np.sqrt(rng.uniform(0.0, 0.99, 1000))
    semi_thickness = 0.13
    body = np.column_stack(
        (
            1.0 + radius * np.cos(angle),
            semi_thickness * radius * np.sin(angle),
        )
    )
    world = _body_to_world(body, 10.0)
    projected = project_to_domain(world, beta=0.5, alpha_deg=10.0)
    projected_body = _world_to_body(projected, 10.0)
    level = (projected_body[:, 0] - 1.0) ** 2 + (
        projected_body[:, 1] / semi_thickness
    ) ** 2
    np.testing.assert_allclose(level, 1.0, rtol=0.0, atol=1.0e-10)


def test_projection_is_identity_for_fluid_points() -> None:
    points = np.array([[0.1, -3.5], [8.0, 2.0], [4.5, -1.0]], dtype=np.float64)
    np.testing.assert_allclose(
        project_to_domain(points, 0.5, 10.0), points, rtol=0.0, atol=2.0e-15
    )
