from pathlib import Path
import sys

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from generate_oscillatory_abc_gt import (  # noqa: E402
    abc_parameters,
    oscillatory_abc_velocity,
)


def periodic_derivative(values: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (
        np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
    ) / (2.0 * spacing)


def test_oscillatory_abc_is_periodic_divergence_free_beltrami():
    n = 32
    coordinates = np.linspace(-np.pi, np.pi, n, endpoint=False)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    velocity = oscillatory_abc_velocity(x, y, z, 1.234).astype(np.float64)
    spacing = 2.0 * np.pi / n
    divergence = sum(
        periodic_derivative(velocity[..., axis], axis, spacing)
        for axis in range(3)
    )
    curl = np.stack(
        (
            periodic_derivative(velocity[..., 2], 1, spacing)
            - periodic_derivative(velocity[..., 1], 2, spacing),
            periodic_derivative(velocity[..., 0], 2, spacing)
            - periodic_derivative(velocity[..., 2], 0, spacing),
            periodic_derivative(velocity[..., 1], 0, spacing)
            - periodic_derivative(velocity[..., 0], 1, spacing),
        ),
        axis=-1,
    )
    assert np.sqrt(np.mean(divergence**2)) < 1.0e-6
    assert np.linalg.norm(curl - velocity) / np.linalg.norm(velocity) < 7.0e-3


def test_oscillatory_parameters_close_after_one_period():
    coefficients0, phases0 = abc_parameters(0.0)
    coefficients1, phases1 = abc_parameters(2.0 * np.pi)
    np.testing.assert_allclose(coefficients0, coefficients1, atol=1.0e-12)
    np.testing.assert_allclose(phases0, phases1, atol=1.0e-12)
