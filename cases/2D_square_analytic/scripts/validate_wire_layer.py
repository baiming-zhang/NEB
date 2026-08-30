"""Validate the real-Gabor WIRE Jet3 forward and handwritten transpose."""

from __future__ import annotations

import json
import math

import cupy as cp
import numpy as np

from square_runtime import backend, make_wire_activation


def derivative(
    x: np.ndarray,
    y: np.ndarray,
    order_x: int,
    order_y: int,
    omega: float,
    sigma: float,
    coefficients: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Closed-form mixed derivative for affine ``u(x,y)`` and ``v(x,y)``."""
    ax, ay, c, bx, by, d = coefficients
    u = ax * x + ay * y + c
    v = bx * x + by * y + d
    frequency = [
        np.cos(omega * u),
        -omega * np.sin(omega * u),
        -(omega**2) * np.cos(omega * u),
        omega**3 * np.sin(omega * u),
    ]
    a = sigma * sigma
    gaussian = np.exp(-a * v * v)
    scale = [
        gaussian,
        -2.0 * a * v * gaussian,
        (4.0 * a * a * v * v - 2.0 * a) * gaussian,
        (12.0 * a * a * v - 8.0 * a**3 * v**3) * gaussian,
    ]
    result = np.zeros_like(x)
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            first_order = i + j
            second_order = order_x + order_y - first_order
            result += (
                math.comb(order_x, i)
                * math.comb(order_y, j)
                * frequency[first_order]
                * ax**i
                * ay**j
                * scale[second_order]
                * bx ** (order_x - i)
                * by ** (order_y - j)
            )
    return result


def affine_jet(
    x: np.ndarray, y: np.ndarray, cx: float, cy: float, bias: float
) -> backend.Jet3:
    count = len(x)
    return backend.Jet3(
        cp.asarray((cx * x + cy * y + bias)[:, None], dtype=cp.float64),
        cp.asarray(np.tile([cx, cy], (count, 1))[:, None, :], dtype=cp.float64),
        cp.zeros((count, 1, 3), dtype=cp.float64),
        cp.zeros((count, 1, 4), dtype=cp.float64),
    )


def inner(left: backend.Jet3, right: backend.Jet3) -> float:
    value = cp.sum(left.value * right.value)
    value += cp.sum(left.grad * right.grad)
    value += cp.sum(left.hess * right.hess)
    value += cp.sum(left.third * right.third)
    return float(value.get())


def shifted(source: backend.Jet3, direction: backend.Jet3, scale: float) -> backend.Jet3:
    return backend.Jet3(
        source.value + scale * direction.value,
        source.grad + scale * direction.grad,
        source.hess + scale * direction.hess,
        source.third + scale * direction.third,
    )


def random_jet(rng: np.random.Generator, shape: tuple[int, int]) -> backend.Jet3:
    count, width = shape
    return backend.Jet3(
        cp.asarray(rng.normal(size=shape), dtype=cp.float64),
        cp.asarray(rng.normal(size=(count, width, 2)), dtype=cp.float64),
        cp.asarray(rng.normal(size=(count, width, 3)), dtype=cp.float64),
        cp.asarray(rng.normal(size=(count, width, 4)), dtype=cp.float64),
    )


def main() -> None:
    omega, sigma = 10.0, 2.5
    forward, backward = make_wire_activation(
        {"wire_omega0": omega, "wire_sigma0": sigma}
    )
    x = np.asarray([-0.8, -0.15, 0.3, 0.9])
    y = np.asarray([0.4, -0.7, 0.1, 0.6])
    coefficients = (0.7, -0.2, 0.1, -0.3, 0.5, -0.05)
    frequency = affine_jet(x, y, *coefficients[:3])
    scale = affine_jet(x, y, *coefficients[3:])
    source = backend.Jet3(
        cp.concatenate((frequency.value, scale.value), axis=-1),
        cp.concatenate((frequency.grad, scale.grad), axis=-2),
        cp.concatenate((frequency.hess, scale.hess), axis=-2),
        cp.concatenate((frequency.third, scale.third), axis=-2),
    )
    actual = forward(source)
    expected = {
        "value": derivative(x, y, 0, 0, omega, sigma, coefficients),
        "x": derivative(x, y, 1, 0, omega, sigma, coefficients),
        "y": derivative(x, y, 0, 1, omega, sigma, coefficients),
        "xx": derivative(x, y, 2, 0, omega, sigma, coefficients),
        "xy": derivative(x, y, 1, 1, omega, sigma, coefficients),
        "yy": derivative(x, y, 0, 2, omega, sigma, coefficients),
        "xxx": derivative(x, y, 3, 0, omega, sigma, coefficients),
        "xxy": derivative(x, y, 2, 1, omega, sigma, coefficients),
        "xyy": derivative(x, y, 1, 2, omega, sigma, coefficients),
        "yyy": derivative(x, y, 0, 3, omega, sigma, coefficients),
    }
    observed = {
        "value": cp.asnumpy(actual.value[:, 0]),
        "x": cp.asnumpy(actual.grad[:, 0, 0]),
        "y": cp.asnumpy(actual.grad[:, 0, 1]),
        "xx": cp.asnumpy(actual.hess[:, 0, 0]),
        "xy": cp.asnumpy(actual.hess[:, 0, 1]),
        "yy": cp.asnumpy(actual.hess[:, 0, 2]),
        "xxx": cp.asnumpy(actual.third[:, 0, 0]),
        "xxy": cp.asnumpy(actual.third[:, 0, 1]),
        "xyy": cp.asnumpy(actual.third[:, 0, 2]),
        "yyy": cp.asnumpy(actual.third[:, 0, 3]),
    }
    maximum_error = 0.0
    for name in expected:
        np.testing.assert_allclose(observed[name], expected[name], rtol=2.0e-11, atol=2.0e-11)
        maximum_error = max(maximum_error, float(np.max(np.abs(observed[name] - expected[name]))))

    # Directional transpose check covers arbitrary compact Jet3 storage. This
    # is validation only; training never uses numerical differentiation.
    rng = np.random.default_rng(20260819)
    source = random_jet(rng, (3, 8))
    direction = random_jet(rng, (3, 8))
    output_adjoint = random_jet(rng, (3, 4))
    source_adjoint = backward(source, output_adjoint)
    epsilon = 2.0e-6
    numerical = (
        inner(forward(shifted(source, direction, epsilon)), output_adjoint)
        - inner(forward(shifted(source, direction, -epsilon)), output_adjoint)
    ) / (2.0 * epsilon)
    analytic = inner(source_adjoint, direction)
    relative_transpose_error = abs(numerical - analytic) / max(1.0, abs(analytic))
    if relative_transpose_error > 2.0e-7:
        raise AssertionError(
            f"WIRE transpose mismatch: numerical={numerical}, analytic={analytic}"
        )
    print(json.dumps({
        "status": "pass",
        "wire_variant": "official_real_gabor",
        "maximum_forward_derivative_error": maximum_error,
        "relative_transpose_error": relative_transpose_error,
    }, indent=2))


if __name__ == "__main__":
    main()
