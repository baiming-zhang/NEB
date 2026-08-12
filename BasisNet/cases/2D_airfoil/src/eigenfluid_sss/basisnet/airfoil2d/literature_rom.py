"""Numerically ranked literature ROM building blocks for airfoil rollouts."""

from __future__ import annotations

import numpy as np


def backward_error_rank(values: np.ndarray, rows: int, columns: int) -> int:
    """Float64 rank using the standard spectral backward-error bound."""

    singular = np.asarray(values, dtype=np.float64)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    tolerance = np.finfo(np.float64).eps * max(rows, columns) * singular[0]
    return int(np.count_nonzero(singular > tolerance))


def symmetric_quadratic_features(states: np.ndarray) -> np.ndarray:
    """Return ``[1, a, vech(a a^T)]`` row features."""

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("states must be a 2-D (samples, reduced_dimension) array")
    samples, dimension = states.shape
    rows, columns = np.triu_indices(dimension)
    quadratic = states[:, rows] * states[:, columns]
    return np.column_stack((np.ones(samples), states, quadratic))


def fit_exact_dmd(states: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit the projected exact-DMD one-step map with no rank tuning."""

    states = np.asarray(states, dtype=np.float64)
    left, right = states[:-1].T, states[1:].T
    u, singular, vh = np.linalg.svd(left, full_matrices=False)
    rank = backward_error_rank(singular, *left.shape)
    if rank == 0:
        raise RuntimeError("DMD snapshot matrix has numerical rank zero")
    operator = right @ (vh[:rank].T / singular[:rank]) @ u[:, :rank].T
    residual = right - operator @ left
    return operator, {
        "snapshot_rank": rank,
        "snapshot_columns": left.shape[1],
        "relative_training_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(right), 1.0e-30)
        ),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(operator)))),
    }


def select_quadratic_opinf_order(states: np.ndarray) -> tuple[int, dict[str, int]]:
    """Largest POD prefix whose quadratic library is independently resolved.

    Central differences leave ``m-2`` derivative samples.  A candidate order
    is accepted only when every constant, linear, and unique quadratic column
    is resolved under the float64 backward-error rank bound.
    """

    states = np.asarray(states, dtype=np.float64)
    samples, available = states.shape
    if samples < 4:
        raise ValueError("quadratic Operator Inference needs at least four snapshots")
    derivative_samples = samples - 2
    for order in range(available, 0, -1):
        features = symmetric_quadratic_features(states[1:-1, :order])
        if features.shape[1] > derivative_samples:
            continue
        singular = np.linalg.svd(features, compute_uv=False)
        rank = backward_error_rank(singular, *features.shape)
        if rank == features.shape[1]:
            return order, {
                "derivative_samples": derivative_samples,
                "feature_columns": features.shape[1],
                "feature_rank": rank,
            }
    raise RuntimeError("no independently resolved quadratic Operator Inference order")


def fit_quadratic_opinf(
    states: np.ndarray,
    sample_dt: float,
    regularization_linear: float = 0.0,
    regularization_quadratic: float = 0.0,
) -> tuple[np.ndarray, int, dict[str, float | int]]:
    """Fit continuous quadratic Operator Inference by float64 least squares."""

    states = np.asarray(states, dtype=np.float64)
    order, diagnostics = select_quadratic_opinf_order(states)
    reduced = states[:, :order]
    derivatives = (reduced[2:] - reduced[:-2]) / (2.0 * sample_dt)
    features = symmetric_quadratic_features(reduced[1:-1])
    penalties = np.empty(features.shape[1], dtype=np.float64)
    penalties[: 1 + order] = regularization_linear
    penalties[1 + order :] = regularization_quadratic
    if np.any(penalties > 0.0):
        augmented_features = np.vstack((features, np.diag(np.sqrt(penalties))))
        augmented_derivatives = np.vstack(
            (derivatives, np.zeros((features.shape[1], order), dtype=np.float64))
        )
    else:
        augmented_features, augmented_derivatives = features, derivatives
    coefficients, residuals, rank, singular = np.linalg.lstsq(
        augmented_features, augmented_derivatives, rcond=None
    )
    predicted = features @ coefficients
    diagnostics.update(
        {
            "order": order,
            "least_squares_rank": int(rank),
            "relative_training_residual": float(
                np.linalg.norm(predicted - derivatives)
                / max(np.linalg.norm(derivatives), 1.0e-30)
            ),
            "feature_condition_number": float(singular[0] / singular[-1]),
            "residual_sum_squares": float(np.sum(residuals)) if residuals.size else 0.0,
            "regularization_linear": regularization_linear,
            "regularization_quadratic": regularization_quadratic,
        }
    )
    return coefficients, order, diagnostics


def evaluate_quadratic_opinf(coefficients: np.ndarray, state: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        feature = symmetric_quadratic_features(np.asarray(state)[None, :])[0]
        return feature @ coefficients
