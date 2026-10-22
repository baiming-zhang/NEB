














from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import scipy.sparse


@dataclass(frozen=True)
class MaskedCartesianComplex:


    x: np.ndarray
    y: np.ndarray
    solid: np.ndarray
    unknown: np.ndarray
    node_index: np.ndarray
    gradient_x: scipy.sparse.csr_matrix
    gradient_y: scipy.sparse.csr_matrix
    average_x: scipy.sparse.csr_matrix
    average_y: scipy.sparse.csr_matrix
    cell_divergence: scipy.sparse.csr_matrix
    laplacian: scipy.sparse.csr_matrix
    weights: np.ndarray
    outlet_rows: np.ndarray
    outlet_indices: np.ndarray
    outlet_weights: np.ndarray
    x_edge_flat_nodes: np.ndarray
    y_edge_flat_nodes: np.ndarray
    dx: float
    dy: float

    @property
    def node_count(self) -> int:
        return int(np.count_nonzero(self.unknown))


def _incidence_and_average(
    pairs: list[tuple[int, int]],
    node_index_flat: np.ndarray,
    inverse_h: float,
    node_count: int,
) -> tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray]:


    gradient_rows: list[int] = []
    gradient_columns: list[int] = []
    gradient_data: list[float] = []
    active_endpoints: list[tuple[int, ...]] = []
    retained_pairs: list[tuple[int, int]] = []
    for first_flat, second_flat in pairs:
        first = int(node_index_flat[first_flat])
        second = int(node_index_flat[second_flat])
        if first < 0 and second < 0:
            continue
        edge = len(retained_pairs)
        if first >= 0:
            gradient_rows.append(edge)
            gradient_columns.append(first)
            gradient_data.append(-inverse_h)
        if second >= 0:
            gradient_rows.append(edge)
            gradient_columns.append(second)
            gradient_data.append(inverse_h)
        active_endpoints.append(tuple(index for index in (first, second) if index >= 0))
        retained_pairs.append((first_flat, second_flat))

    gradient = scipy.sparse.csr_matrix(
        (np.asarray(gradient_data, dtype=np.float64), (gradient_rows, gradient_columns)),
        shape=(len(retained_pairs), node_count),
        dtype=np.float64,
    )
    incidence_count = np.zeros(node_count, dtype=np.int32)
    for endpoints in active_endpoints:
        for node in endpoints:
            incidence_count[node] += 1
    if np.any(incidence_count == 0):
        raise RuntimeError("an active scalar node has no incident edge")
    average_rows: list[int] = []
    average_columns: list[int] = []
    average_data: list[float] = []
    for edge, endpoints in enumerate(active_endpoints):
        for node in endpoints:
            average_rows.append(node)
            average_columns.append(edge)
            average_data.append(1.0 / float(incidence_count[node]))
    average = scipy.sparse.csr_matrix(
        (np.asarray(average_data, dtype=np.float64), (average_rows, average_columns)),
        shape=(node_count, len(retained_pairs)),
        dtype=np.float64,
    )
    return gradient, average, np.asarray(retained_pairs, dtype=np.int64)


def build_masked_cartesian_complex(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
) -> MaskedCartesianComplex:


    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    solid = np.asarray(solid, dtype=bool)
    if solid.shape != (len(y), len(x)):
        raise ValueError(f"solid mask shape {solid.shape} does not match {(len(y), len(x))}")
    if len(x) < 3 or len(y) < 3:
        raise ValueError("the mimetic complex requires at least a 3x3 grid")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if not np.allclose(np.diff(x), dx, rtol=5.0e-5, atol=5.0e-7):
        raise ValueError("x grid is not uniform")
    if not np.allclose(np.diff(y), dy, rtol=5.0e-5, atol=5.0e-7):
        raise ValueError("y grid is not uniform")
    if not math.isclose(dx, dy, rel_tol=1.0e-5, abs_tol=1.0e-7):
        raise ValueError("the current Cartesian Hodge star requires dx=dy")

    unknown = (~solid).copy()
    unknown[:, 0] = False
    unknown[0, :] = False
    unknown[-1, :] = False


    node_index = np.full(unknown.shape, -1, dtype=np.int32)
    node_index[unknown] = np.arange(np.count_nonzero(unknown), dtype=np.int32)
    node_count = int(np.count_nonzero(unknown))
    flat = np.arange(unknown.size, dtype=np.int64).reshape(unknown.shape)
    x_pairs = list(zip(flat[:, :-1].ravel().tolist(), flat[:, 1:].ravel().tolist()))
    y_pairs = list(zip(flat[:-1, :].ravel().tolist(), flat[1:, :].ravel().tolist()))
    gx, hx, x_edge_nodes = _incidence_and_average(
        x_pairs, node_index.ravel(), 1.0 / dx, node_count
    )
    gy, hy, y_edge_nodes = _incidence_and_average(
        y_pairs, node_index.ravel(), 1.0 / dy, node_count
    )
    laplacian = (gx.T @ gx + gy.T @ gy).tocsr().astype(np.float64)

    x_edge_lookup = {tuple(pair): edge for edge, pair in enumerate(x_edge_nodes.tolist())}
    y_edge_lookup = {tuple(pair): edge for edge, pair in enumerate(y_edge_nodes.tolist())}
    divergence_rows: list[int] = []
    divergence_columns: list[int] = []
    divergence_data: list[float] = []
    active_cell = 0
    for iy in range(len(y) - 1):
        for ix in range(len(x) - 1):
            bottom = (int(flat[iy, ix]), int(flat[iy, ix + 1]))
            top = (int(flat[iy + 1, ix]), int(flat[iy + 1, ix + 1]))
            left = (int(flat[iy, ix]), int(flat[iy + 1, ix]))
            right = (int(flat[iy, ix + 1]), int(flat[iy + 1, ix + 1]))
            if not (
                bottom in x_edge_lookup
                and top in x_edge_lookup
                and left in y_edge_lookup
                and right in y_edge_lookup
            ):
                continue

            for column, value in (
                (x_edge_lookup[bottom], -1.0 / dy),
                (x_edge_lookup[top], 1.0 / dy),
                (len(x_edge_nodes) + y_edge_lookup[left], -1.0 / dx),
                (len(x_edge_nodes) + y_edge_lookup[right], 1.0 / dx),
            ):
                divergence_rows.append(active_cell)
                divergence_columns.append(column)
                divergence_data.append(value)
            active_cell += 1
    cell_divergence = scipy.sparse.csr_matrix(
        (
            np.asarray(divergence_data, dtype=np.float64),
            (divergence_rows, divergence_columns),
        ),
        shape=(active_cell, len(x_edge_nodes) + len(y_edge_nodes)),
        dtype=np.float64,
    )

    x_weights = np.ones(len(x), dtype=np.float64)
    y_weights = np.ones(len(y), dtype=np.float64)
    x_weights[[0, -1]] = 0.5
    y_weights[[0, -1]] = 0.5
    weights = (dx * dy * np.multiply.outer(y_weights, x_weights))[unknown].astype(np.float32)
    outlet_rows = np.flatnonzero(unknown[:, -1]).astype(np.int64)
    outlet_indices = node_index[outlet_rows, -1].astype(np.int64)
    outlet_weights = (dy * y_weights[outlet_rows]).astype(np.float32)
    return MaskedCartesianComplex(
        x=x,
        y=y,
        solid=solid,
        unknown=unknown,
        node_index=node_index,
        gradient_x=gx,
        gradient_y=gy,
        average_x=hx,
        average_y=hy,
        cell_divergence=cell_divergence,
        laplacian=laplacian,
        weights=weights,
        outlet_rows=outlet_rows,
        outlet_indices=outlet_indices,
        outlet_weights=outlet_weights,
        x_edge_flat_nodes=x_edge_nodes,
        y_edge_flat_nodes=y_edge_nodes,
        dx=dx,
        dy=dy,
    )


def evaluate_streamfunction_block(
    complex_: MaskedCartesianComplex,
    psi_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:


    psi_grid = np.asarray(psi_grid, dtype=np.float32)
    if psi_grid.ndim == 2:
        psi_grid = psi_grid[..., None]
    if psi_grid.shape[:2] != complex_.unknown.shape:
        raise ValueError(
            f"psi block grid {psi_grid.shape[:2]} does not match {complex_.unknown.shape}"
        )



    values = np.asarray(psi_grid[complex_.unknown], dtype=np.float64)
    grad_x = complex_.gradient_x @ values
    grad_y = complex_.gradient_y @ values


    u64 = np.asarray(complex_.average_y @ grad_y, dtype=np.float64)
    v64 = np.asarray(-(complex_.average_x @ grad_x), dtype=np.float64)
    omega64 = np.asarray(complex_.laplacian @ values, dtype=np.float64)
    weak_curl = complex_.gradient_y.T @ grad_y + complex_.gradient_x.T @ grad_x
    curl_error = float(np.max(np.abs(weak_curl - omega64)))
    curl_scale = max(float(np.max(np.abs(omega64))), 1.0e-30)

    edge_velocity = np.concatenate((-grad_x, grad_y), axis=0)
    divergence = complex_.cell_divergence @ edge_velocity
    scale = max(float(np.max(np.abs(grad_x))), float(np.max(np.abs(grad_y))), 1.0e-30)
    scaled_divergence = float(
        np.max(np.abs(divergence)) * max(complex_.dx, complex_.dy) / scale
    )
    return u64.astype(np.float32), v64.astype(np.float32), omega64.astype(np.float32), {
        "weak_curl_max_abs": curl_error,
        "weak_curl_relative_max_abs": curl_error / curl_scale,
        "rotated_gradient_divergence_max_abs": float(np.max(np.abs(divergence))),
        "rotated_gradient_divergence_scaled_max_abs": scaled_divergence,
    }


def weak_curl_from_grid_velocity(
    complex_: MaskedCartesianComplex,
    u_grid: np.ndarray,
    v_grid: np.ndarray,
) -> np.ndarray:


    u_flat = np.asarray(u_grid, dtype=np.float32).ravel()
    v_flat = np.asarray(v_grid, dtype=np.float32).ravel()
    if u_flat.size != complex_.unknown.size or v_flat.size != complex_.unknown.size:
        raise ValueError("mean velocity grids do not match the mimetic complex")
    y_edges = complex_.y_edge_flat_nodes
    x_edges = complex_.x_edge_flat_nodes
    u_edge = np.float32(0.5) * (u_flat[y_edges[:, 0]] + u_flat[y_edges[:, 1]])
    v_edge = np.float32(0.5) * (v_flat[x_edges[:, 0]] + v_flat[x_edges[:, 1]])
    return np.asarray(
        complex_.gradient_y.T @ u_edge - complex_.gradient_x.T @ v_edge,
        dtype=np.float32,
    )


def factorization_max_abs(complex_: MaskedCartesianComplex) -> float:


    difference = complex_.laplacian - (
        complex_.gradient_x.T @ complex_.gradient_x
        + complex_.gradient_y.T @ complex_.gradient_y
    )
    if difference.nnz == 0:
        return 0.0
    return float(np.max(np.abs(difference.data)))


def diagnostics(complex_: MaskedCartesianComplex) -> dict[str, Any]:
    return {
        "formulation": "L=Gx^T Gx+Gy^T Gy; edge velocity=rot(G psi); omega=L psi",
        "grid_shape": [int(len(complex_.y)), int(len(complex_.x))],
        "active_scalar_nodes": complex_.node_count,
        "x_edges": int(complex_.gradient_x.shape[0]),
        "y_edges": int(complex_.gradient_y.shape[0]),
        "active_cells_with_complete_boundary": int(complex_.cell_divergence.shape[0]),
        "spacing": float(complex_.dx),
        "laplacian_incidence_factorization_max_abs": factorization_max_abs(complex_),
        "dirichlet_boundaries": ["airfoil", "inlet", "ymin", "ymax"],
        "natural_boundaries": ["outlet"],
        "cache_operator_metric": "uniform lumped h^2 scalar mass used by archived eigensolve",
        "nonlinear_quadrature": "native active-node tensor trapezoidal weights",
        "collocation": "arithmetic edge-to-active-node averaging for nonlinear quadrature",
    }
