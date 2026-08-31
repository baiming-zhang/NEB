from __future__ import annotations

import numpy as np
import scipy.sparse

from eigenfluid.gen.airfoil2d.mimetic_fd import (
    build_masked_cartesian_complex,
    diagnostics,
    evaluate_streamfunction_block,
)


def _reference_five_point(unknown: np.ndarray, h: float) -> scipy.sparse.csr_matrix:
    index = np.full(unknown.shape, -1, dtype=np.int32)
    index[unknown] = np.arange(np.count_nonzero(unknown), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    inverse_h2 = 1.0 / (h * h)
    ny, nx = unknown.shape
    for iy, ix in np.argwhere(unknown):
        row = int(index[iy, ix])
        diagonal = 0.0
        for jy, jx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            diagonal += inverse_h2
            if unknown[jy, jx]:
                rows.append(row)
                columns.append(int(index[jy, jx]))
                data.append(-inverse_h2)
        rows.append(row)
        columns.append(row)
        data.append(diagonal)
    return scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(np.count_nonzero(unknown),) * 2
    )


def test_incidence_factorization_matches_archived_five_point_operator() -> None:
    x = np.linspace(0.0, 1.4, 8)
    y = np.linspace(-0.6, 0.6, 7)
    solid = np.zeros((len(y), len(x)), dtype=bool)
    solid[3, 3:5] = True
    complex_ = build_masked_cartesian_complex(x, y, solid)
    reference = _reference_five_point(complex_.unknown, complex_.dx)
    difference = complex_.laplacian.astype(np.float64) - reference
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 2.0e-5
    assert diagnostics(complex_)["laplacian_incidence_factorization_max_abs"] == 0.0


def test_rotated_gradient_weak_curl_and_cell_divergence_are_compatible() -> None:
    rng = np.random.default_rng(20260830)
    x = np.linspace(0.0, 1.4, 8)
    y = np.linspace(-0.6, 0.6, 7)
    solid = np.zeros((len(y), len(x)), dtype=bool)
    solid[3, 3:5] = True
    complex_ = build_masked_cartesian_complex(x, y, solid)
    psi = np.zeros((len(y), len(x), 5), dtype=np.float32)
    psi[complex_.unknown] = rng.normal(size=(complex_.node_count, 5)).astype(np.float32)
    u, v, omega, audit = evaluate_streamfunction_block(complex_, psi)
    assert u.shape == v.shape == omega.shape == (complex_.node_count, 5)
    assert audit["weak_curl_relative_max_abs"] < 2.0e-6
    assert audit["rotated_gradient_divergence_scaled_max_abs"] < 2.0e-6


def test_natural_outlet_has_no_exterior_east_edge() -> None:
    x = np.linspace(0.0, 1.0, 6)
    y = np.linspace(-0.4, 0.4, 5)
    solid = np.zeros((len(y), len(x)), dtype=bool)
    complex_ = build_masked_cartesian_complex(x, y, solid)
    assert len(complex_.outlet_rows) == len(y) - 2
    # Every outlet scalar node has only its west x-edge, whereas interior nodes
    # have both west and east x-edges.
    x_incidence = np.diff(complex_.average_x.indptr)
    assert np.all(x_incidence[complex_.outlet_indices] == 1)
