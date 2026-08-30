from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CASE_ROOT / "scripts" / "evaluate_unseen_vs_laplacian.py"
SPEC = importlib.util.spec_from_file_location("three_sphere_reconstruction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_dirichlet_laplacian_is_symmetric_positive_definite() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    matrix, index = MODULE.assemble_dirichlet_laplacian(mask, 0.25)
    assert matrix.shape == (27, 27)
    assert np.max(np.abs((matrix - matrix.T).data), initial=0.0) == 0.0
    assert np.linalg.eigvalsh(matrix.toarray())[0] > 0.0
    center = index[2, 2, 2]
    assert matrix.getrow(center).nnz == 7


def test_common_velocity_operator_and_antisymmetric_divergence() -> None:
    mask = np.zeros((7, 7, 7), dtype=bool)
    mask[1:6, 1:6, 1:6] = True
    z, y, x = np.meshgrid(
        np.linspace(0.0, 1.0, 7),
        np.linspace(0.0, 1.0, 7),
        np.linspace(0.0, 1.0, 7),
        indexing="ij",
    )
    potential = np.column_stack((y[mask], z[mask], x[mask]))
    gradient = MODULE.vector_potential_gradient_on_mask(potential, mask, 1.0 / 6.0)
    velocity = MODULE.velocity_from_potential_gradient(gradient)
    center = np.ravel_multi_index((3, 3, 3), mask.shape)
    valid_flat = np.flatnonzero(mask.ravel())
    center_local = int(np.flatnonzero(valid_flat == center)[0])
    np.testing.assert_allclose(velocity[center_local], [2.0, 0.0, 0.0], atol=1.0e-12)
