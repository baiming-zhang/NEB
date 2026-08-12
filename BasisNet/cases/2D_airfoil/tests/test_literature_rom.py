from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from eigenfluid_sss.basisnet.airfoil2d.literature_rom import (  # noqa: E402
    fit_exact_dmd,
    fit_quadratic_opinf,
    symmetric_quadratic_features,
)


def test_exact_dmd_recovers_a_linear_map() -> None:
    operator = np.array([[0.92, -0.1], [0.1, 0.92]])
    states = [np.array([0.8, -0.2])]
    for _ in range(20):
        states.append(operator @ states[-1])
    fitted, diagnostics = fit_exact_dmd(np.asarray(states))
    np.testing.assert_allclose(fitted, operator, rtol=2.0e-13, atol=2.0e-13)
    assert diagnostics["snapshot_rank"] == 2


def test_symmetric_quadratic_feature_layout() -> None:
    features = symmetric_quadratic_features(np.array([[2.0, 3.0]]))
    np.testing.assert_array_equal(features, [[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])


def test_quadratic_opinf_uses_only_identifiable_order() -> None:
    rng = np.random.default_rng(4)
    states = rng.normal(size=(41, 8))
    coefficients, order, diagnostics = fit_quadratic_opinf(states, 0.1)
    assert order == 7
    assert diagnostics["feature_columns"] == 36
    assert diagnostics["feature_rank"] == 36
    assert coefficients.shape == (36, 7)
