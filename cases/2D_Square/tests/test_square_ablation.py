import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_activation_ablation.py"
SPEC = importlib.util.spec_from_file_location("square_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_analytic_square_spectrum_and_normalization():
    points, weights = MODULE.evaluation_grid(33)
    pairs = MODULE.analytic_pairs(8)
    _, velocity, eigenvalues = MODULE.analytic_basis(points, pairs, weights)
    gram = np.einsum("n,nid,njd->ij", weights, velocity, velocity)
    expected = np.asarray([np.pi**2 * (m * m + n * n) for m, n in pairs])
    assert np.allclose(eigenvalues, expected)
    assert np.allclose(gram, np.eye(8), atol=1.0e-6)


def test_square_envelope_is_zero_on_every_wall():
    points = torch.tensor(
        [[0.0, 0.3], [1.0, 0.4], [0.2, 0.0], [0.7, 1.0], [0.5, 0.5]]
    )
    values = MODULE.square_envelope(points, torch.empty(0))[:, 0]
    assert torch.count_nonzero(values[:4]).item() == 0
    assert values[-1] > 0
