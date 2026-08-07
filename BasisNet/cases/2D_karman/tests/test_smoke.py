import json
from pathlib import Path

import numpy as np
import torch

from eigenfluid_sss.basisnet.karman2d.postprocess import generalized_symmetric_eig
from eigenfluid_sss.basisnet.karman2d.training import (
    Config,
    OfficialPeelingFields,
    fluid_points,
)


def test_final_defaults():
    cfg = Config()
    assert cfg.modes == 96
    assert cfg.envelope_mode == "localized"
    assert cfg.sampling_mode == "mixed"
    assert cfg.learning_rate == 8.0e-4


def test_forward_shape_and_sampling():
    cfg = Config(modes=3, hidden=8, hidden_layers=2, frequencies=2)
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 16, 0.35, cfg.seed)
    xy = torch.tensor(points)
    alpha = torch.full((len(points), 1), 0.35)
    assert model(xy, alpha).shape == (16, 3)
    assert np.all((points[:, 0] >= 0) & (points[:, 0] <= 10))


def test_generalized_eigenproblem():
    mass = np.diag([1.0, 2.0, 3.0])
    stiffness = np.diag([3.0, 4.0, 5.0])
    values, vectors = generalized_symmetric_eig(stiffness, mass, 3)
    assert np.all(np.diff(values) >= 0)
    assert np.allclose(vectors.T @ mass @ vectors, np.eye(3), atol=1.0e-10)


def test_published_best_result_report():
    case_root = Path(__file__).resolve().parents[1]
    report_path = case_root / "reports" / "k96_best_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["training"]["selected_epoch"] == 4950
    assert report["training"]["validation_rank"] == 96
    cases = report["evaluation"]["cases"]
    assert len(cases) == 2
    assert all(np.isfinite(case["velocity_error"]) for case in cases)
    for artifact in report["evaluation"]["artifacts"].values():
        assert (report_path.parent / artifact).resolve().is_file()
