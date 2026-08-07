import json
from pathlib import Path
import sys

import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.periodic3d.training import (  # noqa: E402
    Config,
    OfficialPeelingFields,
    discrete_all_mode_fields,
    fluid_points,
)


def test_final_defaults():
    cfg = Config()
    assert cfg.modes == 256
    assert cfg.domain_type == "periodic_closed"
    assert cfg.sss_ordering == "dynamic"
    assert cfg.points_per_step == 4096
    assert cfg.lr_scheduler == "reduce_on_plateau"


def test_periodic_sampling_and_forward_shape():
    cfg = Config(modes=3, hidden=8, hidden_layers=2, frequencies=2)
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 16, 0.0, cfg.seed)
    xyz = torch.tensor(points)
    alpha = torch.zeros((len(points), 1))
    assert model(xyz, alpha).shape == (16, 3, 3)
    assert np.all(points >= -np.pi)
    assert np.all(points <= np.pi)


def test_antisymmetric_tensor_is_hard_discrete_divergence_free():
    cfg = Config(modes=3, hidden=8, hidden_layers=2, frequencies=2)
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 8, 0.0, cfg.seed + 1)
    with torch.inference_mode():
        fields = discrete_all_mode_fields(cfg, model, points, 0.0)
    assert torch.count_nonzero(fields["divergence"]).item() == 0


def test_published_best_result_report():
    report_path = CASE_ROOT / "reports" / "k256_best_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "complete"
    assert report["training"]["selected_epoch"] == 10000
    assert report["training"]["validation_rank"] == 256
    benchmarks = report["evaluation"]["benchmarks"]
    assert len(benchmarks) == 2
    assert all(np.isfinite(item["mean_velocity_error"]) for item in benchmarks)
    for artifact in report["evaluation"]["artifacts"].values():
        assert (report_path.parent / artifact).resolve().is_file()
