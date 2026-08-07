from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_wake3d.training import (  # noqa: E402
    Config,
    OfficialPeelingFields,
    continuous_alpha,
    fluid_points,
    obstacle_sdf_numpy,
    peel_shape,
)


def small_config() -> Config:
    return Config(
        modes=3,
        domain_type="periodic_obstacle",
        hidden=8,
        hidden_layers=2,
        frequencies=2,
        points_per_step=64,
        validation_points=64,
        validation_alphas=3,
        difference_step=0.03,
        xmin=-np.pi,
        xmax=np.pi,
        ymin=-np.pi,
        ymax=np.pi,
        zmin=-np.pi,
        zmax=np.pi,
        obstacle_x=0.0,
        obstacle_y=0.0,
        obstacle_z=0.0,
        obstacle_radius=1.0,
        sampling_mode="mixed",
        global_fraction=0.5,
        obstacle_fraction=0.5,
        wake_fraction=0.0,
    )


def test_final_defaults() -> None:
    cfg = Config()
    assert cfg.modes == 256
    assert cfg.epochs == 10000
    assert cfg.domain_type == "periodic_obstacle"
    assert cfg.points_per_step == 4096
    assert cfg.sss_ordering == "dynamic"
    assert cfg.lr_scheduler == "reduce_on_plateau"


def test_sdf_endpoints_and_samples_are_fluid() -> None:
    cfg = small_config()
    sphere_axis = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    cube_corner = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    assert abs(float(obstacle_sdf_numpy(cfg, sphere_axis, 0.0)[0])) < 1e-6
    assert abs(float(obstacle_sdf_numpy(cfg, cube_corner, 1.0)[0])) < 1e-6
    points = fluid_points(cfg, 256, 0.37, 123)
    assert np.all(obstacle_sdf_numpy(cfg, points, 0.37) > 0.0)


def test_alpha_is_continuous_and_deterministic() -> None:
    cfg = small_config()
    values = [continuous_alpha(cfg, epoch) for epoch in range(1, 20)]
    assert values == [continuous_alpha(cfg, epoch) for epoch in range(1, 20)]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert len(set(values)) > 15


def test_hard_divergence_and_rank() -> None:
    cfg = small_config()
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 64, 0.43, 456)
    with torch.no_grad():
        _, diagnostic, _ = peel_shape(cfg, model, points, 0.43, None)
    assert diagnostic["max_discrete_divergence"] < 1e-8
    assert diagnostic["effective_raw_rank"] == cfg.modes


def test_published_best_result_report() -> None:
    report_path = ROOT / "reports" / "k256_re500_best_results.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(
        (ROOT / "configs" / "k256_re500_published.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "complete"
    assert report["training"]["selected_epoch"] == 5000
    assert report["training"]["validation_rank"] == 256
    assert config["modes"] == 256
    assert config["epochs"] == 5000
    errors = report["evaluation"]["frame_velocity_errors"]
    assert len(errors) == 30
    assert np.isfinite(errors).all()
    for artifact in report["evaluation"]["artifacts"].values():
        assert (report_path.parent / artifact).resolve().is_file()
