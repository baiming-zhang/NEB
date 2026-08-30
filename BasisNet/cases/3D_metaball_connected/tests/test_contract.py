from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = CASE_ROOT / "scripts" / "train_validate.py"
SPEC = importlib.util.spec_from_file_location("three_sphere_training", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config():
    path = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
    return MODULE.Config(**json.loads(path.read_text(encoding="utf-8")))


def test_connected_geometry_and_iid_points() -> None:
    cfg = config()
    geometry = MODULE.geometry_from_token(cfg, 17)
    assert geometry.shape == (3, 4)
    assert MODULE.overlap_graph_connected(geometry, cfg.minimum_overlap)
    points = MODULE.iid_fluid_points(cfg, 512, geometry, 123)
    assert points.shape == (512, 3)
    assert np.all(MODULE.fluid_mask(cfg, points, geometry))
    assert len(np.unique(points, axis=0)) == len(points)


def test_softmin_and_contract() -> None:
    cfg = config()
    MODULE.validate_contract(cfg)
    geometry = MODULE.geometry_from_token(cfg, 29)
    centers = geometry[:, :3]
    level = MODULE.obstacle_level_numpy(cfg, centers, geometry)
    assert np.all(level < 0.0)
    assert cfg.metaball_softmin_tau == 0.02
    assert cfg.reorthogonalization_passes == 2
    assert cfg.report_raw_rank is False
