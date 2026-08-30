import json
from pathlib import Path

import numpy as np
import torch

from eigenfluid.gen.airfoil2d.ns_reference import (
    LBMSolverConfig,
    separation_diagnostics,
    solve_reference,
)
from eigenfluid.gen.airfoil2d.postprocess import (
    generalized_symmetric_eig,
)
from eigenfluid.gen.airfoil2d.training import (
    Config,
    OfficialPeelingFields,
    airfoil_level_numpy,
    body_coordinates_numpy,
    continuous_parameters,
    fluid_points,
    hard_open_outlet_envelope,
)


def test_final_protocol_config():
    case_root = Path(__file__).resolve().parents[1]
    values = json.loads(
        (case_root / "configs" / "k256_final.json").read_text(encoding="utf-8")
    )
    cfg = Config(
        **{
            key: value
            for key, value in values.items()
            if key in Config.__dataclass_fields__
        }
    )
    assert values["modes"] == 256
    assert values["epochs"] == 10000
    assert values["mlp_compute_dtype"] == "float16"
    assert values["mode_forward_batch"] == 32
    assert values["points_per_step"] == 4096
    assert values["validation_points"] == 4096
    assert values["learning_rate"] == 8.0e-4
    assert values["early_stopping_patience_epochs"] == 1000
    assert values["lr_scheduler_patience_epochs"] == 200
    assert values["rayleigh_scale"] == 80.0
    assert values["sampling_mode"] == "mixed"
    assert (
        values["global_fraction"],
        values["airfoil_fraction"],
        values["wake_fraction"],
    ) == (
        0.5,
        0.25,
        0.25,
    )
    assert cfg.geometry_mode == "c2_ellipse"


def test_k1024_legacy_protocols_are_comparable():
    case_root = Path(__file__).resolve().parents[1]
    hybrid = json.loads(
        (case_root / "configs" / "k1024_c2_sobol_10000_fp16_hybrid50.json").read_text(
            encoding="utf-8"
        )
    )
    velocity = json.loads(
        (
            case_root / "configs" / "k1024_c2_sobol_1000_fp16_velocity_only.json"
        ).read_text(encoding="utf-8")
    )
    vorticity = json.loads(
        (
            case_root / "configs" / "k1024_c2_sobol_10000_fp16_vorticity100.json"
        ).read_text(encoding="utf-8")
    )
    assert hybrid["modes"] == velocity["modes"] == 1024
    assert hybrid["seed"] == velocity["seed"] == 20260801
    assert hybrid["points_per_step"] == velocity["points_per_step"] == 4096
    assert hybrid["mode_forward_batch"] == velocity["mode_forward_batch"] == 32
    assert hybrid["fp32_boundary_layers"]
    assert velocity["fp32_boundary_layers"]
    assert hybrid["basis_objective"] == "velocity_vorticity_hybrid"
    assert hybrid["hybrid_velocity_weight"] == 0.5
    assert hybrid["hybrid_vorticity_weight"] == 0.5
    assert hybrid["hybrid_vorticity_anchor_points"] == 512
    assert velocity["basis_objective"] == "velocity_vorticity_hybrid"
    assert velocity["hybrid_velocity_weight"] == 1.0
    assert velocity["hybrid_vorticity_weight"] == 0.0
    assert velocity["epochs"] == 1000
    assert vorticity["basis_objective"] == "velocity_vorticity_hybrid"
    assert vorticity["hybrid_velocity_weight"] == 0.0
    assert vorticity["hybrid_vorticity_weight"] == 1.0

    run_only = {
        "epochs",
        "early_stopping_patience_epochs",
        "result_root",
    }
    objective_only = {
        "hybrid_velocity_weight",
        "hybrid_vorticity_weight",
    }
    common_keys = set(hybrid) - run_only - objective_only
    assert common_keys == set(velocity) - run_only - objective_only
    assert common_keys == set(vorticity) - run_only - objective_only
    for key in common_keys:
        assert hybrid[key] == velocity[key] == vorticity[key], key

def test_certified_stokes_protocol_has_no_empirical_smoothing_term():
    case_root = Path(__file__).resolve().parents[1]
    values = json.loads(
        (
            case_root
            / "configs"
            / "k1024_c2_stokes_certified_fp64_5000.json"
        ).read_text(encoding="utf-8")
    )
    assert values["basis_objective"] == "stokes_certified"
    assert values["sampling_mode"] == "uniform_nested_physical_area"
    assert values["stokes_min_quadrature"] == 4096
    assert values["stokes_max_quadrature"] > values["stokes_min_quadrature"]
    assert values["mlp_compute_dtype"] == "float64"
    assert values["modes"] == 1024
    for name in (
        "vorticity_regularization_weight",
        "small_vorticity_regularization_weight",
        "large_vorticity_preservation_weight",
        "peeled_subspace_vorticity_regularization_weight",
    ):
        assert values[name] == 0.0


def test_joint_geometry_and_sampling():
    cfg = Config()
    beta, alpha_deg = 0.7, 9.0
    points = fluid_points(cfg, 64, beta, alpha_deg, cfg.seed)
    assert points.shape == (64, 2)
    assert np.all(airfoil_level_numpy(cfg, points, beta, alpha_deg) > 0.0)
    assert np.all((points[:, 0] >= cfg.xmin) & (points[:, 0] <= cfg.xmax))
    assert np.all((points[:, 1] >= cfg.ymin) & (points[:, 1] <= cfg.ymax))


def test_default_training_draw_is_4096_mesh_free_points():
    cfg = Config()
    points = fluid_points(
        cfg, cfg.points_per_step, beta=0.35, alpha_deg=-4.0, seed=cfg.seed
    )
    assert points.shape == (4096, 2)
    # A Cartesian mesh repeats each coordinate many times. These direct,
    # scrambled-Sobol coordinates are essentially unique along both axes.
    assert len(np.unique(points[:, 0])) > 4000
    assert len(np.unique(points[:, 1])) > 4000
    assert np.all(airfoil_level_numpy(cfg, points, beta=0.35, alpha_deg=-4.0) > 0.0)


def test_c2_ellipse_surface_and_rotation():
    cfg = Config()
    beta, alpha_deg = 0.4, -7.0
    s = np.linspace(0.0, 1.0, 17)
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    half = (
        0.5
        * thickness
        * cfg.chord
        * np.sqrt(np.maximum(0.0, 1.0 - (2.0 * s - 1.0) ** 2))
    )
    angle = np.deg2rad(-alpha_deg)
    xb = s * cfg.chord
    yb = half
    dx = xb - 0.25 * cfg.chord
    points = np.column_stack(
        (
            cfg.quarter_chord_x + np.cos(angle) * dx - np.sin(angle) * yb,
            cfg.quarter_chord_y + np.sin(angle) * dx + np.cos(angle) * yb,
        )
    )
    recovered_x, recovered_y = body_coordinates_numpy(cfg, points, alpha_deg)
    assert np.allclose(recovered_x, xb)
    assert np.allclose(recovered_y, yb)
    assert np.max(np.abs(airfoil_level_numpy(cfg, points, beta, alpha_deg))) < 1e-12


def test_forward_uses_beta_and_alpha():
    cfg = Config(modes=3, hidden=8, hidden_layers=2, frequencies=2)
    model = OfficialPeelingFields(cfg)
    points = fluid_points(cfg, 16, 0.3, 4.0, cfg.seed)
    xy = torch.tensor(points)
    beta = torch.full((16, 1), 0.3)
    alpha = torch.full((16, 1), 4.0)
    output = model(xy, beta, alpha)
    assert output.shape == (16, 3)
    changed_beta = model(xy, beta + 0.2, alpha)
    changed_alpha = model(xy, beta, alpha - 3.0)
    assert not torch.allclose(output, changed_beta)
    assert not torch.allclose(output, changed_alpha)


def test_top_and_bottom_are_natural_free_boundaries():
    cfg = Config()
    points = torch.tensor([[8.0, cfg.ymin], [8.0, cfg.ymax]], dtype=torch.float32)
    beta = torch.full((2, 1), 0.5)
    alpha = torch.zeros((2, 1))
    envelope = hard_open_outlet_envelope(cfg, points, beta, alpha)
    assert torch.all(envelope > 0.0)


def test_endpoint_anchors_cover_all_corners():
    cfg = Config(endpoint_period=20)
    observed = {continuous_parameters(cfg, epoch) for epoch in (20, 40, 60, 80)}
    expected = {
        (0.0, cfg.alpha_min_deg),
        (0.0, cfg.alpha_max_deg),
        (1.0, cfg.alpha_min_deg),
        (1.0, cfg.alpha_max_deg),
    }
    assert observed == expected


def test_generalized_eigenproblem():
    mass = np.diag([1.0, 2.0, 3.0])
    stiffness = np.diag([3.0, 4.0, 5.0])
    values, vectors = generalized_symmetric_eig(stiffness, mass, 3)
    assert np.all(np.diff(values) >= 0.0)
    assert np.allclose(vectors.T @ mass @ vectors, np.eye(3), atol=1.0e-8)


def test_independent_ns_smoke(tmp_path):
    cfg = Config()
    output = tmp_path / "reference.npz"
    solve_reference(
        cfg,
        LBMSolverConfig(
            nx=61,
            ny=25,
            reynolds=100.0,
            steps=4,
            warmup_steps=2,
            snapshot_every=2,
            require_separated_gt=False,
            device="cpu",
        ),
        beta=0.5,
        alpha_deg=6.0,
        output=output,
    )
    cache = np.load(output)
    assert cache["u"].shape == (2, 25, 61)
    assert cache["vorticity"].shape == (2, 25, 61)
    assert cache["airfoil"].any()
    assert not cache["solid"][0].any()
    assert not cache["solid"][-1].any()
    assert np.allclose(cache["u"][:, 0, :], 0.08, atol=1.0e-6)
    assert np.allclose(cache["u"][:, -1, :], 0.08, atol=1.0e-6)
    assert np.isfinite(cache["u"]).all()


def test_monotone_high_angle_field_is_rejected_as_gt():
    cfg = Config()
    x = np.linspace(cfg.xmin, cfg.xmax, 101)
    y = np.linspace(cfg.ymin, cfg.ymax, 41)
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    fluid = (
        airfoil_level_numpy(cfg, points, beta=0.5, alpha_deg=10.0).reshape(xgrid.shape)
        > 0.0
    )
    velocity = np.zeros((2, *xgrid.shape), dtype=np.float32)
    vorticity = np.zeros_like(velocity)
    result = separation_diagnostics(
        cfg,
        beta=0.5,
        alpha_deg=10.0,
        xgrid=xgrid,
        ygrid=ygrid,
        fluid=fluid,
        u=velocity,
        vorticity=vorticity,
        inlet_speed=0.08,
    )
    assert result["separation_expected"]
    assert not result["separated_gt_quality_pass"]
    assert result["maximum_coherent_vortex_count"] == 0
