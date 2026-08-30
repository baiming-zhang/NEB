"""No-PyTorch fixed-graph trainer for the parameterized 2D airfoil."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    peel_step,
    uniform_fluid_points,
)
from eigenfluid.gen.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    compute_certified_stokes_basis,
)


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "hardcoded_train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def load_neutral(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def save_neutral(
    path: Path, model: HardcodedModel, cfg: dict[str, Any], epoch: int,
    best_validation: float, training_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **model.arrays())
    os.replace(temporary, path)
    metadata = {
        "format": "eigenfluid_airfoil_hardcoded_v1",
        "epoch": epoch,
        "best_validation": best_validation,
        "config": cfg,
        "training_uses_pytorch": False,
        "spatial_ad": "hardcoded_xy_jet2",
        "parameter_backward": "hardcoded_reverse_jet2",
        "training_state": training_state or {},
    }
    meta_tmp = path.with_suffix(".json.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(meta_tmp, path.with_suffix(".json"))


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data = run_dir / "data"
    data.mkdir(exist_ok=True)
    target = data / "hardcoded_training_history.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    keys = [
        "epoch", "seconds", "beta", "alpha_deg", "train_loss",
        "train_base_loss", "train_vorticity_regularization_loss",
        "train_enstrophy_ratio",
        "train_small_vorticity_regularization_loss",
        "train_large_vorticity_preservation_loss",
        "train_small_vorticity_highpass_ratio",
        "train_large_vorticity_drift_ratio",
        "train_peeled_subspace_vorticity_regularization_loss",
        "train_peeled_subspace_vorticity_highpass_trace",
        "train_velocity_rayleigh", "train_vorticity_rayleigh",
        "train_mean_energy", "train_max_energy", "train_mean_projection_ratio",
        "train_orth_error", "train_rank", "validation_loss",
        "validation_base_loss",
        "validation_vorticity_regularization_loss",
        "validation_enstrophy_ratio",
        "validation_small_vorticity_regularization_loss",
        "validation_large_vorticity_preservation_loss",
        "validation_small_vorticity_highpass_ratio",
        "validation_large_vorticity_drift_ratio",
        "validation_peeled_subspace_vorticity_regularization_loss",
        "validation_peeled_subspace_vorticity_highpass_trace",
        "validation_velocity_rayleigh", "validation_vorticity_rayleigh",
        "validation_mean_energy", "validation_max_energy",
        "validation_mean_projection_ratio", "validation_orth_error",
        "validation_min_rank",
        "validation_loss_p95", "validation_loss_max", "learning_rate",
        "best_validation_epoch", "early_stop_counter_epochs",
        "lr_plateau_counter_epochs",
        "stokes_construction_quadrature", "stokes_certified_modes",
        "stokes_min_construction_rank", "stokes_max_mass_defect",
        "stokes_max_ritz_residual", "stokes_max_eigenvalue_drift",
    ]
    with (data / "hardcoded_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in history)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    case_losses = np.asarray([item["loss"] for item in items], dtype=np.float64)
    return {
        "loss": float(np.mean(case_losses)),
        "base_loss": float(np.mean([item["base_loss"] for item in items])),
        "vorticity_regularization_loss": float(
            np.mean(
                [
                    item["vorticity_regularization_loss"]
                    for item in items
                ]
            )
        ),
        "enstrophy_ratio": float(
            np.mean([item["enstrophy_ratio"] for item in items])
        ),
        "small_vorticity_regularization_loss": float(np.mean([
            item["small_vorticity_regularization_loss"] for item in items
        ])),
        "large_vorticity_preservation_loss": float(np.mean([
            item["large_vorticity_preservation_loss"] for item in items
        ])),
        "small_vorticity_highpass_ratio": float(np.mean([
            item["small_vorticity_highpass_ratio"] for item in items
        ])),
        "large_vorticity_drift_ratio": float(np.mean([
            item["large_vorticity_drift_ratio"] for item in items
        ])),
        "peeled_subspace_vorticity_regularization_loss": float(np.mean([
            item["peeled_subspace_vorticity_regularization_loss"]
            for item in items
        ])),
        "peeled_subspace_vorticity_highpass_trace": float(np.mean([
            item["peeled_subspace_vorticity_highpass_trace"]
            for item in items
        ])),
        "velocity_rayleigh": float(np.mean([
            item.get("velocity_rayleigh", item["mean_energy"])
            for item in items
        ])),
        "vorticity_rayleigh": float(np.mean([
            item.get("vorticity_rayleigh", item["enstrophy_ratio"])
            for item in items
        ])),
        "loss_p95": float(np.percentile(case_losses, 95.0)),
        "loss_max": float(np.max(case_losses)),
        "mean_energy": float(np.mean([item["mean_energy"] for item in items])),
        "max_energy": float(np.max([item["max_energy"] for item in items])),
        "mean_projection_ratio": float(
            np.mean([item["mean_projection_ratio"] for item in items])
        ),
        "orth_error": float(np.max([item["orth_error"] for item in items])),
        "min_rank": int(np.min([
            item["effective_raw_rank"]
            for item in items
            if item["effective_raw_rank"] >= 0
        ])),
    }


def validate(
    cfg: Any,
    model: HardcodedModel,
    sample_pack: dict[str, np.ndarray] | None = None,
    reference_model: HardcodedModel | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples
        )
    ]
    for index, (beta, alpha) in enumerate(pairs):
        if sample_pack is None:
            point_generator = (
                uniform_fluid_points
                if str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
                == "stokes_certified"
                else fluid_points
            )
            points = point_generator(
                cfg, cfg.validation_points, beta, alpha, cfg.seed + 900_000 + index
            )
        else:
            beta = float(sample_pack["validation_beta"][index])
            alpha = float(sample_pack["validation_alpha"][index])
            points = sample_pack["validation_points"][index]
        details.append(
            peel_step(
                cfg, model, points, beta, alpha, False,
                reference_model=reference_model,
                full_diagnostics=(index == 0),
            )
        )
    return aggregate(details), details


def certify_validation_stokes(
    cfg: Any,
    model: HardcodedModel,
    quadrature: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Certify every configured parameter validation case independently."""

    case_rows: list[dict[str, Any]] = []
    for index, beta in enumerate(
        np.linspace(0.0, 1.0, cfg.validation_beta_samples)
    ):
        for alpha_index, alpha in enumerate(
            np.linspace(
                cfg.alpha_min_deg,
                cfg.alpha_max_deg,
                cfg.validation_alpha_samples,
            )
        ):
            case_index = index * cfg.validation_alpha_samples + alpha_index
            certificate = compute_certified_stokes_basis(
                model,
                cfg,
                float(beta),
                float(alpha),
                quadrature,
                quadrature,
                seed=cfg.seed + 2_000_000 + 10_007 * case_index,
            )
            certified = int(certificate["certified_count"])
            mass_slice = certificate["mass_defect"][:certified, :certified]
            case_rows.append(
                {
                    "beta": float(beta),
                    "alpha_deg": float(alpha),
                    "quadrature": int(quadrature),
                    "construction_rank": int(
                        certificate["construction_rank"]
                    ),
                    "certified_modes": certified,
                    "max_mass_defect": (
                        float(np.max(mass_slice)) if certified else float("inf")
                    ),
                    "max_ritz_residual": (
                        float(
                            np.max(
                                certificate["ritz_residual_relative"][:certified]
                            )
                        )
                        if certified
                        else float("inf")
                    ),
                    "max_eigenvalue_drift": (
                        float(
                            np.max(
                                certificate["eigenvalue_drift_relative"][
                                    :certified
                                ]
                            )
                        )
                        if certified
                        else float("inf")
                    ),
                    "mass_defect": certificate["mass_defect"].diagonal().tolist(),
                    "mass_budget": certificate["mass_budget"].diagonal().tolist(),
                    "ritz_residual_relative": certificate[
                        "ritz_residual_relative"
                    ].tolist(),
                    "ritz_residual_budget": certificate[
                        "ritz_residual_budget"
                    ].tolist(),
                    "eigenvalue_drift_relative": certificate[
                        "eigenvalue_drift_relative"
                    ].tolist(),
                    "eigenvalue_drift_budget": certificate[
                        "eigenvalue_drift_budget"
                    ].tolist(),
                }
            )
    finite_mass = [
        item["max_mass_defect"]
        for item in case_rows
        if np.isfinite(item["max_mass_defect"])
    ]
    finite_residual = [
        item["max_ritz_residual"]
        for item in case_rows
        if np.isfinite(item["max_ritz_residual"])
    ]
    finite_drift = [
        item["max_eigenvalue_drift"]
        for item in case_rows
        if np.isfinite(item["max_eigenvalue_drift"])
    ]
    summary = {
        "quadrature": int(quadrature),
        "certified_modes": int(
            min(item["certified_modes"] for item in case_rows)
        ),
        "min_construction_rank": int(
            min(item["construction_rank"] for item in case_rows)
        ),
        "max_mass_defect": max(finite_mass, default=float("inf")),
        "max_ritz_residual": max(finite_residual, default=float("inf")),
        "max_eigenvalue_drift": max(finite_drift, default=float("inf")),
    }
    return summary, case_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sample-pack", type=Path)
    parser.add_argument("--validation-pack", type=Path)
    parser.add_argument("--self-test-epochs", type=int, default=0)
    parser.add_argument("--self-test-points", type=int, default=256)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument(
        "--vorticity-regularization-weight", type=float
    )
    parser.add_argument(
        "--small-vorticity-regularization-weight", type=float
    )
    parser.add_argument(
        "--large-vorticity-preservation-weight", type=float
    )
    parser.add_argument(
        "--peeled-subspace-vorticity-regularization-weight", type=float
    )
    parser.add_argument("--small-vorticity-scale-chords", type=float)
    parser.add_argument("--small-vorticity-anchor-points", type=int)
    parser.add_argument("--regularization-reference", type=Path)
    parser.add_argument("--freeze-prefix-modes", type=int)
    parser.add_argument(
        "--orthonormal-peeling-fast-path", action="store_true"
    )
    parser.add_argument("--early-stopping-patience-epochs", type=int)
    parser.add_argument("--reset-validation-state", action="store_true")
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("hardcoded trainer must not import PyTorch")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    (args.run_dir / "data").mkdir(exist_ok=True)
    metadata, arrays = load_neutral(args.resume)
    sample_pack = None
    if args.sample_pack is not None:
        with np.load(args.sample_pack, allow_pickle=False) as archive:
            sample_pack = {name: archive[name] for name in archive.files}
    validation_pack = None
    if args.validation_pack is not None:
        with np.load(args.validation_pack, allow_pickle=False) as archive:
            validation_pack = {name: archive[name] for name in archive.files}
    config_dict = metadata["config"]
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.learning_rate is not None:
        config_dict["learning_rate"] = args.learning_rate
    if args.vorticity_regularization_weight is not None:
        config_dict["vorticity_regularization_weight"] = (
            args.vorticity_regularization_weight
        )
    for argument, key in (
        (
            args.small_vorticity_regularization_weight,
            "small_vorticity_regularization_weight",
        ),
        (
            args.large_vorticity_preservation_weight,
            "large_vorticity_preservation_weight",
        ),
        (
            args.peeled_subspace_vorticity_regularization_weight,
            "peeled_subspace_vorticity_regularization_weight",
        ),
        (
            args.small_vorticity_scale_chords,
            "small_vorticity_scale_chords",
        ),
        (
            args.small_vorticity_anchor_points,
            "small_vorticity_anchor_points",
        ),
        (
            args.freeze_prefix_modes,
            "freeze_prefix_modes",
        ),
    ):
        if argument is not None:
            config_dict[key] = argument
    if args.orthonormal_peeling_fast_path:
        config_dict["orthonormal_peeling_fast_path"] = True
    if args.early_stopping_patience_epochs is not None:
        config_dict["early_stopping_patience_epochs"] = (
            args.early_stopping_patience_epochs
        )
    cfg = cfg_namespace(config_dict)
    model = HardcodedModel(cfg, arrays)
    scale_selective = (
        float(getattr(cfg, "small_vorticity_regularization_weight", 0.0)) > 0.0
        or float(getattr(cfg, "large_vorticity_preservation_weight", 0.0)) > 0.0
    )
    reference_model = None
    if scale_selective:
        if args.regularization_reference is None:
            raise ValueError(
                "--regularization-reference is required for scale-selective "
                "vorticity regularization"
            )
        _, reference_arrays = load_neutral(args.regularization_reference)
        reference_model = HardcodedModel(cfg, reference_arrays)
    start_epoch = int(metadata["epoch"]) + 1
    stop_epoch = (
        start_epoch + args.self_test_epochs - 1
        if args.self_test_epochs > 0
        else int(cfg.epochs)
    )
    hardcoded_history_path = (
        args.run_dir / "data" / "hardcoded_training_history.json"
    )
    legacy_history_path = args.run_dir / "data" / "training_history.json"
    if hardcoded_history_path.exists():
        history = json.loads(hardcoded_history_path.read_text(encoding="utf-8"))
    elif legacy_history_path.exists():
        history = json.loads(legacy_history_path.read_text(encoding="utf-8"))
    else:
        history = []
    best_validation = float(metadata.get("best_validation", float("inf")))
    restored_state = metadata.get("training_state", {})
    current_learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else restored_state.get("current_learning_rate", cfg.learning_rate)
    )
    cfg.learning_rate = current_learning_rate
    config_dict["learning_rate"] = current_learning_rate
    best_validation_epoch = int(restored_state.get("best_validation_epoch", 0))
    lr_reference_loss = float(
        restored_state.get("lr_reference_loss", best_validation)
    )
    lr_reference_epoch = int(restored_state.get("lr_reference_epoch", 0))
    lr_reductions = int(restored_state.get("lr_reductions", 0))
    certified_stokes = (
        str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
        == "stokes_certified"
    )
    current_construction_quadrature = int(
        restored_state.get(
            "stokes_construction_quadrature",
            getattr(cfg, "stokes_min_quadrature", cfg.points_per_step),
        )
    )
    maximum_construction_quadrature = int(
        getattr(cfg, "stokes_max_quadrature", current_construction_quadrature)
    )
    best_certified_count = int(
        restored_state.get("best_certified_count", 0)
    )
    last_certified_count = int(
        restored_state.get("last_certified_count", best_certified_count)
    )
    if args.reset_validation_state:
        best_validation = float("inf")
        best_validation_epoch = start_epoch - 1
        lr_reference_loss = float("inf")
        lr_reference_epoch = start_epoch - 1
        lr_reductions = 0
        best_certified_count = 0
        last_certified_count = 0
    protocol = {
        "algorithm": (
            "certified_variational_stokes_generalized_eigen_peeling"
            if certified_stokes
            else "shared_velocity_vorticity_hybrid_residual_peeling"
            if getattr(cfg, "basis_objective", "velocity_rayleigh")
            == "velocity_vorticity_hybrid"
            else "official_sss_fixed_order_residual_peeling_joint_beta_alpha"
        ),
        "backend": "cupy_cuda_fixed_graph",
        "mlp_compute_dtype": getattr(cfg, "mlp_compute_dtype", "float32"),
        "mode_forward_batch": getattr(cfg, "mode_forward_batch", 1),
        "fp16_shadow_weights": (
            getattr(cfg, "mlp_compute_dtype", "float32") == "float16"
        ),
        "batched_forward_weights_synchronized_after_adam": True,
        "fp32_boundary_layers": getattr(cfg, "fp32_boundary_layers", False),
        "jet2_linear_channels_fused": True,
        "mlp_accumulation_dtype": (
            "float64"
            if getattr(cfg, "mlp_compute_dtype", "float32") == "float64"
            else "float32"
        ),
        "hardcoded_reverse_dtype": (
            "float64"
            if getattr(cfg, "mlp_compute_dtype", "float32") == "float64"
            else "float32"
        ),
        "master_weight_dtype": (
            "float64"
            if getattr(cfg, "mlp_compute_dtype", "float32") == "float64"
            else "float32"
        ),
        "adam_state_dtype": (
            "float64"
            if getattr(cfg, "mlp_compute_dtype", "float32") == "float64"
            else "float32"
        ),
        "fp16_loss_scale_maximum": getattr(cfg, "fp16_loss_scale", 1.0),
        "fp16_loss_scaling": "dynamic bidirectional with overflow headroom",
        "peeling_ritz_dtype": "float64",
        "pytorch_imported_by_trainer": False,
        "automatic_differentiation_used": False,
        "spatial_derivatives": "hardcoded Jet2 propagation for x,y only",
        "parameter_backward": "hardcoded reverse Jet2 pass",
        "one_network_and_adam_per_mode": True,
        "fixed_mode_order": True,
        "freeze_prefix_modes": getattr(cfg, "freeze_prefix_modes", 0),
        "orthonormal_peeling_fast_path": getattr(
            cfg, "orthonormal_peeling_fast_path", False
        ) or certified_stokes,
        "mode_sorting_used": False,
        "least_squares_residual_peeling": True,
        "previous_modes_detached": True,
        "points_per_epoch": cfg.points_per_step,
        "stokes_variational_problem": {
            "enabled": certified_stokes,
            "mass_form": "integral u dot v over physical fluid area",
            "stiffness_form": "integral grad(u) colon grad(v) over physical fluid area",
            "divergence_free": "exact curl of C2 streamfunction",
            "quadrature": "uniform nested scrambled Sobol with solid rejection",
            "minimum_points": current_construction_quadrature,
            "maximum_points": maximum_construction_quadrature,
            "adaptive_rule": (
                "double after independent-grid certification failure; retain "
                "only the backward-resolved certified spectral prefix"
            ),
            "rank_cutoff": "float64 Gram backward-error bound",
            "empirical_regularization": False,
        },
        "reference_sample_pack": (
            str(args.sample_pack.resolve()) if args.sample_pack is not None else None
        ),
        "fixed_validation_pack": (
            str(args.validation_pack.resolve())
            if args.validation_pack is not None
            else None
        ),
        "geometry": "C2 cubic envelope of a C-infinity ellipse implicit function",
        "farfield_boundary": (
            "double-zero perturbation envelope; total velocity remains free stream"
        ),
        "hybrid_basis_objective": {
            "type": getattr(cfg, "basis_objective", "velocity_rayleigh"),
            "velocity_weight": getattr(cfg, "hybrid_velocity_weight", 1.0),
            "vorticity_weight": getattr(cfg, "hybrid_vorticity_weight", 0.0),
            "velocity_rayleigh_scale": getattr(
                cfg, "hybrid_velocity_rayleigh_scale", cfg.rayleigh_scale
            ),
            "vorticity_rayleigh_scale": getattr(
                cfg, "hybrid_vorticity_rayleigh_scale", cfg.rayleigh_scale
            ),
            "vorticity_mass_length_scale_chords": getattr(
                cfg, "hybrid_vorticity_mass_length_scale_chords", 1.0
            ),
            "vorticity_difference_chords": getattr(
                cfg, "hybrid_vorticity_difference_chords", 0.05
            ),
            "vorticity_anchor_points": getattr(
                cfg, "hybrid_vorticity_anchor_points", 0
            ),
            "reorthogonalization_passes": getattr(
                cfg, "hybrid_reorthogonalization_passes", 1
            ),
            "vorticity_gradient": (
                "symmetric four-neighbor Jet2-Hessian differences"
            ),
            "jet3_avoided": True,
        },
        "vorticity_regularization": {
            "weight": getattr(
                cfg, "vorticity_regularization_weight", 0.0
            ),
            "objective": (
                "full-domain vorticity L2 divided by modal velocity L2"
            ),
            "spatial_mask": "none",
            "geometry_prior": False,
            "requires_streamfunction_derivative_order": 2,
        },
        "scale_selective_vorticity_regularization": {
            "small_scale_weight": getattr(
                cfg, "small_vorticity_regularization_weight", 0.0
            ),
            "large_scale_preservation_weight": getattr(
                cfg, "large_vorticity_preservation_weight", 0.0
            ),
            "scale_chords": getattr(
                cfg, "small_vorticity_scale_chords", 0.05
            ),
            "anchor_points": getattr(
                cfg, "small_vorticity_anchor_points", 512
            ),
            "reference_checkpoint": (
                str(args.regularization_reference.resolve())
                if args.regularization_reference is not None
                else None
            ),
            "spatial_mask": "none",
            "geometry_prior": False,
            "highpass_operator": "center minus normalized four-neighbor mean",
            "lowpass_operator": "half center plus half normalized neighbor mean",
            "peeled_subspace_weight": getattr(
                cfg,
                "peeled_subspace_vorticity_regularization_weight",
                0.0,
            ),
            "peeled_subspace_objective": (
                "trace of fixed-scale vorticity highpass energy over the "
                "mass-orthonormal residual-peeled basis"
            ),
            "basis_invariance": (
                "Frobenius trace invariant under orthogonal changes of the "
                "mass-orthonormal basis"
            ),
        },
        "early_stopping_patience_epochs": cfg.early_stopping_patience_epochs,
        "lr_scheduler": {
            "type": "validation_plateau",
            "patience_epochs": cfg.lr_scheduler_patience_epochs,
            "factor": cfg.lr_scheduler_factor,
            "minimum_learning_rate": cfg.min_learning_rate,
        },
    }
    (args.run_dir / "hardcoded_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    append_log(
        args.run_dir,
        f"start hardcoded-no-pytorch device=cuda modes={cfg.modes} "
        f"points={current_construction_quadrature if certified_stokes else cfg.points_per_step} "
        f"resume_epoch={metadata['epoch']} "
        f"lr={current_learning_rate:.6e} "
        f"vorticity_reg="
        f"{getattr(cfg, 'vorticity_regularization_weight', 0.0):.3e} "
        f"small_vorticity_reg="
        f"{getattr(cfg, 'small_vorticity_regularization_weight', 0.0):.3e} "
        f"large_vorticity_keep="
        f"{getattr(cfg, 'large_vorticity_preservation_weight', 0.0):.3e} "
        f"peeled_subspace_reg="
        f"{getattr(cfg, 'peeled_subspace_vorticity_regularization_weight', 0.0):.3e}",
    )
    started = time.perf_counter()
    stopped_early = False
    final_epoch = int(metadata["epoch"])
    for epoch in range(start_epoch, stop_epoch + 1):
        count = (
            args.self_test_points
            if args.self_test_epochs > 0
            else current_construction_quadrature
            if certified_stokes
            else cfg.points_per_step
        )
        if sample_pack is not None and args.self_test_epochs <= 0:
            pack_index = epoch - 1
            if pack_index >= len(sample_pack["train_beta"]):
                raise IndexError(f"sample pack has no entry for epoch {epoch}")
            beta = float(sample_pack["train_beta"][pack_index])
            alpha = float(sample_pack["train_alpha"][pack_index])
            points = sample_pack["train_points"][pack_index]
        else:
            beta, alpha = continuous_parameters(cfg, epoch)
            point_generator = uniform_fluid_points if certified_stokes else fluid_points
            points = point_generator(
                cfg, count, beta, alpha, cfg.seed + epoch * 1009
            )
        diagnostic = peel_step(
            cfg, model, points, beta, alpha, True,
            reference_model=reference_model,
            full_diagnostics=(
                args.self_test_epochs > 0
                or epoch == 1
                or epoch % cfg.validate_every == 0
            ),
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "beta": beta,
            "alpha_deg": alpha,
            "train_loss": diagnostic["loss"],
            "train_base_loss": diagnostic["base_loss"],
            "train_vorticity_regularization_loss": diagnostic[
                "vorticity_regularization_loss"
            ],
            "train_enstrophy_ratio": diagnostic[
                "enstrophy_ratio"
            ],
            "train_small_vorticity_regularization_loss": diagnostic[
                "small_vorticity_regularization_loss"
            ],
            "train_large_vorticity_preservation_loss": diagnostic[
                "large_vorticity_preservation_loss"
            ],
            "train_small_vorticity_highpass_ratio": diagnostic[
                "small_vorticity_highpass_ratio"
            ],
            "train_large_vorticity_drift_ratio": diagnostic[
                "large_vorticity_drift_ratio"
            ],
            "train_peeled_subspace_vorticity_regularization_loss": diagnostic[
                "peeled_subspace_vorticity_regularization_loss"
            ],
            "train_peeled_subspace_vorticity_highpass_trace": diagnostic[
                "peeled_subspace_vorticity_highpass_trace"
            ],
            "train_velocity_rayleigh": diagnostic.get("velocity_rayleigh"),
            "train_vorticity_rayleigh": diagnostic.get("vorticity_rayleigh"),
            "train_mean_energy": diagnostic["mean_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic["mean_projection_ratio"],
            "train_orth_error": diagnostic["orth_error"],
            "train_rank": diagnostic["effective_raw_rank"],
            "backend": "hardcoded_cupy_no_pytorch",
            "learning_rate": current_learning_rate,
            "stokes_construction_quadrature": (
                current_construction_quadrature if certified_stokes else None
            ),
        }
        if args.self_test_epochs > 0:
            append_log(
                args.run_dir,
                f"hardcoded-test epoch={epoch:04d} train={diagnostic['loss']:.6e} "
                f"base={diagnostic['base_loss']:.6e} "
                f"omega_reg="
                f"{diagnostic['vorticity_regularization_loss']:.3e} "
                f"small_ratio="
                f"{diagnostic['small_vorticity_highpass_ratio']:.3e} "
                f"large_drift="
                f"{diagnostic['large_vorticity_drift_ratio']:.3e} "
                f"peeled_trace="
                f"{diagnostic['peeled_subspace_vorticity_highpass_trace']:.3e} "
                f"velocity_R={diagnostic.get('velocity_rayleigh', float('nan')):.3e} "
                f"vorticity_R={diagnostic.get('vorticity_rayleigh', float('nan')):.3e} "
                f"orth={diagnostic['orth_error']:.2e} rank={diagnostic['effective_raw_rank']}",
            )
        elif epoch == 1 or epoch % cfg.validate_every == 0:
            validation_source = (
                validation_pack if validation_pack is not None else sample_pack
            )
            validation, details = validate(
                cfg, model, validation_source,
                reference_model=reference_model,
            )
            case_rows = [
                {
                    "beta": float(
                        validation_source["validation_beta"][index]
                        if validation_source is not None
                        else np.linspace(0.0, 1.0, cfg.validation_beta_samples)[
                            index // cfg.validation_alpha_samples
                        ]
                    ),
                    "alpha_deg": float(
                        validation_source["validation_alpha"][index]
                        if validation_source is not None
                        else np.linspace(
                            cfg.alpha_min_deg,
                            cfg.alpha_max_deg,
                            cfg.validation_alpha_samples,
                        )[index % cfg.validation_alpha_samples]
                    ),
                    "loss": item["loss"],
                    "base_loss": item["base_loss"],
                    "vorticity_regularization_loss": item[
                        "vorticity_regularization_loss"
                    ],
                    "enstrophy_ratio": item[
                        "enstrophy_ratio"
                    ],
                    "small_vorticity_highpass_ratio": item[
                        "small_vorticity_highpass_ratio"
                    ],
                    "large_vorticity_drift_ratio": item[
                        "large_vorticity_drift_ratio"
                    ],
                    "peeled_subspace_vorticity_highpass_trace": item[
                        "peeled_subspace_vorticity_highpass_trace"
                    ],
                    "mean_energy": item["mean_energy"],
                    "max_energy": item["max_energy"],
                    "orth_error": item["orth_error"],
                    "rank": item["effective_raw_rank"],
                }
                for index, item in enumerate(details)
            ]
            row.update(
                {
                    "validation_loss": validation["loss"],
                    "validation_base_loss": validation["base_loss"],
                    "validation_vorticity_regularization_loss": (
                        validation[
                            "vorticity_regularization_loss"
                        ]
                    ),
                    "validation_enstrophy_ratio": validation[
                        "enstrophy_ratio"
                    ],
                    "validation_small_vorticity_regularization_loss": (
                        validation["small_vorticity_regularization_loss"]
                    ),
                    "validation_large_vorticity_preservation_loss": (
                        validation["large_vorticity_preservation_loss"]
                    ),
                    "validation_small_vorticity_highpass_ratio": validation[
                        "small_vorticity_highpass_ratio"
                    ],
                    "validation_large_vorticity_drift_ratio": validation[
                        "large_vorticity_drift_ratio"
                    ],
                    "validation_peeled_subspace_vorticity_regularization_loss": (
                        validation[
                            "peeled_subspace_vorticity_regularization_loss"
                        ]
                    ),
                    "validation_peeled_subspace_vorticity_highpass_trace": (
                        validation[
                            "peeled_subspace_vorticity_highpass_trace"
                        ]
                    ),
                    "validation_velocity_rayleigh": validation[
                        "velocity_rayleigh"
                    ],
                    "validation_vorticity_rayleigh": validation[
                        "vorticity_rayleigh"
                    ],
                    "validation_loss_p95": validation["loss_p95"],
                    "validation_loss_max": validation["loss_max"],
                    "validation_mean_energy": validation["mean_energy"],
                    "validation_max_energy": validation["max_energy"],
                    "validation_mean_projection_ratio": validation[
                        "mean_projection_ratio"
                    ],
                    "validation_orth_error": validation["orth_error"],
                    "validation_min_rank": validation["min_rank"],
                    "validation_mode_energies": [
                        item["mode_energies"] for item in details
                    ],
                    "validation_cases": case_rows,
                }
            )
            certify_every = int(
                getattr(cfg, "stokes_certify_every", cfg.validate_every)
            )
            certification_due = certified_stokes and (
                epoch == 1 or epoch % max(1, certify_every) == 0
            )
            if certification_due:
                certification, certification_cases = certify_validation_stokes(
                    cfg, model, current_construction_quadrature
                )
                last_certified_count = int(
                    certification["certified_modes"]
                )
                row.update(
                    {
                        "stokes_certified_modes": last_certified_count,
                        "stokes_min_construction_rank": certification[
                            "min_construction_rank"
                        ],
                        "stokes_max_mass_defect": certification[
                            "max_mass_defect"
                        ],
                        "stokes_max_ritz_residual": certification[
                            "max_ritz_residual"
                        ],
                        "stokes_max_eigenvalue_drift": certification[
                            "max_eigenvalue_drift"
                        ],
                    }
                )
                certification_payload = {
                    "epoch": epoch,
                    "objective": "variational_stokes_generalized_eigenproblem",
                    "decision_rule": (
                        "independent nested coarse/fine integration uncertainty "
                        "plus float64 backward error"
                    ),
                    "summary": certification,
                    "cases": certification_cases,
                }
                certification_path = (
                    args.run_dir
                    / "data"
                    / f"stokes_certification_epoch{epoch:04d}.json"
                )
                certification_path.write_text(
                    json.dumps(certification_payload, indent=2),
                    encoding="utf-8",
                )
                latest_certification = (
                    args.run_dir / "data" / "stokes_certification_latest.json"
                )
                latest_tmp = latest_certification.with_suffix(".json.tmp")
                latest_tmp.write_text(
                    json.dumps(certification_payload, indent=2),
                    encoding="utf-8",
                )
                os.replace(latest_tmp, latest_certification)
                append_log(
                    args.run_dir,
                    f"stokes certificate epoch={epoch:04d} "
                    f"q={current_construction_quadrature} "
                    f"certified={last_certified_count}/{cfg.modes} "
                    f"rank={certification['min_construction_rank']}",
                )
                if (
                    last_certified_count < cfg.modes
                    and current_construction_quadrature
                    < maximum_construction_quadrature
                ):
                    current_construction_quadrature = min(
                        maximum_construction_quadrature,
                        2 * current_construction_quadrature,
                    )
                    append_log(
                        args.run_dir,
                        "stokes quadrature refined for following epochs "
                        f"to q={current_construction_quadrature}",
                    )
            relative_threshold = cfg.lr_scheduler_relative_threshold
            validation_improved = float(validation["loss"]) < best_validation * (
                1.0 - relative_threshold
            )
            improved = (
                last_certified_count > best_certified_count
                or (
                    last_certified_count == best_certified_count
                    and validation_improved
                )
                if certified_stokes
                else validation_improved
            )
            if improved:
                best_validation = float(validation["loss"])
                best_certified_count = max(
                    best_certified_count, last_certified_count
                )
                best_validation_epoch = epoch
                lr_reference_loss = best_validation
                lr_reference_epoch = epoch
                save_neutral(
                    args.run_dir / "checkpoints" / "hardcoded_best_model.npz",
                    model,
                    config_dict,
                    epoch,
                    best_validation,
                    {
                        "current_learning_rate": current_learning_rate,
                        "best_validation_epoch": best_validation_epoch,
                        "lr_reference_loss": lr_reference_loss,
                        "lr_reference_epoch": lr_reference_epoch,
                        "lr_reductions": lr_reductions,
                        "stokes_construction_quadrature": (
                            current_construction_quadrature
                            if certified_stokes
                            else None
                        ),
                        "best_certified_count": best_certified_count,
                        "last_certified_count": last_certified_count,
                    },
                )
            elif (
                epoch - lr_reference_epoch
                >= cfg.lr_scheduler_patience_epochs
                and current_learning_rate > cfg.min_learning_rate
            ):
                current_learning_rate = max(
                    cfg.min_learning_rate,
                    current_learning_rate * cfg.lr_scheduler_factor,
                )
                cfg.learning_rate = current_learning_rate
                config_dict["learning_rate"] = current_learning_rate
                lr_reference_loss = float(validation["loss"])
                lr_reference_epoch = epoch
                lr_reductions += 1
                append_log(
                    args.run_dir,
                    f"lr reduced epoch={epoch:04d} lr={current_learning_rate:.6e}",
                )
            row.update(
                {
                    "learning_rate": current_learning_rate,
                    "best_validation_epoch": best_validation_epoch,
                    "early_stop_counter_epochs": epoch - best_validation_epoch,
                    "lr_plateau_counter_epochs": epoch - lr_reference_epoch,
                }
            )
            append_log(
                args.run_dir,
                f"epoch={epoch:04d} beta={beta:.5f} alpha={alpha:+.3f} "
                f"train={diagnostic['loss']:.6e} "
                f"base={diagnostic['base_loss']:.6e} "
                f"omega_reg="
                f"{diagnostic['vorticity_regularization_loss']:.3e} "
                f"small_ratio="
                f"{diagnostic['small_vorticity_highpass_ratio']:.3e} "
                f"large_drift="
                f"{diagnostic['large_vorticity_drift_ratio']:.3e} "
                f"peeled_trace="
                f"{diagnostic['peeled_subspace_vorticity_highpass_trace']:.3e} "
                f"velocity_R={diagnostic.get('velocity_rayleigh', float('nan')):.3e} "
                f"vorticity_R={diagnostic.get('vorticity_rayleigh', float('nan')):.3e} "
                f"val_mean={validation['loss']:.6e} "
                f"val_p95={validation['loss_p95']:.6e} "
                f"val_max={validation['loss_max']:.6e} "
                f"lr={current_learning_rate:.3e} "
                f"orth={validation['orth_error']:.2e} rank={validation['min_rank']}",
            )
        if (
            args.self_test_epochs <= 0
            and cfg.probe_every > 0
            and epoch % cfg.probe_every == 0
        ):
            probe_points = (
                validation_pack["probe_points"]
                if validation_pack is not None and "probe_points" in validation_pack
                else sample_pack["probe_points"]
                if sample_pack is not None and "probe_points" in sample_pack
                else (
                    uniform_fluid_points if certified_stokes else fluid_points
                )(
                    cfg,
                    cfg.validation_points,
                    cfg.probe_beta,
                    cfg.probe_alpha_deg,
                    cfg.seed + 1_500_000,
                )
            )
            probe = peel_step(
                cfg, model, probe_points, cfg.probe_beta, cfg.probe_alpha_deg,
                False, reference_model=reference_model, full_diagnostics=True,
            )
            row.update(
                {
                    "probe_loss": probe["loss"],
                    "probe_mean_energy": probe["mean_energy"],
                    "probe_max_energy": probe["max_energy"],
                    "probe_mean_projection_ratio": probe[
                        "mean_projection_ratio"
                    ],
                    "probe_orth_error": probe["orth_error"],
                    "probe_rank": probe["effective_raw_rank"],
                }
            )
        history.append(row)
        final_epoch = epoch
        training_state = {
            "current_learning_rate": current_learning_rate,
            "best_validation_epoch": best_validation_epoch,
            "lr_reference_loss": lr_reference_loss,
            "lr_reference_epoch": lr_reference_epoch,
            "lr_reductions": lr_reductions,
            "stokes_construction_quadrature": (
                current_construction_quadrature if certified_stokes else None
            ),
            "best_certified_count": best_certified_count,
            "last_certified_count": last_certified_count,
        }
        if (
            args.self_test_epochs <= 0
            and (epoch == 1 or epoch % cfg.checkpoint_every == 0)
        ):
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config_dict, epoch, best_validation, training_state,
            )
            save_history(args.run_dir, history)
        if (
            args.self_test_epochs <= 0
            and best_validation_epoch > 0
            and epoch - best_validation_epoch
            >= cfg.early_stopping_patience_epochs
        ):
            stopped_early = True
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config_dict, epoch, best_validation, training_state,
            )
            save_history(args.run_dir, history)
            append_log(
                args.run_dir,
                f"early stopping epoch={epoch:04d} "
                f"best_epoch={best_validation_epoch:04d} "
                f"patience={cfg.early_stopping_patience_epochs}",
            )
            break
    if args.self_test_epochs > 0:
        append_log(
            args.run_dir,
            f"hardcoded self-test complete epochs={args.self_test_epochs}",
        )
        return
    frozen = args.run_dir / "checkpoints" / "hardcoded_sss_frozen.npz"
    save_neutral(
        frozen,
        model,
        config_dict,
        final_epoch,
        best_validation,
        {
            "current_learning_rate": current_learning_rate,
            "best_validation_epoch": best_validation_epoch,
            "lr_reference_loss": lr_reference_loss,
            "lr_reference_epoch": lr_reference_epoch,
            "lr_reductions": lr_reductions,
            "stokes_construction_quadrature": (
                current_construction_quadrature if certified_stokes else None
            ),
            "best_certified_count": best_certified_count,
            "last_certified_count": last_certified_count,
            "stopped_early": stopped_early,
        },
    )
    save_history(args.run_dir, history)
    append_log(
        args.run_dir,
        f"training complete epoch={final_epoch} stopped_early={stopped_early} "
        f"frozen={frozen}",
    )


if __name__ == "__main__":
    main()
