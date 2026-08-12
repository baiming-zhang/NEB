"""Train the continuous-Jet3 pure-vorticity Rayleigh airfoil basis.

This entry point deliberately supports one mathematical objective only.  It
uses the explicit CuPy kernels and handwritten parameter reverse pass in
``hardcoded_backend``; PyTorch and automatic differentiation are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    peel_step,
)


OBJECTIVE = "pure_vorticity_rayleigh"
FORBIDDEN_CONFIG_KEYS = {
    "rayleigh_scale",
    "hybrid_velocity_weight",
    "hybrid_vorticity_weight",
    "hybrid_velocity_rayleigh_scale",
    "hybrid_vorticity_rayleigh_scale",
    "hybrid_vorticity_mass_length_scale_chords",
    "hybrid_vorticity_difference_chords",
    "hybrid_vorticity_anchor_points",
    "hybrid_reorthogonalization_passes",
    "vorticity_regularization_weight",
    "small_vorticity_regularization_weight",
    "large_vorticity_preservation_weight",
    "peeled_subspace_vorticity_regularization_weight",
    "small_vorticity_scale_chords",
    "small_vorticity_anchor_points",
    "global_fraction",
    "airfoil_fraction",
    "wake_fraction",
    "airfoil_band_chords",
    "wake_length_chords",
    "wake_half_height_chords",
}
FP32_CONFIG_KEYS = (
    "mlp_accumulation_dtype",
    "jet_dtype",
    "parameter_backward_dtype",
    "quadrature_dtype",
    "ritz_dtype",
    "master_weight_dtype",
    "optimizer_dtype",
)
HISTORY_KEYS = [
    "epoch",
    "seconds",
    "beta",
    "alpha_deg",
    "train_loss",
    "train_vorticity_rayleigh",
    "train_mean_energy",
    "train_min_energy",
    "train_max_energy",
    "train_mean_projection_ratio",
    "train_orth_error",
    "train_rank",
    "validation_loss",
    "validation_vorticity_rayleigh",
    "validation_loss_p95",
    "validation_loss_max",
    "validation_mean_energy",
    "validation_min_energy",
    "validation_max_energy",
    "validation_mean_projection_ratio",
    "validation_orth_error",
    "validation_min_rank",
    "probe_loss",
    "probe_vorticity_rayleigh",
    "probe_orth_error",
    "probe_rank",
    "learning_rate",
    "best_validation_epoch",
    "early_stop_counter_epochs",
    "lr_plateau_counter_epochs",
    "resolved_raw_rank",
    "requested_modes",
]


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "hardcoded_train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def load_neutral(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint or sidecar is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def save_neutral(
    path: Path,
    model: HardcodedModel,
    config: dict[str, Any],
    epoch: int,
    best_validation: float,
    training_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **model.arrays())
    os.replace(temporary, path)
    metadata = {
        "format": "eigenfluid_airfoil_continuous_jet3_v1",
        "epoch": epoch,
        "best_validation": best_validation,
        "config": config,
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
        "spatial_ad": "analytic_compact_symmetric_xy_jet3",
        "parameter_backward": "handwritten_reverse_jet3",
        "training_state": training_state,
    }
    metadata_path = path.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(metadata_temporary, metadata_path)


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    target = data_dir / "hardcoded_training_history.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    with (data_dir / "hardcoded_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_KEYS)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in HISTORY_KEYS} for row in history
        )


def validate_config(config: dict[str, Any]) -> None:
    if config.get("basis_objective") != OBJECTIVE:
        raise ValueError(
            f"this trainer accepts basis_objective={OBJECTIVE!r} only; got "
            f"{config.get('basis_objective')!r}"
        )
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
    if forbidden:
        raise ValueError(f"forbidden auxiliary/scaled objective fields: {forbidden}")
    if config.get("sampling_mode") != "uniform_fluid_area_sobol":
        raise ValueError("sampling must be uniform physical-fluid-area Sobol")
    if config.get("quadrature_measure") != "physical_fluid_area":
        raise ValueError("quadrature measure must be physical fluid area")
    if config.get("geometry_mode") != "cinf_ellipse":
        raise ValueError("pure-vorticity Jet3 training requires cinf_ellipse geometry")
    if config.get("envelope_mode") != "cinf_farfield_freestream":
        raise ValueError("pure-vorticity Jet3 training requires a C-infinity envelope")
    if int(config.get("spatial_derivative_order", -1)) != 3:
        raise ValueError("analytic third spatial derivatives are required")
    if str(config.get("mlp_compute_dtype", "")).lower() != "float16":
        raise ValueError("the neural-network forward must use float16")
    wrong_precision = {
        key: config.get(key) for key in FP32_CONFIG_KEYS
        if str(config.get(key, "")).lower() != "float32"
    }
    if wrong_precision:
        raise ValueError(f"all non-network numerical paths must be float32: {wrong_precision}")
    if bool(config.get("fp32_boundary_layers", True)):
        raise ValueError("all neural-network layers, including first/output, must use FP16 shadows")
    if int(config.get("checkpoint_every", 0)) != 100:
        raise ValueError("hardcoded_latest checkpoint cadence must be 100 epochs")
    if int(config.get("archive_checkpoint_every", 0)) != 500:
        raise ValueError("permanent checkpoint cadence must be 500 epochs")


def require_pure_diagnostic(item: dict[str, Any]) -> None:
    required = {
        "loss",
        "vorticity_rayleigh",
        "mean_energy",
        "min_energy",
        "max_energy",
        "orth_error",
        "effective_raw_rank",
        "mode_energies",
    }
    missing = sorted(required.difference(item))
    if missing:
        raise KeyError(f"backend pure-vorticity diagnostics missing {missing}")
    if not math.isfinite(float(item["loss"])):
        raise FloatingPointError("non-finite pure-vorticity Rayleigh loss")
    if float(item["loss"]) != float(item["vorticity_rayleigh"]):
        raise RuntimeError("loss must be exactly the unscaled vorticity Rayleigh mean")


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        require_pure_diagnostic(item)
    losses = np.asarray([item["loss"] for item in items], dtype=np.float32)
    ranks = [int(item["effective_raw_rank"]) for item in items]
    finite_ranks = [rank for rank in ranks if rank >= 0]
    return {
        "loss": float(np.mean(losses)),
        "vorticity_rayleigh": float(np.mean(losses)),
        "loss_p95": float(np.percentile(losses, 95.0)),
        "loss_max": float(np.max(losses)),
        "mean_energy": float(np.mean([item["mean_energy"] for item in items])),
        "min_energy": float(np.min([item["min_energy"] for item in items])),
        "max_energy": float(np.max([item["max_energy"] for item in items])),
        "mean_projection_ratio": float(np.mean([
            item.get("mean_projection_ratio", float("nan")) for item in items
        ])),
        "orth_error": float(np.max([item["orth_error"] for item in items])),
        "min_rank": min(finite_ranks) if finite_ranks else -1,
        "requested_modes": int(items[0].get("requested_modes", len(items[0]["mode_energies"]))),
    }


def validate(
    cfg: Any,
    model: HardcodedModel,
    sample_pack: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples
        )
    ]
    for index, (beta, alpha) in enumerate(pairs):
        if sample_pack is None:
            points = fluid_points(
                cfg, cfg.validation_points, beta, alpha, cfg.seed + 900_000 + index
            )
        else:
            beta = float(sample_pack["validation_beta"][index])
            alpha = float(sample_pack["validation_alpha"][index])
            points = sample_pack["validation_points"][index]
        item = peel_step(
            cfg, model, points, beta, alpha, False,
            reference_model=None, full_diagnostics=True,
        )
        require_pure_diagnostic(item)
        details.append(item)
    return aggregate(details), details


def training_state(
    learning_rate: float,
    best_epoch: int,
    reference_loss: float,
    reference_epoch: int,
    reductions: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "current_learning_rate": learning_rate,
        "best_validation_epoch": best_epoch,
        "lr_reference_loss": reference_loss,
        "lr_reference_epoch": reference_epoch,
        "lr_reductions": reductions,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sample-pack", type=Path)
    parser.add_argument("--validation-pack", type=Path)
    parser.add_argument("--self-test-epochs", type=int, default=0)
    parser.add_argument("--self-test-points", type=int, default=256)
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("the hardcoded trainer must not import PyTorch")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    (args.run_dir / "data").mkdir(exist_ok=True)
    metadata, arrays = load_neutral(args.resume)
    config = dict(metadata["config"])
    if args.epochs is not None:
        if args.epochs <= int(metadata["epoch"]):
            raise ValueError("--epochs must exceed the resume epoch")
        config["epochs"] = args.epochs
    validate_config(config)
    spatial_ad = str(metadata.get("spatial_ad", ""))
    if "jet3" not in spatial_ad.lower():
        raise ValueError(f"refusing non-Jet3 checkpoint metadata: {spatial_ad!r}")

    def load_pack(path: Path | None) -> dict[str, np.ndarray] | None:
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    sample_pack = load_pack(args.sample_pack)
    validation_pack = load_pack(args.validation_pack)
    cfg = cfg_namespace(config)
    model = HardcodedModel(cfg, arrays)
    start_epoch = int(metadata["epoch"]) + 1
    stop_epoch = (
        start_epoch + args.self_test_epochs - 1
        if args.self_test_epochs > 0 else int(cfg.epochs)
    )
    history_path = args.run_dir / "data" / "hardcoded_training_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists() else []
    )
    best_validation = float(metadata.get("best_validation", float("inf")))
    restored = metadata.get("training_state", {})
    current_lr = float(restored.get("current_learning_rate", cfg.learning_rate))
    cfg.learning_rate = current_lr
    config["learning_rate"] = current_lr
    best_epoch = int(restored.get("best_validation_epoch", 0))
    lr_reference_loss = float(restored.get("lr_reference_loss", best_validation))
    lr_reference_epoch = int(restored.get("lr_reference_epoch", 0))
    lr_reductions = int(restored.get("lr_reductions", 0))

    protocol = {
        "algorithm": "fixed_order_residual_peeling_pure_vorticity_rayleigh",
        "backend": "explicit_cupy_cuda_kernels",
        "basis_objective": OBJECTIVE,
        "loss_terms": [
            {
                "name": "vorticity_rayleigh",
                "formula": "integral(|grad(omega)|^2)/integral(omega^2)",
                "coefficient": 1.0,
                "scaled": False,
            }
        ],
        "auxiliary_loss_terms": [],
        "finite_difference_used": False,
        "positional_encoding_derivatives": "analytic through third order",
        "spatial_derivatives": "compact symmetric analytic Jet3 in x,y",
        "parameter_backward": "handwritten reverse Jet3; no tensor autograd graph",
        "pytorch_imported_by_trainer": False,
        "automatic_differentiation_used": False,
        "precision": {
            "neural_network_forward": "float16 shadows with float32 accumulation",
            "spatial_jet": "float32",
            "parameter_reverse": "float32",
            "quadrature": "float32",
            "ritz_and_reorthogonalization": "float32",
            "master_weights": "float32",
            "adam_state": "float32",
        },
        "sampling": {
            "sequence": "scrambled Sobol",
            "measure": "uniform physical fluid area",
            "points_per_epoch": cfg.points_per_step,
            "importance_mixture": False,
        },
        "fixed_mode_order": True,
        "mode_sorting_used": False,
        "previous_modes_detached": True,
        "gradient_clipping": {
            "maximum_global_norm": cfg.grad_clip,
            "role": "finite-arithmetic safeguard only; not a loss term",
        },
        "checkpoint_every": cfg.checkpoint_every,
        "archive_checkpoint_every": cfg.archive_checkpoint_every,
        "early_stopping_patience_epochs": cfg.early_stopping_patience_epochs,
        "lr_scheduler": {
            "type": "validation_plateau",
            "patience_epochs": cfg.lr_scheduler_patience_epochs,
            "factor": cfg.lr_scheduler_factor,
            "minimum_learning_rate": cfg.min_learning_rate,
        },
        "reference_sample_pack": (
            str(args.sample_pack.resolve()) if args.sample_pack else None
        ),
        "fixed_validation_pack": (
            str(args.validation_pack.resolve()) if args.validation_pack else None
        ),
    }
    (args.run_dir / "hardcoded_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    append_log(
        args.run_dir,
        f"start continuous-Jet3 pure-vorticity no-PyTorch device=cuda "
        f"modes={cfg.modes} points={cfg.points_per_step} "
        f"resume_epoch={metadata['epoch']} lr={current_lr:.6e}",
    )

    started = time.perf_counter()
    stopped_early = False
    final_epoch = int(metadata["epoch"])
    for epoch in range(start_epoch, stop_epoch + 1):
        count = args.self_test_points if args.self_test_epochs > 0 else cfg.points_per_step
        if sample_pack is not None and args.self_test_epochs <= 0:
            index = epoch - 1
            if index >= len(sample_pack["train_beta"]):
                raise IndexError(f"sample pack has no entry for epoch {epoch}")
            beta = float(sample_pack["train_beta"][index])
            alpha = float(sample_pack["train_alpha"][index])
            points = sample_pack["train_points"][index]
        else:
            beta, alpha = continuous_parameters(cfg, epoch)
            points = fluid_points(cfg, count, beta, alpha, cfg.seed + epoch * 1009)
        diagnostic = peel_step(
            cfg, model, points, beta, alpha, True,
            reference_model=None, full_diagnostics=True,
        )
        require_pure_diagnostic(diagnostic)
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "beta": beta,
            "alpha_deg": alpha,
            "train_loss": diagnostic["loss"],
            "train_vorticity_rayleigh": diagnostic["vorticity_rayleigh"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_min_energy": diagnostic["min_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic.get(
                "mean_projection_ratio", float("nan")
            ),
            "train_orth_error": diagnostic["orth_error"],
            "train_rank": diagnostic["effective_raw_rank"],
            "learning_rate": current_lr,
        }
        if args.self_test_epochs > 0:
            append_log(
                args.run_dir,
                f"self-test epoch={epoch:04d} "
                f"vorticity_R={diagnostic['loss']:.6e} "
                f"orth={diagnostic['orth_error']:.2e} "
                f"rank={diagnostic['effective_raw_rank']}",
            )
        elif epoch % cfg.validate_every == 0:
            validation_source = validation_pack or sample_pack
            validation, details = validate(cfg, model, validation_source)
            row.update({
                "validation_loss": validation["loss"],
                "validation_vorticity_rayleigh": validation["vorticity_rayleigh"],
                "validation_loss_p95": validation["loss_p95"],
                "validation_loss_max": validation["loss_max"],
                "validation_mean_energy": validation["mean_energy"],
                "validation_min_energy": validation["min_energy"],
                "validation_max_energy": validation["max_energy"],
                "validation_mean_projection_ratio": validation[
                    "mean_projection_ratio"
                ],
                "validation_orth_error": validation["orth_error"],
                "validation_min_rank": validation["min_rank"],
                "validation_mode_energies": [item["mode_energies"] for item in details],
            })
            threshold = cfg.lr_scheduler_relative_threshold
            improved = validation["loss"] < best_validation * (1.0 - threshold)
            if improved:
                best_validation = validation["loss"]
                best_epoch = epoch
                lr_reference_loss = best_validation
                lr_reference_epoch = epoch
                state = training_state(
                    current_lr, best_epoch, lr_reference_loss,
                    lr_reference_epoch, lr_reductions,
                )
                save_neutral(
                    args.run_dir / "checkpoints" / "hardcoded_best_model.npz",
                    model, config, epoch, best_validation, state,
                )
            elif (
                epoch - lr_reference_epoch >= cfg.lr_scheduler_patience_epochs
                and current_lr > cfg.min_learning_rate
            ):
                current_lr = max(
                    cfg.min_learning_rate, current_lr * cfg.lr_scheduler_factor
                )
                cfg.learning_rate = current_lr
                config["learning_rate"] = current_lr
                lr_reference_loss = validation["loss"]
                lr_reference_epoch = epoch
                lr_reductions += 1
                append_log(args.run_dir, f"lr reduced epoch={epoch:04d} lr={current_lr:.6e}")
            row.update({
                "learning_rate": current_lr,
                "best_validation_epoch": best_epoch,
                "early_stop_counter_epochs": epoch - best_epoch,
                "lr_plateau_counter_epochs": epoch - lr_reference_epoch,
            })
            append_log(
                args.run_dir,
                f"epoch={epoch:04d} beta={beta:.5f} alpha={alpha:+.3f} "
                f"train_vorticity_R={diagnostic['loss']:.6e} "
                f"validation_vorticity_R={validation['loss']:.6e} "
                f"val_p95={validation['loss_p95']:.6e} lr={current_lr:.3e} "
                f"orth={validation['orth_error']:.2e} rank={validation['min_rank']}",
            )
        if (
            args.self_test_epochs <= 0 and cfg.probe_every > 0
            and epoch % cfg.probe_every == 0
        ):
            if validation_pack is not None and "probe_points" in validation_pack:
                probe_points = validation_pack["probe_points"]
            elif sample_pack is not None and "probe_points" in sample_pack:
                probe_points = sample_pack["probe_points"]
            else:
                probe_points = fluid_points(
                    cfg, cfg.validation_points, cfg.probe_beta, cfg.probe_alpha_deg,
                    cfg.seed + 1_500_000,
                )
            probe = peel_step(
                cfg, model, probe_points, cfg.probe_beta, cfg.probe_alpha_deg,
                False, reference_model=None, full_diagnostics=True,
            )
            require_pure_diagnostic(probe)
            row.update({
                "probe_loss": probe["loss"],
                "probe_vorticity_rayleigh": probe["vorticity_rayleigh"],
                "probe_orth_error": probe["orth_error"],
                "probe_rank": probe["effective_raw_rank"],
            })
        row["resolved_raw_rank"] = diagnostic["effective_raw_rank"]
        row["requested_modes"] = int(diagnostic.get("requested_modes", cfg.modes))
        history.append(row)
        final_epoch = epoch
        state = training_state(
            current_lr, best_epoch, lr_reference_loss,
            lr_reference_epoch, lr_reductions,
        )
        if args.self_test_epochs <= 0 and epoch % cfg.checkpoint_every == 0:
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config, epoch, best_validation, state,
            )
            if epoch % cfg.archive_checkpoint_every == 0:
                archive = (
                    args.run_dir / "checkpoints" / f"hardcoded_epoch{epoch:04d}.npz"
                )
                save_neutral(archive, model, config, epoch, best_validation, state)
                append_log(args.run_dir, f"archived checkpoint epoch={epoch:04d} path={archive}")
            save_history(args.run_dir, history)
        if (
            args.self_test_epochs <= 0 and best_epoch > 0
            and epoch - best_epoch >= cfg.early_stopping_patience_epochs
        ):
            stopped_early = True
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config, epoch, best_validation, state,
            )
            save_history(args.run_dir, history)
            append_log(
                args.run_dir,
                f"early stopping epoch={epoch:04d} best_epoch={best_epoch:04d}",
            )
            break

    if args.self_test_epochs > 0:
        append_log(args.run_dir, f"hardcoded self-test complete epochs={args.self_test_epochs}")
        return
    final_state = training_state(
        current_lr, best_epoch, lr_reference_loss, lr_reference_epoch,
        lr_reductions, stopped_early=stopped_early,
    )
    frozen = args.run_dir / "checkpoints" / "hardcoded_sss_frozen.npz"
    save_neutral(
        frozen, model, config, final_epoch, best_validation, final_state
    )
    save_history(args.run_dir, history)
    completion = {
        "complete": True,
        "final_epoch": final_epoch,
        "configured_epochs": int(cfg.epochs),
        "stopped_early": stopped_early,
        "best_validation": best_validation,
        "best_validation_epoch": best_epoch,
        "basis_objective": OBJECTIVE,
        "finite_difference_used": False,
        "training_uses_pytorch": False,
        "precision": {
            "neural_network": "float16 operands/shadows; float32 accumulation",
            "all_other_numerical_paths": "float32",
        },
        "frozen_checkpoint": str(frozen.resolve()),
    }
    completion_path = args.run_dir / "completion.json"
    completion_temporary = completion_path.with_suffix(".json.tmp")
    completion_temporary.write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    os.replace(completion_temporary, completion_path)
    append_log(
        args.run_dir,
        f"training complete epoch={final_epoch} stopped_early={stopped_early} "
        f"frozen={frozen}",
    )


if __name__ == "__main__":
    main()
