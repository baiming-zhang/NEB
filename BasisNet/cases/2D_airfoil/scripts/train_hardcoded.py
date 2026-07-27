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

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    peel_step,
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
        "train_mean_energy", "train_max_energy", "train_mean_projection_ratio",
        "train_orth_error", "train_rank", "validation_loss",
        "validation_mean_energy", "validation_max_energy",
        "validation_mean_projection_ratio", "validation_orth_error",
        "validation_min_rank",
        "validation_loss_p95", "validation_loss_max", "learning_rate",
        "best_validation_epoch", "early_stop_counter_epochs",
        "lr_plateau_counter_epochs",
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
        "loss_p95": float(np.percentile(case_losses, 95.0)),
        "loss_max": float(np.max(case_losses)),
        "mean_energy": float(np.mean([item["mean_energy"] for item in items])),
        "max_energy": float(np.max([item["max_energy"] for item in items])),
        "mean_projection_ratio": float(
            np.mean([item["mean_projection_ratio"] for item in items])
        ),
        "orth_error": float(np.max([item["orth_error"] for item in items])),
        "min_rank": int(np.min([item["effective_raw_rank"] for item in items])),
    }


def validate(
    cfg: Any,
    model: HardcodedModel,
    sample_pack: dict[str, np.ndarray] | None = None,
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
            points = fluid_points(
                cfg, cfg.validation_points, beta, alpha, cfg.seed + 900_000 + index
            )
        else:
            beta = float(sample_pack["validation_beta"][index])
            alpha = float(sample_pack["validation_alpha"][index])
            points = sample_pack["validation_points"][index]
        details.append(peel_step(cfg, model, points, beta, alpha, False))
    return aggregate(details), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--sample-pack", type=Path)
    parser.add_argument("--validation-pack", type=Path)
    parser.add_argument("--self-test-epochs", type=int, default=0)
    parser.add_argument("--self-test-points", type=int, default=256)
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
    cfg = cfg_namespace(config_dict)
    model = HardcodedModel(cfg, arrays)
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
        restored_state.get("current_learning_rate", cfg.learning_rate)
    )
    cfg.learning_rate = current_learning_rate
    config_dict["learning_rate"] = current_learning_rate
    best_validation_epoch = int(restored_state.get("best_validation_epoch", 0))
    lr_reference_loss = float(
        restored_state.get("lr_reference_loss", best_validation)
    )
    lr_reference_epoch = int(restored_state.get("lr_reference_epoch", 0))
    lr_reductions = int(restored_state.get("lr_reductions", 0))
    protocol = {
        "algorithm": "official_sss_fixed_order_residual_peeling_joint_beta_alpha",
        "backend": "cupy_cuda_fixed_graph",
        "mlp_compute_dtype": getattr(cfg, "mlp_compute_dtype", "float32"),
        "mode_forward_batch": getattr(cfg, "mode_forward_batch", 1),
        "fp16_shadow_weights": (
            getattr(cfg, "mlp_compute_dtype", "float32") == "float16"
        ),
        "jet2_linear_channels_fused": True,
        "mlp_accumulation_dtype": "float32",
        "hardcoded_reverse_dtype": "float32",
        "master_weight_dtype": "float32",
        "adam_state_dtype": "float32",
        "fp16_loss_scale_maximum": getattr(cfg, "fp16_loss_scale", 1.0),
        "fp16_loss_scaling": "dynamic bidirectional with overflow headroom",
        "peeling_ritz_dtype": "float64",
        "pytorch_imported_by_trainer": False,
        "automatic_differentiation_used": False,
        "spatial_derivatives": "hardcoded Jet2 propagation for x,y only",
        "parameter_backward": "hardcoded reverse Jet2 pass",
        "one_network_and_adam_per_mode": True,
        "fixed_mode_order": True,
        "mode_sorting_used": False,
        "least_squares_residual_peeling": True,
        "previous_modes_detached": True,
        "points_per_epoch": cfg.points_per_step,
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
        f"points={cfg.points_per_step} resume_epoch={metadata['epoch']} "
        f"lr={current_learning_rate:.6e}",
    )
    started = time.perf_counter()
    stopped_early = False
    final_epoch = int(metadata["epoch"])
    for epoch in range(start_epoch, stop_epoch + 1):
        count = args.self_test_points if args.self_test_epochs > 0 else cfg.points_per_step
        if sample_pack is not None and args.self_test_epochs <= 0:
            pack_index = epoch - 1
            if pack_index >= len(sample_pack["train_beta"]):
                raise IndexError(f"sample pack has no entry for epoch {epoch}")
            beta = float(sample_pack["train_beta"][pack_index])
            alpha = float(sample_pack["train_alpha"][pack_index])
            points = sample_pack["train_points"][pack_index]
        else:
            beta, alpha = continuous_parameters(cfg, epoch)
            points = fluid_points(cfg, count, beta, alpha, cfg.seed + epoch * 1009)
        diagnostic = peel_step(cfg, model, points, beta, alpha, True)
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "beta": beta,
            "alpha_deg": alpha,
            "train_loss": diagnostic["loss"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic["mean_projection_ratio"],
            "train_orth_error": diagnostic["orth_error"],
            "train_rank": diagnostic["effective_raw_rank"],
            "backend": "hardcoded_cupy_no_pytorch",
            "learning_rate": current_learning_rate,
        }
        if args.self_test_epochs > 0:
            append_log(
                args.run_dir,
                f"hardcoded-test epoch={epoch:04d} train={diagnostic['loss']:.6e} "
                f"orth={diagnostic['orth_error']:.2e} rank={diagnostic['effective_raw_rank']}",
            )
        elif epoch == 1 or epoch % cfg.validate_every == 0:
            validation_source = (
                validation_pack if validation_pack is not None else sample_pack
            )
            validation, details = validate(cfg, model, validation_source)
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
            relative_threshold = cfg.lr_scheduler_relative_threshold
            improved = float(validation["loss"]) < best_validation * (
                1.0 - relative_threshold
            )
            if improved:
                best_validation = float(validation["loss"])
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
                else fluid_points(
                    cfg,
                    cfg.validation_points,
                    cfg.probe_beta,
                    cfg.probe_alpha_deg,
                    cfg.seed + 1_500_000,
                )
            )
            probe = peel_step(
                cfg, model, probe_points, cfg.probe_beta, cfg.probe_alpha_deg, False
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
