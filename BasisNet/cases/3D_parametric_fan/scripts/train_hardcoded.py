"""Train the 3D four-blade fan basis without any ML framework or autodiff."""

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

import cupy as cp
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (  # noqa: E402
    HardcodedFanModel,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    initialize_arrays,
    peel_and_adjoint,
    raw_fields,
    train_step,
)

if "torch" in sys.modules:
    raise RuntimeError(
        "PyTorch was imported transitively; refusing framework-free training"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CASE_ROOT / "configs" / "k1024_pilot_100.json",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="reset the manual Adam state when resuming inference-only weights",
    )
    parser.add_argument("--modes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--points", type=int)
    parser.add_argument("--mode-forward-batch", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def save_checkpoint(
    run_dir: Path,
    model: HardcodedFanModel,
    config: dict[str, Any],
    epoch: int,
    learning_rate: float,
    best_validation: float,
    best_validation_epoch: int,
    lr_plateau_count: int,
    early_stopping_count: int,
    history: list[dict[str, Any]],
    final_validation: dict[str, Any] | None,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    full_path = checkpoint_dir / f"epoch_{epoch:04d}_full.npz"
    inference_path = checkpoint_dir / f"epoch_{epoch:04d}_inference.npz"
    atomic_npz(full_path, model.arrays(include_adam=True))
    atomic_npz(inference_path, model.arrays(include_adam=False))
    metadata = {
        "format": "eigenfluid_fan3d_hardcoded_v1",
        "epoch": epoch,
        "learning_rate": learning_rate,
        "best_validation": best_validation,
        "best_validation_epoch": best_validation_epoch,
        "lr_plateau_count": lr_plateau_count,
        "early_stopping_count": early_stopping_count,
        "config": config,
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
        "network_forward": "explicit stacked SIREN matrix products",
        "network_backward": "explicit handwritten reverse pass",
        "optimizer": "handwritten independent Adam state per mode",
        "spatial_derivatives": "19-point commuting centered differences",
        "solenoidal_representation": "antisymmetric tensor potential",
        "sss": "dynamic raw-Rayleigh order; detached gradient-causal peeling",
        "theta_policy": "excluded from network; exact SO(2) rotation at inference",
        "blade_count": 4,
        "final_validation": final_validation,
    }
    atomic_json(full_path.with_suffix(".json"), metadata)
    atomic_json(inference_path.with_suffix(".json"), metadata)
    atomic_npz(checkpoint_dir / "latest_full.npz", model.arrays(include_adam=True))
    atomic_npz(
        checkpoint_dir / "latest_inference.npz",
        model.arrays(include_adam=False),
    )
    atomic_json(checkpoint_dir / "latest_full.json", metadata)
    atomic_json(checkpoint_dir / "latest_inference.json", metadata)
    save_history(run_dir, history)


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    atomic_json(data_dir / "training_history.json", history)
    if not history:
        return
    keys = list(history[0])
    path = data_dir / "training_history.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary, path)


def serializable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "q_velocity",
        "coefficient_matrix",
        "adjoint_velocity",
        "adjoint_gradient",
        "mode_energies",
        "mode_energies_by_network",
        "mode_order",
    }
    return {
        key: (
            value.item()
            if isinstance(value, np.generic)
            else value
        )
        for key, value in result.items()
        if key not in excluded
    }


def validate(
    cfg: Any,
    model: HardcodedFanModel,
    epoch: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    points = fluid_points(
        cfg,
        cfg.validation_points,
        parameters,
        cfg.seed + 9_000_000,
    )
    raw = raw_fields(cfg, model, points, parameters)
    result = peel_and_adjoint(
        cfg,
        raw,
        len(points),
        need_adjoint=False,
        need_coefficients=False,
    )
    summary = serializable_metrics(result)
    detail = {
        "epoch": epoch,
        "parameters_normalized": parameters.tolist(),
        "mode_order": result["mode_order"].tolist(),
        "mode_energies": result["mode_energies"].tolist(),
        "mode_energies_by_network": result[
            "mode_energies_by_network"
        ].tolist(),
        **summary,
    }
    return summary, detail


def load_resume(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def assert_resume_config_compatible(
    checkpoint_config: dict[str, Any],
    requested_config: dict[str, Any],
) -> None:
    """Allow only operational continuation settings to change on resume."""
    mutable_keys = {
        "epochs",
        "checkpoint_every",
        "result_root",
        "lr_patience_validations",
        "early_stopping_patience_validations",
        "lr_scheduler",
        "one_cycle_start_epoch",
        "one_cycle_max_lr",
        "one_cycle_div_factor",
        "one_cycle_final_div_factor",
        "one_cycle_pct_start",
        "one_cycle_anneal_strategy",
    }
    checkpoint_contract = {
        key: value
        for key, value in checkpoint_config.items()
        if key not in mutable_keys
    }
    requested_contract = {
        key: value
        for key, value in requested_config.items()
        if key not in mutable_keys
    }
    if checkpoint_contract != requested_contract:
        changed = sorted(
            key
            for key in checkpoint_contract.keys() | requested_contract.keys()
            if checkpoint_contract.get(key) != requested_contract.get(key)
        )
        raise ValueError(
            "resume config changes the training contract: "
            + ", ".join(changed)
        )


def scheduled_learning_rate(cfg: Any, epoch: int) -> float | None:
    """Framework-free cosine OneCycle schedule evaluated once per epoch."""
    if getattr(cfg, "lr_scheduler", "plateau") != "one_cycle":
        return None
    if getattr(cfg, "one_cycle_anneal_strategy", "cos") != "cos":
        raise ValueError("only cosine OneCycle annealing is supported")
    first = int(cfg.one_cycle_start_epoch)
    last = int(cfg.epochs)
    if not first <= epoch <= last:
        raise ValueError(
            f"epoch {epoch} is outside OneCycle interval [{first}, {last}]"
        )
    intervals = max(1, last - first)
    progress = (epoch - first) / intervals
    warm_fraction = float(cfg.one_cycle_pct_start)
    maximum = float(cfg.one_cycle_max_lr)
    initial = maximum / float(cfg.one_cycle_div_factor)
    final = initial / float(cfg.one_cycle_final_div_factor)
    if progress <= warm_fraction:
        phase = progress / max(warm_fraction, 1.0e-12)
        return initial + 0.5 * (maximum - initial) * (
            1.0 - math.cos(math.pi * phase)
        )
    phase = (progress - warm_fraction) / max(
        1.0 - warm_fraction, 1.0e-12
    )
    return final + 0.5 * (maximum - final) * (
        1.0 + math.cos(math.pi * phase)
    )


def reconstruct_validation_state(
    history: list[dict[str, Any]],
    best_validation: float,
) -> tuple[int, int]:
    """Recover the best epoch and consecutive non-improving validations."""
    validation_rows = [
        row
        for row in history
        if np.isfinite(float(row.get("validation_loss", float("nan"))))
    ]
    if not validation_rows:
        return 0, 0
    best_row = min(
        validation_rows,
        key=lambda row: abs(float(row["validation_loss"]) - best_validation),
    )
    best_epoch = int(best_row["epoch"])
    count = sum(int(row["epoch"]) > best_epoch for row in validation_rows)
    return best_epoch, count


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in (
        ("modes", args.modes),
        ("epochs", args.epochs),
        ("points_per_step", args.points),
        ("mode_forward_batch", args.mode_forward_batch),
    ):
        if value is not None:
            config[key] = value
    if config["blade_count"] != 4:
        raise ValueError("this benchmark is contractually fixed to four blades")
    if config["parameter_dim"] != 3:
        raise ValueError("network input must contain exactly beta(r), c(r), and t")
    if 3 * config["points_per_step"] < config["modes"]:
        raise ValueError(
            "quadrature has fewer velocity degrees of freedom than requested modes"
        )
    cfg = cfg_namespace(config)
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (CASE_ROOT.parents[2] / config["result_root"]).resolve()
    )
    if args.resume is None:
        run_dir.mkdir(parents=True, exist_ok=False)
        arrays = initialize_arrays(cfg)
        start_epoch = 1
        history: list[dict[str, Any]] = []
        current_learning_rate = float(cfg.learning_rate)
        best_validation = float("inf")
        best_validation_epoch = 0
        plateau_count = 0
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata, arrays = load_resume(args.resume.resolve())
        assert_resume_config_compatible(metadata["config"], config)
        start_epoch = int(metadata["epoch"]) + 1
        if cfg.epochs < start_epoch:
            raise ValueError(
                f"target epochs {cfg.epochs} precede resume epoch "
                f"{metadata['epoch']}"
            )
        if args.reset_optimizer:
            initialized = initialize_arrays(cfg)
            initialized.update(arrays)
            arrays = initialized
        current_learning_rate = float(
            metadata.get("learning_rate", cfg.learning_rate)
        )
        best_validation = float(
            metadata.get(
                "best_validation",
                metadata.get("validation", {}).get("loss", float("inf")),
            )
        )
        history_path = run_dir / "data" / "training_history.json"
        history = (
            json.loads(history_path.read_text(encoding="utf-8"))
            if history_path.exists()
            else []
        )
        resume_epoch = int(metadata["epoch"])
        history = [
            row for row in history
            if int(row["epoch"]) <= resume_epoch
        ]
        recovered_best_epoch, recovered_count = reconstruct_validation_state(
            history,
            best_validation,
        )
        best_validation_epoch = int(
            metadata.get("best_validation_epoch")
            or recovered_best_epoch
            or metadata["epoch"]
        )
        plateau_count = int(
            metadata.get(
                "lr_plateau_count",
                recovered_count % cfg.lr_patience_validations,
            )
        )
        early_stopping_count = int(
            metadata.get("early_stopping_count", recovered_count)
        )
    if args.resume is None:
        early_stopping_count = 0
    model = HardcodedFanModel(cfg, arrays)
    free, total = cp.cuda.runtime.memGetInfo()
    protocol = {
        "case": "3D_parametric_fan",
        "network_parameters": ["beta(r)", "c(r)", "t"],
        "known_state": "theta",
        "theta_enters_network": False,
        "theta_operator": "exact SO(2) coordinate/vector rotation",
        "modes": cfg.modes,
        "epochs": cfg.epochs,
        "blade_count": cfg.blade_count,
        "domain": {
            "type": "cylinder",
            "radius": cfg.farfield_radius,
            "radius_over_tip_radius": cfg.farfield_radius / cfg.tip_radius,
            "inlet": "x=xmin, homogeneous perturbation",
            "outlet": "x=xmax, natural/open",
            "lateral": "r=2Rtip, homogeneous perturbation/far-field",
        },
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
        "frameworks_prohibited": ["PyTorch", "TensorFlow", "JAX"],
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "gpu_total_bytes": int(total),
        "config": config,
    }
    atomic_json(run_dir / "protocol.json", protocol)
    append_log(
        run_dir,
        f"start manual-cupy modes={cfg.modes} points={cfg.points_per_step} "
        f"epochs={cfg.epochs} start_epoch={start_epoch} "
        f"gpu_free={free / 2**30:.2f}GiB",
    )
    atomic_json(
        run_dir / "completion.json",
        {
            "complete": False,
            "epochs_requested": cfg.epochs,
            "resume_epoch": start_epoch - 1,
            "best_validation": best_validation,
            "best_validation_epoch": best_validation_epoch,
        },
    )
    started = time.perf_counter()
    final_detail: dict[str, Any] | None = None
    last_epoch = start_epoch - 1
    stopped_early = False
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        scheduled = scheduled_learning_rate(cfg, epoch)
        if scheduled is not None:
            current_learning_rate = scheduled
        parameters = continuous_parameters(cfg, epoch)
        points = fluid_points(
            cfg,
            cfg.points_per_step,
            parameters,
            cfg.seed + 100_000 + epoch,
        )
        train_result = train_step(
            cfg,
            model,
            points,
            parameters,
            current_learning_rate,
        )
        cp.cuda.Stream.null.synchronize()
        validation_summary: dict[str, Any] = {}
        validation_loss = float("nan")
        should_validate = (
            not args.skip_validation
            and (
                epoch % cfg.validate_every == 0
                or epoch == cfg.epochs
            )
        )
        if should_validate:
            validation_summary, final_detail = validate(cfg, model, epoch)
            validation_loss = float(validation_summary["loss"])
            if validation_loss < best_validation - cfg.early_stopping_min_delta:
                best_validation = validation_loss
                best_validation_epoch = epoch
                plateau_count = 0
                early_stopping_count = 0
                atomic_npz(
                    run_dir / "checkpoints" / "best_inference.npz",
                    model.arrays(include_adam=False),
                )
                atomic_json(
                    run_dir / "checkpoints" / "best_inference.json",
                    {
                        "epoch": epoch,
                        "validation": final_detail,
                        "config": config,
                        "training_uses_pytorch": False,
                    },
                )
            else:
                plateau_count += 1
                early_stopping_count += 1
                if (
                    getattr(cfg, "lr_scheduler", "plateau") != "one_cycle"
                    and plateau_count >= cfg.lr_patience_validations
                ):
                    current_learning_rate = max(
                        cfg.min_learning_rate,
                        current_learning_rate * cfg.lr_reduce_factor,
                    )
                    plateau_count = 0
        should_stop_early = (
            should_validate
            and early_stopping_count
            >= cfg.early_stopping_patience_validations
        )
        elapsed = time.perf_counter() - epoch_started
        beta, chord, thickness = (
            cfg.beta_tip_min_deg
            + parameters[0] * (cfg.beta_tip_max_deg - cfg.beta_tip_min_deg),
            cfg.chord_tip_min
            + parameters[1] * (cfg.chord_tip_max - cfg.chord_tip_min),
            cfg.thickness_min
            + parameters[2] * (cfg.thickness_max - cfg.thickness_min),
        )
        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "beta_tip_deg": float(beta),
            "chord_tip": float(chord),
            "thickness_ratio": float(thickness),
            "train_loss": float(train_result["loss"]),
            "train_mean_energy": float(train_result["mean_energy"]),
            "train_max_energy": float(train_result["max_energy"]),
            "train_orth_error": float(train_result["orth_error"]),
            "train_effective_rank": int(train_result["effective_raw_rank"]),
            "validation_loss": validation_loss,
            "validation_orth_error": float(
                validation_summary.get("orth_error", float("nan"))
            ),
            "learning_rate": current_learning_rate,
            "gpu_pool_used_gib": cp.get_default_memory_pool().used_bytes() / 2**30,
        }
        history.append(row)
        last_epoch = epoch
        save_history(run_dir, history)
        append_log(
            run_dir,
            f"epoch={epoch:04d}/{cfg.epochs} sec={elapsed:.2f} "
            f"loss={row['train_loss']:.6e} rank={row['train_effective_rank']} "
            f"orth={row['train_orth_error']:.3e} "
            f"val={validation_loss:.6e}",
        )
        if (
            epoch % cfg.checkpoint_every == 0
            or epoch == cfg.epochs
            or should_stop_early
        ):
            save_checkpoint(
                run_dir,
                model,
                config,
                epoch,
                current_learning_rate,
                best_validation,
                best_validation_epoch,
                plateau_count,
                early_stopping_count,
                history,
                final_detail,
            )
        if should_stop_early:
            stopped_early = True
            append_log(
                run_dir,
                f"early stopping epoch={epoch:04d} "
                f"best_epoch={best_validation_epoch:04d} "
                f"patience_validations="
                f"{cfg.early_stopping_patience_validations}",
            )
            break
    atomic_json(
        run_dir / "completion.json",
        {
            "complete": True,
            "epochs_requested": cfg.epochs,
            "epochs_completed": last_epoch,
            "stopped_early": stopped_early,
            "wall_seconds": time.perf_counter() - started,
            "best_validation": best_validation,
            "best_validation_epoch": best_validation_epoch,
            "final_validation": final_detail,
        },
    )
    append_log(
        run_dir,
        f"complete epochs={last_epoch}/{cfg.epochs} "
        f"stopped_early={stopped_early} "
        f"wall={time.perf_counter() - started:.1f}s",
    )


if __name__ == "__main__":
    main()
