#!/usr/bin/env python3
"""Train and visualize the free-boundary raw-xy K=1024 vortex width sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parent
BASE_CONFIG = (
    CASE_ROOT / "configs/k1024_free_rawxy_potential_jet1_depth5_width032_e2000.json"
)
WIDTHS = (32, 64, 128, 256, 512)
REQUESTED_WIDTHS = (32, 64)
STOP_AFTER_WIDTH = 64
MODE_BATCHES = {32: 32, 64: 16, 128: 8, 256: 2, 512: 1}
MODES = 1024
TARGET_EPOCH = 2000
CHECKPOINT_EPOCHS = tuple(range(200, TARGET_EPOCH + 1, 200))
SWEEP_ROOT = (
    CASE_ROOT
    / "outputs/eigenmodes_k1024_free_rawxy_potential_jet1_depth5_width_sweep_e2000"
)
CONFIG_ROOT = SWEEP_ROOT / "configs"
ARM_ROOT = SWEEP_ROOT / "arms"
STATE_PATH = SWEEP_ROOT / "sweep_state.json"
MANIFEST_PATH = SWEEP_ROOT / "sweep_manifest.json"
PID_PATH = SWEEP_ROOT / "runner_pid.json"
GT_PATH = CASE_ROOT / "outputs/gt_128_re350_l0p90_t6pi/vortex_merger_gt.h5"


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_path(width: int) -> Path:
    return CONFIG_ROOT / f"k1024_depth5_width{width:03d}_e2000.json"


def result_root(width: int) -> Path:
    return ARM_ROOT / f"width_{width:03d}"


def run_dir(width: int) -> Path:
    return result_root(width) / "run_001"


def visualization_dir(width: int) -> Path:
    return result_root(width) / "visualizations"


def rollout_dir(width: int) -> Path:
    return result_root(width) / "witt_rollout"


def update_state(**updates: Any) -> dict[str, Any]:
    state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    state["updated_unix_ns"] = time.time_ns()
    write_json(STATE_PATH, state)
    return state


def update_manifest(**updates: Any) -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    manifest.update(updates)
    manifest["updated_at"] = now()
    manifest["updated_unix_ns"] = time.time_ns()
    write_json(MANIFEST_PATH, manifest)
    return manifest


def set_arm_manifest(width: int, **updates: Any) -> None:
    manifest = read_json(MANIFEST_PATH)
    for arm in manifest["arms"]:
        if int(arm["hidden_width"]) == width:
            arm.update(updates)
            arm["updated_at"] = now()
            arm["updated_unix_ns"] = time.time_ns()
            break
    else:
        raise KeyError(f"manifest missing width={width}")
    manifest["updated_at"] = now()
    manifest["updated_unix_ns"] = time.time_ns()
    write_json(MANIFEST_PATH, manifest)


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = [
            item.strip()
            for item in subprocess.check_output(command, text=True).strip().split(",")
        ]
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "name": fields[0] if fields else None,
        "uuid": fields[1] if len(fields) > 1 else None,
        "driver_version": fields[2] if len(fields) > 2 else None,
        "memory_total_mib": int(fields[3]) if len(fields) > 3 else None,
    }


def create_or_validate_configs() -> dict[int, dict[str, Any]]:
    base = read_json(BASE_CONFIG)
    configs: dict[int, dict[str, Any]] = {}
    for width in WIDTHS:
        relative_root = result_root(width).resolve().relative_to(
            WORKTREE_ROOT.resolve()
        ).as_posix()
        expected = {
            **base,
            "modes": MODES,
            "hidden": width,
            "hidden_layers": 4,
            "mode_forward_batch": MODE_BATCHES[width],
            "mode_backward_batch": MODE_BATCHES[width],
            "checkpoint_every": 100,
            "archive_checkpoint_every": 200,
            "result_root": relative_root,
        }
        path = config_path(width)
        if path.exists():
            if read_json(path) != expected:
                raise RuntimeError(f"existing width={width} config is not exact")
        else:
            write_json(path, expected)
        differences = {
            key: (base.get(key), expected.get(key))
            for key in sorted(set(base) | set(expected))
            if base.get(key) != expected.get(key)
        }
        permitted = {
            "modes",
            "hidden",
            "mode_forward_batch",
            "mode_backward_batch",
            "archive_checkpoint_every",
            "result_root",
        }
        if not set(differences).issubset(permitted):
            raise RuntimeError(
                f"width={width} has unexpected config differences: {differences}"
            )
        if int(expected["hidden_layers"]) + 1 != 5:
            raise RuntimeError("architecture is not five total affine layers")
        configs[width] = expected
    return configs


def ensure_manifest(configs: dict[int, dict[str, Any]]) -> None:
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
        if tuple(int(item) for item in manifest["strict_order"]) != WIDTHS:
            raise RuntimeError("existing sweep manifest order mismatch")
        return
    created_at = now()
    created_ns = time.time_ns()
    write_json(
        MANIFEST_PATH,
        {
            "experiment": "vortex_merger_k1024_free_rawxy_potential_jet1_depth5_width_sweep_e2000",
            "status": "initialized",
            "created_at": created_at,
            "created_unix_ns": created_ns,
            "strict_order": list(WIDTHS),
            "modes": MODES,
            "target_epoch": TARGET_EPOCH,
            "architecture": {
                "hidden_layers": 4,
                "total_affine_layers": 5,
                "swept_hidden_widths": list(WIDTHS),
            },
            "protocol": {
                "case": "2D_Vortex_Merger",
                "domain": "[0,2pi]x[0,2pi]",
                "boundary_condition": "free on all four sides",
                "boundary_realization": "raw physical x,y; identity envelope; no seam constraint",
                "positional_encoding": False,
                "mean_zero_streamfunction_gauge": True,
                "basis_objective": "Potential-Rayleigh",
                "rayleigh_form": "integral(|grad psi|^2) / integral(psi^2)",
                "spatial_jet": "analytic Jet1: value,x,y",
                "causal_sort": "stable geometry-Rayleigh on GPU",
                "peeling": "two-pass causal modified Gram-Schmidt",
                "post_training_order": "vorticity modes, GT-frame-0 reconstruction, de Witt rollout, then next width",
                "rollout": "de Witt vorticity-Galerkin over every one of the 61 GT frames",
                "error_metric": "plain elementwise MSE; first, last, and all-frame mean",
                "velocity_visualization": "streamlines with arrows; never speed magnitude",
            },
            "base_config": str(BASE_CONFIG.resolve()),
            "base_config_sha256": sha256(BASE_CONFIG),
            "numerically_neutral_batch_schedule": MODE_BATCHES,
            "arms": [
                {
                    "hidden_width": width,
                    "status": "pending",
                    "visualization_status": "pending",
                    "rollout_status": "pending",
                    "config": str(config_path(width).resolve()),
                    "config_sha256": sha256(config_path(width)),
                    "result_root": str(result_root(width).resolve()),
                    "run_dir": str(run_dir(width).resolve()),
                    "visualization_dir": str(visualization_dir(width).resolve()),
                    "rollout_dir": str(rollout_dir(width).resolve()),
                    "archive_epochs": list(CHECKPOINT_EPOCHS),
                }
                for width in WIDTHS
            ],
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "gpu": gpu_snapshot(),
            },
        },
    )
    update_state(
        status="initialized",
        created_at=created_at,
        created_unix_ns=created_ns,
        sweep_root=str(SWEEP_ROOT.resolve()),
        planned_widths=list(WIDTHS),
        completed_widths=[],
        visualized_widths=[],
        rolled_out_widths=[],
        current_width=None,
    )


def other_gpu_training_processes() -> list[dict[str, Any]]:
    own_pid = os.getpid()
    blockers: list[dict[str, Any]] = []
    markers = (
        "train_free_hardcoded.py",
        "train_metaball.py",
        "evaluate_original_k64_to_k1024_vs_laplacian.py",
    )
    for process in psutil.process_iter(("pid", "name", "cmdline", "create_time")):
        if process.pid == own_pid:
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if str(process.info.get("name") or "").lower() not in {
            "python.exe",
            "pythonw.exe",
        }:
            continue
        if any(marker in command.lower() for marker in markers):
            blockers.append(
                {
                    "pid": int(process.pid),
                    "created_at_unix": float(process.info.get("create_time") or 0.0),
                    "command": command,
                }
            )
    return blockers


def wait_for_gpu() -> None:
    while True:
        blockers = other_gpu_training_processes()
        if not blockers:
            update_state(status="gpu_released", blockers=[])
            return
        update_state(status="waiting_for_gpu", blockers=blockers)
        time.sleep(30.0)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def ensure_inputs(width: int) -> tuple[Path, Path, float]:
    inputs = result_root(width) / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    metadata_path = initial.with_suffix(".json")
    expected_config = read_json(config_path(width))
    needs_preparation = not initial.exists() or not metadata_path.exists()
    if not needs_preparation:
        metadata = read_json(metadata_path)
        needs_preparation = (
            metadata.get("format") != "eigenfluid_vortex_merger_free_rawxy_jet1_v1"
            or metadata.get("config") != expected_config
        )
    timer = time.perf_counter_ns()
    if needs_preparation:
        update_state(
            status="preparing_inputs",
            current_width=width,
            preparation_started_at=now(),
        )
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_free_hardcoded_run.py"),
                "--config",
                str(config_path(width)),
                "--output-dir",
                str(inputs),
            ]
        )
    preparation_seconds = (time.perf_counter_ns() - timer) / 1.0e9
    validation = inputs / "fixed_validation_iid_uniform_v1.npz"
    if not initial.exists() or not metadata_path.exists() or not validation.exists():
        raise FileNotFoundError(f"width={width} epoch-0 inputs are incomplete")
    metadata = read_json(metadata_path)
    config = metadata["config"]
    if int(metadata["epoch"]) != 0 or "jet1" not in str(
        metadata.get("spatial_ad", "")
    ).lower():
        raise RuntimeError(f"width={width} is not a fresh Jet1 epoch-0 checkpoint")
    if int(config["modes"]) != MODES or int(config["hidden"]) != width:
        raise RuntimeError(f"width={width} epoch-0 architecture mismatch")
    required_protocol = {
        "geometry_mode": "free_box",
        "envelope_mode": "free_identity",
        "input_encoding": "raw_xy",
        "input_features": 2,
        "basis_objective": "potential_rayleigh",
        "spatial_derivative_order": 1,
        "causal_sorting": "geometry_rayleigh",
        "reorthogonalization_passes": 2,
        "archive_checkpoint_every": 200,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required_protocol.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"width={width} protocol mismatch: {mismatches}")
    validation_hash = sha256(validation)
    manifest = read_json(MANIFEST_PATH)
    established_hashes = {
        str(arm["validation_pack_sha256"])
        for arm in manifest["arms"]
        if arm.get("validation_pack_sha256")
    }
    if established_hashes and validation_hash not in established_hashes:
        raise RuntimeError(f"width={width} fixed-validation hash mismatch")
    set_arm_manifest(
        width,
        preparation_wall_seconds=preparation_seconds,
        initial_checkpoint=str(initial.resolve()),
        initial_checkpoint_sha256=sha256(initial),
        validation_pack=str(validation.resolve()),
        validation_pack_sha256=validation_hash,
    )
    return initial, validation, preparation_seconds


def completion_ok(width: int) -> bool:
    completion = run_dir(width) / "completion.json"
    frozen = run_dir(width) / "checkpoints/hardcoded_sss_frozen.npz"
    if not completion.exists() or not frozen.exists():
        return False
    data = read_json(completion)
    return bool(data.get("complete")) and int(data.get("final_epoch", 0)) == TARGET_EPOCH


def visualization_ok(width: int) -> bool:
    summary_path = visualization_dir(width) / "mode_visualization_summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    figures = summary.get("figures", {})
    required = (
        "first16_png",
        "first16_pdf",
        "spectrum_png",
        "spectrum_pdf",
        "potential_first16_png",
        "potential_first16_pdf",
        "potential_spectrum_png",
        "potential_spectrum_pdf",
    )
    return bool(summary.get("complete")) and all(
        Path(figures.get(key, "")).exists() for key in required
    )


def rollout_ok(width: int) -> bool:
    summary_path = rollout_dir(width) / "witt_rollout_summary.json"
    if not summary_path.exists():
        return False
    summary = read_json(summary_path)
    files = summary.get("files", {})
    required = (
        "metrics_csv",
        "fields_npz",
        "streamlines_png",
        "streamlines_pdf",
        "mse_png",
        "mse_pdf",
        "video_mp4",
    )
    return bool(summary.get("complete")) and all(
        Path(files.get(key, "")).exists() for key in required
    )


def trim_history(resume: Path, target_run_dir: Path) -> int:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
    history_path = target_run_dir / "data/hardcoded_training_history.json"
    if history_path.exists():
        history = read_json(history_path)
        kept = [row for row in history if int(row.get("epoch", 0)) <= resume_epoch]
        if len(kept) != len(history):
            backup = history_path.with_name(
                f"history_before_resume_epoch{resume_epoch:04d}_"
                f"{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            write_json(backup, history)
            write_json(history_path, kept)
    return resume_epoch


def final_metrics(width: int) -> dict[str, Any]:
    target_run_dir = run_dir(width)
    history = read_json(target_run_dir / "data/hardcoded_training_history.json")
    if len(history) != TARGET_EPOCH or int(history[-1].get("epoch", 0)) != TARGET_EPOCH:
        raise RuntimeError(f"width={width} history is not a complete trace")
    row = history[-1]
    metrics = {
        "final_epoch": TARGET_EPOCH,
        "validation_mean": float(row["validation_loss"]),
        "validation_p95": float(row["validation_loss_p95"]),
        "validation_max": float(row["validation_loss_max"]),
        "peeled_rank": int(row["validation_peeled_rank"]),
        "orthogonality_max_abs": float(row["validation_orth_error"]),
        "learning_rate": float(row["learning_rate"]),
        "training_elapsed_seconds": float(row["training_elapsed_seconds"]),
        "spatial_jet_order_used": int(row["spatial_jet_order_used"]),
        "causal_sort_device": str(row["causal_sort_device"]),
        "optimizer_device": str(row["optimizer_device"]),
    }
    numeric = [
        value
        for key, value in metrics.items()
        if key not in {"causal_sort_device", "optimizer_device"}
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError(f"width={width} has non-finite final metrics")
    if metrics["peeled_rank"] != MODES:
        raise RuntimeError(f"width={width} peeled rank is {metrics['peeled_rank']}")
    if metrics["spatial_jet_order_used"] != 1:
        raise RuntimeError(f"width={width} did not use Jet1")
    if metrics["causal_sort_device"] != "gpu":
        raise RuntimeError(f"width={width} causal sort was not on GPU")
    if metrics["optimizer_device"] != "gpu_batched_all_modes":
        raise RuntimeError(f"width={width} optimizer was not batched on GPU")
    missing = [
        epoch
        for epoch in CHECKPOINT_EPOCHS
        if not (
            target_run_dir / f"checkpoints/hardcoded_epoch{epoch:04d}.npz"
        ).exists()
        or not (
            target_run_dir / f"checkpoints/hardcoded_epoch{epoch:04d}.json"
        ).exists()
    ]
    if missing:
        raise RuntimeError(f"width={width} missing archive epochs {missing}")
    return metrics


def completed_widths() -> list[int]:
    return [width for width in WIDTHS if completion_ok(width)]


def visualized_widths() -> list[int]:
    return [width for width in WIDTHS if visualization_ok(width)]


def rolled_out_widths() -> list[int]:
    return [width for width in WIDTHS if rollout_ok(width)]


def train_arm(width: int) -> dict[str, Any]:
    if completion_ok(width):
        metrics = final_metrics(width)
        set_arm_manifest(
            width,
            status="training_complete",
            skipped_existing_training=True,
            final_metrics=metrics,
        )
        return metrics
    initial, validation, preparation_seconds = ensure_inputs(width)
    target_run_dir = run_dir(width)
    arm = next(
        item
        for item in read_json(MANIFEST_PATH)["arms"]
        if int(item["hidden_width"]) == width
    )
    first_started_at = arm.get("training_started_at")
    first_started_ns = arm.get("training_started_unix_ns")
    for attempt in range(1, 4):
        latest = target_run_dir / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        resume_epoch = trim_history(resume, target_run_dir)
        if first_started_ns is None:
            first_started_at = now()
            first_started_ns = time.time_ns()
        attempt_started_ns = time.time_ns()
        fields = {
            "status": "training",
            "error": None,
            "attempt_failed_at": None,
            "failed_at": None,
            "attempt": attempt,
            "resumed_from_epoch": resume_epoch,
            "resume_checkpoint": str(resume.resolve()),
            "training_started_at": first_started_at,
            "training_started_unix_ns": first_started_ns,
            "attempt_started_at": now(),
            "attempt_started_unix_ns": attempt_started_ns,
            "preparation_wall_seconds": preparation_seconds,
        }
        set_arm_manifest(width, **fields)
        update_state(
            current_width=width,
            completed_widths=completed_widths(),
            visualized_widths=visualized_widths(),
            **fields,
        )
        try:
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts/train_free_hardcoded.py"),
                    "--resume",
                    str(resume),
                    "--run-dir",
                    str(target_run_dir),
                    "--epochs",
                    str(TARGET_EPOCH),
                    "--validation-pack",
                    str(validation),
                ]
            )
            if not completion_ok(width):
                raise RuntimeError(f"width={width} trainer exited without completion")
            metrics = final_metrics(width)
            completed_ns = time.time_ns()
            metrics.update(
                {
                    "training_process_wall_seconds": (
                        completed_ns - int(first_started_ns)
                    )
                    / 1.0e9,
                    "completed_at": now(),
                    "completed_unix_ns": completed_ns,
                }
            )
            set_arm_manifest(width, status="training_complete", final_metrics=metrics)
            update_state(
                status="training_complete",
                current_width=width,
                completed_widths=completed_widths(),
                visualized_widths=visualized_widths(),
                last_completed_metrics=metrics,
            )
            print(f"DONE training width={width}", flush=True)
            return metrics
        except Exception as error:
            status = "retrying_training" if attempt < 3 else "failed"
            details = {
                "status": status,
                "attempt": attempt,
                "error": repr(error),
                "attempt_failed_at": now(),
                "attempt_wall_seconds": (
                    time.time_ns() - attempt_started_ns
                )
                / 1.0e9,
            }
            set_arm_manifest(width, **details)
            update_state(current_width=width, **details)
            if attempt == 3:
                raise
            time.sleep(30.0)
    raise AssertionError("unreachable")


def visualize_arm(width: int) -> dict[str, Any]:
    if visualization_ok(width):
        summary = read_json(
            visualization_dir(width) / "mode_visualization_summary.json"
        )
        set_arm_manifest(
            width,
            status="modes_visualized",
            visualization_status="complete",
            skipped_existing_visualization=True,
            visualization=summary,
        )
        return summary
    checkpoint = run_dir(width) / "checkpoints/hardcoded_sss_frozen.npz"
    set_arm_manifest(width, visualization_status="running")
    update_state(
        status="visualizing",
        current_width=width,
        completed_widths=completed_widths(),
        visualized_widths=visualized_widths(),
    )
    run(
        [
            sys.executable,
            "-u",
            str(CASE_ROOT / "scripts/visualize_free_k1024_vorticity_modes.py"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(visualization_dir(width)),
        ]
    )
    if not visualization_ok(width):
        raise RuntimeError(f"width={width} visualization is incomplete")
    summary = read_json(
        visualization_dir(width) / "mode_visualization_summary.json"
    )
    set_arm_manifest(
        width,
        status="modes_visualized",
        visualization_status="complete",
        visualization_completed_at=now(),
        visualization=summary,
    )
    update_state(
        status="arm_complete",
        current_width=width,
        completed_widths=completed_widths(),
        visualized_widths=visualized_widths(),
        latest_visualization=summary["figures"],
    )
    print(f"DONE visualization width={width}", flush=True)
    return summary


def rollout_arm(width: int) -> dict[str, Any]:
    summary_path = rollout_dir(width) / "witt_rollout_summary.json"
    if rollout_ok(width):
        summary = read_json(summary_path)
        set_arm_manifest(
            width,
            status="complete",
            rollout_status="complete",
            skipped_existing_rollout=True,
            rollout=summary,
        )
        return summary
    checkpoint = run_dir(width) / "checkpoints/hardcoded_sss_frozen.npz"
    for attempt in range(1, 4):
        fields = {
            "status": "rolling_out",
            "rollout_status": "running",
            "rollout_attempt": attempt,
            "rollout_error": None,
            "rollout_started_at": now(),
        }
        set_arm_manifest(width, **fields)
        update_state(
            current_width=width,
            completed_widths=completed_widths(),
            visualized_widths=visualized_widths(),
            rolled_out_widths=rolled_out_widths(),
            **fields,
        )
        try:
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts/rollout_free_rawxy_witt.py"),
                    "--checkpoint",
                    str(checkpoint),
                    "--gt",
                    str(GT_PATH),
                    "--output-dir",
                    str(rollout_dir(width)),
                ]
            )
            if not rollout_ok(width):
                raise RuntimeError(f"width={width} de Witt rollout is incomplete")
            summary = read_json(summary_path)
            set_arm_manifest(
                width,
                status="complete",
                rollout_status="complete",
                rollout_completed_at=now(),
                rollout=summary,
            )
            update_state(
                status="arm_complete",
                current_width=width,
                completed_widths=completed_widths(),
                visualized_widths=visualized_widths(),
                rolled_out_widths=rolled_out_widths(),
                latest_rollout=summary["files"],
                latest_mse=summary["mse"],
            )
            print(f"DONE de Witt rollout width={width}", flush=True)
            return summary
        except Exception as error:
            status = "retrying_rollout" if attempt < 3 else "failed"
            details = {
                "status": status,
                "rollout_status": status,
                "rollout_attempt": attempt,
                "rollout_error": repr(error),
                "rollout_failed_at": now(),
            }
            set_arm_manifest(width, **details)
            update_state(current_width=width, **details)
            if attempt == 3:
                raise
            time.sleep(30.0)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    configs = create_or_validate_configs()
    ensure_manifest(configs)
    if args.initialize_only:
        update_state(status="initialized")
        print(f"INITIALIZED {SWEEP_ROOT}", flush=True)
        return
    write_json(
        PID_PATH,
        {
            "pid": os.getpid(),
            "started_at": now(),
            "started_unix_ns": time.time_ns(),
            "python": sys.executable,
        },
    )
    update_manifest(status="waiting_for_gpu", runner_pid=os.getpid())
    wait_for_gpu()
    started_ns = int(
        read_json(MANIFEST_PATH).get("sweep_started_unix_ns") or time.time_ns()
    )
    update_manifest(
        status="running",
        sweep_started_at=read_json(MANIFEST_PATH).get("sweep_started_at") or now(),
        sweep_started_unix_ns=started_ns,
    )
    for width in REQUESTED_WIDTHS:
        train_arm(width)
        visualize_arm(width)
        rollout_arm(width)
    completed_ns = time.time_ns()
    summary = {
        str(width): {
            "metrics": final_metrics(width),
            "visualization": read_json(
                visualization_dir(width) / "mode_visualization_summary.json"
            ),
            "witt_rollout": read_json(
                rollout_dir(width) / "witt_rollout_summary.json"
            ),
        }
        for width in REQUESTED_WIDTHS
    }
    update_manifest(
        status="complete_requested_scope",
        stopped_after_width=STOP_AFTER_WIDTH,
        completed_at=now(),
        completed_unix_ns=completed_ns,
        sweep_wall_seconds=(completed_ns - started_ns) / 1.0e9,
        final_results=summary,
    )
    update_state(
        status="complete_requested_scope",
        stopped_after_width=STOP_AFTER_WIDTH,
        current_width=None,
        completed_widths=list(REQUESTED_WIDTHS),
        visualized_widths=list(REQUESTED_WIDTHS),
        rolled_out_widths=list(REQUESTED_WIDTHS),
        completed_at=now(),
        completed_unix_ns=completed_ns,
        sweep_wall_seconds=(completed_ns - started_ns) / 1.0e9,
    )
    print(
        "DONE requested free-boundary raw-xy vortex K1024 scope through width=64",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if MANIFEST_PATH.exists():
            update_manifest(status="failed", error=repr(error), failed_at=now())
        update_state(status="failed", error=repr(error), failed_at=now())
        raise
