"""Run fresh Jet1/GPU airfoil K128--K2048 trainings sequentially."""

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
WORKTREE_ROOT = CASE_ROOT.parents[1]
BASE_CONFIG = (
    CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_jet1_gpu_2000.json"
)
MODE_COUNTS = (128, 256, 512, 1024, 2048)
TARGET_EPOCH = 2000
SEQUENCE_ROOT = (
    CASE_ROOT
    / "results/siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
)
CONFIG_ROOT = SEQUENCE_ROOT / "configs"
ARM_ROOT = SEQUENCE_ROOT / "arms"
STATE_PATH = SEQUENCE_ROOT / "sequence_state.json"
MANIFEST_PATH = SEQUENCE_ROOT / "sequence_manifest.json"
PID_PATH = SEQUENCE_ROOT / "runner_pid.json"
EXPECTED_VALIDATION_SHA256 = (
    "4b723fe1f504bd6f197c2b134a25ad9696817b427e098a11df0f85896c307162"
)


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


def config_path(modes: int) -> Path:
    return CONFIG_ROOT / f"k{modes}_airfoil_jet1_gpu_e2000.json"


def result_root(modes: int) -> Path:
    return ARM_ROOT / f"k{modes:04d}"


def run_dir(modes: int) -> Path:
    return result_root(modes) / "run_001"


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


def set_arm_manifest(modes: int, **updates: Any) -> None:
    manifest = read_json(MANIFEST_PATH)
    for arm in manifest["arms"]:
        if int(arm["modes"]) == modes:
            arm.update(updates)
            arm["updated_at"] = now()
            arm["updated_unix_ns"] = time.time_ns()
            break
    else:
        raise KeyError(f"manifest missing K{modes}")
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
    for modes in MODE_COUNTS:
        relative_root = result_root(modes).resolve().relative_to(
            WORKTREE_ROOT.resolve()
        ).as_posix()
        expected = {**base, "modes": modes, "result_root": relative_root}
        path = config_path(modes)
        if path.exists():
            actual = read_json(path)
            if actual != expected:
                raise RuntimeError(f"existing generated K{modes} config is not exact")
        else:
            write_json(path, expected)
        differences = {
            key: (base.get(key), expected.get(key))
            for key in sorted(set(base) | set(expected))
            if base.get(key) != expected.get(key)
        }
        if set(differences) != {"modes", "result_root"}:
            raise RuntimeError(f"K{modes} has unexpected config differences: {differences}")
        configs[modes] = expected
    return configs


def ensure_manifest(configs: dict[int, dict[str, Any]]) -> None:
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
        if tuple(int(item) for item in manifest["strict_order"]) != MODE_COUNTS:
            raise RuntimeError("existing sequence manifest order mismatch")
        return
    created_at = now()
    created_ns = time.time_ns()
    manifest = {
        "experiment": "airfoil_original_jet1_gpu_k128_to_k2048_e2000",
        "status": "initialized",
        "created_at": created_at,
        "created_unix_ns": created_ns,
        "strict_order": list(MODE_COUNTS),
        "fresh_epoch0_per_arm": True,
        "reuses_previous_k128_to_k2048_checkpoints": False,
        "target_epoch": TARGET_EPOCH,
        "base_config": str(BASE_CONFIG.resolve()),
        "base_config_sha256": sha256(BASE_CONFIG),
        "only_permitted_config_changes": ["modes", "result_root"],
        "backend": {
            "spatial_jet": "analytic Jet1: value,x,y",
            "causal_sort": "stable geometry-Rayleigh on GPU",
            "optimizer": "GPU batched all-mode Adam transaction",
            "hot_path_get_calls": 0,
            "diagnostic_host_transfers_per_step": 1,
            "optimizer_host_transfers_per_step": 0,
        },
        "strict_timing": {
            "history": "training_elapsed_seconds",
            "process_wall": "persisted Unix nanosecond boundaries",
            "preparation_reported_separately": True,
        },
        "arms": [
            {
                "modes": modes,
                "status": "pending",
                "config": str(config_path(modes).resolve()),
                "config_sha256": sha256(config_path(modes)),
                "result_root": str(result_root(modes).resolve()),
                "run_dir": str(run_dir(modes).resolve()),
                "archive_epochs": [500, 1000, 1500, 2000],
            }
            for modes in MODE_COUNTS
        ],
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu": gpu_snapshot(),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    update_state(
        status="initialized",
        created_at=created_at,
        created_unix_ns=created_ns,
        sequence_root=str(SEQUENCE_ROOT.resolve()),
        planned_arms=list(MODE_COUNTS),
        completed_arms=[],
        current_arm=None,
    )


def other_gpu_training_processes() -> list[dict[str, Any]]:
    own_pid = os.getpid()
    blockers: list[dict[str, Any]] = []
    markers = (
        "train_hardcoded.py",
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


def ensure_inputs(modes: int) -> tuple[Path, Path, float]:
    inputs = result_root(modes) / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    timer = time.perf_counter_ns()
    if not initial.exists():
        update_state(
            status="preparing_inputs",
            current_arm=modes,
            preparation_started_at=now(),
        )
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config",
                str(config_path(modes)),
                "--output-dir",
                str(inputs),
            ]
        )
    preparation_seconds = (time.perf_counter_ns() - timer) / 1.0e9
    validation = inputs / "fixed_validation_iid_uniform_v1.npz"
    if not initial.exists() or not initial.with_suffix(".json").exists():
        raise FileNotFoundError(f"K{modes} epoch-0 input is incomplete")
    if not validation.exists() or sha256(validation) != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError(f"K{modes} fixed-validation hash mismatch")
    metadata = read_json(initial.with_suffix(".json"))
    if int(metadata["epoch"]) != 0 or "jet1" not in str(
        metadata.get("spatial_ad", "")
    ).lower():
        raise RuntimeError(f"K{modes} is not a fresh Jet1 epoch-0 checkpoint")
    set_arm_manifest(
        modes,
        preparation_wall_seconds=preparation_seconds,
        initial_checkpoint=str(initial.resolve()),
        initial_checkpoint_sha256=sha256(initial),
        validation_pack=str(validation.resolve()),
        validation_pack_sha256=sha256(validation),
    )
    return initial, validation, preparation_seconds


def completion_ok(modes: int) -> bool:
    run_path = run_dir(modes)
    completion = run_path / "completion.json"
    frozen = run_path / "checkpoints/hardcoded_sss_frozen.npz"
    if not completion.exists() or not frozen.exists():
        return False
    data = read_json(completion)
    return bool(data.get("complete")) and int(data.get("final_epoch", 0)) == TARGET_EPOCH


def trim_history(resume: Path, run_path: Path) -> int:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
    history_path = run_path / "data/hardcoded_training_history.json"
    if history_path.exists():
        history = read_json(history_path)
        kept = [row for row in history if int(row.get("epoch", 0)) <= resume_epoch]
        if len(kept) != len(history):
            backup = history_path.with_name(
                f"history_before_resume_epoch{resume_epoch:04d}_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            write_json(backup, history)
            write_json(history_path, kept)
    return resume_epoch


def final_metrics(modes: int) -> dict[str, Any]:
    run_path = run_dir(modes)
    history = read_json(run_path / "data/hardcoded_training_history.json")
    if len(history) != TARGET_EPOCH or int(history[-1].get("epoch", 0)) != TARGET_EPOCH:
        raise RuntimeError(f"K{modes} history is not a complete 2000-epoch trace")
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
        "diagnostic_host_transfers_per_step": int(
            row["diagnostic_host_transfers_per_step"]
        ),
        "optimizer_host_transfers_per_step": int(
            row["optimizer_host_transfers_per_step"]
        ),
    }
    numeric = [
        metrics[key]
        for key in (
            "validation_mean",
            "validation_p95",
            "validation_max",
            "orthogonality_max_abs",
            "learning_rate",
            "training_elapsed_seconds",
        )
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError(f"K{modes} has non-finite final metrics")
    if metrics["peeled_rank"] != modes:
        raise RuntimeError(f"K{modes} peeled rank is {metrics['peeled_rank']}")
    if metrics["spatial_jet_order_used"] != 1:
        raise RuntimeError(f"K{modes} did not use Jet1")
    if metrics["causal_sort_device"] != "gpu":
        raise RuntimeError(f"K{modes} causal sort was not on GPU")
    if metrics["optimizer_device"] != "gpu_batched_all_modes":
        raise RuntimeError(f"K{modes} optimizer was not batched on GPU")
    if metrics["diagnostic_host_transfers_per_step"] != 1 or metrics[
        "optimizer_host_transfers_per_step"
    ] != 0:
        raise RuntimeError(f"K{modes} host-transfer audit failed")
    missing = [
        epoch
        for epoch in (500, 1000, 1500, 2000)
        if not (run_path / f"checkpoints/hardcoded_epoch{epoch:04d}.npz").exists()
        or not (run_path / f"checkpoints/hardcoded_epoch{epoch:04d}.json").exists()
    ]
    if missing:
        raise RuntimeError(f"K{modes} missing archive epochs {missing}")
    return metrics


def completed_modes() -> list[int]:
    return [modes for modes in MODE_COUNTS if completion_ok(modes)]


def train_arm(modes: int) -> dict[str, Any]:
    if completion_ok(modes):
        metrics = final_metrics(modes)
        set_arm_manifest(
            modes, status="complete", skipped_existing_sequence_arm=True, final_metrics=metrics
        )
        return metrics
    initial, validation, preparation_seconds = ensure_inputs(modes)
    run_path = run_dir(modes)
    previous_arm = next(
        arm
        for arm in read_json(MANIFEST_PATH)["arms"]
        if int(arm["modes"]) == modes
    )
    first_started_at = previous_arm.get("training_started_at")
    first_started_ns = previous_arm.get("training_started_unix_ns")
    for attempt in range(1, 4):
        latest = run_path / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        resume_epoch = trim_history(resume, run_path)
        if first_started_ns is None:
            first_started_at = now()
            first_started_ns = time.time_ns()
        attempt_started_at = now()
        attempt_started_ns = time.time_ns()
        fields = {
            "status": "training",
            "attempt": attempt,
            "resumed_from_epoch": resume_epoch,
            "resume_checkpoint": str(resume.resolve()),
            "training_started_at": first_started_at,
            "training_started_unix_ns": first_started_ns,
            "attempt_started_at": attempt_started_at,
            "attempt_started_unix_ns": attempt_started_ns,
            "preparation_wall_seconds": preparation_seconds,
        }
        set_arm_manifest(modes, **fields)
        update_state(current_arm=modes, completed_arms=completed_modes(), **fields)
        try:
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts/train_hardcoded.py"),
                    "--resume",
                    str(resume),
                    "--run-dir",
                    str(run_path),
                    "--epochs",
                    str(TARGET_EPOCH),
                    "--validation-pack",
                    str(validation),
                ]
            )
            if not completion_ok(modes):
                raise RuntimeError(f"K{modes} trainer exited without completion")
            metrics = final_metrics(modes)
            finished_ns = time.time_ns()
            metrics.update(
                {
                    "training_process_wall_seconds": (
                        finished_ns - int(first_started_ns)
                    )
                    / 1.0e9,
                    "completed_at": now(),
                    "completed_unix_ns": finished_ns,
                }
            )
            set_arm_manifest(modes, status="complete", final_metrics=metrics)
            update_state(
                status="arm_complete",
                current_arm=modes,
                completed_arms=completed_modes(),
                last_completed_metrics=metrics,
            )
            print(f"DONE Jet1/GPU K{modes} E2000", flush=True)
            return metrics
        except Exception as error:
            status = "retrying" if attempt < 3 else "failed"
            details = {
                "status": status,
                "attempt": attempt,
                "error": repr(error),
                "attempt_failed_at": now(),
                "attempt_wall_seconds": (time.time_ns() - attempt_started_ns) / 1.0e9,
            }
            set_arm_manifest(modes, **details)
            update_state(current_arm=modes, **details)
            if attempt == 3:
                raise
            time.sleep(30.0)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    SEQUENCE_ROOT.mkdir(parents=True, exist_ok=True)
    configs = create_or_validate_configs()
    ensure_manifest(configs)
    if args.initialize_only:
        update_state(status="initialized")
        print(f"INITIALIZED {SEQUENCE_ROOT}", flush=True)
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
    sequence_started_ns = int(
        read_json(MANIFEST_PATH).get("sequence_started_unix_ns") or time.time_ns()
    )
    update_manifest(
        status="training",
        sequence_started_at=read_json(MANIFEST_PATH).get("sequence_started_at") or now(),
        sequence_started_unix_ns=sequence_started_ns,
    )
    for modes in MODE_COUNTS:
        train_arm(modes)
    completed_ns = time.time_ns()
    summary = {str(modes): final_metrics(modes) for modes in MODE_COUNTS}
    update_manifest(
        status="complete",
        completed_at=now(),
        completed_unix_ns=completed_ns,
        sequence_wall_seconds=(completed_ns - sequence_started_ns) / 1.0e9,
        final_metrics=summary,
    )
    update_state(
        status="complete",
        current_arm=None,
        completed_arms=list(MODE_COUNTS),
        completed_at=now(),
        completed_unix_ns=completed_ns,
        sequence_wall_seconds=(completed_ns - sequence_started_ns) / 1.0e9,
    )
    print("DONE Jet1/GPU K128-K2048 E2000 sequence", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if MANIFEST_PATH.exists():
            update_manifest(status="failed", error=repr(error), failed_at=now())
        update_state(status="failed", error=repr(error), failed_at=now())
        raise
