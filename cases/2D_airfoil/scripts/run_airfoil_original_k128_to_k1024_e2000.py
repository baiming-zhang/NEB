"""Queue and run Original airfoil K128/K256/K512/K1024 through epoch 2000."""

from __future__ import annotations

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
BASE_CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_2000.json"
MODE_COUNTS = (128, 256, 512, 1024)
CONFIGS = {
    modes: CASE_ROOT / f"configs/k{modes}_airfoil_siren_original_2000.json"
    for modes in MODE_COUNTS
}
SEQUENCE_ROOT = CASE_ROOT / "results/siren_original_k64_to_k1024_e2000_sequence"
STATE_PATH = SEQUENCE_ROOT / "sequence_state.json"
MANIFEST_PATH = SEQUENCE_ROOT / "sequence_manifest.json"
EVALUATION_ROOT = SEQUENCE_ROOT / "evaluation_three_geometries_three_frames"
EVALUATION_COMPLETE = EVALUATION_ROOT / "comparison_complete.json"
EVALUATION_SCRIPT = CASE_ROOT / "scripts/evaluate_original_k64_to_k1024_vs_laplacian.py"
CONTROL_PATH = SEQUENCE_ROOT / "sequence_control.json"
ORIGINAL_K64_CHECKPOINT = (
    CASE_ROOT
    / "results/siren_k64_dirichlet_psi_iid_dynamic_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz"
)
GT_ROOT = Path(
    r"C:\Users\admin\Desktop\Eigenbasis\EigenFluid-SSS\cases\2D_airfoil\results\k128_c2_sobol_10000\run_001\postprocess\gt"
)
EXPECTED_VALIDATION_SHA256 = (
    "4b723fe1f504bd6f197c2b134a25ad9696817b427e098a11df0f85896c307162"
)
TARGET_EPOCH = 2000
QUIET_RELEASE_SECONDS = 120.0


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_control() -> dict[str, Any]:
    defaults = {
        "train_through": 256,
        "evaluate_after_each_arm_from": 256,
        "pause_after_train_through": True,
    }
    if CONTROL_PATH.exists():
        defaults.update(read_json(CONTROL_PATH))
    train_through = int(defaults["train_through"])
    if train_through not in MODE_COUNTS:
        raise ValueError(f"invalid train_through={train_through}")
    return defaults


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def result_root(config: dict[str, Any]) -> Path:
    return WORKTREE_ROOT / str(config["result_root"])


def run_dir(config: dict[str, Any]) -> Path:
    return result_root(config) / "run_001"


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = [item.strip() for item in subprocess.check_output(command, text=True).strip().split(",")]
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "name": fields[0] if fields else None,
        "uuid": fields[1] if len(fields) > 1 else None,
        "driver_version": fields[2] if len(fields) > 2 else None,
        "memory_total_mib": int(fields[3]) if len(fields) > 3 else None,
    }


def validate_configs() -> dict[int, dict[str, Any]]:
    base = read_json(BASE_CONFIG)
    configs: dict[int, dict[str, Any]] = {}
    for modes in MODE_COUNTS:
        config = read_json(CONFIGS[modes])
        differences = {
            key: (base.get(key), config.get(key))
            for key in sorted(set(base) | set(config))
            if base.get(key) != config.get(key)
        }
        if set(differences) != {"modes", "result_root"}:
            raise RuntimeError(f"K{modes} differs from Original in unexpected fields: {differences}")
        if int(config["modes"]) != modes:
            raise RuntimeError(f"K{modes} config modes mismatch")
        required = {
            "epochs": 2000,
            "hidden": 32,
            "hidden_layers": 4,
            "siren_first_omega0": 8.0,
            "siren_hidden_omega0": 30.0,
            "points_per_step": 4096,
            "base_learning_rate": 2.0e-4,
            "min_learning_rate": 1.0e-6,
            "grad_clip": 5.0,
            "endpoint_period": 20,
            "reorthogonalization_passes": 2,
            "basis_objective": "streamfunction_dirichlet_rayleigh",
            "causal_sorting": "geometry_rayleigh",
            "archive_checkpoint_every": 500,
            "d4_symmetry_augmentation": False,
            "analytic_eigenmodes": False,
            "degenerate_block_direction_alignment": False,
            "rank_reporting": "peeled_only",
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in required.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"K{modes} Original config mismatch: {mismatches}")
        configs[modes] = config
    return configs


def create_manifest(configs: dict[int, dict[str, Any]]) -> None:
    if MANIFEST_PATH.exists():
        return
    created = now()
    created_ns = time.time_ns()
    arms = []
    for modes in MODE_COUNTS:
        config = configs[modes]
        root = result_root(config)
        arms.append(
            {
                "modes": modes,
                "status": "pending",
                "config": str(CONFIGS[modes].resolve()),
                "config_sha256": sha256(CONFIGS[modes]),
                "result_root": str(root.resolve()),
                "run_dir": str((root / "run_001").resolve()),
                "archive_epochs": [500, 1000, 1500, 2000],
            }
        )
    manifest = {
        "experiment": "airfoil_original_k128_k256_k512_k1024_e2000_sequential",
        "created_at": created,
        "created_unix_ns": created_ns,
        "status": "queued",
        "strict_order": list(MODE_COUNTS),
        "fresh_epoch0_per_arm": True,
        "base_config": str(BASE_CONFIG.resolve()),
        "base_config_sha256": sha256(BASE_CONFIG),
        "only_permitted_config_changes": ["modes", "result_root"],
        "fixed_validation_sha256": EXPECTED_VALIDATION_SHA256,
        "strict_timing": {
            "wall_clock": "time.time_ns",
            "elapsed_clock": "time.perf_counter_ns",
            "history_fields": [
                "seconds",
                "training_elapsed_seconds",
                "wall_time_iso",
                "wall_time_unix_ns",
            ],
            "queue_excluded_from_training_time": True,
        },
        "arms": arms,
        "evaluation": {
            "status": "pending",
            "k_values": [64, 128, 256, 512, 1024],
            "geometries": ["up", "middle", "down"],
            "frames_per_geometry": 3,
            "layout": "2x6 per frame: GT plus five Original reconstructions; GT plus five same-domain FD Laplacian reconstructions",
            "gt_root": str(GT_ROOT.resolve()),
            "output_root": str(EVALUATION_ROOT.resolve()),
        },
        "staged_evaluations": {},
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu": gpu_snapshot(),
        },
    }
    write_json(MANIFEST_PATH, manifest)
    update_state(
        status="queued_for_existing_2d_training",
        created_at=created,
        created_unix_ns=created_ns,
        sequence_root=str(SEQUENCE_ROOT.resolve()),
        current_arm=None,
        completed_arms=[],
        planned_arms=list(MODE_COUNTS),
    )


def active_gpu_blockers() -> list[dict[str, Any]]:
    markers = (
        "train_metaball.py",
        "train_hardcoded.py",
        "evaluate_original_k64_to_k1024_vs_laplacian.py",
        "compare_three_gt_eigenfluid_vs_laplacian.py",
        "evaluate_checkpoint_vs_laplacian.py",
        "\\basisnet\\cases\\2d_metaball\\",
        "/basisnet/cases/2d_metaball/",
        "\\basisnet\\cases\\3d_metaball\\",
        "/basisnet/cases/3d_metaball/",
    )
    blockers = []
    own_pid = os.getpid()
    for process in psutil.process_iter(("pid", "name", "cmdline", "create_time")):
        if process.pid == own_pid:
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        process_name = str(process.info.get("name") or "").lower()
        if process_name in {"powershell.exe", "pwsh.exe", "cmd.exe", "conhost.exe"}:
            continue
        lowered = command.lower()
        if any(marker in lowered for marker in markers):
            blockers.append(
                {
                    "pid": int(process.pid),
                    "created_at_unix": float(process.info.get("create_time") or 0.0),
                    "command": command,
                }
            )
    return blockers


def wait_for_gpu() -> None:
    resumed_after_initial_release = bool(read_json(STATE_PATH).get("gpu_released_at"))
    required_quiet_seconds = 0.0 if resumed_after_initial_release else QUIET_RELEASE_SECONDS
    quiet_since: float | None = None
    while True:
        blockers = active_gpu_blockers()
        if blockers:
            quiet_since = None
            state = read_json(STATE_PATH)
            already_released = bool(state.get("gpu_released_at"))
            queue_wall_seconds = (
                float(state.get("queue_wall_seconds", 0.0))
                if already_released
                else (time.time_ns() - int(read_json(MANIFEST_PATH)["created_unix_ns"])) / 1.0e9
            )
            update_state(
                status=(
                    "waiting_for_active_sequence_training"
                    if already_released
                    else "queued_for_existing_2d_training"
                ),
                blockers=blockers,
                queue_wall_seconds=queue_wall_seconds,
            )
            time.sleep(30.0)
            continue
        if quiet_since is None:
            quiet_since = time.monotonic()
        quiet_seconds = time.monotonic() - quiet_since
        manifest_created_ns = int(read_json(MANIFEST_PATH)["created_unix_ns"])
        queue_wall_seconds = (time.time_ns() - manifest_created_ns) / 1.0e9
        update_state(
            status="gpu_quiet_period",
            blockers=[],
            quiet_seconds=quiet_seconds,
            quiet_required_seconds=required_quiet_seconds,
            queue_wall_seconds=queue_wall_seconds,
        )
        if quiet_seconds >= required_quiet_seconds:
            if resumed_after_initial_release:
                update_state(
                    status="gpu_released",
                    blockers=[],
                    scheduler_resumed_at=now(),
                    scheduler_resumed_unix_ns=time.time_ns(),
                )
                update_manifest(status="training")
                return
            released_at = now()
            released_ns = time.time_ns()
            queue_wall_seconds = (released_ns - manifest_created_ns) / 1.0e9
            update_state(
                status="gpu_released",
                gpu_released_at=released_at,
                gpu_released_unix_ns=released_ns,
                queue_wall_seconds=queue_wall_seconds,
            )
            update_manifest(
                status="training",
                gpu_released_at=released_at,
                gpu_released_unix_ns=released_ns,
                queue_wall_seconds=queue_wall_seconds,
            )
            return
        time.sleep(15.0)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def ensure_inputs(modes: int, config: dict[str, Any]) -> tuple[Path, Path]:
    root = result_root(config)
    inputs = root / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    if not initial.exists():
        timer = time.perf_counter_ns()
        update_state(status="preparing_inputs", current_arm=modes, preparation_started_at=now())
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config",
                str(CONFIGS[modes]),
                "--output-dir",
                str(inputs),
            ]
        )
        update_state(
            preparation_finished_at=now(),
            preparation_wall_seconds=(time.perf_counter_ns() - timer) / 1.0e9,
        )
    validation = inputs / "fixed_validation_iid_uniform_v1.npz"
    if not validation.exists() or sha256(validation) != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError(f"K{modes} fixed-validation hash mismatch")
    return initial, validation


def completion_ok(run: Path, modes: int) -> bool:
    path = run / "completion.json"
    if not path.exists():
        return False
    data = read_json(path)
    return (
        bool(data.get("complete"))
        and int(data.get("final_epoch", 0)) >= TARGET_EPOCH
        and (run / "checkpoints/hardcoded_sss_frozen.npz").exists()
        and modes > 0
    )


def trim_history(resume: Path, run: Path) -> int:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
    history_path = run / "data/hardcoded_training_history.json"
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


def final_metrics(run: Path, modes: int) -> dict[str, Any]:
    history = read_json(run / "data/hardcoded_training_history.json")
    row = history[-1]
    if int(row.get("epoch", 0)) != TARGET_EPOCH:
        raise RuntimeError(f"K{modes} history does not end at epoch 2000")
    metrics = {
        "final_epoch": TARGET_EPOCH,
        "validation_mean": float(row["validation_loss"]),
        "validation_p95": float(row["validation_loss_p95"]),
        "validation_max": float(row["validation_loss_max"]),
        "peeled_rank": int(row["validation_peeled_rank"]),
        "orthogonality_max_abs": float(row["validation_orth_error"]),
        "learning_rate": float(row["learning_rate"]),
        "training_elapsed_seconds": float(row.get("training_elapsed_seconds", row["seconds"])),
        "wall_time_iso": row.get("wall_time_iso"),
        "wall_time_unix_ns": int(row.get("wall_time_unix_ns", 0)),
    }
    finite = [
        metrics["validation_mean"],
        metrics["validation_p95"],
        metrics["validation_max"],
        metrics["orthogonality_max_abs"],
        metrics["learning_rate"],
        metrics["training_elapsed_seconds"],
    ]
    if not all(math.isfinite(value) for value in finite):
        raise FloatingPointError(f"K{modes} has non-finite final metrics: {finite}")
    if metrics["peeled_rank"] != modes:
        raise RuntimeError(f"K{modes} peeled rank is {metrics['peeled_rank']}")
    missing_archives = [
        epoch
        for epoch in (500, 1000, 1500, 2000)
        if not (run / f"checkpoints/hardcoded_epoch{epoch:04d}.npz").exists()
    ]
    if missing_archives:
        raise RuntimeError(f"K{modes} missing archives: {missing_archives}")
    return metrics


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


def train_arm(modes: int, config: dict[str, Any]) -> dict[str, Any]:
    run_path = run_dir(config)
    if completion_ok(run_path, modes):
        metrics = final_metrics(run_path, modes)
        set_arm_manifest(modes, status="complete", skipped_existing=True, final_metrics=metrics)
        return metrics
    initial, validation = ensure_inputs(modes, config)
    overall_timer = time.perf_counter_ns()
    first_start_ns: int | None = None
    first_start_at: str | None = None
    for attempt in range(1, 4):
        latest = run_path / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        resume_epoch = trim_history(resume, run_path)
        attempt_start_at = now()
        attempt_start_ns = time.time_ns()
        attempt_timer = time.perf_counter_ns()
        if first_start_ns is None:
            first_start_ns = attempt_start_ns
            first_start_at = attempt_start_at
        set_arm_manifest(
            modes,
            status="training",
            attempt=attempt,
            resumed_from_epoch=resume_epoch,
            resume_checkpoint=str(resume.resolve()),
            training_started_at=first_start_at,
            training_started_unix_ns=first_start_ns,
            attempt_started_at=attempt_start_at,
            attempt_started_unix_ns=attempt_start_ns,
        )
        update_state(
            status="training",
            current_arm=modes,
            attempt=attempt,
            resumed_from_epoch=resume_epoch,
            training_started_at=first_start_at,
            training_started_unix_ns=first_start_ns,
            attempt_started_at=attempt_start_at,
            attempt_started_unix_ns=attempt_start_ns,
        )
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
            if not completion_ok(run_path, modes):
                raise RuntimeError(f"K{modes} trainer exited without completion")
            metrics = final_metrics(run_path, modes)
            finished_at = now()
            finished_ns = time.time_ns()
            metrics.update(
                {
                    "training_process_wall_seconds": (time.perf_counter_ns() - overall_timer) / 1.0e9,
                    "completed_at": finished_at,
                    "completed_unix_ns": finished_ns,
                }
            )
            set_arm_manifest(modes, status="complete", final_metrics=metrics)
            completed = [
                int(arm["modes"])
                for arm in read_json(MANIFEST_PATH)["arms"]
                if arm.get("status") == "complete"
            ]
            update_state(
                status="arm_complete",
                current_arm=modes,
                completed_arms=completed,
                last_completed_metrics=metrics,
            )
            print(f"DONE Original K{modes} E2000", flush=True)
            return metrics
        except Exception as error:
            status = "retrying" if attempt < 3 else "failed"
            details = {
                "status": status,
                "attempt": attempt,
                "error": repr(error),
                "attempt_wall_seconds": (time.perf_counter_ns() - attempt_timer) / 1.0e9,
                "failed_at": now(),
            }
            set_arm_manifest(modes, **details)
            update_state(current_arm=modes, **details)
            if attempt == 3:
                raise
            time.sleep(30.0)
    raise AssertionError("unreachable")


def evaluate() -> None:
    if EVALUATION_COMPLETE.exists() and read_json(EVALUATION_COMPLETE).get("complete"):
        update_manifest(
            status="complete",
            evaluation={
                **read_json(MANIFEST_PATH)["evaluation"],
                "status": "complete",
                "completed_at": now(),
                "summary": str(EVALUATION_COMPLETE.resolve()),
                "reused_staged_k1024_evaluation": True,
            },
        )
        update_state(status="complete", evaluation_summary=str(EVALUATION_COMPLETE.resolve()))
        return
    if not EVALUATION_SCRIPT.exists():
        raise FileNotFoundError(EVALUATION_SCRIPT)
    checkpoints = {
        64: ORIGINAL_K64_CHECKPOINT,
        **{
            modes: run_dir(read_json(CONFIGS[modes])) / "checkpoints/hardcoded_sss_frozen.npz"
            for modes in MODE_COUNTS
        },
    }
    command = [
        sys.executable,
        "-u",
        str(EVALUATION_SCRIPT),
        "--gt-root",
        str(GT_ROOT),
        "--output-dir",
        str(EVALUATION_ROOT),
        "--random-seed",
        "20260824",
    ]
    for modes in (64,) + MODE_COUNTS:
        command.extend([f"--k{modes}-checkpoint", str(checkpoints[modes])])
    for attempt in range(1, 4):
        timer = time.perf_counter_ns()
        update_state(status="evaluation", evaluation_attempt=attempt, evaluation_started_at=now())
        try:
            run(command)
            if not EVALUATION_COMPLETE.exists() or not read_json(EVALUATION_COMPLETE).get("complete"):
                raise RuntimeError("evaluation did not write complete marker")
            elapsed = (time.perf_counter_ns() - timer) / 1.0e9
            update_manifest(
                status="complete",
                evaluation={
                    **read_json(MANIFEST_PATH)["evaluation"],
                    "status": "complete",
                    "completed_at": now(),
                    "wall_seconds": elapsed,
                    "summary": str(EVALUATION_COMPLETE.resolve()),
                },
            )
            update_state(
                status="complete",
                completed_at=now(),
                completed_unix_ns=time.time_ns(),
                evaluation_wall_seconds=elapsed,
                evaluation_summary=str(EVALUATION_COMPLETE.resolve()),
            )
            return
        except Exception as error:
            update_state(
                status="evaluation_retrying" if attempt < 3 else "failed",
                evaluation_attempt=attempt,
                evaluation_error=repr(error),
                evaluation_attempt_wall_seconds=(time.perf_counter_ns() - timer) / 1.0e9,
            )
            if attempt == 3:
                raise
            time.sleep(30.0)


def staged_evaluation_root(max_modes: int) -> Path:
    if max_modes == max(MODE_COUNTS):
        return EVALUATION_ROOT
    return SEQUENCE_ROOT / f"evaluation_after_k{max_modes:04d}_three_geometries_three_frames"


def set_staged_evaluation_manifest(max_modes: int, **updates: Any) -> None:
    manifest = read_json(MANIFEST_PATH)
    staged = manifest.setdefault("staged_evaluations", {})
    record = staged.setdefault(f"k{max_modes}", {})
    record.update(updates)
    record["updated_at"] = now()
    record["updated_unix_ns"] = time.time_ns()
    manifest["updated_at"] = now()
    manifest["updated_unix_ns"] = time.time_ns()
    write_json(MANIFEST_PATH, manifest)


def evaluate_stage(max_modes: int) -> Path:
    active_modes = (64,) + tuple(modes for modes in MODE_COUNTS if modes <= max_modes)
    output_root = staged_evaluation_root(max_modes)
    complete = output_root / "comparison_complete.json"
    if complete.exists() and read_json(complete).get("complete"):
        set_staged_evaluation_manifest(
            max_modes,
            status="complete",
            k_values=list(active_modes),
            output_root=str(output_root.resolve()),
            summary=str(complete.resolve()),
            skipped_existing=True,
        )
        return complete
    checkpoints = {64: ORIGINAL_K64_CHECKPOINT}
    checkpoints.update(
        {
            modes: run_dir(read_json(CONFIGS[modes])) / "checkpoints/hardcoded_sss_frozen.npz"
            for modes in active_modes
            if modes != 64
        }
    )
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"staged K{max_modes} evaluation missing checkpoints: {missing}")
    command = [
        sys.executable,
        "-u",
        str(EVALUATION_SCRIPT),
        "--gt-root",
        str(GT_ROOT),
        "--output-dir",
        str(output_root),
        "--random-seed",
        "20260824",
        "--k-values",
        *[str(modes) for modes in active_modes],
    ]
    for modes in active_modes:
        command.extend([f"--k{modes}-checkpoint", str(checkpoints[modes])])
    for attempt in range(1, 4):
        timer = time.perf_counter_ns()
        started_at = now()
        set_staged_evaluation_manifest(
            max_modes,
            status="running",
            attempt=attempt,
            k_values=list(active_modes),
            output_root=str(output_root.resolve()),
            started_at=started_at,
        )
        update_state(
            status="staged_evaluation",
            current_arm=max_modes,
            staged_evaluation_k_values=list(active_modes),
            staged_evaluation_attempt=attempt,
            staged_evaluation_started_at=started_at,
        )
        try:
            run(command)
            if not complete.exists() or not read_json(complete).get("complete"):
                raise RuntimeError(f"K{max_modes} staged evaluation did not write completion marker")
            elapsed = (time.perf_counter_ns() - timer) / 1.0e9
            set_staged_evaluation_manifest(
                max_modes,
                status="complete",
                completed_at=now(),
                wall_seconds=elapsed,
                summary=str(complete.resolve()),
            )
            update_state(
                status="staged_evaluation_complete",
                current_arm=max_modes,
                staged_evaluation_wall_seconds=elapsed,
                staged_evaluation_summary=str(complete.resolve()),
            )
            return complete
        except Exception as error:
            set_staged_evaluation_manifest(
                max_modes,
                status="retrying" if attempt < 3 else "failed",
                attempt=attempt,
                error=repr(error),
                attempt_wall_seconds=(time.perf_counter_ns() - timer) / 1.0e9,
            )
            update_state(
                status="staged_evaluation_retrying" if attempt < 3 else "failed",
                staged_evaluation_error=repr(error),
            )
            if attempt == 3:
                raise
            time.sleep(30.0)
    raise AssertionError("unreachable")


def main() -> None:
    configs = validate_configs()
    create_manifest(configs)
    control = read_control()
    if not ORIGINAL_K64_CHECKPOINT.exists():
        raise FileNotFoundError(ORIGINAL_K64_CHECKPOINT)
    wait_for_gpu()
    for modes in MODE_COUNTS:
        if modes > int(control["train_through"]):
            break
        train_arm(modes, configs[modes])
        if modes >= int(control["evaluate_after_each_arm_from"]):
            evaluate_stage(modes)
    if bool(control["pause_after_train_through"]) and int(control["train_through"]) < max(MODE_COUNTS):
        completed = [
            int(arm["modes"])
            for arm in read_json(MANIFEST_PATH)["arms"]
            if arm.get("status") == "complete"
        ]
        update_manifest(status="paused_after_staged_evaluation")
        update_state(
            status="paused_after_staged_evaluation",
            current_arm=int(control["train_through"]),
            completed_arms=completed,
            pause_reason="awaiting user decision before higher-K training",
            control_file=str(CONTROL_PATH.resolve()),
        )
        return
    update_manifest(status="evaluation")
    evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update_state(status="failed", error=repr(error), failed_at=now())
        raise
