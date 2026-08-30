"""Queue and run the original-config K64 airfoil SIREN through epoch 2000."""

from __future__ import annotations

import hashlib
import json
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
CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_2000.json"
ORIGINAL_CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_500.json"
RESULT_ROOT = CASE_ROOT / "results/siren_k64_dirichlet_psi_iid_dynamic_e2000"
INPUTS = RESULT_ROOT / "inputs"
RUN_DIR = RESULT_ROOT / "run_001"
STATE_PATH = RESULT_ROOT / "launcher_state.json"
RUN_MANIFEST_PATH = RESULT_ROOT / "training_run_manifest.json"
TARGET_EPOCH = 2000
QUIET_RELEASE_SECONDS = 120.0


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_state(**updates: Any) -> dict[str, Any]:
    state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    state["updated_unix_ns"] = time.time_ns()
    write_json(STATE_PATH, state)
    return state


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": repr(error)}
    fields = [item.strip() for item in output.split(",")]
    return {
        "available": True,
        "name": fields[0] if len(fields) > 0 else None,
        "uuid": fields[1] if len(fields) > 1 else None,
        "driver_version": fields[2] if len(fields) > 2 else None,
        "memory_total_mib": int(fields[3]) if len(fields) > 3 else None,
    }


def write_run_manifest(config: dict[str, Any], created_at: str, created_ns: int) -> None:
    original = read_json(ORIGINAL_CONFIG)
    differences = {
        key: {"original": original.get(key), "epoch2000": config.get(key)}
        for key in sorted(set(original) | set(config))
        if original.get(key) != config.get(key)
    }
    command = [
        sys.executable,
        "-u",
        str(CASE_ROOT / "scripts/train_hardcoded.py"),
        "--resume",
        "<initial_epoch0000.npz-or-hardcoded_latest.npz>",
        "--run-dir",
        str(RUN_DIR.resolve()),
        "--epochs",
        str(TARGET_EPOCH),
        "--validation-pack",
        "<fixed_validation_iid_uniform_v1.npz>",
    ]
    manifest = {
        "experiment": "airfoil_k64_original_configuration_epoch2000",
        "status": "queued_waiting_for_metaball",
        "created_at": created_at,
        "created_unix_ns": created_ns,
        "strict_timing": {
            "wall_clock": "time.time_ns",
            "elapsed_clock": "time.perf_counter_ns",
            "units": "nanoseconds internally; seconds serialized",
            "queue_included_in_training_time": False,
        },
        "config_path": str(CONFIG.resolve()),
        "config_sha256": sha256_file(CONFIG),
        "config_snapshot": config,
        "differences_from_original_epoch500_config": differences,
        "planned_archive_epochs": [500, 1000, 1500, 2000],
        "training_command_template": command,
        "important_artifacts": {
            "state": str(STATE_PATH.resolve()),
            "training_history_json": str(
                (RUN_DIR / "data/hardcoded_training_history.json").resolve()
            ),
            "training_history_csv": str(
                (RUN_DIR / "data/hardcoded_training_history.csv").resolve()
            ),
            "completion": str((RUN_DIR / "completion.json").resolve()),
            "checkpoint_directory": str((RUN_DIR / "checkpoints").resolve()),
        },
        "history_fields_of_interest": [
            "epoch",
            "seconds",
            "training_elapsed_seconds",
            "wall_time_iso",
            "wall_time_unix_ns",
            "beta",
            "alpha_deg",
            "train_loss",
            "validation_loss",
            "validation_loss_p95",
            "validation_loss_max",
            "validation_peeled_rank",
            "validation_orth_error",
            "learning_rate",
            "best_validation_epoch",
        ],
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu": gpu_snapshot(),
        },
        "timing": {
            "queue_started_at": created_at,
            "queue_started_unix_ns": created_ns,
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def update_run_manifest(status: str, **updates: Any) -> None:
    manifest = read_json(RUN_MANIFEST_PATH)
    manifest["status"] = status
    manifest.setdefault("timing", {}).update(updates)
    manifest["updated_at"] = now()
    manifest["updated_unix_ns"] = time.time_ns()
    write_json(RUN_MANIFEST_PATH, manifest)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def metaball_blockers() -> list[dict[str, Any]]:
    needles = (
        "train_metaball.py",
        "run_connected_k64_hyperparameter_sweep.py",
        "run_connected_k64_original_e2000.py",
        "run_connected_k64_strict_fairness.py",
        "run_deferred_e2000.py",
        "compare_three_gt_eigenfluid_vs_laplacian.py",
        "evaluate_checkpoint_vs_laplacian.py",
        "\\basisnet\\cases\\2d_metaball\\",
        "/basisnet/cases/2d_metaball/",
        "\\basisnet\\cases\\3d_metaball\\",
        "/basisnet/cases/3d_metaball/",
        "\\basisnet\\cases\\3d_metaball_connected\\",
        "/basisnet/cases/3d_metaball_connected/",
    )
    blockers: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "cmdline", "create_time")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        lowered = command.lower()
        if any(needle in lowered for needle in needles):
            blockers.append(
                {
                    "pid": int(process.info["pid"]),
                    "created_at_unix": float(process.info.get("create_time") or 0.0),
                    "command": command,
                }
            )
    return blockers


def wait_for_metaball(queue_started_ns: int) -> tuple[str, int, float]:
    quiet_since: float | None = None
    last_report = 0.0
    while True:
        blockers = metaball_blockers()
        if blockers:
            quiet_since = None
            update_state(
                status="queued_waiting_for_metaball",
                queue_blockers=blockers,
                quiet_release_seconds=QUIET_RELEASE_SECONDS,
                queue_wall_seconds=(time.time_ns() - queue_started_ns) / 1.0e9,
            )
            if time.monotonic() - last_report >= 1800.0 or last_report == 0.0:
                print(
                    "GPU QUEUE waiting for metaball work: "
                    + ", ".join(str(item["pid"]) for item in blockers),
                    flush=True,
                )
                last_report = time.monotonic()
            time.sleep(30.0)
            continue
        if quiet_since is None:
            quiet_since = time.monotonic()
        quiet_elapsed = time.monotonic() - quiet_since
        update_state(
            status="queued_metaball_released_quiet_period",
            queue_blockers=[],
            quiet_elapsed_seconds=quiet_elapsed,
            quiet_release_seconds=QUIET_RELEASE_SECONDS,
            queue_wall_seconds=(time.time_ns() - queue_started_ns) / 1.0e9,
        )
        if quiet_elapsed >= QUIET_RELEASE_SECONDS:
            print("GPU QUEUE released; starting airfoil epoch-2000 run", flush=True)
            released_at = now()
            released_ns = time.time_ns()
            queue_seconds = (released_ns - queue_started_ns) / 1.0e9
            update_state(
                status="queue_released",
                queue_released_at=released_at,
                queue_released_unix_ns=released_ns,
                queue_wall_seconds=queue_seconds,
            )
            update_run_manifest(
                "queue_released",
                queue_released_at=released_at,
                queue_released_unix_ns=released_ns,
                queue_wall_seconds=queue_seconds,
            )
            return released_at, released_ns, queue_seconds
        time.sleep(15.0)


def validation_pack() -> Path:
    matches = sorted(INPUTS.glob("fixed_validation_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation pack under {INPUTS}")
    return matches[0]


def ensure_inputs() -> tuple[Path, Path]:
    initial = INPUTS / "initial_epoch0000.npz"
    if not initial.exists():
        started_at = now()
        started_ns = time.time_ns()
        timer_ns = time.perf_counter_ns()
        update_state(
            status="preparing_inputs",
            preparation_started_at=started_at,
            preparation_started_unix_ns=started_ns,
        )
        update_run_manifest(
            "preparing_inputs",
            preparation_started_at=started_at,
            preparation_started_unix_ns=started_ns,
        )
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config",
                str(CONFIG),
                "--output-dir",
                str(INPUTS),
            ]
        )
        finished_at = now()
        finished_ns = time.time_ns()
        elapsed_seconds = (time.perf_counter_ns() - timer_ns) / 1.0e9
        update_state(
            preparation_finished_at=finished_at,
            preparation_finished_unix_ns=finished_ns,
            preparation_wall_seconds=elapsed_seconds,
        )
        update_run_manifest(
            "inputs_ready",
            preparation_finished_at=finished_at,
            preparation_finished_unix_ns=finished_ns,
            preparation_wall_seconds=elapsed_seconds,
        )
    return initial, validation_pack()


def trim_history_to_resume(resume: Path) -> int:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
    history_path = RUN_DIR / "data/hardcoded_training_history.json"
    if not history_path.exists():
        return resume_epoch
    history = read_json(history_path)
    kept = [row for row in history if int(row.get("epoch", 0)) <= resume_epoch]
    if len(kept) != len(history):
        backup = history_path.with_name(
            f"recovery_history_before_epoch{resume_epoch:04d}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        write_json(backup, history)
        write_json(history_path, kept)
    return resume_epoch


def final_metrics() -> dict[str, Any]:
    history_path = RUN_DIR / "data/hardcoded_training_history.json"
    history = read_json(history_path)
    rows = [
        row
        for row in history
        if int(row.get("epoch", 0)) == TARGET_EPOCH
        and "validation_loss" in row
    ]
    if not rows:
        raise RuntimeError("missing epoch-2000 validation row")
    row = rows[-1]
    return {
        "final_epoch": TARGET_EPOCH,
        "final_validation_mean": float(row["validation_loss"]),
        "final_validation_p95": float(row["validation_loss_p95"]),
        "final_validation_max": float(row["validation_loss_max"]),
        "final_peeled_rank": int(row["validation_peeled_rank"]),
        "final_orth_error": float(row["validation_orth_error"]),
        "final_learning_rate": float(row["learning_rate"]),
        "elapsed_seconds": float(row["seconds"]),
    }


def already_complete() -> bool:
    completion_path = RUN_DIR / "completion.json"
    if not completion_path.exists():
        return False
    completion = read_json(completion_path)
    return bool(completion.get("complete")) and int(
        completion.get("final_epoch", 0)
    ) >= TARGET_EPOCH


def train() -> None:
    if already_complete():
        update_state(status="complete", **final_metrics())
        print("SKIP already-complete airfoil epoch-2000 run", flush=True)
        return
    initial, fixed_validation = ensure_inputs()
    first_training_started_at: str | None = None
    first_training_started_ns: int | None = None
    total_training_timer_ns = time.perf_counter_ns()
    for attempt in range(1, 4):
        latest = RUN_DIR / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        resume_epoch = trim_history_to_resume(resume)
        attempt_started_at = now()
        attempt_started_ns = time.time_ns()
        attempt_timer_ns = time.perf_counter_ns()
        if first_training_started_at is None:
            first_training_started_at = attempt_started_at
            first_training_started_ns = attempt_started_ns
        update_state(
            status="training",
            attempt=attempt,
            resumed_from_epoch=resume_epoch,
            resume_checkpoint=str(resume.resolve()),
            queue_blockers=[],
            training_started_at=first_training_started_at,
            training_started_unix_ns=first_training_started_ns,
            attempt_started_at=attempt_started_at,
            attempt_started_unix_ns=attempt_started_ns,
        )
        update_run_manifest(
            "training",
            training_started_at=first_training_started_at,
            training_started_unix_ns=first_training_started_ns,
            attempt=attempt,
            attempt_started_at=attempt_started_at,
            attempt_started_unix_ns=attempt_started_ns,
            resumed_from_epoch=resume_epoch,
            resume_checkpoint=str(resume.resolve()),
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
                    str(RUN_DIR),
                    "--epochs",
                    str(TARGET_EPOCH),
                    "--validation-pack",
                    str(fixed_validation),
                ]
            )
            if not already_complete():
                raise RuntimeError("trainer exited without epoch-2000 completion")
            completed_at = now()
            completed_ns = time.time_ns()
            attempt_seconds = (time.perf_counter_ns() - attempt_timer_ns) / 1.0e9
            training_seconds = (
                time.perf_counter_ns() - total_training_timer_ns
            ) / 1.0e9
            metrics = final_metrics()
            update_state(
                status="complete",
                completed_at=completed_at,
                completed_unix_ns=completed_ns,
                attempt_wall_seconds=attempt_seconds,
                training_wall_seconds=training_seconds,
                **metrics,
            )
            update_run_manifest(
                "complete",
                completed_at=completed_at,
                completed_unix_ns=completed_ns,
                final_attempt_wall_seconds=attempt_seconds,
                total_training_wall_seconds=training_seconds,
                final_metrics=metrics,
            )
            print("DONE airfoil K64 original config epoch 2000", flush=True)
            return
        except Exception as error:
            failed_at = now()
            failed_ns = time.time_ns()
            attempt_seconds = (time.perf_counter_ns() - attempt_timer_ns) / 1.0e9
            update_state(
                status="retrying" if attempt < 3 else "failed",
                attempt=attempt,
                error=repr(error),
                attempt_failed_at=failed_at,
                attempt_failed_unix_ns=failed_ns,
                attempt_wall_seconds=attempt_seconds,
            )
            update_run_manifest(
                "retrying" if attempt < 3 else "failed",
                attempt=attempt,
                attempt_failed_at=failed_at,
                attempt_failed_unix_ns=failed_ns,
                attempt_wall_seconds=attempt_seconds,
                error=repr(error),
            )
            if attempt == 3:
                raise
            print(f"TRAIN attempt {attempt} failed; resuming latest checkpoint", flush=True)
            time.sleep(30.0)


def main() -> None:
    config = read_json(CONFIG)
    required = {
        "modes": 64,
        "epochs": TARGET_EPOCH,
        "hidden": 32,
        "hidden_layers": 4,
        "siren_first_omega0": 8.0,
        "siren_hidden_omega0": 30.0,
        "points_per_step": 4096,
        "archive_checkpoint_every": 500,
        "rank_reporting": "peeled_only",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"epoch-2000 config mismatch: {mismatches}")
    existing_state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
    created_at = str(existing_state.get("created_at") or now())
    created_ns = int(
        existing_state.get("created_unix_ns")
        or datetime.fromisoformat(created_at).timestamp() * 1.0e9
    )
    write_run_manifest(config, created_at, created_ns)
    update_state(
        status="initializing",
        created_at=created_at,
        created_unix_ns=created_ns,
        config=str(CONFIG.resolve()),
        result_root=str(RESULT_ROOT.resolve()),
        target_epoch=TARGET_EPOCH,
        archive_checkpoint_every=500,
    )
    wait_for_metaball(created_ns)
    train()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update_state(status="failed", error=repr(error), failed_at=now())
        raise
