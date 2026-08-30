"""Train the K64 sweep winner for 2000 epochs, then run the fair comparison."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parents[1]
CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_sweep_winner_2000.json"
WINNER_E500_CONFIG = (
    CASE_ROOT
    / "results/siren_k64_airfoil_hyperparameter_sweep_e500/06_endpoint_period/period-10/config.json"
)
RESULT_ROOT = CASE_ROOT / "results/siren_k64_airfoil_sweep_winner_e2000"
INPUTS = RESULT_ROOT / "inputs"
RUN_DIR = RESULT_ROOT / "run_001"
STATE = RESULT_ROOT / "pipeline_state.json"
MANIFEST = RESULT_ROOT / "pipeline_manifest.json"
EVALUATION_DIR = RESULT_ROOT / "comparison_original2000_winner2000_laplacian"
ORIGINAL_E2000 = (
    CASE_ROOT
    / "results/siren_k64_dirichlet_psi_iid_dynamic_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz"
)
GT_ROOT = Path(
    r"C:\Users\admin\Desktop\Eigenbasis\EigenFluid-SSS\cases\2D_airfoil\results\k128_c2_sobol_10000\run_001\postprocess\gt"
)
MODE_GT = GT_ROOT / "gt_003.npz"
TARGET_EPOCH = 2000
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


def update_state(**updates: Any) -> dict[str, Any]:
    state = read_json(STATE) if STATE.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    state["updated_unix_ns"] = time.time_ns()
    write_json(STATE, state)
    return state


def active_gpu_blockers() -> list[dict[str, Any]]:
    markers = (
        "train_hardcoded.py",
        "train_metaball.py",
        "train_validate.py",
        "evaluate_unseen_k64_vs_laplacian.py",
        "compare_original2000_winner2000_laplacian.py",
    )
    blockers = []
    for process in psutil.process_iter(("pid", "cmdline", "create_time")):
        if process.pid == psutil.Process().pid:
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
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
    quiet_since: float | None = None
    while True:
        blockers = active_gpu_blockers()
        if blockers:
            quiet_since = None
            update_state(status="queued_for_gpu", blockers=blockers)
            time.sleep(30.0)
            continue
        if quiet_since is None:
            quiet_since = time.monotonic()
        quiet = time.monotonic() - quiet_since
        update_state(status="gpu_quiet_period", blockers=[], quiet_seconds=quiet)
        if quiet >= 30.0:
            update_state(status="gpu_released", gpu_released_at=now())
            return
        time.sleep(5.0)


def validate_config() -> dict[str, Any]:
    config = read_json(CONFIG)
    winner = read_json(WINNER_E500_CONFIG)
    expected = {
        "modes": 64,
        "epochs": 2000,
        "hidden": 32,
        "hidden_layers": 4,
        "siren_first_omega0": 6.0,
        "siren_hidden_omega0": 36.0,
        "points_per_step": 5120,
        "base_learning_rate": 2.0e-4,
        "min_learning_rate": 1.0e-6,
        "grad_clip": 10.0,
        "endpoint_period": 10,
        "reorthogonalization_passes": 2,
        "basis_objective": "streamfunction_dirichlet_rayleigh",
        "causal_sorting": "geometry_rayleigh",
        "d4_symmetry_augmentation": False,
        "analytic_eigenmodes": False,
        "degenerate_block_direction_alignment": False,
        "rank_reporting": "peeled_only",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"winner E2000 config mismatch: {mismatches}")
    differences = {
        key: (winner.get(key), config.get(key))
        for key in sorted(set(winner) | set(config))
        if winner.get(key) != config.get(key)
    }
    allowed = {"epochs", "result_root"}
    if set(differences) != allowed:
        raise RuntimeError(f"unexpected differences from winner E500: {differences}")
    return config


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def ensure_inputs() -> tuple[Path, Path]:
    initial = INPUTS / "initial_epoch0000.npz"
    if not initial.exists():
        start_ns = time.time_ns()
        timer = time.perf_counter_ns()
        update_state(status="preparing_inputs", preparation_started_unix_ns=start_ns)
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
        update_state(
            preparation_finished_unix_ns=time.time_ns(),
            preparation_wall_seconds=(time.perf_counter_ns() - timer) / 1.0e9,
        )
    validation = INPUTS / "fixed_validation_iid_uniform_v1.npz"
    if sha256(validation) != EXPECTED_VALIDATION_SHA256:
        raise RuntimeError("winner E2000 fixed validation pack hash mismatch")
    return initial, validation


def completion_ok() -> bool:
    path = RUN_DIR / "completion.json"
    if not path.exists():
        return False
    completion = read_json(path)
    return bool(completion.get("complete")) and int(completion.get("final_epoch", 0)) >= 2000


def trim_history(resume: Path) -> int:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
    history_path = RUN_DIR / "data/hardcoded_training_history.json"
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


def train(initial: Path, validation: Path) -> None:
    if completion_ok():
        update_state(status="training_complete_skipped")
        return
    total_timer = time.perf_counter_ns()
    first_start_ns: int | None = None
    for attempt in range(1, 4):
        latest = RUN_DIR / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        resume_epoch = trim_history(resume)
        attempt_ns = time.time_ns()
        attempt_timer = time.perf_counter_ns()
        if first_start_ns is None:
            first_start_ns = attempt_ns
        update_state(
            status="training",
            attempt=attempt,
            resumed_from_epoch=resume_epoch,
            resume_checkpoint=str(resume.resolve()),
            training_started_unix_ns=first_start_ns,
            attempt_started_unix_ns=attempt_ns,
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
                    "2000",
                    "--validation-pack",
                    str(validation),
                ]
            )
            if not completion_ok():
                raise RuntimeError("trainer exited without epoch-2000 completion")
            history = read_json(RUN_DIR / "data/hardcoded_training_history.json")
            final = history[-1]
            values = [
                final["validation_loss"],
                final["validation_loss_p95"],
                final["validation_loss_max"],
                final["validation_orth_error"],
            ]
            if not all(math.isfinite(float(value)) for value in values):
                raise FloatingPointError(f"non-finite final metrics: {values}")
            update_state(
                status="training_complete",
                training_completed_at=now(),
                training_completed_unix_ns=time.time_ns(),
                attempt_wall_seconds=(time.perf_counter_ns() - attempt_timer) / 1.0e9,
                training_wall_seconds=(time.perf_counter_ns() - total_timer) / 1.0e9,
                final_epoch=2000,
                final_validation_mean=float(final["validation_loss"]),
                final_validation_p95=float(final["validation_loss_p95"]),
                final_validation_max=float(final["validation_loss_max"]),
                final_peeled_rank=int(final["validation_peeled_rank"]),
                final_orth_error=float(final["validation_orth_error"]),
                final_learning_rate=float(final["learning_rate"]),
            )
            return
        except Exception as error:
            update_state(
                status="retrying_training" if attempt < 3 else "failed",
                error=repr(error),
                failed_attempt=attempt,
                failed_attempt_wall_seconds=(time.perf_counter_ns() - attempt_timer)
                / 1.0e9,
            )
            if attempt == 3:
                raise
            time.sleep(30.0)


def evaluate() -> None:
    summary = EVALUATION_DIR / "comparison_complete.json"
    if summary.exists() and read_json(summary).get("complete"):
        update_state(status="evaluation_complete_skipped")
        return
    winner_checkpoint = RUN_DIR / "checkpoints/hardcoded_sss_frozen.npz"
    timer = time.perf_counter_ns()
    update_state(status="evaluation", evaluation_started_at=now())
    run(
        [
            sys.executable,
            "-u",
            str(CASE_ROOT / "scripts/compare_original2000_winner2000_laplacian.py"),
            "--original-checkpoint",
            str(ORIGINAL_E2000),
            "--winner-checkpoint",
            str(winner_checkpoint),
            "--mode-gt",
            str(MODE_GT),
            "--gt-root",
            str(GT_ROOT),
            "--output-dir",
            str(EVALUATION_DIR),
            "--random-seed",
            "20260824",
        ]
    )
    if not summary.exists() or not read_json(summary).get("complete"):
        raise RuntimeError("comparison evaluation did not write complete marker")
    update_state(
        status="complete",
        completed_at=now(),
        completed_unix_ns=time.time_ns(),
        evaluation_wall_seconds=(time.perf_counter_ns() - timer) / 1.0e9,
        evaluation_summary=str(summary.resolve()),
    )


def main() -> None:
    config = validate_config()
    created_at = now()
    created_ns = time.time_ns()
    manifest = {
        "experiment": "airfoil_k64_sweep_winner_e2000_and_fair_comparison",
        "created_at": created_at,
        "created_unix_ns": created_ns,
        "config": config,
        "config_sha256": sha256(CONFIG),
        "differences_from_winner_e500": {
            "epochs": [500, 2000],
            "result_root": [str(WINNER_E500_CONFIG.parent), str(RESULT_ROOT)],
        },
        "fresh_epoch0_training": True,
        "full_2000_epoch_cosine_schedule": True,
        "fixed_validation_sha256": EXPECTED_VALIDATION_SHA256,
        "archive_epochs": [500, 1000, 1500, 2000],
        "original_e2000_checkpoint": str(ORIGINAL_E2000.resolve()),
        "mode_gt": str(MODE_GT.resolve()),
        "random_case_gt_root": str(GT_ROOT.resolve()),
        "random_seed": 20260824,
        "outputs": {
            "modes_3x8": ["modes_3x8.png", "modes_3x8.pdf"],
            "random_reconstruction_pdf": "three_random_reconstructions.pdf",
            "summary": "comparison_complete.json",
        },
    }
    write_json(MANIFEST, manifest)
    update_state(
        status="initializing",
        created_at=created_at,
        created_unix_ns=created_ns,
        config=str(CONFIG.resolve()),
        result_root=str(RESULT_ROOT.resolve()),
        target_epoch=2000,
    )
    wait_for_gpu()
    initial, validation = ensure_inputs()
    train(initial, validation)
    evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update_state(status="failed", error=repr(error), failed_at=now())
        raise
