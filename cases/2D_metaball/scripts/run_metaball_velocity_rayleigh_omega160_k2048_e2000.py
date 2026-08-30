"""Run the staged K=2048/E=2000 velocity-Rayleigh campaign.

Training pauses exactly at epoch 500, runs the fixed three-unseen-case audit
with the GPU training process stopped, and then resumes from the epoch-500
checkpoint to epoch 2000.  The strict training timer remains checkpointed and
does not include the unseen evaluation wall time.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE = CASE_ROOT.parents[1]
CONFIG = CASE_ROOT / (
    "configs/"
    "k2048_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json"
)
RESULT_ROOT = CASE_ROOT / (
    "results/siren_k2048_connected_velocity_rayleigh_omega160_onepass_e2000"
)
RUN_DIR = RESULT_ROOT / "run_001"
INPUTS = RESULT_ROOT / "inputs"
LAUNCH_MANIFEST = RESULT_ROOT / "launch_manifest.json"
BANDWIDTH_AUDIT = RESULT_ROOT / "initial_bandwidth_audit.json"
EPOCH500_EVALUATION = RESULT_ROOT / "evaluation/epoch0500_three_unseen_k2048_128"
RECOVERY_TRAINING_ELAPSED_FLOOR = 45458.479101299825


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print(f"[{now()}] RUN {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=WORKTREE, check=True)


def validate_fixed_config(config: dict[str, Any]) -> None:
    expected = {
        "modes": 2048,
        "epochs": 2000,
        "hidden": 32,
        "hidden_layers": 4,
        "siren_first_omega0": float(config["siren_first_omega0"]),
        "siren_hidden_omega0": 30.0,
        "learning_rate": 2.0e-4,
        "min_learning_rate": 1.0e-6,
        "basis_objective": "velocity_rayleigh",
        "spatial_derivative_order": 3,
        "causal_sorting": "geometry_rayleigh",
        "reorthogonalization_passes": 1,
        "peeling_relative_mass_floor": 1.0e-8,
        "peeling_overlap_tolerance": 1.0e-3,
        "sampling_mode": "uniform_random",
        "points_per_step": 4096,
        "validation_points": 4096,
        "checkpoint_every": 500,
        "archive_checkpoint_every": 500,
    }
    changed = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise ValueError(f"K=2048 campaign config drifted: {changed}")


def checkpoint_epoch(path: Path) -> int:
    return int(read_json(path.with_suffix(".json"))["epoch"])


def validation_pack() -> Path:
    matches = sorted(INPUTS.glob("fixed_validation_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one validation pack, got {matches}")
    return matches[0]


def ensure_inputs() -> tuple[Path, Path]:
    initial = INPUTS / "initial_epoch0000.npz"
    if not initial.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_metaball_run.py"),
                "--config",
                str(CONFIG),
                "--output-dir",
                str(INPUTS),
            ]
        )
    metadata = read_json(initial.with_suffix(".json"))
    if int(metadata.get("config", {}).get("modes", 0)) != 2048:
        raise RuntimeError("prepared checkpoint is not K=2048")
    if metadata.get("spatial_ad") != "analytic_compact_symmetric_xy_jet3":
        raise RuntimeError("prepared checkpoint is not analytic Jet3")
    if metadata.get("config", {}).get("basis_objective") != "velocity_rayleigh":
        raise RuntimeError("prepared checkpoint has the wrong objective")
    return initial, validation_pack()


def audit_initialization(initial: Path) -> None:
    baseline = CASE_ROOT / (
        "results/siren_k1024_connected_velocity_rayleigh_omega160_onepass_e2000/"
        "inputs/initial_epoch0000.npz"
    )
    audit_path = RESULT_ROOT / "initialization_audit.json"
    audit: dict[str, Any] = {
        "k2048_initial": str(initial.resolve()),
        "k1024_prefix_baseline": str(baseline.resolve()),
        "audited_at": now(),
        "initialization_policy": (
            "fresh deterministic seed; first 1024 mode tensors must match the "
            "K=1024 campaign bitwise, with 1024 newly initialized suffix modes"
        ),
    }
    if not baseline.exists():
        audit["status"] = "baseline_missing"
        write_json(audit_path, audit)
        return
    with np.load(initial, allow_pickle=False) as expanded, np.load(
        baseline, allow_pickle=False
    ) as prefix:
        common = list(prefix.files)
        missing = [name for name in common if name not in expanded.files]
        exact = not missing and all(
            np.array_equal(expanded[name], prefix[name]) for name in common
        )
        extra = [name for name in expanded.files if name not in prefix.files]
        audit.update(
            status="exact_prefix_match" if exact else "mismatch",
            common_tensor_count=len(common),
            extra_tensor_count=len(extra),
            missing_tensor_count=len(missing),
            all_common_tensors_bitwise_equal=exact,
        )
    write_json(audit_path, audit)
    if not exact:
        raise RuntimeError("K=2048 initialization does not preserve K=1024 prefix")


def ensure_bandwidth_audit(initial: Path, fixed_validation: Path) -> dict[str, Any]:
    omega = float(read_json(CONFIG)["siren_first_omega0"])
    if not BANDWIDTH_AUDIT.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/audit_velocity_rayleigh_bandwidth.py"),
                "--config",
                str(CONFIG),
                "--initial",
                str(initial),
                "--omegas",
                f"{omega:g}",
                "--tokens",
                "1",
                "2",
                "3",
                "20",
                "--validation-pack",
                str(fixed_validation),
                "--output",
                str(BANDWIDTH_AUDIT),
            ]
        )
    audit = read_json(BANDWIDTH_AUDIT)
    rows = audit.get("rows", [])
    if len(rows) != 8:
        raise RuntimeError(f"expected eight bandwidth audit rows, got {len(rows)}")
    failed = [row for row in rows if not row["accepted_under_production_rule"]]
    if failed:
        raise RuntimeError(
            f"omega={omega:g} K=2048 bandwidth audit failed: {failed}"
        )
    return audit


def trim_history_to_resume(resume: Path) -> None:
    history_path = RUN_DIR / "data/hardcoded_training_history.json"
    if not history_path.exists():
        return
    epoch = checkpoint_epoch(resume)
    history = read_json(history_path)
    kept = [row for row in history if int(row.get("epoch", 0)) <= epoch]
    if len(kept) == len(history):
        return
    backup = history_path.with_name(
        f"recovery_history_before_epoch{epoch:04d}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(history_path, backup)
    write_json(history_path, kept)


def run_epoch500_stage(resume: Path, fixed_validation: Path) -> Path:
    if checkpoint_epoch(resume) < 500:
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/train_metaball.py"),
                "--resume",
                str(resume),
                "--run-dir",
                str(RUN_DIR),
                "--validation-pack",
                str(fixed_validation),
                "--stop-after-epoch",
                "500",
            ]
        )
    checkpoint = RUN_DIR / "checkpoints/hardcoded_epoch0500.npz"
    if not checkpoint.exists() or checkpoint_epoch(checkpoint) != 500:
        raise RuntimeError("epoch-500 checkpoint was not created")
    pause_path = RUN_DIR / "planned_pause.json"
    if not pause_path.exists():
        pause_path = RUN_DIR / "planned_pause_epoch0500.json"
    pause = read_json(pause_path)
    if not pause.get("paused") or int(pause.get("epoch", 0)) != 500:
        raise RuntimeError(f"invalid epoch-500 pause record: {pause}")
    return checkpoint


def run_epoch500_unseen(checkpoint: Path) -> Path:
    summary = EPOCH500_EVALUATION / "three_unseen_reconstruction_summary.json"
    if not summary.exists() or read_json(summary).get("status") != "complete":
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/evaluate_velocity_checkpoint_three_unseen.py"),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(EPOCH500_EVALUATION),
            ]
        )
    value = read_json(summary)
    if int(value.get("checkpoint_epoch", -1)) != 500 or int(value.get("modes", -1)) != 2048:
        raise RuntimeError("epoch-500 unseen evaluation metadata mismatch")
    return summary


def verify_completion() -> None:
    completion = read_json(RUN_DIR / "completion.json")
    if not completion.get("complete") or int(completion.get("final_epoch", 0)) != 2000:
        raise RuntimeError(f"incomplete training record: {completion}")
    required = [
        RUN_DIR / f"checkpoints/hardcoded_epoch{epoch:04d}.npz"
        for epoch in (500, 1000, 1500, 2000)
    ] + [RUN_DIR / "checkpoints/hardcoded_sss_frozen.npz"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing final checkpoints: {missing}")
    for path in required:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise RuntimeError(f"empty checkpoint: {path}")


def main() -> None:
    config = read_json(CONFIG)
    validate_fixed_config(config)
    manifest: dict[str, Any] = {
        "status": "preparing",
        "queued_at": now(),
        "config": str(CONFIG.resolve()),
        "result_root": str(RESULT_ROOT.resolve()),
        "pid": os.getpid(),
        "python": sys.executable,
        "strict_training_seconds_source": (
            "run_001/data/hardcoded_training_history.json:training_elapsed_seconds"
        ),
        "epoch500_protocol": (
            "stop GPU training, preserve timer, evaluate three fixed unseen "
            "geometries, resume from epoch-500 archive"
        ),
        "method_lock": {
            "objective": "velocity_rayleigh",
            "modes": 2048,
            "epochs": 2000,
            "causal_sorting": "geometry_rayleigh",
            "peeling_passes": 1,
            "peeling_relative_mass_floor": float(
                config["peeling_relative_mass_floor"]
            ),
            "first_omega": float(config["siren_first_omega0"]),
            "hidden_omega": 30.0,
            "points_per_epoch": 4096,
        },
    }
    write_json(LAUNCH_MANIFEST, manifest)
    completion_path = RUN_DIR / "completion.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        if completion.get("complete") and int(completion.get("final_epoch", 0)) == 2000:
            verify_completion()
            manifest.update(status="complete", completed_at=now(), already_complete=True)
            write_json(LAUNCH_MANIFEST, manifest)
            return

    initial, fixed_validation = ensure_inputs()
    audit_initialization(initial)
    bandwidth = ensure_bandwidth_audit(initial, fixed_validation)
    manifest.update(
        status="training_to_epoch500",
        bandwidth_audit=str(BANDWIDTH_AUDIT.resolve()),
        bandwidth_min_relative_mass=min(
            float(row["min_relative_residual_mass"]) for row in bandwidth["rows"]
        ),
        bandwidth_max_overlap=max(
            float(row["max_post_peel_overlap"]) for row in bandwidth["rows"]
        ),
    )
    write_json(LAUNCH_MANIFEST, manifest)

    latest = RUN_DIR / "checkpoints/hardcoded_latest.npz"
    resume = latest if latest.exists() else initial
    trim_history_to_resume(resume)
    epoch500_checkpoint = run_epoch500_stage(resume, fixed_validation)
    manifest.update(
        status="epoch500_unseen_evaluation",
        epoch500_checkpoint=str(epoch500_checkpoint.resolve()),
        epoch500_training_paused_at=now(),
    )
    write_json(LAUNCH_MANIFEST, manifest)

    summary = run_epoch500_unseen(epoch500_checkpoint)
    pause_path = RUN_DIR / "planned_pause.json"
    preserved_pause = RUN_DIR / "planned_pause_epoch0500.json"
    if pause_path.exists() and not preserved_pause.exists():
        os.replace(pause_path, preserved_pause)
    manifest.update(
        status="training_to_epoch2000",
        epoch500_unseen_summary=str(summary.resolve()),
        resumed_after_epoch500_evaluation_at=now(),
    )
    write_json(LAUNCH_MANIFEST, manifest)

    started = time.perf_counter()
    try:
        live_status_path = RUN_DIR / "live_status.json"
        training_elapsed_floor = 0.0
        if live_status_path.exists():
            training_elapsed_floor = float(
                read_json(live_status_path).get("training_elapsed_seconds", 0.0)
            )
        training_elapsed_floor = max(
            training_elapsed_floor, RECOVERY_TRAINING_ELAPSED_FLOOR
        )
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/train_metaball.py"),
                "--resume",
                str(RUN_DIR / "checkpoints/hardcoded_latest.npz"),
                "--run-dir",
                str(RUN_DIR),
                "--validation-pack",
                str(fixed_validation),
                "--invalid-candidate-max-resamples",
                "8",
                "--training-elapsed-floor",
                f"{training_elapsed_floor:.17g}",
                "--extra-archive-epochs",
                "1269",
                "--peeling-relative-mass-floor",
                "1e-8",
            ]
        )
        verify_completion()
    except BaseException as error:
        manifest.update(
            status="failed",
            failed_at=now(),
            launcher_active_seconds=time.perf_counter() - started,
            error=repr(error),
        )
        write_json(LAUNCH_MANIFEST, manifest)
        raise
    manifest.update(
        status="complete",
        completed_at=now(),
        launcher_active_seconds=time.perf_counter() - started,
    )
    write_json(LAUNCH_MANIFEST, manifest)


if __name__ == "__main__":
    main()
