"""Run connected-metaball velocity-Rayleigh K=1024/E=2000 training.

The campaign uses geometry-Rayleigh causal sorting and exactly one velocity-L2
projection pass.  A collapsed/nonorthogonal candidate aborts before the GPU
optimizer transaction.  The runner is deterministic, restart-safe at each
500-epoch checkpoint, and preserves a monotonic strict-training timer.
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
    "k1024_metaball_connected_siren_velocity_rayleigh_omega128_onepass_2000.json"
)
RESULT_ROOT = CASE_ROOT / (
    "results/siren_k1024_connected_velocity_rayleigh_omega128_onepass_e2000"
)
RUN_DIR = RESULT_ROOT / "run_001"
INPUTS = RESULT_ROOT / "inputs"
LAUNCH_MANIFEST = RESULT_ROOT / "launch_manifest.json"


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
        "modes": 1024,
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
        "peeling_relative_mass_floor": 1.0e-4,
        "peeling_overlap_tolerance": 1.0e-3,
        "sampling_mode": "uniform_random",
        "points_per_step": 4096,
        "checkpoint_every": 500,
        "archive_checkpoint_every": 500,
    }
    changed = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if changed:
        raise ValueError(f"velocity-Rayleigh campaign config drifted: {changed}")


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
    if metadata.get("spatial_ad") != "analytic_compact_symmetric_xy_jet3":
        raise RuntimeError("prepared velocity checkpoint is not analytic Jet3")
    if metadata.get("config", {}).get("basis_objective") != "velocity_rayleigh":
        raise RuntimeError("prepared checkpoint has the wrong objective")
    return initial, validation_pack()


def audit_initialization(initial: Path) -> None:
    baseline = CASE_ROOT / (
        "results/siren_k1024_connected_original_config_e2000/"
        "run_001/data/initial_epoch0000.npz"
    )
    audit_path = RESULT_ROOT / "initialization_audit.json"
    audit: dict[str, Any] = {
        "velocity_initial": str(initial.resolve()),
        "original_initial": str(baseline.resolve()),
        "audited_at": now(),
        "expected_difference": (
            "identical parameter tensors; forward bandwidth changes only through "
            f"siren_first_omega0 8 -> {read_json(CONFIG)['siren_first_omega0']:g}"
        ),
    }
    if not baseline.exists():
        audit["status"] = "baseline_missing"
        write_json(audit_path, audit)
        return
    with np.load(initial, allow_pickle=False) as first, np.load(
        baseline, allow_pickle=False
    ) as second:
        names_match = first.files == second.files
        exact = names_match and all(
            np.array_equal(first[name], second[name]) for name in first.files
        )
        audit.update(
            status="exact_match" if exact else "mismatch",
            tensor_names_match=names_match,
            tensor_count=len(first.files),
            all_tensors_bitwise_equal=exact,
        )
    write_json(audit_path, audit)
    if not exact:
        raise RuntimeError("velocity and Original initial parameter tensors differ")


def checkpoint_epoch(path: Path) -> int:
    return int(read_json(path.with_suffix(".json"))["epoch"])


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
    print(
        f"[{now()}] trimmed history {len(history)} -> {len(kept)} for epoch {epoch}",
        flush=True,
    )


def verify_completion() -> None:
    completion = read_json(RUN_DIR / "completion.json")
    if not completion.get("complete") or int(completion.get("final_epoch", 0)) != 2000:
        raise RuntimeError(f"incomplete training record: {completion}")
    if completion.get("basis_objective") != "velocity_rayleigh":
        raise RuntimeError("completion objective is not velocity-Rayleigh")
    required = [
        RUN_DIR / f"checkpoints/hardcoded_epoch{epoch:04d}.npz"
        for epoch in (500, 1000, 1500, 2000)
    ] + [RUN_DIR / "checkpoints/hardcoded_sss_frozen.npz"]
    missing = [str(path) for path in required if not path.is_file()]
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
        "method_lock": {
            "objective": "velocity_rayleigh",
            "modes": 1024,
            "epochs": 2000,
            "causal_sorting": "geometry_rayleigh",
            "peeling_passes": 1,
            "invalid_candidate_policy": "reject_before_optimizer_transaction",
            "first_omega": float(config["siren_first_omega0"]),
            "hidden_omega": 30.0,
        },
    }
    write_json(LAUNCH_MANIFEST, manifest)
    completion = RUN_DIR / "completion.json"
    if completion.exists():
        value = read_json(completion)
        if value.get("complete") and int(value.get("final_epoch", 0)) == 2000:
            verify_completion()
            manifest.update(status="complete", completed_at=now(), already_complete=True)
            write_json(LAUNCH_MANIFEST, manifest)
            return

    initial, fixed_validation = ensure_inputs()
    audit_initialization(initial)
    latest = RUN_DIR / "checkpoints/hardcoded_latest.npz"
    resume = latest if latest.exists() else initial
    trim_history_to_resume(resume)
    manifest.update(
        status="training",
        training_process_started_at=now(),
        resume_checkpoint=str(resume.resolve()),
        resume_epoch=checkpoint_epoch(resume),
        fixed_validation=str(fixed_validation.resolve()),
    )
    write_json(LAUNCH_MANIFEST, manifest)
    started = time.perf_counter()
    try:
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
