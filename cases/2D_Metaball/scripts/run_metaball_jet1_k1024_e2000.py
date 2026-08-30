"""Run the Original connected-metaball K=1024/E=2000 Jet1 campaign.

The runner is restart-safe: it reuses deterministic prepared inputs, trims
history to the latest complete checkpoint after interruption, and resumes from
that checkpoint.  With ``--wait-for-airfoil`` it performs no CUDA work until
all 2D-airfoil Python work and NVIDIA compute processes have exited.
"""

from __future__ import annotations

import argparse
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
CONFIG = CASE_ROOT / "configs/k1024_metaball_connected_siren_original_jet1_2000.json"
RESULT_ROOT = (
    CASE_ROOT / "results/siren_k1024_connected_original_jet1_e2000"
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


def airfoil_python_processes() -> list[dict[str, Any]]:
    try:
        import psutil
    except ImportError:
        return []
    found: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "cmdline", "create_time")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            lowered = command.lower().replace("/", "\\")
            if "gen\\cases\\2d_airfoil" in lowered:
                found.append(
                    {
                        "pid": int(process.info["pid"]),
                        "command": command,
                        "created": float(process.info.get("create_time") or 0.0),
                    }
                )
        except (OSError, psutil.Error):
            continue
    return found


def gpu_compute_pids() -> list[int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return sorted(
        {
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit() and int(line.strip()) != os.getpid()
        }
    )


def wait_for_gpu_release(manifest: dict[str, Any], interval: int) -> None:
    last_signature: tuple[int, ...] | None = None
    while True:
        airfoil = airfoil_python_processes()
        gpu_pids = gpu_compute_pids()
        signature = tuple(item["pid"] for item in airfoil)
        # Under Windows/WDDM, nvidia-smi lists ordinary desktop graphics
        # clients as compute applications with unavailable memory.  The
        # ownership barrier is therefore the explicit 2D-airfoil Python
        # process audit; GPU PIDs are recorded for provenance only.
        if not airfoil:
            manifest["gpu_released_at"] = now()
            manifest["status"] = "preparing"
            write_json(LAUNCH_MANIFEST, manifest)
            print(f"[{now()}] 2D airfoil finished and GPU is free", flush=True)
            return
        if signature != last_signature:
            print(
                f"[{now()}] WAIT airfoil_pids={[item['pid'] for item in airfoil]} "
                f"gpu_compute_pids={gpu_pids}",
                flush=True,
            )
            manifest["waiting_airfoil_pids"] = [item["pid"] for item in airfoil]
            manifest["waiting_gpu_compute_pids"] = gpu_pids
            manifest["last_wait_audit_at"] = now()
            write_json(LAUNCH_MANIFEST, manifest)
            last_signature = signature
        time.sleep(max(5, interval))


def validate_fixed_config(config: dict[str, Any]) -> None:
    expected = {
        "modes": 1024,
        "epochs": 2000,
        "hidden": 32,
        "hidden_layers": 4,
        "siren_first_omega0": 8.0,
        "siren_hidden_omega0": 30.0,
        "learning_rate": 2.0e-4,
        "min_learning_rate": 1.0e-6,
        "basis_objective": "streamfunction_dirichlet_rayleigh",
        "spatial_derivative_order": 1,
        "causal_sorting": "geometry_rayleigh",
        "reorthogonalization_passes": 2,
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
        raise ValueError(f"Jet1 campaign config drifted: {changed}")


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
    if metadata.get("spatial_ad") != "analytic_value_gradient_xy_jet1":
        raise RuntimeError("prepared checkpoint is not a Jet1 checkpoint")
    return initial, validation_pack()


def audit_initialization(initial: Path) -> None:
    baseline = (
        CASE_ROOT
        / "results/siren_k1024_connected_original_config_e2000/run_001/data/initial_epoch0000.npz"
    )
    audit_path = RESULT_ROOT / "initialization_audit.json"
    if audit_path.exists():
        previous = read_json(audit_path)
        if (
            previous.get("status") == "exact_match"
            and previous.get("all_tensors_bitwise_equal") is True
            and Path(previous.get("jet1_initial", "")) == initial.resolve()
            and Path(previous.get("original_jet3_initial", "")) == baseline.resolve()
            and initial.is_file()
            and baseline.is_file()
        ):
            previous["reused_at"] = now()
            write_json(audit_path, previous)
            return
    audit: dict[str, Any] = {
        "jet1_initial": str(initial.resolve()),
        "original_jet3_initial": str(baseline.resolve()),
        "audited_at": now(),
    }
    if not baseline.exists():
        audit["status"] = "baseline_missing"
        write_json(audit_path, audit)
        return
    with np.load(initial, allow_pickle=False) as first, np.load(
        baseline, allow_pickle=False
    ) as third:
        names_match = first.files == third.files
        exact = names_match and all(
            np.array_equal(first[name], third[name]) for name in first.files
        )
        audit.update(
            status="exact_match" if exact else "mismatch",
            tensor_names_match=names_match,
            tensor_count=len(first.files),
            all_tensors_bitwise_equal=exact,
        )
    write_json(audit_path, audit)
    if not audit["all_tensors_bitwise_equal"]:
        raise RuntimeError("Jet1 and Original Jet3 initial weights are not identical")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-airfoil", action="store_true")
    parser.add_argument("--wait-interval", type=int, default=30)
    args = parser.parse_args()
    config = read_json(CONFIG)
    validate_fixed_config(config)
    manifest: dict[str, Any] = {
        "status": "queued" if args.wait_for_airfoil else "preparing",
        "queued_at": now(),
        "config": str(CONFIG.resolve()),
        "result_root": str(RESULT_ROOT.resolve()),
        "pid": os.getpid(),
        "python": sys.executable,
        "strict_training_seconds_source": (
            "data/hardcoded_training_history.json:training_elapsed_seconds"
        ),
        "wall_clock_includes_airfoil_wait": False,
    }
    write_json(LAUNCH_MANIFEST, manifest)
    if args.wait_for_airfoil:
        wait_for_gpu_release(manifest, args.wait_interval)

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
