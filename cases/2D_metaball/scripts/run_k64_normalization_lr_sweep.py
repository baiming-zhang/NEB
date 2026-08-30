"""Run the 500-epoch K=64 normalization-by-learning-rate metaball sweep."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    CASE_ROOT
    / "configs/k64_metaball_siren_centered_unit_lr2em4_cosine_500.json"
)
DEFAULT_ROOT = CASE_ROOT / "results/siren_k64_norm_lr_sweep_e500_q4096"
NORMALIZATIONS = {
    "unit_interval": 16.0,
    "centered_unit": 8.0,
    "standardized_uniform": 16.0 / math.sqrt(12.0),
}
LEARNING_RATES = (1.0e-4, 2.0e-4, 4.0e-4)
HIDDEN_OMEGAS = (1.0, 4.0, 8.0, 16.0, 30.0, 64.0)


def label(value: float) -> str:
    return {1.0e-4: "1em04", 2.0e-4: "2em04", 4.0e-4: "4em04"}[value]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=CASE_ROOT.parents[1], check=True)


def validation_metrics(run_dir: Path) -> dict[str, float | int]:
    history_path = run_dir / "data/hardcoded_training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    validated = [row for row in history if "validation_loss" in row]
    if not validated:
        return {
            "final_validation_rank": 0,
            "final_validation_loss": math.inf,
            "best_validation_loss": math.inf,
        }
    final = validated[-1]
    return {
        "final_validation_rank": int(final.get("validation_min_rank", 0)),
        "final_validation_loss": float(final["validation_loss"]),
        "best_validation_loss": min(float(row["validation_loss"]) for row in validated),
    }


def selection_key(arm: dict) -> tuple[int, float]:
    return (-int(arm["final_validation_rank"]), float(arm["final_validation_loss"]))


def preflight_arm(arm: dict) -> None:
    run_dir = Path(arm["run_dir"])
    arm_root = run_dir.parent
    inputs = arm_root / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    validation = inputs / "fixed_validation_scipy_sobol_v1.npz"
    if not initial.exists() or not validation.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_metaball_run.py"),
                "--config",
                arm["config"],
                "--output-dir",
                str(inputs),
            ]
        )
    preflight_dir = arm_root / "preflight"
    log_path = preflight_dir / "hardcoded_train.log"
    if not log_path.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/train_metaball.py"),
                "--resume",
                str(initial),
                "--run-dir",
                str(preflight_dir),
                "--validation-pack",
                str(validation),
                "--self-test-epochs",
                "1",
                "--self-test-points",
                "4096",
            ]
        )
    match = re.findall(r"rank=(\d+)", log_path.read_text(encoding="utf-8"))
    arm["preflight_rank"] = int(match[-1]) if match else 0


def train_arm(arm: dict) -> None:
    run_dir = Path(arm["run_dir"])
    completion = run_dir / "completion.json"
    if completion.exists():
        completed = json.loads(completion.read_text(encoding="utf-8"))
        if completed.get("complete") and int(completed.get("final_epoch", 0)) >= 500:
            print(f"SKIP completed {arm['name']}", flush=True)
            arm.update(validation_metrics(run_dir))
            return
    arm_root = run_dir.parent
    inputs = arm_root / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    validation = inputs / "fixed_validation_scipy_sobol_v1.npz"
    if not initial.exists() or not validation.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_metaball_run.py"),
                "--config",
                arm["config"],
                "--output-dir",
                str(inputs),
            ]
        )
    latest = run_dir / "checkpoints/hardcoded_latest.npz"
    resume = latest if latest.exists() else initial
    run(
        [
            sys.executable,
            str(CASE_ROOT / "scripts/train_metaball.py"),
            "--resume",
            str(resume),
            "--run-dir",
            str(run_dir),
            "--validation-pack",
            str(validation),
        ]
    )
    arm["completed_at"] = datetime.now().astimezone().isoformat()
    arm.update(validation_metrics(run_dir))


def make_arm(
    base: dict,
    stage_root: Path,
    name: str,
    normalization: str,
    first_omega: float,
    hidden_omega: float,
    learning_rate: float,
) -> dict:
    arm_root = stage_root / name
    config = dict(base)
    config.update(
        {
            "coordinate_normalization": normalization,
            "siren_first_omega0": first_omega,
            "siren_hidden_omega0": hidden_omega,
            "learning_rate": learning_rate,
            "base_learning_rate": learning_rate,
            "result_root": str(arm_root),
        }
    )
    config_path = arm_root / "config.json"
    write_json(config_path, config)
    return {
        "name": name,
        "normalization": normalization,
        "normalization_scale": 16.0 / first_omega,
        "siren_first_omega0": first_omega,
        "siren_hidden_omega0": hidden_omega,
        "effective_spatial_frequency": 16.0,
        "learning_rate": learning_rate,
        "config": str(config_path),
        "run_dir": str(arm_root / "run_001"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "modes": 64,
        "epochs": 500,
        "points_per_step": 4096,
        "design": {
            "stage_1": "hidden omega sweep at centered_unit and LR=2e-4",
            "stage_2": "normalization by LR sweep using the best hidden omega",
        },
        "fixed": {
            "hidden": 32,
            "hidden_layers": 4,
            "lr_scheduler_type": "cosine",
            "effective_spatial_frequency": 16.0,
        },
        "hidden_omega_candidates": list(HIDDEN_OMEGAS),
    }
    manifest_path = result_root / "sweep_manifest.json"
    write_json(manifest_path, manifest)

    stage_1 = [
        make_arm(
            base,
            result_root / "01_hidden_omega_e500",
            f"hidden_omega_{str(value).replace('.', 'p')}",
            "centered_unit",
            8.0,
            value,
            2.0e-4,
        )
        for value in HIDDEN_OMEGAS
    ]
    manifest["stage_1_arms"] = stage_1
    write_json(manifest_path, manifest)
    for arm in stage_1:
        preflight_arm(arm)
        write_json(manifest_path, manifest)
    for index, arm in enumerate(stage_1, start=1):
        train_arm(arm)
        manifest["stage_1_completed_arms"] = index
        write_json(manifest_path, manifest)

    winner_1 = min(stage_1, key=selection_key)
    best_hidden_omega = float(winner_1["siren_hidden_omega0"])
    manifest["stage_1_winner"] = winner_1
    manifest["selected_hidden_omega0"] = best_hidden_omega
    write_json(manifest_path, manifest)

    stage_2 = []
    for normalization, first_omega in NORMALIZATIONS.items():
        for learning_rate in LEARNING_RATES:
            stage_2.append(
                make_arm(
                    base,
                    result_root / "02_normalization_lr_e500",
                    f"norm_{normalization}_lr{label(learning_rate)}",
                    normalization,
                    first_omega,
                    best_hidden_omega,
                    learning_rate,
                )
            )
    manifest["stage_2_arms"] = stage_2
    write_json(manifest_path, manifest)
    for arm in stage_2:
        preflight_arm(arm)
        write_json(manifest_path, manifest)
    for index, arm in enumerate(stage_2, start=1):
        train_arm(arm)
        manifest["stage_2_completed_arms"] = index
        write_json(manifest_path, manifest)

    winner_2 = min(stage_2, key=selection_key)
    manifest["stage_2_winner"] = winner_2
    manifest["overall_winner"] = min(
        [*stage_1, *stage_2], key=selection_key
    )

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
