"""Run a resumable local K=64, 500-epoch hyperparameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any


CASE_ROOT = Path(__file__).resolve().parents[1]
TRAINER = CASE_ROOT / "scripts" / "train_validate.py"
BASE_CONFIG = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
BASELINE_RUN = CASE_ROOT / "results" / "k64_siren_dirichlet_e500" / "run_001"
DEFAULT_ROOT = CASE_ROOT / "results" / "k64_siren_dirichlet_e500_hyperparameter_sweep"

SWEEP_VALUES: dict[str, tuple[float | int, float | int]] = {
    "learning_rate": (1.4e-4, 2.8e-4),
    "min_learning_rate": (5.0e-7, 2.0e-6),
    "siren_first_omega0": (6.0, 10.0),
    "siren_hidden_omega0": (24.0, 36.0),
    "grad_clip": (3.0, 8.0),
    "points_per_step": (3072, 6144),
    "outer_envelope_transition": (0.12, 0.18),
    "obstacle_envelope_transition": (0.08, 0.12),
}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def slug_value(value: float | int) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def stage_a_arms(base: dict[str, Any]) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for parameter, values in SWEEP_VALUES.items():
        for value in values:
            if value == base[parameter]:
                raise ValueError(f"sweep value duplicates baseline: {parameter}={value}")
            arms.append(
                {
                    "name": f"ofat_{parameter}_{slug_value(value)}",
                    "stage": "A",
                    "changes": {parameter: value},
                }
            )
    return arms


def read_completion(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "completion.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("complete") else None


def score_from_completion(completion: dict[str, Any] | None) -> tuple[float, float, float]:
    if completion is None:
        return math.inf, math.inf, math.inf
    validation = completion.get("validation") or {}
    if int(validation.get("min_peeled_rank", 0)) != 64:
        return math.inf, math.inf, math.inf
    values = (
        float(validation.get("mean_loss", math.inf)),
        float(validation.get("max_loss", math.inf)),
        float(validation.get("max_peeled_orthogonality_abs", math.inf)),
    )
    return values if all(math.isfinite(value) for value in values) else (math.inf,) * 3


def arm_record(arm: dict[str, Any], root: Path) -> dict[str, Any]:
    run_dir = root / arm["name"] / "run"
    completion = read_completion(run_dir)
    validation = {} if completion is None else completion.get("validation") or {}
    return {
        "name": arm["name"],
        "stage": arm["stage"],
        "changes": arm["changes"],
        "complete": completion is not None,
        "epochs_completed": 0 if completion is None else int(completion.get("epochs_completed", 0)),
        "wall_seconds": None if completion is None else completion.get("wall_seconds"),
        "validation_mean_rayleigh": validation.get("mean_loss"),
        "validation_max_rayleigh": validation.get("max_loss"),
        "min_peeled_rank": validation.get("min_peeled_rank"),
        "orthogonality_max_abs": validation.get("max_peeled_orthogonality_abs"),
        "run_dir": str(run_dir.resolve()),
    }


def baseline_record() -> dict[str, Any]:
    completion = read_completion(BASELINE_RUN)
    validation = {} if completion is None else completion.get("validation") or {}
    return {
        "name": "baseline_run_001",
        "stage": "baseline",
        "changes": {},
        "complete": completion is not None,
        "epochs_completed": 0 if completion is None else int(completion.get("epochs_completed", 0)),
        "wall_seconds": None if completion is None else completion.get("wall_seconds"),
        "validation_mean_rayleigh": validation.get("mean_loss"),
        "validation_max_rayleigh": validation.get("max_loss"),
        "min_peeled_rank": validation.get("min_peeled_rank"),
        "orthogonality_max_abs": validation.get("max_peeled_orthogonality_abs"),
        "run_dir": str(BASELINE_RUN.resolve()),
    }


def write_manifest(root: Path, arms: list[dict[str, Any]], status: str, current: str | None) -> None:
    records = [baseline_record(), *(arm_record(arm, root) for arm in arms)]
    complete = sum(bool(record["complete"]) for record in records) - 1
    payload = {
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_reused": True,
        "current_arm": current,
        "candidate_count": len(arms),
        "completed_candidates": complete,
        "fixed_contract": {
            "modes": 64,
            "epochs": 500,
            "hidden": 32,
            "hidden_layers": 4,
            "metaball_softmin_tau": 0.02,
            "reorthogonalization_passes": 2,
            "reporting_scope": "peeled_basis_only",
        },
        "records": records,
    }
    atomic_json(root / "sweep_manifest.json", payload)
    csv_path = root / "sweep_results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    flat_rows = []
    for record in records:
        flat_rows.append(
            {
                **{key: value for key, value in record.items() if key != "changes"},
                "changes": json.dumps(record["changes"], sort_keys=True),
            }
        )
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    os.replace(temporary, csv_path)


def materialize_config(base: dict[str, Any], arm: dict[str, Any], root: Path) -> Path:
    payload = deepcopy(base)
    payload.update(arm["changes"])
    payload["result_root"] = str((root / arm["name"]).resolve())
    config_path = root / arm["name"] / "config.json"
    atomic_json(config_path, payload)
    return config_path


def run_arm(base: dict[str, Any], arm: dict[str, Any], root: Path, maximum_restarts: int) -> None:
    run_dir = root / arm["name"] / "run"
    if read_completion(run_dir) is not None:
        print(f"skip complete arm={arm['name']}", flush=True)
        return
    config_path = materialize_config(base, arm, root)
    checkpoint = run_dir / "checkpoints" / "latest.pt"
    for attempt in range(maximum_restarts + 1):
        arguments = [
            sys.executable,
            str(TRAINER),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ]
        if checkpoint.exists():
            arguments.extend(("--resume", str(checkpoint)))
        print(
            f"launch arm={arm['name']} stage={arm['stage']} attempt={attempt + 1} "
            f"resume={checkpoint.exists()}",
            flush=True,
        )
        completed = subprocess.run(arguments, cwd=CASE_ROOT.parents[1], check=False)
        if completed.returncode == 0 and read_completion(run_dir) is not None:
            return
        restart_log = run_dir / "restart.log"
        restart_log.parent.mkdir(parents=True, exist_ok=True)
        with restart_log.open("a", encoding="utf-8") as stream:
            stream.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] exit={completed.returncode} "
                f"attempt={attempt + 1}\n"
            )
    raise RuntimeError(f"arm failed after restarts: {arm['name']}")


def stage_b_arms(base: dict[str, Any], stage_a: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    baseline_score = score_from_completion(read_completion(BASELINE_RUN))
    best_values: dict[str, float | int] = {}
    improvements: list[tuple[float, str]] = []
    for parameter in SWEEP_VALUES:
        candidates = [arm for arm in stage_a if parameter in arm["changes"]]
        scored = [
            (score_from_completion(read_completion(root / arm["name"] / "run")), arm)
            for arm in candidates
        ]
        score, winner = min(scored, key=lambda item: item[0])
        if score < baseline_score:
            best_values[parameter] = winner["changes"][parameter]
            improvements.append((baseline_score[0] - score[0], parameter))
        else:
            best_values[parameter] = base[parameter]
            improvements.append((0.0, parameter))
    improvements.sort(reverse=True)
    optimizer_parameters = {
        "learning_rate",
        "min_learning_rate",
        "grad_clip",
        "points_per_step",
    }
    representation_parameters = set(SWEEP_VALUES) - optimizer_parameters
    change_sets = [
        {key: value for key, value in best_values.items() if value != base[key]},
        {
            key: best_values[key]
            for key in optimizer_parameters
            if best_values[key] != base[key]
        },
        {
            key: best_values[key]
            for key in representation_parameters
            if best_values[key] != base[key]
        },
        {
            key: best_values[key]
            for _, key in improvements[:4]
            if best_values[key] != base[key]
        },
    ]
    arms: list[dict[str, Any]] = []
    signatures: set[str] = set()
    names = ("joint_best_all", "joint_optimizer", "joint_representation", "joint_top4")
    for name, changes in zip(names, change_sets):
        signature = json.dumps(changes, sort_keys=True)
        if not changes or signature in signatures:
            continue
        signatures.add(signature)
        arms.append({"name": name, "stage": "B", "changes": changes})
    atomic_json(
        root / "stage_b_selection.json",
        {
            "baseline_score": baseline_score,
            "best_values": best_values,
            "ranked_parameter_improvements": improvements,
            "joint_arms": arms,
        },
    )
    return arms[:4]


def select_winner(base: dict[str, Any], arms: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    candidates: list[tuple[tuple[float, float, float], dict[str, Any], Path]] = [
        (score_from_completion(read_completion(BASELINE_RUN)), {"name": "baseline_run_001", "stage": "baseline", "changes": {}}, BASELINE_RUN)
    ]
    for arm in arms:
        run_dir = root / arm["name"] / "run"
        candidates.append((score_from_completion(read_completion(run_dir)), arm, run_dir))
    score, winner, run_dir = min(candidates, key=lambda item: item[0])
    config = deepcopy(base)
    config.update(winner["changes"])
    result = {
        "selected_without_final_unseen_geometry": True,
        "selection_metric": "epoch500 fixed-validation mean Rayleigh, max Rayleigh, orthogonality",
        "winner_name": winner["name"],
        "winner_stage": winner["stage"],
        "winner_changes": winner["changes"],
        "winner_score": score,
        "winner_run_dir": str(run_dir.resolve()),
        "winner_checkpoint": str((run_dir / "checkpoints" / "final.pt").resolve()),
        "winner_config": config,
    }
    atomic_json(root / "winner.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--maximum-restarts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    stage_a = stage_a_arms(base)
    all_arms = list(stage_a)
    write_manifest(root, all_arms, "stage_a_running", None)
    for arm in stage_a:
        write_manifest(root, all_arms, "stage_a_running", arm["name"])
        run_arm(base, arm, root, args.maximum_restarts)
        write_manifest(root, all_arms, "stage_a_running", None)

    stage_b = stage_b_arms(base, stage_a, root)
    all_arms.extend(stage_b)
    write_manifest(root, all_arms, "stage_b_running", None)
    for arm in stage_b:
        write_manifest(root, all_arms, "stage_b_running", arm["name"])
        run_arm(base, arm, root, args.maximum_restarts)
        write_manifest(root, all_arms, "stage_b_running", None)

    winner = select_winner(base, all_arms, root)
    write_manifest(root, all_arms, "training_complete", None)
    print(json.dumps(winner, indent=2), flush=True)


if __name__ == "__main__":
    main()
