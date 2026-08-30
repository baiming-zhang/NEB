"""Greedy local 500-epoch hyperparameter sweep for connected metaballs.

The network architecture, geometry family, variational objective, sampling
distribution, dynamic causal sorting, and peeled rank are held fixed.  Only
active optimization/frequency/preprocessing parameters are varied.  Every
candidate is compared against the incumbent on its full epoch-500 fixed
validation result; completed arms are skipped and interrupted arms resume
from ``hardcoded_latest.npz``.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parents[1]
DEFAULT_BASE_CONFIG = (
    CASE_ROOT
    / "configs/k64_metaball_connected_siren_dirichlet_psi_generic_random_dynamic_500.json"
)
DEFAULT_BASELINE_RUN = (
    CASE_ROOT
    / "results/siren_k64_connected_dirichlet_psi_generic_random_dynamic_e500/run_002"
)
DEFAULT_RESULT_ROOT = (
    CASE_ROOT / "results/siren_k64_connected_hyperparameter_sweep_e500"
)

ARCHITECTURE_KEYS = (
    "modes",
    "activation",
    "canonical_siren",
    "input_encoding",
    "input_features",
    "hidden",
    "hidden_layers",
    "mode_forward_batch",
    "mode_backward_batch",
    "mlp_compute_dtype",
    "mlp_accumulation_dtype",
)

PROBLEM_KEYS = (
    "case_domain",
    "geometry_mode",
    "metaball_connectivity",
    "metaball_minimum_overlap",
    "metaball_outer_clearance",
    "metaball_softmin_tau",
    "outer_envelope_transition",
    "obstacle_envelope_transition",
    "envelope_mode",
    "basis_objective",
    "sampling_mode",
    "quadrature_measure",
    "causal_sorting",
    "spatial_derivative_order",
    "seed",
    "epochs",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def validation_metrics(run_dir: Path) -> dict[str, float | int]:
    history_path = run_dir / "data/hardcoded_training_history.json"
    history = read_json(history_path)
    rows = [
        row
        for row in history
        if int(row.get("epoch", 0)) == 500 and "validation_loss" in row
    ]
    if not rows:
        raise RuntimeError(f"missing epoch-500 validation row: {history_path}")
    row = rows[-1]
    return {
        "final_epoch": 500,
        "final_validation_rank": int(row.get("validation_min_rank", 0)),
        "final_validation_loss": float(row["validation_loss"]),
        "final_validation_p95": float(row["validation_loss_p95"]),
        "final_validation_max": float(row["validation_loss_max"]),
        "final_validation_orth_error": float(row["validation_orth_error"]),
        "final_train_loss": float(row["train_loss"]),
        "final_learning_rate": float(row["learning_rate"]),
    }


def selection_key(arm: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        -int(arm["final_validation_rank"]),
        float(arm["final_validation_loss"]),
        float(arm["final_validation_p95"]),
        float(arm["final_validation_max"]),
    )


def validation_pack(inputs: Path) -> Path:
    matches = sorted(inputs.glob("fixed_validation_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one fixed validation pack under {inputs}, got {matches}"
        )
    return matches[0]


def ensure_inputs(config_path: Path, arm_root: Path) -> tuple[Path, Path]:
    inputs = arm_root / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    if not initial.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_metaball_run.py"),
                "--config",
                str(config_path),
                "--output-dir",
                str(inputs),
            ]
        )
    return initial, validation_pack(inputs)


def trim_history_to_resume(run_dir: Path, resume: Path) -> None:
    """Keep history consistent with the checkpoint used after interruption."""
    metadata_path = resume.with_suffix(".json")
    resume_epoch = int(read_json(metadata_path).get("epoch", 0))
    history_path = run_dir / "data/hardcoded_training_history.json"
    if not history_path.exists():
        return
    history = read_json(history_path)
    kept = [row for row in history if int(row.get("epoch", 0)) <= resume_epoch]
    if len(kept) == len(history):
        return
    backup = history_path.with_name(
        f"recovery_history_before_epoch{resume_epoch:04d}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    write_json(backup, history)
    write_json(history_path, kept)
    print(
        f"TRIM history {len(history)} -> {len(kept)} rows for resume epoch "
        f"{resume_epoch}; backup={backup}",
        flush=True,
    )


def train_arm(arm: dict[str, Any], manifest_path: Path) -> None:
    run_dir = Path(arm["run_dir"])
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        if completion.get("complete") and int(completion.get("final_epoch", 0)) >= 500:
            arm.update(validation_metrics(run_dir))
            arm["status"] = "complete"
            print(f"SKIP completed {arm['name']}", flush=True)
            return

    arm["status"] = "preparing"
    arm["updated_at"] = now()
    initial, fixed_validation = ensure_inputs(Path(arm["config"]), run_dir.parent)
    latest = run_dir / "checkpoints/hardcoded_latest.npz"
    resume = latest if latest.exists() else initial
    trim_history_to_resume(run_dir, resume)
    arm["status"] = "training"
    arm["resume_checkpoint"] = str(resume.resolve())
    arm["updated_at"] = now()
    # The caller persists the enclosing manifest before entering this function;
    # this marker is still useful in the arm-local config/result tree.
    write_json(run_dir.parent / "arm_status.json", arm)
    try:
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/train_metaball.py"),
                "--resume",
                str(resume),
                "--run-dir",
                str(run_dir),
                "--validation-pack",
                str(fixed_validation),
            ]
        )
    except Exception as error:
        arm["status"] = "failed"
        arm["error"] = repr(error)
        arm["updated_at"] = now()
        write_json(run_dir.parent / "arm_status.json", arm)
        raise
    arm.update(validation_metrics(run_dir))
    arm["status"] = "complete"
    arm["completed_at"] = now()
    arm.pop("error", None)
    write_json(run_dir.parent / "arm_status.json", arm)
    print(
        f"DONE {arm['name']} rank={arm['final_validation_rank']} "
        f"val={arm['final_validation_loss']:.9g}",
        flush=True,
    )


def assert_fixed(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    changed = {
        key: (reference.get(key), candidate.get(key))
        for key in (*ARCHITECTURE_KEYS, *PROBLEM_KEYS)
        if reference.get(key) != candidate.get(key)
    }
    if changed:
        raise RuntimeError(f"sweep changed fixed architecture/problem fields: {changed}")


def slug(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}".replace("-", "m").replace(".", "p").replace("+", "")
    return str(value).replace("_", "-")


def make_arm(
    fixed_reference: dict[str, Any],
    incumbent_config: dict[str, Any],
    result_root: Path,
    stage_index: int,
    stage_name: str,
    candidate_name: str,
    updates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(incumbent_config)
    config.update(updates)
    config["epochs"] = 500
    assert_fixed(fixed_reference, config)
    arm_root = result_root / f"{stage_index:02d}_{stage_name}" / candidate_name
    config["result_root"] = str(arm_root.resolve())
    config_path = arm_root / "config.json"
    write_json(config_path, config)
    arm = {
        "name": candidate_name,
        "stage": stage_name,
        "updates": updates,
        "config": str(config_path.resolve()),
        "run_dir": str((arm_root / "run_001").resolve()),
        "status": "pending",
    }
    return arm, config


def parameter_value(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config[key] if key in config else default


def same_updates(config: dict[str, Any], updates: dict[str, Any]) -> bool:
    return all(parameter_value(config, key) == value for key, value in updates.items())


def frequency_candidates(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    first_values = (6.0, 8.0, 10.0)
    hidden_values = (24.0, 30.0, 36.0)
    return [
        (
            f"first-{slug(first)}_hidden-{slug(hidden)}",
            {"siren_first_omega0": first, "siren_hidden_omega0": hidden},
        )
        for first in first_values
        for hidden in hidden_values
    ]


def lr_candidates(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            f"lr-{slug(value)}",
            {"learning_rate": value, "base_learning_rate": value},
        )
        for value in (1.5e-4, 2.0e-4, 2.5e-4)
    ]


def scheduler_candidates(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"scheduler-{value}", {"lr_scheduler_type": value})
        for value in ("cosine", "constant", "validation_plateau")
    ]


def scheduler_detail_candidates(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    scheduler = str(config["lr_scheduler_type"])
    if scheduler == "cosine":
        return [
            (f"minlr-{slug(value)}", {"min_learning_rate": value})
            for value in (3.0e-7, 1.0e-6, 3.0e-6)
        ]
    if scheduler == "validation_plateau":
        current = {
            "lr_scheduler_patience_epochs": int(config["lr_scheduler_patience_epochs"]),
            "lr_scheduler_factor": float(config["lr_scheduler_factor"]),
        }
        return [
            ("plateau-fast", {**current, "lr_scheduler_patience_epochs": 25}),
            ("plateau-baseline", current),
            ("plateau-gentle", {**current, "lr_scheduler_factor": 0.7}),
        ]
    return [("constant-no-detail", {})]


def scalar_candidates(
    key: str, values: tuple[Any, ...], prefix: str
) -> Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    def build(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        del config
        return [(f"{prefix}-{slug(value)}", {key: value}) for value in values]

    return build


def normalization_candidates(
    config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    # Preserve the first-layer spatial phase scale while changing the raw
    # normalization convention: scale(x) * first_omega stays constant.
    scales = {
        "unit_interval": 1.0,
        "centered_unit": 2.0,
        "standardized_uniform": math.sqrt(12.0),
    }
    current_name = str(config["coordinate_normalization"])
    effective = scales[current_name] * float(config["siren_first_omega0"])
    return [
        (
            f"norm-{name}",
            {
                "coordinate_normalization": name,
                "siren_first_omega0": effective / scale,
            },
        )
        for name, scale in scales.items()
    ]


def baseline_arm(config_path: Path, run_dir: Path) -> dict[str, Any]:
    completion = read_json(run_dir / "completion.json")
    if not completion.get("complete") or int(completion.get("final_epoch", 0)) < 500:
        raise RuntimeError("the requested original 500-epoch baseline is incomplete")
    arm: dict[str, Any] = {
        "name": "original_epoch500_baseline",
        "stage": "baseline",
        "updates": {},
        "config": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "status": "complete",
        "checkpoint": str((run_dir / "checkpoints/hardcoded_sss_frozen.npz").resolve()),
    }
    arm.update(validation_metrics(run_dir))
    return arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()

    base_path = args.base_config.resolve()
    baseline_run = args.baseline_run.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "sweep_manifest.json"
    base = read_json(base_path)
    if int(base["epochs"]) != 500 or int(base["modes"]) != 64:
        raise ValueError("this sweep is fixed to K=64 and 500 epochs")

    baseline = baseline_arm(base_path, baseline_run)
    manifest: dict[str, Any] = (
        read_json(manifest_path) if manifest_path.exists() else {}
    )
    manifest.update(
        {
            "status": "running",
            "started_at": manifest.get("started_at", now()),
            "updated_at": now(),
            "design": "greedy local coordinate sweep; every arm trains through epoch 500",
            "selection": (
                "maximize epoch-500 peeled validation rank, then minimize mean, "
                "P95, and maximum streamfunction Rayleigh quotient"
            ),
            "baseline": baseline,
            "fixed_architecture": {key: base.get(key) for key in ARCHITECTURE_KEYS},
            "fixed_problem_and_algorithm": {key: base.get(key) for key in PROBLEM_KEYS},
            "inactive_or_nonlearning_fields_not_swept": {
                "initial_*_gain": "inactive under canonical SIREN initialization",
                "siren_first_layer_spatial_omega0": "inactive under canonical SIREN encoding",
                "mode_*_batch": "performance batching only",
                "checkpoint/validation cadence": "bookkeeping only for cosine/constant runs",
                "mass_floor": "finite-arithmetic guard",
                "geometry/envelope parameters": "define the physical problem, not learning",
            },
            "result_root": str(result_root),
        }
    )
    write_json(manifest_path, manifest)

    incumbent_arm = baseline
    incumbent_config = dict(base)
    stages: list[
        tuple[str, Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]]]]
    ] = [
        ("frequency", frequency_candidates),
        ("learning_rate", lr_candidates),
        ("scheduler", scheduler_candidates),
        ("scheduler_detail", scheduler_detail_candidates),
        ("grad_clip", scalar_candidates("grad_clip", (2.5, 5.0, 10.0), "clip")),
        (
            "reorthogonalization",
            scalar_candidates("reorthogonalization_passes", (1, 2, 3), "passes"),
        ),
        (
            "points_per_step",
            # K=64, batch=32 at 4096 points already occupies about 15.8 GiB
            # on the 16-GiB RTX 4080 SUPER.  Keep batching fixed and sweep
            # downward rather than confounding the experiment with a batch
            # change or knowingly scheduling an OOM arm.
            scalar_candidates("points_per_step", (3072, 3584, 4096), "points"),
        ),
        ("normalization", normalization_candidates),
    ]

    completed_stages: list[dict[str, Any]] = []
    for stage_index, (stage_name, builder) in enumerate(stages, start=1):
        candidates = [
            (name, updates)
            for name, updates in builder(incumbent_config)
            if not same_updates(incumbent_config, updates)
        ]
        stage: dict[str, Any] = {
            "index": stage_index,
            "name": stage_name,
            "status": "running",
            "incumbent_before": incumbent_arm,
            "arms": [],
            "started_at": now(),
        }
        manifest["current_stage"] = stage_name
        manifest["stages"] = completed_stages + [stage]
        write_json(manifest_path, manifest)

        arm_configs: dict[str, dict[str, Any]] = {}
        for candidate_name, updates in candidates:
            arm, arm_config = make_arm(
                base,
                incumbent_config,
                result_root,
                stage_index,
                stage_name,
                candidate_name,
                updates,
            )
            stage["arms"].append(arm)
            arm_configs[arm["name"]] = arm_config
            manifest["stages"] = completed_stages + [stage]
            write_json(manifest_path, manifest)
            train_arm(arm, manifest_path)
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)

        winner = min([incumbent_arm, *stage["arms"]], key=selection_key)
        if winner is not incumbent_arm:
            incumbent_arm = winner
            incumbent_config = arm_configs[winner["name"]]
        stage["winner"] = winner
        stage["improved_over_incumbent"] = winner is not stage["incumbent_before"]
        stage["status"] = "complete"
        stage["completed_at"] = now()
        completed_stages.append(stage)
        manifest["stages"] = completed_stages
        manifest["incumbent"] = incumbent_arm
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)

    baseline_loss = float(baseline["final_validation_loss"])
    winner_loss = float(incumbent_arm["final_validation_loss"])
    comparison = {
        "baseline": baseline,
        "winner": incumbent_arm,
        "winner_effective_config": incumbent_config,
        "absolute_validation_loss_change": winner_loss - baseline_loss,
        "relative_validation_loss_change": (winner_loss - baseline_loss) / baseline_loss,
        "improvement_percent": 100.0 * (baseline_loss - winner_loss) / baseline_loss,
        "all_arms_train_to_epoch": 500,
        "network_architecture_changed": False,
    }
    write_json(result_root / "comparison_summary.json", comparison)
    manifest["status"] = "training_complete_pending_independent_evaluation"
    manifest["current_stage"] = None
    manifest["winner"] = incumbent_arm
    manifest["completed_at"] = now()
    manifest["updated_at"] = now()
    write_json(manifest_path, manifest)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
