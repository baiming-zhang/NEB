"""Greedy local 500-epoch hyperparameter sweep for the K64 airfoil SIREN.

The network topology and physical problem are fixed.  Active continuous
training parameters are varied locally, one family at a time, and every arm
trains through epoch 500.  Selection uses only the fixed validation pack.  The
winning configuration is evaluated once on the independent unseen NS case and
compared with the original epoch-500 checkpoint and the same-domain airfoil FD
Laplacian baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import psutil


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parents[1]
BASE_CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_500.json"
BASELINE_RUN = CASE_ROOT / "results/siren_k64_dirichlet_psi_iid_dynamic_e500"
RESULT_ROOT = CASE_ROOT / "results/siren_k64_airfoil_hyperparameter_sweep_e500"
UNSEEN_GT = (
    WORKTREE_ROOT.parent.parent
    / "EigenFluid-SSS/cases/2D_airfoil/results/k128_c2_sobol_10000"
    / "run_001/postprocess/gt/gt_003.npz"
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

FIXED_PROBLEM_KEYS = (
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "chord",
    "quarter_chord_x",
    "quarter_chord_y",
    "thickness_min",
    "thickness_max",
    "geometry_mode",
    "alpha_min_deg",
    "alpha_max_deg",
    "envelope_mode",
    "sampling_mode",
    "quadrature_measure",
    "basis_objective",
    "spatial_derivative_order",
    "causal_sorting",
    "d4_symmetry_augmentation",
    "analytic_eigenmodes",
    "degenerate_block_direction_alignment",
    "rank_reporting",
    "seed",
    "epochs",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def gpu_blockers() -> list[dict[str, Any]]:
    needles = (
        "run_connected_k64_hyperparameter_sweep.py",
        "train_metaball.py",
        "run_connected_k64_strict_fairness.py",
        "compare_three_gt_eigenfluid_vs_laplacian.py",
        "evaluate_checkpoint_vs_laplacian.py",
    )
    blockers: list[dict[str, Any]] = []
    for process in psutil.process_iter(("pid", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if any(needle in command for needle in needles):
            blockers.append({"pid": int(process.info["pid"]), "command": command})
    return blockers


def wait_for_existing_gpu_sweep(manifest_path: Path) -> None:
    last_report = 0.0
    while True:
        blockers = gpu_blockers()
        if not blockers:
            print("GPU QUEUE released; starting airfoil sweep", flush=True)
            return
        current = read_json(manifest_path)
        current["status"] = "queued_waiting_for_existing_gpu_sweep"
        current["queue_blockers"] = blockers
        current["updated_at"] = now()
        write_json(manifest_path, current)
        if time.monotonic() - last_report >= 1800.0 or last_report == 0.0:
            print(
                "GPU QUEUE waiting for existing metaball sweep: "
                + ", ".join(str(item["pid"]) for item in blockers),
                flush=True,
            )
            last_report = time.monotonic()
        time.sleep(60.0)


def validation_metrics(run_dir: Path) -> dict[str, float | int]:
    history = read_json(run_dir / "data/hardcoded_training_history.json")
    rows = [
        row
        for row in history
        if int(row.get("epoch", 0)) == 500 and "validation_loss" in row
    ]
    if not rows:
        raise RuntimeError(f"missing epoch-500 validation row under {run_dir}")
    row = rows[-1]
    return {
        "final_epoch": 500,
        "final_validation_rank": int(row["validation_min_rank"]),
        "final_validation_loss": float(row["validation_loss"]),
        "final_validation_p95": float(row["validation_loss_p95"]),
        "final_validation_max": float(row["validation_loss_max"]),
        "final_validation_orth_error": float(row["validation_orth_error"]),
        "final_train_loss": float(row["train_loss"]),
        "final_learning_rate": float(row["learning_rate"]),
        "elapsed_seconds": float(row["seconds"]),
    }


def selection_key(arm: dict[str, Any]) -> tuple[int, int, float, float, float]:
    orth_error = float(arm["final_validation_orth_error"])
    return (
        -int(arm["final_validation_rank"]),
        int(not math.isfinite(orth_error) or orth_error > 1.0e-4),
        float(arm["final_validation_loss"]),
        float(arm["final_validation_p95"]),
        float(arm["final_validation_max"]),
    )


def trim_history_to_resume(run_dir: Path, resume: Path) -> None:
    resume_epoch = int(read_json(resume.with_suffix(".json"))["epoch"])
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


def validation_pack(inputs: Path) -> Path:
    matches = sorted(inputs.glob("fixed_validation_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one fixed validation pack under {inputs}")
    return matches[0]


def ensure_inputs(config_path: Path, arm_root: Path) -> tuple[Path, Path]:
    inputs = arm_root / "inputs"
    initial = inputs / "initial_epoch0000.npz"
    if not initial.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config",
                str(config_path),
                "--output-dir",
                str(inputs),
            ]
        )
    return initial, validation_pack(inputs)


def persist_arm(arm: dict[str, Any]) -> None:
    write_json(Path(arm["run_dir"]).parent / "arm_status.json", arm)


def train_arm(arm: dict[str, Any]) -> None:
    run_dir = Path(arm["run_dir"])
    completion_path = run_dir / "completion.json"
    if completion_path.exists():
        completion = read_json(completion_path)
        if completion.get("complete") and int(completion.get("final_epoch", 0)) >= 500:
            arm.update(validation_metrics(run_dir))
            arm["status"] = "complete"
            persist_arm(arm)
            print(f"SKIP completed {arm['name']}", flush=True)
            return

    arm["status"] = "preparing"
    arm["updated_at"] = now()
    persist_arm(arm)
    initial, fixed_validation = ensure_inputs(Path(arm["config"]), run_dir.parent)
    for attempt in range(1, 3):
        latest = run_dir / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        trim_history_to_resume(run_dir, resume)
        arm.update(
            {
                "status": "training",
                "attempt": attempt,
                "resume_checkpoint": str(resume.resolve()),
                "updated_at": now(),
            }
        )
        persist_arm(arm)
        try:
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts/train_hardcoded.py"),
                    "--resume",
                    str(resume),
                    "--run-dir",
                    str(run_dir),
                    "--epochs",
                    "500",
                    "--validation-pack",
                    str(fixed_validation),
                ]
            )
            break
        except Exception as error:
            arm["status"] = "retrying" if attempt == 1 else "failed"
            arm["error"] = repr(error)
            arm["updated_at"] = now()
            persist_arm(arm)
            if attempt == 2:
                raise
            time.sleep(10.0)
    arm.update(validation_metrics(run_dir))
    arm["status"] = "complete"
    arm["completed_at"] = now()
    arm.pop("error", None)
    persist_arm(arm)
    print(
        f"DONE {arm['name']} rank={arm['final_validation_rank']} "
        f"val={arm['final_validation_loss']:.9g}",
        flush=True,
    )


def assert_fixed(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    changed = {
        key: (reference.get(key), candidate.get(key))
        for key in (*ARCHITECTURE_KEYS, *FIXED_PROBLEM_KEYS)
        if reference.get(key) != candidate.get(key)
    }
    if changed:
        raise RuntimeError(f"sweep changed fixed architecture/problem fields: {changed}")


def slug(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}".replace("-", "m").replace(".", "p")
    return str(value).replace("_", "-")


def same_updates(config: dict[str, Any], updates: dict[str, Any]) -> bool:
    return all(config.get(key) == value for key, value in updates.items())


def scalar_candidates(
    key: str, values: tuple[Any, ...], prefix: str
) -> Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    def build(_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return [(f"{prefix}-{slug(value)}", {key: value}) for value in values]

    return build


def frequency_candidates(_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            f"first-{slug(first)}_hidden-{slug(hidden)}",
            {"siren_first_omega0": first, "siren_hidden_omega0": hidden},
        )
        for first in (6.0, 8.0, 10.0)
        for hidden in (24.0, 30.0, 36.0)
    ]


def learning_rate_candidates(_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (
            f"lr-{slug(value)}",
            {"learning_rate": value, "base_learning_rate": value},
        )
        for value in (1.5e-4, 2.0e-4, 2.5e-4)
    ]


def make_arm(
    reference: dict[str, Any],
    incumbent: dict[str, Any],
    result_root: Path,
    stage_index: int,
    stage_name: str,
    candidate_name: str,
    updates: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(incumbent)
    config.update(updates)
    config["epochs"] = 500
    assert_fixed(reference, config)
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


def baseline_arm(config_path: Path, run_dir: Path) -> dict[str, Any]:
    completion = read_json(run_dir / "completion.json")
    if not completion.get("complete") or int(completion["final_epoch"]) < 500:
        raise RuntimeError("original epoch-500 baseline is incomplete")
    arm: dict[str, Any] = {
        "name": "original_epoch500_baseline",
        "stage": "baseline",
        "updates": {},
        "config": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(
            (run_dir / "checkpoints/hardcoded_sss_frozen.npz").resolve()
        ),
        "status": "complete",
    }
    arm.update(validation_metrics(run_dir))
    return arm


def plot_sweep(manifest: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [arm for stage in manifest["stages"] for arm in stage["arms"]]
    values = [float(arm["final_validation_loss"]) for arm in arms]
    labels = [f"{arm['stage']}\n{arm['name']}" for arm in arms]
    winner = manifest["winner"]
    colors = ["#B2182B" if arm["run_dir"] == winner["run_dir"] else "#2166AC" for arm in arms]
    figure, axis = plt.subplots(figsize=(max(13.0, 0.48 * len(arms)), 5.8), constrained_layout=True)
    positions = list(range(len(arms)))
    axis.scatter(positions, values, c=colors, s=34, zorder=3)
    axis.axhline(
        float(manifest["baseline"]["final_validation_loss"]),
        color="#252525",
        linewidth=1.2,
        linestyle="--",
        label="Original epoch-500 baseline",
    )
    axis.set_xticks(positions, labels, rotation=75, ha="right", fontsize=7)
    axis.set_ylabel("Epoch-500 validation mean Rayleigh")
    axis.set_title("Airfoil K64 SIREN local hyperparameter sweep")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False)
    figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def evaluate_winner(
    winner: dict[str, Any], result_root: Path, baseline: dict[str, Any]
) -> dict[str, Any]:
    winner_checkpoint = Path(winner["run_dir"]) / "checkpoints/hardcoded_sss_frozen.npz"
    evaluation_dir = result_root / "winner_unseen_fair_airfoil_laplacian"
    summary_path = evaluation_dir / "unseen_evaluation_summary.json"
    if not summary_path.exists():
        run(
            [
                sys.executable,
                "-u",
                str(CASE_ROOT / "scripts/evaluate_unseen_k64_vs_laplacian.py"),
                "--checkpoint",
                str(winner_checkpoint),
                "--gt",
                str(UNSEEN_GT),
                "--output-dir",
                str(evaluation_dir),
                "--frame",
                "-1",
                "--peeling-points",
                "32768",
                "--projection-points",
                "65536",
                "--field-batch",
                "2048",
                "--laplacian-stride",
                "2",
            ]
        )
    winner_evaluation = read_json(summary_path)
    baseline_summary = read_json(
        BASELINE_RUN
        / "unseen_beta0p783156_alpha5p87785_fair_airfoil_laplacian"
        / "unseen_evaluation_summary.json"
    )
    metrics = (
        "velocity_error",
        "vorticity_error",
        "fluctuation_velocity_error",
        "fluctuation_vorticity_error",
        "wake_velocity_error",
        "wake_vorticity_error",
    )
    return {
        "selection_uses_unseen_case": False,
        "baseline": baseline,
        "winner": winner,
        "baseline_unseen": baseline_summary["learned"],
        "winner_unseen": winner_evaluation["learned"],
        "same_airfoil_fd_laplacian": winner_evaluation["laplacian"],
        "winner_relative_error_change_vs_original": {
            key: (
                winner_evaluation["learned"][key] / baseline_summary["learned"][key]
                - 1.0
            )
            for key in metrics
        },
        "winner_evaluation_summary": str(summary_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--skip-gpu-queue", action="store_true")
    args = parser.parse_args()

    base_path = args.base_config.resolve()
    baseline_run = args.baseline_run.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "sweep_manifest.json"
    base = read_json(base_path)
    if int(base["epochs"]) != 500 or int(base["modes"]) != 64:
        raise ValueError("sweep is fixed to K=64 and 500 epochs")
    baseline = baseline_arm(base_path, baseline_run)
    manifest: dict[str, Any] = read_json(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "status": "initializing",
            "started_at": manifest.get("started_at", now()),
            "updated_at": now(),
            "design": "greedy local coordinate sweep; every arm trains to epoch 500",
            "selection": (
                "maximize peeled rank, reject orthogonality error above 1e-4, "
                "then minimize epoch-500 fixed-validation mean, P95, and "
                "maximum Rayleigh"
            ),
            "baseline": baseline,
            "fixed_architecture": {key: base.get(key) for key in ARCHITECTURE_KEYS},
            "fixed_problem": {key: base.get(key) for key in FIXED_PROBLEM_KEYS},
            "swept_parameters": {
                "siren_frequencies": [[6, 8, 10], [24, 30, 36]],
                "initial_learning_rate": [1.5e-4, 2.0e-4, 2.5e-4],
                "cosine_min_learning_rate": [5e-7, 1e-6, 2e-6],
                "grad_clip": [2.5, 5.0, 10.0],
                "points_per_step": [3072, 4096, 5120],
                "endpoint_period": [10, 20, 40],
                "reorthogonalization_passes": [1, 2, 3],
            },
            "not_swept": {
                "width/depth/activation/input_encoding": "network form fixed",
                "scheduler_type": "cosine form fixed",
                "validation/checkpoint cadence": "bookkeeping only",
                "mode batches": "throughput only",
                "mass_floor": "inactive numerical guard",
                "dtype": "FP32 protocol fixed",
                "validation_points": "metric precision, not training",
                "coordinate_normalization": (
                    "currently inactive in the airfoil hardcoded encoding; "
                    "the active centered-unit scale is swept through first omega"
                ),
            },
            "result_root": str(result_root),
        }
    )
    write_json(manifest_path, manifest)
    if not args.skip_gpu_queue:
        wait_for_existing_gpu_sweep(manifest_path)

    incumbent_arm = baseline
    incumbent_config = dict(base)
    stages: list[
        tuple[str, Callable[[dict[str, Any]], list[tuple[str, dict[str, Any]]]]]
    ] = [
        ("frequency", frequency_candidates),
        ("learning_rate", learning_rate_candidates),
        ("cosine_min_lr", scalar_candidates("min_learning_rate", (5e-7, 1e-6, 2e-6), "minlr")),
        ("grad_clip", scalar_candidates("grad_clip", (2.5, 5.0, 10.0), "clip")),
        ("points_per_step", scalar_candidates("points_per_step", (3072, 4096, 5120), "points")),
        ("endpoint_period", scalar_candidates("endpoint_period", (10, 20, 40), "period")),
        ("reorthogonalization", scalar_candidates("reorthogonalization_passes", (1, 2, 3), "passes")),
    ]
    completed_stages: list[dict[str, Any]] = []
    manifest["status"] = "running"
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
            write_json(manifest_path, manifest)
            train_arm(arm)
            manifest["updated_at"] = now()
            write_json(manifest_path, manifest)
        winner = min([incumbent_arm, *stage["arms"]], key=selection_key)
        if winner is not incumbent_arm:
            incumbent_arm = winner
            incumbent_config = arm_configs[winner["name"]]
        stage.update(
            {
                "winner": winner,
                "improved_over_incumbent": winner is not stage["incumbent_before"],
                "status": "complete",
                "completed_at": now(),
            }
        )
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
        "validation_improvement_percent": 100.0 * (baseline_loss - winner_loss) / baseline_loss,
        "all_arms_train_to_epoch": 500,
        "network_architecture_changed": False,
    }
    manifest["winner"] = incumbent_arm
    manifest["status"] = "training_complete_running_independent_evaluation"
    manifest["current_stage"] = None
    write_json(result_root / "comparison_summary.json", comparison)
    write_json(manifest_path, manifest)
    plot_sweep(manifest, result_root / "sweep_validation_comparison")
    independent = evaluate_winner(incumbent_arm, result_root, baseline)
    write_json(result_root / "best_vs_original_epoch500.json", independent)
    manifest["status"] = "complete"
    manifest["completed_at"] = now()
    manifest["updated_at"] = now()
    manifest["independent_evaluation"] = independent
    write_json(manifest_path, manifest)
    print(json.dumps({**comparison, "independent_evaluation": independent}, indent=2), flush=True)


if __name__ == "__main__":
    main()
