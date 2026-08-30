"""Run Original airfoil K2048 to epoch 2000, then extend the fair K64-K2048 evaluation."""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import run_airfoil_original_k128_to_k1024_e2000 as base


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parents[1]
CONFIG = CASE_ROOT / "configs/k2048_airfoil_siren_original_2000.json"
BASE_CONFIG = CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_2000.json"
RESULT_ROOT = CASE_ROOT / "results/siren_original_k2048_e2000"
RUN_DIR = RESULT_ROOT / "run_001"
STATE_PATH = RESULT_ROOT / "pipeline_state.json"
MANIFEST_PATH = RESULT_ROOT / "pipeline_manifest.json"
TARGET_EPOCH = 2000
TARGET_MODES = 2048
ARCHIVE_EPOCHS = (500, 1000, 1500, 2000)
ALL_K_VALUES = (64, 128, 256, 512, 1024, 2048)

PREVIOUS_SEQUENCE_ROOT = CASE_ROOT / "results/siren_original_k64_to_k1024_e2000_sequence"
EVALUATION_ROOT = PREVIOUS_SEQUENCE_ROOT / "evaluation_three_geometries_three_frames"
EVALUATION_COMPLETE = EVALUATION_ROOT / "comparison_complete.json"
LAPLACIAN_CACHE_ROOT = PREVIOUS_SEQUENCE_ROOT / "shared_geometry_laplacian_cache_k2048"
EVALUATION_SCRIPT = CASE_ROOT / "scripts/evaluate_original_k64_to_k1024_vs_laplacian.py"
SUMMARY_SCRIPT = CASE_ROOT / "scripts/compose_original_laplacian_results.py"
GT_ROOT = Path(
    r"C:\Users\admin\Desktop\Eigenbasis\EigenFluid-SSS\cases\2D_airfoil\results\k128_c2_sobol_10000\run_001\postprocess\gt"
)
CHECKPOINTS = {
    64: CASE_ROOT
    / "results/siren_k64_dirichlet_psi_iid_dynamic_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz",
    128: CASE_ROOT
    / "results/siren_original_k128_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz",
    256: CASE_ROOT
    / "results/siren_original_k256_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz",
    512: CASE_ROOT
    / "results/siren_original_k512_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz",
    1024: CASE_ROOT
    / "results/siren_original_k1024_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz",
    2048: RUN_DIR / "checkpoints/hardcoded_sss_frozen.npz",
}


def configure_base() -> None:
    base.MODE_COUNTS = (TARGET_MODES,)
    base.CONFIGS = {TARGET_MODES: CONFIG}
    base.SEQUENCE_ROOT = RESULT_ROOT
    base.STATE_PATH = STATE_PATH
    base.MANIFEST_PATH = MANIFEST_PATH
    base.TARGET_EPOCH = TARGET_EPOCH


def update_manifest(**updates: Any) -> dict[str, Any]:
    manifest = base.read_json(MANIFEST_PATH)
    manifest.update(updates)
    manifest["updated_at"] = base.now()
    manifest["updated_unix_ns"] = time.time_ns()
    base.write_json(MANIFEST_PATH, manifest)
    return manifest


def validate_config() -> dict[str, Any]:
    reference = base.read_json(BASE_CONFIG)
    config = base.read_json(CONFIG)
    differences = {
        key: (reference.get(key), config.get(key))
        for key in sorted(set(reference) | set(config))
        if reference.get(key) != config.get(key)
    }
    if set(differences) != {"modes", "result_root"}:
        raise RuntimeError(f"K2048 differs from Original in unexpected fields: {differences}")
    required = {
        "modes": TARGET_MODES,
        "epochs": TARGET_EPOCH,
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
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"K2048 Original config mismatch: {mismatches}")
    return config


def create_manifest(config: dict[str, Any]) -> None:
    if MANIFEST_PATH.exists():
        manifest = base.read_json(MANIFEST_PATH)
        if manifest.get("config_sha256") != base.sha256(CONFIG):
            raise RuntimeError("existing K2048 manifest config hash does not match")
        return
    created_at = base.now()
    created_ns = time.time_ns()
    result_root = WORKTREE_ROOT / str(config["result_root"])
    manifest = {
        "experiment": "airfoil_original_k2048_e2000_with_k64_to_k2048_fair_evaluation",
        "status": "queued",
        "created_at": created_at,
        "created_unix_ns": created_ns,
        "config": str(CONFIG.resolve()),
        "config_sha256": base.sha256(CONFIG),
        "config_snapshot": config,
        "base_original_config": str(BASE_CONFIG.resolve()),
        "base_original_config_sha256": base.sha256(BASE_CONFIG),
        "only_permitted_config_changes": ["modes", "result_root"],
        "fresh_epoch0": True,
        "archive_epochs": list(ARCHIVE_EPOCHS),
        "strict_timing": {
            "wall_clock": "time.time_ns",
            "elapsed_clock": "time.perf_counter_ns",
            "queue_excluded_from_training_time": True,
            "history_fields": [
                "seconds",
                "training_elapsed_seconds",
                "wall_time_iso",
                "wall_time_unix_ns",
            ],
        },
        "arms": [
            {
                "modes": TARGET_MODES,
                "status": "pending",
                "config": str(CONFIG.resolve()),
                "config_sha256": base.sha256(CONFIG),
                "result_root": str(result_root.resolve()),
                "run_dir": str((result_root / "run_001").resolve()),
                "archive_epochs": list(ARCHIVE_EPOCHS),
            }
        ],
        "evaluation": {
            "status": "pending",
            "k_values": list(ALL_K_VALUES),
            "geometries": ["up", "middle", "down"],
            "frames": [30],
            "sample_count": 3,
            "output_root": str(EVALUATION_ROOT.resolve()),
            "laplacian_cache_root": str(LAPLACIAN_CACHE_ROOT.resolve()),
            "fd_policy": "one geometry-aware K2048 eigensolve per geometry, nested prefixes reused for all K",
            "reuse_lower_learned_peeling_states": True,
            "metrics": [
                "velocity_error",
                "vorticity_error",
                "fluctuation_velocity_error",
                "fluctuation_vorticity_error",
                "wake_velocity_error",
                "wake_vorticity_error",
            ],
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu": base.gpu_snapshot(),
        },
    }
    base.write_json(MANIFEST_PATH, manifest)
    base.update_state(
        status="queued",
        created_at=created_at,
        created_unix_ns=created_ns,
        target_modes=TARGET_MODES,
        target_epoch=TARGET_EPOCH,
        result_root=str(RESULT_ROOT.resolve()),
        run_dir=str(RUN_DIR.resolve()),
        completed_arms=[],
        planned_arms=[TARGET_MODES],
    )


def evaluation_complete() -> bool:
    if not EVALUATION_COMPLETE.exists():
        return False
    summary = base.read_json(EVALUATION_COMPLETE)
    protocol = summary.get("protocol", {})
    return (
        bool(summary.get("complete"))
        and tuple(protocol.get("k_values", [])) == ALL_K_VALUES
        and tuple(protocol.get("frames", [])) == (30,)
        and int(summary.get("metric_rows", 0)) == 3 * 2 * len(ALL_K_VALUES)
    )


def validate_evaluation_outputs() -> dict[str, Any]:
    if not evaluation_complete():
        raise RuntimeError("K64-K2048 evaluation completion contract is not satisfied")
    required = [
        EVALUATION_ROOT / "reconstruction_metrics.csv",
        EVALUATION_ROOT / "original_vs_airfoil_laplacian_k64_to_k2048_speed.pdf",
        EVALUATION_ROOT / "original_vs_airfoil_laplacian_k64_to_k2048_vorticity.pdf",
        EVALUATION_ROOT / "original_vs_laplacian_three_cases_combined.png",
        EVALUATION_ROOT / "original_vs_laplacian_three_cases_combined.pdf",
        EVALUATION_ROOT / "reconstruction_summary_by_k.csv",
        EVALUATION_ROOT / "reconstruction_summary_by_case.csv",
        EVALUATION_ROOT / "reconstruction_summary.json",
        EVALUATION_ROOT / "reconstruction_error_summary_all_metrics.png",
        EVALUATION_ROOT / "reconstruction_error_summary_all_metrics.pdf",
    ]
    for orientation in ("up", "middle", "down"):
        required.extend(
            [
                EVALUATION_ROOT / f"{orientation}_frame30_speed_2x7.png",
                EVALUATION_ROOT / f"{orientation}_frame30_vorticity_2x7.png",
                EVALUATION_ROOT / orientation / "basis_state/learned_k2048_state.npz",
                EVALUATION_ROOT / orientation / "basis_state/learned_k2048_state.json",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing K2048 evaluation outputs: {missing}")
    summary = base.read_json(EVALUATION_COMPLETE)
    wall_seconds = float(summary["evaluation_wall_seconds"])
    if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise FloatingPointError(f"invalid evaluation wall time: {wall_seconds}")
    return {
        "comparison_complete": str(EVALUATION_COMPLETE.resolve()),
        "evaluation_wall_seconds": wall_seconds,
        "metric_rows": int(summary["metric_rows"]),
        "required_outputs_verified": len(required),
    }


def evaluate() -> None:
    if evaluation_complete():
        try:
            details = validate_evaluation_outputs()
        except FileNotFoundError:
            base.run([sys.executable, str(SUMMARY_SCRIPT), str(EVALUATION_ROOT)])
            details = validate_evaluation_outputs()
        update_manifest(status="complete", evaluation={**base.read_json(MANIFEST_PATH)["evaluation"], "status": "complete", **details})
        base.update_state(status="complete", **details)
        print("SKIP existing complete K64-K2048 evaluation", flush=True)
        return
    missing_checkpoints = [str(path) for path in CHECKPOINTS.values() if not path.exists()]
    if missing_checkpoints:
        raise FileNotFoundError(f"missing evaluation checkpoints: {missing_checkpoints}")
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
        "--frames",
        "30",
        "--k-values",
        *[str(value) for value in ALL_K_VALUES],
        "--laplacian-cache-root",
        str(LAPLACIAN_CACHE_ROOT),
    ]
    for modes in ALL_K_VALUES:
        command.extend([f"--k{modes}-checkpoint", str(CHECKPOINTS[modes])])
    overall_timer = time.perf_counter_ns()
    first_started_at: str | None = None
    first_started_ns: int | None = None
    for attempt in range(1, 4):
        attempt_started_at = base.now()
        attempt_started_ns = time.time_ns()
        attempt_timer = time.perf_counter_ns()
        if first_started_at is None:
            first_started_at = attempt_started_at
            first_started_ns = attempt_started_ns
        evaluation = {
            **base.read_json(MANIFEST_PATH)["evaluation"],
            "status": "running",
            "attempt": attempt,
            "started_at": first_started_at,
            "started_unix_ns": first_started_ns,
            "attempt_started_at": attempt_started_at,
            "attempt_started_unix_ns": attempt_started_ns,
        }
        update_manifest(status="evaluation", evaluation=evaluation)
        base.update_state(
            status="evaluation",
            current_arm=TARGET_MODES,
            evaluation_attempt=attempt,
            evaluation_started_at=first_started_at,
            evaluation_started_unix_ns=first_started_ns,
            evaluation_attempt_started_at=attempt_started_at,
            evaluation_attempt_started_unix_ns=attempt_started_ns,
        )
        try:
            if evaluation_complete():
                print("REUSE completed K64-K2048 evaluator output", flush=True)
            else:
                base.run(command)
            if not evaluation_complete():
                raise RuntimeError("evaluator exited without the K2048 completion contract")
            base.run([sys.executable, str(SUMMARY_SCRIPT), str(EVALUATION_ROOT)])
            details = validate_evaluation_outputs()
            completed_at = base.now()
            completed_ns = time.time_ns()
            details.update(
                {
                    "completed_at": completed_at,
                    "completed_unix_ns": completed_ns,
                    "pipeline_evaluation_process_wall_seconds": (
                        time.perf_counter_ns() - overall_timer
                    )
                    / 1.0e9,
                }
            )
            update_manifest(
                status="complete",
                completed_at=completed_at,
                completed_unix_ns=completed_ns,
                evaluation={**evaluation, "status": "complete", **details},
            )
            base.update_state(status="complete", **details)
            print("DONE Original K2048 E2000 and K64-K2048 fair evaluation", flush=True)
            return
        except Exception as error:
            status = "evaluation_retrying" if attempt < 3 else "failed"
            failure = {
                "status": status,
                "attempt": attempt,
                "error": repr(error),
                "attempt_wall_seconds": (time.perf_counter_ns() - attempt_timer) / 1.0e9,
                "failed_at": base.now(),
                "failed_unix_ns": time.time_ns(),
            }
            update_manifest(status=status, evaluation={**evaluation, **failure})
            base.update_state(**failure)
            if attempt == 3:
                raise
            time.sleep(30.0)


def main() -> None:
    configure_base()
    config = validate_config()
    create_manifest(config)
    for modes, checkpoint in CHECKPOINTS.items():
        if modes != TARGET_MODES and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    base.wait_for_gpu()
    base.train_arm(TARGET_MODES, config)
    update_manifest(status="evaluation")
    evaluate()


if __name__ == "__main__":
    configure_base()
    try:
        main()
    except Exception as error:
        if MANIFEST_PATH.exists():
            update_manifest(status="failed", error=repr(error), failed_at=base.now())
        base.update_state(status="failed", error=repr(error), failed_at=base.now())
        raise
