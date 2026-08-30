"""Run and aggregate the post-sweep strict matched-grid fairness evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = (
    HERE.parent
    / "results"
    / "siren_k64_connected_hyperparameter_sweep_e500"
)
COMPARE = HERE / "compare_three_gt_eigenfluid_vs_laplacian.py"
RESOLUTIONS = (192, 256, 320)
TOKENS = (4_000_001, 4_000_002, 4_000_003)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def valid_result(path: Path, checkpoint: Path, resolution: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("comparison_status") == "strict_matched_grid_fairness"
        and Path(result["checkpoint"]).resolve() == checkpoint.resolve()
        and int(result["grid_resolution"]) == resolution
        and tuple(result["geometry_tokens"]) == TOKENS
    )


def update_manifest(path: Path, **updates: Any) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    manifest["updated_at"] = now()
    write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest_path = root / "sweep_manifest.json"
    comparison_path = root / "comparison_summary.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    checkpoints = {
        "winner": Path(comparison["winner"]["run_dir"])
        / "checkpoints"
        / "hardcoded_sss_frozen.npz",
        "original_run_002": Path(comparison["baseline"]["checkpoint"]),
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    output_root = root / "09_strict_matched_grid_fairness"
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "evaluation_progress.json"
    aggregate_json = output_root / "strict_matched_grid_fairness_aggregate.json"
    aggregate_csv = output_root / "strict_matched_grid_fairness_aggregate.csv"
    spectrum_json = output_root / "peeled_ritz_first16_spectrum.json"
    spectrum_csv = output_root / "peeled_ritz_first16_spectrum.csv"
    best_json = output_root / "best_configuration_vs_original_epoch500.json"
    best_csv = output_root / "best_configuration_vs_original_epoch500.csv"
    stdout_log = output_root / "evaluation.stdout.log"
    stderr_log = output_root / "evaluation.stderr.log"

    update_manifest(
        manifest_path,
        status="independent_evaluation_running",
        independent_evaluation={
            "started_at": now(),
            "runner": str(Path(__file__).resolve()),
            "resolutions": list(RESOLUTIONS),
            "tokens": list(TOKENS),
            "output_root": str(output_root),
        },
    )

    jobs = [
        (label, resolution, checkpoint)
        for resolution in RESOLUTIONS
        for label, checkpoint in checkpoints.items()
    ]
    progress = {
        "status": "running",
        "started_at": now(),
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "jobs": [],
    }
    write_json(progress_path, progress)
    try:
        for label, resolution, checkpoint in jobs:
            output = output_root / label / f"resolution_{resolution}"
            output.mkdir(parents=True, exist_ok=True)
            summary = output / "three_gt_reconstruction_summary.json"
            job = {
                "label": label,
                "resolution": resolution,
                "checkpoint": str(checkpoint.resolve()),
                "output_dir": str(output),
            }
            if valid_result(summary, checkpoint, resolution):
                job["status"] = "skipped_complete"
            else:
                job["status"] = "running"
                job["started_at"] = now()
                progress["current_job"] = job
                write_json(progress_path, progress)
                command = [
                    sys.executable,
                    str(COMPARE),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(output),
                    "--tokens",
                    *(str(token) for token in TOKENS),
                    "--resolution",
                    str(resolution),
                    "--peeling-points",
                    "16384",
                    "--seed",
                    "20260823",
                ]
                with stdout_log.open("a", encoding="utf-8") as stdout, stderr_log.open(
                    "a", encoding="utf-8"
                ) as stderr:
                    stdout.write(f"[{now()}] START {label} resolution={resolution}\n")
                    stdout.flush()
                    subprocess.run(
                        command,
                        cwd=HERE,
                        stdout=stdout,
                        stderr=stderr,
                        check=True,
                    )
                    stdout.write(f"[{now()}] COMPLETE {label} resolution={resolution}\n")
                job["status"] = "complete"
                job["completed_at"] = now()
            progress["jobs"].append(job)
            progress["completed_jobs"] = len(progress["jobs"])
            progress.pop("current_job", None)
            write_json(progress_path, progress)

        results: dict[str, dict[str, Any]] = {}
        rows: list[dict[str, Any]] = []
        spectra: list[dict[str, Any]] = []
        metric_names = (
            "test_psi_relative_l2",
            "test_velocity_relative_l2",
            "test_vorticity_relative_l2",
        )
        for label, resolution, checkpoint in jobs:
            result_path = (
                output_root
                / label
                / f"resolution_{resolution}"
                / "three_gt_reconstruction_summary.json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.setdefault(label, {})[str(resolution)] = result
            for case in result["cases"]:
                for method, metrics in case["metrics"].items():
                    rows.append(
                        {
                            "checkpoint_label": label,
                            "resolution": resolution,
                            "case": case["case"],
                            "geometry_token": case["geometry_token"],
                            "method": method,
                            **{name: metrics[name] for name in metric_names},
                        }
                    )
                spectrum_sets = {
                    "Eigenfluid_peeled_grid_Ritz": case[
                        "eigenfluid_matched_grid_ritz"
                    ]["first16_eigenvalues"],
                    "Laplacian": case["laplacian"]["first16_eigenvalues"],
                }
                for method, values in spectrum_sets.items():
                    for mode, value in enumerate(values, start=1):
                        spectra.append(
                            {
                                "checkpoint_label": label,
                                "resolution": resolution,
                                "case": case["case"],
                                "geometry_token": case["geometry_token"],
                                "method": method,
                                "mode": mode,
                                "eigenvalue": value,
                            }
                        )

        aggregate = {
            "status": "complete",
            "completed_at": now(),
            "comparison_status": "strict_matched_grid_fairness",
            "resolutions": list(RESOLUTIONS),
            "geometry_tokens": list(TOKENS),
            "checkpoints": {key: str(value.resolve()) for key, value in checkpoints.items()},
            "fairness_contract": results["winner"]["320"]["fairness_contract"],
            "old_mixed_derivative_metrics": (
                "provisional_only; excluded from every conclusion in this aggregate"
            ),
            "laplacian_convergence_review": {
                str(resolution): results["winner"][str(resolution)]["mean_test_errors"][
                    "Laplacian"
                ]
                for resolution in RESOLUTIONS
            },
            "winner_vs_original_at_320": {
                label: results[label]["320"]["mean_test_errors"]
                for label in checkpoints
            },
            "results": results,
        }
        write_json(aggregate_json, aggregate)
        with aggregate_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(spectrum_json, {"spectra": spectra})
        with spectrum_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(spectra[0]))
            writer.writeheader()
            writer.writerows(spectra)

        best = {
            "winner": comparison["winner"],
            "baseline": comparison["baseline"],
            "winner_effective_config": comparison["winner_effective_config"],
            "validation_improvement_percent": comparison["improvement_percent"],
            "strict_fairness_resolution_320": aggregate["winner_vs_original_at_320"],
        }
        write_json(best_json, best)
        config_rows = [
            {"parameter": key, "winner_value": value}
            for key, value in comparison["winner_effective_config"].items()
        ]
        with best_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["parameter", "winner_value"])
            writer.writeheader()
            writer.writerows(config_rows)

        progress["status"] = "complete"
        progress["completed_at"] = now()
        write_json(progress_path, progress)
        update_manifest(
            manifest_path,
            status="complete",
            independent_evaluation={
                "status": "complete",
                "completed_at": now(),
                "aggregate_json": str(aggregate_json),
                "aggregate_csv": str(aggregate_csv),
                "spectrum_json": str(spectrum_json),
                "spectrum_csv": str(spectrum_csv),
                "best_configuration_json": str(best_json),
                "best_configuration_csv": str(best_csv),
            },
        )
    except BaseException as error:
        progress["status"] = "failed"
        progress["failed_at"] = now()
        progress["error"] = repr(error)
        write_json(progress_path, progress)
        update_manifest(
            manifest_path,
            status="independent_evaluation_failed",
            independent_evaluation={
                "status": "failed",
                "failed_at": now(),
                "error": repr(error),
                "progress": str(progress_path),
            },
        )
        raise


if __name__ == "__main__":
    main()
