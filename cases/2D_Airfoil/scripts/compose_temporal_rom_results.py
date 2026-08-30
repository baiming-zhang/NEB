#!/usr/bin/env python3
"""Compose three-geometry K2048 temporal-ROM errors and timings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ORIENTATIONS = ("up", "middle", "down")
METHODS = ("potential", "fd_laplacian")
LABELS = {"potential": "Potential--Rayleigh", "fd_laplacian": "FD--Laplacian"}
COLORS = {"potential": "#003F88", "fd_laplacian": "#8E1B1B"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    case_root = Path(__file__).resolve().parents[1]
    default_root = (
        case_root
        / "results"
        / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
        / "temporal_rollout_gt30_k2048"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=default_root)
    args = parser.parse_args()
    root = args.root
    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    per_frame_rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, list[dict[str, str]]]] = {}
    for orientation in ORIENTATIONS:
        metrics[orientation] = {}
        for method in METHODS:
            payload = read_json(root / orientation / f"{method}_summary.json")
            rows = read_csv(root / orientation / f"{method}_temporal_metrics.csv")
            metrics[orientation][method] = rows
            aggregate = payload["aggregate_errors"]
            summary_rows.append(
                {
                    "geometry": orientation,
                    "method": method,
                    "modes": payload["modes"],
                    "initial_projection_error": payload["initial_projection_relative_l2"],
                    "velocity_mean_error": aggregate["velocity_error"]["mean"],
                    "velocity_final_error": aggregate["velocity_error"]["final"],
                    "vorticity_mean_error": aggregate["vorticity_error"]["mean"],
                    "vorticity_final_error": aggregate["vorticity_error"]["final"],
                    "fluctuation_velocity_mean_error": aggregate["fluctuation_velocity_error"]["mean"],
                    "fluctuation_velocity_final_error": aggregate["fluctuation_velocity_error"]["final"],
                    "fluctuation_vorticity_mean_error": aggregate["fluctuation_vorticity_error"]["mean"],
                    "fluctuation_vorticity_final_error": aggregate["fluctuation_vorticity_error"]["final"],
                    "wake_velocity_mean_error": aggregate["wake_velocity_error"]["mean"],
                    "wake_velocity_final_error": aggregate["wake_velocity_error"]["final"],
                    "wake_vorticity_mean_error": aggregate["wake_vorticity_error"]["mean"],
                    "wake_vorticity_final_error": aggregate["wake_vorticity_error"]["final"],
                }
            )
            timing = payload["timings"]
            timing_rows.append(
                {
                    "geometry": orientation,
                    "method": method,
                    "basis_field_seconds": timing["basis_field_seconds"],
                    "ritz_seconds": timing["ritz_seconds"],
                    "outlet_basis_seconds": timing["outlet_basis_seconds"],
                    "initial_projection_seconds": timing["initial_projection_seconds"],
                    "coefficient_rollout_30_frames_seconds": timing["coefficient_rollout_seconds"],
                    "coefficient_rollout_mean_per_frame_seconds": float(
                        np.mean(timing["coefficient_rollout_per_frame_seconds"])
                    ),
                    "coefficient_rollout_p95_per_frame_seconds": float(
                        np.percentile(timing["coefficient_rollout_per_frame_seconds"], 95.0)
                    ),
                    "reconstruction_seconds": timing["reconstruction_seconds"],
                    "metric_computation_seconds": timing["metric_computation_seconds"],
                    "serialization_seconds": timing["serialization_seconds"],
                    "total_method_seconds": timing["total_method_seconds"],
                }
            )
            for row, seconds in zip(
                rows, timing["coefficient_rollout_per_frame_seconds"], strict=True
            ):
                per_frame_rows.append(
                    {
                        "geometry": orientation,
                        "method": method,
                        **row,
                        "coefficient_rollout_frame_seconds": seconds,
                    }
                )
    write_csv(root / "temporal_error_summary.csv", summary_rows)
    write_csv(root / "temporal_timing_summary.csv", timing_rows)
    write_csv(root / "temporal_metrics_by_frame.csv", per_frame_rows)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.5), sharex=True)
    for row_index, orientation in enumerate(ORIENTATIONS):
        for column, key in enumerate(("velocity_error", "vorticity_error")):
            axis = axes[row_index, column]
            for method in METHODS:
                rows = metrics[orientation][method]
                axis.plot(
                    [int(item["frame"]) for item in rows],
                    [100.0 * float(item[key]) for item in rows],
                    color=COLORS[method],
                    linewidth=1.3,
                    marker="o",
                    markersize=2.0,
                    markevery=3,
                    label=LABELS[method],
                )
            axis.set_xlim(1, 30)
            axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
            axis.set_ylabel(rf"{orientation.capitalize()} error (\%)")
            axis.grid(color="#D9D9D9", linewidth=0.4, alpha=0.65)
            if row_index == 0:
                axis.set_title("Velocity" if column == 0 else "Vorticity")
    for axis in axes[-1]:
        axis.set_xlabel("Predicted frame")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96), pad=0.8)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"temporal_error_all_geometries.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)

    output = {
        "complete": True,
        "definition": "GT frame 0 is projected into each K2048 space and advanced to GT frames 1--30 with the same de Witt-type variational ROM protocol.",
        "error_summary_csv": str((root / "temporal_error_summary.csv").resolve()),
        "timing_summary_csv": str((root / "temporal_timing_summary.csv").resolve()),
        "per_frame_csv": str((root / "temporal_metrics_by_frame.csv").resolve()),
        "figure_pdf": str((root / "temporal_error_all_geometries.pdf").resolve()),
        "summary_rows": summary_rows,
        "timing_rows": timing_rows,
    }
    (root / "temporal_rollout_summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
