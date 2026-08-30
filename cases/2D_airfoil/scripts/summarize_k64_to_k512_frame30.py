#!/usr/bin/env python3
"""Summarize the three unseen frame-30 reconstructions across K and method."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("velocity_error", "Velocity L2 error"),
    ("vorticity_error", "Vorticity L2 error"),
    ("fluctuation_velocity_error", "Fluctuation velocity L2"),
    ("fluctuation_vorticity_error", "Fluctuation vorticity L2"),
    ("wake_velocity_error", "Wake velocity L2"),
    ("wake_vorticity_error", "Wake vorticity L2"),
)
METHODS = ("learned", "laplacian")
METHOD_LABELS = {"learned": "Original learned", "laplacian": "FD Laplacian"}
COLORS = {"learned": "#2F5DA8", "laplacian": "#C9473A"}
MARKERS = {"learned": "o", "laplacian": "s"}


def load_rows(path: Path, frame: int) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream))
    selected: list[dict[str, object]] = []
    for row in source:
        if int(row["frame"]) != frame:
            continue
        parsed: dict[str, object] = {
            "orientation": row["orientation"],
            "frame": int(row["frame"]),
            "method": row["method"],
            "modes": int(row["modes"]),
        }
        for name, _label in METRICS:
            parsed[name] = float(row[name])
        selected.append(parsed)
    expected = {(orientation, method, modes) for orientation in ("up", "middle", "down") for method in METHODS for modes in (64, 128, 256, 512)}
    actual = {(str(row["orientation"]), str(row["method"]), int(row["modes"])) for row in selected}
    if actual != expected or len(selected) != len(expected):
        raise ValueError(f"frame {frame} selection is incomplete: {len(selected)} rows")
    return selected


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["modes"]))].append(row)
    summary: list[dict[str, object]] = []
    for method in METHODS:
        for modes in (64, 128, 256, 512):
            group = grouped[(method, modes)]
            record: dict[str, object] = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "modes": modes,
                "cases": len(group),
            }
            for name, _label in METRICS:
                values = np.asarray([float(row[name]) for row in group], dtype=np.float64) * 100.0
                record[f"{name}_mean_pct"] = float(np.mean(values))
                record[f"{name}_std_pct"] = float(np.std(values, ddof=0))
                record[f"{name}_min_pct"] = float(np.min(values))
                record[f"{name}_max_pct"] = float(np.max(values))
            summary.append(record)
    return summary


def write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    fields = ["method", "method_label", "modes", "cases"]
    for name, _label in METRICS:
        fields.extend(
            [
                f"{name}_mean_pct",
                f"{name}_std_pct",
                f"{name}_min_pct",
                f"{name}_max_pct",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def plot(path_png: Path, path_pdf: Path, summary: list[dict[str, object]], frame: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.7), constrained_layout=True)
    modes = np.asarray((64, 128, 256, 512), dtype=np.float64)
    for axis, (metric, label) in zip(axes.ravel(), METRICS, strict=True):
        for method in METHODS:
            records = [row for row in summary if row["method"] == method]
            mean = np.asarray([float(row[f"{metric}_mean_pct"]) for row in records])
            lower = np.asarray([float(row[f"{metric}_min_pct"]) for row in records])
            upper = np.asarray([float(row[f"{metric}_max_pct"]) for row in records])
            axis.fill_between(modes, lower, upper, color=COLORS[method], alpha=0.12, linewidth=0)
            axis.plot(
                modes,
                mean,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.8,
                linewidth=1.7,
                label=METHOD_LABELS[method],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(modes, labels=[str(int(value)) for value in modes])
        axis.set_xlabel("Number of modes K")
        axis.set_ylabel("Relative L2 error (%)")
        axis.set_title(label, loc="left", fontsize=10.0)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.55, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    figure.suptitle(
        f"Airfoil reconstruction versus basis size — three unseen geometries, frame {frame}",
        fontsize=12.0,
        y=1.075,
    )
    figure.savefig(path_png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(path_pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics, args.frame)
    summary = summarize(rows)
    csv_path = args.output_dir / "frame30_reconstruction_summary.csv"
    json_path = args.output_dir / "frame30_reconstruction_summary.json"
    png_path = args.output_dir / "frame30_reconstruction_comparison.png"
    pdf_path = args.output_dir / "frame30_reconstruction_comparison.pdf"
    write_csv(csv_path, summary)
    json_path.write_text(
        json.dumps(
            {
                "frame": args.frame,
                "orientations": ["up", "middle", "down"],
                "aggregation": "arithmetic mean and population standard deviation across three unseen geometries",
                "rows": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot(png_path, pdf_path, summary, args.frame)
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
