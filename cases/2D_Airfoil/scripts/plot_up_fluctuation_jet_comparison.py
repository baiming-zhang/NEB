#!/usr/bin/env python3
"""Plot one airfoil case's fluctuation errors for two objectives and FD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


K_VALUES = (64, 128, 256, 512, 1024, 2048)
COLORS = {
    "Vorticity-Rayleigh": "#3C5488",  # Nature-style blue
    "Potential-Rayleigh": "#E64B35",  # Nature-style red
    "FD-Laplacian": "#4D4D4D",
}
MARKERS = {
    "Vorticity-Rayleigh": "o",
    "Potential-Rayleigh": "^",
    "FD-Laplacian": "D",
}
LINESTYLES = {
    "Vorticity-Rayleigh": "-",
    "Potential-Rayleigh": "-",
    "FD-Laplacian": "--",
}


def metrics(summary: Path, family: str) -> dict[int, dict[str, float]]:
    payload = json.loads(summary.read_text(encoding="utf-8"))
    source = payload[family]
    return {
        k: {
            key: float(source[f"k{k}"]["metrics_by_frame"][0][key]) * 100.0
            for key in (
                "fluctuation_velocity_error",
                "fluctuation_vorticity_error",
            )
        }
        for k in K_VALUES
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jet3-summary", type=Path, required=True)
    parser.add_argument("--jet1-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case-label",
        choices=("Up", "Middle", "Down"),
        default="Up",
        help="Human-readable geometry label written to the companion CSV",
    )
    parser.add_argument(
        "--x-axis",
        choices=("log2", "linear", "categorical"),
        default="log2",
        help="Numerical log2, numerical linear, or equally spaced discrete K axis",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    series = {
        "Vorticity-Rayleigh": metrics(args.jet3_summary, "learned"),
        "Potential-Rayleigh": metrics(args.jet1_summary, "learned"),
        "FD-Laplacian": metrics(args.jet3_summary, "laplacian"),
    }
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "method",
                "K",
                "fluctuation_velocity_relative_l2_percent",
                "fluctuation_vorticity_relative_l2_percent",
            ),
        )
        writer.writeheader()
        for method, values in series.items():
            for k in K_VALUES:
                writer.writerow(
                    {
                        "case": args.case_label,
                        "method": method,
                        "K": k,
                        "fluctuation_velocity_relative_l2_percent": values[k][
                            "fluctuation_velocity_error"
                        ],
                        "fluctuation_vorticity_relative_l2_percent": values[k][
                            "fluctuation_vorticity_error"
                        ],
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.75), constrained_layout=False)
    panels = (
        ("fluctuation_velocity_error", "Fluctuation velocity error (%)", (10, 102)),
        ("fluctuation_vorticity_error", "Fluctuation vorticity error (%)", (30, 103)),
    )
    x_positions = np.arange(len(K_VALUES)) if args.x_axis == "categorical" else K_VALUES
    handles = []
    for axis, (key, ylabel, limits), panel in zip(axes, panels, ("a", "b")):
        for method in (
            "Vorticity-Rayleigh",
            "Potential-Rayleigh",
            "FD-Laplacian",
        ):
            line, = axis.plot(
                x_positions,
                [series[method][k][key] for k in K_VALUES],
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                linewidth=1.55,
                markersize=4.5,
                markerfacecolor="white" if method == "FD-Laplacian" else COLORS[method],
                markeredgecolor=COLORS[method],
                markeredgewidth=0.8,
                label=method,
                zorder=3,
            )
            if axis is axes[0]:
                handles.append(line)
        if args.x_axis == "log2":
            axis.set_xscale("log", base=2)
            axis.set_xticks(K_VALUES, [str(k) for k in K_VALUES])
            axis.set_xlim(55, 2380)
        elif args.x_axis == "linear":
            axis.set_xscale("linear")
            axis.set_xticks(K_VALUES[1:], [str(k) for k in K_VALUES[1:]])
            axis.set_xlim(0, 2120)
        else:
            axis.set_xscale("linear")
            axis.set_xticks(x_positions, [str(k) for k in K_VALUES])
            axis.set_xlim(-0.25, len(K_VALUES) - 0.75)
        axis.set_ylim(*limits)
        axis.set_xlabel("Number of modes, $K$")
        axis.set_ylabel(ylabel)
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out")
        if args.x_axis == "linear":
            axis.annotate(
                "64",
                xy=(64, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(-17, -19),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=7.5,
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#333333",
                    "linewidth": 0.65,
                    "shrinkA": 1.0,
                    "shrinkB": 1.0,
                },
            )
        axis.text(
            -0.16,
            1.04,
            f"({panel})",
            transform=axis.transAxes,
            fontsize=9.0,
            va="bottom",
        )
    fig.legend(
        handles,
        [item.get_label() for item in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.225, top=0.81, wspace=0.32)
    fig.savefig(args.output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(
        json.dumps(
            {
                "png": str(args.output.with_suffix(".png").resolve()),
                "pdf": str(args.output.with_suffix(".pdf").resolve()),
                "csv": str(csv_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
