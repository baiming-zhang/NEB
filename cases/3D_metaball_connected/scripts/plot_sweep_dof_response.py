"""Plot the Stage-A one-degree-of-freedom hyperparameter responses."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = CASE_ROOT / "results" / "k64_siren_dirichlet_e500_hyperparameter_sweep"
INPUT_CSV = SWEEP_ROOT / "sweep_results.csv"
OUTPUT_PNG = SWEEP_ROOT / "sweep_single_dof_response.png"
NATURE_BLUE = "#3B6FB6"
NATURE_RED = "#D1495B"
NATURE_DARK = "#243447"
NATURE_GREY = "#7A8793"

BASELINE = {
    "learning_rate": 2.0e-4,
    "min_learning_rate": 1.0e-6,
    "siren_first_omega0": 8.0,
    "siren_hidden_omega0": 30.0,
    "grad_clip": 5.0,
    "points_per_step": 4096,
    "outer_envelope_transition": 0.15,
    "obstacle_envelope_transition": 0.10,
}

PARAMETERS: list[tuple[str, str, str]] = [
    ("learning_rate", "Learning rate", "scientific"),
    ("min_learning_rate", "Cosine minimum LR", "scientific"),
    ("siren_first_omega0", r"First $\omega_0$", "plain"),
    ("siren_hidden_omega0", r"Hidden $\omega_0$", "plain"),
    ("grad_clip", "Gradient clip", "plain"),
    ("points_per_step", "IID points per step", "integer"),
    ("outer_envelope_transition", r"Outer envelope $\delta$", "plain"),
    ("obstacle_envelope_transition", r"Obstacle envelope $\delta$", "plain"),
]


def format_value(value: float, style: str) -> str:
    if style == "scientific":
        return f"{value:.1e}"
    if style == "integer":
        return f"{int(value)}"
    return f"{value:g}"


def load_records() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    baseline_record = next(record for record in records if record["stage"] == "baseline")
    by_parameter: dict[str, list[dict[str, Any]]] = {parameter: [] for parameter in BASELINE}
    for record in records:
        if record["stage"] != "A":
            continue
        changes = json.loads(record["changes"])
        if len(changes) != 1:
            raise ValueError(f"Stage-A arm is not OFAT: {record['name']}")
        parameter, value = next(iter(changes.items()))
        by_parameter[parameter].append(
            {
                "name": record["name"],
                "value": float(value),
                "mean": float(record["validation_mean_rayleigh"]),
                "max": float(record["validation_max_rayleigh"]),
            }
        )
    return baseline_record, by_parameter


def render() -> None:
    baseline_record, by_parameter = load_records()
    baseline_mean = float(baseline_record["validation_mean_rayleigh"])
    baseline_max = float(baseline_record["validation_max_rayleigh"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 8.2))
    figure.subplots_adjust(
        left=0.055, right=0.99, bottom=0.075, top=0.80, wspace=0.19, hspace=0.43
    )
    legend_handles = None
    for axis, (parameter, title, label_style) in zip(axes.ravel(), PARAMETERS):
        records = [
            *by_parameter[parameter],
            {
                "name": "baseline",
                "value": float(BASELINE[parameter]),
                "mean": baseline_mean,
                "max": baseline_max,
            },
        ]
        records.sort(key=lambda record: record["value"])
        values = np.asarray([record["value"] for record in records], dtype=np.float64)
        means = np.asarray([record["mean"] for record in records], dtype=np.float64)
        maxima = np.asarray([record["max"] for record in records], dtype=np.float64)
        positions = (values - values.min()) / (values.max() - values.min())
        mean_line = axis.plot(
            positions,
            means,
            "o-",
            color=NATURE_BLUE,
            linewidth=1.8,
            markersize=5.5,
            label="Validation mean Rayleigh",
            zorder=3,
        )[0]
        max_line = axis.plot(
            positions,
            maxima,
            "s--",
            color=NATURE_RED,
            linewidth=1.45,
            markersize=4.8,
            label="Validation max Rayleigh",
            zorder=2,
        )[0]
        baseline_index = int(np.flatnonzero(np.isclose(values, BASELINE[parameter]))[0])
        axis.scatter(
            [positions[baseline_index]],
            [means[baseline_index]],
            s=72,
            facecolors="white",
            edgecolors=NATURE_DARK,
            linewidths=1.25,
            zorder=4,
            label="Baseline",
        )
        combined = np.concatenate((means, maxima))
        span = max(float(combined.max() - combined.min()), 20.0)
        axis.set_ylim(float(combined.min() - 0.16 * span), float(combined.max() + 0.24 * span))
        for index, (position, mean) in enumerate(zip(positions, means)):
            change = 100.0 * (mean / baseline_mean - 1.0)
            label = "base" if index == baseline_index else f"{change:+.1f}%"
            axis.annotate(
                label,
                (position, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=NATURE_DARK,
                fontsize=8,
            )
        axis.set_xticks(positions, [format_value(value, label_style) for value in values])
        axis.set_title(title, color=NATURE_DARK, fontweight="medium")
        axis.set_xlabel("Parameter value")
        axis.set_ylabel("Dirichlet Rayleigh")
        axis.grid(axis="y", color=NATURE_GREY, alpha=0.20, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(NATURE_GREY)
            spine.set_linewidth(0.65)
        legend_handles = (mean_line, max_line)
    assert legend_handles is not None
    figure.legend(
        legend_handles,
        ("Validation mean Rayleigh", "Validation max Rayleigh"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "K=64 connected three-sphere SIREN — single-parameter response at epoch 500\n"
        "Percent labels show validation mean change relative to the original configuration",
        color=NATURE_DARK,
        fontsize=13,
        fontweight="medium",
        y=0.985,
    )
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PNG.resolve())


if __name__ == "__main__":
    render()
