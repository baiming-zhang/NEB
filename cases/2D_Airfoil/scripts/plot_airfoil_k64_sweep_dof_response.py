#!/usr/bin/env python3
"""Plot local one-degree-of-freedom responses from the completed airfoil sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BLUE = "#3C5488"
SKY = "#4DBBD5"
RED = "#E64B35"
GREY = "#7E7E7E"
GRID = "#D9D9D9"
TEXT = "#252525"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_for(record: dict[str, Any]) -> dict[str, Any]:
    return read_json(Path(record["config"]))


def mean_loss(record: dict[str, Any]) -> float:
    return float(record["final_validation_loss"])


def improvement_percent(record: dict[str, Any], reference: dict[str, Any]) -> float:
    reference_loss = mean_loss(reference)
    return 100.0 * (reference_loss - mean_loss(record)) / reference_loss


def stage_by_name(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return next(stage for stage in manifest["stages"] if stage["name"] == name)


def unique_records_by_parameter(
    stage: dict[str, Any], parameter: str
) -> list[tuple[float, dict[str, Any]]]:
    records = [stage["incumbent_before"], *stage["arms"]]
    by_value: dict[float, dict[str, Any]] = {}
    for record in records:
        value = float(config_for(record)[parameter])
        by_value[value] = record
    return sorted(by_value.items())


def format_tick(parameter: str, value: float) -> str:
    if parameter in {"learning_rate", "min_learning_rate"}:
        return f"{value:.1e}".replace("e-0", "e-").replace("e+0", "e+")
    if parameter in {
        "points_per_step",
        "endpoint_period",
        "reorthogonalization_passes",
    }:
        return str(int(value))
    return f"{value:g}"


def style_axis(axis: plt.Axes) -> None:
    axis.axhline(0.0, color=GREY, linewidth=1.0, linestyle="--", zorder=0)
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    axis.tick_params(colors=TEXT, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(0.8)


def mark_selected(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    annotate: bool = True,
) -> None:
    axis.scatter(
        [x],
        [y],
        marker="*",
        s=150,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=6,
    )
    if annotate:
        axis.annotate(
            label,
            (x, y),
            xytext=(0, 10 if y <= 0.8 else -17),
            textcoords="offset points",
            ha="center",
            va="bottom" if y <= 0.8 else "top",
            color=RED,
            fontsize=7.5,
            fontweight="medium",
        )


def plot_frequency_panel(
    axis: plt.Axes,
    stage: dict[str, Any],
    x_parameter: str,
    conditional_parameter: str,
    title: str,
    x_label: str,
) -> None:
    reference = stage["incumbent_before"]
    records = [reference, *stage["arms"]]
    grouped: dict[float, list[tuple[float, dict[str, Any]]]] = {}
    for record in records:
        config = config_for(record)
        condition = float(config[conditional_parameter])
        x_value = float(config[x_parameter])
        grouped.setdefault(condition, []).append((x_value, record))

    colors = [BLUE, SKY, RED]
    for color, condition in zip(colors, sorted(grouped)):
        observations = sorted(grouped[condition])
        x_values = [item[0] for item in observations]
        changes = [improvement_percent(item[1], reference) for item in observations]
        condition_symbol = r"$\omega_h$" if conditional_parameter == "siren_hidden_omega0" else r"$\omega_1$"
        axis.plot(
            x_values,
            changes,
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=5.0,
            label=f"{condition_symbol}={condition:g}",
            zorder=3,
        )

    winner = stage["winner"]
    winner_config = config_for(winner)
    winner_x = float(winner_config[x_parameter])
    winner_y = improvement_percent(winner, reference)
    mark_selected(axis, winner_x, winner_y, f"selected\n{mean_loss(winner):.3f}")
    values = sorted({float(config_for(record)[x_parameter]) for record in records})
    axis.set_xticks(values, [format_tick(x_parameter, value) for value in values])
    axis.set_title(title, fontsize=10, color=TEXT, pad=7)
    axis.set_xlabel(x_label, fontsize=8.5, color=TEXT)
    axis.legend(frameon=False, fontsize=7, loc="lower left", handlelength=2.2)
    style_axis(axis)


def plot_scalar_panel(
    axis: plt.Axes,
    stage: dict[str, Any],
    parameter: str,
    title: str,
    x_label: str,
) -> None:
    reference = stage["incumbent_before"]
    observations = unique_records_by_parameter(stage, parameter)
    positions = list(range(len(observations)))
    changes = [improvement_percent(record, reference) for _, record in observations]
    axis.plot(
        positions,
        changes,
        color=BLUE,
        linewidth=1.8,
        marker="o",
        markersize=5.5,
        zorder=3,
    )
    for position, change, (_, record) in zip(positions, changes, observations):
        axis.annotate(
            f"{mean_loss(record):.3f}",
            (position, change),
            xytext=(0, 6 if change <= 0.8 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if change <= 0.8 else "top",
            fontsize=7.2,
            color=TEXT,
        )

    winner = stage["winner"]
    winner_value = float(config_for(winner)[parameter])
    selected_position = next(
        position
        for position, (value, _) in enumerate(observations)
        if value == winner_value
    )
    selected_change = improvement_percent(winner, reference)
    mark_selected(axis, selected_position, selected_change, "", annotate=False)
    axis.set_xticks(
        positions,
        [format_tick(parameter, value) for value, _ in observations],
    )
    axis.set_title(title, fontsize=10, color=TEXT, pad=7)
    axis.set_xlabel(x_label, fontsize=8.5, color=TEXT)
    style_axis(axis)


def plot(manifest: dict[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelcolor": TEXT,
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 7.8),
        sharey=True,
        constrained_layout=False,
    )
    frequency = stage_by_name(manifest, "frequency")
    plot_frequency_panel(
        axes[0, 0],
        frequency,
        "siren_first_omega0",
        "siren_hidden_omega0",
        r"First-layer frequency $\omega_1$",
        r"$\omega_1$",
    )
    plot_frequency_panel(
        axes[0, 1],
        frequency,
        "siren_hidden_omega0",
        "siren_first_omega0",
        r"Hidden-layer frequency $\omega_h$",
        r"$\omega_h$",
    )
    scalar_panels = [
        ("learning_rate", "learning_rate", "Initial learning rate", "Learning rate"),
        ("cosine_min_lr", "min_learning_rate", "Cosine minimum LR", "Minimum LR"),
        ("grad_clip", "grad_clip", "Gradient clipping", "Clip norm"),
        ("points_per_step", "points_per_step", "IID fluid points per step", "Points per step"),
        ("endpoint_period", "endpoint_period", "Endpoint period", "Epoch period"),
        (
            "reorthogonalization",
            "reorthogonalization_passes",
            "MGS peeling passes",
            "MGS passes",
        ),
    ]
    scalar_axes = [axes[0, 2], axes[0, 3], *axes[1, :]]
    for axis, (stage_name, parameter, title, x_label) in zip(
        scalar_axes, scalar_panels
    ):
        plot_scalar_panel(
            axis,
            stage_by_name(manifest, stage_name),
            parameter,
            title,
            x_label,
        )

    all_changes: list[float] = []
    for stage in manifest["stages"]:
        reference = stage["incumbent_before"]
        all_changes.extend(
            improvement_percent(record, reference) for record in stage["arms"]
        )
    lower = min(-4.0, min(all_changes) - 0.25)
    upper = max(1.35, max(all_changes) + 0.35)
    for axis in axes.flat:
        axis.set_ylim(lower, upper)
    figure.supylabel(
        "Epoch-500 validation mean improvement vs stage incumbent (%)\nPositive is better",
        x=0.012,
        fontsize=9.5,
    )
    figure.suptitle(
        "Airfoil K64 SIREN: local response of each swept degree of freedom",
        fontsize=14,
        fontweight="medium",
        y=0.985,
    )
    legend_handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=1.8, label="Epoch-500 validation mean"),
        Line2D([0], [0], color=GREY, linestyle="--", linewidth=1.0, label="Stage incumbent (0%)"),
        Line2D([0], [0], color=RED, marker="*", linewidth=0, markersize=10, label="Selected for next stage"),
    ]
    figure.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        fontsize=8,
    )
    figure.text(
        0.5,
        0.018,
        "Each panel is conditioned on the incumbent entering that stage; frequency panels show the complete 3 x 3 grid. "
        "Numbers beside scalar points are absolute validation means.",
        ha="center",
        va="bottom",
        fontsize=8,
        color=GREY,
    )
    figure.subplots_adjust(left=0.072, right=0.988, top=0.875, bottom=0.105, wspace=0.22, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(read_json(args.manifest.resolve()), args.output.resolve())


if __name__ == "__main__":
    main()
