"""Compare original-E500, sweep-winner-E500, and original-E2000 airfoil runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = CASE_ROOT / "results"
BASELINE_ROOT = RESULTS / "siren_k64_dirichlet_psi_iid_dynamic_e500"
SWEEP_ROOT = RESULTS / "siren_k64_airfoil_hyperparameter_sweep_e500"
LONG_ROOT = RESULTS / "siren_k64_dirichlet_psi_iid_dynamic_e2000"
OUTPUT = LONG_ROOT / "comparison_original500_winner500_original2000"

BLUE = "#3C5488"
RED = "#E64B35"
LIGHT_BLUE = "#4DBBD5"
GRAY = "#7E7E7E"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validation_rows(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in history if "validation_loss" in row]


def row_at(history: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    rows = [row for row in history if int(row["epoch"]) == epoch]
    if len(rows) != 1 or "validation_loss" not in rows[0]:
        raise RuntimeError(f"missing unique validation row at epoch {epoch}")
    return rows[0]


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": int(config["hidden"]),
        "depth": int(config["hidden_layers"]),
        "first_omega": float(config["siren_first_omega0"]),
        "hidden_omega": float(config["siren_hidden_omega0"]),
        "points_per_step": int(config["points_per_step"]),
        "initial_lr": float(config.get("base_learning_rate", config["learning_rate"])),
        "min_lr": float(config["min_learning_rate"]),
        "grad_clip": float(config["grad_clip"]),
        "endpoint_period": int(config["endpoint_period"]),
        "mgs_passes": int(config["reorthogonalization_passes"]),
    }


def make_record(
    label: str,
    family: str,
    row: dict[str, Any],
    configured_epochs: int,
    config: dict[str, Any],
    strict_wall_seconds: float | None = None,
) -> dict[str, Any]:
    epoch = int(row["epoch"])
    elapsed = float(row.get("training_elapsed_seconds", row["seconds"]))
    record = {
        "label": label,
        "family": family,
        "epoch": epoch,
        "configured_epochs": configured_epochs,
        **compact_config(config),
        "train_loss": float(row["train_loss"]),
        "validation_mean": float(row["validation_loss"]),
        "validation_p95": float(row["validation_loss_p95"]),
        "validation_max": float(row["validation_loss_max"]),
        "peeled_rank": int(row["validation_peeled_rank"]),
        "orth_error": float(row["validation_orth_error"]),
        "learning_rate": float(row["learning_rate"]),
        "history_elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / epoch,
        "strict_wall_seconds": strict_wall_seconds,
    }
    return record


def add_improvements(records: list[dict[str, Any]], reference: dict[str, Any]) -> None:
    for record in records:
        for key in ("validation_mean", "validation_p95", "validation_max"):
            record[f"{key}_improvement_vs_original500_pct"] = (
                (reference[key] - record[key]) / reference[key] * 100.0
            )


def write_table(records: list[dict[str, Any]]) -> None:
    fields = list(records[0])
    with (OUTPUT / "comparison_table.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def plot(
    baseline_history: list[dict[str, Any]],
    winner_history: list[dict[str, Any]],
    long_history: list[dict[str, Any]],
    headline: list[dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)
    series = [
        ("Original E500", validation_rows(baseline_history), BLUE, "-", "o"),
        ("Winner E500", validation_rows(winner_history), RED, "-", "s"),
        ("Original E2000", validation_rows(long_history), LIGHT_BLUE, "-", "^"),
    ]
    for ax, metric, title, ylabel in [
        (axes[0, 0], "validation_loss", "a  Validation mean convergence", "Mean Rayleigh value"),
        (axes[0, 1], "validation_loss_p95", "b  Validation P95 convergence", "P95 Rayleigh value"),
    ]:
        for label, rows, color, linestyle, marker in series:
            xs = np.asarray([r["epoch"] for r in rows], dtype=float)
            ys = np.asarray([r[metric] for r in rows], dtype=float)
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                marker=marker,
                markersize=2.6,
                markevery=max(len(xs) // 14, 1),
                label=label,
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axes[0, 0].legend(frameon=False, ncol=1, loc="upper right")

    ax = axes[1, 0]
    metric_names = ["Mean", "P95", "Max"]
    metric_keys = ["validation_mean", "validation_p95", "validation_max"]
    x = np.arange(len(metric_names), dtype=float)
    width = 0.24
    for index, (record, color) in enumerate(zip(headline, (BLUE, RED, LIGHT_BLUE))):
        values = [record[key] for key in metric_keys]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            color=color,
            edgecolor="none",
            label=record["label"],
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
    ax.set_title("c  Final fixed-validation metrics", loc="left", fontweight="bold")
    ax.set_xticks(x, metric_names)
    ax.set_ylabel("Rayleigh value (lower is better)")
    ymin = min(record[key] for record in headline for key in metric_keys)
    ymax = max(record[key] for record in headline for key in metric_keys)
    margin = max((ymax - ymin) * 0.55, 0.15)
    ax.set_ylim(ymin - margin * 0.25, ymax + margin)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    runtime_minutes = [record["history_elapsed_seconds"] / 60.0 for record in headline]
    bars = ax.bar(
        np.arange(3), runtime_minutes, color=(BLUE, RED, LIGHT_BLUE), edgecolor="none"
    )
    ax.bar_label(bars, fmt="%.1f min", padding=3, fontsize=8)
    ax.set_title("d  Training cost", loc="left", fontweight="bold")
    ax.set_xticks(np.arange(3), ["Original\nE500", "Winner\nE500", "Original\nE2000"])
    ax.set_ylabel("History elapsed time (min)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75)

    fig.suptitle(
        "2D airfoil K=64: original E500, sweep winner E500, and original E2000",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sweep_manifest = read_json(SWEEP_ROOT / "sweep_manifest.json")
    winner = sweep_manifest["winner"]
    winner_run = Path(winner["run_dir"])

    baseline_history = read_json(BASELINE_ROOT / "data/hardcoded_training_history.json")
    winner_history = read_json(winner_run / "data/hardcoded_training_history.json")
    long_history = read_json(LONG_ROOT / "run_001/data/hardcoded_training_history.json")
    baseline_config = read_json(
        CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_500.json"
    )
    winner_config = read_json(Path(winner["config"]))
    long_config = read_json(
        CASE_ROOT / "configs/k64_airfoil_siren_dirichlet_psi_iid_dynamic_2000.json"
    )
    long_state = read_json(LONG_ROOT / "launcher_state.json")

    baseline = make_record(
        "Original E500",
        "original500",
        row_at(baseline_history, 500),
        500,
        baseline_config,
    )
    winner_record = make_record(
        "Winner E500",
        "winner500",
        row_at(winner_history, 500),
        500,
        winner_config,
    )
    long_checkpoints = [
        make_record(
            f"Original E2000 @ {epoch}",
            "original2000",
            row_at(long_history, epoch),
            2000,
            long_config,
            float(long_state["training_wall_seconds"]) if epoch == 2000 else None,
        )
        for epoch in (500, 1000, 1500, 2000)
    ]
    records = [baseline, winner_record, *long_checkpoints]
    add_improvements(records, baseline)
    write_table(records)

    validation_hash = "4b723fe1f504bd6f197c2b134a25ad9696817b427e098a11df0f85896c307162"
    headline = [baseline, winner_record, long_checkpoints[-1]]
    summary = {
        "comparison_scope": (
            "same K64 fixed-validation pack; final metrics plus original-E2000 "
            "500-epoch checkpoint trajectory"
        ),
        "fixed_validation_sha256": validation_hash,
        "headline_records": headline,
        "original2000_checkpoint_records": long_checkpoints,
        "all_records": records,
        "best_validation": {
            "original500": read_json(BASELINE_ROOT / "completion.json")["best_validation"],
            "original500_epoch": read_json(BASELINE_ROOT / "completion.json")[
                "best_validation_epoch"
            ],
            "winner500": read_json(winner_run / "completion.json")["best_validation"],
            "winner500_epoch": read_json(winner_run / "completion.json")[
                "best_validation_epoch"
            ],
            "original2000": read_json(LONG_ROOT / "run_001/completion.json")[
                "best_validation"
            ],
            "original2000_epoch": read_json(
                LONG_ROOT / "run_001/completion.json"
            )["best_validation_epoch"],
        },
        "winner_configuration": compact_config(winner_config),
        "original_configuration": compact_config(long_config),
        "key_deltas_pct": {
            "winner500_vs_original500_mean": winner_record[
                "validation_mean_improvement_vs_original500_pct"
            ],
            "original2000_vs_original500_mean": long_checkpoints[-1][
                "validation_mean_improvement_vs_original500_pct"
            ],
            "original2000_vs_winner500_mean": (
                (winner_record["validation_mean"] - long_checkpoints[-1]["validation_mean"])
                / winner_record["validation_mean"]
                * 100.0
            ),
        },
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot(baseline_history, winner_history, long_history, headline)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
