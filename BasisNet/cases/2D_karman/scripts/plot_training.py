from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def values(history, key):
    x, y = [], []
    for row in history:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            x.append(int(row["epoch"]))
            y.append(float(value))
    return np.asarray(x), np.asarray(y)


def moving_average(y, width=51):
    if len(y) < width:
        return y
    kernel = np.ones(width) / width
    padded = np.pad(y, (width // 2, width // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    history = json.loads(
        (args.run_dir / "data" / "training_history.json").read_text(encoding="utf-8")
    )
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    epochs = np.asarray([int(row["epoch"]) for row in history])
    learning_rate = float(config["learning_rate"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
            "axes.edgecolor": "#777777",
            "grid.color": "#d8d8d8",
            "grid.linewidth": 0.55,
        }
    )
    colors = {
        "train": "#4C78A8",
        "validation": "#F58518",
        "probe": "#54A24B",
        "accent": "#B279A2",
    }
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0), dpi=150)

    # Loss
    ax = axes[0, 0]
    tx, ty = values(history, "train_loss")
    vx, vy = values(history, "validation_loss")
    px, py = values(history, "probe_loss")
    ax.plot(tx, ty, color=colors["train"], alpha=0.14, lw=0.55)
    ax.plot(
        tx,
        moving_average(ty),
        color=colors["train"],
        lw=1.7,
        label="Train (51-epoch mean)",
    )
    ax.plot(
        vx,
        vy,
        "o-",
        ms=2.3,
        lw=1.15,
        color=colors["validation"],
        label="Validation (every 50)",
    )
    ax.plot(
        px,
        py,
        "s-",
        ms=2.5,
        lw=1.1,
        color=colors["probe"],
        label=r"Probe $\alpha=0.15$ (every 100)",
    )
    best_index = int(np.argmin(vy))
    ax.scatter(
        [vx[best_index]],
        [vy[best_index]],
        s=45,
        marker="*",
        color=colors["accent"],
        zorder=5,
        label=f"Best val: {vy[best_index]:.6f} @ {vx[best_index]}",
    )
    ax.set_yscale("log")
    ax.set_title("Rayleigh loss / 80")
    ax.set_ylabel("Loss (log scale)")
    ax.legend(loc="upper right", ncol=1)

    # Learning rate
    ax = axes[0, 1]
    ax.plot(
        epochs,
        np.full_like(epochs, learning_rate, dtype=float),
        color=colors["accent"],
        lw=2.0,
    )
    ax.set_ylim(learning_rate * 0.94, learning_rate * 1.06)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.text(
        0.5,
        0.52,
        f"Adam learning rate = {learning_rate:.1e}\n"
        "constant for all 5000 epochs · no scheduler",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("Learning rate")
    ax.set_ylabel("Learning rate")

    # Energy
    ax = axes[1, 0]
    for key, label, color, marker in (
        ("train_mean_energy", "Train (51-epoch mean)", colors["train"], None),
        ("validation_mean_energy", "Validation", colors["validation"], "o"),
        ("probe_mean_energy", r"Probe $\alpha=0.15$", colors["probe"], "s"),
    ):
        x, y = values(history, key)
        if key.startswith("train"):
            ax.plot(x, y, color=color, alpha=0.12, lw=0.5)
            y = moving_average(y)
        ax.plot(x, y, marker=marker, ms=2.2, lw=1.3, color=color, label=label)
    ax.set_yscale("log")
    ax.set_title("Mean Rayleigh energy")
    ax.set_ylabel("Mean energy (log scale)")
    ax.legend()

    # Projection ratio (diagnostic only)
    ax = axes[1, 1]
    for key, label, color, marker in (
        (
            "train_mean_projection_ratio",
            "Train (51-epoch mean)",
            colors["train"],
            None,
        ),
        (
            "validation_mean_projection_ratio",
            "Validation",
            colors["validation"],
            "o",
        ),
        (
            "probe_mean_projection_ratio",
            r"Probe $\alpha=0.15$",
            colors["probe"],
            "s",
        ),
    ):
        x, y = values(history, key)
        if key.startswith("train"):
            ax.plot(x, y, color=color, alpha=0.12, lw=0.5)
            y = moving_average(y)
        ax.plot(x, y, marker=marker, ms=2.2, lw=1.3, color=color, label=label)
    ax.axhline(1.0, color="#777777", lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_title("Raw projection ratio (diagnostic only)")
    ax.set_ylabel("Raw overlap ratio (log scale)")
    ax.legend()

    # Orthogonality
    ax = axes[2, 0]
    for key, label, color, marker in (
        ("train_orth_error", "Train", colors["train"], None),
        ("validation_orth_error", "Validation", colors["validation"], "o"),
        ("probe_orth_error", r"Probe $\alpha=0.15$", colors["probe"], "s"),
    ):
        x, y = values(history, key)
        if key.startswith("train"):
            x, y = x[::10], y[::10]
        ax.plot(x, y, marker=marker, ms=2.2, lw=1.0, color=color, label=label)
    ax.set_yscale("log")
    ax.set_title("Peeling orthogonality error")
    ax.set_ylabel("Orthogonality error (log scale)")
    ax.legend()

    # Rank
    ax = axes[2, 1]
    for key, label, color, marker in (
        ("train_rank", "Train", colors["train"], None),
        ("validation_min_rank", "Validation minimum", colors["validation"], "o"),
        ("probe_rank", r"Probe $\alpha=0.15$", colors["probe"], "s"),
    ):
        x, y = values(history, key)
        ax.plot(x, y, marker=marker, ms=2.2, lw=1.25, color=color, label=label)
    ax.set_ylim(94.5, 96.5)
    ax.set_yticks([95, 96])
    ax.set_title("Effective raw basis rank")
    ax.set_ylabel("Rank")
    ax.text(
        0.5,
        0.18,
        "rank = 96 throughout training",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    ax.legend(loc="lower right")

    for ax in axes.flat:
        ax.set_xlim(1, 5000)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.75)
    fig.suptitle(
        "K=96 training history — no projection penalty, no frequency curriculum",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.005,
        "All six Fourier gates remained 1.0; projection ratio did not "
        "participate in the gradient.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.015, 0.025, 0.985, 0.975), h_pad=1.4, w_pad=1.2)

    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "training_curves_with_lr.png"
    svg = output_dir / "training_curves_with_lr.svg"
    pdf = output_dir / "training_curves_with_lr.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = {
        "epochs": len(history),
        "learning_rate": learning_rate,
        "learning_rate_schedule": "constant",
        "best_validation_epoch": int(vx[best_index]),
        "best_validation_loss": float(vy[best_index]),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_validation_loss": float(history[-1]["validation_loss"]),
        "final_probe_loss": float(history[-1]["probe_loss"]),
        "minimum_train_rank": int(min(row["train_rank"] for row in history)),
        "projection_penalty_in_loss": False,
        "frequency_curriculum": False,
    }
    (output_dir / "training_curves_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
