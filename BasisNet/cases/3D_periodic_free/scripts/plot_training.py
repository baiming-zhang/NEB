from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def values(history, key):
    pairs = [
        (int(row["epoch"]), float(row[key]))
        for row in history
        if key in row and np.isfinite(row[key])
    ]
    if not pairs:
        return np.empty(0), np.empty(0)
    return np.asarray([x for x, _ in pairs]), np.asarray([y for _, y in pairs])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    history = json.loads(
        (args.run_dir / "data" / "training_history.json").read_text(
            encoding="utf-8"
        )
    )
    panels = (
        ("train_loss", "validation_loss", "Rayleigh energy / 80", True),
        (
            "train_mean_energy",
            "validation_mean_energy",
            "Mean Rayleigh energy",
            True,
        ),
        (
            "train_mean_projection_ratio",
            "validation_mean_projection_ratio",
            "Projection ratio (diagnostic only)",
            True,
        ),
        (
            "train_orth_error",
            "validation_orth_error",
            "Peeling orthogonality error",
            True,
        ),
        (
            "train_max_discrete_divergence",
            "validation_max_discrete_divergence",
            "Discrete divergence",
            True,
        ),
        ("train_rank", "validation_min_rank", "Effective raw rank", False),
        ("learning_rate", "learning_rate", "Learning rate", True),
    )
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), dpi=150)
    for ax, (train_key, validation_key, title, log_scale) in zip(
        axes.flat, panels
    ):
        x, y = values(history, train_key)
        ax.plot(x, y, color="#4C78A8", lw=1.1, label="train")
        x, y = values(history, validation_key)
        if len(x):
            ax.plot(
                x, y, "o-", color="#F58518", ms=2.5, lw=1.1, label="validation"
            )
        if log_scale and np.any(y > 0):
            ax.set_yscale("log")
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.suptitle(
        "3D TG BasisNet: dynamic SSS with hard discrete divergence"
    )
    fig.tight_layout()
    output = args.output_dir or args.run_dir
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"training_curves.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
