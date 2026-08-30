"""Plot the joint beta/alpha training history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in history if key in row]
    return (
        np.asarray([row["epoch"] for row in rows]),
        np.asarray([row[key] for row in rows]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    history = json.loads(
        (args.run_dir / "data" / "training_history.json").read_text(
            encoding="utf-8"
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), constrained_layout=True)
    for key, label, color in (
        ("train_loss", "train", "#2364aa"),
        ("validation_loss", "validation", "#d1495b"),
        ("probe_loss", "probe", "#2a9d8f"),
    ):
        x, y = series(history, key)
        if len(x):
            axes[0, 0].semilogy(x, y, label=label, color=color)
    axes[0, 0].set_title("Rayleigh loss / 80")
    axes[0, 0].legend()
    for key, label in (
        ("train_mean_projection_ratio", "projection ratio"),
        ("train_orth_error", "orthogonality error"),
    ):
        x, y = series(history, key)
        axes[0, 1].semilogy(x, np.maximum(y, 1.0e-18), label=label)
    axes[0, 1].set_title("Peeling diagnostics (not in loss)")
    axes[0, 1].legend()
    epoch = np.asarray([row["epoch"] for row in history])
    beta = np.asarray([row["beta"] for row in history])
    alpha = np.asarray([row["alpha_deg"] for row in history])
    axes[1, 0].scatter(epoch, beta, c=alpha, cmap="coolwarm", s=8)
    axes[1, 0].set_title(r"Joint samples: $\beta$ (color is $\alpha$)")
    axes[1, 0].set_ylabel(r"$\beta$")
    image = axes[1, 1].scatter(
        beta, alpha, c=epoch, cmap="viridis", s=10
    )
    axes[1, 1].set_title("Coverage of joint parameter domain")
    axes[1, 1].set_xlabel(r"$\beta$")
    axes[1, 1].set_ylabel(r"$\alpha$ [deg]")
    figure.colorbar(image, ax=axes[1, 1], label="epoch")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        if axis is not axes[1, 1]:
            axis.set_xlabel("epoch")
    figure.suptitle(
        "K=96 2D airfoil — fixed-order SSS peeling, no NS supervision"
    )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(args.run_dir / f"training_history.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
