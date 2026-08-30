"""Render the fair K=64 unseen reconstruction comparison as one PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction"
NATURE_BLUE = "#3B6FB6"
NATURE_RED = "#D1495B"
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def volume_slice(values: np.ndarray, mask: np.ndarray, z_index: int) -> np.ndarray:
    volume = np.full(mask.shape, np.nan, dtype=np.float64)
    volume[mask] = values
    return volume[z_index]


def robust_signed_limit(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([np.abs(array[np.isfinite(array)]) for array in arrays])
    return max(float(np.percentile(values, 99.0)), 1.0e-12)


def robust_positive_limit(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return max(float(np.percentile(values, 99.0)), 1.0e-12)


def render(result_dir: Path, output: Path) -> None:
    metrics = json.loads((result_dir / "reconstruction_metrics.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "reconstruction_fields.npz") as fields:
        coordinates = fields["coordinates"]
        mask = fields["mask"]
        geometry = fields["geometry"]
        target_potential = fields["target_potential"]
        learned_potential = fields["learned_potential"]
        laplacian_potential = fields["laplacian_potential"]
        target_velocity = fields["target_velocity"]
        learned_velocity = fields["learned_velocity"]
        laplacian_velocity = fields["laplacian_velocity"]

    obstacle_center_z = float(np.average(geometry[:, 2], weights=geometry[:, 3] ** 3))
    z_index = int(np.argmin(np.abs(coordinates - obstacle_center_z)))
    z_value = float(coordinates[z_index])
    mask_slice = mask[z_index]

    potential_fields = [
        volume_slice(target_potential[:, 0], mask, z_index),
        volume_slice(learned_potential[:, 0], mask, z_index),
        volume_slice(laplacian_potential[:, 0], mask, z_index),
    ]
    potential_errors = [
        volume_slice(np.abs(learned_potential[:, 0] - target_potential[:, 0]), mask, z_index),
        volume_slice(
            np.abs(laplacian_potential[:, 0] - target_potential[:, 0]), mask, z_index
        ),
    ]
    velocity_x_fields = [
        volume_slice(target_velocity[:, 0], mask, z_index),
        volume_slice(learned_velocity[:, 0], mask, z_index),
        volume_slice(laplacian_velocity[:, 0], mask, z_index),
    ]
    velocity_errors = [
        volume_slice(np.linalg.norm(learned_velocity - target_velocity, axis=1), mask, z_index),
        volume_slice(
            np.linalg.norm(laplacian_velocity - target_velocity, axis=1), mask, z_index
        ),
    ]
    speed_fields = [
        volume_slice(np.linalg.norm(target_velocity, axis=1), mask, z_index),
        volume_slice(np.linalg.norm(learned_velocity, axis=1), mask, z_index),
        volume_slice(np.linalg.norm(laplacian_velocity, axis=1), mask, z_index),
    ]
    speed_errors = velocity_errors

    rows = [
        (potential_fields, potential_errors, r"$A_{12}$", "signed"),
        (velocity_x_fields, velocity_errors, r"$u_x$", "signed"),
        (speed_fields, speed_errors, r"$|\mathbf{u}|$", "positive"),
    ]
    column_titles = (
        "Target",
        "Learned, K=64",
        "Laplacian, K=64",
        "|Learned error|",
        "|Laplacian error|",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    figure = plt.figure(figsize=(14.2, 8.6), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(
        3, 7, width_ratios=(1, 1, 1, 1, 1, 0.045, 0.045), wspace=0.08, hspace=0.16
    )
    axes = np.empty((3, 5), dtype=object)
    for row_index, (physical, errors, row_label, scale_kind) in enumerate(rows):
        signed_limit = robust_signed_limit(physical)
        positive_limit = robust_positive_limit(physical)
        error_limit = robust_positive_limit(errors)
        if scale_kind == "signed":
            physical_norm = TwoSlopeNorm(vmin=-signed_limit, vcenter=0.0, vmax=signed_limit)
            physical_cmap = "RdBu_r"
        else:
            physical_norm = Normalize(vmin=0.0, vmax=positive_limit)
            physical_cmap = "Blues"
        error_norm = Normalize(vmin=0.0, vmax=error_limit)
        for column in range(5):
            axis = figure.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            image = physical[column] if column < 3 else errors[column - 3]
            axis.imshow(
                image,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=(physical_cmap if column < 3 else "Reds"),
                norm=(physical_norm if column < 3 else error_norm),
                interpolation="bilinear",
            )
            axis.contour(
                mask_slice.astype(np.float64),
                levels=[0.5],
                colors=[NATURE_DARK],
                linewidths=0.45,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            if row_index == 0:
                axis.set_title(column_titles[column], color=NATURE_DARK, pad=5)
            if row_index == 2:
                axis.set_xlabel("x")
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(f"{row_label}\n y")
            else:
                axis.set_yticklabels([])
            for spine in axis.spines.values():
                spine.set_color(NATURE_GREY)
                spine.set_linewidth(0.55)
        physical_bar_axis = figure.add_subplot(grid[row_index, 5])
        error_bar_axis = figure.add_subplot(grid[row_index, 6])
        physical_bar = figure.colorbar(
            axes[row_index, 2].images[0], cax=physical_bar_axis, orientation="vertical"
        )
        error_bar = figure.colorbar(
            axes[row_index, 4].images[0], cax=error_bar_axis, orientation="vertical"
        )
        physical_bar.ax.tick_params(labelsize=7, width=0.5, length=2)
        error_bar.ax.tick_params(labelsize=7, width=0.5, length=2)
        physical_bar.set_label("value", fontsize=7.5)
        error_bar.set_label("abs. error", fontsize=7.5)

    learned = metrics["k64_reconstruction"]["learned_peeled"]
    laplacian = metrics["k64_reconstruction"]["numerical_dirichlet_laplacian"]
    figure.suptitle(
        "Unseen connected three-sphere reconstruction — identical Dirichlet domain and K=64\n"
        f"token {metrics['geometry']['token']} · z={z_value:.3f} · "
        f"held-out potential L2: learned {learned['test_potential_relative_l2']:.3f}, "
        f"Laplacian {laplacian['test_potential_relative_l2']:.3f} · "
        f"velocity L2: learned {learned['test_velocity_relative_l2']:.3f}, "
        f"Laplacian {laplacian['test_velocity_relative_l2']:.3f}",
        color=NATURE_DARK,
        fontsize=12.5,
        fontweight="medium",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_DIR / "unseen_k64_reconstruction_comparison.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.result_dir.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
