"""Render orthogonal-slice vector comparisons for the unseen K=64 case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction"
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def vector_volume(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    volume = np.full((*mask.shape, 3), np.nan, dtype=np.float64)
    volume[mask] = values
    return volume


def plane_slice(
    volume: np.ndarray,
    mask: np.ndarray,
    plane: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    if plane == "xy":
        vectors = volume[index, :, :, :]
        mask_slice = mask[index, :, :]
        horizontal, vertical = 0, 1
        horizontal_label, vertical_label = "x", "y"
    elif plane == "xz":
        vectors = volume[:, index, :, :]
        mask_slice = mask[:, index, :]
        horizontal, vertical = 0, 2
        horizontal_label, vertical_label = "x", "z"
    elif plane == "yz":
        vectors = volume[:, :, index, :]
        mask_slice = mask[:, :, index]
        horizontal, vertical = 1, 2
        horizontal_label, vertical_label = "y", "z"
    else:
        raise ValueError(f"unsupported plane: {plane}")
    magnitude = np.linalg.norm(vectors, axis=2)
    return (
        magnitude,
        vectors[:, :, horizontal],
        vectors[:, :, vertical],
        mask_slice,
        horizontal_label,
        vertical_label,
    )


def robust_limit(arrays: list[np.ndarray]) -> float:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return max(float(np.percentile(finite, 99.0)), 1.0e-12)


def draw_vectors(
    axis: plt.Axes,
    coordinates: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    mask_slice: np.ndarray,
    color: str,
) -> None:
    step = 2
    coordinate_horizontal, coordinate_vertical = np.meshgrid(coordinates, coordinates)
    magnitude = np.hypot(horizontal, vertical)
    stable = binary_erosion(mask_slice, iterations=1, border_value=0)
    threshold = max(float(np.nanpercentile(magnitude[mask_slice], 8.0)), 1.0e-12)
    selection = stable & np.isfinite(magnitude) & (magnitude >= threshold)
    selection &= np.indices(mask_slice.shape)[0] % step == 0
    selection &= np.indices(mask_slice.shape)[1] % step == 0
    normalized_horizontal = np.divide(
        horizontal,
        magnitude,
        out=np.zeros_like(horizontal),
        where=magnitude > 1.0e-12,
    )
    normalized_vertical = np.divide(
        vertical,
        magnitude,
        out=np.zeros_like(vertical),
        where=magnitude > 1.0e-12,
    )
    axis.quiver(
        coordinate_horizontal[selection],
        coordinate_vertical[selection],
        normalized_horizontal[selection],
        normalized_vertical[selection],
        angles="xy",
        scale_units="xy",
        scale=18.5,
        pivot="mid",
        color=color,
        width=0.0042,
        headwidth=3.4,
        headlength=4.3,
        headaxislength=3.8,
        alpha=0.92,
        zorder=3,
    )


def render(result_dir: Path, output: Path) -> None:
    metrics = json.loads((result_dir / "reconstruction_metrics.json").read_text(encoding="utf-8"))
    with np.load(result_dir / "reconstruction_fields.npz") as fields:
        coordinates = fields["coordinates"]
        mask = fields["mask"]
        geometry = fields["geometry"]
        target_velocity = fields["target_velocity"]
        learned_velocity = fields["learned_velocity"]
        laplacian_velocity = fields["laplacian_velocity"]

    physical_volumes = [
        vector_volume(target_velocity, mask),
        vector_volume(learned_velocity, mask),
        vector_volume(laplacian_velocity, mask),
    ]
    error_volumes = [
        vector_volume(learned_velocity - target_velocity, mask),
        vector_volume(laplacian_velocity - target_velocity, mask),
    ]
    weights = geometry[:, 3] ** 3
    obstacle_center = np.average(geometry[:, :3], axis=0, weights=weights)
    plane_configuration = [
        ("xy", int(np.argmin(np.abs(coordinates - obstacle_center[2]))), "z"),
        ("xz", int(np.argmin(np.abs(coordinates - obstacle_center[1]))), "y"),
        ("yz", int(np.argmin(np.abs(coordinates - obstacle_center[0]))), "x"),
    ]
    column_titles = (
        "Target vector field",
        "Learned, K=64",
        "Laplacian, K=64",
        "Learned vector error",
        "Laplacian vector error",
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
    figure = plt.figure(figsize=(14.4, 8.8), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(
        3, 7, width_ratios=(1, 1, 1, 1, 1, 0.045, 0.045), wspace=0.08, hspace=0.17
    )
    axes = np.empty((3, 5), dtype=object)
    for row, (plane, plane_index, fixed_axis) in enumerate(plane_configuration):
        physical_slices = [
            plane_slice(volume, mask, plane, plane_index) for volume in physical_volumes
        ]
        error_slices = [
            plane_slice(volume, mask, plane, plane_index) for volume in error_volumes
        ]
        physical_limit = robust_limit([item[0] for item in physical_slices])
        error_limit = robust_limit([item[0] for item in error_slices])
        physical_norm = Normalize(vmin=0.0, vmax=physical_limit)
        error_norm = Normalize(vmin=0.0, vmax=error_limit)
        combined = [*physical_slices, *error_slices]
        for column, item in enumerate(combined):
            magnitude, horizontal, vertical, mask_slice, horizontal_label, vertical_label = item
            axis = figure.add_subplot(grid[row, column])
            axes[row, column] = axis
            axis.imshow(
                magnitude,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=("Blues" if column < 3 else "Reds"),
                norm=(physical_norm if column < 3 else error_norm),
                interpolation="bilinear",
                zorder=1,
            )
            axis.contour(
                mask_slice.astype(np.float64),
                levels=[0.5],
                colors=[NATURE_GREY],
                linewidths=0.5,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                zorder=2,
            )
            draw_vectors(
                axis,
                coordinates,
                horizontal,
                vertical,
                mask_slice,
                NATURE_DARK,
            )
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            if row == 0:
                axis.set_title(column_titles[column], color=NATURE_DARK, pad=5)
            if row == 2:
                axis.set_xlabel(horizontal_label)
            else:
                axis.set_xticklabels([])
            if column == 0:
                fixed_value = float(coordinates[plane_index])
                axis.set_ylabel(
                    f"{plane} plane ({fixed_axis}={fixed_value:.3f})\n{vertical_label}"
                )
            else:
                axis.set_yticklabels([])
            for spine in axis.spines.values():
                spine.set_color(NATURE_GREY)
                spine.set_linewidth(0.55)
        physical_bar_axis = figure.add_subplot(grid[row, 5])
        error_bar_axis = figure.add_subplot(grid[row, 6])
        physical_bar = figure.colorbar(
            axes[row, 2].images[0], cax=physical_bar_axis, orientation="vertical"
        )
        error_bar = figure.colorbar(
            axes[row, 4].images[0], cax=error_bar_axis, orientation="vertical"
        )
        physical_bar.ax.tick_params(labelsize=7, width=0.5, length=2)
        error_bar.ax.tick_params(labelsize=7, width=0.5, length=2)
        physical_bar.set_label(r"$|\mathbf{u}|$", fontsize=8)
        error_bar.set_label(r"$|\Delta\mathbf{u}|$", fontsize=8)

    learned = metrics["k64_reconstruction"]["learned_peeled"]
    laplacian = metrics["k64_reconstruction"]["numerical_dirichlet_laplacian"]
    figure.suptitle(
        "Unseen connected three-sphere vector reconstruction — identical Dirichlet domain and K=64\n"
        "Normalized arrows show projected vector direction; background shows full 3D magnitude · "
        f"held-out velocity L2: learned {learned['test_velocity_relative_l2']:.3f}, "
        f"Laplacian {laplacian['test_velocity_relative_l2']:.3f}",
        color=NATURE_DARK,
        fontsize=12.3,
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
        default=DEFAULT_RESULT_DIR / "unseen_k64_vector_reconstruction_comparison.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.result_dir.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
