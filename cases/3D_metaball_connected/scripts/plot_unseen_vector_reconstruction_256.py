"""Render 256x256 orthogonal-slice vector reconstruction comparisons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction"
CONFIG_PATH = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def analytic_fluid_mask(
    config: dict[str, Any], points: np.ndarray, geometry: np.ndarray
) -> np.ndarray:
    relative_outer = (points - 0.5) / float(config["outer_half_extent"])
    outer_level = 1.0 - np.sum(
        relative_outer ** int(config["outer_superellipsoid_power"]), axis=1
    )
    relative = points[:, None, :] - geometry[None, :, :3]
    distances = np.linalg.norm(relative, axis=2) - geometry[None, :, 3]
    minimum = distances.min(axis=1, keepdims=True)
    tau = float(config["metaball_softmin_tau"])
    obstacle_level = minimum[:, 0] - tau * np.log(
        np.exp(-(distances - minimum) / tau).sum(axis=1)
    )
    return (outer_level > 0.0) & (obstacle_level > 0.0)


def vector_interpolators(
    coordinates: np.ndarray,
    mask: np.ndarray,
    values: np.ndarray,
) -> list[RegularGridInterpolator]:
    volume = np.zeros((*mask.shape, 3), dtype=np.float64)
    volume[mask] = values
    axes = (coordinates, coordinates, coordinates)
    return [
        RegularGridInterpolator(
            axes,
            volume[:, :, :, component],
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        for component in range(3)
    ]


def dense_plane_points(
    plane: str,
    fixed_value: float,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    if plane == "xy":
        points = np.column_stack(
            (
                horizontal.ravel(),
                vertical.ravel(),
                np.full(horizontal.size, fixed_value),
            )
        )
        return points, "x", "y"
    if plane == "xz":
        points = np.column_stack(
            (
                horizontal.ravel(),
                np.full(horizontal.size, fixed_value),
                vertical.ravel(),
            )
        )
        return points, "x", "z"
    if plane == "yz":
        points = np.column_stack(
            (
                np.full(horizontal.size, fixed_value),
                horizontal.ravel(),
                vertical.ravel(),
            )
        )
        return points, "y", "z"
    raise ValueError(f"unsupported plane: {plane}")


def projected_components(vectors: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
    if plane == "xy":
        return vectors[:, :, 0], vectors[:, :, 1]
    if plane == "xz":
        return vectors[:, :, 0], vectors[:, :, 2]
    if plane == "yz":
        return vectors[:, :, 1], vectors[:, :, 2]
    raise ValueError(f"unsupported plane: {plane}")


def interpolate_vectors(
    interpolators: list[RegularGridInterpolator],
    points_xyz: np.ndarray,
    resolution: int,
) -> np.ndarray:
    points_zyx = points_xyz[:, (2, 1, 0)]
    values = np.column_stack(
        [interpolator(points_zyx) for interpolator in interpolators]
    )
    return values.reshape(resolution, resolution, 3)


def robust_limit(arrays: list[np.ndarray]) -> float:
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    return max(float(np.percentile(finite, 99.0)), 1.0e-12)


def relative_l2(reference: np.ndarray, approximation: np.ndarray, mask: np.ndarray) -> float:
    difference = approximation[mask] - reference[mask]
    denominator = float(np.sum(reference[mask] ** 2))
    return float(np.sqrt(np.sum(difference**2) / max(denominator, 1.0e-30)))


def draw_streamlines(
    axis: plt.Axes,
    coordinates: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    mask: np.ndarray,
) -> None:
    horizontal_masked = np.ma.array(horizontal, mask=~mask)
    vertical_masked = np.ma.array(vertical, mask=~mask)
    axis.streamplot(
        coordinates,
        coordinates,
        horizontal_masked,
        vertical_masked,
        color=NATURE_DARK,
        density=0.82,
        linewidth=0.55,
        arrowsize=0.62,
        minlength=0.08,
        maxlength=2.2,
        integration_direction="both",
        broken_streamlines=True,
        zorder=3,
    )


def render(result_dir: Path, output: Path, resolution: int) -> dict[str, Any]:
    metrics = json.loads((result_dir / "reconstruction_metrics.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    with np.load(result_dir / "reconstruction_fields.npz") as fields:
        coarse_coordinates = fields["coordinates"]
        coarse_mask = fields["mask"]
        geometry = fields["geometry"]
        coarse_fields = {
            "target": fields["target_velocity"],
            "learned": fields["learned_velocity"],
            "laplacian": fields["laplacian_velocity"],
        }

    interpolators = {
        name: vector_interpolators(coarse_coordinates, coarse_mask, values)
        for name, values in coarse_fields.items()
    }
    dense_coordinates = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    weights = geometry[:, 3] ** 3
    obstacle_center = np.average(geometry[:, :3], axis=0, weights=weights)
    plane_configuration = [
        ("xy", float(obstacle_center[2]), "z"),
        ("xz", float(obstacle_center[1]), "y"),
        ("yz", float(obstacle_center[0]), "x"),
    ]
    rows: list[dict[str, Any]] = []
    learned_numerator = 0.0
    laplacian_numerator = 0.0
    shared_denominator = 0.0
    for plane, fixed_value, fixed_axis in plane_configuration:
        points_xyz, horizontal_label, vertical_label = dense_plane_points(
            plane, fixed_value, dense_coordinates
        )
        dense_mask = analytic_fluid_mask(config, points_xyz, geometry).reshape(
            resolution, resolution
        )
        target = interpolate_vectors(
            interpolators["target"], points_xyz, resolution
        )
        learned = interpolate_vectors(
            interpolators["learned"], points_xyz, resolution
        )
        laplacian = interpolate_vectors(
            interpolators["laplacian"], points_xyz, resolution
        )
        learned_error = learned - target
        laplacian_error = laplacian - target
        learned_rl2 = relative_l2(target, learned, dense_mask)
        laplacian_rl2 = relative_l2(target, laplacian, dense_mask)
        learned_numerator += float(np.sum(learned_error[dense_mask] ** 2))
        laplacian_numerator += float(np.sum(laplacian_error[dense_mask] ** 2))
        shared_denominator += float(np.sum(target[dense_mask] ** 2))
        rows.append(
            {
                "plane": plane,
                "fixed_axis": fixed_axis,
                "fixed_value": fixed_value,
                "horizontal_label": horizontal_label,
                "vertical_label": vertical_label,
                "mask": dense_mask,
                "vectors": [target, learned, laplacian, learned_error, laplacian_error],
                "learned_velocity_rl2": learned_rl2,
                "laplacian_velocity_rl2": laplacian_rl2,
            }
        )

    learned_combined_rl2 = float(
        np.sqrt(learned_numerator / max(shared_denominator, 1.0e-30))
    )
    laplacian_combined_rl2 = float(
        np.sqrt(laplacian_numerator / max(shared_denominator, 1.0e-30))
    )
    column_titles = (
        "Target vector field\nvelocity rL2=0",
        f"Learned, K=64\nvelocity rL2={learned_combined_rl2:.3f}",
        f"Laplacian, K=64\nvelocity rL2={laplacian_combined_rl2:.3f}",
        f"Learned vector error\nrL2={learned_combined_rl2:.3f}",
        f"Laplacian vector error\nrL2={laplacian_combined_rl2:.3f}",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    figure = plt.figure(figsize=(14.4, 8.9), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(
        3, 7, width_ratios=(1, 1, 1, 1, 1, 0.045, 0.045), wspace=0.08, hspace=0.17
    )
    axes = np.empty((3, 5), dtype=object)
    for row_index, row in enumerate(rows):
        magnitude_fields = [np.linalg.norm(vectors, axis=2) for vectors in row["vectors"]]
        physical_limit = robust_limit(magnitude_fields[:3])
        error_limit = robust_limit(magnitude_fields[3:])
        physical_norm = Normalize(vmin=0.0, vmax=physical_limit)
        error_norm = Normalize(vmin=0.0, vmax=error_limit)
        for column, (vectors, magnitude) in enumerate(zip(row["vectors"], magnitude_fields)):
            axis = figure.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            masked_magnitude = np.ma.array(magnitude, mask=~row["mask"])
            axis.imshow(
                masked_magnitude,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=("Blues" if column < 3 else "Reds"),
                norm=(physical_norm if column < 3 else error_norm),
                interpolation="bilinear",
                zorder=1,
            )
            axis.contour(
                row["mask"].astype(np.float64),
                levels=[0.5],
                colors=[NATURE_GREY],
                linewidths=0.52,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                zorder=2,
            )
            horizontal, vertical = projected_components(vectors, row["plane"])
            draw_streamlines(
                axis,
                dense_coordinates,
                horizontal,
                vertical,
                row["mask"],
            )
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            if row_index == 0:
                axis.set_title(column_titles[column], color=NATURE_DARK, pad=5)
            if row_index == 2:
                axis.set_xlabel(row["horizontal_label"])
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(
                    f"{row['plane']} ({row['fixed_axis']}={row['fixed_value']:.3f})\n"
                    f"{row['vertical_label']} · rL2 L={row['learned_velocity_rl2']:.3f}, "
                    f"Δ={row['laplacian_velocity_rl2']:.3f}"
                )
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
        physical_bar.set_label(r"$|\mathbf{u}|$", fontsize=8)
        error_bar.set_label(r"$|\Delta\mathbf{u}|$", fontsize=8)

    figure.suptitle(
        f"Unseen connected three-sphere vector reconstruction — {resolution}² evaluation per slice\n"
        "Streamlines show projected direction; background shows full 3D magnitude · "
        f"Laplacian source volume {len(coarse_coordinates)}³, trilinearly sampled",
        color=NATURE_DARK,
        fontsize=12.2,
        fontweight="medium",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)

    dense_metrics = {
        "complete": True,
        "geometry_token": metrics["geometry"]["token"],
        "evaluation_grid_per_slice": [resolution, resolution],
        "evaluation_points_per_slice": resolution**2,
        "planes": [
            {
                "plane": row["plane"],
                "fixed_axis": row["fixed_axis"],
                "fixed_value": row["fixed_value"],
                "fluid_points": int(np.count_nonzero(row["mask"])),
                "learned_velocity_rl2": row["learned_velocity_rl2"],
                "laplacian_velocity_rl2": row["laplacian_velocity_rl2"],
            }
            for row in rows
        ],
        "combined_three_plane_velocity_rl2": {
            "learned_peeled_k64": learned_combined_rl2,
            "numerical_dirichlet_laplacian_k64": laplacian_combined_rl2,
        },
        "boundary_mask": "analytic rounded-cube and tau=0.02 soft-min level sets",
        "vector_sampling": "trilinear from the common reconstructed 34^3 volume",
        "laplacian_source_grid": [len(coarse_coordinates)] * 3,
        "note": (
            "256^2 is the orthogonal-slice evaluation grid; it does not replace the "
            "underlying 34^3 Laplacian eigensolve"
        ),
        "png": str(output.resolve()),
    }
    atomic_json(result_dir / "reconstruction_256x256_slice_metrics.json", dense_metrics)
    return dense_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULT_DIR / "unseen_k64_vector_reconstruction_256x256.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resolution < 32:
        raise ValueError("resolution must be at least 32")
    metrics = render(args.result_dir.resolve(), args.output.resolve(), args.resolution)
    print(json.dumps(metrics["combined_three_plane_velocity_rl2"], indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
