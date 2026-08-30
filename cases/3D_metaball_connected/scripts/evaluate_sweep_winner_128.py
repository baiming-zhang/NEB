"""Compare the original and swept K=64 bases against the cached 128^3 Laplacian."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))
import evaluate_unseen_vs_laplacian as base  # noqa: E402
import plot_unseen_vector_reconstruction_256 as plot_utils  # noqa: E402


SWEEP_ROOT = CASE_ROOT / "results" / "k64_siren_dirichlet_e500_hyperparameter_sweep"
BASELINE_CHECKPOINT = (
    CASE_ROOT / "results" / "k64_siren_dirichlet_e500" / "run_001" / "checkpoints" / "final.pt"
)
DEFAULT_OUTPUT = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction_128"
CONFIG_PATH = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def peel_with_transform(
    raw_potential: np.ndarray,
    raw_gradient: np.ndarray,
    weight: float,
    passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    count, _, modes = raw_potential.shape
    potential = raw_potential.reshape(count * 3, modes)
    gradient = raw_gradient.reshape(count * 9, modes)
    raw_rayleigh = np.sum(gradient**2, axis=0) / np.maximum(
        np.sum(potential**2, axis=0), 1.0e-30
    )
    order = np.argsort(raw_rayleigh, kind="stable")
    basis = np.empty_like(potential)
    transform = np.zeros((modes, modes), dtype=np.float64)
    residual_masses = np.empty(modes)
    for rank, source in enumerate(order):
        residual = potential[:, source].copy()
        coefficient = np.zeros(modes)
        coefficient[source] = 1.0
        if rank:
            for _ in range(passes):
                projection = weight * (basis[:, :rank].T @ residual)
                residual -= basis[:, :rank] @ projection
                coefficient -= transform[:, :rank] @ projection
        mass = weight * float(residual @ residual)
        if not np.isfinite(mass) or mass <= 1.0e-14:
            raise FloatingPointError(f"unresolved peeled mode {rank + 1}")
        scale = 1.0 / np.sqrt(mass)
        basis[:, rank] = residual * scale
        transform[:, rank] = coefficient * scale
        residual_masses[rank] = mass
    peeled_potential = (potential @ transform).reshape(count, 3, modes)
    peeled_gradient = (gradient @ transform).reshape(count, 3, 3, modes)
    gram = weight * (basis.T @ basis)
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    rayleigh = np.sum(peeled_gradient.reshape(count * 9, modes) ** 2, axis=0) / np.maximum(
        np.sum(peeled_potential.reshape(count * 3, modes) ** 2, axis=0), 1.0e-30
    )
    metrics = {
        "reporting_scope": "peeled_basis_only",
        "peeled_effective_rank": int(np.count_nonzero(eigenvalues > 1.0e-10)),
        "peeled_orthogonality_max_abs": float(np.max(np.abs(gram - np.eye(modes)))),
        "peeled_residual_mass_min": float(np.min(residual_masses)),
        "rayleigh_mean": float(np.mean(rayleigh)),
        "rayleigh_min": float(np.min(rayleigh)),
        "rayleigh_max": float(np.max(rayleigh)),
        "causal_order_1based": [int(value) + 1 for value in order],
    }
    return peeled_potential, peeled_gradient, transform, rayleigh, metrics


def scalar_gradients_at_indices(
    eigenvectors: np.ndarray,
    mask: np.ndarray,
    spacing: float,
    sample_indices: np.ndarray,
) -> np.ndarray:
    flat_valid = np.flatnonzero(mask.ravel())
    selected_flat = flat_valid[sample_indices]
    gradients = np.empty((len(sample_indices), 3, eigenvectors.shape[1]), dtype=np.float64)
    for mode in range(eigenvectors.shape[1]):
        volume = np.zeros(mask.shape, dtype=np.float32)
        volume[mask] = eigenvectors[:, mode]
        dz, dy, dx = np.gradient(volume, spacing, edge_order=2)
        gradients[:, 0, mode] = dx.ravel()[selected_flat]
        gradients[:, 1, mode] = dy.ravel()[selected_flat]
        gradients[:, 2, mode] = dz.ravel()[selected_flat]
    return gradients


def fit_method(
    name: str,
    checkpoint: Path,
    points: np.ndarray,
    geometry: np.ndarray,
    weight: float,
    target_potential: np.ndarray,
    target_velocity: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    cfg, model, payload = base.load_model(checkpoint)
    raw_potential, raw_gradient = base.evaluate_raw_learned_basis(
        cfg, model, points, geometry, 4096
    )
    potential, gradient, transform, rayleigh, peeled_metrics = peel_with_transform(
        raw_potential,
        raw_gradient,
        weight,
        cfg.reorthogonalization_passes,
    )
    velocity = base.velocity_from_potential_gradient(gradient)
    reconstruction = base.fit_reconstruction(
        potential,
        velocity,
        target_potential,
        target_velocity,
        fit_indices,
        test_indices,
        cfg.modes,
        1.0e-8,
    )
    coefficients, _ = base.ridge_fit_coefficients(
        potential, target_potential, fit_indices, cfg.modes, 1.0e-8
    )
    result = {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "config": cfg.__dict__,
        "peeled_basis": {
            **peeled_metrics,
            "rayleigh_first16": rayleigh[:16].tolist(),
        },
        "reconstruction": reconstruction,
        "raw_reconstruction_coefficients": (transform @ coefficients).tolist(),
    }
    del raw_potential, raw_gradient, potential, gradient, velocity, model
    return result


def evaluate_raw_reconstruction(
    checkpoint: Path,
    raw_coefficients: np.ndarray,
    points: np.ndarray,
    geometry: np.ndarray,
) -> np.ndarray:
    cfg, model, _ = base.load_model(checkpoint)
    potential = np.zeros((len(points), 3), dtype=np.float64)
    gradient = np.zeros((len(points), 3, 3), dtype=np.float64)
    raw_potential, raw_gradient = base.evaluate_raw_learned_basis(
        cfg, model, points, geometry, 4096
    )
    potential[:] = np.einsum("nck,k->nc", raw_potential, raw_coefficients)
    gradient[:] = np.einsum("ncdk,k->ncd", raw_gradient, raw_coefficients)
    return base.velocity_from_potential_gradient(gradient)


def reconstructed_laplacian_velocity(
    eigenvectors: np.ndarray,
    coefficients: np.ndarray,
    mask: np.ndarray,
    spacing: float,
) -> np.ndarray:
    scalar_modes = eigenvectors.shape[1]
    component_coefficients = np.zeros((scalar_modes, 3), dtype=np.float64)
    for mode, coefficient in enumerate(coefficients):
        component_coefficients[mode // 3, mode % 3] += coefficient
    potential = eigenvectors @ component_coefficients
    gradient = np.empty((len(eigenvectors), 3, 3), dtype=np.float64)
    for component in range(3):
        volume = np.zeros(mask.shape, dtype=np.float64)
        volume[mask] = potential[:, component]
        dz, dy, dx = np.gradient(volume, spacing, edge_order=2)
        gradient[:, component, 0] = dx[mask]
        gradient[:, component, 1] = dy[mask]
        gradient[:, component, 2] = dz[mask]
    return base.velocity_from_potential_gradient(gradient)


def slice_points(
    coordinates: np.ndarray,
    plane: str,
    index: int,
) -> tuple[np.ndarray, str, str, str]:
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    fixed = float(coordinates[index])
    if plane == "xy":
        return np.column_stack((horizontal.ravel(), vertical.ravel(), np.full(horizontal.size, fixed))), "x", "y", "z"
    if plane == "xz":
        return np.column_stack((horizontal.ravel(), np.full(horizontal.size, fixed), vertical.ravel())), "x", "z", "y"
    return np.column_stack((np.full(horizontal.size, fixed), horizontal.ravel(), vertical.ravel())), "y", "z", "x"


def plot_comparison(
    rows: list[dict[str, Any]],
    combined_rl2: dict[str, float],
    output: Path,
) -> None:
    titles = (
        "Target\nvelocity rL2=0",
        f"Original 500 epoch\nvelocity rL2={combined_rl2['baseline']:.3f}",
        f"Sweep winner\nvelocity rL2={combined_rl2['winner']:.3f}",
        f"Laplacian, K=64\nvelocity rL2={combined_rl2['laplacian']:.3f}",
        f"Original error\nrL2={combined_rl2['baseline']:.3f}",
        f"Winner error\nrL2={combined_rl2['winner']:.3f}",
        f"Laplacian error\nrL2={combined_rl2['laplacian']:.3f}",
    )
    figure = plt.figure(figsize=(19.0, 8.9), constrained_layout=True, facecolor="white")
    grid = figure.add_gridspec(3, 9, width_ratios=(1, 1, 1, 1, 1, 1, 1, 0.04, 0.04))
    axes = np.empty((3, 7), dtype=object)
    for row_index, row in enumerate(rows):
        magnitudes = [np.linalg.norm(field, axis=2) for field in row["fields"]]
        physical_limit = plot_utils.robust_limit(magnitudes[:4])
        error_limit = plot_utils.robust_limit(magnitudes[4:])
        for column, (field, magnitude) in enumerate(zip(row["fields"], magnitudes)):
            axis = figure.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            image = np.ma.array(magnitude, mask=~row["mask"])
            axis.imshow(
                image,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="Blues" if column < 4 else "Reds",
                vmin=0.0,
                vmax=physical_limit if column < 4 else error_limit,
                interpolation="bilinear",
            )
            axis.contour(
                row["mask"].astype(float), levels=[0.5], colors=[NATURE_GREY],
                linewidths=0.5, origin="lower", extent=(0, 1, 0, 1),
            )
            horizontal, vertical = plot_utils.projected_components(field, row["plane"])
            plot_utils.draw_streamlines(
                axis, row["coordinates"], horizontal, vertical, row["mask"]
            )
            axis.set_aspect("equal")
            axis.set_xticks((0, 0.5, 1))
            axis.set_yticks((0, 0.5, 1))
            if row_index == 0:
                axis.set_title(titles[column], color=NATURE_DARK, fontsize=9)
            if row_index == 2:
                axis.set_xlabel(row["horizontal_label"])
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(
                    f"{row['plane']} ({row['fixed_axis']}={row['fixed_value']:.3f})\n{row['vertical_label']}"
                )
            else:
                axis.set_yticklabels([])
        physical_axis = figure.add_subplot(grid[row_index, 7])
        error_axis = figure.add_subplot(grid[row_index, 8])
        figure.colorbar(axes[row_index, 3].images[0], cax=physical_axis).set_label(r"$|\mathbf{u}|$")
        figure.colorbar(axes[row_index, 6].images[0], cax=error_axis).set_label(r"$|\Delta\mathbf{u}|$")
    figure.suptitle(
        "Unseen connected three-sphere reconstruction — true 128³ Dirichlet Laplacian and 128×128 slices",
        color=NATURE_DARK,
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=260, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=32768)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    metadata = json.loads((output / "laplacian_scalar_metadata_128.json").read_text(encoding="utf-8"))
    if not metadata.get("complete"):
        raise RuntimeError("128^3 Laplacian cache is incomplete")
    eigenvectors = np.load(output / "laplacian_scalar_eigenvectors_128.npy", mmap_mode="r")
    baseline_config = base.training.Config(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    geometry = base.training.geometry_from_token(baseline_config, 12_000_001)
    grid = base.build_grid(baseline_config, 128, geometry)
    winner = json.loads((SWEEP_ROOT / "winner.json").read_text(encoding="utf-8"))
    winner_checkpoint = Path(winner["winner_checkpoint"])

    rng = np.random.default_rng(20260825)
    sample_indices = np.sort(rng.choice(len(grid["points"]), size=args.sample_count, replace=False))
    points = grid["points"][sample_indices]
    fit_count = int(round(0.70 * args.sample_count))
    permutation = rng.permutation(args.sample_count)
    fit_indices, test_indices = permutation[:fit_count], permutation[fit_count:]
    target_potential, target_gradient, target_spec = base.target_potential_and_gradient(
        baseline_config, points, geometry, 20_260_824
    )
    target_velocity = base.velocity_from_potential_gradient(target_gradient)
    weight = len(grid["points"]) * grid["spacing"] ** 3 / args.sample_count

    baseline = fit_method(
        "original_500_epoch", BASELINE_CHECKPOINT, points, geometry, weight,
        target_potential, target_velocity, fit_indices, test_indices,
    )
    swept = fit_method(
        "sweep_winner", winner_checkpoint, points, geometry, weight,
        target_potential, target_velocity, fit_indices, test_indices,
    )
    scalar_gradient = scalar_gradients_at_indices(
        eigenvectors, grid["mask"], grid["spacing"], sample_indices
    )
    lap_potential, lap_gradient, lap_rayleigh, _ = base.expand_laplacian_basis(
        np.asarray(eigenvectors[sample_indices], dtype=np.float64),
        scalar_gradient,
        np.asarray(metadata["eigenvalues"]),
        64,
    )
    lap_velocity = base.velocity_from_potential_gradient(lap_gradient)
    lap_reconstruction = base.fit_reconstruction(
        lap_potential, lap_velocity, target_potential, target_velocity,
        fit_indices, test_indices, 64, 1.0e-8,
    )
    lap_coefficients, _ = base.ridge_fit_coefficients(
        lap_potential, target_potential, fit_indices, 64, 1.0e-8
    )

    lap_full_velocity = reconstructed_laplacian_velocity(
        eigenvectors, lap_coefficients, grid["mask"], grid["spacing"]
    )
    index = np.full(grid["shape"], -1, dtype=np.int64)
    index[grid["mask"]] = np.arange(len(grid["points"]))
    center = np.average(geometry[:, :3], axis=0, weights=geometry[:, 3] ** 3)
    plane_specs = (
        ("xy", int(np.argmin(abs(grid["coordinates"] - center[2])))),
        ("xz", int(np.argmin(abs(grid["coordinates"] - center[1])))),
        ("yz", int(np.argmin(abs(grid["coordinates"] - center[0])))),
    )
    rows: list[dict[str, Any]] = []
    squared_errors = {"baseline": 0.0, "winner": 0.0, "laplacian": 0.0}
    target_square = 0.0
    for plane, plane_index in plane_specs:
        all_points, horizontal_label, vertical_label, fixed_axis = slice_points(
            grid["coordinates"], plane, plane_index
        )
        if plane == "xy":
            slice_mask = grid["mask"][plane_index, :, :]
        elif plane == "xz":
            slice_mask = grid["mask"][:, plane_index, :]
        else:
            slice_mask = grid["mask"][:, :, plane_index]
        valid_points = all_points[slice_mask.ravel()].astype(np.float32)
        _, target_slice_gradient, _ = base.target_potential_and_gradient(
            baseline_config, valid_points, geometry, 20_260_824
        )
        target_slice = base.velocity_from_potential_gradient(target_slice_gradient)
        baseline_slice = evaluate_raw_reconstruction(
            BASELINE_CHECKPOINT,
            np.asarray(baseline["raw_reconstruction_coefficients"]),
            valid_points,
            geometry,
        )
        winner_slice = evaluate_raw_reconstruction(
            winner_checkpoint,
            np.asarray(swept["raw_reconstruction_coefficients"]),
            valid_points,
            geometry,
        )
        if plane == "xy":
            lap_indices = index[plane_index, :, :][slice_mask]
        elif plane == "xz":
            lap_indices = index[:, plane_index, :][slice_mask]
        else:
            lap_indices = index[:, :, plane_index][slice_mask]
        lap_slice = lap_full_velocity[lap_indices]
        fields = []
        for values in (target_slice, baseline_slice, winner_slice, lap_slice):
            field = np.full((128, 128, 3), np.nan)
            field[slice_mask] = values
            fields.append(field)
        fields.extend((fields[1] - fields[0], fields[2] - fields[0], fields[3] - fields[0]))
        target_square += float(np.sum(target_slice**2))
        squared_errors["baseline"] += float(np.sum((baseline_slice - target_slice) ** 2))
        squared_errors["winner"] += float(np.sum((winner_slice - target_slice) ** 2))
        squared_errors["laplacian"] += float(np.sum((lap_slice - target_slice) ** 2))
        rows.append(
            {
                "plane": plane,
                "fixed_axis": fixed_axis,
                "fixed_value": float(grid["coordinates"][plane_index]),
                "horizontal_label": horizontal_label,
                "vertical_label": vertical_label,
                "coordinates": grid["coordinates"],
                "mask": slice_mask,
                "fields": fields,
            }
        )
    combined_rl2 = {
        name: float(np.sqrt(value / target_square)) for name, value in squared_errors.items()
    }
    png = output / "baseline_winner_laplacian_vector_comparison_128.png"
    plot_comparison(rows, combined_rl2, png)

    result = {
        "complete": True,
        "geometry_token": 12_000_001,
        "target": target_spec,
        "grid": {
            "resolution": 128,
            "fluid_nodes": len(grid["points"]),
            "spacing": grid["spacing"],
            "sample_nodes": args.sample_count,
            "fit_nodes": len(fit_indices),
            "test_nodes": len(test_indices),
        },
        "laplacian": {
            "eigensolve": metadata,
            "rayleigh_mean": float(np.mean(lap_rayleigh)),
            "reconstruction": lap_reconstruction,
        },
        "baseline": baseline,
        "winner": swept,
        "winner_selection": winner,
        "combined_three_slice_velocity_rl2": combined_rl2,
        "validation_mean_improvement_fraction": float(
            1.0 - winner["winner_score"][0] / 310.5197274237871
        ),
        "reporting_scope": "peeled_basis_only",
        "png": str(png),
    }
    atomic_json(output / "baseline_winner_laplacian_comparison.json", result)
    csv_path = output / "baseline_winner_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("method", "test_potential_rl2", "test_velocity_rl2", "slice_velocity_rl2"),
        )
        writer.writeheader()
        for name, metrics in (
            ("baseline", baseline["reconstruction"]),
            ("winner", swept["reconstruction"]),
            ("laplacian", lap_reconstruction),
        ):
            writer.writerow(
                {
                    "method": name,
                    "test_potential_rl2": metrics["test_potential_relative_l2"],
                    "test_velocity_rl2": metrics["test_velocity_relative_l2"],
                    "slice_velocity_rl2": combined_rl2[name],
                }
            )
    print(json.dumps(result["combined_three_slice_velocity_rl2"], indent=2))


if __name__ == "__main__":
    main()
