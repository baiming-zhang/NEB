"""Plot the first 16 vector modes for baseline, sweep winner, and Laplacian."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))
import evaluate_sweep_winner_128 as comparison  # noqa: E402
import evaluate_unseen_vs_laplacian as base  # noqa: E402
import plot_unseen_vector_reconstruction_256 as plot_utils  # noqa: E402


SWEEP_ROOT = CASE_ROOT / "results" / "k64_siren_dirichlet_e500_hyperparameter_sweep"
OUTPUT_ROOT = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction_128"
CONFIG_PATH = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
BASELINE_CHECKPOINT = (
    CASE_ROOT / "results" / "k64_siren_dirichlet_e500" / "run_001" / "checkpoints" / "final.pt"
)
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def extract_modes(
    name: str,
    checkpoint: Path,
    sample_points: np.ndarray,
    slice_points: np.ndarray,
    slice_mask: np.ndarray,
    geometry: np.ndarray,
    weight: float,
) -> dict[str, Any]:
    cfg, model, payload = base.load_model(checkpoint)
    raw_potential, raw_gradient = base.evaluate_raw_learned_basis(
        cfg, model, sample_points, geometry, 4096
    )
    _, _, transform, rayleigh, metrics = comparison.peel_with_transform(
        raw_potential,
        raw_gradient,
        weight,
        cfg.reorthogonalization_passes,
    )
    del raw_potential, raw_gradient
    _, slice_raw_gradient = base.evaluate_raw_learned_basis(
        cfg, model, slice_points, geometry, 4096
    )
    peeled_gradient = np.einsum(
        "ncdj,jk->ncdk", slice_raw_gradient, transform[:, :16], optimize=True
    )
    velocity = base.velocity_from_potential_gradient(peeled_gradient)
    fields: list[np.ndarray] = []
    for mode in range(16):
        field = np.full((128, 128, 3), np.nan, dtype=np.float64)
        field[slice_mask] = velocity[:, :, mode]
        fields.append(field)
    return {
        "name": name,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "config": cfg.__dict__,
        "rayleigh_first16": rayleigh[:16].tolist(),
        "peeled_metrics": metrics,
        "fields": fields,
        "mode_labels": [f"mode {mode + 1:02d}" for mode in range(16)],
        "atlas_scope": "peeled vector modes",
        "subtitle": (
            f"LR={cfg.learning_rate:.1e}, first ω={cfg.siren_first_omega0:g}, "
            f"hidden ω={cfg.siren_hidden_omega0:g} · no cross-model sign or direction alignment"
        ),
    }


def extract_laplacian_modes(
    eigenvectors: np.ndarray,
    metadata: dict[str, Any],
    mask: np.ndarray,
    spacing: float,
    z_index: int,
) -> dict[str, Any]:
    """Expand six scalar eigenfunctions into the first 16 antisymmetric modes."""
    fields: list[np.ndarray] = []
    labels: list[str] = []
    rayleigh: list[float] = []
    eigenvalues = np.asarray(metadata["eigenvalues"], dtype=np.float64)
    component_names = ("A12", "A13", "A23")
    gradients: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scalar_mode in range(6):
        volume = np.zeros(mask.shape, dtype=np.float64)
        volume[mask] = np.asarray(eigenvectors[:, scalar_mode], dtype=np.float64)
        derivative_z, derivative_y, derivative_x = np.gradient(
            volume, spacing, edge_order=2
        )
        gradients[scalar_mode] = (
            derivative_x[z_index],
            derivative_y[z_index],
            derivative_z[z_index],
        )
    slice_mask = mask[z_index]
    for mode in range(16):
        scalar_mode = mode // 3
        component = mode % 3
        derivative_x, derivative_y, derivative_z = gradients[scalar_mode]
        field = np.zeros((*slice_mask.shape, 3), dtype=np.float64)
        if component == 0:
            field[..., 0] = derivative_y
            field[..., 1] = -derivative_x
        elif component == 1:
            field[..., 0] = derivative_z
            field[..., 2] = -derivative_x
        else:
            field[..., 1] = derivative_z
            field[..., 2] = -derivative_y
        field[~slice_mask] = np.nan
        fields.append(field)
        labels.append(
            f"mode {mode + 1:02d} · {component_names[component]}"
        )
        rayleigh.append(float(eigenvalues[scalar_mode]))
    return {
        "name": "128³ homogeneous-Dirichlet Laplacian",
        "rayleigh_first16": rayleigh,
        "fields": fields,
        "mode_labels": labels,
        "atlas_scope": "vector modes",
        "subtitle": (
            "22 scalar eigenfunctions × (A12, A13, A23) → K=64 · "
            f"max eigensolver relative residual={metadata['relative_residual_max']:.2e}"
        ),
        "eigensolve": metadata,
    }


def render_atlas(
    result: dict[str, Any],
    coordinates: np.ndarray,
    slice_mask: np.ndarray,
    z_value: float,
    shared_limit: float,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure, axes = plt.subplots(4, 4, figsize=(11.3, 10.6), constrained_layout=True)
    norm = Normalize(vmin=0.0, vmax=shared_limit)
    image_handle = None
    for mode, axis in enumerate(axes.ravel()):
        field = result["fields"][mode]
        magnitude = np.linalg.norm(field, axis=2)
        image_handle = axis.imshow(
            np.ma.array(magnitude, mask=~slice_mask),
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="Blues",
            norm=norm,
            interpolation="bilinear",
            zorder=1,
        )
        axis.contour(
            slice_mask.astype(float),
            levels=[0.5],
            colors=[NATURE_GREY],
            linewidths=0.45,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            zorder=2,
        )
        horizontal, vertical = plot_utils.projected_components(field, "xy")
        plot_utils.draw_streamlines(
            axis,
            coordinates,
            horizontal,
            vertical,
            slice_mask,
        )
        axis.set_title(
            f"{result['mode_labels'][mode]}  ·  Rayleigh={result['rayleigh_first16'][mode]:.3f}",
            color=NATURE_DARK,
            pad=3,
        )
        axis.set_aspect("equal")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_yticks((0.0, 0.5, 1.0))
        row, column = divmod(mode, 4)
        if row == 3:
            axis.set_xlabel("x")
        else:
            axis.set_xticklabels([])
        if column == 0:
            axis.set_ylabel("y")
        else:
            axis.set_yticklabels([])
        for spine in axis.spines.values():
            spine.set_color(NATURE_GREY)
            spine.set_linewidth(0.5)
    assert image_handle is not None
    colorbar = figure.colorbar(image_handle, ax=axes, location="right", shrink=0.88, pad=0.015)
    colorbar.set_label(r"$|\mathbf{u}_k|$ (shared scale)")
    figure.suptitle(
        f"{result['name']} — first 16 {result['atlas_scope']} on unseen geometry\n"
        f"xy slice z={z_value:.3f} · {result['subtitle']}",
        color=NATURE_DARK,
        fontsize=12.2,
        fontweight="medium",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    baseline_config = base.training.Config(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    geometry = base.training.geometry_from_token(baseline_config, 12_000_001)
    grid = base.build_grid(baseline_config, 128, geometry)
    rng = np.random.default_rng(20260825)
    sample_indices = np.sort(rng.choice(len(grid["points"]), size=32768, replace=False))
    sample_points = grid["points"][sample_indices]
    weight = len(grid["points"]) * grid["spacing"] ** 3 / len(sample_points)
    center_z = float(np.average(geometry[:, 2], weights=geometry[:, 3] ** 3))
    z_index = int(np.argmin(np.abs(grid["coordinates"] - center_z)))
    slice_mask = grid["mask"][z_index]
    horizontal, vertical = np.meshgrid(grid["coordinates"], grid["coordinates"])
    all_slice_points = np.column_stack(
        (
            horizontal.ravel(),
            vertical.ravel(),
            np.full(horizontal.size, grid["coordinates"][z_index]),
        )
    )
    valid_slice_points = all_slice_points[slice_mask.ravel()].astype(np.float32)
    winner = json.loads((SWEEP_ROOT / "winner.json").read_text(encoding="utf-8"))
    laplacian_metadata = json.loads(
        (OUTPUT_ROOT / "laplacian_scalar_metadata_128.json").read_text(encoding="utf-8")
    )
    if not laplacian_metadata.get("complete"):
        raise RuntimeError("128³ Laplacian cache is incomplete")
    eigenvectors = np.load(
        OUTPUT_ROOT / "laplacian_scalar_eigenvectors_128.npy", mmap_mode="r"
    )
    results = [
        extract_modes(
            "Original 500-epoch baseline",
            BASELINE_CHECKPOINT,
            sample_points,
            valid_slice_points,
            slice_mask,
            geometry,
            weight,
        ),
        extract_modes(
            "Sweep winner: joint_best_all",
            Path(winner["winner_checkpoint"]),
            sample_points,
            valid_slice_points,
            slice_mask,
            geometry,
            weight,
        ),
        extract_laplacian_modes(
            eigenvectors,
            laplacian_metadata,
            grid["mask"],
            grid["spacing"],
            z_index,
        ),
    ]
    finite_speeds = []
    for result in results:
        for field in result["fields"]:
            speed = np.linalg.norm(field, axis=2)
            finite_speeds.append(speed[np.isfinite(speed)])
    shared_limit = float(np.percentile(np.concatenate(finite_speeds), 99.0))
    output_paths = (
        OUTPUT_ROOT / "baseline_first16_peeled_vector_modes.png",
        OUTPUT_ROOT / "winner_first16_peeled_vector_modes.png",
        OUTPUT_ROOT / "laplacian_first16_vector_modes.png",
    )
    for result, output in zip(results, output_paths):
        render_atlas(
            result,
            grid["coordinates"],
            slice_mask,
            float(grid["coordinates"][z_index]),
            shared_limit,
            output,
        )
    payload = {
        "complete": True,
        "geometry_token": 12_000_001,
        "grid_resolution": 128,
        "slice": {"plane": "xy", "z_index": z_index, "z_value": float(grid["coordinates"][z_index])},
        "peeling_nodes": len(sample_points),
        "shared_speed_color_limit_p99": shared_limit,
        "alignment": "none; no sign, degenerate-block, or direction alignment",
        "reporting_scope": "peeled_basis_only",
        "baseline": {
            key: value for key, value in results[0].items() if key != "fields"
        },
        "winner": {
            key: value for key, value in results[1].items() if key != "fields"
        },
        "laplacian": {
            key: value for key, value in results[2].items() if key != "fields"
        },
        "pngs": [str(path.resolve()) for path in output_paths],
    }
    json_path = OUTPUT_ROOT / "baseline_winner_laplacian_first16_mode_rayleigh.json"
    csv_path = OUTPUT_ROOT / "baseline_winner_laplacian_first16_mode_rayleigh.csv"
    comparison.atomic_json(json_path, payload)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "baseline_rayleigh", "winner_rayleigh", "laplacian_rayleigh"))
        for mode in range(16):
            writer.writerow(
                (
                    mode + 1,
                    payload["baseline"]["rayleigh_first16"][mode],
                    payload["winner"]["rayleigh_first16"][mode],
                    payload["laplacian"]["rayleigh_first16"][mode],
                )
            )
    print(
        json.dumps(
            {
                "baseline": payload["baseline"]["rayleigh_first16"],
                "winner": payload["winner"]["rayleigh_first16"],
                "laplacian": payload["laplacian"]["rayleigh_first16"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
