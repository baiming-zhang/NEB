"""Compare the first 16 strict matched-grid Ritz modes for two checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from compare_three_gt_eigenfluid_vs_laplacian import (
    eigenfluid_basis,
    matched_grid_ritz,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_connected_dirichlet_psi_basis import load_model
from metaball_runtime import (
    fluid_mask,
    geometry_from_token,
    obstacle_level_numpy,
    outer_level_numpy,
)


F32 = np.float32
BLUE = "#3B4CC0"
RED = "#B40426"
CHARCOAL = "#303030"


def evaluate(
    checkpoint: Path,
    token: int,
    resolution: int,
    peeling_points: int,
    seed: int,
) -> dict[str, Any]:
    metadata, cfg, model = load_model(checkpoint)
    coordinates = np.linspace(0.0, 1.0, resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    valid_grid = fluid_mask(cfg, all_points, float(token)).reshape(
        resolution, resolution
    )
    points = all_points[valid_grid.ravel()]
    psi, _, _, peeling, _ = eigenfluid_basis(
        cfg,
        model,
        points,
        float(token),
        peeling_points,
        seed + 200_003,
    )
    psi, eigenvalues, ritz = matched_grid_ritz(psi, valid_grid, spacing)
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": int(metadata["epoch"]),
        "cfg": cfg,
        "coordinates": coordinates,
        "spacing": spacing,
        "all_points": all_points,
        "valid": valid_grid,
        "psi": psi[:, :16],
        "eigenvalues": eigenvalues[:16],
        "peeling": peeling,
        "ritz": ritz,
    }


def fields_on_grid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    count = values.shape[1]
    fields = np.full((count, valid.size), np.nan, dtype=F32)
    fields[:, valid.ravel()] = values.T.astype(F32, copy=False)
    return fields.reshape(count, *valid.shape)


def save_atlas(
    original: dict[str, Any],
    winner: dict[str, Any],
    laplacian: dict[str, Any],
    token: int,
    output: Path,
) -> dict[str, list[float]]:
    valid = original["valid"]
    if not np.array_equal(valid, winner["valid"]):
        raise RuntimeError("original and winner do not share the same evaluation mask")
    original_values = original["psi"].copy()
    winner_values = winner["psi"].copy()
    laplacian_values = laplacian["psi"].copy()
    winner_correlations: list[float] = []
    laplacian_correlations: list[float] = []
    for mode in range(16):
        left = original_values[:, mode].astype(np.float64, copy=False)
        right = winner_values[:, mode].astype(np.float64, copy=False)
        correlation = float(
            np.dot(left, right)
            / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1.0e-30)
        )
        if correlation < 0.0:
            winner_values[:, mode] *= -1.0
            correlation *= -1.0
        winner_correlations.append(correlation)
        lap_right = laplacian_values[:, mode].astype(np.float64, copy=False)
        lap_correlation = float(
            np.dot(left, lap_right)
            / max(float(np.linalg.norm(left) * np.linalg.norm(lap_right)), 1.0e-30)
        )
        if lap_correlation < 0.0:
            laplacian_values[:, mode] *= -1.0
            lap_correlation *= -1.0
        laplacian_correlations.append(lap_correlation)

    original_fields = fields_on_grid(original_values, valid)
    winner_fields = fields_on_grid(winner_values, valid)
    laplacian_fields = fields_on_grid(laplacian_values, valid)
    cfg = original["cfg"]
    coordinates = original["coordinates"]
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = original["all_points"]
    geometry = geometry_from_token(cfg, float(token))
    outer = outer_level_numpy(cfg, points).reshape(valid.shape)
    obstacle = obstacle_level_numpy(cfg, points, geometry).reshape(valid.shape)

    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.titlesize": 11.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(6, 8, figsize=(22.8, 15.6))
    for group in range(2):
        for column in range(8):
            mode = 8 * group + column
            original_axis = axes[3 * group, column]
            winner_axis = axes[3 * group + 1, column]
            laplacian_axis = axes[3 * group + 2, column]
            combined = np.concatenate(
                (
                    np.abs(original_fields[mode][np.isfinite(original_fields[mode])]),
                    np.abs(winner_fields[mode][np.isfinite(winner_fields[mode])]),
                    np.abs(laplacian_fields[mode][np.isfinite(laplacian_fields[mode])]),
                )
            )
            limit = max(float(np.percentile(combined, 99.5)), 1.0e-12)
            for axis, field, eigenvalue, label_color in (
                (
                    original_axis,
                    original_fields[mode],
                    original["eigenvalues"][mode],
                    BLUE,
                ),
                (
                    winner_axis,
                    winner_fields[mode],
                    winner["eigenvalues"][mode],
                    RED,
                ),
                (
                    laplacian_axis,
                    laplacian_fields[mode],
                    laplacian["eigenvalues"][mode],
                    CHARCOAL,
                ),
            ):
                axis.imshow(
                    field,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    interpolation="bilinear",
                )
                axis.contour(
                    xx, yy, outer, levels=[0.0], colors=[CHARCOAL], linewidths=0.55
                )
                axis.contour(
                    xx,
                    yy,
                    obstacle,
                    levels=[0.0],
                    colors=[CHARCOAL],
                    linewidths=0.7,
                )
                axis.set_title(
                    f"Mode {mode + 1:02d}  |  $R$={eigenvalue:.2f}",
                    color=label_color,
                    fontweight="bold",
                    pad=3.0,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal")
                for spine in axis.spines.values():
                    spine.set_linewidth(0.65)
                    spine.set_color("#606060")

    row_centers = [0.795, 0.660, 0.525, 0.365, 0.230, 0.095]
    row_labels = [
        ("Original run_002", BLUE),
        ("Sweep winner", RED),
        ("FD Laplacian", CHARCOAL),
        ("Original run_002", BLUE),
        ("Sweep winner", RED),
        ("FD Laplacian", CHARCOAL),
    ]
    for center, (label, color) in zip(row_centers, row_labels):
        fig.text(
            0.006,
            center,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13.5,
            fontweight="bold",
            color=color,
        )
    fig.suptitle(
        "First 16 geometry-conditioned streamfunction modes",
        fontsize=22.0,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        (
            f"Unseen connected-metaball geometry token {token} · "
            "strict 320-grid generalized Ritz ordering · K=64"
        ),
        ha="center",
        va="center",
        fontsize=14.0,
        color=CHARCOAL,
    )
    fig.text(
        0.5,
        0.918,
        (
            "Rayleigh value R shown above each mode; color limits are shared within "
            "each three-method group; signs aligned to run_002 only for display"
        ),
        ha="center",
        va="center",
        fontsize=12.0,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.027,
        right=0.996,
        bottom=0.025,
        top=0.89,
        wspace=0.035,
        hspace=0.32,
    )
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "winner": winner_correlations,
        "laplacian": laplacian_correlations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--winner-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", type=int, default=4_000_001)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--peeling-points", type=int, default=16_384)
    parser.add_argument("--seed", type=int, default=20_260_823)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = evaluate(
        args.original_checkpoint,
        args.token,
        args.resolution,
        args.peeling_points,
        args.seed,
    )
    winner = evaluate(
        args.winner_checkpoint,
        args.token,
        args.resolution,
        args.peeling_points,
        args.seed,
    )
    original_geometry = geometry_from_token(original["cfg"], float(args.token))
    winner_geometry = geometry_from_token(winner["cfg"], float(args.token))
    if not np.array_equal(original_geometry, winner_geometry):
        raise RuntimeError("checkpoint geometry definitions differ")

    laplacian_eigenvalues, laplacian_psi, _, _ = finite_difference_laplacian(
        original["valid"], original["spacing"], 16
    )
    laplacian = {
        "psi": laplacian_psi[:, :16],
        "eigenvalues": laplacian_eigenvalues[:16],
    }

    stem = args.output_dir / "original_winner_laplacian_first16_streamfunction_modes"
    correlations = save_atlas(original, winner, laplacian, args.token, stem)
    strict_root = (
        args.winner_checkpoint.parents[5]
        / "siren_k64_connected_hyperparameter_sweep_e500"
        / "09_strict_matched_grid_fairness"
    )
    reference_paths = {
        "original": strict_root
        / "original_run_002"
        / f"resolution_{args.resolution}"
        / "three_gt_reconstruction_summary.json",
        "winner": strict_root
        / "winner"
        / f"resolution_{args.resolution}"
        / "three_gt_reconstruction_summary.json",
    }
    reference_checks: dict[str, float] = {}
    for label, result in (("original", original), ("winner", winner)):
        path = reference_paths[label]
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            case = next(
                item for item in stored["cases"] if item["geometry_token"] == args.token
            )
            expected = np.asarray(
                case["eigenfluid_matched_grid_ritz"]["first16_eigenvalues"]
            )
            difference = float(np.max(np.abs(expected - result["eigenvalues"])))
            reference_checks[label] = difference
            if difference > 1.0e-8:
                raise RuntimeError(
                    f"{label} spectrum does not reproduce strict evaluation: {difference}"
                )
    winner_reference = json.loads(
        reference_paths["winner"].read_text(encoding="utf-8")
    )
    winner_case = next(
        item
        for item in winner_reference["cases"]
        if item["geometry_token"] == args.token
    )
    expected_laplacian = np.asarray(
        winner_case["laplacian"]["first16_eigenvalues"]
    )
    laplacian_difference = float(
        np.max(np.abs(expected_laplacian - laplacian["eigenvalues"]))
    )
    reference_checks["laplacian"] = laplacian_difference
    if laplacian_difference > 1.0e-8:
        raise RuntimeError(
            "Laplacian spectrum does not reproduce strict evaluation: "
            f"{laplacian_difference}"
        )

    rows: list[dict[str, Any]] = []
    for mode in range(16):
        original_value = float(original["eigenvalues"][mode])
        winner_value = float(winner["eigenvalues"][mode])
        laplacian_value = float(laplacian["eigenvalues"][mode])
        rows.append(
            {
                "mode": mode + 1,
                "original_rayleigh": original_value,
                "winner_rayleigh": winner_value,
                "laplacian_rayleigh": laplacian_value,
                "winner_minus_original": winner_value - original_value,
                "laplacian_minus_original": laplacian_value - original_value,
                "relative_difference": abs(winner_value - original_value)
                / original_value,
                "winner_original_field_correlation_after_sign_alignment": (
                    correlations["winner"][mode]
                ),
                "laplacian_original_field_correlation_after_sign_alignment": (
                    correlations["laplacian"][mode]
                ),
            }
        )
    csv_path = args.output_dir / "original_vs_winner_first16_rayleigh.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "geometry_token": args.token,
        "geometry_parameters": original_geometry.tolist(),
        "resolution": args.resolution,
        "peeling_points": args.peeling_points,
        "mode_definition": (
            "geometry-Rayleigh causal sorting, two-pass peeling, then strict "
            "matched-grid generalized Ritz ordering in the common five-point "
            "Dirichlet Laplacian metric"
        ),
        "field": "streamfunction psi",
        "original_checkpoint": original["checkpoint"],
        "winner_checkpoint": winner["checkpoint"],
        "strict_evaluation_reproduction_max_abs": reference_checks,
        "modes": rows,
        "outputs": {
            "atlas_png": str(stem.with_suffix(".png")),
            "atlas_pdf": str(stem.with_suffix(".pdf")),
            "rayleigh_csv": str(csv_path),
        },
    }
    json_path = args.output_dir / "original_vs_winner_first16_summary.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
