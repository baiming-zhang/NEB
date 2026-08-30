#!/usr/bin/env python3
"""Compare the first 16 peeled modes of the original and sweep-winner bases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_unseen_k64_vs_laplacian import (
    BLUE,
    RED,
    add_solid,
    airfoil_area,
    as_grid,
    fluid_points,
    geometry_laplacian,
    load_model,
    peeled_transform,
    raw_fields,
    save_figure,
    write_json,
)


LAPLACIAN_COLOR = "#00A087"


def ensure_matching_problem(original_cfg: Any, winner_cfg: Any) -> None:
    fields = (
        "modes",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "chord",
        "quarter_chord_x",
        "quarter_chord_y",
        "thickness_min",
        "thickness_max",
        "geometry_mode",
        "alpha_min_deg",
        "alpha_max_deg",
        "basis_objective",
        "causal_sorting",
        "mass_floor",
    )
    mismatches = [
        field
        for field in fields
        if getattr(original_cfg, field) != getattr(winner_cfg, field)
    ]
    if mismatches:
        raise ValueError(f"original/winner problem mismatch: {mismatches}")
    if int(original_cfg.reorthogonalization_passes) != 2:
        raise ValueError("original checkpoint does not use two-pass MGS")
    if int(winner_cfg.reorthogonalization_passes) != 2:
        raise ValueError("winner checkpoint does not use two-pass MGS")


def canonicalize_display_sign(modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = modes.copy()
    signs = np.ones(len(result), dtype=np.float32)
    for index, field in enumerate(result):
        pivot = int(np.argmax(np.abs(field)))
        if field[pivot] < 0.0:
            result[index] *= -1.0
            signs[index] = -1.0
    return result, signs


def evaluate_first16(
    model: Any,
    cfg: Any,
    transform: np.ndarray,
    points: np.ndarray,
    beta: float,
    alpha: float,
    batch_size: int,
) -> np.ndarray:
    modes = np.empty((16, len(points)), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        psi, gx, gy, u, v, omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha
        )
        modes[:, start:stop] = (psi @ transform[:, :16]).T
        del psi, gx, gy, u, v, omega
    cp.get_default_memory_pool().free_all_blocks()
    return modes


def plot_modes(
    original: np.ndarray,
    winner: np.ndarray,
    laplacian: np.ndarray,
    original_rayleigh: np.ndarray,
    winner_rayleigh: np.ndarray,
    laplacian_rayleigh: np.ndarray,
    solid: np.ndarray,
    extent: tuple[float, float, float, float],
    beta: float,
    alpha: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(4, 12, figsize=(28.0, 8.8), constrained_layout=False)
    for row in range(4):
        for pair in range(4):
            rank = 4 * row + pair
            for offset, (field, label, value, color) in enumerate(
                (
                    (original[rank], "Original", original_rayleigh[rank], BLUE),
                    (winner[rank], "Winner", winner_rayleigh[rank], RED),
                    (
                        laplacian[rank],
                        "Airfoil FD",
                        laplacian_rayleigh[rank],
                        LAPLACIAN_COLOR,
                    ),
                )
            ):
                axis = axes[row, 3 * pair + offset]
                normalized = field / max(float(np.max(np.abs(field))), 1.0e-12)
                axis.imshow(
                    np.ma.masked_where(solid, normalized),
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-1.0,
                    vmax=1.0,
                    interpolation="bilinear",
                )
                add_solid(axis, solid, extent)
                axis.set_title(
                    f"{label} {rank + 1:02d}\n$R_\\psi$={value:.3f}",
                    fontsize=8.0,
                    color=color,
                    pad=1.5,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal")
    figure.suptitle(
        "Unseen airfoil: original, sweep winner, and geometry-aware FD Laplacian - first 16 modes\n"
        rf"$\beta={beta:.6f}$, $\alpha={alpha:.5f}^\circ$; same airfoil domain and open outlet",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.006,
        "Each mode is normalized by max |psi| for display. Learned signs are canonicalized independently; "
        "no cross-basis or degenerate-block direction alignment is used.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.012,
        right=0.995,
        top=0.89,
        bottom=0.085,
        wspace=0.14,
        hspace=0.34,
    )
    save_figure(figure, output)


def config_summary(cfg: Any) -> dict[str, Any]:
    return {
        "width": int(cfg.hidden),
        "depth": int(cfg.hidden_layers),
        "first_omega": float(cfg.siren_first_omega0),
        "hidden_omega": float(cfg.siren_hidden_omega0),
        "learning_rate": float(cfg.learning_rate),
        "min_learning_rate": float(cfg.min_learning_rate),
        "grad_clip": float(cfg.grad_clip),
        "points_per_step": int(cfg.points_per_step),
        "endpoint_period": int(cfg.endpoint_period),
        "reorthogonalization_passes": int(cfg.reorthogonalization_passes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--winner-checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument("--grid-stride", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_metadata, original_cfg, original_model = load_model(
        args.original_checkpoint
    )
    winner_metadata, winner_cfg, winner_model = load_model(args.winner_checkpoint)
    ensure_matching_problem(original_cfg, winner_cfg)

    with np.load(args.gt, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
        gt_metadata = json.loads(str(archive["metadata"]))
    beta = float(gt_metadata["beta"])
    alpha = float(gt_metadata["alpha_deg"])
    domain_area = (
        (original_cfg.xmax - original_cfg.xmin)
        * (original_cfg.ymax - original_cfg.ymin)
        - airfoil_area(original_cfg, beta)
    )
    integration_points = fluid_points(
        original_cfg,
        args.peeling_points,
        beta,
        alpha,
        original_cfg.seed + 2_300_001,
    )

    print("[1/5] original geometry-Rayleigh sorting and peeling", flush=True)
    original_psi, original_gx, original_gy, original_u, original_v, original_omega = raw_fields(
        original_model, original_cfg, integration_points, beta, alpha
    )
    original_transform, original_rayleigh, original_diagnostics = peeled_transform(
        original_cfg,
        original_psi,
        original_gx,
        original_gy,
        domain_area,
    )
    del original_psi, original_gx, original_gy, original_u, original_v, original_omega
    cp.get_default_memory_pool().free_all_blocks()

    print("[2/5] winner geometry-Rayleigh sorting and peeling", flush=True)
    winner_psi, winner_gx, winner_gy, winner_u, winner_v, winner_omega = raw_fields(
        winner_model, winner_cfg, integration_points, beta, alpha
    )
    winner_transform, winner_rayleigh, winner_diagnostics = peeled_transform(
        winner_cfg,
        winner_psi,
        winner_gx,
        winner_gy,
        domain_area,
    )
    del winner_psi, winner_gx, winner_gy, winner_u, winner_v, winner_omega
    cp.get_default_memory_pool().free_all_blocks()

    stride = int(args.grid_stride)
    if stride < 1:
        raise ValueError("grid stride must be positive")
    x_plot = x[::stride]
    y_plot = y[::stride]
    solid_plot = solid[::stride, ::stride]
    x_grid, y_grid = np.meshgrid(x_plot, y_plot)
    fluid_plot = ~solid_plot
    plot_points = np.column_stack(
        (x_grid[fluid_plot], y_grid[fluid_plot])
    ).astype(np.float32)

    print("[3/5] first 16 modes on the shared plotting grid", flush=True)
    original_modes = evaluate_first16(
        original_model,
        original_cfg,
        original_transform,
        plot_points,
        beta,
        alpha,
        args.field_batch,
    )
    winner_modes = evaluate_first16(
        winner_model,
        winner_cfg,
        winner_transform,
        plot_points,
        beta,
        alpha,
        args.field_batch,
    )
    original_modes, original_signs = canonicalize_display_sign(original_modes)
    winner_modes, winner_signs = canonicalize_display_sign(winner_modes)
    original_grids = np.stack(
        [as_grid(mode, fluid_plot) for mode in original_modes]
    )
    winner_grids = np.stack([as_grid(mode, fluid_plot) for mode in winner_modes])

    print("[4/5] geometry-aware finite-difference Laplacian", flush=True)
    laplacian_basis = geometry_laplacian(
        x,
        y,
        solid,
        int(original_cfg.modes),
        stride,
        int(original_cfg.seed) + 2_400_001,
    )
    if not np.array_equal(laplacian_basis["solid"], solid_plot):
        raise ValueError("Laplacian and learned plotting masks differ")
    laplacian_grids = np.moveaxis(
        laplacian_basis["psi"][:, :, :16], -1, 0
    ).astype(np.float32)

    print("[5/5] PNG, vector PDF, JSON, and Rayleigh CSV", flush=True)
    stem = args.output_dir / "unseen_first16_original_winner_laplacian"
    plot_modes(
        original_grids,
        winner_grids,
        laplacian_grids,
        original_rayleigh[:16],
        winner_rayleigh[:16],
        laplacian_basis["eigenvalues"][:16],
        solid_plot,
        (float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        beta,
        alpha,
        stem,
    )
    summary = {
        "case": {
            "gt": str(args.gt.resolve()),
            "beta": beta,
            "alpha_deg": alpha,
            "grid_stride": stride,
            "grid_shape": [int(len(y_plot)), int(len(x_plot))],
        },
        "protocol": {
            "same_iid_peeling_points": True,
            "peeling_points": int(args.peeling_points),
            "geometry_rayleigh_causal_sorting": True,
            "two_pass_mgs": True,
            "raw_rank_reported": False,
            "ritz_rotation_used": False,
            "cross_basis_direction_alignment_used": False,
            "degenerate_block_direction_alignment_used": False,
            "display_sign_convention": "largest-magnitude plotted fluid sample positive independently per mode",
        },
        "original": {
            "checkpoint": str(args.original_checkpoint.resolve()),
            "epoch": int(original_metadata["epoch"]),
            "config": config_summary(original_cfg),
            "peeled_rank": int(original_diagnostics["peeled_rank"]),
            "orthogonality_max_abs": float(
                original_diagnostics["peeled_orthogonality_max_abs"]
            ),
            "causal_order_1based": original_diagnostics["causal_order_1based"],
            "first16_rayleigh": original_rayleigh[:16].tolist(),
            "display_signs": original_signs.tolist(),
        },
        "winner": {
            "checkpoint": str(args.winner_checkpoint.resolve()),
            "epoch": int(winner_metadata["epoch"]),
            "config": config_summary(winner_cfg),
            "peeled_rank": int(winner_diagnostics["peeled_rank"]),
            "orthogonality_max_abs": float(
                winner_diagnostics["peeled_orthogonality_max_abs"]
            ),
            "causal_order_1based": winner_diagnostics["causal_order_1based"],
            "first16_rayleigh": winner_rayleigh[:16].tolist(),
            "display_signs": winner_signs.tolist(),
        },
        "laplacian": {
            "label": "geometry-aware five-point finite-difference Laplacian",
            "first16_rayleigh": laplacian_basis["eigenvalues"][:16].tolist(),
            "eigensolve": laplacian_basis["diagnostics"],
        },
        "figures": {
            "png": str(stem.with_suffix(".png").resolve()),
            "pdf": str(stem.with_suffix(".pdf").resolve()),
        },
    }
    summary_path = (
        args.output_dir / "unseen_original_winner_laplacian_modes_summary.json"
    )
    write_json(summary_path, summary)
    csv_path = args.output_dir / "unseen_first16_rayleigh_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "mode",
                "original_rayleigh",
                "winner_rayleigh",
                "airfoil_fd_laplacian_rayleigh",
                "winner_change_vs_original_percent",
            )
        )
        for index in range(16):
            original_value = float(original_rayleigh[index])
            winner_value = float(winner_rayleigh[index])
            writer.writerow(
                (
                    index + 1,
                    original_value,
                    winner_value,
                    float(laplacian_basis["eigenvalues"][index]),
                    100.0 * (winner_value - original_value) / original_value,
                )
            )
    print(json.dumps({
        "original_first16_rayleigh": original_rayleigh[:16].tolist(),
        "winner_first16_rayleigh": winner_rayleigh[:16].tolist(),
        "laplacian_first16_rayleigh": laplacian_basis["eigenvalues"][:16].tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
