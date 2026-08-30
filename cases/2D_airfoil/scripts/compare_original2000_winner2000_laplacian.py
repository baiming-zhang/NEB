#!/usr/bin/env python3
"""Fair K64 comparison of Original-E2000, Winner-E2000, and airfoil FD modes.

The learned bases are independently geometry-Rayleigh sorted and peeled with
their configured two-pass MGS.  The comparison never rotates or aligns the
learned bases.  Three reproducibly selected independent NS cases (airfoil
pointing up, centered, and down) are reconstructed from the same temporal
fluctuation and velocity projection samples with each K=64 basis.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from evaluate_unseen_k64_vs_laplacian import (
    BLUE,
    CHARCOAL,
    RED,
    add_solid,
    airfoil_area,
    as_grid,
    body_numpy,
    fluid_points,
    geometry_laplacian,
    geometry_laplacian_fields,
    geometry_laplacian_interpolators,
    load_model,
    peeled_transform,
    raw_fields,
    relative_scalar,
    relative_velocity,
    save_figure,
    velocity_projection,
    write_json,
)


GREEN = "#008176"
METHODS = ("original_e2000", "winner_e2000", "laplacian")
METHOD_LABELS = {
    "original_e2000": "Original E2000",
    "winner_e2000": "Winner E2000",
    "laplacian": "Airfoil FD Laplacian",
}
METHOD_COLORS = {
    "original_e2000": BLUE,
    "winner_e2000": RED,
    "laplacian": GREEN,
}


def now() -> str:
    return datetime.now().astimezone().isoformat()


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
        "reorthogonalization_passes",
        "rank_reporting",
    )
    mismatches = [
        name
        for name in fields
        if getattr(original_cfg, name) != getattr(winner_cfg, name)
    ]
    if mismatches:
        raise ValueError(f"original/winner problem mismatch: {mismatches}")
    for label, cfg in (("original", original_cfg), ("winner", winner_cfg)):
        if int(cfg.modes) != 64:
            raise ValueError(f"{label} basis is K={cfg.modes}, expected K=64")
        if int(cfg.reorthogonalization_passes) != 2:
            raise ValueError(f"{label} basis does not use two-pass MGS")
        if bool(cfg.d4_symmetry_augmentation):
            raise ValueError(f"{label} basis unexpectedly uses D4")
        if bool(cfg.analytic_eigenmodes):
            raise ValueError(f"{label} basis unexpectedly uses analytic modes")
        if bool(cfg.degenerate_block_direction_alignment):
            raise ValueError(f"{label} basis unexpectedly aligns degenerate blocks")


def config_summary(cfg: Any) -> dict[str, Any]:
    return {
        "epochs": int(cfg.epochs),
        "modes": int(cfg.modes),
        "width": int(cfg.hidden),
        "depth": int(cfg.hidden_layers),
        "first_omega": float(cfg.siren_first_omega0),
        "hidden_omega": float(cfg.siren_hidden_omega0),
        "learning_rate": float(cfg.learning_rate),
        "min_learning_rate": float(cfg.min_learning_rate),
        "points_per_step": int(cfg.points_per_step),
        "grad_clip": float(cfg.grad_clip),
        "endpoint_period": int(cfg.endpoint_period),
        "reorthogonalization_passes": int(cfg.reorthogonalization_passes),
        "basis_objective": cfg.basis_objective,
        "causal_sorting": cfg.causal_sorting,
        "rank_reporting": cfg.rank_reporting,
    }


def final_training_metrics(checkpoint: Path) -> dict[str, Any]:
    history_path = checkpoint.parents[1] / "data/hardcoded_training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    row = history[-1]
    return {
        "epoch": int(row["epoch"]),
        "validation_mean": float(row["validation_loss"]),
        "validation_p95": float(row["validation_loss_p95"]),
        "validation_max": float(row["validation_loss_max"]),
        "peeled_rank": int(row["validation_peeled_rank"]),
        "orthogonality_max_abs": float(row["validation_orth_error"]),
        "learning_rate": float(row["learning_rate"]),
        "training_elapsed_seconds": float(
            row.get("training_elapsed_seconds", row.get("seconds", math.nan))
        ),
    }


def learned_geometry_state(
    model: Any,
    cfg: Any,
    beta: float,
    alpha: float,
    count: int,
    seed_offset: int,
) -> dict[str, Any]:
    area = (
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    points = fluid_points(cfg, count, beta, alpha, int(cfg.seed) + seed_offset)
    psi, gx, gy, u, v, omega = raw_fields(model, cfg, points, beta, alpha)
    transform, rayleigh, diagnostics = peeled_transform(cfg, psi, gx, gy, area)
    del points, psi, gx, gy, u, v, omega
    cp.get_default_memory_pool().free_all_blocks()
    return {
        "transform": transform,
        "rayleigh": rayleigh,
        "diagnostics": diagnostics,
    }


def first_modes(
    model: Any,
    cfg: Any,
    transform: np.ndarray,
    points: np.ndarray,
    beta: float,
    alpha: float,
    count: int,
    batch_size: int,
) -> np.ndarray:
    modes = np.empty((count, len(points)), dtype=np.float32)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        psi, gx, gy, u, v, omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha
        )
        modes[:, start:stop] = (psi @ transform[:, :count]).T
        del psi, gx, gy, u, v, omega
    cp.get_default_memory_pool().free_all_blocks()
    return modes


def canonicalize_display_sign(modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = modes.copy()
    signs = np.ones(len(result), dtype=np.float32)
    for index, field in enumerate(result):
        valid = np.isfinite(field)
        pivot = int(np.nanargmax(np.where(valid, np.abs(field), -np.inf)))
        flat = field.reshape(-1)
        if flat[pivot] < 0.0:
            result[index] *= -1.0
            signs[index] = -1.0
    return result, signs


def plot_modes_3x8(
    grids: dict[str, np.ndarray],
    spectra: dict[str, np.ndarray],
    solid: np.ndarray,
    extent: tuple[float, float, float, float],
    beta: float,
    alpha: float,
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 8, figsize=(20.0, 6.7), constrained_layout=False)
    for row, method in enumerate(METHODS):
        for rank in range(8):
            axis = axes[row, rank]
            field = grids[method][rank]
            limit = max(float(np.nanmax(np.abs(field))), 1.0e-12)
            normalized = field / limit
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
            symbol = r"\lambda" if method == "laplacian" else r"R_\psi"
            axis.set_title(
                f"mode {rank + 1}\n" + rf"${symbol}$={spectra[method][rank]:.3f}",
                fontsize=8.1,
                pad=1.5,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            if rank == 0:
                axis.set_ylabel(
                    METHOD_LABELS[method],
                    fontsize=10.0,
                    color=METHOD_COLORS[method],
                    weight="bold",
                )
    figure.suptitle(
        "First eight K=64 streamfunction modes on one unseen airfoil\n"
        rf"$\beta={beta:.6f}$, $\alpha={alpha:.5f}^\circ$; identical fluid domain and boundary protocol",
        fontsize=12.0,
        y=0.985,
    )
    figure.text(
        0.5,
        0.008,
        "Each panel is independently normalized by max |psi|. Learned values are peeled-basis "
        "Rayleigh diagonals, while FD values are discrete eigenvalues. No cross-basis alignment.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.885,
        bottom=0.09,
        wspace=0.08,
        hspace=0.23,
    )
    save_figure(figure, output)


def discover_cases(gt_root: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(gt_root.glob("gt_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            frame_count = int(len(archive["steps"]))
        cases.append(
            {
                "path": path,
                "beta": float(metadata["beta"]),
                "alpha_deg": float(metadata["alpha_deg"]),
                "frames": frame_count,
                "metadata": metadata,
            }
        )
    if not cases:
        raise FileNotFoundError(f"no gt_*.npz cases under {gt_root}")
    return cases


def select_random_cases(
    gt_root: Path, seed: int
) -> list[dict[str, Any]]:
    cases = discover_cases(gt_root)
    bins = {
        "up": [item for item in cases if item["alpha_deg"] >= 3.0],
        "middle": [item for item in cases if abs(item["alpha_deg"]) <= 0.5],
        "down": [item for item in cases if item["alpha_deg"] <= -3.0],
    }
    if any(not options for options in bins.values()):
        raise ValueError("GT archive does not contain up/middle/down angle bins")
    rng = np.random.default_rng(seed)
    selected = []
    for orientation in ("up", "middle", "down"):
        options = bins[orientation]
        item = dict(options[int(rng.integers(0, len(options)))])
        # Skip the startup transient and draw a reproducible developed-flow frame.
        low = min(20, item["frames"] - 1)
        item["frame"] = int(rng.integers(low, item["frames"]))
        item["orientation"] = orientation
        selected.append(item)
    return selected


def load_case(item: dict[str, Any]) -> dict[str, Any]:
    with np.load(item["path"], allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
        frame = int(item["frame"])
        actual_u = archive["u"][frame].astype(np.float32)
        actual_v = archive["v"][frame].astype(np.float32)
        actual_omega = archive["vorticity"][frame].astype(np.float32)
        mean_u = np.mean(archive["u"], axis=0, dtype=np.float64).astype(np.float32)
        mean_v = np.mean(archive["v"], axis=0, dtype=np.float64).astype(np.float32)
        mean_omega = np.mean(
            archive["vorticity"], axis=0, dtype=np.float64
        ).astype(np.float32)
    return {
        "x": x,
        "y": y,
        "solid": solid,
        "actual_u": actual_u,
        "actual_v": actual_v,
        "actual_omega": actual_omega,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "mean_omega": mean_omega,
    }


def reconstruct_learned(
    model: Any,
    cfg: Any,
    state: dict[str, Any],
    points: np.ndarray,
    subset: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    beta: float,
    alpha: float,
    field_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    transform = state["transform"]
    psi, gx, gy, raw_u, raw_v, omega = raw_fields(
        model, cfg, points[subset], beta, alpha
    )
    basis_u = raw_u @ transform
    basis_v = raw_v @ transform
    coefficients, rank, residual = velocity_projection(
        basis_u, basis_v, target_u[subset], target_v[subset]
    )
    raw_coefficients = transform @ coefficients
    del psi, gx, gy, raw_u, raw_v, omega, basis_u, basis_v
    fluctuation_u = np.empty(len(points), dtype=np.float32)
    fluctuation_v = np.empty(len(points), dtype=np.float32)
    fluctuation_omega = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        psi, gx, gy, raw_u, raw_v, omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha
        )
        fluctuation_u[start:stop] = raw_u @ raw_coefficients
        fluctuation_v[start:stop] = raw_v @ raw_coefficients
        fluctuation_omega[start:stop] = omega @ raw_coefficients
        del psi, gx, gy, raw_u, raw_v, omega
    cp.get_default_memory_pool().free_all_blocks()
    return (
        fluctuation_u,
        fluctuation_v,
        fluctuation_omega,
        {
            "projection_rank": rank,
            "projection_relative_residual": residual,
            "coefficients": coefficients,
        },
    )


def reconstruct_laplacian(
    basis: dict[str, Any],
    points: np.ndarray,
    subset: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    field_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    interpolators = geometry_laplacian_interpolators(basis)
    psi, basis_u, basis_v, omega = geometry_laplacian_fields(
        points[subset], interpolators
    )
    coefficients, rank, residual = velocity_projection(
        basis_u, basis_v, target_u[subset], target_v[subset]
    )
    del psi, basis_u, basis_v, omega
    fluctuation_u = np.empty(len(points), dtype=np.float32)
    fluctuation_v = np.empty(len(points), dtype=np.float32)
    fluctuation_omega = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        psi, basis_u, basis_v, omega = geometry_laplacian_fields(
            points[start:stop], interpolators
        )
        fluctuation_u[start:stop] = basis_u @ coefficients
        fluctuation_v[start:stop] = basis_v @ coefficients
        fluctuation_omega[start:stop] = omega @ coefficients
        del psi, basis_u, basis_v, omega
    return (
        fluctuation_u,
        fluctuation_v,
        fluctuation_omega,
        {
            "projection_rank": rank,
            "projection_relative_residual": residual,
            "coefficients": coefficients,
            "eigensolve": basis["diagnostics"],
        },
    )


def reconstruction_metrics(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    estimate_omega: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    target_omega: np.ndarray,
    wake: np.ndarray,
) -> dict[str, float]:
    return {
        "velocity_error": relative_velocity(
            actual_u, actual_v, estimate_u, estimate_v
        ),
        "vorticity_error": relative_scalar(actual_omega, estimate_omega),
        "fluctuation_velocity_error": relative_velocity(
            target_u,
            target_v,
            estimate_u - mean_u,
            estimate_v - mean_v,
        ),
        "fluctuation_vorticity_error": relative_scalar(
            target_omega, estimate_omega - mean_omega
        ),
        "wake_velocity_error": relative_velocity(
            actual_u, actual_v, estimate_u, estimate_v, wake
        ),
        "wake_vorticity_error": relative_scalar(
            actual_omega, estimate_omega, wake
        ),
    }


def reconstruction_figure(
    case: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    stride: int,
) -> plt.Figure:
    x = case["x"]
    y = case["y"]
    solid = case["solid"][::stride, ::stride]
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    display = {
        method: {
            name: fields[name][::stride, ::stride]
            for name in ("u", "v", "omega")
        }
        for method, fields in arrays.items()
    }
    speeds = {
        method: np.sqrt(fields["u"] ** 2 + fields["v"] ** 2)
        for method, fields in display.items()
    }
    omega_limit = max(
        float(np.nanpercentile(np.abs(field[~solid]), 99.4))
        for field in (fields["omega"] for fields in display.values())
    )
    speed_min = min(
        float(np.nanpercentile(field[~solid], 0.5)) for field in speeds.values()
    )
    speed_max = max(
        float(np.nanpercentile(field[~solid], 99.5)) for field in speeds.values()
    )
    columns = ("ground_truth",) + METHODS
    figure, axes = plt.subplots(2, 4, figsize=(16.0, 6.25), constrained_layout=False)
    speed_image = None
    omega_image = None
    for column, method in enumerate(columns):
        title = "Ground truth" if method == "ground_truth" else METHOD_LABELS[method]
        speed_image = axes[0, column].imshow(
            np.ma.masked_where(solid, speeds[method]),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=speed_min,
            vmax=speed_max,
            interpolation="bilinear",
        )
        omega_image = axes[1, column].imshow(
            np.ma.masked_where(solid, display[method]["omega"]),
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-omega_limit,
            vmax=omega_limit,
            interpolation="bilinear",
        )
        for axis in axes[:, column]:
            add_solid(axis, solid, extent)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
        if method == "ground_truth":
            axes[0, column].set_title(title, fontsize=10.2, color=CHARCOAL)
        else:
            axes[0, column].set_title(
                f"{title}\nvelocity L2={100.0 * metrics[method]['velocity_error']:.2f}%",
                fontsize=9.4,
                color=METHOD_COLORS[method],
            )
            axes[1, column].set_title(
                f"vorticity L2={100.0 * metrics[method]['vorticity_error']:.2f}%",
                fontsize=8.8,
                color=METHOD_COLORS[method],
                pad=1.5,
            )
    axes[0, 0].set_ylabel("Speed", fontsize=10.0, weight="bold")
    axes[1, 0].set_ylabel("Vorticity", fontsize=10.0, weight="bold")
    assert speed_image is not None and omega_image is not None
    figure.colorbar(
        speed_image, ax=axes[0, :].tolist(), orientation="horizontal", fraction=0.045, pad=0.055
    ).set_label("speed", fontsize=8.5)
    figure.colorbar(
        omega_image, ax=axes[1, :].tolist(), orientation="horizontal", fraction=0.045, pad=0.055
    ).set_label("vorticity", fontsize=8.5)
    figure.suptitle(
        f"{case['orientation'].capitalize()} random case | "
        rf"$\beta={case['beta']:.6f}$, $\alpha={case['alpha_deg']:.5f}^\circ$, frame {case['frame']} | K=64",
        fontsize=12.0,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.995,
        top=0.87,
        bottom=0.14,
        wspace=0.06,
        hspace=0.18,
    )
    return figure


def evaluate_case(
    item: dict[str, Any],
    original_model: Any,
    original_cfg: Any,
    winner_model: Any,
    winner_cfg: Any,
    peeling_points: int,
    projection_points: int,
    field_batch: int,
    laplacian_stride: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    loaded = load_case(item)
    x, y, solid = loaded["x"], loaded["y"], loaded["solid"]
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    actual_u = loaded["actual_u"][fluid]
    actual_v = loaded["actual_v"][fluid]
    actual_omega = loaded["actual_omega"][fluid]
    mean_u = loaded["mean_u"][fluid]
    mean_v = loaded["mean_v"][fluid]
    mean_omega = loaded["mean_omega"][fluid]
    target_u = actual_u - mean_u
    target_v = actual_v - mean_v
    target_omega = actual_omega - mean_omega
    subset = np.linspace(
        0, len(points) - 1, min(projection_points, len(points)), dtype=np.int64
    )
    beta, alpha = float(item["beta"]), float(item["alpha_deg"])
    states = {
        "original_e2000": learned_geometry_state(
            original_model, original_cfg, beta, alpha, peeling_points, 2_310_001
        ),
        "winner_e2000": learned_geometry_state(
            winner_model, winner_cfg, beta, alpha, peeling_points, 2_310_001
        ),
    }
    fluctuations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for method, model, cfg in (
        ("original_e2000", original_model, original_cfg),
        ("winner_e2000", winner_model, winner_cfg),
    ):
        fu, fv, fo, projection = reconstruct_learned(
            model,
            cfg,
            states[method],
            points,
            subset,
            target_u,
            target_v,
            beta,
            alpha,
            field_batch,
        )
        fluctuations[method] = (fu, fv, fo)
        diagnostics[method] = {
            **projection,
            "peeled_basis": states[method]["diagnostics"],
            "first8_rayleigh": states[method]["rayleigh"][:8].tolist(),
        }
    lap_basis = geometry_laplacian(
        x,
        y,
        solid,
        int(original_cfg.modes),
        laplacian_stride,
        seed + 2_410_001 + int(item["frame"]),
    )
    fu, fv, fo, projection = reconstruct_laplacian(
        lap_basis,
        points,
        subset,
        target_u,
        target_v,
        field_batch,
    )
    fluctuations["laplacian"] = (fu, fv, fo)
    diagnostics["laplacian"] = {
        **projection,
        "first8_eigenvalues": lap_basis["eigenvalues"][:8].tolist(),
    }
    xb, yb = body_numpy(original_cfg, points, alpha)
    wake = (
        (xb >= original_cfg.chord)
        & (xb <= 4.0 * original_cfg.chord)
        & (np.abs(yb) <= 0.75 * original_cfg.chord)
    )
    arrays: dict[str, dict[str, np.ndarray]] = {
        "ground_truth": {
            "u": loaded["actual_u"],
            "v": loaded["actual_v"],
            "omega": loaded["actual_omega"],
        }
    }
    metrics: dict[str, dict[str, float]] = {}
    for method in METHODS:
        fu, fv, fo = fluctuations[method]
        estimate_u = mean_u + fu
        estimate_v = mean_v + fv
        estimate_omega = mean_omega + fo
        metrics[method] = reconstruction_metrics(
            actual_u,
            actual_v,
            actual_omega,
            mean_u,
            mean_v,
            mean_omega,
            estimate_u,
            estimate_v,
            estimate_omega,
            target_u,
            target_v,
            target_omega,
            wake,
        )
        arrays[method] = {
            "u": as_grid(estimate_u, fluid),
            "v": as_grid(estimate_v, fluid),
            "omega": as_grid(estimate_omega, fluid),
        }
    loaded["orientation"] = item["orientation"]
    loaded["beta"] = beta
    loaded["alpha_deg"] = alpha
    loaded["frame"] = int(item["frame"])
    result = {
        "orientation": item["orientation"],
        "gt": str(Path(item["path"]).resolve()),
        "beta": beta,
        "alpha_deg": alpha,
        "frame": int(item["frame"]),
        "reynolds": float(item["metadata"]["solver_config"]["reynolds"]),
        "fluid_points": int(len(points)),
        "projection_points": int(len(subset)),
        "methods": {
            method: {
                **metrics[method],
                **{
                    key: value
                    for key, value in diagnostics[method].items()
                    if key != "coefficients"
                },
            }
            for method in METHODS
        },
    }
    for method in METHODS:
        arrays[method]["coefficients"] = diagnostics[method]["coefficients"]
    del points, xgrid, ygrid, fluctuations, states, lap_basis
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return result, {"case": loaded, **arrays}


def save_case_fields(
    output: Path,
    arrays: dict[str, dict[str, np.ndarray]],
    stride: int,
) -> None:
    case = arrays["case"]
    payload: dict[str, Any] = {
        "x": case["x"][::stride],
        "y": case["y"][::stride],
        "solid": case["solid"][::stride, ::stride],
        "beta": np.asarray(case["beta"], dtype=np.float64),
        "alpha_deg": np.asarray(case["alpha_deg"], dtype=np.float64),
        "frame": np.asarray(case["frame"], dtype=np.int32),
    }
    for method in ("ground_truth",) + METHODS:
        for field in ("u", "v", "omega"):
            payload[f"{method}_{field}"] = arrays[method][field][::stride, ::stride]
        if method != "ground_truth":
            payload[f"{method}_coefficients"] = arrays[method]["coefficients"]
    np.savez_compressed(output, **payload)


def write_metrics_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = (
        "velocity_error",
        "vorticity_error",
        "fluctuation_velocity_error",
        "fluctuation_vorticity_error",
        "wake_velocity_error",
        "wake_vorticity_error",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("orientation", "gt", "beta", "alpha_deg", "frame", "method")
            + fields
            + ("projection_rank", "projection_relative_residual")
        )
        for result in results:
            for method in METHODS:
                values = result["methods"][method]
                writer.writerow(
                    (
                        result["orientation"],
                        result["gt"],
                        result["beta"],
                        result["alpha_deg"],
                        result["frame"],
                        method,
                    )
                    + tuple(values[field] for field in fields)
                    + (
                        values["projection_rank"],
                        values["projection_relative_residual"],
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--winner-checkpoint", type=Path, required=True)
    parser.add_argument("--mode-gt", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument("--laplacian-stride", type=int, default=2)
    parser.add_argument("--archive-stride", type=int, default=2)
    parser.add_argument(
        "--modes-only",
        action="store_true",
        help="regenerate only the 3x8 mode products and stop",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    started_at = now()

    original_metadata, original_cfg, original_model = load_model(
        args.original_checkpoint
    )
    winner_metadata, winner_cfg, winner_model = load_model(args.winner_checkpoint)
    ensure_matching_problem(original_cfg, winner_cfg)
    if int(original_metadata["epoch"]) != 2000 or int(winner_metadata["epoch"]) != 2000:
        raise ValueError("both learned checkpoints must be exactly epoch 2000")

    print("[1/8] 3x8 mode comparison: load unseen geometry", flush=True)
    with np.load(args.mode_gt, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
        metadata = json.loads(str(archive["metadata"]))
    beta = float(metadata["beta"])
    alpha = float(metadata["alpha_deg"])
    original_state = learned_geometry_state(
        original_model,
        original_cfg,
        beta,
        alpha,
        args.peeling_points,
        2_300_001,
    )
    winner_state = learned_geometry_state(
        winner_model,
        winner_cfg,
        beta,
        alpha,
        args.peeling_points,
        2_300_001,
    )
    stride = int(args.laplacian_stride)
    x_plot, y_plot = x[::stride], y[::stride]
    solid_plot = solid[::stride, ::stride]
    xgrid, ygrid = np.meshgrid(x_plot, y_plot)
    fluid_plot = ~solid_plot
    plot_points = np.column_stack((xgrid[fluid_plot], ygrid[fluid_plot])).astype(
        np.float32
    )
    print("[2/8] evaluate Original/Winner first eight modes", flush=True)
    original_modes = first_modes(
        original_model,
        original_cfg,
        original_state["transform"],
        plot_points,
        beta,
        alpha,
        8,
        args.field_batch,
    )
    winner_modes = first_modes(
        winner_model,
        winner_cfg,
        winner_state["transform"],
        plot_points,
        beta,
        alpha,
        8,
        args.field_batch,
    )
    original_modes, original_signs = canonicalize_display_sign(original_modes)
    winner_modes, winner_signs = canonicalize_display_sign(winner_modes)
    print("[3/8] solve same-domain geometry-aware FD Laplacian", flush=True)
    mode_laplacian = geometry_laplacian(
        x,
        y,
        solid,
        int(original_cfg.modes),
        stride,
        int(args.random_seed) + 2_400_001,
    )
    lap_modes = np.moveaxis(mode_laplacian["psi"][:, :, :8], -1, 0)
    lap_modes, lap_signs = canonicalize_display_sign(lap_modes)
    mode_grids = {
        "original_e2000": np.stack(
            [as_grid(field, fluid_plot) for field in original_modes]
        ),
        "winner_e2000": np.stack(
            [as_grid(field, fluid_plot) for field in winner_modes]
        ),
        "laplacian": lap_modes,
    }
    spectra = {
        "original_e2000": original_state["rayleigh"][:8],
        "winner_e2000": winner_state["rayleigh"][:8],
        "laplacian": mode_laplacian["eigenvalues"][:8],
    }
    mode_stem = args.output_dir / "modes_3x8"
    plot_modes_3x8(
        mode_grids,
        spectra,
        solid_plot,
        (float(x[0]), float(x[-1]), float(y[0]), float(y[-1])),
        beta,
        alpha,
        mode_stem,
    )
    with (args.output_dir / "modes_first8_values.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("mode", "original_e2000_rayleigh", "winner_e2000_rayleigh", "fd_laplacian_eigenvalue")
        )
        for index in range(8):
            writer.writerow(
                (index + 1, spectra["original_e2000"][index], spectra["winner_e2000"][index], spectra["laplacian"][index])
            )
    mode_summary = {
        "case": {
            "gt": str(args.mode_gt.resolve()),
            "beta": beta,
            "alpha_deg": alpha,
            "grid_stride": stride,
        },
        "original_e2000": {
            "first8_rayleigh": spectra["original_e2000"].tolist(),
            "display_signs": original_signs.tolist(),
            "peeled_basis": original_state["diagnostics"],
        },
        "winner_e2000": {
            "first8_rayleigh": spectra["winner_e2000"].tolist(),
            "display_signs": winner_signs.tolist(),
            "peeled_basis": winner_state["diagnostics"],
        },
        "laplacian": {
            "first8_eigenvalues": spectra["laplacian"].tolist(),
            "display_signs": lap_signs.tolist(),
            "eigensolve": mode_laplacian["diagnostics"],
        },
        "figure_png": str(mode_stem.with_suffix(".png").resolve()),
        "figure_pdf": str(mode_stem.with_suffix(".pdf").resolve()),
    }
    write_json(args.output_dir / "modes_3x8_summary.json", mode_summary)
    del original_modes, winner_modes, lap_modes, mode_grids, mode_laplacian
    del original_state, winner_state, plot_points, xgrid, ygrid
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    if args.modes_only:
        print("[4/4] modes-only regeneration complete", flush=True)
        return

    print("[4/8] reproducibly select up/middle/down NS cases", flush=True)
    selected = select_random_cases(args.gt_root, args.random_seed)
    selection_json = [
        {
            "orientation": item["orientation"],
            "gt": str(Path(item["path"]).resolve()),
            "beta": item["beta"],
            "alpha_deg": item["alpha_deg"],
            "frame": item["frame"],
        }
        for item in selected
    ]
    write_json(
        args.output_dir / "random_case_selection.json",
        {"seed": args.random_seed, "cases": selection_json},
    )

    reconstruction_pdf = args.output_dir / "three_random_reconstructions.pdf"
    results: list[dict[str, Any]] = []
    print("[5/8] reconstruct three cases with all three K64 bases", flush=True)
    with PdfPages(reconstruction_pdf) as pdf:
        for case_index, item in enumerate(selected, start=1):
            print(
                f"  case {case_index}/3 {item['orientation']} "
                f"beta={item['beta']:.6f} alpha={item['alpha_deg']:.5f} frame={item['frame']}",
                flush=True,
            )
            result, arrays = evaluate_case(
                item,
                original_model,
                original_cfg,
                winner_model,
                winner_cfg,
                args.peeling_points,
                args.projection_points,
                args.field_batch,
                args.laplacian_stride,
                args.random_seed + 1000 * case_index,
            )
            results.append(result)
            figure = reconstruction_figure(
                arrays["case"],
                {key: arrays[key] for key in ("ground_truth",) + METHODS},
                {method: result["methods"][method] for method in METHODS},
                args.archive_stride,
            )
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.04)
            page_png = args.output_dir / f"random_{case_index}_{item['orientation']}.png"
            figure.savefig(page_png, dpi=260, bbox_inches="tight", pad_inches=0.04)
            plt.close(figure)
            save_case_fields(
                args.output_dir / f"random_{case_index}_{item['orientation']}_fields_stride{args.archive_stride}.npz",
                arrays,
                args.archive_stride,
            )
            del arrays
            gc.collect()

    print("[6/8] write reconstruction metrics and training comparison", flush=True)
    metrics_csv = args.output_dir / "three_random_reconstruction_metrics.csv"
    write_metrics_csv(metrics_csv, results)
    training = {
        "original_e2000": {
            "checkpoint": str(args.original_checkpoint.resolve()),
            "checkpoint_epoch": int(original_metadata["epoch"]),
            "config": config_summary(original_cfg),
            "final": final_training_metrics(args.original_checkpoint),
        },
        "winner_e2000": {
            "checkpoint": str(args.winner_checkpoint.resolve()),
            "checkpoint_epoch": int(winner_metadata["epoch"]),
            "config": config_summary(winner_cfg),
            "final": final_training_metrics(args.winner_checkpoint),
        },
    }
    with (args.output_dir / "training_e2000_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "validation_mean",
                "validation_p95",
                "validation_max",
                "peeled_rank",
                "orthogonality_max_abs",
                "learning_rate",
                "training_elapsed_seconds",
            )
        )
        for method in ("original_e2000", "winner_e2000"):
            final = training[method]["final"]
            writer.writerow(
                (
                    method,
                    final["validation_mean"],
                    final["validation_p95"],
                    final["validation_max"],
                    final["peeled_rank"],
                    final["orthogonality_max_abs"],
                    final["learning_rate"],
                    final["training_elapsed_seconds"],
                )
            )

    print("[7/8] finalize protocol and completion marker", flush=True)
    summary = {
        "complete": True,
        "started_at": started_at,
        "completed_at": now(),
        "evaluation_wall_seconds": (time.perf_counter_ns() - started) / 1.0e9,
        "random_seed": args.random_seed,
        "training": training,
        "mode_comparison": mode_summary,
        "random_case_selection": selection_json,
        "reconstruction_protocol": {
            "modes": 64,
            "peeling_points": args.peeling_points,
            "projection_points": args.projection_points,
            "coefficient_fit": "velocity L2 of 41-frame temporal fluctuation",
            "same_projection_samples_for_all_three_methods": True,
            "same_temporal_mean_for_all_three_methods": True,
            "learned_dynamic_processing": "geometry-Rayleigh causal sorting plus two-pass MGS",
            "laplacian": "geometry-aware five-point FD on the same airfoil fluid domain",
            "dirichlet_boundaries": ["airfoil", "inlet", "ymin", "ymax"],
            "natural_boundaries": ["outlet"],
            "d4": False,
            "analytic_eigenmodes": False,
            "degenerate_block_direction_alignment": False,
            "ritz_rotation": False,
            "raw_rank_reported": False,
        },
        "random_reconstructions": results,
        "outputs": {
            "modes_png": str(mode_stem.with_suffix(".png").resolve()),
            "modes_pdf": str(mode_stem.with_suffix(".pdf").resolve()),
            "modes_values_csv": str((args.output_dir / "modes_first8_values.csv").resolve()),
            "reconstructions_pdf": str(reconstruction_pdf.resolve()),
            "metrics_csv": str(metrics_csv.resolve()),
            "training_csv": str((args.output_dir / "training_e2000_comparison.csv").resolve()),
        },
    }
    write_json(args.output_dir / "comparison_complete.json", summary)
    print("[8/8] complete", flush=True)
    print(
        json.dumps(
            {
                "training": {key: value["final"] for key, value in training.items()},
                "random_case_selection": selection_json,
                "reconstruction_pdf": str(reconstruction_pdf.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
