#!/usr/bin/env python3
"""Validate the strong Laplacian relation after velocity-space Ritz rotation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
)


def load_model(checkpoint: Path) -> tuple[Any, HardcodedModel]:
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    return cfg, HardcodedModel(cfg, arrays)


def evaluate_gauge_mean(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    point_batch: int,
) -> cp.ndarray:
    total = cp.zeros(int(cfg.modes), dtype=cp.float64)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        raw_value = cp.empty((stop - start, int(cfg.modes)), dtype=cp.float32)
        for mode, field, _cache in model.forward_stream_jet1(
            cfg, device_points, 0.0, 0.0, False
        ):
            raw_value[:, int(mode.mode_index)] = field.value[:, 0]
        total += cp.sum(raw_value, axis=0, dtype=cp.float64)
        print(f"[gauge] points {stop}/{len(points)}", flush=True)
    return (total / len(points)).astype(cp.float32)


def strong_residuals(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    interior_mask: np.ndarray,
    rotation: np.ndarray,
    eigenvalues: np.ndarray,
    point_batch: int,
) -> dict[str, np.ndarray]:
    gauge_mean = evaluate_gauge_mean(model, cfg, points, point_batch)
    gpu_rotation = cp.asarray(rotation, dtype=cp.float32)
    gpu_eigenvalues = cp.asarray(eigenvalues, dtype=cp.float32)
    modes = int(cfg.modes)
    sums = {
        name: cp.zeros(modes, dtype=cp.float64)
        for name in ("psi2", "omega2", "psiomega", "residual2")
    }
    interior_sums = {
        name: cp.zeros(modes, dtype=cp.float64)
        for name in ("psi2", "residual2")
    }
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        raw_value = cp.empty((stop - start, modes), dtype=cp.float32)
        raw_omega = cp.empty_like(raw_value)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, device_points, 0.0, 0.0, False
        ):
            index = int(mode.mode_index)
            raw_value[:, index] = field.value[:, 0]
            raw_omega[:, index] = -(
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        psi = (raw_value - gauge_mean) @ gpu_rotation
        omega = raw_omega @ gpu_rotation
        residual = omega - psi * gpu_eigenvalues[None, :]
        sums["psi2"] += cp.sum(psi * psi, axis=0, dtype=cp.float64)
        sums["omega2"] += cp.sum(omega * omega, axis=0, dtype=cp.float64)
        sums["psiomega"] += cp.sum(psi * omega, axis=0, dtype=cp.float64)
        sums["residual2"] += cp.sum(residual * residual, axis=0, dtype=cp.float64)
        local_mask = cp.asarray(interior_mask[start:stop])
        if bool(cp.any(local_mask)):
            interior_psi = psi[local_mask]
            interior_residual = residual[local_mask]
            interior_sums["psi2"] += cp.sum(
                interior_psi * interior_psi, axis=0, dtype=cp.float64
            )
            interior_sums["residual2"] += cp.sum(
                interior_residual * interior_residual, axis=0, dtype=cp.float64
            )
        print(f"[strong residual] points {stop}/{len(points)}", flush=True)

    host = {name: cp.asnumpy(value) for name, value in sums.items()}
    host_interior = {
        name: cp.asnumpy(value) for name, value in interior_sums.items()
    }
    eps = np.finfo(np.float64).tiny
    denominator = np.maximum(np.abs(eigenvalues) * np.sqrt(host["psi2"]), eps)
    denominator_interior = np.maximum(
        np.abs(eigenvalues) * np.sqrt(host_interior["psi2"]), eps
    )
    best_lambda = host["psiomega"] / np.maximum(host["psi2"], eps)
    best_residual2 = (
        host["omega2"]
        - 2.0 * best_lambda * host["psiomega"]
        + best_lambda * best_lambda * host["psi2"]
    )
    best_residual2 = np.maximum(best_residual2, 0.0)
    return {
        "ritz_lambda": eigenvalues,
        "best_strong_lambda": best_lambda,
        "relative_strong_residual": np.sqrt(host["residual2"]) / denominator,
        "relative_strong_residual_interior": (
            np.sqrt(host_interior["residual2"]) / denominator_interior
        ),
        "relative_best_fit_residual": (
            np.sqrt(best_residual2)
            / np.maximum(np.abs(best_lambda) * np.sqrt(host["psi2"]), eps)
        ),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def write_outputs(output_dir: Path, width: int, result: dict[str, np.ndarray], grid: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    residual = result["relative_strong_residual"]
    interior = result["relative_strong_residual_interior"]
    best = result["relative_best_fit_residual"]
    payload = {
        "width": width,
        "validation_grid": f"{grid}x{grid} cell-centered independent grid",
        "definition": "||(-Delta psi_tilde)-lambda_Ritz psi_tilde||_2 / (|lambda_Ritz| ||psi_tilde||_2)",
        "all_1024": summarize(residual),
        "first_16": summarize(residual[:16]),
        "first_16_excluding_four_edge_cells": summarize(interior[:16]),
        "all_1024_best_scalar_lambda": summarize(best),
        "first_16_best_scalar_lambda": summarize(best[:16]),
        "selected_modes": [],
    }
    selected = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
    for rank in selected:
        index = rank - 1
        payload["selected_modes"].append(
            {
                "rank": rank,
                "ritz_lambda": float(result["ritz_lambda"][index]),
                "best_strong_lambda": float(result["best_strong_lambda"][index]),
                "relative_strong_residual": float(residual[index]),
                "relative_strong_residual_interior": float(interior[index]),
                "relative_best_fit_residual": float(best[index]),
            }
        )
    (output_dir / "velocity_ritz_eigenrelation_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "velocity_ritz_eigenrelation_all_modes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "rank",
                "ritz_lambda",
                "best_strong_lambda",
                "relative_strong_residual",
                "relative_strong_residual_interior",
                "relative_best_fit_residual",
            ]
        )
        for index in range(len(residual)):
            writer.writerow(
                [
                    index + 1,
                    result["ritz_lambda"][index],
                    result["best_strong_lambda"][index],
                    residual[index],
                    interior[index],
                    best[index],
                ]
            )

    ranks = np.arange(1, len(residual) + 1)
    plt.style.use("default")
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    axis.semilogy(ranks, residual, color="#B2182B", lw=1.1, label="full domain")
    axis.semilogy(ranks, interior, color="#2166AC", lw=1.0, label="exclude 4 edge cells")
    axis.axhline(0.1, color="#555555", ls="--", lw=0.8, label="10% residual")
    axis.set_xlabel("Ritz mode rank")
    axis.set_ylabel("Normalized strong eigenrelation residual")
    axis.set_title(f"Velocity-space Ritz eigenrelation | width {width}")
    axis.grid(True, which="both", color="#D9D9D9", lw=0.55)
    axis.legend(frameon=False, ncol=3, fontsize=8)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"velocity_ritz_eigenrelation.{suffix}", dpi=240)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--point-batch", type=int, default=2048)
    args = parser.parse_args()

    cfg, model = load_model(args.checkpoint.resolve())
    with np.load(args.rollout_npz.resolve(), allow_pickle=False) as archive:
        rotation = archive["raw_to_ritz"].astype(np.float64)
        eigenvalues = archive["ritz_eigenvalues"].astype(np.float64)
    xmin, xmax = float(cfg.xmin), float(cfg.xmax)
    ymin, ymax = float(cfg.ymin), float(cfg.ymax)
    spacing_x = (xmax - xmin) / args.grid
    spacing_y = (ymax - ymin) / args.grid
    axis_x = xmin + (np.arange(args.grid, dtype=np.float64) + 0.5) * spacing_x
    axis_y = ymin + (np.arange(args.grid, dtype=np.float64) + 0.5) * spacing_y
    grid_x, grid_y = np.meshgrid(axis_x, axis_y, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    index_x, index_y = np.meshgrid(
        np.arange(args.grid), np.arange(args.grid), indexing="xy"
    )
    interior_mask = (
        (index_x >= 4)
        & (index_x < args.grid - 4)
        & (index_y >= 4)
        & (index_y < args.grid - 4)
    ).ravel()
    result = strong_residuals(
        model,
        cfg,
        points,
        interior_mask,
        rotation,
        eigenvalues,
        args.point_batch,
    )
    write_outputs(args.output_dir.resolve(), args.width, result, args.grid)
    print(json.dumps({"first_16": summarize(result["relative_strong_residual"][:16]), "all": summarize(result["relative_strong_residual"])}, indent=2))


if __name__ == "__main__":
    main()
