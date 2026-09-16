#!/usr/bin/env python3
"""Strong-form validation of trained, potential-Ritz, and velocity-Ritz modes."""

from __future__ import annotations

import argparse
import csv
import json
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
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import fluid_points  # noqa: E402
from eigenfluid.gen.vortex_merger2d.witt_rom import (  # noqa: E402
    generalized_symmetric_eig,
)
from visualize_free_k1024_vorticity_modes import (  # noqa: E402
    causal_peeling_transform,
    load_model,
    raw_potential_jet1,
)


def evaluate_raw_omega(
    model: Any, cfg: Any, points: np.ndarray, point_batch: int
) -> cp.ndarray:
    result = cp.empty((len(points), int(cfg.modes)), dtype=cp.float32)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, device_points, 0.0, 0.0, False
        ):
            result[start:stop, int(mode.mode_index)] = -(
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        print(f"[Ritz omega] points {stop}/{len(points)}", flush=True)
    return result


def validation_gauge_mean(
    model: Any, cfg: Any, points: np.ndarray, point_batch: int
) -> cp.ndarray:
    total = cp.zeros(int(cfg.modes), dtype=cp.float64)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        raw = cp.empty((stop - start, int(cfg.modes)), dtype=cp.float32)
        for mode, field, _cache in model.forward_stream_jet1(
            cfg, device_points, 0.0, 0.0, False
        ):
            raw[:, int(mode.mode_index)] = field.value[:, 0]
        total += cp.sum(raw, axis=0, dtype=cp.float64)
    return (total / len(points)).astype(cp.float32)


def validate_transforms(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    interior_mask: np.ndarray,
    transforms: dict[str, np.ndarray],
    eigenvalues: dict[str, np.ndarray],
    point_batch: int,
) -> dict[str, dict[str, np.ndarray]]:
    gauge = validation_gauge_mean(model, cfg, points, point_batch)
    gpu_transforms = {
        name: cp.asarray(transform, dtype=cp.float32)
        for name, transform in transforms.items()
    }
    gpu_eigenvalues = {
        name: cp.asarray(values, dtype=cp.float32)
        for name, values in eigenvalues.items()
    }
    accumulators = {
        name: {
            key: cp.zeros(int(cfg.modes), dtype=cp.float64)
            for key in ("psi2", "omega2", "psiomega", "r2", "psi2_in", "r2_in")
        }
        for name in transforms
    }
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        raw_value = cp.empty((stop - start, int(cfg.modes)), dtype=cp.float32)
        raw_omega = cp.empty_like(raw_value)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, device_points, 0.0, 0.0, False
        ):
            index = int(mode.mode_index)
            raw_value[:, index] = field.value[:, 0]
            raw_omega[:, index] = -(
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        centered = raw_value - gauge
        local_mask = cp.asarray(interior_mask[start:stop])
        for name, transform in gpu_transforms.items():
            psi = centered @ transform
            omega = raw_omega @ transform
            residual = omega - psi * gpu_eigenvalues[name][None, :]
            acc = accumulators[name]
            acc["psi2"] += cp.sum(psi * psi, axis=0, dtype=cp.float64)
            acc["omega2"] += cp.sum(omega * omega, axis=0, dtype=cp.float64)
            acc["psiomega"] += cp.sum(psi * omega, axis=0, dtype=cp.float64)
            acc["r2"] += cp.sum(residual * residual, axis=0, dtype=cp.float64)
            acc["psi2_in"] += cp.sum(
                psi[local_mask] * psi[local_mask], axis=0, dtype=cp.float64
            )
            acc["r2_in"] += cp.sum(
                residual[local_mask] * residual[local_mask],
                axis=0,
                dtype=cp.float64,
            )
        print(f"[validation Jet2] points {stop}/{len(points)}", flush=True)

    result: dict[str, dict[str, np.ndarray]] = {}
    tiny = np.finfo(np.float64).tiny
    for name, values in accumulators.items():
        host = {key: cp.asnumpy(value) for key, value in values.items()}
        lam = eigenvalues[name]
        best_lam = host["psiomega"] / np.maximum(host["psi2"], tiny)
        best_r2 = np.maximum(
            host["omega2"]
            - 2.0 * best_lam * host["psiomega"]
            + best_lam**2 * host["psi2"],
            0.0,
        )
        result[name] = {
            "lambda": lam,
            "best_lambda": best_lam,
            "residual": np.sqrt(host["r2"])
            / np.maximum(np.abs(lam) * np.sqrt(host["psi2"]), tiny),
            "interior_residual": np.sqrt(host["r2_in"])
            / np.maximum(np.abs(lam) * np.sqrt(host["psi2_in"]), tiny),
            "best_fit_residual": np.sqrt(best_r2)
            / np.maximum(np.abs(best_lam) * np.sqrt(host["psi2"]), tiny),
        }
    return result


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def write_outputs(
    output_dir: Path,
    result: dict[str, dict[str, np.ndarray]],
    grid_size: int,
    quadrature_points: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = len(next(iter(result.values()))["residual"])
    summary: dict[str, Any] = {
        "modes": modes,
        "ritz_quadrature_points": quadrature_points,
        "validation_grid": f"independent {grid_size}x{grid_size} cell-centered grid",
        "residual_definition": "||(-Delta psi)-lambda psi||_2/(|lambda| ||psi||_2)",
        "methods": {},
    }
    for name, values in result.items():
        residual = values["residual"]
        summary["methods"][name] = {
            "full_domain": stats(residual),
            "excluding_four_edge_cells": stats(values["interior_residual"]),
            "best_scalar_lambda": stats(values["best_fit_residual"]),
            "count_below_1pct": int(np.sum(residual < 0.01)),
            "count_below_5pct": int(np.sum(residual < 0.05)),
            "count_below_10pct": int(np.sum(residual < 0.10)),
        }
    (output_dir / "laplacian_eigenrelation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "laplacian_eigenrelation_all_modes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        header = ["rank"]
        for name in result:
            header += [
                f"{name}_lambda",
                f"{name}_strong_residual",
                f"{name}_interior_residual",
                f"{name}_best_lambda",
                f"{name}_best_fit_residual",
            ]
        writer.writerow(header)
        for index in range(modes):
            row: list[float | int] = [index + 1]
            for values in result.values():
                row += [
                    values["lambda"][index],
                    values["residual"][index],
                    values["interior_residual"][index],
                    values["best_lambda"][index],
                    values["best_fit_residual"][index],
                ]
            writer.writerow(row)

    palette = {
        "trained_peeled": "#555555",
        "potential_ritz": "#2166AC",
        "velocity_ritz": "#B2182B",
    }
    labels = {
        "trained_peeled": "trained + causal peeling",
        "potential_ritz": "potential-space Ritz",
        "velocity_ritz": "velocity-space Ritz",
    }
    ranks = np.arange(1, modes + 1)
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for name, values in result.items():
        axis.semilogy(
            ranks,
            values["residual"],
            marker="o",
            ms=2.8,
            lw=1.15,
            color=palette[name],
            label=labels[name],
        )
    axis.axhline(0.10, color="#777777", ls="--", lw=0.8, label="10%")
    axis.axhline(0.05, color="#999999", ls=":", lw=0.8, label="5%")
    axis.set_xlabel("Mode rank")
    axis.set_ylabel("Normalized strong Laplacian residual")
    axis.set_title(
        f"Potential-Rayleigh eigenrelation · K={modes} · Q={quadrature_points}"
    )
    axis.grid(True, which="both", color="#D9D9D9", lw=0.55)
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    figure.savefig(output_dir / "laplacian_eigenrelation.png", dpi=240)
    figure.savefig(output_dir / "laplacian_eigenrelation.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quadrature-points", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=192)
    parser.add_argument("--point-batch", type=int, default=2048)
    args = parser.parse_args()

    _metadata, cfg, model = load_model(args.checkpoint.resolve())
    integration_points = fluid_points(
        cfg,
        args.quadrature_points,
        0.0,
        0.0,
        int(cfg.seed) + 3_100_000,
    ).astype(np.float32)
    area = (float(cfg.xmax) - float(cfg.xmin)) * (
        float(cfg.ymax) - float(cfg.ymin)
    )
    raw_value, raw_gx, raw_gy, _gauge = raw_potential_jet1(
        model, cfg, integration_points
    )
    peeling, trained_lambda, _diagnostics = causal_peeling_transform(
        cfg, raw_value, raw_gx, raw_gy, area
    )
    psi = raw_value @ peeling
    gx = raw_gx @ peeling
    gy = raw_gy @ peeling
    weight = area / args.quadrature_points
    mass = cp.asnumpy(cp.float32(weight) * (psi.T @ psi)).astype(np.float64)
    potential_stiffness = cp.asnumpy(
        cp.float32(weight) * (gx.T @ gx + gy.T @ gy)
    ).astype(np.float64)
    potential_lambda, potential_rotation, _rank = generalized_symmetric_eig(
        potential_stiffness,
        mass,
        int(cfg.modes),
        sample_rows=args.quadrature_points,
    )
    raw_omega = evaluate_raw_omega(
        model, cfg, integration_points, args.point_batch
    )
    omega = raw_omega @ peeling
    velocity_stiffness = cp.asnumpy(
        cp.float32(weight) * (omega.T @ omega)
    ).astype(np.float64)
    velocity_mass = potential_stiffness
    velocity_lambda, velocity_rotation, _rank = generalized_symmetric_eig(
        velocity_stiffness,
        velocity_mass,
        int(cfg.modes),
        sample_rows=2 * args.quadrature_points,
    )

    transforms = {
        "trained_peeled": cp.asnumpy(peeling).astype(np.float64),
        "potential_ritz": cp.asnumpy(peeling).astype(np.float64)
        @ potential_rotation,
        "velocity_ritz": cp.asnumpy(peeling).astype(np.float64)
        @ velocity_rotation,
    }
    eigenvalues = {
        "trained_peeled": trained_lambda.astype(np.float64),
        "potential_ritz": potential_lambda,
        "velocity_ritz": velocity_lambda,
    }
    spacing_x = (float(cfg.xmax) - float(cfg.xmin)) / args.grid_size
    spacing_y = (float(cfg.ymax) - float(cfg.ymin)) / args.grid_size
    axis_x = float(cfg.xmin) + (
        np.arange(args.grid_size, dtype=np.float64) + 0.5
    ) * spacing_x
    axis_y = float(cfg.ymin) + (
        np.arange(args.grid_size, dtype=np.float64) + 0.5
    ) * spacing_y
    grid_x, grid_y = np.meshgrid(axis_x, axis_y, indexing="xy")
    validation_points = np.column_stack(
        (grid_x.ravel(), grid_y.ravel())
    ).astype(np.float32)
    ix, iy = np.meshgrid(
        np.arange(args.grid_size), np.arange(args.grid_size), indexing="xy"
    )
    interior_mask = (
        (ix >= 4)
        & (ix < args.grid_size - 4)
        & (iy >= 4)
        & (iy < args.grid_size - 4)
    ).ravel()
    result = validate_transforms(
        model,
        cfg,
        validation_points,
        interior_mask,
        transforms,
        eigenvalues,
        args.point_batch,
    )
    write_outputs(
        args.output_dir.resolve(), result, args.grid_size, args.quadrature_points
    )
    print(
        json.dumps(
            {name: stats(values["residual"]) for name, values in result.items()},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
