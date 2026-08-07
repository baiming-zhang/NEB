"""Reconstruct the converged rotating-fan reference with all 1024 PINN modes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import cupy as cp
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from evaluate_laplacian_gt_projection import (
    CASE_ROOT,
    InferenceEngine,
    fluid_points,
)
from reconstruct_laplacian_gt_with_all_modes import (
    evaluate_projection_errors,
    fit_all_mode_projection,
    full_peeled_matrix,
)
from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (
    raw_velocity,
    rotor_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "checkpoints"
            / "best_inference.npz"
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "gt_rotating_stable_re500_nominal_n96"
            / "stable_fan_reference.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "stable_fan_n96_all1024_reconstruction"
        ),
    )
    parser.add_argument("--fit-points", type=int, default=16384)
    parser.add_argument("--validation-points", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=1024)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


class ReferenceField:
    def __init__(self, path: Path) -> None:
        with h5py.File(path, "r") as archive:
            self.axis = archive["axis"][:].astype(np.float64)
            self.velocity = archive["velocity"][:].astype(np.float64)
            self.vorticity = archive["vorticity"][:].astype(np.float64)
            self.rotor_mask = archive["rotor_mask"][:].astype(np.float64)
        self.dx = float(self.axis[1] - self.axis[0])
        self.xmin = float(self.axis[0])

    def interpolate(
        self,
        field: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        coordinates = ((points - self.xmin) / self.dx).T
        return np.column_stack(
            [
                map_coordinates(
                    field[component],
                    coordinates,
                    order=1,
                    mode="grid-wrap",
                    prefilter=False,
                )
                for component in range(field.shape[0])
            ]
        )

    def disturbance(self, points: np.ndarray) -> np.ndarray:
        values = self.interpolate(self.velocity, points)
        values[:, 0] -= 1.0
        return values


def evaluate_raw_reconstruction(
    engine: InferenceEngine,
    points: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    chunk: int,
) -> np.ndarray:
    values = np.empty((len(points), 3), dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        raw = raw_velocity(
            engine.cfg,
            engine.model,
            points[start:stop],
            parameters,
        )
        values[start:stop] = cp.asnumpy(raw @ coefficients).reshape(
            3, stop - start
        ).T
    return values


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(
        np.linalg.norm((approximation - reference).ravel())
        / max(np.linalg.norm(reference.ravel()), 1.0e-30)
    )


def make_comparison_figure(
    engine: InferenceEngine,
    reference: ReferenceField,
    parameters: np.ndarray,
    raw_coefficients: cp.ndarray,
    chunk: int,
    output: Path,
) -> dict[str, float]:
    cfg = engine.cfg
    x = reference.axis.astype(np.float32)
    y = reference.axis.astype(np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zeros = np.zeros_like(xx)
    center = np.column_stack((xx.ravel(), yy.ravel(), zeros.ravel()))
    dy = reference.dx
    y_plus = center.copy()
    y_minus = center.copy()
    z_plus = center.copy()
    z_minus = center.copy()
    y_plus[:, 1] += dy
    y_minus[:, 1] -= dy
    z_plus[:, 2] += dy
    z_minus[:, 2] -= dy
    stacked = np.concatenate((center, y_plus, y_minus, z_plus, z_minus))
    predicted = evaluate_raw_reconstruction(
        engine, stacked, parameters, raw_coefficients, chunk
    )
    count = len(center)
    center_prediction = predicted[:count]
    omega_x_prediction = (
        predicted[count : 2 * count, 2]
        - predicted[2 * count : 3 * count, 2]
        - predicted[3 * count : 4 * count, 1]
        + predicted[4 * count : 5 * count, 1]
    ) / (2.0 * dy)

    middle = int(np.argmin(np.abs(reference.axis)))
    gt_disturbance_x = reference.velocity[0, :, :, middle] - 1.0
    gt_omega_x = reference.vorticity[0, :, :, middle]
    predicted_disturbance_x = center_prediction[:, 0].reshape(xx.shape)
    predicted_omega_x = omega_x_prediction.reshape(xx.shape)
    physical = (
        (np.abs(yy) <= float(cfg.farfield_radius))
        & (
            np.asarray(rotor_distance(cfg, center, parameters)).reshape(xx.shape)
            > 0.0
        )
    )
    fields = [
        gt_disturbance_x,
        predicted_disturbance_x,
        predicted_disturbance_x - gt_disturbance_x,
        gt_omega_x,
        predicted_omega_x,
        predicted_omega_x - gt_omega_x,
    ]
    fields = [np.where(physical, field, np.nan) for field in fields]
    velocity_limit = max(
        np.nanquantile(np.abs(fields[0]), 0.99),
        np.nanquantile(np.abs(fields[1]), 0.99),
        1.0e-6,
    )
    vorticity_limit = max(
        np.nanquantile(np.abs(fields[3]), 0.99),
        np.nanquantile(np.abs(fields[4]), 0.99),
        1.0e-6,
    )
    limits = [
        velocity_limit,
        velocity_limit,
        velocity_limit,
        vorticity_limit,
        vorticity_limit,
        vorticity_limit,
    ]
    titles = [
        r"GT $u_x-U_\infty$",
        r"1024-basis reconstruction",
        r"velocity error",
        r"GT $\omega_x$",
        r"1024-basis reconstruction",
        r"vorticity error",
    ]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 7.4),
        constrained_layout=True,
        facecolor="white",
    )
    for axis_plot, field, limit, title in zip(
        axes.ravel(), fields, limits, titles
    ):
        image = axis_plot.pcolormesh(
            x,
            y,
            field.T,
            shading="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            rasterized=True,
        )
        axis_plot.contour(
            x,
            y,
            reference.rotor_mask[:, :, middle].T,
            levels=[0.5],
            colors=["#555555"],
            linewidths=1.2,
        )
        axis_plot.set_aspect("equal")
        axis_plot.set_xlim(cfg.xmin, cfg.xmax)
        axis_plot.set_ylim(-cfg.farfield_radius, cfg.farfield_radius)
        axis_plot.set_title(title, color="#17365d", fontsize=11)
        axis_plot.set_xlabel("x")
        axis_plot.set_ylabel("y")
        figure.colorbar(image, ax=axis_plot, shrink=0.83, pad=0.02)
    figure.suptitle(
        "Converged four-blade fan: numerical GT vs full PINN basis",
        color="#17365d",
        fontsize=15,
        fontweight="semibold",
    )
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    return {
        "centerplane_velocity_disturbance_relative_l2": relative_error(
            gt_disturbance_x[physical],
            predicted_disturbance_x[physical],
        ),
        "centerplane_axial_vorticity_relative_l2": relative_error(
            gt_omega_x[physical],
            predicted_omega_x[physical],
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = InferenceEngine(args.checkpoint)
    reference = ReferenceField(args.reference)
    parameters = engine.normalize(23.0, 0.28, 0.14)

    fit_points = fluid_points(
        engine.cfg,
        args.fit_points,
        parameters,
        engine.cfg.seed + 31_000_000,
    )
    fit_targets = reference.disturbance(fit_points)[:, None, :]
    raw_coefficients, fit_diagnostics = fit_all_mode_projection(
        engine,
        fit_points,
        fit_targets,
        parameters,
        args.chunk,
    )

    validation_points = fluid_points(
        engine.cfg,
        args.validation_points,
        parameters,
        engine.cfg.seed + 32_000_000,
    )
    validation_targets = reference.disturbance(validation_points)[:, None, :]
    validation_reconstruction, disturbance_errors = evaluate_projection_errors(
        engine,
        validation_points,
        validation_targets,
        parameters,
        raw_coefficients,
        args.chunk,
    )
    target_full = validation_targets[:, 0].copy()
    target_full[:, 0] += 1.0
    reconstruction_full = validation_reconstruction[:, 0].copy()
    reconstruction_full[:, 0] += 1.0

    core = (
        (np.abs(validation_points[:, 0]) < engine.cfg.xmax - 0.72)
        & (
            np.hypot(validation_points[:, 1], validation_points[:, 2])
            < engine.cfg.farfield_radius - 0.25
        )
    )
    peeled_matrix, peeling_diagnostics = full_peeled_matrix(engine, parameters)
    peeled_coefficients = cp.linalg.solve(
        peeled_matrix,
        raw_coefficients,
    )
    transform_residual = float(
        (
            cp.linalg.norm(
                peeled_matrix @ peeled_coefficients - raw_coefficients
            )
            / cp.maximum(cp.linalg.norm(raw_coefficients), 1.0e-30)
        ).get()
    )

    figure_metrics = make_comparison_figure(
        engine,
        reference,
        parameters,
        raw_coefficients,
        args.chunk,
        args.output_dir / "gt_vs_all1024_reconstruction.png",
    )
    np.savez_compressed(
        args.output_dir / "all1024_projection_coefficients.npz",
        raw_network_coefficients=cp.asnumpy(raw_coefficients[:, 0]),
        peeled_pinn_basis_coefficients=cp.asnumpy(peeled_coefficients[:, 0]),
        parameters_normalized=parameters,
    )
    metrics = {
        "status": "completed",
        "reference": str(args.reference.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "basis": "all 1024 trained divergence-free PINN modes",
        "projection_target": "u - U_infinity e_x",
        "lifting": "U_infinity e_x is added back after reconstruction",
        "parameters_physical": {
            "beta_tip_deg": 23.0,
            "chord_tip": 0.28,
            "thickness_ratio": 0.14,
        },
        "parameters_normalized": parameters.tolist(),
        "fit_points": len(fit_points),
        "validation_points": len(validation_points),
        "fit": fit_diagnostics,
        "validation_disturbance_relative_l2": disturbance_errors[0],
        "validation_full_velocity_relative_l2": relative_error(
            target_full,
            reconstruction_full,
        ),
        "validation_core_disturbance_relative_l2": relative_error(
            validation_targets[core, 0],
            validation_reconstruction[core, 0],
        ),
        "validation_core_full_velocity_relative_l2": relative_error(
            target_full[core],
            reconstruction_full[core],
        ),
        "peeling": peeling_diagnostics,
        "raw_to_peeled_transform_relative_residual": transform_residual,
        **figure_metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
