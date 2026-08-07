"""Reconstruct the first six Laplacian GT modes with all 1024 PINN modes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from evaluate_laplacian_gt_projection import (
    CASE_ROOT,
    InferenceEngine,
    evaluate_laplacian_modes,
    fluid_points,
    laplacian_mode_table,
    peel_and_adjoint,
    raw_fields,
    raw_velocity,
    write_json,
)
from render_first_six_isovorticity import vorticity_field
from render_first_six_mode_video import draw_cylinder, rotor_mesh
from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (
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
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "laplacian_gt_all1024_reconstruction"
        ),
    )
    parser.add_argument("--fit-points", type=int, default=16384)
    parser.add_argument("--validation-points", type=int, default=8192)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--surface-quantile", type=float, default=0.90)
    parser.add_argument("--surface-step", type=int, default=2)
    return parser.parse_args()


def component_major(fields: np.ndarray) -> np.ndarray:
    return fields.transpose(2, 0, 1).reshape(
        3 * len(fields),
        fields.shape[1],
    )


def full_peeled_matrix(
    engine: InferenceEngine,
    parameters: np.ndarray,
) -> tuple[cp.ndarray, dict[str, object]]:
    cfg = engine.cfg
    points = fluid_points(
        cfg,
        cfg.validation_points,
        parameters,
        cfg.seed + 9_000_000,
    )
    raw = raw_fields(cfg, engine.model, points, parameters)
    peeled = peel_and_adjoint(
        cfg,
        raw,
        len(points),
        need_adjoint=False,
        need_coefficients=True,
    )
    return peeled["coefficient_matrix"], {
        "peeling_points": len(points),
        "peeling_orthogonality_error": float(peeled["orth_error"]),
        "effective_raw_rank": int(peeled["effective_raw_rank"]),
    }


def fit_all_mode_projection(
    engine: InferenceEngine,
    points: np.ndarray,
    targets: np.ndarray,
    parameters: np.ndarray,
    chunk: int,
) -> tuple[cp.ndarray, dict[str, object]]:
    """Fit each physical GT mode in the complete raw/PINN span."""
    cfg = engine.cfg
    target_matrix = component_major(targets)
    gram = cp.zeros((cfg.modes, cfg.modes), dtype=cp.float64)
    right = cp.zeros((cfg.modes, targets.shape[1]), dtype=cp.float64)
    target_mass = np.sum(target_matrix * target_matrix, axis=0, dtype=np.float64)

    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        raw = raw_velocity(
            cfg,
            engine.model,
            points[start:stop],
            parameters,
        )
        target_chunk = np.concatenate(
            [
                target_matrix[
                    component * len(points) + start :
                    component * len(points) + stop
                ]
                for component in range(3)
            ],
            axis=0,
        )
        target_device = cp.asarray(target_chunk, dtype=cp.float64)
        gram += raw.T @ raw
        right += raw.T @ target_device
        print(f"[fit {stop:05d}/{len(points):05d}]", flush=True)

    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    threshold = cp.maximum(eigenvalues[-1], cfg.mass_floor) * 1.0e-8
    keep = eigenvalues > threshold
    transformed = eigenvectors[:, keep].T @ right
    coefficients = eigenvectors[:, keep] @ (
        transformed / eigenvalues[keep, None]
    )
    captured = cp.asnumpy(cp.sum(right * coefficients, axis=0))
    fit_errors = np.sqrt(
        np.maximum(0.0, (target_mass - captured) / target_mass)
    )
    return coefficients, {
        "effective_rank": int(cp.count_nonzero(keep).get()),
        "relative_rank_threshold": 1.0e-8,
        "gram_largest_eigenvalue": float(eigenvalues[-1].get()),
        "gram_smallest_retained_eigenvalue": float(
            eigenvalues[keep][0].get()
        ),
        "fit_projection_errors": fit_errors.tolist(),
        "fit_mean_error": float(np.mean(fit_errors)),
        "fit_rms_error": float(np.sqrt(np.mean(fit_errors**2))),
    }


def evaluate_projection_errors(
    engine: InferenceEngine,
    points: np.ndarray,
    targets: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    chunk: int,
) -> tuple[np.ndarray, list[float]]:
    reconstruction = np.empty_like(targets, dtype=np.float64)
    numerator = np.zeros(targets.shape[1], dtype=np.float64)
    denominator = np.zeros(targets.shape[1], dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        raw = raw_velocity(
            engine.cfg,
            engine.model,
            points[start:stop],
            parameters,
        )
        values = (
            cp.asnumpy(raw @ coefficients)
            .reshape(3, stop - start, targets.shape[1])
            .transpose(1, 2, 0)
        )
        reconstruction[start:stop] = values
        difference = values - targets[start:stop]
        numerator += np.sum(difference * difference, axis=(0, 2))
        denominator += np.sum(
            targets[start:stop] * targets[start:stop],
            axis=(0, 2),
        )
        print(f"[validate {stop:05d}/{len(points):05d}]", flush=True)
    errors = np.sqrt(numerator / np.maximum(denominator, 1.0e-30))
    return reconstruction, errors.tolist()


def evaluate_grid_reconstruction(
    engine: InferenceEngine,
    descriptors: list[dict[str, object]],
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    nx: int,
    nr: int,
    chunk: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    cfg = engine.cfg
    x = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    y = np.linspace(-cfg.farfield_radius, cfg.farfield_radius, nr, dtype=np.float32)
    z = np.linspace(-cfg.farfield_radius, cfg.farfield_radius, nr, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    target = evaluate_laplacian_modes(points, cfg.xmin, descriptors)
    reconstruction = np.empty_like(target, dtype=np.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        raw = raw_velocity(
            cfg,
            engine.model,
            points[start:stop],
            parameters,
        )
        reconstruction[start:stop] = (
            cp.asnumpy(raw @ coefficients)
            .reshape(3, stop - start, len(descriptors))
            .transpose(1, 2, 0)
        )
        if stop % (8 * chunk) == 0 or stop == len(points):
            print(f"[grid {stop:06d}/{len(points):06d}]", flush=True)
    target = target.reshape(nx, nr, nr, len(descriptors), 3).astype(np.float32)
    reconstruction = reconstruction.reshape(
        nx,
        nr,
        nr,
        len(descriptors),
        3,
    ).astype(np.float32)
    radius = np.sqrt(yy * yy + zz * zz)
    obstacle = np.asarray(rotor_distance(cfg, points, parameters)).reshape(
        nx,
        nr,
        nr,
    )
    valid = (
        (radius < 1.82)
        & (obstacle > 0.05)
        & (xx > x[1])
        & (xx < x[-2])
    )
    return x, y, z, target, reconstruction, valid


def surface_mesh(
    omega: np.ndarray,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    level: float,
    color_limit: float,
    color_component: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    magnitude = np.linalg.norm(omega, axis=-1)
    volume = np.where(valid, magnitude, 0.0)
    if float(np.max(volume)) <= level:
        return None
    spacing = np.asarray(
        (x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        dtype=np.float64,
    )
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=level,
        spacing=tuple(float(value) for value in spacing),
        step_size=step,
        allow_degenerate=False,
    )
    origin = np.asarray((x[0], y[0], z[0]))
    vertices += origin
    indices = np.rint((vertices - origin) / spacing).astype(np.int64)
    indices[:, 0] = np.clip(indices[:, 0], 0, len(x) - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, len(y) - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, len(z) - 1)
    signed_component = omega[
        indices[:, 0],
        indices[:, 1],
        indices[:, 2],
        color_component,
    ]
    face_value = np.mean(signed_component[faces], axis=1)
    normalized = np.clip(face_value / max(color_limit, 1.0e-12), -1.0, 1.0)
    return vertices, faces.astype(np.int32), normalized


def setup_axis(ax, cfg, title: str) -> None:
    radial_limit = 1.08 * cfg.farfield_radius
    axial_padding = 0.04 * (cfg.xmax - cfg.xmin)
    ax.set_xlim(cfg.xmin - axial_padding, cfg.xmax + axial_padding)
    ax.set_ylim(-radial_limit, radial_limit)
    ax.set_zlim(-radial_limit, radial_limit)
    ax.set_box_aspect(
        ((cfg.xmax - cfg.xmin) / (2.0 * radial_limit), 1.0, 1.0),
        zoom=1.12,
    )
    ax.set_proj_type("persp", focal_length=0.95)
    ax.view_init(elev=23.0, azim=-43.0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5, pad=1)
    draw_cylinder(ax, cfg.xmin, cfg.xmax, cfg.farfield_radius)


def render_comparison(
    output: Path,
    engine: InferenceEngine,
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_omega: np.ndarray,
    reconstruction_omega: np.ndarray,
    valid: np.ndarray,
    errors: list[float],
    quantile: float,
    surface_step: int,
) -> list[dict[str, object]]:
    cfg = engine.cfg
    figure = plt.figure(figsize=(20.0, 11.25), dpi=120, facecolor="white")
    cmap = matplotlib.colormaps["RdBu_r"]
    rotor_vertices, rotor_faces = rotor_mesh(cfg, parameters)
    surface_rows: list[dict[str, object]] = []

    for mode in range(6):
        gt_magnitude = np.linalg.norm(target_omega[:, :, :, mode], axis=-1)
        usable = gt_magnitude[valid & np.isfinite(gt_magnitude)]
        level = float(np.quantile(usable, quantile))
        component_rms = np.asarray(
            [
                np.sqrt(
                    np.mean(
                        target_omega[:, :, :, mode, component][valid] ** 2
                    )
                )
                for component in range(3)
            ]
        )
        color_component = int(np.argmax(component_rms))
        color_limit = max(
            float(
                np.quantile(
                    np.abs(
                        target_omega[
                            :, :, :, mode, color_component
                        ][valid]
                    ),
                    0.98,
                )
            ),
            1.0e-8,
        )
        surface_rows.append(
            {
                "mode": mode + 1,
                "vorticity_magnitude_level": level,
                "signed_vorticity_component": (
                    ("omega_x", "omega_y", "omega_z")[color_component]
                ),
                "signed_vorticity_color_limit": color_limit,
            }
        )
        for variant in range(2):
            panel = 2 * mode + variant
            ax = figure.add_subplot(3, 4, panel + 1, projection="3d")
            if variant == 0:
                title = f"GT mode {mode + 1}"
                omega = target_omega[:, :, :, mode]
            else:
                title = (
                    f"all-1024 reconstruction {mode + 1}  "
                    f"L2={100.0 * errors[mode]:.1f}%"
                )
                omega = reconstruction_omega[:, :, :, mode]
            setup_axis(ax, cfg, title)
            mesh = surface_mesh(
                omega,
                valid,
                x,
                y,
                z,
                level,
                color_limit,
                color_component,
                surface_step,
            )
            if mesh is not None:
                vertices, faces, normalized = mesh
                colors = cmap(Normalize(-1.0, 1.0)(normalized))
                colors[:, 3] = 0.92
                ax.add_collection3d(
                    Poly3DCollection(
                        vertices[faces],
                        facecolors=colors,
                        edgecolors="none",
                        linewidths=0.0,
                        alpha=0.92,
                        zorder=3,
                    )
                )
            else:
                ax.text2D(
                    0.5,
                    0.5,
                    "no surface at GT level",
                    transform=ax.transAxes,
                    ha="center",
                    color="#666666",
                    fontsize=8,
                )
            ax.add_collection3d(
                Poly3DCollection(
                    rotor_vertices[rotor_faces],
                    facecolors="#8d9499",
                    edgecolors=(0.30, 0.32, 0.34, 0.22),
                    linewidths=0.08,
                    alpha=1.0,
                    zorder=4,
                )
            )
            ax.text2D(
                0.02,
                0.01,
                (
                    rf"$|\omega|_{{iso}}={level:.3f}$"
                    "\n"
                    rf"color: $\omega_{{{'xyz'[color_component]}}}$"
                ),
                transform=ax.transAxes,
                fontsize=7,
                color="#555555",
            )

    color_axis = figure.add_axes((0.935, 0.19, 0.012, 0.62))
    colorbar = figure.colorbar(
        cm.ScalarMappable(norm=Normalize(-1.0, 1.0), cmap="RdBu_r"),
        cax=color_axis,
    )
    colorbar.set_label(
        r"dominant signed vorticity component / GT-p98",
        fontsize=9,
    )
    figure.suptitle(
        "First six cylindrical Laplacian GT modes reconstructed by all "
        "1024 trained PINN modes",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Each GT/reconstruction pair uses the same GT-derived "
            "isosurface level and axial-vorticity color scale"
        ),
        ha="center",
        fontsize=9,
        color="#4f4f4f",
    )
    figure.subplots_adjust(
        left=0.01,
        right=0.92,
        bottom=0.045,
        top=0.93,
        wspace=-0.06,
        hspace=0.02,
    )
    figure.savefig(output, facecolor="white")
    plt.close(figure)
    return surface_rows


def write_error_csv(
    path: Path,
    descriptors: list[dict[str, object]],
    fit_errors: list[float],
    validation_errors: list[float],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "gt_mode",
                "family",
                "m",
                "q",
                "n",
                "parity",
                "eigenvalue",
                "fit_relative_velocity_l2_error",
                "independent_relative_velocity_l2_error",
            ]
        )
        for index, descriptor in enumerate(descriptors):
            writer.writerow(
                [
                    index + 1,
                    descriptor["family"],
                    descriptor["m"],
                    descriptor["q"],
                    descriptor["n"],
                    descriptor["parity"],
                    descriptor["eigenvalue"],
                    fit_errors[index],
                    validation_errors[index],
                ]
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "pipeline_state.json"
    write_json(state_path, {"status": "running", "stage": "loading"})
    started = time.perf_counter()
    engine = InferenceEngine(args.checkpoint)
    cfg = engine.cfg
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    descriptors = laplacian_mode_table(
        6,
        cfg.xmax - cfg.xmin,
        cfg.farfield_radius,
    )

    fit_points = fluid_points(
        cfg,
        args.fit_points,
        parameters,
        cfg.seed + 12_000_000,
    )
    fit_targets = evaluate_laplacian_modes(
        fit_points,
        cfg.xmin,
        descriptors,
    )
    write_json(state_path, {"status": "running", "stage": "fitting"})
    raw_coefficients, fit_metrics = fit_all_mode_projection(
        engine,
        fit_points,
        fit_targets,
        parameters,
        args.chunk,
    )

    peeled_matrix, peeling_diagnostics = full_peeled_matrix(
        engine,
        parameters,
    )
    peeled_coefficients = cp.linalg.solve(
        peeled_matrix,
        raw_coefficients,
    )
    coefficient_transform_residual = float(
        cp.linalg.norm(
            peeled_matrix @ peeled_coefficients - raw_coefficients
        ).get()
        / max(float(cp.linalg.norm(raw_coefficients).get()), 1.0e-30)
    )

    validation_points = fluid_points(
        cfg,
        args.validation_points,
        parameters,
        cfg.seed + 13_000_000,
    )
    validation_targets = evaluate_laplacian_modes(
        validation_points,
        cfg.xmin,
        descriptors,
    )
    write_json(state_path, {"status": "running", "stage": "validating"})
    _, validation_errors = evaluate_projection_errors(
        engine,
        validation_points,
        validation_targets,
        parameters,
        raw_coefficients,
        args.chunk,
    )

    write_json(state_path, {"status": "running", "stage": "grid_reconstruction"})
    x, y, z, target, reconstruction, valid = evaluate_grid_reconstruction(
        engine,
        descriptors,
        parameters,
        raw_coefficients,
        args.nx,
        args.nr,
        args.chunk,
    )
    target_omega = vorticity_field(target, x, y, z)
    reconstruction_omega = vorticity_field(reconstruction, x, y, z)
    figure_path = args.output_dir / "gt_vs_all1024_reconstruction.png"
    surface_rows = render_comparison(
        figure_path,
        engine,
        parameters,
        x,
        y,
        z,
        target_omega,
        reconstruction_omega,
        valid,
        validation_errors,
        args.surface_quantile,
        args.surface_step,
    )

    fit_errors = fit_metrics["fit_projection_errors"]
    csv_path = args.output_dir / "mode_reconstruction_errors.csv"
    write_error_csv(csv_path, descriptors, fit_errors, validation_errors)
    coefficient_path = args.output_dir / "all1024_projection_coefficients.npz"
    np.savez_compressed(
        coefficient_path,
        raw_network_coefficients=cp.asnumpy(raw_coefficients),
        peeled_mode_coefficients=cp.asnumpy(peeled_coefficients),
        gt_eigenvalues=np.asarray(
            [descriptor["eigenvalue"] for descriptor in descriptors]
        ),
        fit_errors=np.asarray(fit_errors),
        independent_validation_errors=np.asarray(validation_errors),
    )

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "objective": (
            "reconstruct each of the first six analytic Laplacian GT modes "
            "with the complete 1024-mode trained PINN span"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "span_equivalence": (
            "all raw networks and all peeled PINN modes have the same "
            "1024-dimensional span"
        ),
        "parameters_normalized": parameters.tolist(),
        "fit_points": args.fit_points,
        "independent_validation_points": args.validation_points,
        "fit_metrics": fit_metrics,
        "independent_validation_errors": validation_errors,
        "independent_mean_error": float(np.mean(validation_errors)),
        "independent_rms_error": float(
            np.sqrt(np.mean(np.asarray(validation_errors) ** 2))
        ),
        "peeling_diagnostics": peeling_diagnostics,
        "raw_to_peeled_coefficient_transform_relative_residual": (
            coefficient_transform_residual
        ),
        "laplacian_gt_modes": descriptors,
        "surface_settings": {
            "quantile": args.surface_quantile,
            "step_size": args.surface_step,
            "rows": surface_rows,
        },
        "outputs": {
            "comparison_figure": str(figure_path.resolve()),
            "errors_csv": str(csv_path.resolve()),
            "coefficients": str(coefficient_path.resolve()),
        },
        "elapsed_seconds": elapsed,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        state_path,
        {
            "status": "complete",
            "stage": "complete",
            "elapsed_seconds": elapsed,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
