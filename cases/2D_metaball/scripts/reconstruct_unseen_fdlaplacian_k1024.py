"""Build K=1024 FD Laplacian modes and reconstruct one unseen metaball flow."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from compare_three_gt_eigenfluid_vs_laplacian import (
    fit_velocity,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
    serializable_gt,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from metaball_runtime import (
    backend,
    fluid_mask,
    geometry_from_token,
    install_backend,
)
from simulate_ns_vs_eigenfluid_video import positive_dirichlet_laplacian


F32 = np.float32
F64 = np.float64
CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    CASE_ROOT / "configs/k1024_metaball_connected_siren_original_jet1_2000.json"
)


def masked_grid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(valid.shape, np.nan, dtype=F32)
    result[valid] = np.asarray(values, dtype=F32)
    return result


def render(
    output_stem: Path,
    valid: np.ndarray,
    gt_omega: np.ndarray,
    reconstructed_omega: np.ndarray,
    gt_velocity: np.ndarray,
    reconstructed_velocity: np.ndarray,
    metrics: dict[str, float],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
        }
    )
    gt_omega_grid = masked_grid(gt_omega, valid)
    reconstructed_omega_grid = masked_grid(reconstructed_omega, valid)
    omega_error = np.abs(reconstructed_omega_grid - gt_omega_grid)
    gt_speed = masked_grid(np.linalg.norm(gt_velocity, axis=1), valid)
    reconstructed_speed = masked_grid(
        np.linalg.norm(reconstructed_velocity, axis=1), valid
    )
    speed_error = np.abs(reconstructed_speed - gt_speed)
    omega_limit = max(float(np.nanpercentile(np.abs(gt_omega_grid), 99.0)), 1.0e-8)
    speed_limit = max(float(np.nanpercentile(gt_speed, 99.0)), 1.0e-8)
    error_limits = (
        max(float(np.nanpercentile(omega_error, 99.0)), 1.0e-8),
        max(float(np.nanpercentile(speed_error, 99.0)), 1.0e-8),
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.7, 6.4), constrained_layout=True)
    rows = (
        (gt_omega_grid, reconstructed_omega_grid, omega_error),
        (gt_speed, reconstructed_speed, speed_error),
    )
    for row, fields in enumerate(rows):
        for column, field in enumerate(fields):
            axis = axes[row, column]
            if row == 0 and column < 2:
                image = axis.imshow(
                    field,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=-omega_limit,
                    vmax=omega_limit,
                )
            elif row == 1 and column < 2:
                image = axis.imshow(
                    field,
                    origin="lower",
                    cmap="Blues",
                    vmin=0.0,
                    vmax=speed_limit,
                )
            else:
                image = axis.imshow(
                    field,
                    origin="lower",
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_limits[row],
                )
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    axes[0, 0].set_title("Ground-truth vorticity")
    axes[0, 1].set_title(
        rf"FD Laplacian $K=1024$ ($e_\omega={metrics['test_vorticity_relative_l2']:.3f}$)"
    )
    axes[0, 2].set_title("Absolute vorticity error")
    axes[1, 0].set_title("Ground-truth speed")
    axes[1, 1].set_title(
        rf"FD Laplacian $K=1024$ ($e_u={metrics['test_velocity_relative_l2']:.3f}$)"
    )
    axes[1, 2].set_title("Absolute speed error")
    fig.suptitle(
        "Unseen connected-metaball reconstruction · 128 × 128 matched grid",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", type=int, default=4_000_001)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--modes", type=int, default=1024)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.modes != 1024:
        raise ValueError("this audit is fixed to K=1024")
    if args.resolution != 128:
        raise ValueError("this audit is fixed to a 128 x 128 grid")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    token = float(args.token)
    valid = fluid_mask(cfg, all_points, token).reshape(
        args.resolution, args.resolution
    )
    points = all_points[valid.ravel()]
    if len(points) <= args.modes + 1:
        raise ValueError(
            f"only {len(points)} fluid nodes are available for K={args.modes}"
        )

    started = time.perf_counter()
    eigensolve_started = time.perf_counter()
    eigenvalues, laplacian_psi, _, _ = finite_difference_laplacian(
        valid, spacing, args.modes
    )
    eigensolve_seconds = time.perf_counter() - eigensolve_started
    laplacian_velocity, laplacian_omega = matched_grid_observables(
        laplacian_psi, valid, spacing
    )

    operator = positive_dirichlet_laplacian(valid, spacing)
    residual = operator @ laplacian_psi.astype(F64) - (
        laplacian_psi.astype(F64) * eigenvalues[None, :]
    )
    relative_eigen_residual = float(
        np.linalg.norm(residual)
        / max(
            float(
                np.linalg.norm(
                    laplacian_psi.astype(F64) * eigenvalues[None, :]
                )
            ),
            np.finfo(F64).tiny,
        )
    )
    mass_gram = (spacing * spacing) * (
        laplacian_psi.astype(F64).T @ laplacian_psi.astype(F64)
    )
    mass_orthogonality = float(
        np.max(np.abs(mass_gram - np.eye(args.modes)))
    )

    split_rng = np.random.default_rng(args.seed)
    permutation = split_rng.permutation(len(points))
    fit_count = int(0.70 * len(points))
    fit_indices = permutation[:fit_count]
    test_indices = permutation[fit_count:]
    gt_seed = args.seed + 100_003
    gt_parameters = generate_gt_parameters(
        cfg, token, gt_seed, args.vortices
    )
    gt_psi, _, _, gt_normalization = ground_truth_fields(
        cfg, points, token, gt_parameters
    )
    gt_velocity, gt_omega = matched_grid_observables(gt_psi, valid, spacing)

    reconstruction = fit_velocity(
        laplacian_psi,
        laplacian_velocity,
        laplacian_omega,
        gt_psi,
        gt_velocity,
        gt_omega,
        fit_indices,
        test_indices,
        args.ridge_relative,
    )
    total_seconds = time.perf_counter() - started

    basis_path = args.output_dir / "fd_laplacian_k1024_modes_128x128.npz"
    np.savez(
        basis_path,
        eigenvalues=eigenvalues.astype(F64),
        psi_valid=laplacian_psi.astype(F32),
        valid_mask=valid,
        coordinates=coordinates,
        spacing=np.asarray(spacing, dtype=F64),
        geometry_token=np.asarray(args.token, dtype=np.int64),
        geometry_parameters=geometry_from_token(cfg, token).astype(F64),
    )
    fields_path = args.output_dir / "unseen_reconstruction_fields.npz"
    np.savez_compressed(
        fields_path,
        valid_mask=valid,
        points=points,
        fit_indices=fit_indices,
        test_indices=test_indices,
        gt_psi=gt_psi,
        gt_velocity=gt_velocity,
        gt_omega=gt_omega,
        reconstructed_psi=reconstruction["psi"],
        reconstructed_velocity=reconstruction["velocity"],
        reconstructed_omega=reconstruction["omega"],
        coefficients=reconstruction["coefficients"],
    )

    summary = {
        "status": "complete",
        "case": "2D connected three-circle smooth metaball",
        "geometry_status": "unseen",
        "geometry_token": args.token,
        "geometry_parameters": geometry_from_token(cfg, token).tolist(),
        "grid_resolution": args.resolution,
        "fluid_nodes": int(len(points)),
        "fit_nodes": int(len(fit_indices)),
        "held_out_test_nodes": int(len(test_indices)),
        "modes": args.modes,
        "boundary_condition": "psi=0 by zero extension on every outer and obstacle boundary component",
        "operator": "geometry-specific five-point finite-difference positive Dirichlet -Delta",
        "eigensolver": "SciPy symmetric shift-invert eigsh",
        "coefficient_fit": "velocity least squares on 70% of valid nodes",
        "evaluation": "relative L2 on the disjoint held-out 30% valid nodes",
        "ridge_relative": args.ridge_relative,
        "gt_seed": gt_seed,
        "gt_velocity_rms_before_normalization": gt_normalization,
        "gt_parameters": serializable_gt(gt_parameters),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "first16_eigenvalues": [float(value) for value in eigenvalues[:16]],
        "mass_orthogonality_max_abs": mass_orthogonality,
        "relative_eigen_residual_frobenius": relative_eigen_residual,
        "metrics": reconstruction["metrics"],
        "timing_seconds": {
            "eigensolve": eigensolve_seconds,
            "total": total_seconds,
        },
        "outputs": {
            "basis": str(basis_path.resolve()),
            "fields": str(fields_path.resolve()),
            "figure_png": str(
                (args.output_dir / "unseen_fdlaplacian_k1024_reconstruction.png").resolve()
            ),
            "figure_pdf": str(
                (args.output_dir / "unseen_fdlaplacian_k1024_reconstruction.pdf").resolve()
            ),
        },
        "peeling_policy_note": (
            "No peeling is used for the FD Laplacian basis. Future learned-basis "
            "runs use one projection pass and reject candidates that fail the "
            "independence criterion."
        ),
    }
    summary_path = args.output_dir / "unseen_fdlaplacian_k1024_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "unseen_fdlaplacian_k1024_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for name, value in reconstruction["metrics"].items():
            writer.writerow((name, value))
    render(
        args.output_dir / "unseen_fdlaplacian_k1024_reconstruction",
        valid,
        gt_omega,
        reconstruction["omega"],
        gt_velocity,
        reconstruction["velocity"],
        reconstruction["metrics"],
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
