"""Project analytic cylindrical Laplacian GT modes onto trained fan modes.

This is the 3D analogue of the 2D airfoil Laplacian baseline: analytic
divergence-free outer-domain modes are evaluated only in the physical fluid,
the rotor interior is excluded, and projection errors use the uniform-volume
velocity L2 inner product.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import jn_zeros, jnp_zeros, jv, jvp

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from serve_interactive import InferenceEngine  # noqa: E402
from eigenfluid.gen.fan3d.hardcoded_backend import (  # noqa: E402
    fluid_points,
    peel_and_adjoint,
    raw_fields,
    raw_velocity,
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
            / "laplacian_gt_projection"
        ),
    )
    parser.add_argument("--gt-count", type=int, default=6)
    parser.add_argument("--learned-count", type=int, default=64)
    parser.add_argument("--points", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--seed-offset", type=int, default=12_000_000)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def laplacian_mode_table(
    count: int,
    length: float,
    radius: float,
) -> list[dict[str, Any]]:
    """Return low cylindrical vector-Laplacian TE/TM mode descriptors.

    TE modes use radial Dirichlet Bessel roots and
    ``u = curl(Psi e_x)``.  TM modes use radial Neumann Bessel roots and
    ``u = curl curl(Psi e_x) / sqrt(lambda)``.  Both are solenoidal vector
    Helmholtz eigenfunctions in the closed outer cylinder.
    """
    upper = max(5, int(math.ceil(count ** (1.0 / 3.0))) + 3)
    candidates: list[dict[str, Any]] = []
    for family in ("TE", "TM"):
        for azimuthal in range(0, upper + 1):
            roots = (
                jn_zeros(azimuthal, upper)
                if family == "TE"
                else jnp_zeros(azimuthal, upper)
            )
            for radial_index, root in enumerate(roots, start=1):
                transverse_wave = float(root) / radius
                for axial_index in range(1, upper + 1):
                    axial_wave = axial_index * math.pi / length
                    eigenvalue = axial_wave**2 + transverse_wave**2
                    parities = ("cos",) if azimuthal == 0 else ("cos", "sin")
                    for parity in parities:
                        candidates.append(
                            {
                                "family": family,
                                "m": azimuthal,
                                "q": radial_index,
                                "n": axial_index,
                                "parity": parity,
                                "bessel_root": float(root),
                                "kappa": transverse_wave,
                                "kx": axial_wave,
                                "eigenvalue": eigenvalue,
                            }
                        )
    candidates.sort(
        key=lambda item: (
            item["eigenvalue"],
            item["family"],
            item["m"],
            item["q"],
            item["n"],
            item["parity"],
        )
    )
    return candidates[:count]


def evaluate_laplacian_modes(
    points: np.ndarray,
    xmin: float,
    descriptors: list[dict[str, Any]],
) -> np.ndarray:
    """Evaluate analytic divergence-free TE/TM modes at Cartesian points."""
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    z = points[:, 2].astype(np.float64)
    radius = np.hypot(y, z)
    angle = np.arctan2(z, y)
    safe_radius = np.maximum(radius, 1.0e-12)
    cosine = y / safe_radius
    sine = z / safe_radius
    modes = np.empty((len(points), len(descriptors), 3), dtype=np.float64)

    for mode_index, descriptor in enumerate(descriptors):
        m = int(descriptor["m"])
        kappa = float(descriptor["kappa"])
        kx = float(descriptor["kx"])
        eigenvalue = float(descriptor["eigenvalue"])
        phase = kx * (x - xmin)
        axial = np.sin(phase)
        axial_derivative = kx * np.cos(phase)
        bessel = jv(m, kappa * radius)
        bessel_derivative = kappa * jvp(m, kappa * radius, 1)
        if descriptor["parity"] == "cos":
            angular = np.cos(m * angle)
            angular_derivative = -m * np.sin(m * angle)
        else:
            angular = np.sin(m * angle)
            angular_derivative = m * np.cos(m * angle)
        radial_field = bessel * angular
        derivative_r = bessel_derivative * angular
        derivative_angle_over_r = (
            bessel * angular_derivative / safe_radius
        )
        derivative_y = (
            cosine * derivative_r - sine * derivative_angle_over_r
        )
        derivative_z = (
            sine * derivative_r + cosine * derivative_angle_over_r
        )

        if descriptor["family"] == "TE":
            modes[:, mode_index, 0] = 0.0
            modes[:, mode_index, 1] = axial * derivative_z
            modes[:, mode_index, 2] = -axial * derivative_y
        else:
            scale = 1.0 / math.sqrt(eigenvalue)
            modes[:, mode_index, 0] = (
                kappa**2 * axial * radial_field * scale
            )
            modes[:, mode_index, 1] = (
                axial_derivative * derivative_y * scale
            )
            modes[:, mode_index, 2] = (
                axial_derivative * derivative_z * scale
            )
    return modes


def validate_laplacian_modes(
    points: np.ndarray,
    xmin: float,
    descriptors: list[dict[str, Any]],
    step: float = 5.0e-4,
) -> dict[str, Any]:
    """Finite-difference check of div(u)=0 and -Delta(u)=lambda*u."""
    selected = points[: min(512, len(points))].astype(np.float64)
    center = evaluate_laplacian_modes(selected, xmin, descriptors)
    divergence = np.zeros((len(selected), len(descriptors)), dtype=np.float64)
    laplacian = np.zeros_like(center)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = step
        plus = evaluate_laplacian_modes(selected + offset, xmin, descriptors)
        minus = evaluate_laplacian_modes(selected - offset, xmin, descriptors)
        divergence += (plus[:, :, axis] - minus[:, :, axis]) / (2.0 * step)
        laplacian += (plus - 2.0 * center + minus) / (step * step)
    rows = []
    for index, descriptor in enumerate(descriptors):
        eigenvalue = float(descriptor["eigenvalue"])
        velocity_rms = float(np.sqrt(np.mean(center[:, index] ** 2)))
        divergence_rms = float(np.sqrt(np.mean(divergence[:, index] ** 2)))
        eigen_residual_rms = float(
            np.sqrt(
                np.mean(
                    (
                        laplacian[:, index]
                        + eigenvalue * center[:, index]
                    )
                    ** 2
                )
            )
        )
        rows.append(
            {
                "mode": index + 1,
                "scaled_divergence_rms": divergence_rms
                / max(math.sqrt(eigenvalue) * velocity_rms, 1.0e-30),
                "relative_vector_laplacian_residual": eigen_residual_rms
                / max(eigenvalue * velocity_rms, 1.0e-30),
            }
        )
    return {
        "finite_difference_step": step,
        "points": len(selected),
        "modes": rows,
        "maximum_scaled_divergence_rms": max(
            row["scaled_divergence_rms"] for row in rows
        ),
        "maximum_relative_vector_laplacian_residual": max(
            row["relative_vector_laplacian_residual"] for row in rows
        ),
    }


def learned_coefficients(
    engine: InferenceEngine,
    parameters: np.ndarray,
    learned_count: int,
) -> tuple[cp.ndarray, np.ndarray, dict[str, Any]]:
    cfg = engine.cfg
    quadrature = fluid_points(
        cfg,
        cfg.validation_points,
        parameters,
        cfg.seed + 9_000_000,
    )
    raw = raw_fields(cfg, engine.model, quadrature, parameters)
    peeled = peel_and_adjoint(
        cfg,
        raw,
        len(quadrature),
        need_adjoint=False,
        need_coefficients=True,
    )
    # Match the browser and the first-six renderer: peeling is causal, while
    # presentation/comparison is stably reordered by the final Rayleigh value.
    energy_order = np.argsort(peeled["mode_energies"], kind="stable")
    selected_ranks = energy_order[:learned_count]
    coefficients = peeled["coefficient_matrix"][:, selected_ranks]
    energies = np.asarray(peeled["mode_energies"])[selected_ranks]
    diagnostics = {
        "peeling_points": len(quadrature),
        "peeling_orthogonality_error": float(peeled["orth_error"]),
        "effective_raw_rank": int(peeled["effective_raw_rank"]),
        "comparison_order": "stable ascending final Rayleigh energy",
        "selected_peeling_ranks": [
            int(value) for value in selected_ranks
        ],
        "mode_networks": [
            int(peeled["mode_order"][rank]) for rank in selected_ranks
        ],
    }
    return coefficients, energies, diagnostics


def evaluate_learned_modes(
    engine: InferenceEngine,
    points: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    gt_q_component_major: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_count = coefficients.shape[1]
    velocity = np.empty((len(points), mode_count, 3), dtype=np.float64)
    raw_gram = cp.zeros((engine.cfg.modes, engine.cfg.modes), dtype=cp.float64)
    raw_rhs = cp.zeros(
        (engine.cfg.modes, gt_q_component_major.shape[1]),
        dtype=cp.float64,
    )
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        raw = raw_velocity(
            engine.cfg,
            engine.model,
            points[start:stop],
            parameters,
        )
        gt_chunk = np.concatenate(
            [
                gt_q_component_major[
                    component * len(points) + start :
                    component * len(points) + stop
                ]
                for component in range(3)
            ],
            axis=0,
        )
        gt_chunk_device = cp.asarray(gt_chunk, dtype=cp.float64)
        raw_gram += raw.T @ raw
        raw_rhs += raw.T @ gt_chunk_device
        evaluated = raw @ coefficients
        velocity[start:stop] = (
            cp.asnumpy(evaluated)
            .reshape(3, stop - start, mode_count)
            .transpose(1, 2, 0)
        )
        print(
            f"[learned field {stop:05d}/{len(points):05d}]",
            flush=True,
        )
    scale = 1.0 / (3.0 * len(points))
    raw_gram *= scale
    raw_rhs *= scale
    raw_gram = 0.5 * (raw_gram + raw_gram.T)
    eigenvalues, eigenvectors = cp.linalg.eigh(raw_gram)
    threshold = cp.maximum(eigenvalues[-1], engine.cfg.mass_floor) * 1.0e-8
    keep = eigenvalues > threshold
    transformed_rhs = eigenvectors[:, keep].T @ raw_rhs
    capture = transformed_rhs.T @ (
        transformed_rhs / eigenvalues[keep, None]
    )
    capture = 0.5 * (capture + capture.T)
    capture_host = cp.asnumpy(capture)
    capture_host = np.clip(capture_host, -1.0, 1.0)
    squared_cosines = np.clip(
        np.linalg.eigvalsh(capture_host)[::-1],
        0.0,
        1.0,
    )
    residuals = np.sqrt(
        np.maximum(0.0, 1.0 - np.diag(capture_host))
    )
    full_span = {
        "basis": "all 1024 raw networks; same span as full peeled basis",
        "gram_relative_rank_threshold": 1.0e-8,
        "effective_rank": int(cp.count_nonzero(keep).get()),
        "gram_largest_eigenvalue": float(eigenvalues[-1].get()),
        "gram_smallest_retained_eigenvalue": float(
            eigenvalues[keep][0].get()
        ),
        "gt_projection_residuals": residuals.tolist(),
        "mean_gt_projection_error": float(np.mean(residuals)),
        "rms_gt_subspace_error": float(
            np.sqrt(
                max(
                    0.0,
                    1.0 - float(np.sum(squared_cosines))
                    / len(squared_cosines),
                )
            )
        ),
        "principal_cosines": np.sqrt(squared_cosines).tolist(),
        "principal_angles_deg": np.degrees(
            np.arccos(np.sqrt(squared_cosines))
        ).tolist(),
        "maximum_principal_angle_deg": float(
            np.max(
                np.degrees(np.arccos(np.sqrt(squared_cosines)))
            )
        ),
    }
    return velocity, full_span


def flattened(velocity: np.ndarray) -> np.ndarray:
    return velocity.transpose(0, 2, 1).reshape(-1, velocity.shape[1])


def orthonormalize(
    fields: np.ndarray,
    *,
    relative_floor: float = 1.0e-11,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Symmetrically whiten columns under the sampled uniform-volume L2."""
    gram = (fields.T @ fields) / len(fields)
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    threshold = max(float(values[-1]) * relative_floor, 1.0e-14)
    keep = values > threshold
    if np.count_nonzero(keep) != fields.shape[1]:
        raise RuntimeError(
            f"mode set lost rank: {np.count_nonzero(keep)}/{fields.shape[1]}"
        )
    transform = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    orthonormal = fields @ transform
    check = (orthonormal.T @ orthonormal) / len(fields)
    return orthonormal, {
        "rank": int(np.count_nonzero(keep)),
        "gram_condition_number": float(values[-1] / values[keep][0]),
        "orthogonality_error": float(
            np.max(np.abs(check - np.eye(check.shape[0])))
        ),
    }


def comparison_metrics(
    learned_velocity: np.ndarray,
    gt_velocity: np.ndarray,
    prefixes: list[int],
) -> dict[str, Any]:
    learned_flat = flattened(learned_velocity)
    gt_flat = flattened(gt_velocity)
    gt_q, gt_diag = orthonormalize(gt_flat)
    prefix_rows = []
    prefix_cross: dict[int, np.ndarray] = {}
    for prefix in prefixes:
        learned_q, learned_diag = orthonormalize(learned_flat[:, :prefix])
        cross = (learned_q.T @ gt_q) / len(learned_flat)
        prefix_cross[prefix] = cross
        squared_capture = np.sum(cross * cross, axis=0)
        residuals = np.sqrt(np.maximum(0.0, 1.0 - squared_capture))
        singular = np.linalg.svd(cross, compute_uv=False)
        singular = np.clip(singular, 0.0, 1.0)
        principal_angles = np.degrees(np.arccos(singular))
        prefix_rows.append(
            {
                "learned_modes": prefix,
                "gt_modes": gt_velocity.shape[1],
                "gt_projection_residuals": residuals.tolist(),
                "mean_gt_projection_error": float(np.mean(residuals)),
                "rms_gt_subspace_error": float(
                    np.sqrt(
                        max(
                            0.0,
                            1.0
                            - float(np.sum(singular**2))
                            / gt_velocity.shape[1],
                        )
                    )
                ),
                "principal_angles_deg": principal_angles.tolist(),
                "maximum_principal_angle_deg": float(
                    np.max(principal_angles)
                ),
                "learned_orthonormalization": learned_diag,
            }
        )

    comparison_count = gt_velocity.shape[1]
    learned6_q, learned6_diag = orthonormalize(
        learned_flat[:, :comparison_count]
    )
    cross6 = (learned6_q.T @ gt_q) / len(learned_flat)
    learned_to_gt = np.sqrt(
        np.maximum(0.0, 1.0 - np.sum(cross6 * cross6, axis=1))
    )
    gt_to_learned = np.sqrt(
        np.maximum(0.0, 1.0 - np.sum(cross6 * cross6, axis=0))
    )
    row_indices, column_indices = linear_sum_assignment(-np.abs(cross6))
    matched = []
    for learned_index, gt_index in zip(row_indices, column_indices):
        correlation = float(abs(cross6[learned_index, gt_index]))
        matched.append(
            {
                "learned_mode": int(learned_index + 1),
                "gt_mode": int(gt_index + 1),
                "absolute_correlation": correlation,
                "MAC": correlation**2,
                "pair_projection_error": float(
                    math.sqrt(max(0.0, 1.0 - correlation**2))
                ),
                "sign_invariant_vector_error": float(
                    math.sqrt(max(0.0, 2.0 - 2.0 * correlation))
                ),
            }
        )
    singular6 = np.clip(np.linalg.svd(cross6, compute_uv=False), 0.0, 1.0)
    return {
        "gt_orthonormalization": gt_diag,
        "learned_first_gt_count_orthonormalization": learned6_diag,
        "cross_gram_first_six": cross6.tolist(),
        "MAC_first_six": (cross6 * cross6).tolist(),
        "learned_to_gt6_projection_errors": learned_to_gt.tolist(),
        "gt_to_learned6_projection_errors": gt_to_learned.tolist(),
        "matched_pairs": matched,
        "first_six_principal_cosines": singular6.tolist(),
        "first_six_principal_angles_deg": np.degrees(
            np.arccos(singular6)
        ).tolist(),
        "first_six_rms_subspace_error": float(
            np.sqrt(
                max(
                    0.0,
                    1.0 - float(np.sum(singular6**2)) / comparison_count,
                )
            )
        ),
        "prefix_comparison": prefix_rows,
        "_cross6_array": cross6,
    }


def convergence_metrics(
    learned: np.ndarray,
    gt: np.ndarray,
    sample_counts: list[int],
) -> list[dict[str, Any]]:
    rows = []
    comparison_count = gt.shape[1]
    for count in sample_counts:
        learned_q, _ = orthonormalize(
            flattened(learned[:count, :comparison_count])
        )
        gt_q, _ = orthonormalize(flattened(gt[:count]))
        cross = (learned_q.T @ gt_q) / (3 * count)
        singular = np.clip(np.linalg.svd(cross, compute_uv=False), 0.0, 1.0)
        rows.append(
            {
                "points": count,
                "rms_subspace_error": float(
                    np.sqrt(
                        max(
                            0.0,
                            1.0
                            - float(np.sum(singular**2))
                            / comparison_count,
                        )
                    )
                ),
                "maximum_principal_angle_deg": float(
                    np.max(np.degrees(np.arccos(singular)))
                ),
            }
        )
    return rows


def render_summary(
    path: Path,
    metrics: dict[str, Any],
    descriptors: list[dict[str, Any]],
) -> None:
    cross = np.asarray(metrics["_cross6_array"])
    mac = cross * cross
    matched = metrics["matched_pairs"]
    prefix = metrics["prefix_comparison"]
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))

    image = axes[0, 0].imshow(mac, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0, 0].set(
        xlabel="Laplacian GT mode",
        ylabel="learned mode",
        title="Modal assurance criterion (MAC)",
        xticks=np.arange(len(descriptors)),
        yticks=np.arange(len(descriptors)),
        xticklabels=np.arange(1, len(descriptors) + 1),
        yticklabels=np.arange(1, len(descriptors) + 1),
    )
    for i in range(mac.shape[0]):
        for j in range(mac.shape[1]):
            axes[0, 0].text(
                j,
                i,
                f"{mac[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mac[i, j] > 0.55 else "#222222",
            )
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.04)

    pair_labels = [
        f"L{item['learned_mode']}→G{item['gt_mode']}" for item in matched
    ]
    pair_errors = [item["pair_projection_error"] for item in matched]
    axes[0, 1].bar(pair_labels, pair_errors, color="#4C78A8")
    axes[0, 1].set(
        ylim=(0.0, 1.02),
        ylabel="relative velocity-L2 projection error",
        title="Optimal one-to-one matching",
    )
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].grid(axis="y", alpha=0.25)

    prefix_count = [item["learned_modes"] for item in prefix]
    prefix_error = [item["rms_gt_subspace_error"] for item in prefix]
    full_span = metrics.get("full_1024_mode_span")
    if full_span is not None:
        prefix_count.append(1024)
        prefix_error.append(full_span["rms_gt_subspace_error"])
    axes[1, 0].plot(
        prefix_count,
        prefix_error,
        marker="o",
        color="#D64B40",
        linewidth=2.0,
    )
    axes[1, 0].set(
        xlabel="number of learned modes used",
        ylabel="RMS GT-subspace projection error",
        ylim=(0.0, 1.02),
        title="Coverage of the six Laplacian GT modes",
    )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(prefix_count)
    axes[1, 0].set_xticklabels([str(value) for value in prefix_count])
    axes[1, 0].grid(alpha=0.25)

    angles = np.asarray(metrics["first_six_principal_angles_deg"])
    axes[1, 1].bar(
        np.arange(1, len(angles) + 1),
        angles,
        color="#72B7B2",
    )
    axes[1, 1].set(
        xlabel="principal direction",
        ylabel="angle (degrees)",
        ylim=(0.0, 90.0),
        title="Principal angles: learned-6 vs Laplacian-GT-6",
        xticks=np.arange(1, len(angles) + 1),
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "3D four-blade fan: learned basis vs cylindrical Laplacian GT modes",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)


def write_pair_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "learned_mode",
                "gt_mode",
                "absolute_correlation",
                "MAC",
                "pair_projection_error",
                "sign_invariant_vector_error",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics["matched_pairs"])


def main() -> None:
    args = parse_args()
    if args.gt_count < 1:
        raise ValueError("gt-count must be positive")
    if args.learned_count < args.gt_count:
        raise ValueError("learned-count must be at least gt-count")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state_path = args.output_dir / "pipeline_state.json"
    write_json(state_path, {"status": "running", "stage": "loading"})

    engine = InferenceEngine(args.checkpoint)
    cfg = engine.cfg
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    coefficients, learned_energies, peeling_diagnostics = learned_coefficients(
        engine,
        parameters,
        args.learned_count,
    )
    write_json(state_path, {"status": "running", "stage": "evaluating_fields"})
    points = fluid_points(
        cfg,
        args.points,
        parameters,
        cfg.seed + args.seed_offset,
    )
    descriptors = laplacian_mode_table(
        args.gt_count,
        cfg.xmax - cfg.xmin,
        cfg.farfield_radius,
    )
    gt = evaluate_laplacian_modes(points, cfg.xmin, descriptors)
    gt_validation = validate_laplacian_modes(
        points,
        cfg.xmin,
        descriptors,
    )
    gt_component_major = gt.transpose(2, 0, 1).reshape(
        3 * len(points),
        args.gt_count,
    )
    gt_q_component_major, _ = orthonormalize(gt_component_major)
    learned, full_span_metrics = evaluate_learned_modes(
        engine,
        points,
        parameters,
        coefficients,
        gt_q_component_major,
        args.chunk,
    )

    prefixes = sorted(
        {
            args.gt_count,
            min(16, args.learned_count),
            min(32, args.learned_count),
            args.learned_count,
        }
    )
    metrics = comparison_metrics(learned, gt, prefixes)
    metrics["full_1024_mode_span"] = full_span_metrics
    cross_array = np.asarray(metrics.pop("_cross6_array"))
    convergence_counts = sorted(
        {
            min(args.points, 4096),
            min(args.points, 8192),
            args.points,
        }
    )
    convergence = convergence_metrics(
        learned,
        gt,
        convergence_counts,
    )
    metrics["_cross6_array"] = cross_array
    render_summary(
        args.output_dir / "laplacian_gt_projection_error.png",
        metrics,
        descriptors,
    )
    metrics.pop("_cross6_array")
    write_pair_csv(args.output_dir / "matched_mode_errors.csv", metrics)
    np.savez_compressed(
        args.output_dir / "projection_data.npz",
        points=points,
        learned_velocity=learned[:, : args.gt_count].astype(np.float32),
        gt_velocity=gt.astype(np.float32),
        cross_gram_first_six=cross_array,
        learned_energies=learned_energies,
        gt_eigenvalues=np.asarray(
            [item["eigenvalue"] for item in descriptors]
        ),
    )

    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "definition": {
            "name": "closed_cylindrical_solenoidal_vector_laplacian",
            "families": {
                "TE": "curl(Psi e_x), radial Dirichlet Bessel roots",
                "TM": (
                    "curl curl(Psi e_x)/sqrt(lambda), "
                    "radial Neumann Bessel roots"
                ),
            },
            "axial_factor": "sin(n*pi*(x-xmin)/L)",
            "ordering": "ascending analytic vector-Laplacian eigenvalue",
            "projection_inner_product": (
                "uniform-volume sampled velocity L2 in cylinder minus rotor"
            ),
            "rotor_boundary": (
                "not enforced by analytic GT; rotor interior excluded, "
                "matching the 2D Laplacian-baseline convention"
            ),
            "outer_boundary": "closed-cylinder no-through vector harmonics",
            "learned_boundary": (
                "inlet/lateral/rotor double-zero envelope; natural outlet"
            ),
            "error": "relative velocity-L2 orthogonal projection residual",
            "subspace_metric": (
                "principal angles after independent sampled-L2 whitening"
            ),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "parameters_normalized": parameters.tolist(),
        "points": args.points,
        "gt_count": args.gt_count,
        "learned_count": args.learned_count,
        "peeling_diagnostics": peeling_diagnostics,
        "learned_rayleigh_energies": learned_energies.tolist(),
        "laplacian_gt_modes": descriptors,
        "laplacian_gt_validation": gt_validation,
        "metrics": metrics,
        "quadrature_convergence": convergence,
        "outputs": {
            "figure": str(
                (args.output_dir / "laplacian_gt_projection_error.png").resolve()
            ),
            "matched_errors_csv": str(
                (args.output_dir / "matched_mode_errors.csv").resolve()
            ),
            "projection_data": str(
                (args.output_dir / "projection_data.npz").resolve()
            ),
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
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
