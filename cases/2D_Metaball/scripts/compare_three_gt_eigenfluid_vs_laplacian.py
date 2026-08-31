"""Compare K=64 Eigenfluid and Laplacian reconstruction on three unseen flows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np
import scipy.linalg

from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_connected_dirichlet_psi_basis import (
    load_model,
    numerical_ritz_transform,
    peeled_transform,
    raw_fields,
)
from metaball_runtime import (
    backend,
    excluded_area,
    fluid_mask,
    geometry_from_token,
    metaball_envelope_jet,
    metaball_fluid_points,
)


F32 = np.float32


def matched_grid_observables(
    psi: np.ndarray,
    valid: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one common zero-Dirichlet grid operator to any psi field.

    Fair reconstruction comparisons must not use analytic derivatives for the
    learned field while using finite differences for the Laplacian field.
    Both methods (and the GT) are therefore embedded into the same staircase
    mask with zero values outside, differentiated by the same centered stencil,
    and acted on by the exact same five-point positive ``-Delta`` operator.
    """
    values = np.asarray(psi, dtype=F32)
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != int(np.count_nonzero(valid)):
        raise ValueError("psi must contain one value per valid grid node")
    ny, nx = valid.shape
    grid = np.zeros((ny, nx, values.shape[1]), dtype=F32)
    grid[valid] = values
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="constant")
    center = padded[1:-1, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    down = padded[:-2, 1:-1]
    up = padded[2:, 1:-1]
    inverse_2h = F32(0.5 / spacing)
    inverse_h2 = F32(1.0 / (spacing * spacing))
    dpsi_dx = (right - left) * inverse_2h
    dpsi_dy = (up - down) * inverse_2h
    velocity = np.stack((dpsi_dy[valid], -dpsi_dx[valid]), axis=1)
    omega = ((F32(4.0) * center - left - right - down - up) * inverse_h2)[valid]
    if squeeze:
        return velocity[:, :, 0], omega[:, 0]
    return velocity, omega


def matched_grid_ritz(
    psi: np.ndarray,
    valid: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rotate/normalize a fixed span in the common discrete Laplacian metric."""
    values = np.asarray(psi, dtype=np.float64)
    _, omega = matched_grid_observables(values.astype(F32), valid, spacing)
    weight = float(spacing * spacing)
    mass = weight * (values.T @ values)
    stiffness = weight * (
        values.T @ omega.astype(np.float64, copy=False)
    )
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, transform = scipy.linalg.eigh(
        stiffness, mass, check_finite=False, driver="gvd"
    )
    rotated = (values @ transform).astype(F32)
    diagnostics = {
        "mass_orthogonality_max_abs": float(
            np.max(np.abs(transform.T @ mass @ transform - np.eye(values.shape[1])))
        ),
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "first16_eigenvalues": [float(value) for value in eigenvalues[:16]],
    }
    return rotated, eigenvalues, diagnostics


def _exp_jet(source: backend.Jet3) -> backend.Jet3:
    value = cp.exp(source.value)
    return backend.unary(source, value, value, value, value)


def generate_gt_parameters(
    cfg: SimpleNamespace,
    token: float,
    seed: int,
    vortices: int,
) -> dict[str, Any]:
    centers = metaball_fluid_points(cfg, vortices, token, 0.0, seed + 101)
    rng = np.random.default_rng(seed)
    sigmas = rng.uniform(0.075, 0.165, size=vortices).astype(F32)
    amplitudes = rng.normal(0.0, 1.0, size=vortices).astype(F32)
    amplitudes -= F32(0.35 * float(np.mean(amplitudes)))
    return {
        "centers": centers,
        "sigmas": sigmas,
        "amplitudes": amplitudes,
    }


def ground_truth_fields(
    cfg: SimpleNamespace,
    points: np.ndarray,
    token: float,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build an independent divergence-free, hard-slip GT from Gaussian vortices."""
    x, y = backend.xy_jets(points, backend.F32)
    envelope = metaball_envelope_jet(cfg, x, y, token, 0.0)
    source = backend.Jet3.constant(F32(0.0), x.value.shape, backend.F32)
    for center, sigma, amplitude in zip(
        parameters["centers"], parameters["sigmas"], parameters["amplitudes"]
    ):
        dx = x - float(center[0])
        dy = y - float(center[1])
        exponent = -F32(0.5) * (dx.square() + dy.square()) / F32(float(sigma) ** 2)
        source = source + F32(amplitude) * _exp_jet(exponent)
    field = envelope * source
    psi = field.value[:, 0]
    velocity = cp.stack((field.grad[:, 0, 1], -field.grad[:, 0, 0]), axis=1)
    omega = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
    velocity_rms = cp.sqrt(cp.mean(cp.sum(velocity * velocity, axis=1)))
    normalization = float(velocity_rms.get())
    if not math.isfinite(normalization) or normalization <= 1.0e-12:
        raise FloatingPointError("degenerate generated GT velocity")
    inverse = F32(1.0 / normalization)
    result = (
        cp.asnumpy(psi * inverse).astype(F32),
        cp.asnumpy(velocity * inverse).astype(F32),
        cp.asnumpy(omega * inverse).astype(F32),
        normalization,
    )
    del x, y, envelope, source, field, psi, velocity, omega, velocity_rms
    cp.get_default_memory_pool().free_all_blocks()
    return result


def full_network_fields(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    token: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    count, modes = len(points), int(cfg.modes)
    value = cp.empty((count, modes), dtype=backend.F32)
    velocity = cp.empty((count, 2, modes), dtype=backend.F32)
    omega = cp.empty((count, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, token, 0.0, False)
    ):
        value[:, index] = field.value[:, 0]
        velocity[:, 0, index] = field.grad[:, 0, 1]
        velocity[:, 1, index] = -field.grad[:, 0, 0]
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
    return value, velocity, omega


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction64 = prediction.astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    return float(
        np.linalg.norm(prediction64 - target64)
        / max(float(np.linalg.norm(target64)), 1.0e-30)
    )


def fit_velocity(
    psi: np.ndarray,
    velocity: np.ndarray,
    omega: np.ndarray,
    gt_psi: np.ndarray,
    gt_velocity: np.ndarray,
    gt_omega: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    ridge_relative: float,
) -> dict[str, Any]:
    modes = psi.shape[1]
    design = velocity[fit_indices].reshape(-1, modes).astype(np.float64, copy=False)
    target = gt_velocity[fit_indices].reshape(-1).astype(np.float64, copy=False)
    gram = (design.T @ design) / len(design)
    rhs = (design.T @ target) / len(design)
    ridge = ridge_relative * float(np.trace(gram) / modes)
    coefficients = scipy.linalg.solve(
        gram + ridge * np.eye(modes), rhs, assume_a="pos", check_finite=False
    )
    reconstruction = {
        "psi": (psi @ coefficients).astype(F32),
        "velocity": np.einsum(
            "ncm,m->nc", velocity, coefficients, optimize=True
        ).astype(F32),
        "omega": (omega @ coefficients).astype(F32),
    }
    eigenvalues = np.linalg.eigvalsh(gram)
    threshold = max(float(eigenvalues[-1]) * 1.0e-8, 1.0e-20)
    metrics: dict[str, Any] = {
        "fit_psi_relative_l2": relative_l2(
            reconstruction["psi"][fit_indices], gt_psi[fit_indices]
        ),
        "fit_velocity_relative_l2": relative_l2(
            reconstruction["velocity"][fit_indices], gt_velocity[fit_indices]
        ),
        "fit_vorticity_relative_l2": relative_l2(
            reconstruction["omega"][fit_indices], gt_omega[fit_indices]
        ),
        "test_psi_relative_l2": relative_l2(
            reconstruction["psi"][test_indices], gt_psi[test_indices]
        ),
        "test_velocity_relative_l2": relative_l2(
            reconstruction["velocity"][test_indices], gt_velocity[test_indices]
        ),
        "test_vorticity_relative_l2": relative_l2(
            reconstruction["omega"][test_indices], gt_omega[test_indices]
        ),
        "fit_velocity_gram_effective_rank": int(
            np.count_nonzero(eigenvalues > threshold)
        ),
        "fit_velocity_gram_condition_number": float(
            eigenvalues[-1] / max(float(eigenvalues[0]), 1.0e-30)
        ),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "ridge": float(ridge),
    }
    return {"metrics": metrics, "coefficients": coefficients, **reconstruction}


def eigenfluid_basis(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    token: float,
    peeling_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    reference = metaball_fluid_points(
        cfg, peeling_points, token, 0.0, seed + 701
    )
    reference_value, reference_gradient = raw_fields(cfg, model, reference, token)
    box_area = float((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin))
    fluid_area = box_area - excluded_area(cfg, token)
    peeling, peeling_diagnostics = peeled_transform(
        cfg, reference_value, reference_gradient, fluid_area
    )
    total, eigenvalues, ritz_diagnostics = numerical_ritz_transform(
        reference_value, reference_gradient, peeling, fluid_area
    )
    raw_psi, raw_velocity, raw_omega = full_network_fields(cfg, model, points, token)
    psi = cp.asnumpy(raw_psi @ total).astype(F32)
    velocity = cp.asnumpy(
        cp.einsum("ncm,mk->nck", raw_velocity, total, optimize=True)
    ).astype(F32)
    omega = cp.asnumpy(raw_omega @ total).astype(F32)
    del reference_value, reference_gradient, peeling, total
    del raw_psi, raw_velocity, raw_omega
    cp.get_default_memory_pool().free_all_blocks()
    diagnostics = {
        **peeling_diagnostics,
        **ritz_diagnostics,
        "ritz_lambda_min": float(eigenvalues[0]),
        "ritz_lambda_max": float(eigenvalues[-1]),
    }
    return psi, velocity, omega, diagnostics, {"fluid_area": fluid_area}


def serializable_gt(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: np.asarray(value).tolist()
        for key, value in parameters.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[4_000_001, 4_000_002, 4_000_003]
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, model = load_model(args.checkpoint)
    if int(cfg.modes) != 64:
        raise ValueError(f"expected K=64 checkpoint, found K={cfg.modes}")
    if str(cfg.basis_objective) != "streamfunction_dirichlet_rayleigh":
        raise ValueError("expected streamfunction Dirichlet checkpoint")

    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for case_index, token_integer in enumerate(args.tokens):
        token = float(token_integer)
        valid_grid = fluid_mask(cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        points = all_points[valid_grid.ravel()]
        if len(points) < int(cfg.modes) + 1024:
            raise ValueError(f"insufficient fluid nodes for token {token_integer}")
        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]

        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(
            cfg, token, gt_seed, args.vortices
        )
        gt_psi, gt_velocity_analytic, gt_omega_analytic, gt_normalization = ground_truth_fields(
            cfg, points, token, gt_parameters
        )
        gt_velocity, gt_omega = matched_grid_observables(
            gt_psi, valid_grid, spacing
        )

        eigen_psi, eigen_velocity_analytic, eigen_omega_analytic, peeling, area = eigenfluid_basis(
            cfg,
            model,
            points,
            token,
            args.peeling_points,
            args.seed + 200_003 * (case_index + 1),
        )
        eigen_psi, _, eigen_grid_ritz = matched_grid_ritz(
            eigen_psi, valid_grid, spacing
        )
        eigen_velocity, eigen_omega = matched_grid_observables(
            eigen_psi, valid_grid, spacing
        )
        eigenfluid = fit_velocity(
            eigen_psi,
            eigen_velocity,
            eigen_omega,
            gt_psi,
            gt_velocity,
            gt_omega,
            fit_indices,
            test_indices,
            args.ridge_relative,
        )

        laplacian_values, lap_psi, lap_omega, lap_velocity = (
            finite_difference_laplacian(valid_grid, spacing, int(cfg.modes))
        )
        lap_velocity_matched, lap_omega_matched = matched_grid_observables(
            lap_psi, valid_grid, spacing
        )
        lap_velocity_consistency = float(
            np.max(np.abs(lap_velocity_matched - lap_velocity))
        )
        lap_velocity_consistency_relative = float(
            lap_velocity_consistency
            / max(float(np.max(np.abs(lap_velocity))), 1.0e-30)
        )
        lap_omega_consistency = float(
            np.linalg.norm(
                lap_omega_matched.astype(np.float64)
                - lap_omega.astype(np.float64)
            )
            / max(float(np.linalg.norm(lap_omega.astype(np.float64))), 1.0e-30)
        )
        # ``finite_difference_laplacian`` forms its legacy diagnostics with
        # FP32 division/multiplication while ``matched_grid_observables`` uses
        # an algebraically identical vectorized stencil.  Their absolute
        # roundoff grows like 1/h, so use a scale-free audit here.  Every
        # reported fit below still uses only the common matched-grid fields.
        if (
            lap_velocity_consistency_relative > 2.0e-6
            or lap_omega_consistency > 2.0e-4
        ):
            raise RuntimeError(
                "shared observable operator is inconsistent with the FD "
                "Laplacian: velocity_relative="
                f"{lap_velocity_consistency_relative:.3e}, "
                f"vorticity={lap_omega_consistency:.3e}"
            )
        laplacian = fit_velocity(
            lap_psi,
            lap_velocity_matched,
            lap_omega_matched,
            gt_psi,
            gt_velocity,
            gt_omega,
            fit_indices,
            test_indices,
            args.ridge_relative,
        )

        for method, result in (("Eigenfluid", eigenfluid), ("Laplacian", laplacian)):
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "method": method,
                    **result["metrics"],
                }
            )
        cases.append(
            {
                "case": case_index + 1,
                "geometry_token": token_integer,
                "geometry_parameters": geometry_from_token(cfg, token).tolist(),
                "fluid_nodes": int(len(points)),
                "fit_nodes": int(len(fit_indices)),
                "held_out_test_nodes": int(len(test_indices)),
                "fluid_area": float(area["fluid_area"]),
                "gt_seed": gt_seed,
                "gt_velocity_rms_before_normalization": gt_normalization,
                "gt_parameters": serializable_gt(gt_parameters),
                "eigenfluid_peeling": peeling,
                "eigenfluid_matched_grid_ritz": eigen_grid_ritz,
                "laplacian": {
                    "eigenvalue_min": float(laplacian_values[0]),
                    "eigenvalue_max": float(laplacian_values[-1]),
                    "first16_eigenvalues": [
                        float(value) for value in laplacian_values[:16]
                    ],
                    "recomputed_for_geometry": True,
                    "matched_velocity_operator_max_abs": lap_velocity_consistency,
                    "matched_velocity_operator_relative_max": (
                        lap_velocity_consistency_relative
                    ),
                    "matched_vorticity_operator_relative_l2": lap_omega_consistency,
                },
                "fairness_audit": {
                    "same_geometry_token_and_parameters": True,
                    "same_fluid_mask": True,
                    "same_outer_and_obstacle_zero_level_sets": True,
                    "same_boundary_condition": "psi=0 on every boundary component",
                    "same_grid_nodes": True,
                    "same_mode_count": int(cfg.modes),
                    "same_streamfunction_mass_normalization": True,
                    "learned_span_only_rotated_not_enlarged": True,
                    "same_velocity_operator": "zero-extended centered difference",
                    "same_vorticity_operator": "five-point positive -Delta",
                    "same_fit_test_split": True,
                    "same_ridge": float(args.ridge_relative),
                },
                "metrics": {
                    "Eigenfluid": eigenfluid["metrics"],
                    "Laplacian": laplacian["metrics"],
                },
            }
        )

    means: dict[str, dict[str, float]] = {}
    for method in ("Eigenfluid", "Laplacian"):
        selected = [row for row in rows if row["method"] == method]
        means[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in (
                "test_psi_relative_l2",
                "test_velocity_relative_l2",
                "test_vorticity_relative_l2",
            )
        }
    winner_counts = {
        metric: sum(
            case["metrics"]["Eigenfluid"][metric]
            < case["metrics"]["Laplacian"][metric]
            for case in cases
        )
        for metric in (
            "test_psi_relative_l2",
            "test_velocity_relative_l2",
            "test_vorticity_relative_l2",
        )
    }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "modes": int(cfg.modes),
        "geometry_tokens": args.tokens,
        "grid_resolution": args.resolution,
        "peeling_points": args.peeling_points,
        "gt_definition": (
            "seven independent Gaussian vortical structures multiplied by the "
            "geometry hard Dirichlet streamfunction envelope; the analytic "
            "velocity is used only for the common amplitude normalization"
        ),
        "coefficient_fit": (
            "matched velocity least squares on 70% grid nodes with relative "
            f"ridge {args.ridge_relative:.3e}"
        ),
        "evaluation": (
            "relative L2 on the disjoint held-out 30% grid nodes; GT, learned, "
            "and Laplacian observables all use identical zero-extended centered "
            "velocity differences and the identical five-point Dirichlet -Delta"
        ),
        "eigenfluid_basis": (
            "geometry-Rayleigh causal sorting, two-pass streamfunction peeling, "
            "numerical generalized Ritz rotation; no raw-rank metric"
        ),
        "laplacian_basis": (
            "geometry-specific five-point finite-difference Dirichlet "
            "streamfunction eigenbasis"
        ),
        "comparison_status": "strict_matched_grid_fairness",
        "fairness_contract": {
            "geometry": "same cfg, geometry token, soft-min obstacle, and p=8 outer mask",
            "boundary": "psi=0 on every outer/obstacle boundary component for both methods",
            "discretization": "same valid nodes and same discrete derivative operators for all reported fields",
            "normalization": "same discrete streamfunction L2 mass; learned span receives only an invertible grid-Ritz rotation",
            "fit": "same K, GT, node split, ridge, coefficient objective, and test metric",
            "only_intended_difference": "construction of the K-dimensional streamfunction span",
        },
        "mean_test_errors": means,
        "eigenfluid_wins_out_of_three": winner_counts,
        "cases": cases,
    }
    csv_path = args.output_dir / "three_gt_reconstruction_metrics.csv"
    json_path = args.output_dir / "three_gt_reconstruction_summary.json"
    spectrum_csv_path = args.output_dir / "peeled_ritz_first16_spectrum.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with spectrum_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["case", "geometry_token", "method", "mode", "eigenvalue"],
        )
        writer.writeheader()
        for case in cases:
            spectra = {
                "Eigenfluid_peeled_grid_Ritz": case[
                    "eigenfluid_matched_grid_ritz"
                ]["first16_eigenvalues"],
                "Laplacian": case["laplacian"]["first16_eigenvalues"],
            }
            for method, values in spectra.items():
                for mode, value in enumerate(values, start=1):
                    writer.writerow(
                        {
                            "case": case["case"],
                            "geometry_token": case["geometry_token"],
                            "method": method,
                            "mode": mode,
                            "eigenvalue": value,
                        }
                    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
