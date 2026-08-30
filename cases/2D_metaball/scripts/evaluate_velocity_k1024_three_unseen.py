"""Strict K=1024 velocity-Rayleigh reconstruction on three unseen geometries.

The learned span is constructed on each target geometry with the same dynamic
raw velocity-Rayleigh ordering and strict single velocity-L2 peel used during
training.  Learned and finite-difference Laplacian spans are then evaluated on
one common 128 x 128 grid, receive the same invertible discrete velocity-mass
whitening, fit the same GT velocity samples, and are scored on the same held-out
nodes with the same finite-difference observables.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import scipy.linalg

from compare_three_gt_eigenfluid_vs_laplacian import (
    fit_velocity,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
    serializable_gt,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_connected_dirichlet_psi_basis import load_model
from metaball_runtime import (
    backend,
    excluded_area,
    fluid_mask,
    geometry_from_token,
    metaball_fluid_points,
)


F32 = np.float32
F64 = np.float64
CASE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    CASE_ROOT
    / "results/siren_k1024_connected_velocity_rayleigh_omega160_onepass_e2000"
)
DEFAULT_BEST = RESULT_ROOT / "run_001/checkpoints/hardcoded_best_model.npz"
DEFAULT_FINAL = RESULT_ROOT / "run_001/checkpoints/hardcoded_sss_frozen.npz"
DEFAULT_OUTPUT = RESULT_ROOT / "evaluation/three_unseen_k1024_128"


def _raw_velocity_data(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    token: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return flattened grad(psi) and compact Hessian in network order."""
    count, modes = len(points), int(cfg.modes)
    gradient = cp.empty((2 * count, modes), dtype=backend.F32)
    hessian = cp.empty((3 * count, modes), dtype=backend.F32)
    for mode, field, _ in model.forward_stream(cfg, points, token, 0.0, False):
        network = int(mode.mode_index)
        gradient[:, network] = field.grad[:, 0].reshape(-1)
        hessian[:, network] = field.hess[:, 0].reshape(-1)
    return gradient, hessian


def _raw_network_psi(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    token: float,
) -> cp.ndarray:
    values = cp.empty((len(points), int(cfg.modes)), dtype=backend.F32)
    for mode, field, _ in model.forward_stream(cfg, points, token, 0.0, False):
        values[:, int(mode.mode_index)] = field.value[:, 0]
    return values


def velocity_peeling_transform(
    cfg: Any,
    model: Any,
    reference_points: np.ndarray,
    token: float,
) -> tuple[cp.ndarray, dict[str, Any]]:
    """Reproduce training-time dynamic sorting and one velocity-L2 peel."""
    if str(cfg.basis_objective) != "velocity_rayleigh":
        raise ValueError("checkpoint is not a velocity-Rayleigh model")
    if int(cfg.reorthogonalization_passes) != 1:
        raise ValueError("evaluation requires exactly one peeling pass")

    gradient, hessian = _raw_velocity_data(cfg, model, reference_points, token)
    count, modes = len(reference_points), int(cfg.modes)
    fluid_area = float(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - excluded_area(cfg, token)
    )
    weight = cp.asarray(F32(fluid_area / count), dtype=backend.F32)
    hessian_metric = cp.tile(cp.asarray((1.0, 2.0, 1.0), dtype=backend.F32), count)
    raw_mass = weight * cp.sum(gradient * gradient, axis=0, dtype=backend.F32)
    raw_stiffness = weight * cp.sum(
        hessian_metric[:, None] * hessian * hessian,
        axis=0,
        dtype=backend.F32,
    )
    raw_rayleigh = raw_stiffness / raw_mass
    if str(cfg.causal_sorting) == "geometry_rayleigh":
        order = cp.argsort(raw_rayleigh, kind="stable")
    else:
        order = cp.arange(modes, dtype=cp.int64)

    sorted_gradient = gradient[:, order]
    sorted_hessian = hessian[:, order]
    basis_gradient = cp.empty_like(sorted_gradient)
    basis_hessian = cp.empty_like(sorted_hessian)
    transform = cp.zeros((modes, modes), dtype=backend.F32)
    relative_masses = cp.empty(modes, dtype=backend.F32)
    overlaps = cp.empty(modes, dtype=backend.F32)
    peeled_rayleigh = cp.empty(modes, dtype=backend.F32)
    mass_floor = F32(float(cfg.mass_floor))

    for rank in range(modes):
        residual_gradient = sorted_gradient[:, rank].copy()
        residual_hessian = sorted_hessian[:, rank].copy()
        coefficients = cp.zeros(modes, dtype=backend.F32)
        coefficients[order[rank]] = F32(1.0)
        if rank:
            previous_gradient = basis_gradient[:, :rank]
            previous_hessian = basis_hessian[:, :rank]
            previous_transform = transform[:, :rank]
            projection = weight * (previous_gradient.T @ residual_gradient)
            residual_gradient -= previous_gradient @ projection
            residual_hessian -= previous_hessian @ projection
            coefficients -= previous_transform @ projection
        else:
            previous_gradient = None

        mass = weight * cp.dot(residual_gradient, residual_gradient)
        stiffness = weight * cp.dot(
            hessian_metric * residual_hessian, residual_hessian
        )
        safe_mass = cp.maximum(mass, mass_floor)
        inverse_norm = cp.reciprocal(cp.sqrt(safe_mass))
        normalized_gradient = residual_gradient * inverse_norm
        basis_gradient[:, rank] = normalized_gradient
        basis_hessian[:, rank] = residual_hessian * inverse_norm
        transform[:, rank] = coefficients * inverse_norm
        relative_masses[rank] = mass / raw_mass[order[rank]]
        normalized_mass = weight * cp.dot(normalized_gradient, normalized_gradient)
        overlap = cp.abs(normalized_mass - F32(1.0))
        if rank:
            overlap = cp.maximum(
                overlap,
                cp.max(cp.abs(weight * (previous_gradient.T @ normalized_gradient))),
            )
        overlaps[rank] = overlap
        peeled_rayleigh[rank] = stiffness / safe_mass

    diagnostic_rows = cp.asnumpy(
        cp.stack((relative_masses, overlaps, peeled_rayleigh, order.astype(backend.F32)))
    )
    relative_host, overlap_host, rayleigh_host, order_float = diagnostic_rows
    order_host = order_float.astype(np.int64)
    relative_floor = float(cfg.peeling_relative_mass_floor)
    overlap_tolerance = float(cfg.peeling_overlap_tolerance)
    rejected = (
        ~np.isfinite(relative_host)
        | ~np.isfinite(overlap_host)
        | ~np.isfinite(rayleigh_host)
        | (relative_host < relative_floor)
        | (overlap_host > overlap_tolerance)
    )
    if np.any(rejected):
        rank = int(np.flatnonzero(rejected)[0])
        raise FloatingPointError(
            f"unseen token {int(token)} rejects network mode "
            f"{int(order_host[rank]) + 1} at causal rank {rank + 1}: "
            f"relative_mass={relative_host[rank]:.8e}, "
            f"overlap={overlap_host[rank]:.8e}"
        )
    diagnostics = {
        "reference_points": count,
        "fluid_area": fluid_area,
        "causal_sorting": str(cfg.causal_sorting),
        "peeling_passes": 1,
        "peeled_rank": modes,
        "min_relative_residual_mass": float(np.min(relative_host)),
        "max_post_peel_overlap": float(np.max(overlap_host)),
        "rayleigh_min": float(np.min(rayleigh_host)),
        "rayleigh_mean": float(np.mean(rayleigh_host, dtype=np.float64)),
        "rayleigh_max": float(np.max(rayleigh_host)),
        "causal_order_1based": (order_host + 1).tolist(),
    }
    del gradient, hessian, sorted_gradient, sorted_hessian
    del basis_gradient, basis_hessian, relative_masses, overlaps, peeled_rayleigh
    return transform, diagnostics


def common_velocity_mass_whitening(
    psi: np.ndarray,
    valid: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Apply an invertible full-grid velocity-mass whitening to one span."""
    velocity, omega = matched_grid_observables(psi, valid, spacing)
    modes = psi.shape[1]
    design = velocity.reshape(-1, modes).astype(F64, copy=False)
    mass = (spacing * spacing) * (design.T @ design)
    mass = 0.5 * (mass + mass.T)
    eigenvalues = np.linalg.eigvalsh(mass)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise np.linalg.LinAlgError(
            f"discrete velocity mass is not positive definite: "
            f"lambda_min={eigenvalues[0]:.8e}"
        )
    lower = scipy.linalg.cholesky(mass, lower=True, check_finite=False)
    transform = scipy.linalg.solve_triangular(
        lower.T,
        np.eye(modes, dtype=F64),
        lower=False,
        check_finite=False,
    )
    whitened_psi = (psi.astype(F64, copy=False) @ transform).astype(F32)
    whitened_velocity, whitened_omega = matched_grid_observables(
        whitened_psi, valid, spacing
    )
    whitened_design = whitened_velocity.reshape(-1, modes).astype(F64, copy=False)
    gram = (spacing * spacing) * (whitened_design.T @ whitened_design)
    diagnostics = {
        "prewhitening_mass_eigenvalue_min": float(eigenvalues[0]),
        "prewhitening_mass_eigenvalue_max": float(eigenvalues[-1]),
        "prewhitening_mass_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "postwhitening_mass_orthogonality_max_abs": float(
            np.max(np.abs(gram - np.eye(modes)))
        ),
    }
    return whitened_psi, whitened_velocity, whitened_omega, diagnostics


def learned_span(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    valid: np.ndarray,
    spacing: float,
    token: float,
    peeling_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    reference = metaball_fluid_points(
        cfg, peeling_points, token, 0.0, seed
    )
    transform, peeling = velocity_peeling_transform(
        cfg, model, reference, token
    )
    raw_psi = _raw_network_psi(cfg, model, points, token)
    psi = cp.asnumpy(raw_psi @ transform).astype(F32)
    del raw_psi, transform
    cp.get_default_memory_pool().free_all_blocks()
    psi, velocity, omega, whitening = common_velocity_mass_whitening(
        psi, valid, spacing
    )
    return psi, velocity, omega, {
        "peeling": peeling,
        "discrete_velocity_mass_whitening": whitening,
    }


def assert_compatible(first_cfg: Any, second_cfg: Any) -> None:
    keys = (
        "modes",
        "basis_objective",
        "geometry_mode",
        "metaball_connectivity",
        "causal_sorting",
        "reorthogonalization_passes",
        "peeling_relative_mass_floor",
        "peeling_overlap_tolerance",
        "siren_first_omega0",
        "siren_hidden_omega0",
    )
    mismatches = {
        key: (getattr(first_cfg, key), getattr(second_cfg, key))
        for key in keys
        if getattr(first_cfg, key) != getattr(second_cfg, key)
    }
    if mismatches:
        raise ValueError(f"best/final checkpoint protocol mismatch: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-checkpoint", type=Path, default=DEFAULT_BEST)
    parser.add_argument("--final-checkpoint", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[4_000_001, 4_000_002, 4_000_003]
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.resolution != 128:
        raise ValueError("this strict audit is fixed to a 128 x 128 grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_metadata, best_cfg, best_model = load_model(args.best_checkpoint)
    final_metadata, final_cfg, final_model = load_model(args.final_checkpoint)
    assert_compatible(best_cfg, final_cfg)
    if int(best_cfg.modes) != 1024:
        raise ValueError(f"expected K=1024, found K={best_cfg.modes}")
    if str(best_cfg.basis_objective) != "velocity_rayleigh":
        raise ValueError("expected velocity-Rayleigh checkpoints")

    models = (
        ("Learned-best-e800", best_metadata, best_cfg, best_model),
        ("Learned-final-e2000", final_metadata, final_cfg, final_model),
    )
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for case_index, token_integer in enumerate(args.tokens):
        case_started = time.perf_counter()
        token = float(token_integer)
        valid = fluid_mask(best_cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        points = all_points[valid.ravel()]
        if len(points) <= int(best_cfg.modes) + 1:
            raise ValueError(f"insufficient fluid nodes for token {token_integer}")
        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]

        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(
            best_cfg, token, gt_seed, args.vortices
        )
        gt_psi, _, _, gt_normalization = ground_truth_fields(
            best_cfg, points, token, gt_parameters
        )
        gt_velocity, gt_omega = matched_grid_observables(
            gt_psi, valid, spacing
        )

        case_results: dict[str, Any] = {}
        for method_index, (label, metadata, cfg, model) in enumerate(models):
            method_started = time.perf_counter()
            psi, velocity, omega, diagnostics = learned_span(
                cfg,
                model,
                points,
                valid,
                spacing,
                token,
                args.peeling_points,
                args.seed + 200_003 * (case_index + 1),
            )
            reconstruction = fit_velocity(
                psi,
                velocity,
                omega,
                gt_psi,
                gt_velocity,
                gt_omega,
                fit_indices,
                test_indices,
                args.ridge_relative,
            )
            result = {
                "checkpoint": str(
                    (args.best_checkpoint if method_index == 0 else args.final_checkpoint).resolve()
                ),
                "checkpoint_epoch": int(metadata["epoch"]),
                "metrics": reconstruction["metrics"],
                "diagnostics": diagnostics,
                "seconds": time.perf_counter() - method_started,
            }
            case_results[label] = result
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "method": label,
                    **reconstruction["metrics"],
                }
            )
            del psi, velocity, omega, reconstruction

        laplacian_started = time.perf_counter()
        laplacian_values, laplacian_psi, _, _ = finite_difference_laplacian(
            valid, spacing, int(best_cfg.modes)
        )
        laplacian_psi, laplacian_velocity, laplacian_omega, laplacian_whitening = (
            common_velocity_mass_whitening(laplacian_psi, valid, spacing)
        )
        laplacian_reconstruction = fit_velocity(
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
        laplacian_result = {
            "metrics": laplacian_reconstruction["metrics"],
            "eigenvalue_min": float(laplacian_values[0]),
            "eigenvalue_max": float(laplacian_values[-1]),
            "discrete_velocity_mass_whitening": laplacian_whitening,
            "seconds": time.perf_counter() - laplacian_started,
        }
        case_results["Laplacian"] = laplacian_result
        rows.append(
            {
                "case": case_index + 1,
                "geometry_token": token_integer,
                "method": "Laplacian",
                **laplacian_reconstruction["metrics"],
            }
        )

        cases.append(
            {
                "case": case_index + 1,
                "geometry_token": token_integer,
                "geometry_parameters": geometry_from_token(best_cfg, token).tolist(),
                "fluid_nodes": int(len(points)),
                "fit_nodes": int(len(fit_indices)),
                "held_out_test_nodes": int(len(test_indices)),
                "gt_seed": gt_seed,
                "gt_velocity_rms_before_normalization": gt_normalization,
                "gt_parameters": serializable_gt(gt_parameters),
                "results": case_results,
                "seconds": time.perf_counter() - case_started,
            }
        )
        print(
            json.dumps(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "test_errors": {
                        label: result["metrics"]
                        for label, result in case_results.items()
                    },
                }
            ),
            flush=True,
        )
        del laplacian_psi, laplacian_velocity, laplacian_omega
        del laplacian_reconstruction

    metric_keys = (
        "test_psi_relative_l2",
        "test_velocity_relative_l2",
        "test_vorticity_relative_l2",
    )
    labels = ("Learned-best-e800", "Learned-final-e2000", "Laplacian")
    means = {
        label: {
            key: float(
                np.mean(
                    [
                        case["results"][label]["metrics"][key]
                        for case in cases
                    ]
                )
            )
            for key in metric_keys
        }
        for label in labels
    }
    learned_best_wins = {
        key: int(
            sum(
                case["results"]["Learned-best-e800"]["metrics"][key]
                < case["results"]["Laplacian"]["metrics"][key]
                for case in cases
            )
        )
        for key in metric_keys
    }
    summary = {
        "status": "complete",
        "case": "2D connected three-circle smooth metaball",
        "geometry_status": "three independently generated unseen cases",
        "geometry_tokens": args.tokens,
        "modes": int(best_cfg.modes),
        "grid_resolution": args.resolution,
        "peeling_points": args.peeling_points,
        "best_checkpoint_epoch": int(best_metadata["epoch"]),
        "final_checkpoint_epoch": int(final_metadata["epoch"]),
        "fit": "velocity least squares on 70% of valid nodes",
        "test": "relative L2 on disjoint held-out 30% valid nodes",
        "gt": (
            "seven deterministic Gaussian vortical structures times the exact "
            "hard Dirichlet geometry envelope"
        ),
        "fairness_contract": {
            "same_geometry_and_mask": True,
            "same_gt_and_fit_test_split": True,
            "same_mode_count": True,
            "same_boundary_condition": "psi=0 on outer and obstacle boundaries",
            "same_velocity_operator": "zero-extended centered difference",
            "same_vorticity_operator": "five-point positive -Delta",
            "same_span_postprocessing": (
                "invertible full-grid discrete velocity-mass whitening"
            ),
            "learned_construction": (
                "geometry-Rayleigh causal sorting plus exactly one velocity-L2 peel"
            ),
            "laplacian_construction": (
                "geometry-specific five-point FD Dirichlet eigensolve"
            ),
        },
        "mean_test_errors": means,
        "learned_best_wins_vs_laplacian_out_of_three": learned_best_wins,
        "cases": cases,
        "total_seconds": time.perf_counter() - total_started,
    }

    csv_path = args.output_dir / "three_unseen_reconstruction_metrics.csv"
    json_path = args.output_dir / "three_unseen_reconstruction_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    print(json.dumps({"summary": str(json_path), "means": means}, indent=2), flush=True)


if __name__ == "__main__":
    main()
