"""Audit numerical rank, orthogonality, and divergence after peeling.

The audit uses the same continuous validation points, geometry-Rayleigh order,
and two-pass streamfunction L2 peeling as training.  It intentionally reports
the spectrum of the *peeled* Gram matrix; no raw-network rank is formed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from evaluate_connected_dirichlet_psi_basis import load_model
from metaball_runtime import backend, excluded_area


F32 = np.float32


def continuous_fields(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    token: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return analytic psi, its full xy gradient, and symmetric Hessian."""
    count, modes = len(points), int(cfg.modes)
    value = cp.empty((count, modes), dtype=backend.F32)
    gradient = cp.empty((count, 2, modes), dtype=backend.F32)
    hessian = cp.empty((count, 3, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, token, 0.0, False)
    ):
        value[:, index] = field.value[:, 0]
        gradient[:, :, index] = field.grad[:, 0]
        # Compact symmetric order: xx, xy, yy.
        hessian[:, :, index] = field.hess[:, 0]
    return value, gradient, hessian


def rank_counts(eigenvalues: np.ndarray, count: int) -> dict[str, Any]:
    clipped = np.maximum(eigenvalues, 0.0)
    singular = np.sqrt(clipped)
    maximum = float(singular[-1])
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise FloatingPointError("peeled Gram spectrum has no positive scale")
    fp32_relative_tolerance = max(count, len(eigenvalues)) * np.finfo(F32).eps
    result: dict[str, Any] = {
        "effective_rank_fp32": int(
            np.count_nonzero(singular > maximum * fp32_relative_tolerance)
        ),
        "fp32_singular_value_relative_tolerance": float(
            fp32_relative_tolerance
        ),
    }
    for exponent in (2, 3, 4, 5, 6, 7, 8):
        threshold = 10.0 ** (-exponent)
        result[f"rank_sigma_relative_1e-{exponent}"] = int(
            np.count_nonzero(singular > maximum * threshold)
        )
    positive = singular[singular > maximum * fp32_relative_tolerance]
    result["condition_number_on_fp32_effective_subspace"] = float(
        maximum / max(float(positive[0]), np.finfo(np.float64).tiny)
    )
    result["sigma_max"] = maximum
    result["sigma_min_clipped"] = float(singular[0])
    result["gram_eigenvalue_min"] = float(eigenvalues[0])
    result["gram_eigenvalue_max"] = float(eigenvalues[-1])
    return result


def direct_peeled_fields(
    cfg: Any,
    value: cp.ndarray,
    gradient: cp.ndarray,
    hessian: cp.ndarray,
    fluid_area: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list[float]]:
    """Reproduce the in-place training residuals without transform roundoff."""
    count, modes = value.shape
    weight = cp.asarray(F32(fluid_area / count), dtype=backend.F32)
    candidate_mass = weight * cp.sum(value * value, axis=0)
    candidate_stiffness = weight * cp.sum(
        gradient * gradient, axis=(0, 1)
    )
    sorting_rayleigh = cp.asnumpy(candidate_stiffness / candidate_mass)
    if not np.all(np.isfinite(sorting_rayleigh)):
        raise FloatingPointError("non-finite sorting Rayleigh quotient")
    if str(getattr(cfg, "causal_sorting", "fixed_order")) == "geometry_rayleigh":
        order = np.argsort(sorting_rayleigh, kind="stable").tolist()
    else:
        order = list(range(modes))
    passes = max(1, int(getattr(cfg, "reorthogonalization_passes", 2)))
    basis_value = cp.empty_like(value)
    basis_gradient = cp.empty_like(gradient)
    basis_hessian = cp.empty_like(hessian)
    transform = cp.zeros((modes, modes), dtype=backend.F32)
    masses: list[float] = []
    for rank, network_index in enumerate(order):
        residual_value = value[:, network_index].copy()
        residual_gradient = gradient[:, :, network_index].copy()
        residual_hessian = hessian[:, :, network_index].copy()
        coefficients = cp.zeros((modes,), dtype=backend.F32)
        coefficients[network_index] = F32(1.0)
        if rank:
            previous_value = basis_value[:, :rank]
            previous_gradient = basis_gradient[:, :, :rank]
            previous_hessian = basis_hessian[:, :, :rank]
            previous_transform = transform[:, :rank]
            for _ in range(passes):
                increment = weight * (previous_value.T @ residual_value)
                residual_value -= previous_value @ increment
                residual_gradient -= cp.einsum(
                    "ncr,r->nc", previous_gradient, increment, optimize=True
                )
                residual_hessian -= cp.einsum(
                    "ncr,r->nc", previous_hessian, increment, optimize=True
                )
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual_value, residual_value)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= float(cfg.mass_floor):
            raise FloatingPointError(
                f"unresolved direct peeled mode {rank + 1}: M={mass_value:.8e}"
            )
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis_value[:, rank] = residual_value * inverse_norm
        basis_gradient[:, :, rank] = residual_gradient * inverse_norm
        basis_hessian[:, :, rank] = residual_hessian * inverse_norm
        transform[:, rank] = coefficients * inverse_norm
        masses.append(mass_value)
    return basis_value, basis_hessian, transform, masses


def diagnose_geometry(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    token: float,
) -> dict[str, Any]:
    value, gradient, hessian = continuous_fields(cfg, model, points, token)
    box_area = float((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin))
    fluid_area = box_area - excluded_area(cfg, token)
    peeled_value, peeled_hessian, transform, residual_masses = (
        direct_peeled_fields(cfg, value, gradient, hessian, fluid_area)
    )
    weight = F32(fluid_area / len(points))
    gram = weight * (peeled_value.T @ peeled_value)
    gram64 = cp.asnumpy(gram).astype(np.float64)
    gram64 = 0.5 * (gram64 + gram64.T)
    eigenvalues = np.linalg.eigvalsh(gram64)
    diagonal = np.diag(gram64)
    off_diagonal = gram64.copy()
    np.fill_diagonal(off_diagonal, 0.0)

    # u=(psi_y,-psi_x).  Because the Jet stores a single symmetric xy entry,
    # dudx and -dvdy are derived from exactly the same analytic mixed partial.
    peeled_hxx = peeled_hessian[:, 0]
    peeled_hxy = peeled_hessian[:, 1]
    peeled_hyy = peeled_hessian[:, 2]
    divergence = peeled_hxy + (-peeled_hxy)
    divergence_rms = float(cp.sqrt(cp.mean(divergence * divergence)).get())
    divergence_max = float(cp.max(cp.abs(divergence)).get())
    velocity_gradient_rms = float(
        cp.sqrt(
            cp.mean(
                peeled_hxx * peeled_hxx
                + F32(2.0) * peeled_hxy * peeled_hxy
                + peeled_hyy * peeled_hyy
            )
        ).get()
    )
    relative_divergence = divergence_rms / max(
        velocity_gradient_rms, np.finfo(np.float32).tiny
    )

    # Applying the accumulated coefficient transform in one GEMM can be much
    # less stable than the in-place residual construction when the raw modes
    # are nearly dependent.  Report it separately as a deployment audit.
    reconstructed = value @ transform
    reconstructed_gram = cp.asnumpy(
        weight * (reconstructed.T @ reconstructed)
    ).astype(np.float64)
    reconstructed_gram = 0.5 * (
        reconstructed_gram + reconstructed_gram.T
    )

    result = {
        "geometry_token": int(round(token)),
        "continuous_validation_points": int(len(points)),
        "requested_modes": int(cfg.modes),
        **rank_counts(eigenvalues, len(points)),
        "peeled_orthogonality_max_abs": float(
            np.max(np.abs(gram64 - np.eye(int(cfg.modes))))
        ),
        "peeled_off_diagonal_max_abs": float(
            np.max(np.abs(off_diagonal))
        ),
        "peeled_diagonal_max_abs_error": float(
            np.max(np.abs(diagonal - 1.0))
        ),
        "peeled_gram_frobenius_error_per_mode": float(
            np.linalg.norm(gram64 - np.eye(int(cfg.modes)), ord="fro")
            / int(cfg.modes)
        ),
        "residual_mass_min": float(min(residual_masses)),
        "residual_mass_max": float(max(residual_masses)),
        "coefficient_transform_max_abs": float(
            cp.max(cp.abs(transform)).get()
        ),
        "transform_reconstruction_orthogonality_max_abs": float(
            np.max(
                np.abs(
                    reconstructed_gram - np.eye(int(cfg.modes))
                )
            )
        ),
        "divergence_rms_analytic": divergence_rms,
        "divergence_max_abs_analytic": divergence_max,
        "velocity_gradient_rms_analytic": velocity_gradient_rms,
        "relative_divergence_l2": float(relative_divergence),
        "divergence_identity": "d_x(psi_y)+d_y(-psi_x)=psi_xy-psi_xy",
        "finite_difference_used": False,
        "grid_used": False,
        "causal_sorting": str(
            getattr(cfg, "causal_sorting", "fixed_order")
        ),
        "reorthogonalization_passes": int(
            getattr(cfg, "reorthogonalization_passes", 2)
        ),
    }
    del value, gradient, hessian, transform, peeled_value, peeled_hessian, gram
    del reconstructed, reconstructed_gram
    del peeled_hxx, peeled_hxy, peeled_hyy, divergence
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-geometries", type=int, default=4)
    args = parser.parse_args()

    metadata, cfg, model = load_model(args.checkpoint)
    if int(cfg.modes) != 1024:
        raise ValueError(f"expected K=1024, found K={cfg.modes}")
    if str(cfg.basis_objective) != "streamfunction_dirichlet_rayleigh":
        raise ValueError("expected streamfunction Dirichlet Rayleigh checkpoint")
    with np.load(args.validation_pack, allow_pickle=False) as archive:
        points = archive["validation_points"].astype(F32, copy=False)
        tokens = archive["validation_beta"].astype(F32, copy=False)
    count = min(len(tokens), max(1, args.max_geometries))
    rows: list[dict[str, Any]] = []
    for index in range(count):
        row = diagnose_geometry(cfg, model, points[index], float(tokens[index]))
        rows.append(row)
        print(
            f"geometry={index + 1}/{count} token={row['geometry_token']} "
            f"rank_fp32={row['effective_rank_fp32']}/{cfg.modes} "
            f"orth={row['peeled_orthogonality_max_abs']:.6e} "
            f"div={row['relative_divergence_l2']:.6e}",
            flush=True,
        )

    aggregate = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "method": (
            "continuous geometry-Rayleigh sorting plus two-pass L2 peeling; "
            "rank is measured only from the peeled Gram spectrum"
        ),
        "primary_rank_definition": (
            "sigma_i > sigma_max * max(points,modes) * eps(float32)"
        ),
        "geometries": rows,
        "summary": {
            "effective_rank_fp32_min": min(
                row["effective_rank_fp32"] for row in rows
            ),
            "effective_rank_fp32_max": max(
                row["effective_rank_fp32"] for row in rows
            ),
            "effective_rank_fp32_mean": float(
                np.mean([row["effective_rank_fp32"] for row in rows])
            ),
            "peeled_orthogonality_max_abs_worst": max(
                row["peeled_orthogonality_max_abs"] for row in rows
            ),
            "peeled_off_diagonal_max_abs_worst": max(
                row["peeled_off_diagonal_max_abs"] for row in rows
            ),
            "relative_divergence_l2_worst": max(
                row["relative_divergence_l2"] for row in rows
            ),
            "divergence_max_abs_analytic_worst": max(
                row["divergence_max_abs_analytic"] for row in rows
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "peeled_rank_orthogonality_divergence.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    csv_path = args.output_dir / "peeled_rank_orthogonality_divergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
