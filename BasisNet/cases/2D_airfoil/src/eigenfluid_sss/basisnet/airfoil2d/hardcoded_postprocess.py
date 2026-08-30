"""No-AD postprocessing for neutral hardcoded airfoil checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from .hardcoded_backend import (
    HardcodedModel,
    _sobol_box,
    airfoil_area,
    cfg_namespace,
    uniform_fluid_points,
)


def load_hardcoded_model(
    checkpoint: Path,
) -> tuple[Any, HardcodedModel, dict[str, Any]]:
    metadata = json.loads(
        checkpoint.with_suffix(".json").read_text(encoding="utf-8")
    )
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    return cfg, HardcodedModel(cfg, arrays), metadata


def ritz_fluid_points(
    cfg: Any,
    count: int,
    beta: float,
    alpha_deg: float,
    seed: int,
) -> np.ndarray:
    """Uniform mesh-free Ritz quadrature, mirror-paired for symmetric geometry."""
    if abs(alpha_deg) > 1.0e-12:
        return _sobol_box(
            cfg,
            count,
            (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
            seed,
            beta,
            alpha_deg,
        )
    half = (count + 1) // 2
    upper = _sobol_box(
        cfg,
        half,
        (cfg.xmin, cfg.xmax, 0.0, cfg.ymax),
        seed,
        beta,
        0.0,
    )
    lower = upper * np.asarray((1.0, -1.0), dtype=np.float64)
    return np.vstack((upper, lower))[:count]


def generalized_symmetric_eig(
    stiffness: np.ndarray,
    mass: np.ndarray,
    modes: int,
    *,
    sample_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve ``K c = lambda M c`` on the backward-resolved mass range.

    The mass cutoff is the standard floating-point backward-error bound for
    an explicitly accumulated Gram matrix.  It contains no model-selection
    or smoothing parameter.
    """

    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    values, vectors = np.linalg.eigh(mass)
    largest = max(float(values[-1]), 0.0)
    rows = len(values) if sample_rows is None else int(sample_rows)
    floor = np.finfo(np.float64).eps * max(rows, len(values)) * largest
    keep = values > floor
    if not np.any(keep):
        raise np.linalg.LinAlgError(
            "Stokes mass matrix has no backward-resolved direction"
        )
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    reduced = whitener.T @ stiffness @ whitener
    reduced_values, reduced_vectors = np.linalg.eigh(
        0.5 * (reduced + reduced.T)
    )
    order = np.argsort(reduced_values)[: min(int(modes), int(np.sum(keep)))]
    return (
        reduced_values[order],
        whitener @ reduced_vectors[:, order],
        int(np.sum(keep)),
    )


def mode_fields(
    model: HardcodedModel,
    cfg: Any,
    mode: int,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
    batch: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_parts, v_parts, omega_parts = [], [], []
    for start in range(0, len(points), batch):
        field, _ = model.modes[mode].forward(
            cfg, points[start : start + batch], beta, alpha_deg, False
        )
        u_parts.append(cp.asnumpy(field.grad[:, 0, 1]))
        v_parts.append(cp.asnumpy(-field.grad[:, 0, 0]))
        omega_parts.append(
            cp.asnumpy(
                -(field.hess[:, 0, 0, 0] + field.hess[:, 0, 1, 1])
            )
        )
    return (
        np.concatenate(u_parts).astype(np.float32),
        np.concatenate(v_parts).astype(np.float32),
        np.concatenate(omega_parts).astype(np.float32),
    )


def raw_fields(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
    batch: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_u = np.empty((len(points), cfg.modes), dtype=np.float32)
    raw_v = np.empty_like(raw_u)
    raw_omega = np.empty_like(raw_u)
    mode_batch = max(1, int(getattr(cfg, "mode_forward_batch", 1)))
    if mode_batch == 1:
        for mode in range(cfg.modes):
            raw_u[:, mode], raw_v[:, mode], raw_omega[:, mode] = mode_fields(
                model, cfg, mode, points, beta, alpha_deg, batch
            )
        return raw_u, raw_v, raw_omega

    # Batched mode evaluation reuses encoded coordinates and turns the MLP
    # layers into strided batched GEMMs.  Cap points per pass so the Jet2
    # Hessian tensors remain comfortably within GPU memory.
    point_budget = int(getattr(cfg, "postprocess_mode_point_budget", 262_144))
    effective_batch = min(batch, max(256, point_budget // mode_batch))
    for start in range(0, len(points), effective_batch):
        stop = min(start + effective_batch, len(points))
        for mode, (_network, field, _cache) in enumerate(
            model.forward_stream(
                cfg, points[start:stop], beta, alpha_deg, False
            )
        ):
            raw_u[start:stop, mode] = cp.asnumpy(field.grad[:, 0, 1])
            raw_v[start:stop, mode] = cp.asnumpy(-field.grad[:, 0, 0])
            raw_omega[start:stop, mode] = cp.asnumpy(
                -(field.hess[:, 0, 0, 0] + field.hess[:, 0, 1, 1])
            )
    return raw_u, raw_v, raw_omega


def compute_ritz_basis(
    model: HardcodedModel,
    cfg: Any,
    beta: float,
    alpha_deg: float,
    quadrature: int,
    seed: int | None = None,
) -> dict[str, Any]:
    points = ritz_fluid_points(
        cfg,
        quadrature,
        beta,
        alpha_deg,
        cfg.seed + 96_300 if seed is None else seed,
    )
    velocity = np.empty((2 * len(points), cfg.modes), dtype=np.float64)
    gradient = np.empty((4 * len(points), cfg.modes), dtype=np.float64)
    if max(1, int(getattr(cfg, "mode_forward_batch", 1))) == 1:
        stream = (
            (model.modes[mode],)
            + model.modes[mode].forward(cfg, points, beta, alpha_deg, False)
            for mode in range(cfg.modes)
        )
    else:
        stream = model.forward_stream(cfg, points, beta, alpha_deg, False)
    for mode, (_network, field, _cache) in enumerate(stream):
        grad = field.grad[:, 0]
        hess = field.hess[:, 0]
        velocity[:, mode] = cp.asnumpy(
            cp.concatenate((grad[:, 1], -grad[:, 0]))
        )
        gradient[:, mode] = cp.asnumpy(
            cp.concatenate(
                (
                    hess[:, 1, 0],
                    hess[:, 1, 1],
                    -hess[:, 0, 0],
                    -hess[:, 0, 1],
                )
            )
        )
    domain_area = (
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = domain_area / len(points)
    mass = weight * velocity.T @ velocity
    stiffness = weight * gradient.T @ gradient
    eigenvalues, coefficients, rank = generalized_symmetric_eig(
        stiffness,
        mass,
        cfg.modes,
        sample_rows=2 * len(points),
    )
    return {
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "rank": rank,
        "mass": mass,
        "stiffness": stiffness,
    }


def _stokes_operators_on_points(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the physical-area Stokes mass and stiffness matrices."""

    velocity = np.empty((2 * len(points), cfg.modes), dtype=np.float64)
    gradient = np.empty((4 * len(points), cfg.modes), dtype=np.float64)
    point_batch = max(
        64, int(getattr(cfg, "stokes_operator_point_batch", 4096))
    )
    point_count = len(points)
    for start in range(0, point_count, point_batch):
        stop = min(start + point_batch, point_count)
        stream = model.forward_stream(
            cfg, points[start:stop], beta, alpha_deg, False
        )
        for mode, (_network, field, _cache) in enumerate(stream):
            grad = field.grad[:, 0]
            hess = field.hess[:, 0]
            velocity[start:stop, mode] = cp.asnumpy(grad[:, 1])
            velocity[
                point_count + start : point_count + stop, mode
            ] = cp.asnumpy(-grad[:, 0])
            gradient[start:stop, mode] = cp.asnumpy(hess[:, 1, 0])
            gradient[
                point_count + start : point_count + stop, mode
            ] = cp.asnumpy(hess[:, 1, 1])
            gradient[
                2 * point_count + start : 2 * point_count + stop, mode
            ] = cp.asnumpy(-hess[:, 0, 0])
            gradient[
                3 * point_count + start : 3 * point_count + stop, mode
            ] = cp.asnumpy(-hess[:, 0, 1])
    area = (
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = area / len(points)
    return weight * velocity.T @ velocity, weight * gradient.T @ gradient


def certify_stokes_operators(
    construction_mass: np.ndarray,
    construction_stiffness: np.ndarray,
    independent_coarse_mass: np.ndarray,
    independent_coarse_stiffness: np.ndarray,
    independent_fine_mass: np.ndarray,
    independent_fine_stiffness: np.ndarray,
    modes: int,
    *,
    construction_rows: int,
    independent_rows: int,
) -> dict[str, Any]:
    """Certify a low Stokes eigenspace against an independent nested rule.

    The coarse-to-fine disagreement of the independent rule is the observed
    integration uncertainty.  A direction passes only when its independent
    mass defect, generalized Ritz residual, and eigenvalue drift are no
    larger than that uncertainty plus the explicit floating-point
    accumulation bound.  No user-tuned residual or spectral cutoff enters
    the decision.
    """

    eigenvalues, coefficients, construction_rank = generalized_symmetric_eig(
        construction_stiffness,
        construction_mass,
        modes,
        sample_rows=construction_rows,
    )
    count = len(eigenvalues)
    fine_mass = 0.5 * (
        np.asarray(independent_fine_mass, dtype=np.float64)
        + np.asarray(independent_fine_mass, dtype=np.float64).T
    )
    fine_stiffness = 0.5 * (
        np.asarray(independent_fine_stiffness, dtype=np.float64)
        + np.asarray(independent_fine_stiffness, dtype=np.float64).T
    )
    coarse_mass = 0.5 * (
        np.asarray(independent_coarse_mass, dtype=np.float64)
        + np.asarray(independent_coarse_mass, dtype=np.float64).T
    )
    coarse_stiffness = 0.5 * (
        np.asarray(independent_coarse_stiffness, dtype=np.float64)
        + np.asarray(independent_coarse_stiffness, dtype=np.float64).T
    )

    projected_mass = coefficients.T @ fine_mass @ coefficients
    projected_mass_coarse = coefficients.T @ coarse_mass @ coefficients
    mass_defect = np.abs(projected_mass - np.eye(count))
    mass_uncertainty = np.abs(projected_mass - projected_mass_coarse)
    eps_factor = np.finfo(np.float64).eps * max(
        int(independent_rows), fine_mass.shape[0]
    )
    absolute_coefficients = np.abs(coefficients)
    mass_roundoff = eps_factor * (
        absolute_coefficients.T @ np.abs(fine_mass) @ absolute_coefficients
    )
    mass_budget = mass_uncertainty + mass_roundoff

    fine_kc = fine_stiffness @ coefficients
    fine_mc = fine_mass @ coefficients
    coarse_kc = coarse_stiffness @ coefficients
    coarse_mc = coarse_mass @ coefficients
    residual = fine_kc - fine_mc * eigenvalues[None, :]
    residual_uncertainty = (
        (fine_kc - coarse_kc)
        - (fine_mc - coarse_mc) * eigenvalues[None, :]
    )
    residual_scale = np.linalg.norm(fine_kc, axis=0) + (
        np.abs(eigenvalues) * np.linalg.norm(fine_mc, axis=0)
    )
    residual_scale = np.maximum(residual_scale, np.finfo(np.float64).tiny)
    residual_relative = np.linalg.norm(residual, axis=0) / residual_scale
    residual_uncertainty_relative = (
        np.linalg.norm(residual_uncertainty, axis=0) / residual_scale
    )
    residual_roundoff = eps_factor * (
        np.linalg.norm(np.abs(fine_stiffness) @ absolute_coefficients, axis=0)
        + np.abs(eigenvalues)
        * np.linalg.norm(np.abs(fine_mass) @ absolute_coefficients, axis=0)
    ) / residual_scale
    residual_budget = residual_uncertainty_relative + residual_roundoff

    fine_mass_diagonal = np.diag(projected_mass)
    coarse_mass_diagonal = np.diag(projected_mass_coarse)
    fine_rayleigh = np.sum(coefficients * fine_kc, axis=0) / fine_mass_diagonal
    coarse_rayleigh = (
        np.sum(coefficients * coarse_kc, axis=0) / coarse_mass_diagonal
    )
    eigenvalue_scale = np.maximum(
        np.maximum(np.abs(eigenvalues), np.abs(fine_rayleigh)),
        np.finfo(np.float64).tiny,
    )
    eigenvalue_drift = np.abs(fine_rayleigh - eigenvalues) / eigenvalue_scale
    eigenvalue_budget = (
        np.abs(fine_rayleigh - coarse_rayleigh) / eigenvalue_scale
        + eps_factor
    )

    certified_count = 0
    certified_mask = np.zeros(count, dtype=bool)
    for prefix in range(1, count + 1):
        mass_pass = bool(
            np.all(
                mass_defect[:prefix, :prefix]
                <= mass_budget[:prefix, :prefix]
            )
        )
        residual_pass = bool(
            np.all(
                residual_relative[:prefix] <= residual_budget[:prefix]
            )
        )
        eigenvalue_pass = bool(
            np.all(eigenvalue_drift[:prefix] <= eigenvalue_budget[:prefix])
        )
        if not (mass_pass and residual_pass and eigenvalue_pass):
            break
        certified_count = prefix
        certified_mask[:prefix] = True

    return {
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "construction_rank": int(construction_rank),
        "certified_count": int(certified_count),
        "certified_mask": certified_mask,
        "mass_defect": mass_defect,
        "mass_budget": mass_budget,
        "ritz_residual_relative": residual_relative,
        "ritz_residual_budget": residual_budget,
        "eigenvalue_drift_relative": eigenvalue_drift,
        "eigenvalue_drift_budget": eigenvalue_budget,
    }


def compute_certified_stokes_basis(
    model: HardcodedModel,
    cfg: Any,
    beta: float,
    alpha_deg: float,
    minimum_quadrature: int,
    maximum_quadrature: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Adapt a nested quadrature and return only its certified prefix."""

    minimum = max(2, int(minimum_quadrature))
    maximum = max(minimum, int(maximum_quadrature))
    construction_seed = cfg.seed + 96_300 if seed is None else int(seed)
    independent_seed = construction_seed + 1_000_003
    quadrature = minimum
    last: dict[str, Any] | None = None
    while True:
        construction_points = uniform_fluid_points(
            cfg, quadrature, beta, alpha_deg, construction_seed
        )
        independent_fine_points = uniform_fluid_points(
            cfg, quadrature, beta, alpha_deg, independent_seed
        )
        independent_coarse_points = independent_fine_points[
            : max(1, quadrature // 2)
        ]
        construction_mass, construction_stiffness = _stokes_operators_on_points(
            model, cfg, construction_points, beta, alpha_deg
        )
        coarse_mass, coarse_stiffness = _stokes_operators_on_points(
            model, cfg, independent_coarse_points, beta, alpha_deg
        )
        fine_mass, fine_stiffness = _stokes_operators_on_points(
            model, cfg, independent_fine_points, beta, alpha_deg
        )
        certificate = certify_stokes_operators(
            construction_mass,
            construction_stiffness,
            coarse_mass,
            coarse_stiffness,
            fine_mass,
            fine_stiffness,
            cfg.modes,
            construction_rows=2 * quadrature,
            independent_rows=2 * quadrature,
        )
        certified_count = int(certificate["certified_count"])
        last = {
            **certificate,
            "quadrature": int(quadrature),
            "maximum_quadrature_reached": bool(quadrature >= maximum),
            "mass": construction_mass,
            "stiffness": construction_stiffness,
        }
        if certified_count == len(certificate["eigenvalues"]):
            break
        if quadrature >= maximum:
            break
        quadrature = min(maximum, 2 * quadrature)

    assert last is not None
    certified_count = int(last["certified_count"])
    last["all_eigenvalues"] = last["eigenvalues"]
    last["all_coefficients"] = last["coefficients"]
    last["eigenvalues"] = last["eigenvalues"][:certified_count]
    last["coefficients"] = last["coefficients"][:, :certified_count]
    last["rank"] = certified_count
    return last


def project_velocity(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> np.ndarray:
    gram = (
        basis_u.astype(np.float64).T @ basis_u.astype(np.float64)
        + basis_v.astype(np.float64).T @ basis_v.astype(np.float64)
    )
    right = (
        basis_u.astype(np.float64).T @ target_u.astype(np.float64)
        + basis_v.astype(np.float64).T @ target_v.astype(np.float64)
    )
    solution, _diagnostics = _minimum_norm_gram_solve(
        gram,
        right,
        sample_rows=2 * len(basis_u),
    )
    return solution


def _minimum_norm_gram_solve(
    gram: np.ndarray,
    right: np.ndarray,
    *,
    sample_rows: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Solve a sampled least-squares normal equation without a ridge.

    The cutoff is a backward-error bound for the explicitly accumulated
    Gram matrix: ``eps * max(sample_rows, n) * lambda_max``.  It therefore
    depends only on floating-point precision and matrix dimensions, not on a
    tunable regularization weight.  Unresolved directions receive the unique
    Moore-Penrose minimum-norm solution.
    """

    symmetric = 0.5 * (
        np.asarray(gram, dtype=np.float64)
        + np.asarray(gram, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(symmetric)
    largest = max(float(values[-1]), 0.0)
    tolerance = (
        np.finfo(np.float64).eps
        * max(int(sample_rows), len(values))
        * largest
    )
    keep = values > tolerance
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise np.linalg.LinAlgError(
            "sampled basis Gram matrix has no numerically resolved direction"
        )
    kept_vectors = vectors[:, keep]
    solution = kept_vectors @ (
        (kept_vectors.T @ np.asarray(right, dtype=np.float64)) / values[keep]
    )
    return solution, {
        "rank": rank,
        "discarded_rank": int(len(values) - rank),
        "rank_tolerance": float(tolerance),
        "condition_number": float(largest / values[keep][0]),
    }


def _mass_whitened_projection(
    gram: np.ndarray,
    right: np.ndarray,
    *,
    sample_rows: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project in coordinates orthonormal for the actual sampled measure.

    A Ritz rotation is orthonormal only for the quadrature that constructed
    it.  This second, target-independent whitening uses the projection grid's
    mass matrix itself.  Consequently the active coordinates are orthonormal
    on the measure where the error is evaluated, while numerical null
    directions are removed by the same backward-error rank certificate used
    by :func:`_minimum_norm_gram_solve`.
    """

    symmetric = 0.5 * (
        np.asarray(gram, dtype=np.float64)
        + np.asarray(gram, dtype=np.float64).T
    )
    values, vectors = np.linalg.eigh(symmetric)
    largest = max(float(values[-1]), 0.0)
    tolerance = (
        np.finfo(np.float64).eps
        * max(int(sample_rows), len(values))
        * largest
    )
    keep = values > tolerance
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise np.linalg.LinAlgError(
            "sampled Ritz Gram matrix has no numerically resolved direction"
        )
    whitener = vectors[:, keep] / np.sqrt(values[keep])[None, :]
    whitened_gram = whitener.T @ symmetric @ whitener
    whitened_right = whitener.T @ np.asarray(right, dtype=np.float64)
    coordinates = whitener @ whitened_right
    orthogonality_error = float(
        np.max(np.abs(whitened_gram - np.eye(rank)))
    )
    mean_diagonal = max(float(np.trace(symmetric)) / len(values), 1.0e-30)
    prewhiten_orthogonality_error = float(
        np.max(np.abs(symmetric / mean_diagonal - np.eye(len(values))))
    )
    return coordinates, {
        "rank": rank,
        "discarded_rank": int(len(values) - rank),
        "rank_tolerance": float(tolerance),
        "condition_number": float(largest / values[keep][0]),
        "orthogonality_error": orthogonality_error,
        "prewhiten_orthogonality_error": prewhiten_orthogonality_error,
    }


def project_ritz_velocity(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    ritz_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Project onto the mass-orthogonal Ritz rotation of the raw fields.

    This is algebraically identical to forming ``raw_u @ C`` and
    ``raw_v @ C`` before least squares, without allocating two additional
    full-resolution ``n_points x n_modes`` arrays.
    """
    raw_u64 = raw_u.astype(np.float64)
    raw_v64 = raw_v.astype(np.float64)
    coefficients = ritz_coefficients.astype(np.float64)
    raw_gram = raw_u64.T @ raw_u64 + raw_v64.T @ raw_v64
    raw_right = (
        raw_u64.T @ target_u.astype(np.float64)
        + raw_v64.T @ target_v.astype(np.float64)
    )
    ritz_gram = coefficients.T @ raw_gram @ coefficients
    ritz_right = coefficients.T @ raw_right
    ritz_modal, diagnostics = _mass_whitened_projection(
        ritz_gram,
        ritz_right,
        sample_rows=2 * len(raw_u),
    )
    raw_modal = coefficients @ ritz_modal
    return ritz_modal, raw_modal, diagnostics


def project_ritz_scalar(
    raw_scalar: np.ndarray,
    target_scalar: np.ndarray,
    ritz_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Least-squares projection of a scalar field in Ritz coordinates.

    A symmetric pseudoinverse is used because curl/vorticity columns can
    contain numerically unresolved directions even when the velocity mass
    matrix is full rank.  The returned solution is the minimum-norm
    least-squares solution over all resolved vorticity directions.
    """
    raw64 = raw_scalar.astype(np.float64)
    coefficients = ritz_coefficients.astype(np.float64)
    raw_gram = raw64.T @ raw64
    raw_right = raw64.T @ target_scalar.astype(np.float64)
    ritz_gram = coefficients.T @ raw_gram @ coefficients
    ritz_right = coefficients.T @ raw_right
    ritz_modal, diagnostics = _minimum_norm_gram_solve(
        ritz_gram,
        ritz_right,
        sample_rows=len(raw_scalar),
    )
    raw_modal = coefficients @ ritz_modal
    residual = raw64 @ raw_modal - target_scalar.astype(np.float64)
    relative_residual = float(
        np.linalg.norm(residual)
        / max(np.linalg.norm(target_scalar.astype(np.float64)), 1.0e-30)
    )
    return (
        ritz_modal,
        raw_modal,
        int(diagnostics["rank"]),
        relative_residual,
    )
