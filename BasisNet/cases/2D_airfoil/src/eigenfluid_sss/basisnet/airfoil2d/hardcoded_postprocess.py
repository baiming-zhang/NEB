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
    # Training evaluates all K modes together, but postprocessing has no
    # cross-mode dependency until the completed field columns are assembled
    # into the same dense mass/stiffness matrices.  Stream contiguous mode
    # blocks here so a 4096-point Jet3 Ritz evaluation does not materialize
    # K=1024 modes and all ten derivative channels in one CUDA allocation.
    # This changes only evaluation memory scheduling: points, weights,
    # precision, column order, and the final FP64 matrix algebra are unchanged.
    cfg.mode_forward_batch = min(
        int(getattr(cfg, "mode_forward_batch", 1)),
        int(getattr(cfg, "postprocess_mode_forward_batch", 64)),
    )
    # These two values define only the reported wake subdomain.  The new
    # uniform-area training config has no wake sampling term, so it correctly
    # omits them; retain the established evaluation mask used by every prior
    # 2D comparison when the checkpoint does not carry display metadata.
    cfg.wake_length_chords = float(getattr(cfg, "wake_length_chords", 3.0))
    cfg.wake_half_height_chords = float(
        getattr(cfg, "wake_half_height_chords", 0.55)
    )
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
    """Solve ``K c = lambda M c`` on the backward-resolved mass range."""

    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    values, vectors = np.linalg.eigh(mass)
    largest = max(float(values[-1]), 0.0)
    rows = len(values) if sample_rows is None else int(sample_rows)
    floor = np.finfo(np.float64).eps * max(rows, len(values)) * largest
    keep = values > floor
    if not np.any(keep):
        raise np.linalg.LinAlgError(
            "mass matrix has no backward-resolved direction"
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
                -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
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
                -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
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
                    hess[:, 1],
                    hess[:, 2],
                    -hess[:, 0],
                    -hess[:, 1],
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
        stiffness, mass, cfg.modes
    )
    return {
        "eigenvalues": eigenvalues,
        "coefficients": coefficients,
        "rank": rank,
        "mass": mass,
        "stiffness": stiffness,
    }


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
    floor = max(float(np.trace(gram)) / max(len(gram), 1) * 1.0e-12, 1.0e-14)
    return np.linalg.solve(gram + floor * np.eye(len(gram)), right)


def project_ritz_velocity(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    ritz_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    floor = max(
        float(np.trace(ritz_gram)) / max(len(ritz_gram), 1) * 1.0e-12,
        1.0e-14,
    )
    ritz_modal = np.linalg.solve(
        ritz_gram + floor * np.eye(len(ritz_gram)), ritz_right
    )
    raw_modal = coefficients @ ritz_modal
    normalized_gram = ritz_gram / max(
        float(np.trace(ritz_gram)) / len(ritz_gram), 1.0e-30
    )
    orthogonality_error = float(
        np.max(np.abs(normalized_gram - np.eye(len(ritz_gram))))
    )
    return ritz_modal, raw_modal, orthogonality_error


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
    ritz_gram = 0.5 * (ritz_gram + ritz_gram.T)
    values, vectors = np.linalg.eigh(ritz_gram)
    largest = max(float(values[-1]), 1.0e-30)
    keep = values > largest * 1.0e-10
    ritz_modal = vectors[:, keep] @ (
        (vectors[:, keep].T @ ritz_right) / values[keep]
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
        int(np.count_nonzero(keep)),
        relative_residual,
    )
