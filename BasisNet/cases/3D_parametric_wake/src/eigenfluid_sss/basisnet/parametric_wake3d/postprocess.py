from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .training import (
    DEVICE,
    Config,
    OfficialPeelingFields,
    approximate_fluid_volume,
    discrete_mode_fields,
    fluid_points,
)


def load_model(checkpoint: Path) -> tuple[Config, OfficialPeelingFields, dict]:
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    allowed = {field.name for field in fields(Config)}
    cfg = Config(
        **{key: value for key, value in payload["config"].items() if key in allowed}
    )
    model = OfficialPeelingFields(cfg).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, model, payload


def generalized_symmetric_eig(
    stiffness: np.ndarray, mass: np.ndarray, modes: int
) -> tuple[np.ndarray, np.ndarray]:
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, eigenvectors = np.linalg.eigh(mass)
    floor = max(float(eigenvalues[-1]) * 1.0e-10, 1.0e-14)
    keep = eigenvalues > floor
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitener.T @ stiffness @ whitener
    values, vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    order = np.argsort(values)[:modes]
    return values[order], whitener @ vectors[:, order]


def compute_ritz_basis(
    model: OfficialPeelingFields,
    cfg: Config,
    alpha: float,
    quadrature: int = 512,
) -> tuple[np.ndarray, np.ndarray, int]:
    points = fluid_points(cfg, quadrature, alpha, cfg.seed + 96_300)
    velocities, gradients = [], []
    for mode in range(cfg.modes):
        fields = discrete_mode_fields(cfg, model, mode, points, alpha)
        velocities.append(fields["velocity"].detach().cpu().numpy())
        gradients.append(fields["gradient"].detach().cpu().numpy())
    velocity = np.stack(velocities, axis=-1)
    gradient = np.stack(gradients, axis=-1)
    weight = approximate_fluid_volume(cfg, alpha) / len(points)
    mass = weight * np.einsum("nci,ncj->ij", velocity, velocity)
    stiffness = weight * np.einsum("ncdi,ncdj->ij", gradient, gradient)
    eigenvalues, coefficients = generalized_symmetric_eig(
        stiffness, mass, cfg.modes
    )
    spectrum = np.linalg.eigvalsh(0.5 * (mass + mass.T))
    rank = int(
        np.sum(spectrum > max(float(spectrum[-1]) * 1.0e-8, 1.0e-14))
    )
    return eigenvalues, coefficients, rank


def basis_fields(
    model: OfficialPeelingFields,
    cfg: Config,
    points: np.ndarray,
    alpha: float,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    potentials, velocities = [], []
    for mode in range(cfg.modes):
        fields = discrete_mode_fields(
            cfg, model, mode, points, alpha, with_gradient=False
        )
        potentials.append(fields["potential"].detach().cpu().numpy())
        velocities.append(fields["velocity"].detach().cpu().numpy())
    potential = np.stack(potentials, axis=-1)
    velocity = np.stack(velocities, axis=-1)
    return (
        np.einsum("nci,ij->ncj", potential, coefficients),
        np.einsum("nci,ij->ncj", velocity, coefficients),
    )


def track_ritz_mode(
    previous: np.ndarray | None,
    candidates: np.ndarray,
    seed_index: int,
) -> tuple[np.ndarray, int, float]:
    """Track one continuous Ritz branch and remove arbitrary sign flips.

    ``candidates`` has shape ``(points, 3, modes)``.  At the first parameter
    value the requested seed index is used.  Later values select the mode with
    maximum absolute normalized correlation to the previous displayed field.
    """
    if previous is None:
        return candidates[:, :, seed_index].copy(), seed_index, 1.0
    flat_previous = previous.reshape(-1)
    flat_candidates = candidates.reshape(-1, candidates.shape[-1])
    previous_norm = max(float(np.linalg.norm(flat_previous)), 1.0e-15)
    candidate_norms = np.maximum(
        np.linalg.norm(flat_candidates, axis=0), 1.0e-15
    )
    correlations = (flat_candidates.T @ flat_previous) / (
        candidate_norms * previous_norm
    )
    index = int(np.argmax(np.abs(correlations)))
    correlation = float(correlations[index])
    sign = 1.0 if correlation >= 0.0 else -1.0
    return sign * candidates[:, :, index], index, abs(correlation)
