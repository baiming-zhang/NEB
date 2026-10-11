

from __future__ import annotations

import json
from _neb_paths import Path

import numpy as np
import torch

from .training import (
    DEVICE,
    Config,
    OfficialPeelingFields,
    fluid_area,
    fluid_points,
    raw_mode_fields,
)


def load_model(
    checkpoint: Path,
) -> tuple[Config, OfficialPeelingFields, dict]:
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    cfg = Config(**payload["config"])
    model = OfficialPeelingFields(cfg).to(DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, model, payload


def generalized_symmetric_eig(
    stiffness: np.ndarray,
    mass: np.ndarray,
    modes: int,
) -> tuple[np.ndarray, np.ndarray]:

    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, eigenvectors = np.linalg.eigh(mass)
    floor = max(float(eigenvalues[-1]) * 1.0e-10, 1.0e-14)
    keep = eigenvalues > floor
    whitener = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitener.T @ stiffness @ whitener
    reduced_values, reduced_vectors = np.linalg.eigh(
        0.5 * (reduced + reduced.T)
    )
    order = np.argsort(reduced_values)[:modes]
    return reduced_values[order], whitener @ reduced_vectors[:, order]


def compute_ritz_basis(
    cfg: Config,
    model: OfficialPeelingFields,
    beta: float,
    alpha_deg: float,
    quadrature: int = 512,
    seed: int | None = None,
) -> dict[str, np.ndarray | float | int]:

    points = fluid_points(
        cfg,
        quadrature,
        beta,
        alpha_deg,
        cfg.seed + 96300 if seed is None else seed,
    )
    raw_velocity: list[np.ndarray] = []
    raw_gradient: list[np.ndarray] = []
    with torch.enable_grad():
        for mode in range(cfg.modes):
            fields = raw_mode_fields(
                model, mode, points, beta, alpha_deg, second_order=True
            )
            raw_velocity.append(
                torch.cat((fields["u"], fields["v"])).detach().cpu().numpy()
            )
            raw_gradient.append(
                torch.cat(
                    (
                        fields["ux"],
                        fields["uy"],
                        fields["vx"],
                        fields["vy"],
                    )
                )
                .detach()
                .cpu()
                .numpy()
            )
    velocity = np.stack(raw_velocity, axis=1).astype(np.float64)
    gradient = np.stack(raw_gradient, axis=1).astype(np.float64)
    weight = fluid_area(cfg, beta) / len(points)
    mass = weight * velocity.T @ velocity
    stiffness = weight * gradient.T @ gradient
    values, coefficients = generalized_symmetric_eig(
        stiffness, mass, cfg.modes
    )
    return {
        "beta": float(beta),
        "alpha_deg": float(alpha_deg),
        "points": points,
        "eigenvalues": values,
        "coefficients": coefficients,
        "mass": mass,
        "stiffness": stiffness,
        "rank": int(np.linalg.matrix_rank(mass)),
    }


def basis_fields(
    model: OfficialPeelingFields,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
    ritz_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_psi: list[np.ndarray] = []
    raw_u: list[np.ndarray] = []
    raw_v: list[np.ndarray] = []
    with torch.enable_grad():
        for mode in range(model.cfg.modes):
            fields = raw_mode_fields(
                model,
                mode,
                points,
                beta,
                alpha_deg,
                second_order=False,
            )
            raw_psi.append(fields["psi"].detach().cpu().numpy())
            raw_u.append(fields["u"].detach().cpu().numpy())
            raw_v.append(fields["v"].detach().cpu().numpy())
    return (
        np.stack(raw_psi, axis=1) @ ritz_coefficients,
        np.stack(raw_u, axis=1) @ ritz_coefficients,
        np.stack(raw_v, axis=1) @ ritz_coefficients,
    )


def weighted_projection(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    matrix = np.concatenate((basis_u, basis_v), axis=0)
    target = np.concatenate((target_u, target_v), axis=0)
    if weights is None:
        return np.linalg.lstsq(matrix, target, rcond=1.0e-10)[0]
    root = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted_matrix = np.concatenate(
        (basis_u * root[:, None], basis_v * root[:, None]), axis=0
    )
    weighted_target = np.concatenate((target_u * root, target_v * root))
    return np.linalg.lstsq(
        weighted_matrix, weighted_target, rcond=1.0e-10
    )[0]


def relative_vector_error(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    predicted_u: np.ndarray,
    predicted_v: np.ndarray,
) -> float:
    numerator = np.sum(
        (predicted_u - actual_u) ** 2 + (predicted_v - actual_v) ** 2
    )
    denominator = np.sum(actual_u**2 + actual_v**2)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
