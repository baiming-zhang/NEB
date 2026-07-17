from __future__ import annotations

import subprocess
from dataclasses import fields
from pathlib import Path

import matplotlib.tri as mtri
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.func import jacfwd, vmap

from .training import DEVICE, DTYPE, Config, OfficialPeelingFields, fluid_area


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


def reflection_paired_fluid_points(
    cfg: Config, n: int, alpha: float, seed: int
) -> np.ndarray:
    half = (n + 1) // 2
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    centre = cfg.centre_left + (cfg.centre_right - cfg.centre_left) * alpha
    accepted: list[np.ndarray] = []
    remaining = half
    while remaining:
        unit = engine.draw(max(3 * remaining, 128)).cpu().numpy()
        candidates = np.column_stack(
            (
                cfg.xmin + (cfg.xmax - cfg.xmin) * unit[:, 0],
                unit[:, 1] * cfg.ymax,
            )
        ).astype(np.float32)
        fluid = (candidates[:, 0] - centre) ** 2 + candidates[:, 1] ** 2 > cfg.radius**2
        chunk = candidates[fluid][:remaining]
        accepted.append(chunk)
        remaining -= len(chunk)
    upper = np.concatenate(accepted)
    paired = np.vstack((upper, upper * np.array([1.0, -1.0], np.float32)))
    return paired[:n]


def derivatives(function, points: np.ndarray, batch: int, second_order: bool):
    first_fn = jacfwd(function)
    second_fn = jacfwd(first_fn) if second_order else None
    values, firsts, seconds = [], [], []
    for start in range(0, len(points), batch):
        z = torch.tensor(points[start : start + batch], device=DEVICE, dtype=DTYPE)
        values.append(vmap(function)(z).detach().cpu().numpy())
        firsts.append(vmap(first_fn)(z).detach().cpu().numpy())
        if second_fn is not None:
            seconds.append(vmap(second_fn)(z).detach().cpu().numpy())
        del z
        torch.cuda.empty_cache()
    return (
        np.concatenate(values),
        np.concatenate(firsts),
        np.concatenate(seconds) if second_order else None,
    )


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
    reduced_values, reduced_vectors = np.linalg.eigh(0.5 * (reduced + reduced.T))
    order = np.argsort(reduced_values)[:modes]
    return reduced_values[order], whitener @ reduced_vectors[:, order]


def compute_ritz_basis(
    model: OfficialPeelingFields,
    cfg: Config,
    alpha: float,
    quadrature: int = 512,
) -> tuple[np.ndarray, np.ndarray, int]:
    q = reflection_paired_fluid_points(cfg, quadrature, alpha, 20260712 + 96300)
    z = np.column_stack((q, np.full(len(q), alpha, dtype=np.float32))).astype(
        np.float32
    )

    def network(point):
        return model(point[:2].reshape(1, 2), point[2:3].reshape(1, 1))[0]

    _, first, second = derivatives(network, z, 32, True)
    u, v = first[:, :, 1], -first[:, :, 0]
    ux, uy = second[:, :, 1, 0], second[:, :, 1, 1]
    vx, vy = -second[:, :, 0, 0], -second[:, :, 0, 1]
    weight = fluid_area(cfg) / len(q)
    mass = weight * (u.T @ u + v.T @ v)
    stiffness = weight * (ux.T @ ux + uy.T @ uy + vx.T @ vx + vy.T @ vy)
    eigenvalues, coefficients = generalized_symmetric_eig(stiffness, mass, cfg.modes)
    spectrum = np.linalg.eigvalsh(0.5 * (mass + mass.T))
    rank = int(np.sum(spectrum > max(float(spectrum[-1]) * 1.0e-8, 1.0e-14)))
    return eigenvalues, coefficients, rank


def basis_fields(
    model: OfficialPeelingFields,
    points: np.ndarray,
    alpha: float,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.column_stack((points, np.full(len(points), alpha, dtype=np.float32))).astype(
        np.float32
    )

    def network(point):
        return model(point[:2].reshape(1, 2), point[2:3].reshape(1, 1))[0]

    raw, jacobian, _ = derivatives(network, z, 48, False)
    return (
        raw @ coefficients,
        jacobian[:, :, 1] @ coefficients,
        -jacobian[:, :, 0] @ coefficients,
    )


def weighted_projection(
    du: np.ndarray,
    dv: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    root = np.sqrt(weights)[:, None]
    matrix = np.vstack((root * basis_u, root * basis_v))
    target = np.r_[root[:, 0] * du, root[:, 0] * dv]
    return np.linalg.lstsq(matrix, target, rcond=1.0e-7)[0]


def quadrature_weights(points: np.ndarray) -> np.ndarray:
    distance, _ = cKDTree(points).query(points, k=9)
    weights = distance[:, -1] ** 2
    low, high = np.percentile(weights, [3.0, 97.0])
    weights = np.clip(weights, low, high)
    return (weights / np.mean(weights)).astype(np.float32)


def triangulation(
    points: np.ndarray, cylinder_x: float, radius: float
) -> mtri.Triangulation:
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    xyz = points[tri.triangles]
    centres = xyz.mean(axis=1)
    hole = (centres[:, 0] - cylinder_x) ** 2 + centres[:, 1] ** 2 < (1.02 * radius) ** 2
    edge = np.max(np.linalg.norm(xyz - np.roll(xyz, 1, axis=1), axis=2), axis=1) > 0.42
    tri.set_mask(hole | edge)
    return tri


def relative_vector_error(
    actual_u, actual_v, predicted_u, predicted_v, weights
) -> float:
    numerator = np.sum(
        weights * ((actual_u - predicted_u) ** 2 + (actual_v - predicted_v) ** 2)
    )
    denominator = np.sum(weights * (actual_u**2 + actual_v**2))
    return float(np.sqrt(numerator / max(denominator, 1.0e-24)))


def relative_scalar_error(actual, predicted, weights) -> float:
    numerator = np.sum(weights * (actual - predicted) ** 2)
    denominator = np.sum(weights * actual**2)
    return float(np.sqrt(numerator / max(denominator, 1.0e-24)))


def encode_mp4(frames: Path, output: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame_%03d.png"),
            "-vf",
            "scale=min(1920\\,iw):min(1080\\,ih):force_original_aspect_ratio=decrease,"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
