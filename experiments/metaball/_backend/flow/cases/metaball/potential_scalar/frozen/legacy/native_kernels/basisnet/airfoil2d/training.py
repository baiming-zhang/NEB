"""Unchanged airfoil geometry and sampling helpers.

Neural models and analytic derivatives are supplied by hardcoded_backend.
PyTorch remains here only for the existing geometry/sampling utilities.
"""












from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from _neb_paths import Path
from typing import Any

import numpy as np
import torch




CASE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_ROOT = CASE_ROOT / "results" / "k256_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
REDUCTION_DTYPE = torch.float64



@dataclass
class Config:
    seed: int = 20260725
    modes: int = 96
    epochs: int = 5000
    xmin: float = 0.0
    xmax: float = 10.0
    ymin: float = -4.0
    ymax: float = 4.0
    chord: float = 2.0
    quarter_chord_x: float = 3.0
    quarter_chord_y: float = 0.0
    thickness_min: float = 0.08
    thickness_max: float = 0.18
    geometry_mode: str = "c2_ellipse"
    alpha_min_deg: float = -12.0
    alpha_max_deg: float = 12.0
    hidden: int = 64
    hidden_layers: int = 4
    frequencies: int = 6
    points_per_step: int = 4096
    validation_points: int = 4096
    validation_beta_samples: int = 3
    validation_alpha_samples: int = 3
    probe_beta: float = 0.5
    probe_alpha_deg: float = 6.0
    probe_every: int = 100
    validate_every: int = 50
    checkpoint_every: int = 25
    learning_rate: float = 8.0e-4
    base_learning_rate: float = 8.0e-4
    early_stopping_patience_epochs: int = 1000
    lr_scheduler_patience_epochs: int = 200
    lr_scheduler_factor: float = 0.5
    min_learning_rate: float = 1.0e-6
    lr_scheduler_relative_threshold: float = 1.0e-4
    rayleigh_scale: float = 80.0
    grad_clip: float = 5.0
    mass_floor: float = 1.0e-10
    endpoint_period: int = 20
    envelope_mode: str = "localized"
    sampling_mode: str = "mixed"
    global_fraction: float = 0.50
    airfoil_fraction: float = 0.25
    wake_fraction: float = 0.25
    airfoil_band_chords: float = 0.55
    wake_length_chords: float = 3.0
    wake_half_height_chords: float = 0.55
    result_root: str = str(DEFAULT_RESULT_ROOT)

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def thickness_ratio(cfg: Config, beta: torch.Tensor | float) -> torch.Tensor | float:
    return cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta

def normalize_alpha(cfg: Config, alpha_deg: torch.Tensor | float):
    return 2.0 * (alpha_deg - cfg.alpha_min_deg) / (
        cfg.alpha_max_deg - cfg.alpha_min_deg
    ) - 1.0

def _expand_parameter(value: torch.Tensor, count: int) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1, 1).expand(count, 1)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.shape[0] == 1:
        return value.expand(count, 1)
    return value

def body_coordinates_torch(
    cfg: Config,
    xy: torch.Tensor,
    alpha_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    alpha_deg = _expand_parameter(alpha_deg, len(xy))


    angle = torch.deg2rad(-alpha_deg)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    dx = xy[:, 0:1] - cfg.quarter_chord_x
    dy = xy[:, 1:2] - cfg.quarter_chord_y
    xb = cosine * dx + sine * dy + 0.25 * cfg.chord
    yb = -sine * dx + cosine * dy
    return xb, yb

def body_coordinates_numpy(
    cfg: Config, points: np.ndarray, alpha_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    angle = np.deg2rad(-alpha_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    dx = points[:, 0] - cfg.quarter_chord_x
    dy = points[:, 1] - cfg.quarter_chord_y
    xb = cosine * dx + sine * dy + 0.25 * cfg.chord
    yb = -sine * dx + cosine * dy
    return xb, yb

def naca_half_thickness_torch(
    cfg: Config, x_over_c: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:

    beta = _expand_parameter(beta, len(x_over_c))
    s = torch.clamp(x_over_c, 0.0, 1.0)



    leading_edge_epsilon = 1.0e-6
    root = (
        torch.sqrt(s + leading_edge_epsilon)
        - math.sqrt(leading_edge_epsilon)
    ) / (
        math.sqrt(1.0 + leading_edge_epsilon)
        - math.sqrt(leading_edge_epsilon)
    )
    polynomial = (
        0.2969 * root
        - 0.1260 * s
        - 0.3516 * s.square()
        + 0.2843 * s.pow(3)
        - 0.1036 * s.pow(4)
    )
    return 5.0 * thickness_ratio(cfg, beta) * cfg.chord * polynomial

def naca_half_thickness_numpy(
    cfg: Config, x_over_c: np.ndarray, beta: float
) -> np.ndarray:
    s = np.clip(x_over_c, 0.0, 1.0)
    leading_edge_epsilon = 1.0e-6
    root = (
        np.sqrt(s + leading_edge_epsilon)
        - math.sqrt(leading_edge_epsilon)
    ) / (
        math.sqrt(1.0 + leading_edge_epsilon)
        - math.sqrt(leading_edge_epsilon)
    )
    polynomial = (
        0.2969 * root
        - 0.1260 * s
        - 0.3516 * s**2
        + 0.2843 * s**3
        - 0.1036 * s**4
    )
    return 5.0 * float(thickness_ratio(cfg, beta)) * cfg.chord * polynomial

def airfoil_level_torch(
    cfg: Config,
    xy: torch.Tensor,
    beta: torch.Tensor,
    alpha_deg: torch.Tensor,
) -> torch.Tensor:

    xb, yb = body_coordinates_torch(cfg, xy, alpha_deg)
    beta = _expand_parameter(beta, len(xy))
    thickness = thickness_ratio(cfg, beta)
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord).square()
        + (yb / semi_thickness).square()
        - 1.0
    )

def airfoil_level_numpy(
    cfg: Config, points: np.ndarray, beta: float, alpha_deg: float
) -> np.ndarray:
    xb, yb = body_coordinates_numpy(cfg, points, alpha_deg)
    thickness = float(thickness_ratio(cfg, beta))
    semi_chord = 0.5 * cfg.chord
    semi_thickness = 0.5 * thickness * cfg.chord
    return (
        ((xb - semi_chord) / semi_chord) ** 2
        + (yb / semi_thickness) ** 2
        - 1.0
    )

def inside_airfoil(
    cfg: Config, points: np.ndarray, beta: float, alpha_deg: float
) -> np.ndarray:
    return airfoil_level_numpy(cfg, points, beta, alpha_deg) <= 0.0

def airfoil_area(cfg: Config, beta: float) -> float:
    thickness = float(thickness_ratio(cfg, beta))
    return float(
        math.pi
        * (0.5 * cfg.chord)
        * (0.5 * thickness * cfg.chord)
    )

def fluid_area(cfg: Config, beta: float) -> float:
    return (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(
        cfg, beta
    )

def hard_open_outlet_envelope(
    cfg: Config,
    xy: torch.Tensor,
    beta: torch.Tensor,
    alpha_deg: torch.Tensor,
) -> torch.Tensor:

    x = xy[:, 0:1]
    scale = 0.25 * cfg.chord
    inlet_distance = torch.relu((x - cfg.xmin) / scale)
    inlet_cubic = inlet_distance.pow(3)
    inlet = inlet_cubic / (1.0 + inlet_cubic)
    clearance = torch.relu(
        airfoil_level_torch(cfg, xy, beta, alpha_deg)
    )
    clearance_cubic = clearance.pow(3)
    airfoil = clearance_cubic / (1.0 + clearance_cubic)
    return inlet * airfoil

def trailing_edge(cfg: Config, alpha_deg: float) -> tuple[float, float]:
    angle = math.radians(-alpha_deg)
    return (
        cfg.quarter_chord_x + 0.75 * cfg.chord * math.cos(angle),
        cfg.quarter_chord_y + 0.75 * cfg.chord * math.sin(angle),
    )

def _counts(n: int, fractions: list[float]) -> np.ndarray:
    values = np.asarray(fractions, dtype=np.float64)
    if np.any(values < 0.0) or float(values.sum()) <= 0.0:
        raise ValueError("sampling fractions must be nonnegative with positive sum")
    values /= values.sum()
    counts = np.floor(n * values).astype(int)
    counts[0] += n - int(counts.sum())
    return counts

def _sobol_fluid_box(
    cfg: Config,
    count: int,
    box: tuple[float, float, float, float],
    seed: int,
    beta: float,
    alpha_deg: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    xmin, xmax, ymin, ymax = box
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"empty sampling box: {box}")
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    pieces: list[np.ndarray] = []
    remaining = count
    while remaining:
        unit = engine.draw(max(3 * remaining, 128)).cpu().numpy()
        candidates = np.column_stack(
            (
                xmin + (xmax - xmin) * unit[:, 0],
                ymin + (ymax - ymin) * unit[:, 1],
            )
        ).astype(np.float32)
        accepted = candidates[
            ~inside_airfoil(cfg, candidates, beta, alpha_deg)
        ][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces)

def fluid_points(
    cfg: Config,
    n: int,
    beta: float,
    alpha_deg: float,
    seed: int,
) -> np.ndarray:

    if cfg.sampling_mode == "uniform":
        return _sobol_fluid_box(
            cfg,
            n,
            (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
            seed,
            beta,
            alpha_deg,
        )
    if cfg.sampling_mode != "mixed":
        raise ValueError(f"unknown sampling mode: {cfg.sampling_mode}")
    counts = _counts(
        n,
        [cfg.global_fraction, cfg.airfoil_fraction, cfg.wake_fraction],
    )
    half_x = 0.5 * cfg.chord + cfg.airfoil_band_chords * cfg.chord
    half_y = 0.5 * cfg.chord
    centre_x = cfg.quarter_chord_x + 0.25 * cfg.chord
    centre_y = cfg.quarter_chord_y
    tx, ty = trailing_edge(cfg, alpha_deg)
    global_points = _sobol_fluid_box(
        cfg,
        int(counts[0]),
        (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax),
        seed,
        beta,
        alpha_deg,
    )
    airfoil_points = _sobol_fluid_box(
        cfg,
        int(counts[1]),
        (
            max(cfg.xmin, centre_x - half_x),
            min(cfg.xmax, centre_x + half_x),
            max(cfg.ymin, centre_y - half_y),
            min(cfg.ymax, centre_y + half_y),
        ),
        seed + 1,
        beta,
        alpha_deg,
    )
    wake_points = _sobol_fluid_box(
        cfg,
        int(counts[2]),
        (
            min(cfg.xmax - 1.0e-6, tx + 0.02 * cfg.chord),
            min(cfg.xmax, tx + cfg.wake_length_chords * cfg.chord),
            max(
                cfg.ymin,
                ty - cfg.wake_half_height_chords * cfg.chord,
            ),
            min(
                cfg.ymax,
                ty + cfg.wake_half_height_chords * cfg.chord,
            ),
        ),
        seed + 2,
        beta,
        alpha_deg,
    )
    points = np.concatenate((global_points, airfoil_points, wake_points))
    order = np.random.default_rng(seed).permutation(len(points))
    return points[order]

def continuous_parameters(cfg: Config, epoch: int) -> tuple[float, float]:

    if cfg.endpoint_period > 0 and epoch % cfg.endpoint_period == 0:
        corners = (
            (0.0, cfg.alpha_min_deg),
            (0.0, cfg.alpha_max_deg),
            (1.0, cfg.alpha_min_deg),
            (1.0, cfg.alpha_max_deg),
        )
        return corners[(epoch // cfg.endpoint_period - 1) % len(corners)]
    engine = torch.quasirandom.SobolEngine(
        2, scramble=True, seed=cfg.seed + 700_000 + epoch
    )
    sample = engine.draw(1)[0]
    beta = float(sample[0])
    alpha_deg = cfg.alpha_min_deg + float(sample[1]) * (
        cfg.alpha_max_deg - cfg.alpha_min_deg
    )
    return beta, alpha_deg
