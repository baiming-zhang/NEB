"""K=96 2D airfoil training with the final 2D Karman SSS protocol.

The geometry has two independent continuous parameters: beta changes the
thickness of a symmetric NACA 00xx airfoil and alpha changes its angle of
attack.  The learning algorithm intentionally keeps the invariants of the
final 2D_karman case: one sinusoidal MLP and Adam optimizer per mode, fixed
mode order, causal least-squares peeling, shared field/gradient coefficients,
all Fourier features active from epoch one, and Rayleigh/80 as the only loss.
No Navier--Stokes field is opened during basis training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.func import jacfwd, vmap


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
    """Coordinates in the airfoil frame, rotating about quarter chord."""
    alpha_deg = _expand_parameter(alpha_deg, len(xy))
    # The leading edge is the nose (the airfoil faces left). Positive pitch
    # therefore rotates the body clockwise in an x-right/y-up plot.
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
    """Closed-trailing-edge NACA 00xx half thickness."""
    beta = _expand_parameter(beta, len(x_over_c))
    s = torch.clamp(x_over_c, 0.0, 1.0)
    # The analytic NACA leading edge has an infinite first derivative.  A
    # shifted epsilon regularization keeps y_t(0)=0 while making the second
    # spatial derivatives used by the Rayleigh loss finite.
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
    """C-infinity ellipse implicit function, positive outside the airfoil."""
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
    """Double-zero inlet/body envelope; outlet and far field are natural."""
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


class ScalarModeField(nn.Module):
    """One independent sinusoidal mode with joint beta/alpha conditioning."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Fourier x/y + beta + normalized alpha + body-relative x/y.
        in_features = 4 * cfg.frequencies + 4
        layers: list[nn.Linear] = [nn.Linear(in_features, cfg.hidden)]
        layers.extend(
            nn.Linear(cfg.hidden, cfg.hidden)
            for _ in range(cfg.hidden_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(cfg.hidden, 1)

    def encode(
        self,
        xy: torch.Tensor,
        beta: torch.Tensor,
        alpha_deg: torch.Tensor,
    ) -> torch.Tensor:
        beta = _expand_parameter(beta, len(xy))
        alpha_deg = _expand_parameter(alpha_deg, len(xy))
        xn = 2.0 * (xy[:, 0:1] - self.cfg.xmin) / (
            self.cfg.xmax - self.cfg.xmin
        ) - 1.0
        yn = 2.0 * (xy[:, 1:2] - self.cfg.ymin) / (
            self.cfg.ymax - self.cfg.ymin
        ) - 1.0
        spatial = torch.cat((xn, yn), dim=1)
        frequencies = 2.0 ** torch.arange(
            self.cfg.frequencies, device=xy.device, dtype=xy.dtype
        )
        phase = spatial.unsqueeze(-1) * frequencies
        fourier = torch.cat(
            (torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)), dim=1
        )
        xb, yb = body_coordinates_torch(self.cfg, xy, alpha_deg)
        body_relative = torch.cat(
            (
                2.0 * xb / self.cfg.chord - 1.0,
                2.0 * yb / self.cfg.chord,
            ),
            dim=1,
        )
        return torch.cat(
            (
                fourier,
                2.0 * beta - 1.0,
                normalize_alpha(self.cfg, alpha_deg),
                body_relative,
            ),
            dim=1,
        )

    def forward(
        self,
        xy: torch.Tensor,
        beta: torch.Tensor,
        alpha_deg: torch.Tensor,
    ) -> torch.Tensor:
        value = self.encode(xy, beta, alpha_deg)
        for layer in self.layers:
            value = torch.sin(layer(value))
        raw = self.output(value)
        return hard_open_outlet_envelope(
            self.cfg, xy, beta, alpha_deg
        ) * raw


class OfficialPeelingFields(nn.Module):
    """Fixed ModuleList; modes are never sorted, swapped, or shared."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.modes = nn.ModuleList(ScalarModeField(cfg) for _ in range(cfg.modes))

    def forward_mode(
        self,
        mode: int,
        xy: torch.Tensor,
        beta: torch.Tensor,
        alpha_deg: torch.Tensor,
    ) -> torch.Tensor:
        return self.modes[mode](xy, beta, alpha_deg)

    def forward(
        self,
        xy: torch.Tensor,
        beta: torch.Tensor,
        alpha_deg: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [mode(xy, beta, alpha_deg) for mode in self.modes], dim=1
        )


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
    """Final Karman 50/25/25 mixed Sobol sampler adapted to the airfoil."""
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
    """One deterministic continuous (beta, alpha) pair per epoch."""
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


def raw_mode_fields(
    model: OfficialPeelingFields,
    mode: int,
    points: np.ndarray | torch.Tensor,
    beta_value: float,
    alpha_deg_value: float,
    second_order: bool = True,
) -> dict[str, torch.Tensor]:
    """Evaluate spatial derivatives with beta and alpha held fixed.

    The Rayleigh objective needs derivatives with respect to x and y only.
    Keeping the geometry parameters outside ``jacfwd`` avoids constructing
    unused beta/alpha derivatives and mixed Hessian entries while preserving
    their role as continuous conditioning inputs.
    """
    xy = torch.as_tensor(
        points, device=DEVICE, dtype=DTYPE
    ).requires_grad_(True)
    beta = torch.as_tensor(
        beta_value, device=DEVICE, dtype=DTYPE
    ).reshape(1, 1)
    alpha = torch.as_tensor(
        alpha_deg_value, device=DEVICE, dtype=DTYPE
    ).reshape(1, 1)

    def one_point(point: torch.Tensor) -> torch.Tensor:
        return model.forward_mode(
            mode,
            point.reshape(1, 2),
            beta,
            alpha,
        )[0, 0]

    first_fn = jacfwd(one_point)
    psi = vmap(one_point)(xy)
    first = vmap(first_fn)(xy)
    result = {"psi": psi, "u": first[:, 1], "v": -first[:, 0]}
    if second_order:
        second = vmap(jacfwd(first_fn))(xy)
        result.update(
            {
                "ux": second[:, 1, 0],
                "uy": second[:, 1, 1],
                "vx": -second[:, 0, 0],
                "vy": -second[:, 0, 1],
            }
        )
    return result


def pack_velocity(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((fields["u"], fields["v"])).to(REDUCTION_DTYPE)


def pack_gradient(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (fields["ux"], fields["uy"], fields["vx"], fields["vy"])
    ).to(REDUCTION_DTYPE)


def least_squares_peel(
    raw_velocity: torch.Tensor,
    raw_gradient: torch.Tensor,
    raw_psi: torch.Tensor,
    previous_velocity: list[torch.Tensor],
    previous_gradient: list[torch.Tensor],
    previous_psi: list[torch.Tensor],
    weight: float,
    mass_floor: float,
) -> dict[str, torch.Tensor]:
    """The exact fixed-order Karman residual peeling operation."""
    if previous_velocity:
        u_previous = torch.stack(previous_velocity, dim=1)
        g_previous = torch.stack(previous_gradient, dim=1)
        psi_previous = torch.stack(previous_psi, dim=1)
        lhs = weight * (u_previous.T @ u_previous)
        rhs = weight * (u_previous.T @ raw_velocity)
        coefficients = torch.linalg.lstsq(
            lhs, rhs.unsqueeze(1)
        ).solution[:, 0]
        reconstructed_velocity = u_previous @ coefficients
        reconstructed_gradient = g_previous @ coefficients
        reconstructed_psi = psi_previous @ coefficients
    else:
        coefficients = torch.empty(
            0, device=raw_velocity.device, dtype=raw_velocity.dtype
        )
        reconstructed_velocity = torch.zeros_like(raw_velocity)
        reconstructed_gradient = torch.zeros_like(raw_gradient)
        reconstructed_psi = torch.zeros_like(raw_psi)
    residual_velocity = raw_velocity - reconstructed_velocity
    residual_gradient = raw_gradient - reconstructed_gradient
    residual_psi = raw_psi - reconstructed_psi
    mass = weight * residual_velocity.square().sum()
    stiffness = weight * residual_gradient.square().sum()
    projection_mass = weight * reconstructed_velocity.square().sum()
    normalizer = torch.rsqrt(torch.clamp(mass, min=mass_floor))
    return {
        "q": coefficients,
        "velocity": residual_velocity,
        "gradient": residual_gradient,
        "psi": residual_psi,
        "normalized_velocity": residual_velocity * normalizer,
        "normalized_gradient": residual_gradient * normalizer,
        "normalized_psi": residual_psi * normalizer,
        "mass": mass,
        "stiffness": stiffness,
        "projection_mass": projection_mass,
    }


def peel_parameters(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray,
    beta_value: float,
    alpha_deg_value: float,
    optimizers: list[torch.optim.Optimizer] | None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    weight = fluid_area(cfg, beta_value) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_gradient: list[torch.Tensor] = []
    previous_psi: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    energies: list[float] = []
    projection_ratios: list[float] = []
    losses: list[float] = []
    for mode in range(cfg.modes):
        raw = raw_mode_fields(
            model, mode, points, beta_value, alpha_deg_value
        )
        raw_velocity = pack_velocity(raw)
        raw_gradient = pack_gradient(raw)
        raw_psi = raw["psi"].to(REDUCTION_DTYPE)
        raw_velocities.append(raw_velocity.detach())
        peeled = least_squares_peel(
            raw_velocity,
            raw_gradient,
            raw_psi,
            previous_velocity,
            previous_gradient,
            previous_psi,
            weight,
            cfg.mass_floor,
        )
        denominator = torch.clamp(peeled["mass"], min=cfg.mass_floor)
        rayleigh = peeled["stiffness"] / denominator
        projection_ratio = peeled["projection_mass"] / denominator
        loss = rayleigh / cfg.rayleigh_scale
        if optimizers is not None:
            optimizers[mode].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.modes[mode].parameters(), cfg.grad_clip
            )
            optimizers[mode].step()
        energies.append(float(rayleigh.detach().cpu()))
        projection_ratios.append(float(projection_ratio.detach().cpu()))
        losses.append(float(loss.detach().cpu()))
        previous_velocity.append(peeled["normalized_velocity"].detach())
        previous_gradient.append(peeled["normalized_gradient"].detach())
        previous_psi.append(peeled["normalized_psi"].detach())
    velocity_matrix = torch.stack(previous_velocity, dim=1)
    gram = weight * velocity_matrix.T @ velocity_matrix
    raw_matrix = torch.stack(raw_velocities, dim=1)
    raw_gram = weight * raw_matrix.T @ raw_matrix
    eigenvalues = torch.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    threshold = torch.clamp(
        eigenvalues[-1], min=cfg.mass_floor
    ) * 1.0e-8
    diagnostics: dict[str, Any] = {
        "beta": beta_value,
        "alpha_deg": alpha_deg_value,
        "loss": float(np.mean(losses)),
        "mean_energy": float(np.mean(energies)),
        "max_energy": float(np.max(energies)),
        "min_energy": float(np.min(energies)),
        "mean_projection_ratio": float(np.mean(projection_ratios)),
        "max_projection_ratio": float(np.max(projection_ratios)),
        "orth_error": float(
            torch.max(
                torch.abs(
                    gram
                    - torch.eye(
                        cfg.modes, device=DEVICE, dtype=REDUCTION_DTYPE
                    )
                )
            ).cpu()
        ),
        "effective_raw_rank": int((eigenvalues > threshold).sum().cpu()),
        "mode_energies": energies,
        "projection_ratios": projection_ratios,
        "fixed_mode_order": list(range(cfg.modes)),
    }
    n = len(points)
    output = {
        "psi": torch.stack(previous_psi, dim=1),
        "u": velocity_matrix[:n],
        "v": velocity_matrix[n:],
        "gram": gram,
    }
    mean_loss = torch.tensor(
        diagnostics["loss"], device=DEVICE, dtype=DTYPE
    )
    return mean_loss, diagnostics, output


def infer_basis(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
) -> dict[str, np.ndarray | float | int]:
    """One-shot joint-parameter inference without optimization."""
    model.eval()
    weight = fluid_area(cfg, beta) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_psi: list[torch.Tensor] = []
    for mode in range(cfg.modes):
        raw = raw_mode_fields(
            model, mode, points, beta, alpha_deg, second_order=False
        )
        raw_velocity = pack_velocity(raw)
        raw_psi = raw["psi"].to(REDUCTION_DTYPE)
        if previous_velocity:
            earlier = torch.stack(previous_velocity, dim=1)
            earlier_psi = torch.stack(previous_psi, dim=1)
            lhs = weight * (earlier.T @ earlier)
            rhs = weight * (earlier.T @ raw_velocity)
            q = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution[:, 0]
            residual_velocity = raw_velocity - earlier @ q
            residual_psi = raw_psi - earlier_psi @ q
        else:
            residual_velocity, residual_psi = raw_velocity, raw_psi
        normalizer = torch.rsqrt(
            torch.clamp(
                weight * residual_velocity.square().sum(),
                min=cfg.mass_floor,
            )
        )
        previous_velocity.append((residual_velocity * normalizer).detach())
        previous_psi.append((residual_psi * normalizer).detach())
    velocity = torch.stack(previous_velocity, dim=1)
    n = len(points)
    gram = weight * velocity.T @ velocity
    return {
        "beta": float(beta),
        "alpha_deg": float(alpha_deg),
        "psi": torch.stack(previous_psi, dim=1).cpu().numpy(),
        "u": velocity[:n].cpu().numpy(),
        "v": velocity[n:].cpu().numpy(),
        "gram": gram.cpu().numpy(),
        "rank": int(torch.linalg.matrix_rank(gram).cpu()),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "loss": float(np.mean([item["loss"] for item in items])),
        "mean_energy": float(
            np.mean([item["mean_energy"] for item in items])
        ),
        "max_energy": float(np.max([item["max_energy"] for item in items])),
        "mean_projection_ratio": float(
            np.mean([item["mean_projection_ratio"] for item in items])
        ),
        "orth_error": float(np.max([item["orth_error"] for item in items])),
        "min_rank": int(
            np.min([item["effective_raw_rank"] for item in items])
        ),
    }


def validation_pass(
    cfg: Config, model: OfficialPeelingFields
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    details: list[dict[str, Any]] = []
    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg,
            cfg.alpha_max_deg,
            cfg.validation_alpha_samples,
        )
    ]
    for index, (beta, alpha_deg) in enumerate(pairs):
        points = fluid_points(
            cfg,
            cfg.validation_points,
            beta,
            alpha_deg,
            cfg.seed + 900_000 + index,
        )
        _, diagnostic, _ = peel_parameters(
            cfg, model, points, beta, alpha_deg, optimizers=None
        )
        details.append(diagnostic)
    model.train()
    return aggregate(details), details


def probe_pass(
    cfg: Config, model: OfficialPeelingFields
) -> dict[str, Any]:
    model.eval()
    points = fluid_points(
        cfg,
        cfg.validation_points,
        cfg.probe_beta,
        cfg.probe_alpha_deg,
        cfg.seed + 1_500_000,
    )
    _, result, _ = peel_parameters(
        cfg,
        model,
        points,
        cfg.probe_beta,
        cfg.probe_alpha_deg,
        optimizers=None,
    )
    model.train()
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.split("_")[-1])
        for path in root.glob("run_*")
        if path.name.split("_")[-1].isdigit()
    ]
    return root / f"run_{max(existing, default=0) + 1:03d}"


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data = run_dir / "data"
    data.mkdir(exist_ok=True)
    (data / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    keys = [
        "epoch",
        "seconds",
        "beta",
        "alpha_deg",
        "train_loss",
        "train_mean_energy",
        "train_max_energy",
        "train_mean_projection_ratio",
        "train_orth_error",
        "train_rank",
        "validation_loss",
        "validation_mean_energy",
        "validation_max_energy",
        "validation_mean_projection_ratio",
        "validation_orth_error",
        "validation_min_rank",
        "probe_loss",
        "probe_mean_energy",
        "probe_max_energy",
        "probe_mean_projection_ratio",
        "probe_orth_error",
        "probe_rank",
    ]
    with (data / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in keys} for row in history
        )


def save_checkpoint(
    path: Path,
    cfg: Config,
    model: OfficialPeelingFields,
    optimizers: list[torch.optim.Optimizer],
    epoch: int,
    history: list[dict[str, Any]],
    best_validation: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": (
            "official_sss_fixed_order_residual_peeling_joint_beta_alpha"
        ),
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "epoch": epoch,
        "history": history,
        "best_validation": best_validation,
        "training_uses_gt_or_ns": False,
        "inference_requires_optimization": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(cfg: Config, run_dir: Path, resume: Path | None = None) -> None:
    set_seed(cfg.seed)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    model = OfficialPeelingFields(cfg).to(DEVICE)
    optimizers = [
        torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate)
        for mode in model.modes
    ]
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    start_epoch = 1
    if resume is not None:
        payload = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model"])
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        history = payload.get("history", [])
        best_validation = float(payload.get("best_validation", math.inf))
        start_epoch = int(payload["epoch"]) + 1
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    protocol = {
        "reference_case": "BasisNet/cases/2D_karman final protocol",
        "one_independent_network_per_mode": True,
        "one_optimizer_per_mode": True,
        "fixed_mode_order": True,
        "mode_sorting_used": False,
        "least_squares_residual_peeling": True,
        "field_and_gradient_share_projection_coefficients": True,
        "previous_modes_detached": True,
        "loss": "mean_k[(||grad u_k||^2 / ||u_k||^2) / 80]",
        "projection_penalty_used_in_loss": False,
        "frequency_curriculum": False,
        "constant_learning_rate": cfg.learning_rate,
        "training_uses_gt_or_ns": False,
        "training_uses_mesh_or_grid": False,
        "geometry_parameterization": (
            "continuous analytic G(x,y; beta,alpha), no mesh representation"
        ),
        "randomized_spatial_points_per_epoch": cfg.points_per_step,
        "spatial_sampling": (
            "fresh deterministic scrambled Sobol; 50% global, "
            "25% analytic-airfoil neighborhood, 25% wake"
        ),
        "joint_continuous_parameter_sampling": ["beta", "alpha_deg"],
        "one_shot_joint_parameter_inference": True,
        "geometry": "symmetric NACA 00xx, beta controls thickness",
        "airflow": (
            "left-to-right; natural top/bottom atmospheric far field "
            "and natural right outlet"
        ),
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    append_log(
        run_dir,
        f"start official-peeling device={DEVICE} modes={cfg.modes} "
        f"parameters={parameter_count(model):,} points={cfg.points_per_step}",
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs + 1):
        beta, alpha_deg = continuous_parameters(cfg, epoch)
        points = fluid_points(
            cfg,
            cfg.points_per_step,
            beta,
            alpha_deg,
            cfg.seed + epoch * 1009,
        )
        model.train()
        _, diagnostic, _ = peel_parameters(
            cfg, model, points, beta, alpha_deg, optimizers
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "beta": beta,
            "alpha_deg": alpha_deg,
            "train_loss": diagnostic["loss"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic[
                "mean_projection_ratio"
            ],
            "train_orth_error": diagnostic["orth_error"],
            "train_rank": diagnostic["effective_raw_rank"],
        }
        if epoch == 1 or epoch % cfg.validate_every == 0:
            validation, details = validation_pass(cfg, model)
            row.update(
                {
                    "validation_loss": validation["loss"],
                    "validation_mean_energy": validation["mean_energy"],
                    "validation_max_energy": validation["max_energy"],
                    "validation_mean_projection_ratio": validation[
                        "mean_projection_ratio"
                    ],
                    "validation_orth_error": validation["orth_error"],
                    "validation_min_rank": validation["min_rank"],
                    "validation_mode_energies": [
                        item["mode_energies"] for item in details
                    ],
                }
            )
            if float(validation["loss"]) < best_validation:
                best_validation = float(validation["loss"])
                save_checkpoint(
                    run_dir / "checkpoints" / "best_model.pt",
                    cfg,
                    model,
                    optimizers,
                    epoch,
                    history + [row],
                    best_validation,
                )
            append_log(
                run_dir,
                f"epoch={epoch:04d} beta={beta:.5f} "
                f"alpha={alpha_deg:+.3f} train={diagnostic['loss']:.6e} "
                f"val={validation['loss']:.6e} "
                f"proj={validation['mean_projection_ratio']:.3e} "
                f"orth={validation['orth_error']:.2e} "
                f"rank={validation['min_rank']}",
            )
        if cfg.probe_every > 0 and epoch % cfg.probe_every == 0:
            probe = probe_pass(cfg, model)
            row.update(
                {
                    "probe_loss": probe["loss"],
                    "probe_mean_energy": probe["mean_energy"],
                    "probe_max_energy": probe["max_energy"],
                    "probe_mean_projection_ratio": probe[
                        "mean_projection_ratio"
                    ],
                    "probe_orth_error": probe["orth_error"],
                    "probe_rank": probe["effective_raw_rank"],
                }
            )
        history.append(row)
        if epoch == 1 or epoch % cfg.checkpoint_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                cfg,
                model,
                optimizers,
                epoch,
                history,
                best_validation,
            )
            save_history(run_dir, history)
    frozen = run_dir / "checkpoints" / "sss_official_peeling_frozen.pt"
    save_checkpoint(
        frozen, cfg, model, optimizers, cfg.epochs, history, best_validation
    )
    save_history(run_dir, history)
    append_log(run_dir, f"training complete frozen={frozen}")


def self_test() -> None:
    cfg = Config(
        modes=3,
        hidden=12,
        hidden_layers=2,
        frequencies=3,
        points_per_step=16,
        validation_points=16,
        validation_beta_samples=2,
        validation_alpha_samples=2,
        epochs=1,
    )
    set_seed(cfg.seed)
    model = OfficialPeelingFields(cfg).to(DEVICE)
    optimizers = [
        torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate)
        for mode in model.modes
    ]
    beta, alpha_deg = 0.37, 5.5
    points = fluid_points(
        cfg, cfg.points_per_step, beta, alpha_deg, cfg.seed + 1
    )
    raw0 = raw_mode_fields(model, 0, points, beta, alpha_deg)
    peeled0 = least_squares_peel(
        pack_velocity(raw0),
        pack_gradient(raw0),
        raw0["psi"].to(REDUCTION_DTYPE),
        [],
        [],
        [],
        fluid_area(cfg, beta) / len(points),
        cfg.mass_floor,
    )
    raw1 = raw_mode_fields(model, 1, points, beta, alpha_deg)
    peeled1 = least_squares_peel(
        pack_velocity(raw1),
        pack_gradient(raw1),
        raw1["psi"].to(REDUCTION_DTYPE),
        [peeled0["normalized_velocity"].detach()],
        [peeled0["normalized_gradient"].detach()],
        [peeled0["normalized_psi"].detach()],
        fluid_area(cfg, beta) / len(points),
        cfg.mass_floor,
    )
    model.zero_grad(set_to_none=True)
    causal_loss = peeled1["stiffness"] / torch.clamp(
        peeled1["mass"], min=cfg.mass_floor
    )
    causal_loss.backward()
    if any(
        parameter.grad is not None
        and float(parameter.grad.abs().max().cpu()) > 0.0
        for parameter in model.modes[0].parameters()
    ):
        raise RuntimeError("later-mode loss leaked into an earlier mode")
    if not any(
        parameter.grad is not None
        and float(parameter.grad.abs().max().cpu()) > 0.0
        for parameter in model.modes[1].parameters()
    ):
        raise RuntimeError("peeled mode received no gradient")
    model.zero_grad(set_to_none=True)
    _, diagnostic, _ = peel_parameters(
        cfg, model, points, beta, alpha_deg, optimizers
    )
    inferred = infer_basis(cfg, model, points, 0.63, -4.0)
    if inferred["u"].shape != (cfg.points_per_step, cfg.modes):
        raise RuntimeError("joint one-shot inference returned wrong shape")
    if not math.isfinite(diagnostic["loss"]):
        raise RuntimeError("non-finite peeling loss")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "device": str(DEVICE),
                "joint_parameters": ["beta", "alpha_deg"],
                "fixed_order": diagnostic["fixed_mode_order"],
                "orth_error": diagnostic["orth_error"],
                "rank": diagnostic["effective_raw_rank"],
                "causal_stop_gradient": "passed",
                "all_frequency_features_active": True,
                "one_shot_shape": list(inferred["u"].shape),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--resume", type=Path)
    for name in (
        "modes",
        "epochs",
        "hidden",
        "hidden_layers",
        "frequencies",
        "points_per_step",
        "validation_points",
        "validation_beta_samples",
        "validation_alpha_samples",
        "probe_every",
        "validate_every",
        "checkpoint_every",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--probe-beta", type=float)
    parser.add_argument("--probe-alpha-deg", type=float)
    parser.add_argument("--envelope-mode", choices=("localized",))
    parser.add_argument("--sampling-mode", choices=("mixed", "uniform"))
    parser.add_argument("--global-fraction", type=float)
    parser.add_argument("--airfoil-fraction", type=float)
    parser.add_argument("--wake-fraction", type=float)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    values: dict[str, Any] = {}
    valid_fields = set(Config.__dataclass_fields__)
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = sorted(set(loaded) - valid_fields)
        if unknown:
            raise ValueError(f"unknown config fields: {', '.join(unknown)}")
        values.update(loaded)
    for field in valid_fields:
        if hasattr(args, field):
            value = getattr(args, field)
            if value is not None:
                values[field] = value
    if args.result_root is not None:
        values["result_root"] = str(args.result_root)
    return Config(**values)


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    cfg = config_from_args(args)
    root = Path(cfg.result_root)
    run_dir = args.run_dir if args.run_dir is not None else make_run_dir(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, run_dir, args.resume)


if __name__ == "__main__":
    main()
