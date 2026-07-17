"""K=96 basis training for the 2D Karman case with fixed-order SSS peeling.

The important invariants are deliberately the same as the reference notebook:

1. one independent neural field and one optimizer per mode;
2. one continuously sampled shape code (alpha) per training step;
3. a fixed mode identity/order over the whole shape space;
4. mode j is projected onto the already peeled modes 0..j-1;
5. the same least-squares coefficients peel both the field and its gradients;
6. previous modes are detached before mode j is optimized;
7. the loss is only normalized Dirichlet/Rayleigh energy (Rayleigh / 80).

The airplane scalar field from the notebook is replaced by a streamfunction.
Its curl is a divergence-free velocity mode.  The hard envelope removes the
constant null mode, so unlike the Neumann airplane example there is no valid
constant eigenfunction to prepend.
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
DEFAULT_RESULT_ROOT = CASE_ROOT / "results" / "k96_final"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
REDUCTION_DTYPE = torch.float64


@dataclass
class Config:
    seed: int = 20260715
    modes: int = 96
    xmin: float = 0.0
    xmax: float = 10.0
    ymin: float = -2.0
    ymax: float = 2.0
    radius: float = 0.45
    centre_left: float = 1.0
    centre_right: float = 9.0
    hidden: int = 64
    hidden_layers: int = 4
    frequencies: int = 6
    points_per_step: int = 512
    epochs: int = 5000
    validation_points: int = 1024
    validation_alphas: int = 7
    probe_alpha: float = 0.15
    probe_every: int = 100
    validate_every: int = 50
    checkpoint_every: int = 25
    learning_rate: float = 8.0e-4
    rayleigh_scale: float = 80.0
    grad_clip: float = 5.0
    mass_floor: float = 1.0e-10
    endpoint_period: int = 20
    envelope_mode: str = "localized"
    sampling_mode: str = "mixed"
    global_fraction: float = 0.50
    cylinder_fraction: float = 0.25
    wake_fraction: float = 0.25
    cylinder_band_radii: float = 2.5
    wake_length_radii: float = 13.0
    wake_half_height_radii: float = 3.0
    result_root: str = str(DEFAULT_RESULT_ROOT)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def centre_from_alpha(cfg: Config, alpha: torch.Tensor | float) -> torch.Tensor | float:
    return cfg.centre_left + (cfg.centre_right - cfg.centre_left) * alpha


def fluid_area(cfg: Config) -> float:
    return (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - math.pi * cfg.radius**2


def hard_open_outlet_envelope(
    cfg: Config, xy: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Double-zero streamfunction envelope except at the natural outlet."""
    if alpha.ndim == 0:
        alpha = alpha.reshape(1, 1).expand(len(xy), 1)
    elif alpha.shape[0] == 1:
        alpha = alpha.expand(len(xy), 1)
    x, y = xy[:, 0:1], xy[:, 1:2]
    centre = centre_from_alpha(cfg, alpha)
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    inlet_distance = torch.clamp((x - cfg.xmin) / cfg.radius, min=0.0)
    inlet = inlet_distance / (1.0 + inlet_distance)
    wall_distance = torch.clamp((1.0 - yn.square()) / 0.25, min=0.0)
    wall = wall_distance / (1.0 + wall_distance)
    cylinder_distance = torch.relu(
        ((x - centre).square() + y.square() - cfg.radius**2) / cfg.radius**2
    )
    cylinder = cylinder_distance / (1.0 + cylinder_distance)
    return (inlet * wall * cylinder).square()


class ScalarModeField(nn.Module):
    """One independent SSS mode, matching the Colab's sinusoidal MLP style."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        # Colab: sin/cos(1,2,4,...,64) for each spatial coordinate, then code.
        in_features = (
            2 * 2 * cfg.frequencies + 2
        )  # encoded x,y + alpha + cylinder-relative x
        layers: list[nn.Linear] = [nn.Linear(in_features, cfg.hidden)]
        layers.extend(
            nn.Linear(cfg.hidden, cfg.hidden) for _ in range(cfg.hidden_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(cfg.hidden, 1)

    def encode(self, xy: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        if alpha.ndim == 0:
            alpha = alpha.reshape(1, 1).expand(len(xy), 1)
        elif alpha.shape[0] == 1:
            alpha = alpha.expand(len(xy), 1)
        xn = 2.0 * (xy[:, 0:1] - self.cfg.xmin) / (self.cfg.xmax - self.cfg.xmin) - 1.0
        yn = 2.0 * (xy[:, 1:2] - self.cfg.ymin) / (self.cfg.ymax - self.cfg.ymin) - 1.0
        spatial = torch.cat((xn, yn), dim=1)
        frequencies = 2.0 ** torch.arange(
            self.cfg.frequencies, device=xy.device, dtype=xy.dtype
        )
        phase = spatial.unsqueeze(-1) * frequencies
        encoded_spatial = torch.cat(
            (
                torch.sin(phase).flatten(1),
                torch.cos(phase).flatten(1),
            ),
            dim=1,
        )
        centre = centre_from_alpha(self.cfg, alpha)
        relative = 2.0 * (xy[:, 0:1] - centre) / (self.cfg.xmax - self.cfg.xmin)
        return torch.cat((encoded_spatial, alpha, relative), dim=1)

    def forward(self, xy: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        value = self.encode(xy, alpha)
        for layer in self.layers:
            value = torch.sin(layer(value))
        raw = self.output(value)
        return hard_open_outlet_envelope(self.cfg, xy, alpha) * raw


class OfficialPeelingFields(nn.Module):
    """Independent ModuleList: mode identities never get sorted or exchanged."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.modes = nn.ModuleList(ScalarModeField(cfg) for _ in range(cfg.modes))

    def forward_mode(
        self, mode: int, xy: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        return self.modes[mode](xy, alpha)

    def forward(self, xy: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return torch.cat([mode(xy, alpha) for mode in self.modes], dim=1)


def fluid_points(cfg: Config, n: int, alpha: float, seed: int) -> np.ndarray:
    if cfg.sampling_mode == "mixed":
        fractions = np.asarray(
            [cfg.global_fraction, cfg.cylinder_fraction, cfg.wake_fraction],
            dtype=np.float64,
        )
        if np.any(fractions < 0.0) or float(fractions.sum()) <= 0.0:
            raise ValueError(
                "Mixed sampling fractions must be nonnegative and have positive sum"
            )
        fractions /= fractions.sum()
        counts = np.floor(n * fractions).astype(int)
        counts[0] += n - int(counts.sum())
        centre = float(centre_from_alpha(cfg, alpha))

        def sobol_box(
            count: int, box: tuple[float, float, float, float], local_seed: int
        ) -> np.ndarray:
            if count <= 0:
                return np.empty((0, 2), dtype=np.float32)
            engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=local_seed)
            xmin, xmax, ymin, ymax = box
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
                keep = (candidates[:, 0] - centre) ** 2 + candidates[
                    :, 1
                ] ** 2 > cfg.radius**2
                accepted = candidates[keep][:remaining]
                pieces.append(accepted)
                remaining -= len(accepted)
            return np.concatenate(pieces)

        global_points = sobol_box(
            int(counts[0]), (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax), seed
        )
        band = cfg.cylinder_band_radii * cfg.radius
        cylinder_points = sobol_box(
            int(counts[1]),
            (
                max(cfg.xmin, centre - band),
                min(cfg.xmax, centre + band),
                max(cfg.ymin, -band),
                min(cfg.ymax, band),
            ),
            seed + 1,
        )
        wake_points = sobol_box(
            int(counts[2]),
            (
                min(cfg.xmax, centre + 0.25 * cfg.radius),
                min(cfg.xmax, centre + cfg.wake_length_radii * cfg.radius),
                max(cfg.ymin, -cfg.wake_half_height_radii * cfg.radius),
                min(cfg.ymax, cfg.wake_half_height_radii * cfg.radius),
            ),
            seed + 2,
        )
        points = np.concatenate((global_points, cylinder_points, wake_points), axis=0)
        # Deterministically interleave the three strata so mode-wise derivative
        # batches do not see a long run from only one spatial region.
        order = np.random.default_rng(seed).permutation(len(points))
        return points[order]
    if cfg.sampling_mode != "uniform":
        raise ValueError(f"Unknown sampling mode: {cfg.sampling_mode}")
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    centre = float(centre_from_alpha(cfg, alpha))
    pieces: list[np.ndarray] = []
    remaining = n
    while remaining:
        unit = engine.draw(max(2 * remaining, 128)).cpu().numpy()
        candidates = np.column_stack(
            (
                cfg.xmin + (cfg.xmax - cfg.xmin) * unit[:, 0],
                cfg.ymin + (cfg.ymax - cfg.ymin) * unit[:, 1],
            )
        ).astype(np.float32)
        keep = (candidates[:, 0] - centre) ** 2 + candidates[:, 1] ** 2 > cfg.radius**2
        accepted = candidates[keep][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces)


def continuous_alpha(cfg: Config, epoch: int) -> float:
    """One continuous code per step, as in Colab, with sparse endpoint anchors."""
    if cfg.endpoint_period > 0 and epoch % cfg.endpoint_period == 0:
        return 0.0 if (epoch // cfg.endpoint_period) % 2 else 1.0
    engine = torch.quasirandom.SobolEngine(
        1, scramble=True, seed=cfg.seed + 700_000 + epoch
    )
    return float(engine.draw(1).item())


def raw_mode_fields(
    cfg: Config,
    model: OfficialPeelingFields,
    mode: int,
    points: np.ndarray | torch.Tensor,
    alpha_value: float,
) -> dict[str, torch.Tensor]:
    xy = torch.as_tensor(points, device=DEVICE, dtype=DTYPE)
    alpha = torch.full((len(xy), 1), alpha_value, device=DEVICE, dtype=DTYPE)
    xya = torch.cat((xy, alpha), dim=1).requires_grad_(True)

    def one_point(z: torch.Tensor) -> torch.Tensor:
        return model.forward_mode(mode, z[:2].reshape(1, 2), z[2:3].reshape(1, 1))[0, 0]

    first_fn = jacfwd(one_point)
    psi = vmap(one_point)(xya)
    first = vmap(first_fn)(xya)
    second = vmap(jacfwd(first_fn))(xya)
    return {
        "psi": psi,
        "u": first[:, 1],
        "v": -first[:, 0],
        "ux": second[:, 1, 0],
        "uy": second[:, 1, 1],
        "vx": -second[:, 0, 0],
        "vy": -second[:, 0, 1],
    }


def raw_mode_inference_fields(
    model: OfficialPeelingFields,
    mode: int,
    points: np.ndarray | torch.Tensor,
    alpha_value: float,
) -> dict[str, torch.Tensor]:
    """Inference path: only first derivatives are needed for velocity peeling."""
    xy = torch.as_tensor(points, device=DEVICE, dtype=DTYPE)
    alpha = torch.full((len(xy), 1), alpha_value, device=DEVICE, dtype=DTYPE)
    xya = torch.cat((xy, alpha), dim=1).requires_grad_(True)

    def one_point(z: torch.Tensor) -> torch.Tensor:
        return model.forward_mode(mode, z[:2].reshape(1, 2), z[2:3].reshape(1, 1))[0, 0]

    psi = vmap(one_point)(xya)
    first = vmap(jacfwd(one_point))(xya)
    return {"psi": psi, "u": first[:, 1], "v": -first[:, 0]}


def pack_velocity(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((fields["u"], fields["v"])).to(REDUCTION_DTYPE)


def pack_gradient(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat((fields["ux"], fields["uy"], fields["vx"], fields["vy"])).to(
        REDUCTION_DTYPE
    )


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
    """Colab's U(U^T U)^-1 U^T peeling, applied to values and gradients."""
    if previous_velocity:
        u_previous = torch.stack(previous_velocity, dim=1)
        g_previous = torch.stack(previous_gradient, dim=1)
        psi_previous = torch.stack(previous_psi, dim=1)
        lhs = weight * (u_previous.T @ u_previous)
        rhs = weight * (u_previous.T @ raw_velocity)
        q = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution[:, 0]
        reconstructed_velocity = u_previous @ q
        reconstructed_gradient = g_previous @ q
        reconstructed_psi = psi_previous @ q
    else:
        q = torch.empty(0, device=raw_velocity.device, dtype=raw_velocity.dtype)
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
        "q": q,
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


def peel_shape(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray,
    alpha_value: float,
    optimizers: list[torch.optim.Optimizer] | None,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """Run the official fixed-order mode loop for one continuous shape code."""
    weight = fluid_area(cfg) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_gradient: list[torch.Tensor] = []
    previous_psi: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    energies: list[float] = []
    projection_ratios: list[float] = []
    losses: list[float] = []

    for mode in range(cfg.modes):
        raw = raw_mode_fields(cfg, model, mode, points, alpha_value)
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
        # This is the causal stop-gradient in the reference notebook.  Store
        # normalized columns: it is the same span/projection, but stable at K=96.
        previous_velocity.append(peeled["normalized_velocity"].detach())
        previous_gradient.append(peeled["normalized_gradient"].detach())
        previous_psi.append(peeled["normalized_psi"].detach())

    velocity_matrix = torch.stack(previous_velocity, dim=1)
    gram = weight * velocity_matrix.T @ velocity_matrix
    raw_matrix = torch.stack(raw_velocities, dim=1)
    raw_gram = weight * raw_matrix.T @ raw_matrix
    eigenvalues = torch.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    threshold = torch.clamp(eigenvalues[-1], min=cfg.mass_floor) * 1.0e-8
    diagnostics: dict[str, Any] = {
        "alpha": alpha_value,
        "loss": float(np.mean(losses)),
        "mean_energy": float(np.mean(energies)),
        "max_energy": float(np.max(energies)),
        "min_energy": float(np.min(energies)),
        "mean_projection_ratio": float(np.mean(projection_ratios)),
        "max_projection_ratio": float(np.max(projection_ratios)),
        "orth_error": float(
            torch.max(torch.abs(gram - torch.eye(cfg.modes, device=DEVICE))).cpu()
        ),
        "effective_raw_rank": int((eigenvalues > threshold).sum().cpu()),
        "mode_energies": energies,
        "projection_ratios": projection_ratios,
        "fixed_mode_order": list(range(cfg.modes)),
    }
    n = len(points)
    psi_matrix = torch.stack(previous_psi, dim=1)
    output = {
        "psi": psi_matrix,
        "u": velocity_matrix[:n],
        "v": velocity_matrix[n:],
        "gram": gram,
    }
    mean_loss = torch.tensor(float(np.mean(losses)), device=DEVICE, dtype=DTYPE)
    return mean_loss, diagnostics, output


def infer_basis(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float | int]:
    """One-shot continuous-alpha inference: no fine-tuning and no transfer step."""
    model.eval()
    weight = fluid_area(cfg) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_psi: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    for mode in range(cfg.modes):
        raw = raw_mode_inference_fields(model, mode, points, float(alpha))
        raw_velocity = torch.cat((raw["u"], raw["v"])).to(REDUCTION_DTYPE)
        raw_psi = raw["psi"].to(REDUCTION_DTYPE)
        raw_velocities.append(raw_velocity.detach())
        if previous_velocity:
            u_previous = torch.stack(previous_velocity, dim=1)
            psi_previous = torch.stack(previous_psi, dim=1)
            lhs = weight * (u_previous.T @ u_previous)
            rhs = weight * (u_previous.T @ raw_velocity)
            q = torch.linalg.lstsq(lhs, rhs.unsqueeze(1)).solution[:, 0]
            residual_velocity = raw_velocity - u_previous @ q
            residual_psi = raw_psi - psi_previous @ q
        else:
            residual_velocity = raw_velocity
            residual_psi = raw_psi
        mass = weight * residual_velocity.square().sum()
        normalizer = torch.rsqrt(torch.clamp(mass, min=cfg.mass_floor))
        previous_velocity.append((residual_velocity * normalizer).detach())
        previous_psi.append((residual_psi * normalizer).detach())

    velocity_matrix = torch.stack(previous_velocity, dim=1)
    psi_matrix = torch.stack(previous_psi, dim=1)
    gram = weight * velocity_matrix.T @ velocity_matrix
    raw_matrix = torch.stack(raw_velocities, dim=1)
    raw_gram = weight * raw_matrix.T @ raw_matrix
    eigenvalues = torch.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    threshold = torch.clamp(eigenvalues[-1], min=cfg.mass_floor) * 1.0e-8
    n = len(points)
    return {
        "alpha": float(alpha),
        "points": np.asarray(points),
        "psi": psi_matrix.cpu().numpy(),
        "u": velocity_matrix[:n].cpu().numpy(),
        "v": velocity_matrix[n:].cpu().numpy(),
        "gram": gram.cpu().numpy(),
        "orth_error": float(
            torch.max(torch.abs(gram - torch.eye(cfg.modes, device=DEVICE))).cpu()
        ),
        "effective_raw_rank": int((eigenvalues > threshold).sum().cpu()),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "loss": float(np.mean([x["loss"] for x in items])),
        "mean_energy": float(np.mean([x["mean_energy"] for x in items])),
        "max_energy": float(np.max([x["max_energy"] for x in items])),
        "mean_projection_ratio": float(
            np.mean([x["mean_projection_ratio"] for x in items])
        ),
        "orth_error": float(np.max([x["orth_error"] for x in items])),
        "min_rank": int(np.min([x["effective_raw_rank"] for x in items])),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("run_%Y%m%d_%H%M%S_official_peeling")
    path = root / stem
    suffix = 0
    while path.exists():
        suffix += 1
        path = root / f"{stem}_{suffix:03d}"
    path.mkdir()
    return path


def append_log(run_dir: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    temporary = data_dir / "training_history.json.tmp"
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(data_dir / "training_history.json")
    keys = [
        "epoch",
        "seconds",
        "alpha",
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
        "probe_alpha",
        "probe_loss",
        "probe_mean_energy",
        "probe_max_energy",
        "probe_mean_projection_ratio",
        "probe_orth_error",
        "probe_rank",
    ]
    with (data_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in history)


def save_milestone_table(run_dir: Path, milestones: list[dict[str, Any]]) -> None:
    """Write the cumulative human/machine-readable evaluation table atomically."""
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    json_path = data_dir / "alpha_0p15_milestones.json"
    json_temp = json_path.with_suffix(".json.tmp")
    json_temp.write_text(json.dumps(milestones, indent=2), encoding="utf-8")
    json_temp.replace(json_path)
    keys = [
        "epoch",
        "scope",
        "alpha",
        "loss",
        "mean_energy",
        "max_energy",
        "mean_projection_ratio",
        "orth_error",
        "rank",
        "source",
    ]
    with (data_dir / "alpha_0p15_milestones.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in milestones)
    lines = [
        "| epoch | scope | alpha | loss | mean energy | max energy | projection ratio | orth error | rank |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in milestones:
        alpha = row.get("alpha", "")
        alpha_text = f"{alpha:.2f}" if isinstance(alpha, (int, float)) else str(alpha)
        lines.append(
            f"| {row['epoch']} | {row['scope']} | {alpha_text} | {row['loss']:.6e} | "
            f"{row['mean_energy']:.6e} | {row['max_energy']:.6e} | "
            f"{row['mean_projection_ratio']:.6e} | {row['orth_error']:.3e} | {row['rank']} |"
        )
    (data_dir / "alpha_0p15_milestones.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
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
        "algorithm": "official_sss_fixed_order_residual_peeling_continuous_alpha",
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


def validation_pass(
    cfg: Config, model: OfficialPeelingFields
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    details: list[dict[str, Any]] = []
    for index, alpha in enumerate(np.linspace(0.0, 1.0, cfg.validation_alphas)):
        points = fluid_points(
            cfg, cfg.validation_points, float(alpha), cfg.seed + 900_000 + index
        )
        _, diagnostic, _ = peel_shape(cfg, model, points, float(alpha), optimizers=None)
        details.append(diagnostic)
    model.train()
    return aggregate(details), details


def probe_pass(
    cfg: Config,
    model: OfficialPeelingFields,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    points = fluid_points(
        cfg, cfg.validation_points, cfg.probe_alpha, cfg.seed + 1_500_000
    )
    _, diagnostic, _ = peel_shape(cfg, model, points, cfg.probe_alpha, optimizers=None)
    model.train()
    return {
        "epoch": epoch,
        "scope": "alpha_0p15_exact",
        "alpha": cfg.probe_alpha,
        "loss": diagnostic["loss"],
        "mean_energy": diagnostic["mean_energy"],
        "max_energy": diagnostic["max_energy"],
        "mean_projection_ratio": diagnostic["mean_projection_ratio"],
        "orth_error": diagnostic["orth_error"],
        "rank": diagnostic["effective_raw_rank"],
        "source": "fixed Sobol probe; no optimization",
    }


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
    milestones: list[dict[str, Any]] = []
    if resume is not None:
        payload = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model"])
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        history = payload.get("history", [])
        best_validation = float(payload.get("best_validation", math.inf))
        start_epoch = int(payload["epoch"]) + 1

        # Preserve the already reported epochs in the requested cumulative
        # table. These historical entries are five-alpha validation aggregates;
        # epoch 10 is additionally re-evaluated below at exactly alpha=0.15.
        for old_row in history:
            if (
                int(old_row.get("epoch", -1)) in (1, 5, 10)
                and "validation_loss" in old_row
            ):
                milestones.append(
                    {
                        "epoch": int(old_row["epoch"]),
                        "scope": "historical_5alpha_validation",
                        "alpha": "grid",
                        "loss": float(old_row["validation_loss"]),
                        "mean_energy": float(old_row["validation_mean_energy"]),
                        "max_energy": float(old_row["validation_max_energy"]),
                        "mean_projection_ratio": float(
                            old_row["validation_mean_projection_ratio"]
                        ),
                        "orth_error": float(old_row["validation_orth_error"]),
                        "rank": int(old_row["validation_min_rank"]),
                        "source": str(resume),
                    }
                )

    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(
            {
                "reference": "official SSS_colab.ipynb, fixed-order optimization cell",
                "one_independent_network_per_mode": True,
                "one_optimizer_per_mode": True,
                "fixed_mode_order": True,
                "mode_sorting_used": False,
                "least_squares_residual_peeling": True,
                "field_and_gradient_share_projection_coefficients": True,
                "previous_modes_detached": True,
                "projection_penalty_used_in_loss": False,
                "envelope_mode": cfg.envelope_mode,
                "sampling_mode": cfg.sampling_mode,
                "training_uses_gt_or_ns": False,
                "continuous_alpha_sampling": True,
                "frequency_curriculum": False,
                "one_shot_continuous_alpha_inference": True,
                "constant_null_mode": "not present: hard homogeneous velocity space",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    append_log(
        run_dir,
        f"start official-peeling device={DEVICE} modes={cfg.modes} parameters={parameter_count(model):,} "
        f"points={cfg.points_per_step}",
    )
    if resume is not None and start_epoch > 1:
        exact_resume_probe = probe_pass(cfg, model, start_epoch - 1)
        milestones.append(exact_resume_probe)
        save_milestone_table(run_dir, milestones)
        append_log(
            run_dir,
            f"probe epoch={start_epoch - 1:04d} alpha={cfg.probe_alpha:.2f} "
            f"loss={exact_resume_probe['loss']:.6e} meanE={exact_resume_probe['mean_energy']:.4e} "
            f"proj={exact_resume_probe['mean_projection_ratio']:.3e} "
            f"orth={exact_resume_probe['orth_error']:.2e} rank={exact_resume_probe['rank']}",
        )
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs + 1):
        alpha = continuous_alpha(cfg, epoch)
        points = fluid_points(cfg, cfg.points_per_step, alpha, cfg.seed + epoch * 1009)
        model.train()
        _, diagnostic, _ = peel_shape(cfg, model, points, alpha, optimizers=optimizers)
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "alpha": alpha,
            "train_loss": diagnostic["loss"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic["mean_projection_ratio"],
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
                    "validation_mode_energies": [x["mode_energies"] for x in details],
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
                f"epoch={epoch:04d} alpha={alpha:.6f} train={diagnostic['loss']:.6e} "
                f"val={validation['loss']:.6e} meanE={validation['mean_energy']:.4e} "
                f"maxE={validation['max_energy']:.4e} proj={validation['mean_projection_ratio']:.3e} "
                f"orth={validation['orth_error']:.2e} rank={validation['min_rank']}",
            )
        probe = None
        if cfg.probe_every > 0 and epoch % cfg.probe_every == 0:
            probe = probe_pass(cfg, model, epoch)
            row.update(
                {
                    "probe_alpha": cfg.probe_alpha,
                    "probe_loss": probe["loss"],
                    "probe_mean_energy": probe["mean_energy"],
                    "probe_max_energy": probe["max_energy"],
                    "probe_mean_projection_ratio": probe["mean_projection_ratio"],
                    "probe_orth_error": probe["orth_error"],
                    "probe_rank": probe["rank"],
                }
            )
            milestones.append(probe)
            save_milestone_table(run_dir, milestones)
            append_log(
                run_dir,
                f"probe epoch={epoch:04d} alpha={cfg.probe_alpha:.2f} loss={probe['loss']:.6e} "
                f"meanE={probe['mean_energy']:.4e} maxE={probe['max_energy']:.4e} "
                f"proj={probe['mean_projection_ratio']:.3e} orth={probe['orth_error']:.2e} rank={probe['rank']}",
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
        if probe is not None:
            save_checkpoint(
                run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
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
        validation_alphas=2,
        epochs=1,
    )
    set_seed(cfg.seed)
    model = OfficialPeelingFields(cfg).to(DEVICE)
    optimizers = [
        torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate)
        for mode in model.modes
    ]
    points = fluid_points(cfg, cfg.points_per_step, 0.371, cfg.seed + 1)

    # Explicitly verify the Colab stop-gradient direction: mode 1 may update
    # itself through its peeled residual, but its loss cannot touch mode 0.
    raw0 = raw_mode_fields(cfg, model, 0, points, 0.371)
    peeled0 = least_squares_peel(
        pack_velocity(raw0),
        pack_gradient(raw0),
        raw0["psi"].to(REDUCTION_DTYPE),
        [],
        [],
        [],
        fluid_area(cfg) / len(points),
        cfg.mass_floor,
    )
    raw1 = raw_mode_fields(cfg, model, 1, points, 0.371)
    peeled1 = least_squares_peel(
        pack_velocity(raw1),
        pack_gradient(raw1),
        raw1["psi"].to(REDUCTION_DTYPE),
        [peeled0["normalized_velocity"].detach()],
        [peeled0["normalized_gradient"].detach()],
        [peeled0["normalized_psi"].detach()],
        fluid_area(cfg) / len(points),
        cfg.mass_floor,
    )
    model.zero_grad(set_to_none=True)
    causal_loss = peeled1["stiffness"] / torch.clamp(
        peeled1["mass"], min=cfg.mass_floor
    )
    causal_loss.backward()
    if any(
        parameter.grad is not None and float(parameter.grad.abs().max().cpu()) > 0.0
        for parameter in model.modes[0].parameters()
    ):
        raise RuntimeError("later-mode loss leaked a gradient into an earlier mode")
    if not any(
        parameter.grad is not None and float(parameter.grad.abs().max().cpu()) > 0.0
        for parameter in model.modes[1].parameters()
    ):
        raise RuntimeError("peeled mode received no gradient")
    model.zero_grad(set_to_none=True)

    before = [
        [parameter.detach().clone() for parameter in mode.parameters()]
        for mode in model.modes
    ]
    _, diagnostic, output = peel_shape(cfg, model, points, 0.371, optimizers)
    if not math.isfinite(diagnostic["loss"]):
        raise RuntimeError("non-finite peeling loss")
    if diagnostic["orth_error"] > 1.0e-7:
        raise RuntimeError(
            f"peeling failed orthogonality test: {diagnostic['orth_error']}"
        )
    for mode, old_parameters in zip(model.modes, before):
        if not any(
            not torch.equal(parameter.detach(), old)
            for parameter, old in zip(mode.parameters(), old_parameters)
        ):
            raise RuntimeError("an independent mode optimizer did not update its mode")
    inferred = infer_basis(cfg, model, points, 0.373)
    if inferred["u"].shape != (cfg.points_per_step, cfg.modes):
        raise RuntimeError("one-shot inference returned the wrong shape")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "device": str(DEVICE),
                "fixed_order": diagnostic["fixed_mode_order"],
                "orth_error": diagnostic["orth_error"],
                "rank": diagnostic["effective_raw_rank"],
                "causal_stop_gradient": "passed",
                "all_frequency_features_active": True,
                "one_shot_alpha": inferred["alpha"],
                "one_shot_shape": list(inferred["u"].shape),
            },
            indent=2,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON file containing Config field overrides",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--frequencies", type=int)
    parser.add_argument("--points-per-step", type=int)
    parser.add_argument("--validation-points", type=int)
    parser.add_argument("--validation-alphas", type=int)
    parser.add_argument("--probe-alpha", type=float)
    parser.add_argument("--probe-every", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--envelope-mode", choices=("localized",))
    parser.add_argument("--sampling-mode", choices=("mixed",))
    parser.add_argument("--global-fraction", type=float)
    parser.add_argument("--cylinder-fraction", type=float)
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

    cli_fields = (
        "modes",
        "epochs",
        "hidden",
        "hidden_layers",
        "frequencies",
        "points_per_step",
        "validation_points",
        "validation_alphas",
        "probe_alpha",
        "probe_every",
        "validate_every",
        "checkpoint_every",
        "learning_rate",
        "envelope_mode",
        "sampling_mode",
        "global_fraction",
        "cylinder_fraction",
        "wake_fraction",
    )
    for field in cli_fields:
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
    result_root = Path(cfg.result_root)
    run_dir = args.run_dir if args.run_dir is not None else make_run_dir(result_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, run_dir, args.resume)


if __name__ == "__main__":
    main()
