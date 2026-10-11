








from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass, fields
from datetime import datetime
from _neb_paths import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NATURE_BLUE = "#3B6FB6"
NATURE_RED = "#D1495B"
NATURE_DARK = "#243447"


def wall_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def replace_with_retry(temporary: Path, path: Path) -> None:

    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


@dataclass(frozen=True)
class Config:
    seed: int
    modes: int
    epochs: int
    points_per_step: int
    validation_points: int
    validation_geometries: int
    validate_every: int
    checkpoint_every: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    outer_half_extent: float
    outer_superellipsoid_power: int
    sphere_count: int
    sphere_center_min: float
    sphere_center_max: float
    sphere_radius_min: float
    sphere_radius_max: float
    metaball_softmin_tau: float
    minimum_overlap: float
    outer_clearance: float
    geometry_max_attempts: int
    outer_envelope_transition: float
    obstacle_envelope_transition: float
    input_features: int
    coordinate_normalization: str
    hidden: int
    hidden_layers: int
    siren_first_omega0: float
    siren_hidden_omega0: float
    learning_rate: float
    min_learning_rate: float
    lr_scheduler: str
    grad_clip: float
    mass_floor: float
    reorthogonalization_passes: int
    causal_sorting: str
    sampling_mode: str
    basis_objective: str
    symmetry_augmentation: str
    analytic_eigenmodes: bool
    degenerate_block_alignment: bool
    report_raw_rank: bool
    dtype: str
    result_root: str


def load_config(path: Path, overrides: dict[str, int | None]) -> Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(Config)}
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError("unknown config fields: " + ", ".join(unknown))
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    cfg = Config(**payload)
    validate_contract(cfg)
    return cfg


def validate_contract(cfg: Config) -> None:
    checks = {
        "three spheres": cfg.sphere_count == 3,
        "15 raw inputs": cfg.input_features == 15,
        "centered-unit normalization": cfg.coordinate_normalization == "centered_unit",
        "positive hidden-layer count": cfg.hidden_layers >= 1,
        "two MGS passes": cfg.reorthogonalization_passes == 2,
        "geometry Rayleigh sorting": cfg.causal_sorting == "geometry_rayleigh",
        "IID PCG64 sampling": cfg.sampling_mode == "iid_pcg64_uniform_fluid_volume",
        "Dirichlet objective": cfg.basis_objective == "antisymmetric_potential_dirichlet_rayleigh",
        "no symmetry augmentation": cfg.symmetry_augmentation == "none",
        "no analytic modes": not cfg.analytic_eigenmodes,
        "no degeneracy alignment": not cfg.degenerate_block_alignment,
        "no raw rank reporting": not cfg.report_raw_rank,
        "FP32": cfg.dtype == "float32",
        "cosine scheduler": cfg.lr_scheduler == "cosine",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("contract violation: " + ", ".join(failed))
    if cfg.modes < 1 or cfg.points_per_step < cfg.modes:
        raise ValueError("quadrature must contain at least as many points as modes")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def append_log(run_dir: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def geometry_from_token(cfg: Config, token: int) -> np.ndarray:

    rng = np.random.default_rng(cfg.seed + 104729 * int(token))
    for _ in range(cfg.geometry_max_attempts):
        sample = rng.random((3, 4))
        geometry = np.empty((3, 4), dtype=np.float64)
        geometry[:, :3] = cfg.sphere_center_min + sample[:, :3] * (
            cfg.sphere_center_max - cfg.sphere_center_min
        )
        geometry[:, 3] = cfg.sphere_radius_min + sample[:, 3] * (
            cfg.sphere_radius_max - cfg.sphere_radius_min
        )
        if not overlap_graph_connected(geometry, cfg.minimum_overlap):
            continue
        if not spheres_inside_rounded_cube(cfg, geometry):
            continue
        return geometry.astype(np.float32)
    raise RuntimeError(
        f"could not sample connected three-sphere geometry for token {token}"
    )


def overlap_graph_connected(geometry: np.ndarray, minimum_overlap: float) -> bool:
    adjacency = [set() for _ in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            distance = float(np.linalg.norm(geometry[left, :3] - geometry[right, :3]))
            overlap = float(geometry[left, 3] + geometry[right, 3] - distance)
            if overlap >= minimum_overlap:
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached, frontier = {0}, [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == 3


def sphere_directions(count: int = 256) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(golden * index), radius * np.sin(golden * index), z))


SPHERE_DIRECTIONS = sphere_directions()


def spheres_inside_rounded_cube(cfg: Config, geometry: np.ndarray) -> bool:
    half = cfg.outer_half_extent - cfg.outer_clearance
    if half <= 0.0:
        return False
    for sphere in geometry:
        boundary = sphere[:3] + sphere[3] * SPHERE_DIRECTIONS
        normalized = np.abs((boundary - 0.5) / half)
        if np.any(np.sum(normalized**cfg.outer_superellipsoid_power, axis=1) >= 1.0):
            return False
    return True


def outer_level_numpy(cfg: Config, points: np.ndarray) -> np.ndarray:
    normalized = (points - 0.5) / cfg.outer_half_extent
    return 1.0 - np.sum(normalized**cfg.outer_superellipsoid_power, axis=1)


def obstacle_level_numpy(cfg: Config, points: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - geometry[None, :, :3], axis=2)
    distances -= geometry[None, :, 3]
    minimum = distances.min(axis=1, keepdims=True)
    tau = cfg.metaball_softmin_tau
    return minimum[:, 0] - tau * np.log(np.exp(-(distances - minimum) / tau).sum(axis=1))


def fluid_mask(cfg: Config, points: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    return (outer_level_numpy(cfg, points) > 0.0) & (
        obstacle_level_numpy(cfg, points, geometry) > 0.0
    )


def iid_fluid_points(cfg: Config, count: int, geometry: np.ndarray, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    remaining = int(count)
    while remaining:
        candidates = rng.random((max(2 * remaining, 256), 3), dtype=np.float32)
        accepted = candidates[fluid_mask(cfg, candidates, geometry)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def approximate_fluid_volume(cfg: Config, geometry: np.ndarray, token: int) -> float:
    rng = np.random.default_rng(cfg.seed + 3_000_003 + int(token))
    points = rng.random((65536, 3), dtype=np.float32)
    return float(np.mean(fluid_mask(cfg, points, geometry), dtype=np.float64))


def normalized_features(cfg: Config, points: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = 2.0 * points - 1.0
    center_u = (geometry[:, :3] - cfg.sphere_center_min) / (
        cfg.sphere_center_max - cfg.sphere_center_min
    )
    radius_u = (geometry[:, 3:4] - cfg.sphere_radius_min) / (
        cfg.sphere_radius_max - cfg.sphere_radius_min
    )
    conditioned = 2.0 * torch.cat((center_u, radius_u), dim=1).reshape(1, 12) - 1.0
    conditioned = conditioned.expand(len(points), -1)
    features = torch.cat((spatial, conditioned), dim=1)
    spatial_jacobian = torch.zeros(
        (len(points), cfg.input_features, 3), device=points.device, dtype=points.dtype
    )
    spatial_jacobian[:, 0, 0] = 2.0
    spatial_jacobian[:, 1, 1] = 2.0
    spatial_jacobian[:, 2, 2] = 2.0
    return features, spatial_jacobian


def envelope_and_gradient(cfg: Config, points: torch.Tensor, geometry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    relative_outer = (points - 0.5) / cfg.outer_half_extent
    power = cfg.outer_superellipsoid_power
    outer_level = 1.0 - torch.sum(relative_outer**power, dim=1)
    outer_gradient = -(power / cfg.outer_half_extent) * relative_outer ** (power - 1)

    relative = points[:, None, :] - geometry[None, :, :3]
    norm = torch.linalg.vector_norm(relative, dim=2).clamp_min(1.0e-8)
    distances = norm - geometry[None, :, 3]
    weights = torch.softmax(-distances / cfg.metaball_softmin_tau, dim=1)
    obstacle_level = -cfg.metaball_softmin_tau * torch.logsumexp(
        -distances / cfg.metaball_softmin_tau, dim=1
    )
    obstacle_gradient = torch.sum(weights[:, :, None] * relative / norm[:, :, None], dim=1)

    outer_delta = cfg.outer_envelope_transition
    obstacle_delta = cfg.obstacle_envelope_transition
    outer_factor = outer_level / (outer_delta + outer_level)
    obstacle_factor = obstacle_level / (obstacle_delta + obstacle_level)
    outer_factor_gradient = (
        outer_delta / (outer_delta + outer_level).square()
    )[:, None] * outer_gradient
    obstacle_factor_gradient = (
        obstacle_delta / (obstacle_delta + obstacle_level).square()
    )[:, None] * obstacle_gradient
    envelope = outer_factor * obstacle_factor
    gradient = (
        obstacle_factor[:, None] * outer_factor_gradient
        + outer_factor[:, None] * obstacle_factor_gradient
    )
    return envelope, gradient


class SirenPotential(nn.Module):
    def __init__(self, cfg: Config, seed: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        fan_in = cfg.input_features
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for layer_index in range(cfg.hidden_layers):
            layer = nn.Linear(fan_in, cfg.hidden)
            bound = (
                1.0 / fan_in
                if layer_index == 0
                else math.sqrt(6.0 / fan_in) / cfg.siren_hidden_omega0
            )
            with torch.no_grad():
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)
            self.layers.append(layer)
            fan_in = cfg.hidden
        self.output = nn.Linear(cfg.hidden, 3)
        bound = math.sqrt(6.0 / fan_in) / cfg.siren_hidden_omega0
        with torch.no_grad():
            self.output.weight.uniform_(-bound, bound, generator=generator)
            self.output.bias.uniform_(-bound, bound, generator=generator)

    def forward_with_gradient(self, features: torch.Tensor, feature_gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = features
        gradient = feature_gradient
        for index, layer in enumerate(self.layers):
            omega = self.cfg.siren_first_omega0 if index == 0 else self.cfg.siren_hidden_omega0
            preactivation = omega * layer(value)
            gradient = omega * torch.einsum("oi,nid->nod", layer.weight, gradient)
            value = torch.sin(preactivation)
            gradient = torch.cos(preactivation)[:, :, None] * gradient
        raw = self.output(value)
        raw_gradient = torch.einsum("oi,nid->nod", self.output.weight, gradient)
        return raw, raw_gradient


class BasisModel(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.modes = nn.ModuleList(
            SirenPotential(cfg, cfg.seed + 7919 * (mode + 1)) for mode in range(cfg.modes)
        )


def mode_field(
    cfg: Config,
    model: BasisModel,
    mode: int,
    features: torch.Tensor,
    feature_gradient: torch.Tensor,
    envelope: torch.Tensor,
    envelope_gradient: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw, raw_gradient = model.modes[mode].forward_with_gradient(features, feature_gradient)
    potential = envelope[:, None] * raw
    gradient = (
        envelope[:, None, None] * raw_gradient
        + raw[:, :, None] * envelope_gradient[:, None, :]
    )
    return potential.reshape(-1), gradient.reshape(-1)


def cosine_learning_rate(cfg: Config, epoch: int) -> float:
    if cfg.epochs <= 1:
        return cfg.min_learning_rate
    progress = (epoch - 1) / (cfg.epochs - 1)
    return cfg.min_learning_rate + 0.5 * (cfg.learning_rate - cfg.min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )


def peeled_step(
    cfg: Config,
    model: BasisModel,
    optimizers: list[torch.optim.Optimizer] | None,
    points_numpy: np.ndarray,
    geometry_numpy: np.ndarray,
    volume: float,
) -> dict[str, Any]:
    points = torch.as_tensor(points_numpy, device=DEVICE, dtype=DTYPE)
    geometry = torch.as_tensor(geometry_numpy, device=DEVICE, dtype=DTYPE)
    features, feature_gradient = normalized_features(cfg, points, geometry)
    envelope, envelope_gradient = envelope_and_gradient(cfg, points, geometry)
    weight = float(volume / len(points_numpy))

    sorting: list[float] = []
    with torch.no_grad():
        for mode in range(cfg.modes):
            potential, gradient = mode_field(
                cfg, model, mode, features, feature_gradient, envelope, envelope_gradient
            )
            mass = weight * torch.dot(potential, potential)
            stiffness = weight * torch.dot(gradient, gradient)
            sorting.append(float((stiffness / mass.clamp_min(cfg.mass_floor)).cpu()))
    order = np.argsort(np.asarray(sorting), kind="stable").tolist()





    basis_potential = torch.empty(
        (points.numel(), cfg.modes), device=DEVICE, dtype=DTYPE
    )
    basis_gradient = torch.empty(
        (points.numel() * 3, cfg.modes), device=DEVICE, dtype=DTYPE
    )
    energies: list[float] = []
    residual_masses: list[float] = []
    projection_ratios: list[float] = []
    for causal_rank, mode in enumerate(order):
        potential, gradient = mode_field(
            cfg, model, mode, features, feature_gradient, envelope, envelope_gradient
        )
        residual_potential = potential
        residual_gradient = gradient
        coefficient_norm_sq = potential.new_zeros(())
        if causal_rank:
            previous_potential = basis_potential[:, :causal_rank]
            previous_gradient = basis_gradient[:, :causal_rank]
            for _ in range(cfg.reorthogonalization_passes):
                increment = weight * (previous_potential.T @ residual_potential)
                coefficient_norm_sq = coefficient_norm_sq + torch.dot(increment, increment)
                residual_potential = residual_potential - previous_potential @ increment
                residual_gradient = residual_gradient - previous_gradient @ increment
        mass = weight * torch.dot(residual_potential, residual_potential)
        stiffness = weight * torch.dot(residual_gradient, residual_gradient)
        if not bool(torch.isfinite(mass)) or not bool(torch.isfinite(stiffness)):
            raise FloatingPointError(f"non-finite peeled mode at causal rank {causal_rank + 1}")
        if float(mass.detach().cpu()) <= cfg.mass_floor:
            raise FloatingPointError(
                f"unresolved peeled mode at causal rank {causal_rank + 1}: "
                f"mass={float(mass.detach().cpu()):.6e}, "
                f"mass_floor={cfg.mass_floor:.6e}, raw_mode={mode + 1}"
            )
        loss = stiffness / mass
        if optimizers is not None:
            optimizer = optimizers[mode]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.modes[mode].parameters(), cfg.grad_clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite gradient for network {mode + 1}")
            optimizer.step()
        normalizer = torch.rsqrt(mass.detach())
        basis_potential[:, causal_rank].copy_(
            residual_potential.detach() * normalizer
        )
        basis_gradient[:, causal_rank].copy_(
            residual_gradient.detach() * normalizer
        )
        energies.append(float(loss.detach().cpu()))
        residual_masses.append(float(mass.detach().cpu()))
        projection_ratios.append(
            float((coefficient_norm_sq.detach() / mass.detach()).cpu()) if causal_rank else 0.0
        )

    matrix = basis_potential
    gram = weight * (matrix.T @ matrix)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (gram + gram.T))
    threshold = torch.finfo(DTYPE).eps * max(matrix.shape) * eigenvalues[-1].abs().clamp_min(cfg.mass_floor)
    peeled_rank = int(torch.count_nonzero(eigenvalues > threshold).cpu())
    identity = torch.eye(cfg.modes, device=DEVICE, dtype=DTYPE)
    return {
        "loss": float(np.mean(energies)),
        "mean_rayleigh": float(np.mean(energies)),
        "min_rayleigh": float(np.min(energies)),
        "max_rayleigh": float(np.max(energies)),
        "mean_projection_ratio": float(np.mean(projection_ratios)),
        "peeled_orthogonality_max_abs": float(torch.max(torch.abs(gram - identity)).cpu()),
        "peeled_effective_rank": peeled_rank,
        "requested_modes": cfg.modes,
        "peeled_gram_eigenvalue_min": float(eigenvalues[0].cpu()),
        "peeled_gram_eigenvalue_max": float(eigenvalues[-1].cpu()),
        "peeled_residual_mass_min": float(np.min(residual_masses)),
        "peeled_residual_mass_max": float(np.max(residual_masses)),
        "causal_order_1based": [int(mode) + 1 for mode in order],
        "reorthogonalization_passes": cfg.reorthogonalization_passes,
    }


def validate(
    cfg: Config, model: BasisModel, epoch: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    details: list[dict[str, Any]] = []
    with torch.enable_grad():
        for index in range(cfg.validation_geometries):
            token = 10_000_000 + index
            geometry = geometry_from_token(cfg, token)
            points = iid_fluid_points(
                cfg, cfg.validation_points, geometry, cfg.seed + 9_000_000 + index
            )
            volume = approximate_fluid_volume(cfg, geometry, token)
            metrics = peeled_step(cfg, model, None, points, geometry, volume)
            details.append(
                {
                    "geometry_index": index,
                    "geometry_token": token,
                    "sphere_parameters": geometry.tolist(),
                    "fluid_volume": volume,
                    **metrics,
                }
            )
    model.train()
    summary = {
        "epoch": epoch,
        "geometries": len(details),
        "points_per_geometry": cfg.validation_points,
        "mean_loss": float(np.mean([item["loss"] for item in details])),
        "max_loss": float(np.max([item["loss"] for item in details])),
        "min_peeled_rank": int(np.min([item["peeled_effective_rank"] for item in details])),
        "max_peeled_orthogonality_abs": float(
            np.max([item["peeled_orthogonality_max_abs"] for item in details])
        ),
        "min_peeled_residual_mass": float(
            np.min([item["peeled_residual_mass_min"] for item in details])
        ),
        "reporting_scope": "peeled_basis_only",
    }
    return summary, details


def checkpoint_payload(
    cfg: Config,
    model: BasisModel,
    optimizers: list[torch.optim.Optimizer],
    epoch: int,
    history: list[dict[str, Any]],
    validation_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format": "eigenfluid_connected_three_sphere_siren_dirichlet_v1",
        "epoch": epoch,
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "history": history,
        "validation_summary": validation_summary,
        "protocol": {
            "geometry_distribution": "Victoria arXiv v1 fixed-three-component distribution, rejection-conditioned on connectivity",
            "center_coordinates": "IID Uniform[0.25,0.75] in x,y,z",
            "radii": "IID Uniform[0.1,0.3]",
            "smooth_union": "soft-min of signed sphere distances, tau=0.02",
            "network": "independent canonical SIREN per peeled mode",
            "potential": "three independent entries of an antisymmetric 3D tensor",
            "velocity_boundary_condition": (
                "free-slip/no-through: homogeneous Dirichlet potential trace "
                "implies n dot div(A)=0 on each smooth boundary component"
            ),
            "hidden_layers_excluding_output": cfg.hidden_layers,
            "hidden_width": cfg.hidden,
            "objective": cfg.basis_objective,
            "sorting": "per-geometry detached raw Dirichlet Rayleigh stable sort",
            "peeling": "two-pass modified Gram-Schmidt in potential L2",
            "spatial_sampling": "fresh fully IID PCG64 rejection sample every optimizer step",
            "excluded": ["D4", "analytic eigenmodes", "degenerate-block direction alignment"],
            "reporting_scope": "peeled_basis_only",
        },
    }


def save_checkpoint(
    run_dir: Path,
    cfg: Config,
    model: BasisModel,
    optimizers: list[torch.optim.Optimizer],
    epoch: int,
    history: list[dict[str, Any]],
    validation_summary: dict[str, Any] | None,
    name: str,
) -> Path:
    path = run_dir / "checkpoints" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        checkpoint_payload(cfg, model, optimizers, epoch, history, validation_summary),
        temporary,
    )
    replace_with_retry(temporary, path)
    return path


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    atomic_json(run_dir / "data" / "training_history.json", history)
    if not history:
        return
    keys = sorted({key for row in history for key in row})
    path = run_dir / "data" / "training_history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    replace_with_retry(temporary, path)


def render_report(
    run_dir: Path,
    cfg: Config,
    history: list[dict[str, Any]],
    summary: dict[str, Any],
    details: list[dict[str, Any]],
) -> None:
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    axis_loss = fig.add_subplot(grid[0, :2])
    epochs = [row["epoch"] for row in history]
    axis_loss.plot(epochs, [row["train_loss"] for row in history], color=NATURE_BLUE, lw=1.5, label="train peeled Rayleigh")
    validation_x = [row["epoch"] for row in history if row.get("validation_loss") is not None]
    validation_y = [row["validation_loss"] for row in history if row.get("validation_loss") is not None]
    if validation_x:
        axis_loss.plot(validation_x, validation_y, "o-", color=NATURE_RED, ms=3, lw=1.2, label="held-out validation")
    axis_loss.set_yscale("log")
    axis_loss.set_xlabel("epoch")
    axis_loss.set_ylabel("Dirichlet Rayleigh loss")
    axis_loss.grid(alpha=0.2)
    axis_loss.legend(frameon=False)

    axis_rank = fig.add_subplot(grid[1, :2])
    axis_rank.plot(epochs, [row["train_peeled_rank"] for row in history], color=NATURE_BLUE, lw=1.5)
    axis_rank.axhline(cfg.modes, color=NATURE_RED, ls="--", lw=1.0)
    axis_rank.set_ylim(0, cfg.modes + 3)
    axis_rank.set_xlabel("epoch")
    axis_rank.set_ylabel("peeled effective rank")
    axis_rank.grid(alpha=0.2)

    axis_geometry = fig.add_subplot(grid[:, 2], projection="3d")
    geometry = np.asarray(details[0]["sphere_parameters"], dtype=np.float64)
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 20)
    for index, (cx, cy, cz, radius) in enumerate(geometry):
        x = cx + radius * np.outer(np.cos(u), np.sin(v))
        y = cy + radius * np.outer(np.sin(u), np.sin(v))
        z = cz + radius * np.outer(np.ones_like(u), np.cos(v))
        axis_geometry.plot_surface(
            x, y, z, color=(NATURE_RED if index != 1 else NATURE_BLUE), alpha=0.52, linewidth=0
        )
    axis_geometry.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x", ylabel="y", zlabel="z")
    axis_geometry.set_box_aspect((1, 1, 1))
    axis_geometry.set_title("held-out connected three-sphere geometry")
    fig.suptitle(
        f"3D connected metaball · K={cfg.modes} · epoch {summary['epoch']}\n"
        f"validation peeled rank ≥ {summary['min_peeled_rank']} · max |G−I|={summary['max_peeled_orthogonality_abs']:.2e}",
        color=NATURE_DARK,
        fontsize=13,
        fontweight="bold",
    )
    png = figure_dir / "training_validation_summary.png"
    fig.savefig(png, dpi=220, facecolor="white")
    plt.close(fig)


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name.split("_")[-1])
        for path in root.glob("run_*")
        if path.is_dir() and path.name.split("_")[-1].isdigit()
    ]
    return root / f"run_{max(indices, default=0) + 1:03d}"


def resolve_result_root(cfg: Config) -> Path:
    path = Path(cfg.result_root)
    return path if path.is_absolute() else REPO_ROOT / path


def load_checkpoint(
    path: Path,
    cfg: Config,
    model: BasisModel,
    optimizers: list[torch.optim.Optimizer],
) -> tuple[int, list[dict[str, Any]]]:
    payload = torch.load(path, map_location=DEVICE, weights_only=False)
    checkpoint_config = payload["config"]
    requested = cfg.__dict__
    mutable = {"epochs", "validate_every", "checkpoint_every", "result_root"}
    changed = [
        key for key in set(checkpoint_config) | set(requested)
        if key not in mutable and checkpoint_config.get(key) != requested.get(key)
    ]
    if changed:
        raise ValueError("resume changes training contract: " + ", ".join(sorted(changed)))
    model.load_state_dict(payload["model"])
    for optimizer, state in zip(optimizers, payload.get("optimizers", [])):
        optimizer.load_state_dict(state)
    return int(payload["epoch"]), list(payload.get("history", []))


def train_and_validate(cfg: Config, run_dir: Path, resume: Path | None) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
    model = BasisModel(cfg).to(DEVICE)
    optimizers = [torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate) for mode in model.modes]
    start_epoch, history = 1, []
    if resume is not None:
        completed, history = load_checkpoint(resume, cfg, model, optimizers)
        start_epoch = completed + 1
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", cfg.__dict__)
    atomic_json(
        run_dir / "protocol.json",
        checkpoint_payload(cfg, model, optimizers, start_epoch - 1, [], None)["protocol"],
    )
    session_started_at = wall_timestamp()
    campaign_started_at = (
        str(history[0].get("epoch_started_at"))
        if history and history[0].get("epoch_started_at")
        else session_started_at
    )
    atomic_json(
        run_dir / "completion.json",
        {
            "complete": False,
            "epochs_requested": cfg.epochs,
            "resume_epoch": start_epoch - 1,
            "campaign_started_at": campaign_started_at,
            "session_started_at": session_started_at,
        },
    )
    device_metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(DEVICE),
        "seed": cfg.seed,
        "deterministic_epoch_sampling": True,
    }
    if DEVICE.type == "cuda":
        properties = torch.cuda.get_device_properties(DEVICE)
        device_metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": properties.total_memory / 2**20,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    atomic_json(run_dir / "data" / "run_metadata.json", device_metadata)
    append_log(
        run_dir,
        f"start device={DEVICE} modes={cfg.modes} points={cfg.points_per_step} epochs={cfg.epochs} start={start_epoch}",
    )
    session_wall_started = time.perf_counter()
    final_summary: dict[str, Any] | None = None
    final_details: list[dict[str, Any]] = []
    checkpoint_timings: list[dict[str, Any]] = []
    checkpoint_timing_path = run_dir / "data" / "checkpoint_timing.json"
    if checkpoint_timing_path.exists():
        try:
            checkpoint_timings = json.loads(checkpoint_timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoint_timings = []
    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_started_at = wall_timestamp()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        learning_rate = cosine_learning_rate(cfg, epoch)
        for optimizer in optimizers:
            optimizer.param_groups[0]["lr"] = learning_rate
        geometry = geometry_from_token(cfg, epoch)
        points = iid_fluid_points(cfg, cfg.points_per_step, geometry, cfg.seed + 100_000 + epoch)
        volume = approximate_fluid_volume(cfg, geometry, epoch)
        model.train()
        optimizer_started = time.perf_counter()
        metrics = peeled_step(cfg, model, optimizers, points, geometry, volume)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - optimizer_started
        should_validate = epoch % cfg.validate_every == 0 or epoch == cfg.epochs
        validation_summary = None
        validation_seconds = 0.0
        if should_validate:
            validation_started = time.perf_counter()
            validation_summary, final_details = validate(cfg, model, epoch)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - validation_started
            final_summary = validation_summary
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - epoch_started
        cumulative_epoch_seconds = float(
            sum(float(item.get("epoch_seconds", item.get("seconds", 0.0))) for item in history)
            + elapsed
        )
        cuda_allocated_mib = None
        cuda_reserved_mib = None
        cuda_max_allocated_mib = None
        if DEVICE.type == "cuda":
            cuda_allocated_mib = torch.cuda.memory_allocated() / 2**20
            cuda_reserved_mib = torch.cuda.memory_reserved() / 2**20
            cuda_max_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "epoch_seconds": elapsed,
            "optimizer_seconds": optimizer_seconds,
            "validation_seconds": validation_seconds,
            "epoch_started_at": epoch_started_at,
            "epoch_finished_at": wall_timestamp(),
            "cumulative_epoch_seconds": cumulative_epoch_seconds,
            "learning_rate": learning_rate,
            "train_loss": metrics["loss"],
            "train_mean_rayleigh": metrics["mean_rayleigh"],
            "train_min_rayleigh": metrics["min_rayleigh"],
            "train_max_rayleigh": metrics["max_rayleigh"],
            "train_mean_projection_ratio": metrics["mean_projection_ratio"],
            "train_peeled_rank": metrics["peeled_effective_rank"],
            "train_peeled_orthogonality_max_abs": metrics["peeled_orthogonality_max_abs"],
            "train_peeled_residual_mass_min": metrics["peeled_residual_mass_min"],
            "validation_loss": None if validation_summary is None else validation_summary["mean_loss"],
            "validation_max_loss": None if validation_summary is None else validation_summary["max_loss"],
            "validation_min_peeled_rank": None if validation_summary is None else validation_summary["min_peeled_rank"],
            "validation_max_peeled_orthogonality_abs": None if validation_summary is None else validation_summary["max_peeled_orthogonality_abs"],
            "validation_min_peeled_residual_mass": None if validation_summary is None else validation_summary["min_peeled_residual_mass"],
            "cuda_memory_allocated_mib": cuda_allocated_mib,
            "cuda_memory_reserved_mib": cuda_reserved_mib,
            "cuda_max_memory_allocated_mib": cuda_max_allocated_mib,
        }
        history.append(row)
        append_log(
            run_dir,
            f"epoch={epoch:04d}/{cfg.epochs} sec={elapsed:.2f} loss={metrics['loss']:.6e} "
            f"peeled_rank={metrics['peeled_effective_rank']} orth={metrics['peeled_orthogonality_max_abs']:.2e} "
            f"val={row['validation_loss']} lr={learning_rate:.3e}",
        )
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            checkpoint_started_at = wall_timestamp()
            checkpoint_started = time.perf_counter()
            epoch_path = save_checkpoint(
                run_dir,
                cfg,
                model,
                optimizers,
                epoch,
                history,
                validation_summary,
                f"epoch_{epoch:04d}.pt",
            )
            latest_path = run_dir / "checkpoints" / "latest.pt"
            latest_temporary = latest_path.with_suffix(latest_path.suffix + ".tmp")
            shutil.copy2(epoch_path, latest_temporary)
            replace_with_retry(latest_temporary, latest_path)
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            checkpoint_timings.append(
                {
                    "epoch": epoch,
                    "started_at": checkpoint_started_at,
                    "finished_at": wall_timestamp(),
                    "seconds": checkpoint_seconds,
                    "epoch_checkpoint": str(epoch_path.resolve()),
                    "latest_checkpoint": str(latest_path.resolve()),
                }
            )
            atomic_json(checkpoint_timing_path, checkpoint_timings)
            row["checkpoint_seconds"] = checkpoint_seconds
        else:
            row["checkpoint_seconds"] = 0.0
        save_history(run_dir, history)
        atomic_json(
            run_dir / "data" / "timing.json",
            {
                "campaign_started_at": campaign_started_at,
                "session_started_at": session_started_at,
                "last_update_at": wall_timestamp(),
                "epochs_completed": epoch,
                "epochs_requested": cfg.epochs,
                "cumulative_epoch_seconds": cumulative_epoch_seconds,
                "cumulative_optimizer_seconds": float(
                    sum(float(item.get("optimizer_seconds", 0.0)) for item in history)
                ),
                "cumulative_validation_seconds": float(
                    sum(float(item.get("validation_seconds", 0.0)) for item in history)
                ),
                "cumulative_checkpoint_seconds": float(
                    sum(float(item.get("seconds", 0.0)) for item in checkpoint_timings)
                ),
                "latest_epoch_seconds": elapsed,
            },
        )

    if final_summary is None:
        final_summary, final_details = validate(cfg, model, cfg.epochs)
    atomic_json(run_dir / "validation" / "summary.json", final_summary)
    atomic_json(run_dir / "validation" / "details.json", final_details)
    final_path = save_checkpoint(
        run_dir, cfg, model, optimizers, cfg.epochs, history, final_summary, "final.pt"
    )
    render_report(run_dir, cfg, history, final_summary, final_details)
    session_wall_seconds = time.perf_counter() - session_wall_started
    completion = {
        "complete": True,
        "epochs_requested": cfg.epochs,
        "epochs_completed": cfg.epochs,
        "campaign_started_at": campaign_started_at,
        "completed_at": wall_timestamp(),
        "session_started_at": session_started_at,
        "session_wall_seconds": session_wall_seconds,
        "wall_seconds": session_wall_seconds,
        "epoch_seconds_sum": float(
            sum(float(item.get("epoch_seconds", item.get("seconds", 0.0))) for item in history)
        ),
        "optimizer_seconds_sum": float(
            sum(float(item.get("optimizer_seconds", 0.0)) for item in history)
        ),
        "validation_seconds_sum": float(
            sum(float(item.get("validation_seconds", 0.0)) for item in history)
        ),
        "checkpoint_seconds_sum": float(
            sum(float(item.get("seconds", 0.0)) for item in checkpoint_timings)
        ),
        "checkpoint": str(final_path.resolve()),
        "validation": final_summary,
        "reporting_scope": "peeled_basis_only",
    }
    atomic_json(run_dir / "completion.json", completion)
    append_log(
        run_dir,
        f"complete epoch_seconds={completion['epoch_seconds_sum']:.1f}s "
        f"session_wall={completion['session_wall_seconds']:.1f}s checkpoint={final_path}",
    )


def validate_only(cfg: Config, run_dir: Path, checkpoint: Path) -> None:
    model = BasisModel(cfg).to(DEVICE)
    optimizers = [torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate) for mode in model.modes]
    epoch, history = load_checkpoint(checkpoint, cfg, model, optimizers)
    summary, details = validate(cfg, model, epoch)
    atomic_json(run_dir / "validation" / "summary.json", summary)
    atomic_json(run_dir / "validation" / "details.json", details)
    render_report(run_dir, cfg, history, summary, details)
    print(json.dumps(summary, indent=2), flush=True)
