"""Dynamic Shape Space Spectra training for a periodic 3D TG case.

All modes are trained jointly, dynamically sorted by pre-projection Rayleigh
energy, and causally peeled with detached dominating modes. Every network
emits the three independent entries of an antisymmetric tensor.

With ``A12=a, A13=b, A23=c``, the row-wise tensor divergence is

    u = Dy(a) + Dz(b)
    v = -Dx(a) + Dz(c)
    w = -Dx(b) - Dy(c).

All D operators are the same centred-difference operator.  Their commutation
makes ``Dx(u)+Dy(v)+Dz(w)`` cancel algebraically at the discrete level.  The
points are continuous Sobol samples; no training grid is constructed.
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
from torch.utils.checkpoint import checkpoint


CASE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_ROOT = CASE_ROOT / "results" / "k256_dynamic_sss"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
REDUCTION_DTYPE = torch.float64


@dataclass
class Config:
    seed: int = 20260719
    modes: int = 256
    domain_type: str = "periodic_closed"
    xmin: float = -math.pi
    xmax: float = math.pi
    ymin: float = -math.pi
    ymax: float = math.pi
    zmin: float = -math.pi
    zmax: float = math.pi
    obstacle_x: float = 0.0
    obstacle_y: float = 0.0
    obstacle_z: float = 0.0
    obstacle_radius: float = 1.0
    inlet_speed: float = 0.0
    hidden: int = 64
    hidden_layers: int = 4
    frequencies: int = 6
    points_per_step: int = 4096
    epochs: int = 10000
    validation_points: int = 4096
    validation_alphas: int = 1
    probe_alpha: float = 0.0
    probe_every: int = 100
    validate_every: int = 50
    checkpoint_every: int = 25
    learning_rate: float = 8.0e-4
    rayleigh_scale: float = 80.0
    grad_clip: float = 5.0
    mass_floor: float = 1.0e-10
    sss_ordering: str = "dynamic"
    activation_checkpointing: bool = True
    lr_scheduler: str = "reduce_on_plateau"
    lr_reduce_factor: float = 0.5
    lr_patience_validations: int = 10
    min_learning_rate: float = 2.5e-5
    early_stopping_patience_validations: int = 24
    early_stopping_min_delta: float = 1.0e-4
    endpoint_period: int = 0
    difference_step: float = 0.02
    envelope_mode: str = "periodic"
    sampling_mode: str = "uniform"
    global_fraction: float = 1.0
    obstacle_fraction: float = 0.0
    wake_fraction: float = 0.0
    obstacle_band_radii: float = 0.0
    wake_length_radii: float = 0.0
    wake_half_width_radii: float = 0.0
    result_root: str = str(DEFAULT_RESULT_ROOT)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _expand_alpha(alpha: torch.Tensor, count: int) -> torch.Tensor:
    if alpha.ndim == 0:
        return alpha.reshape(1, 1).expand(count, 1)
    if alpha.ndim == 1:
        alpha = alpha.reshape(-1, 1)
    if alpha.shape[0] == 1:
        return alpha.expand(count, 1)
    return alpha


def obstacle_sdf_torch(
    cfg: Config, xyz: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Analytic interpolation of the exact sphere and axis-aligned box SDFs."""
    alpha = _expand_alpha(alpha, len(xyz))
    centre = xyz.new_tensor((cfg.obstacle_x, cfg.obstacle_y, cfg.obstacle_z))
    relative = xyz - centre
    sphere_sdf = (
        torch.linalg.vector_norm(relative, ord=2, dim=1, keepdim=True)
        - cfg.obstacle_radius
    )
    box_offset = torch.abs(relative) - cfg.obstacle_radius
    box_sdf = torch.linalg.vector_norm(
        torch.relu(box_offset), ord=2, dim=1, keepdim=True
    ) + torch.minimum(
        torch.amax(box_offset, dim=1, keepdim=True),
        torch.zeros_like(box_offset[:, :1]),
    )
    return (1.0 - alpha) * sphere_sdf + alpha * box_sdf


def obstacle_sdf_numpy(cfg: Config, xyz: np.ndarray, alpha: float) -> np.ndarray:
    """NumPy form of the interpolated analytic obstacle SDF."""
    relative = xyz - np.asarray(
        (cfg.obstacle_x, cfg.obstacle_y, cfg.obstacle_z), dtype=np.float32
    )
    sphere_sdf = np.linalg.norm(relative, axis=1) - cfg.obstacle_radius
    box_offset = np.abs(relative) - cfg.obstacle_radius
    box_sdf = np.linalg.norm(np.maximum(box_offset, 0.0), axis=1) + np.minimum(
        np.max(box_offset, axis=1), 0.0
    )
    return (1.0 - alpha) * sphere_sdf + alpha * box_sdf


def obstacle_surface_radius_numpy(
    cfg: Config,
    directions: np.ndarray,
    alpha: float,
    iterations: int = 40,
) -> np.ndarray:
    """Find the SDF zero level along unit rays without a surface mesh."""
    shape = directions.shape[:-1]
    unit = directions.reshape(-1, 3)
    centre = np.asarray(
        (cfg.obstacle_x, cfg.obstacle_y, cfg.obstacle_z), dtype=np.float64
    )
    lower = np.zeros(len(unit), dtype=np.float64)
    upper = np.full(
        len(unit), np.sqrt(3.0) * cfg.obstacle_radius, dtype=np.float64
    )
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        points = centre + middle[:, None] * unit
        inside = obstacle_sdf_numpy(cfg, points, alpha) <= 0.0
        lower = np.where(inside, middle, lower)
        upper = np.where(inside, upper, middle)
    return (0.5 * (lower + upper)).reshape(shape)


def inside_obstacle(cfg: Config, xyz: np.ndarray, alpha: float) -> np.ndarray:
    if cfg.domain_type == "periodic_closed":
        return np.zeros(len(xyz), dtype=bool)
    return obstacle_sdf_numpy(cfg, xyz, alpha) <= 0.0


def approximate_fluid_volume(cfg: Config, alpha: float) -> float:
    """A scalar quadrature weight; its small morph-volume error cancels in R."""
    domain = (
        (cfg.xmax - cfg.xmin)
        * (cfg.ymax - cfg.ymin)
        * (cfg.zmax - cfg.zmin)
    )
    if cfg.domain_type == "periodic_closed":
        return domain
    sphere = (4.0 / 3.0) * math.pi * cfg.obstacle_radius**3
    cube = (2.0 * cfg.obstacle_radius) ** 3
    return domain - ((1.0 - alpha) * sphere + alpha * cube)


def hard_open_outlet_envelope(
    cfg: Config, xyz: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """Double-zero potential envelope, leaving only the right outlet open."""
    if cfg.domain_type == "periodic_closed":
        return torch.ones((len(xyz), 1), device=xyz.device, dtype=xyz.dtype)
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    yn = 2.0 * (y - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    zn = 2.0 * (z - cfg.zmin) / (cfg.zmax - cfg.zmin) - 1.0
    inlet_distance = torch.clamp(
        (x - cfg.xmin) / cfg.obstacle_radius, min=0.0
    )
    inlet = inlet_distance / (1.0 + inlet_distance)
    ywall_distance = torch.clamp((1.0 - yn.square()) / 0.25, min=0.0)
    zwall_distance = torch.clamp((1.0 - zn.square()) / 0.25, min=0.0)
    ywall = ywall_distance / (1.0 + ywall_distance)
    zwall = zwall_distance / (1.0 + zwall_distance)
    clearance = torch.relu(
        obstacle_sdf_torch(cfg, xyz, alpha) / cfg.obstacle_radius
    )
    obstacle = clearance / (1.0 + clearance)
    return (inlet * ywall * zwall * obstacle).square()


def inlet_lifting_velocity(cfg: Config, xyz: torch.Tensor) -> torch.Tensor:
    """Steady left-inflow lift; basis modes themselves have homogeneous traces."""
    yn = 2.0 * (xyz[:, 1] - cfg.ymin) / (cfg.ymax - cfg.ymin) - 1.0
    zn = 2.0 * (xyz[:, 2] - cfg.zmin) / (cfg.zmax - cfg.zmin) - 1.0
    u = cfg.inlet_speed * torch.relu(1.0 - yn.square()) * torch.relu(
        1.0 - zn.square()
    )
    return torch.stack((u, torch.zeros_like(u), torch.zeros_like(u)), dim=1)


class AntisymmetricPotentialMode(nn.Module):
    """One independent mode producing (A12, A13, A23)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        in_features = 2 * 3 * cfg.frequencies + 1 + 3
        layers: list[nn.Linear] = [nn.Linear(in_features, cfg.hidden)]
        layers.extend(
            nn.Linear(cfg.hidden, cfg.hidden) for _ in range(cfg.hidden_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(cfg.hidden, 3)

    def encode(self, xyz: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha = _expand_alpha(alpha, len(xyz))
        if self.cfg.domain_type == "periodic_closed":
            normal = torch.stack(
                (
                    (xyz[:, 0] - self.cfg.xmin)
                    / (self.cfg.xmax - self.cfg.xmin),
                    (xyz[:, 1] - self.cfg.ymin)
                    / (self.cfg.ymax - self.cfg.ymin),
                    (xyz[:, 2] - self.cfg.zmin)
                    / (self.cfg.zmax - self.cfg.zmin),
                ),
                dim=1,
            )
        else:
            normal = torch.stack(
                (
                    2.0
                    * (xyz[:, 0] - self.cfg.xmin)
                    / (self.cfg.xmax - self.cfg.xmin)
                    - 1.0,
                    2.0
                    * (xyz[:, 1] - self.cfg.ymin)
                    / (self.cfg.ymax - self.cfg.ymin)
                    - 1.0,
                    2.0
                    * (xyz[:, 2] - self.cfg.zmin)
                    / (self.cfg.zmax - self.cfg.zmin)
                    - 1.0,
                ),
                dim=1,
            )
        frequencies = 2.0 ** torch.arange(
            self.cfg.frequencies, device=xyz.device, dtype=xyz.dtype
        )
        phase_scale = 2.0 * math.pi if self.cfg.domain_type == "periodic_closed" else 1.0
        phase = normal.unsqueeze(-1) * frequencies * phase_scale
        encoded = torch.cat(
            (torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)), dim=1
        )
        if self.cfg.domain_type == "periodic_closed":
            relative = torch.zeros_like(xyz)
        else:
            centre = xyz.new_tensor(
                (self.cfg.obstacle_x, self.cfg.obstacle_y, self.cfg.obstacle_z)
            )
            relative = (xyz - centre) / self.cfg.obstacle_radius
        return torch.cat((encoded, alpha, relative), dim=1)

    def forward(self, xyz: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        value = self.encode(xyz, alpha)
        for layer in self.layers:
            value = torch.sin(layer(value))
        raw = self.output(value)
        return hard_open_outlet_envelope(self.cfg, xyz, alpha) * raw


class OfficialPeelingFields(nn.Module):
    """Independent ModuleList: mode identities never get sorted or exchanged."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.modes = nn.ModuleList(
            AntisymmetricPotentialMode(cfg) for _ in range(cfg.modes)
        )

    def forward_mode(
        self, mode: int, xyz: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        return self.modes[mode](xyz, alpha)

    def forward(self, xyz: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        # All parameters remain independent leaf tensors owned by their mode.
        # Stacking only batches the identical algebra into large GPU kernels.
        value = self.modes[0].encode(xyz, alpha)
        value = value.unsqueeze(0).expand(self.cfg.modes, -1, -1)
        for layer_index in range(self.cfg.hidden_layers):
            weight = torch.stack(
                [mode.layers[layer_index].weight for mode in self.modes]
            )
            bias = torch.stack(
                [mode.layers[layer_index].bias for mode in self.modes]
            )
            value = torch.sin(
                torch.einsum("mni,moi->mno", value, weight)
                + bias[:, None, :]
            )
        output_weight = torch.stack(
            [mode.output.weight for mode in self.modes]
        )
        output_bias = torch.stack([mode.output.bias for mode in self.modes])
        raw = (
            torch.einsum("mni,moi->mno", value, output_weight)
            + output_bias[:, None, :]
        )
        envelope = hard_open_outlet_envelope(
            self.cfg, xyz, alpha
        ).unsqueeze(0)
        return (envelope * raw).permute(1, 0, 2)


def antisymmetric_tensor(entries: torch.Tensor) -> torch.Tensor:
    """Hard-code A=-A^T from (...,3) independent entries."""
    a12, a13, a23 = entries.unbind(dim=-1)
    zero = torch.zeros_like(a12)
    row0 = torch.stack((zero, a12, a13), dim=-1)
    row1 = torch.stack((-a12, zero, a23), dim=-1)
    row2 = torch.stack((-a13, -a23, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _offsets_for_second_derivatives() -> list[tuple[int, int, int]]:
    offsets = {(0, 0, 0)}
    for axis in range(3):
        for sign in (-1, 1):
            offset = [0, 0, 0]
            offset[axis] = sign
            offsets.add(tuple(offset))
    for first in range(3):
        for second in range(first + 1, 3):
            for sign_first in (-1, 1):
                for sign_second in (-1, 1):
                    offset = [0, 0, 0]
                    offset[first] = sign_first
                    offset[second] = sign_second
                    offsets.add(tuple(offset))
    return sorted(offsets)


STENCIL_OFFSETS = _offsets_for_second_derivatives()


def discrete_mode_fields(
    cfg: Config,
    model: OfficialPeelingFields,
    mode: int,
    points: np.ndarray | torch.Tensor,
    alpha_value: float,
    *,
    with_gradient: bool = True,
) -> dict[str, torch.Tensor]:
    """Evaluate div(A) with one commuting local stencil on arbitrary points."""
    model_device = next(model.parameters()).device
    xyz = torch.as_tensor(points, device=model_device, dtype=DTYPE)
    h = cfg.difference_step
    offsets = STENCIL_OFFSETS if with_gradient else [
        offset
        for offset in STENCIL_OFFSETS
        if sum(abs(value) for value in offset) <= 1
    ]
    shifted = torch.cat(
        [
            xyz
            + xyz.new_tensor(offset).reshape(1, 3).mul(h)
            for offset in offsets
        ],
        dim=0,
    )
    alpha = torch.full(
        (len(shifted), 1), alpha_value, device=model_device, dtype=DTYPE
    )
    evaluated = model.forward_mode(mode, shifted, alpha).to(REDUCTION_DTYPE)
    chunks = evaluated.reshape(len(offsets), len(xyz), 3)
    values = {offset: chunks[index] for index, offset in enumerate(offsets)}
    origin = values[(0, 0, 0)]

    first = torch.empty(
        (len(xyz), 3, 3), device=model_device, dtype=REDUCTION_DTYPE
    )
    for axis in range(3):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[axis] = 1
        minus[axis] = -1
        first[:, :, axis] = (
            values[tuple(plus)] - values[tuple(minus)]
        ) / (2.0 * h)

    # a=A12, b=A13, c=A23
    a, b, c = 0, 1, 2
    velocity = torch.stack(
        (
            first[:, a, 1] + first[:, b, 2],
            -first[:, a, 0] + first[:, c, 2],
            -first[:, b, 0] - first[:, c, 1],
        ),
        dim=1,
    )
    output = {
        "potential": origin,
        "tensor": antisymmetric_tensor(origin),
        "velocity": velocity,
    }
    if not with_gradient:
        return output

    hessian = torch.empty(
        (len(xyz), 3, 3, 3), device=model_device, dtype=REDUCTION_DTYPE
    )
    for first_axis in range(3):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[first_axis] = 1
        minus[first_axis] = -1
        hessian[:, :, first_axis, first_axis] = (
            values[tuple(plus)] - 2.0 * origin + values[tuple(minus)]
        ) / (h * h)
        for second_axis in range(first_axis + 1, 3):
            pp = [0, 0, 0]
            pm = [0, 0, 0]
            mp = [0, 0, 0]
            mm = [0, 0, 0]
            pp[first_axis], pp[second_axis] = 1, 1
            pm[first_axis], pm[second_axis] = 1, -1
            mp[first_axis], mp[second_axis] = -1, 1
            mm[first_axis], mm[second_axis] = -1, -1
            mixed = (
                values[tuple(pp)]
                - values[tuple(pm)]
                - values[tuple(mp)]
                + values[tuple(mm)]
            ) / (4.0 * h * h)
            hessian[:, :, first_axis, second_axis] = mixed
            hessian[:, :, second_axis, first_axis] = mixed

    gradient = torch.empty(
        (len(xyz), 3, 3), device=model_device, dtype=REDUCTION_DTYPE
    )
    # gradient[:, velocity component, derivative direction]
    for derivative in range(3):
        gradient[:, 0, derivative] = (
            hessian[:, a, 1, derivative] + hessian[:, b, 2, derivative]
        )
        gradient[:, 1, derivative] = (
            -hessian[:, a, 0, derivative] + hessian[:, c, 2, derivative]
        )
        gradient[:, 2, derivative] = (
            -hessian[:, b, 0, derivative] - hessian[:, c, 1, derivative]
        )
    divergence = (
        gradient[:, 0, 0] + gradient[:, 1, 1] + gradient[:, 2, 2]
    )
    output.update(
        {
            "gradient": gradient,
            "divergence": divergence,
            "hessian": hessian,
        }
    )
    return output


def discrete_all_mode_fields(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray | torch.Tensor,
    alpha_value: float,
) -> dict[str, torch.Tensor]:
    """Batched equivalent of ``discrete_mode_fields`` for every mode."""
    model_device = next(model.parameters()).device
    xyz = torch.as_tensor(points, device=model_device, dtype=DTYPE)
    h = cfg.difference_step
    shifted = torch.cat(
        [
            xyz + xyz.new_tensor(offset).reshape(1, 3).mul(h)
            for offset in STENCIL_OFFSETS
        ],
        dim=0,
    )
    alpha = torch.full(
        (len(shifted), 1),
        alpha_value,
        device=model_device,
        dtype=DTYPE,
    )
    if cfg.activation_checkpointing and torch.is_grad_enabled():
        evaluated = checkpoint(
            model, shifted, alpha, use_reentrant=False
        ).to(REDUCTION_DTYPE)
    else:
        evaluated = model(shifted, alpha).to(REDUCTION_DTYPE)
    chunks = evaluated.reshape(
        len(STENCIL_OFFSETS), len(xyz), cfg.modes, 3
    )
    values = {
        offset: chunks[index]
        for index, offset in enumerate(STENCIL_OFFSETS)
    }
    origin = values[(0, 0, 0)]

    first = torch.empty(
        (len(xyz), cfg.modes, 3, 3),
        device=model_device,
        dtype=REDUCTION_DTYPE,
    )
    for axis in range(3):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[axis] = 1
        minus[axis] = -1
        first[:, :, :, axis] = (
            values[tuple(plus)] - values[tuple(minus)]
        ) / (2.0 * h)

    a, b, c = 0, 1, 2
    velocity = torch.stack(
        (
            first[:, :, a, 1] + first[:, :, b, 2],
            -first[:, :, a, 0] + first[:, :, c, 2],
            -first[:, :, b, 0] - first[:, :, c, 1],
        ),
        dim=2,
    )
    hessian = torch.empty(
        (len(xyz), cfg.modes, 3, 3, 3),
        device=model_device,
        dtype=REDUCTION_DTYPE,
    )
    for first_axis in range(3):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[first_axis] = 1
        minus[first_axis] = -1
        hessian[:, :, :, first_axis, first_axis] = (
            values[tuple(plus)] - 2.0 * origin + values[tuple(minus)]
        ) / (h * h)
        for second_axis in range(first_axis + 1, 3):
            pp = [0, 0, 0]
            pm = [0, 0, 0]
            mp = [0, 0, 0]
            mm = [0, 0, 0]
            pp[first_axis], pp[second_axis] = 1, 1
            pm[first_axis], pm[second_axis] = 1, -1
            mp[first_axis], mp[second_axis] = -1, 1
            mm[first_axis], mm[second_axis] = -1, -1
            mixed = (
                values[tuple(pp)]
                - values[tuple(pm)]
                - values[tuple(mp)]
                + values[tuple(mm)]
            ) / (4.0 * h * h)
            hessian[:, :, :, first_axis, second_axis] = mixed
            hessian[:, :, :, second_axis, first_axis] = mixed

    gradient = torch.empty(
        (len(xyz), cfg.modes, 3, 3),
        device=model_device,
        dtype=REDUCTION_DTYPE,
    )
    for derivative in range(3):
        gradient[:, :, 0, derivative] = (
            hessian[:, :, a, 1, derivative]
            + hessian[:, :, b, 2, derivative]
        )
        gradient[:, :, 1, derivative] = (
            -hessian[:, :, a, 0, derivative]
            + hessian[:, :, c, 2, derivative]
        )
        gradient[:, :, 2, derivative] = (
            -hessian[:, :, b, 0, derivative]
            - hessian[:, :, c, 1, derivative]
        )
    divergence = (
        gradient[:, :, 0, 0]
        + gradient[:, :, 1, 1]
        + gradient[:, :, 2, 2]
    )
    return {
        "potential": origin,
        "velocity": velocity,
        "gradient": gradient,
        "divergence": divergence,
    }


def discrete_all_mode_velocity(
    cfg: Config,
    model: OfficialPeelingFields,
    points: np.ndarray | torch.Tensor,
    alpha_value: float,
) -> torch.Tensor:
    """Evaluate all divergence-free velocities with the seven-point stencil."""
    model_device = next(model.parameters()).device
    xyz = torch.as_tensor(points, device=model_device, dtype=DTYPE)
    h = cfg.difference_step
    offsets = [
        (0, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ]
    shifted = torch.cat(
        [
            xyz + xyz.new_tensor(offset).reshape(1, 3).mul(h)
            for offset in offsets
        ],
        dim=0,
    )
    alpha = torch.full(
        (len(shifted), 1),
        alpha_value,
        device=model_device,
        dtype=DTYPE,
    )
    evaluated = model(shifted, alpha).to(REDUCTION_DTYPE)
    chunks = evaluated.reshape(len(offsets), len(xyz), cfg.modes, 3)
    values = {offset: chunks[index] for index, offset in enumerate(offsets)}
    first = torch.empty(
        (len(xyz), cfg.modes, 3, 3),
        device=model_device,
        dtype=REDUCTION_DTYPE,
    )
    for axis in range(3):
        plus = [0, 0, 0]
        minus = [0, 0, 0]
        plus[axis] = 1
        minus[axis] = -1
        first[:, :, :, axis] = (
            values[tuple(plus)] - values[tuple(minus)]
        ) / (2.0 * h)
    a, b, c = 0, 1, 2
    return torch.stack(
        (
            first[:, :, a, 1] + first[:, :, b, 2],
            -first[:, :, a, 0] + first[:, :, c, 2],
            -first[:, :, b, 0] - first[:, :, c, 1],
        ),
        dim=2,
    )


def _sobol_box(
    count: int,
    box: tuple[float, float, float, float, float, float],
    seed: int,
    cfg: Config,
    alpha: float,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    engine = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed)
    xmin, xmax, ymin, ymax, zmin, zmax = box
    pieces: list[np.ndarray] = []
    remaining = count
    while remaining:
        unit = engine.draw(max(3 * remaining, 128)).cpu().numpy()
        candidates = np.column_stack(
            (
                xmin + (xmax - xmin) * unit[:, 0],
                ymin + (ymax - ymin) * unit[:, 1],
                zmin + (zmax - zmin) * unit[:, 2],
            )
        ).astype(np.float32)
        accepted = candidates[~inside_obstacle(cfg, candidates, alpha)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces)


def fluid_points(cfg: Config, n: int, alpha: float, seed: int) -> np.ndarray:
    if cfg.domain_type == "periodic_closed":
        domain = (
            cfg.xmin,
            cfg.xmax,
            cfg.ymin,
            cfg.ymax,
            cfg.zmin,
            cfg.zmax,
        )
        return _sobol_box(n, domain, seed, cfg, alpha)
    if cfg.sampling_mode != "mixed":
        raise ValueError(f"Unknown sampling mode: {cfg.sampling_mode}")
    fractions = np.asarray(
        (cfg.global_fraction, cfg.obstacle_fraction, cfg.wake_fraction),
        dtype=np.float64,
    )
    if np.any(fractions < 0.0) or float(fractions.sum()) <= 0.0:
        raise ValueError("sampling fractions must be nonnegative with positive sum")
    fractions /= fractions.sum()
    counts = np.floor(n * fractions).astype(int)
    counts[0] += n - int(counts.sum())
    domain = (
        cfg.xmin,
        cfg.xmax,
        cfg.ymin,
        cfg.ymax,
        cfg.zmin,
        cfg.zmax,
    )
    global_points = _sobol_box(int(counts[0]), domain, seed, cfg, alpha)
    band = cfg.obstacle_band_radii * cfg.obstacle_radius
    obstacle_box = (
        max(cfg.xmin, cfg.obstacle_x - band),
        min(cfg.xmax, cfg.obstacle_x + band),
        max(cfg.ymin, cfg.obstacle_y - band),
        min(cfg.ymax, cfg.obstacle_y + band),
        max(cfg.zmin, cfg.obstacle_z - band),
        min(cfg.zmax, cfg.obstacle_z + band),
    )
    obstacle_points = _sobol_box(
        int(counts[1]), obstacle_box, seed + 1, cfg, alpha
    )
    half = cfg.wake_half_width_radii * cfg.obstacle_radius
    wake_box = (
        min(cfg.xmax, cfg.obstacle_x + 0.25 * cfg.obstacle_radius),
        min(
            cfg.xmax,
            cfg.obstacle_x + cfg.wake_length_radii * cfg.obstacle_radius,
        ),
        max(cfg.ymin, cfg.obstacle_y - half),
        min(cfg.ymax, cfg.obstacle_y + half),
        max(cfg.zmin, cfg.obstacle_z - half),
        min(cfg.zmax, cfg.obstacle_z + half),
    )
    wake_points = _sobol_box(int(counts[2]), wake_box, seed + 2, cfg, alpha)
    points = np.concatenate((global_points, obstacle_points, wake_points), axis=0)
    return points[np.random.default_rng(seed).permutation(len(points))]


def continuous_alpha(cfg: Config, epoch: int) -> float:
    if cfg.domain_type == "periodic_closed":
        return 0.0
    if cfg.endpoint_period > 0 and epoch % cfg.endpoint_period == 0:
        return 0.0 if (epoch // cfg.endpoint_period) % 2 else 1.0
    engine = torch.quasirandom.SobolEngine(
        1, scramble=True, seed=cfg.seed + 700_000 + epoch
    )
    return float(engine.draw(1).item())


def pack_velocity(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return fields["velocity"].T.reshape(-1)


def pack_gradient(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return fields["gradient"].permute(1, 2, 0).reshape(-1)


def pack_potential(fields: dict[str, torch.Tensor]) -> torch.Tensor:
    return fields["potential"].T.reshape(-1)


def raw_mode_rayleigh_energies(
    cfg: Config,
    raw_fields: dict[str, torch.Tensor],
    weight: float,
) -> torch.Tensor:
    """Evaluate Algorithm 2's pre-projection energies for causal sorting."""
    velocity = raw_fields["velocity"].to(REDUCTION_DTYPE)
    gradient = raw_fields["gradient"].to(REDUCTION_DTYPE)
    mass = weight * velocity.square().sum(dim=(0, 2))
    stiffness = weight * gradient.square().sum(dim=(0, 2, 3))
    return stiffness / torch.clamp(mass, min=cfg.mass_floor)


def causal_mode_order(
    cfg: Config,
    raw_fields: dict[str, torch.Tensor],
    weight: float,
) -> tuple[torch.Tensor, list[int]]:
    """Return raw Rayleigh energies and the SSS causal dominance order."""
    energies = raw_mode_rayleigh_energies(cfg, raw_fields, weight)
    if cfg.sss_ordering == "fixed":
        order = list(range(cfg.modes))
    elif cfg.sss_ordering == "dynamic":
        order = torch.argsort(
            energies.detach(), stable=True
        ).cpu().tolist()
    else:
        raise ValueError(
            "sss_ordering must be 'fixed' or 'dynamic', "
            f"got {cfg.sss_ordering!r}"
        )
    return energies, order


def least_squares_peel(
    raw_velocity: torch.Tensor,
    raw_gradient: torch.Tensor,
    raw_potential: torch.Tensor,
    previous_velocity: list[torch.Tensor],
    previous_gradient: list[torch.Tensor],
    previous_potential: list[torch.Tensor],
    weight: float,
    mass_floor: float,
) -> dict[str, torch.Tensor]:
    if previous_velocity:
        u_previous = torch.stack(previous_velocity, dim=1)
        g_previous = torch.stack(previous_gradient, dim=1)
        a_previous = torch.stack(previous_potential, dim=1)
        # Causal peeling has already made ``u_previous`` orthonormal under
        # this quadrature.  Algorithm 2's projection coefficients therefore
        # reduce exactly to weighted inner products; solving its identity
        # Gram system again is mathematically redundant and becomes cubic in
        # the number of modes.
        q = weight * (u_previous.T @ raw_velocity)
        reconstructed_velocity = u_previous @ q
        reconstructed_gradient = g_previous @ q
        reconstructed_potential = a_previous @ q
    else:
        q = torch.empty(0, device=raw_velocity.device, dtype=raw_velocity.dtype)
        reconstructed_velocity = torch.zeros_like(raw_velocity)
        reconstructed_gradient = torch.zeros_like(raw_gradient)
        reconstructed_potential = torch.zeros_like(raw_potential)
    residual_velocity = raw_velocity - reconstructed_velocity
    residual_gradient = raw_gradient - reconstructed_gradient
    residual_potential = raw_potential - reconstructed_potential
    mass = weight * residual_velocity.square().sum()
    stiffness = weight * residual_gradient.square().sum()
    projection_mass = weight * reconstructed_velocity.square().sum()
    normalizer = torch.rsqrt(torch.clamp(mass, min=mass_floor))
    return {
        "q": q,
        "velocity": residual_velocity,
        "gradient": residual_gradient,
        "potential": residual_potential,
        "normalized_velocity": residual_velocity * normalizer,
        "normalized_gradient": residual_gradient * normalizer,
        "normalized_potential": residual_potential * normalizer,
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
    weight = approximate_fluid_volume(cfg, alpha_value) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_gradient: list[torch.Tensor] = []
    previous_potential: list[torch.Tensor] = []
    raw_velocities: list[torch.Tensor] = []
    energies: list[float] = []
    projection_ratios: list[float] = []
    losses: list[float] = []
    divergence_maxima: list[float] = []
    raw_all = discrete_all_mode_fields(
        cfg, model, points, alpha_value
    )
    raw_rayleigh, mode_order = causal_mode_order(cfg, raw_all, weight)
    differentiable_losses: list[torch.Tensor] = []
    mode_energies_by_network = [0.0] * cfg.modes
    for mode in mode_order:
        raw = {
            key: value[:, mode]
            for key, value in raw_all.items()
        }
        raw_velocity = pack_velocity(raw)
        raw_gradient = pack_gradient(raw)
        raw_potential = pack_potential(raw)
        raw_velocities.append(raw_velocity.detach())
        peeled = least_squares_peel(
            raw_velocity,
            raw_gradient,
            raw_potential,
            previous_velocity,
            previous_gradient,
            previous_potential,
            weight,
            cfg.mass_floor,
        )
        denominator = torch.clamp(peeled["mass"], min=cfg.mass_floor)
        rayleigh = peeled["stiffness"] / denominator
        projection_ratio = peeled["projection_mass"] / denominator
        loss = rayleigh / cfg.rayleigh_scale
        differentiable_losses.append(loss)
        energies.append(float(rayleigh.detach().cpu()))
        projection_ratios.append(float(projection_ratio.detach().cpu()))
        losses.append(float(loss.detach().cpu()))
        mode_energies_by_network[mode] = float(rayleigh.detach().cpu())
        divergence_maxima.append(float(raw["divergence"].abs().max().detach().cpu()))
        previous_velocity.append(peeled["normalized_velocity"].detach())
        previous_gradient.append(peeled["normalized_gradient"].detach())
        previous_potential.append(peeled["normalized_potential"].detach())

    if optimizers is not None:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        torch.stack(differentiable_losses).sum().backward()
        for mode, optimizer in enumerate(optimizers):
            torch.nn.utils.clip_grad_norm_(
                model.modes[mode].parameters(), cfg.grad_clip
            )
            optimizer.step()

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
            torch.max(
                torch.abs(gram - torch.eye(cfg.modes, device=gram.device))
            ).cpu()
        ),
        "max_discrete_divergence": float(np.max(divergence_maxima)),
        "effective_raw_rank": int((eigenvalues > threshold).sum().cpu()),
        "mode_energies": energies,
        "mode_energies_by_network": mode_energies_by_network,
        "raw_mode_energies_by_network": raw_rayleigh.detach().cpu().tolist(),
        "projection_ratios": projection_ratios,
        "mode_order": mode_order,
        "order_displacement": int(
            sum(position != mode for position, mode in enumerate(mode_order))
        ),
        "sss_ordering": cfg.sss_ordering,
    }
    n = len(points)
    potential_matrix = torch.stack(previous_potential, dim=1)
    output = {
        "potential": potential_matrix,
        "u": velocity_matrix[:n],
        "v": velocity_matrix[n : 2 * n],
        "w": velocity_matrix[2 * n :],
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
    model.eval()
    weight = approximate_fluid_volume(cfg, alpha) / len(points)
    previous_velocity: list[torch.Tensor] = []
    previous_potential: list[torch.Tensor] = []
    raw_all = discrete_all_mode_fields(cfg, model, points, float(alpha))
    _, mode_order = causal_mode_order(cfg, raw_all, weight)
    for mode in mode_order:
        raw = {
            key: value[:, mode]
            for key, value in raw_all.items()
        }
        raw_velocity = pack_velocity(raw)
        raw_potential = pack_potential(raw)
        if previous_velocity:
            u_previous = torch.stack(previous_velocity, dim=1)
            a_previous = torch.stack(previous_potential, dim=1)
            lhs = weight * (u_previous.T @ u_previous)
            rhs = weight * (u_previous.T @ raw_velocity)
            q = torch.linalg.solve(lhs, rhs)
            residual_velocity = raw_velocity - u_previous @ q
            residual_potential = raw_potential - a_previous @ q
        else:
            residual_velocity = raw_velocity
            residual_potential = raw_potential
        mass = weight * residual_velocity.square().sum()
        normalizer = torch.rsqrt(torch.clamp(mass, min=cfg.mass_floor))
        previous_velocity.append((residual_velocity * normalizer).detach())
        previous_potential.append((residual_potential * normalizer).detach())
    velocity = torch.stack(previous_velocity, dim=1)
    n = len(points)
    gram = weight * velocity.T @ velocity
    return {
        "u": velocity[:n].cpu().numpy(),
        "v": velocity[n : 2 * n].cpu().numpy(),
        "w": velocity[2 * n :].cpu().numpy(),
        "potential": torch.stack(previous_potential, dim=1).cpu().numpy(),
        "gram": gram.cpu().numpy(),
        "orth_error": float(
            torch.max(
                torch.abs(gram - torch.eye(cfg.modes, device=gram.device))
            ).cpu()
        ),
        "alpha": float(alpha),
        "mode_order": np.asarray(mode_order),
        "sss_ordering": cfg.sss_ordering,
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
        "max_discrete_divergence": float(
            np.max([x["max_discrete_divergence"] for x in items])
        ),
        "min_rank": int(np.min([x["effective_raw_rank"] for x in items])),
    }


def make_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    indices = [
        int(path.name.split("_")[-1])
        for path in root.glob("run_*")
        if path.name.split("_")[-1].isdigit()
    ]
    return root / f"run_{max(indices, default=0) + 1:03d}"


def append_log(run_dir: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    json_path = data_dir / "training_history.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    keys = sorted({key for row in history for key in row})
    with (data_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                key: (
                    json.dumps(value)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in row.items()
            }
            for row in history
        )


def save_checkpoint(
    path: Path,
    cfg: Config,
    model: OfficialPeelingFields,
    optimizers: list[torch.optim.Optimizer],
    epoch: int,
    history: list[dict[str, Any]],
    best_validation: float,
    schedulers: list[torch.optim.lr_scheduler.ReduceLROnPlateau] | None = None,
    early_stopping_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algorithm": (
            "shape_space_spectra_dynamic_causal_sort"
            if cfg.sss_ordering == "dynamic"
            else "sss_3d_antisymmetric_discrete_divergence"
        ),
        "config": asdict(cfg),
        "model": model.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "epoch": epoch,
        "history": history,
        "best_validation": best_validation,
        "schedulers": (
            [scheduler.state_dict() for scheduler in schedulers]
            if schedulers is not None
            else []
        ),
        "early_stopping_state": early_stopping_state or {},
        "training_uses_gt_or_ns": False,
        "inference_requires_optimization": False,
        "divergence_loss_used": False,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validation_pass(
    cfg: Config, model: OfficialPeelingFields
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    details = []
    alphas = (
        np.zeros(1)
        if cfg.domain_type == "periodic_closed"
        else np.linspace(0.0, 1.0, cfg.validation_alphas)
    )
    for index, alpha in enumerate(alphas):
        points = fluid_points(
            cfg, cfg.validation_points, float(alpha), cfg.seed + 900_000 + index
        )
        with torch.no_grad():
            _, diagnostic, _ = peel_shape(
                cfg, model, points, float(alpha), None
            )
        details.append(diagnostic)
    model.train()
    return aggregate(details), details


def train(cfg: Config, run_dir: Path, resume: Path | None = None) -> None:
    set_seed(cfg.seed)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    model = OfficialPeelingFields(cfg).to(DEVICE)
    optimizers = [
        torch.optim.Adam(mode.parameters(), lr=cfg.learning_rate)
        for mode in model.modes
    ]
    if cfg.lr_scheduler == "none":
        schedulers = None
    elif cfg.lr_scheduler == "reduce_on_plateau":
        schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.lr_reduce_factor,
                patience=cfg.lr_patience_validations,
                threshold=cfg.early_stopping_min_delta,
                threshold_mode="abs",
                min_lr=cfg.min_learning_rate,
            )
            for optimizer in optimizers
        ]
    else:
        raise ValueError(
            "lr_scheduler must be 'none' or 'reduce_on_plateau', "
            f"got {cfg.lr_scheduler!r}"
        )
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    early_stopping_state: dict[str, Any] = {
        "best": math.inf,
        "bad_validations": 0,
    }
    start_epoch = 1
    if resume is not None:
        payload = torch.load(resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(payload["model"])
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        history = payload.get("history", [])
        best_validation = float(payload.get("best_validation", math.inf))
        start_epoch = int(payload["epoch"]) + 1
        scheduler_states = payload.get("schedulers", [])
        if schedulers is not None and scheduler_states:
            for scheduler, state in zip(schedulers, scheduler_states):
                scheduler.load_state_dict(state)
        loaded_early_state = payload.get("early_stopping_state", {})
        if loaded_early_state:
            early_stopping_state.update(loaded_early_state)
    (run_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf-8"
    )
    (run_dir / "protocol.json").write_text(
        json.dumps(
            {
                "reference_case": "EigenFluid-SSS/BasisNet/cases/2D_karman",
                "sss_reference": (
                    "Chang et al., Shape Space Spectra, "
                    "ACM TOG 44(4), 2025, DOI 10.1145/3731148"
                ),
                "ncl_reference": "arXiv:2210.01741 equations 3-4",
                "one_independent_network_per_mode": True,
                "one_optimizer_per_mode": True,
                "joint_training": True,
                "sss_ordering": cfg.sss_ordering,
                "dynamic_causal_sorting": cfg.sss_ordering == "dynamic",
                "sorting_criterion": (
                    "pre-projection normalized Dirichlet/Rayleigh energy"
                ),
                "gradient_causal_filtering": True,
                "activation_checkpointing": cfg.activation_checkpointing,
                "lr_scheduler": cfg.lr_scheduler,
                "early_stopping_patience_validations": (
                    cfg.early_stopping_patience_validations
                ),
                "least_squares_residual_peeling": True,
                "potential_velocity_gradient_share_projection_coefficients": True,
                "previous_modes_detached": True,
                "loss": "mean normalized Dirichlet/Rayleigh energy / 80 only",
                "projection_penalty_used_in_loss": False,
                "divergence_penalty_used_in_loss": False,
                "antisymmetric_tensor_hard_coded": True,
                "discrete_derivative": "commuting centred difference",
                "mesh_used": False,
                "geometry": (
                    "periodic closed box without inlet, outlet, or obstacle"
                    if cfg.domain_type == "periodic_closed"
                    else "analytic interpolation of exact sphere and box SDFs"
                ),
                "flow_direction": (
                    "self-contained decaying vortex flow"
                    if cfg.domain_type == "periodic_closed"
                    else "left xmin to right xmax"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    append_log(
        run_dir,
        f"start device={DEVICE} modes={cfg.modes} points={cfg.points_per_step}",
    )
    started = time.perf_counter()
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        final_epoch = epoch
        alpha = continuous_alpha(cfg, epoch)
        points = fluid_points(cfg, cfg.points_per_step, alpha, cfg.seed + 1009 * epoch)
        model.train()
        _, diagnostic, _ = peel_shape(cfg, model, points, alpha, optimizers)
        row: dict[str, Any] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            "alpha": alpha,
            "train_loss": diagnostic["loss"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic["mean_projection_ratio"],
            "train_orth_error": diagnostic["orth_error"],
            "train_max_discrete_divergence": diagnostic[
                "max_discrete_divergence"
            ],
            "train_rank": diagnostic["effective_raw_rank"],
            "train_mode_order": diagnostic["mode_order"],
            "train_order_displacement": diagnostic["order_displacement"],
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
                    "validation_max_discrete_divergence": validation[
                        "max_discrete_divergence"
                    ],
                    "validation_min_rank": validation["min_rank"],
                    "validation_mode_energies": [
                        item["mode_energies"] for item in details
                    ],
                    "validation_mode_orders": [
                        item["mode_order"] for item in details
                    ],
                }
            )
            if validation["loss"] < best_validation:
                best_validation = float(validation["loss"])
                save_checkpoint(
                    run_dir / "checkpoints" / "best_model.pt",
                    cfg,
                    model,
                    optimizers,
                    epoch,
                    history + [row],
                    best_validation,
                    schedulers,
                    early_stopping_state,
                )
            if schedulers is not None:
                for scheduler in schedulers:
                    scheduler.step(validation["loss"])
            current_lr = float(optimizers[0].param_groups[0]["lr"])
            row["learning_rate"] = current_lr
            improved_for_early_stop = (
                validation["loss"]
                < float(early_stopping_state["best"])
                - cfg.early_stopping_min_delta
            )
            if improved_for_early_stop:
                early_stopping_state["best"] = float(validation["loss"])
                early_stopping_state["bad_validations"] = 0
            else:
                early_stopping_state["bad_validations"] += 1
            append_log(
                run_dir,
                f"epoch={epoch:04d} alpha={alpha:.6f} "
                f"train={diagnostic['loss']:.6e} val={validation['loss']:.6e} "
                f"div={validation['max_discrete_divergence']:.3e} "
                f"orth={validation['orth_error']:.2e} rank={validation['min_rank']} "
                f"lr={current_lr:.3e} "
                f"order={diagnostic['mode_order'][:8]}",
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
                schedulers,
                early_stopping_state,
            )
            save_history(run_dir, history)
        if (
            cfg.early_stopping_patience_validations > 0
            and early_stopping_state["bad_validations"]
            >= cfg.early_stopping_patience_validations
        ):
            append_log(
                run_dir,
                f"early stopping epoch={epoch} "
                f"bad_validations={early_stopping_state['bad_validations']} "
                f"best={early_stopping_state['best']:.6e}",
            )
            break
    frozen_name = (
        "sss_dynamic_causal_frozen.pt"
        if cfg.sss_ordering == "dynamic"
        else "sss_official_peeling_frozen.pt"
    )
    frozen = run_dir / "checkpoints" / frozen_name
    save_checkpoint(
        frozen,
        cfg,
        model,
        optimizers,
        final_epoch,
        history,
        best_validation,
        schedulers,
        early_stopping_state,
    )
    save_history(run_dir, history)
    append_log(run_dir, f"training complete frozen={frozen}")


def self_test() -> None:
    cfg = Config(
        modes=3,
        hidden=8,
        hidden_layers=2,
        frequencies=2,
        points_per_step=12,
        validation_points=12,
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
    raw0 = discrete_mode_fields(cfg, model, 0, points, 0.371)
    if float(raw0["divergence"].abs().max().detach().cpu()) > 1.0e-9:
        raise RuntimeError("antisymmetric stencil is not discretely divergence-free")
    tensor_error = torch.max(
        torch.abs(raw0["tensor"] + raw0["tensor"].transpose(-1, -2))
    )
    if float(tensor_error.detach().cpu()) != 0.0:
        raise RuntimeError("tensor construction is not exactly antisymmetric")
    peeled0 = least_squares_peel(
        pack_velocity(raw0),
        pack_gradient(raw0),
        pack_potential(raw0),
        [],
        [],
        [],
        approximate_fluid_volume(cfg, 0.371) / len(points),
        cfg.mass_floor,
    )
    raw1 = discrete_mode_fields(cfg, model, 1, points, 0.371)
    peeled1 = least_squares_peel(
        pack_velocity(raw1),
        pack_gradient(raw1),
        pack_potential(raw1),
        [peeled0["normalized_velocity"].detach()],
        [peeled0["normalized_gradient"].detach()],
        [peeled0["normalized_potential"].detach()],
        approximate_fluid_volume(cfg, 0.371) / len(points),
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
        raise RuntimeError("later-mode loss leaked into an earlier mode")
    if not any(
        parameter.grad is not None and float(parameter.grad.abs().max().cpu()) > 0.0
        for parameter in model.modes[1].parameters()
    ):
        raise RuntimeError("peeled mode received no gradient")
    model.zero_grad(set_to_none=True)
    _, diagnostic, _ = peel_shape(cfg, model, points, 0.371, optimizers)
    if not math.isfinite(diagnostic["loss"]):
        raise RuntimeError("non-finite peeling loss")
    if diagnostic["orth_error"] > 1.0e-7:
        raise RuntimeError("peeling failed orthogonality test")
    inferred = infer_basis(cfg, model, points, 0.373)
    if inferred["u"].shape != (cfg.points_per_step, cfg.modes):
        raise RuntimeError("one-shot inference returned the wrong shape")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "device": str(DEVICE),
                "max_discrete_divergence": diagnostic[
                    "max_discrete_divergence"
                ],
                "antisymmetric_tensor": "exact",
                "mesh_used": False,
                "orth_error": diagnostic["orth_error"],
                "rank": diagnostic["effective_raw_rank"],
                "causal_stop_gradient": "passed",
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
        "validation_alphas",
        "probe_every",
        "validate_every",
        "checkpoint_every",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("probe_alpha", "learning_rate", "difference_step"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
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
        if hasattr(args, field) and getattr(args, field) is not None:
            values[field] = getattr(args, field)
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
