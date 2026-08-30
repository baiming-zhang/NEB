"""Generate an independent rotating-fan Navier--Stokes reference video.

This preview solver follows the existing 3D EigenFluid reference convention:

* the cube is [-pi, pi]^3 and Re=500;
* velocity is advanced by a dealiased GPU pseudo-spectral projection method;
* a smooth Brinkman mask imposes a genuinely rotating no-slip rotor in the
  inertial frame, u_body = Omega x r;
* smooth fringe regions impose left inflow, a right convective/sponge outlet,
  and ambient far-field data on a rotation-invariant cylindrical boundary.

The fringe formulation makes the interior solution compatible with FFT
projection.  It is an independent numerical GT preview, not a claim of
grid-converged engineering CFD.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CASE_ROOT / "configs" / "gt_rotating_open_re500_preview.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value.square() * (3.0 - 2.0 * value)


class RotatingFanReference:
    """GPU incompressible pseudo-spectral LES with a moving Brinkman rotor."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.n = int(cfg["resolution"])
        self.device = DEVICE
        self.xmin = float(cfg["xmin"])
        self.xmax = float(cfg["xmax"])
        self.length = self.xmax - self.xmin
        self.dx = self.length / self.n
        self.nu = float(cfg["inlet_speed"]) * (
            2.0 * float(cfg["tip_radius"])
        ) / float(cfg["reynolds"])
        self.dt_max = float(cfg["dt_max"])
        self.omega = float(cfg["rotation_rate"])
        self.smagorinsky = float(cfg["smagorinsky"])
        self.solid_eta = float(cfg["solid_penalty_eta"])
        self.fringe_eta = float(cfg["fringe_penalty_eta"])
        self.epsilon = 1.35 * self.dx

        self.axis = np.linspace(
            self.xmin,
            self.xmax,
            self.n,
            endpoint=False,
            dtype=np.float32,
        )
        axis_t = torch.as_tensor(self.axis, device=self.device)
        self.x, self.y, self.z = torch.meshgrid(
            axis_t, axis_t, axis_t, indexing="ij"
        )
        self.radius = torch.sqrt(self.y.square() + self.z.square() + 1.0e-12)
        self.azimuth = torch.atan2(self.z, self.y)

        frequency = (
            torch.fft.fftfreq(self.n, d=self.dx, device=self.device)
            * (2.0 * math.pi)
        )
        self.kx, self.ky, self.kz = torch.meshgrid(
            frequency, frequency, frequency, indexing="ij"
        )
        self.k2 = self.kx.square() + self.ky.square() + self.kz.square()
        mode = torch.fft.fftfreq(self.n, d=1.0 / self.n, device=self.device)
        mx, my, mz = torch.meshgrid(mode, mode, mode, indexing="ij")
        cutoff = self.n // 3
        self.dealias = (
            (mx.abs() <= cutoff)
            & (my.abs() <= cutoff)
            & (mz.abs() <= cutoff)
        )
        self.fringe = self._make_fringe()
        self.ambient = torch.zeros(
            (3, self.n, self.n, self.n),
            device=self.device,
            dtype=torch.float32,
        )
        self.ambient[0] = float(cfg["inlet_speed"])
        self.uhat = self._project(self._fft(self.ambient))
        self.time = 0.0
        self.last_chi = self.rotor_mask(0.0)

    @staticmethod
    def _fft(value: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(value, dim=(-3, -2, -1))

    @staticmethod
    def _ifft(value: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(value, dim=(-3, -2, -1)).real

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        dot = self.kx * value[0] + self.ky * value[1] + self.kz * value[2]
        inverse = torch.where(self.k2 > 0.0, self.k2.reciprocal(), 0.0)
        projected = value.clone()
        projected[0] -= self.kx * dot * inverse
        projected[1] -= self.ky * dot * inverse
        projected[2] -= self.kz * dot * inverse
        return projected * self.dealias

    def _curl(self, uhat: torch.Tensor) -> torch.Tensor:
        return self._ifft(
            torch.stack(
                (
                    1j * (self.ky * uhat[2] - self.kz * uhat[1]),
                    1j * (self.kz * uhat[0] - self.kx * uhat[2]),
                    1j * (self.kx * uhat[1] - self.ky * uhat[0]),
                )
            )
        )

    def _make_fringe(self) -> torch.Tensor:
        width = float(self.cfg["fringe_width"])
        left = smoothstep01((self.xmin + width - self.x) / width)
        right = smoothstep01(
            (self.x - (float(self.cfg["xmax"]) - width)) / width
        )
        farfield_radius = float(self.cfg["farfield_radius"])
        lateral = smoothstep01(
            (
                self.radius
                - (farfield_radius - width)
            )
            / width
        )
        return 1.0 - (1.0 - left) * (1.0 - right) * (1.0 - lateral)

    def _camber(self, xi: torch.Tensor) -> torch.Tensor:
        m = float(self.cfg["camber"])
        p = float(self.cfg["camber_position"])
        fore = m / (p * p) * (2.0 * p * xi - xi.square())
        aft = m / ((1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * xi - xi.square()
        )
        return torch.where(xi < p, fore, aft)

    def _half_thickness(self, xi: torch.Tensor) -> torch.Tensor:
        t = float(self.cfg["thickness_ratio"])
        clipped = torch.clamp(xi, 0.0, 1.0)
        return 5.0 * t * (
            0.2969 * torch.sqrt(clipped)
            - 0.1260 * clipped
            - 0.3516 * clipped.square()
            + 0.2843 * clipped.pow(3)
            - 0.1036 * clipped.pow(4)
        )

    def rotor_mask(self, theta: float) -> torch.Tensor:
        """Smooth union of lofted blades and a capped cylindrical hub."""
        count = int(self.cfg["blade_count"])
        sector = 2.0 * math.pi / count
        phase = torch.remainder(
            self.azimuth - theta + 0.5 * sector, sector
        ) - 0.5 * sector
        tangential = self.radius * phase

        root = float(self.cfg["hub_radius"])
        tip = float(self.cfg["tip_radius"])
        span = torch.clamp((self.radius - root) / (tip - root), 0.0, 1.0)
        smooth_span = span.square() * (3.0 - 2.0 * span)
        beta = torch.deg2rad(
            float(self.cfg["beta_root_deg"])
            + (
                float(self.cfg["beta_tip_deg"])
                - float(self.cfg["beta_root_deg"])
            )
            * span.pow(0.82)
        )
        chord = float(self.cfg["chord_root"]) + (
            float(self.cfg["chord_tip"]) - float(self.cfg["chord_root"])
        ) * smooth_span
        chord = chord * (1.0 - 0.20 * span.pow(10))
        sweep = 0.18 * span.square() - 0.035 * span
        rake = -0.08 * span.square()
        xlocal = self.x - rake
        tlocal = tangential - sweep
        chord_coordinate = xlocal * torch.sin(beta) + tlocal * torch.cos(beta)
        normal_coordinate = xlocal * torch.cos(beta) - tlocal * torch.sin(beta)
        xi = (
            chord_coordinate / torch.clamp(chord, min=1.0e-4)
            + float(self.cfg["stacking_fraction"])
        )
        camber = chord * self._camber(torch.clamp(xi, 0.0, 1.0))
        thickness = chord * self._half_thickness(xi)

        normal_distance = torch.abs(normal_coordinate - camber) - thickness
        leading_distance = -xi * chord
        trailing_distance = (xi - 1.0) * chord
        root_distance = root - self.radius
        tip_distance = self.radius - tip
        blade_distance = torch.maximum(
            torch.maximum(normal_distance, leading_distance),
            torch.maximum(
                trailing_distance, torch.maximum(root_distance, tip_distance)
            ),
        )
        blade = torch.sigmoid(-blade_distance / self.epsilon)

        hub_axial = torch.abs(self.x) - 0.5 * float(self.cfg["hub_length"])
        hub_radial = self.radius - root
        hub_distance = torch.maximum(hub_axial, hub_radial)
        hub = torch.sigmoid(-hub_distance / self.epsilon)
        return torch.maximum(blade, hub)

    def _rhs(self, uhat: torch.Tensor) -> torch.Tensor:
        velocity = self._ifft(uhat)
        vorticity = self._curl(uhat)
        rotational = torch.cross(velocity, vorticity, dim=0)
        resolved_rate = torch.sqrt(torch.mean(vorticity.square()))
        eddy_viscosity = (
            self.smagorinsky * self.dx
        ) ** 2 * resolved_rate
        nonlinear = self._project(self._fft(rotational))
        return nonlinear - (self.nu + eddy_viscosity) * self.k2 * uhat

    def _penalize(
        self, uhat: torch.Tensor, dt: float, target_time: float
    ) -> torch.Tensor:
        velocity = self._ifft(uhat)
        chi = self.rotor_mask(self.omega * target_time)
        solid_target = torch.stack(
            (
                torch.zeros_like(self.x),
                -self.omega * self.z,
                self.omega * self.y,
            )
        )
        solid_rate = chi / self.solid_eta
        fringe_rate = self.fringe / self.fringe_eta
        rate = solid_rate + fringe_rate
        target = (
            solid_rate.unsqueeze(0) * solid_target
            + fringe_rate.unsqueeze(0) * self.ambient
        ) / torch.clamp(rate.unsqueeze(0), min=1.0e-8)
        decay = torch.exp(-dt * rate).unsqueeze(0)
        velocity = target + decay * (velocity - target)
        self.last_chi = chi
        return self._project(self._fft(velocity))

    def step(self, target_time: float | None = None) -> float:
        velocity = self._ifft(self.uhat)
        maximum = float(
            torch.linalg.vector_norm(velocity, dim=0).max().detach().cpu()
        )
        tip_speed = self.omega * float(self.cfg["tip_radius"])
        dt = min(
            self.dt_max,
            0.30 * self.dx / max(maximum, tip_speed, 1.0e-6),
        )
        if target_time is not None:
            dt = min(dt, target_time - self.time)
        first = self._rhs(self.uhat)
        predictor = self._penalize(
            self.uhat + dt * first, dt, self.time + dt
        )
        second = self._rhs(predictor)
        self.uhat = self._penalize(
            self.uhat + 0.5 * dt * (first + second),
            dt,
            self.time + dt,
        )
        self.time += dt
        return dt

    def advance_to(self, target_time: float) -> int:
        steps = 0
        while self.time + 1.0e-10 < target_time:
            self.step(target_time)
            steps += 1
        return steps

    def velocity_tensor(self) -> torch.Tensor:
        return self._ifft(self.uhat)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, float]:
        velocity = self.velocity_tensor()
        vorticity = self._curl(self.uhat)
        divergence_hat = (
            self.kx * self.uhat[0]
            + self.ky * self.uhat[1]
            + self.kz * self.uhat[2]
        )
        divergence = float(
            self._ifft(1j * divergence_hat).abs().max().detach().cpu()
        )
        return (
            velocity.detach().cpu().numpy().astype(np.float32),
            vorticity.detach().cpu().numpy().astype(np.float32),
            divergence,
        )

    def boundary_diagnostics(self, velocity: np.ndarray) -> dict[str, float]:
        band = max(1, int(round(float(self.cfg["fringe_width"]) / self.dx / 3)))
        return {
            "inlet_mean_u": float(velocity[0, :band].mean()),
            "outlet_mean_u": float(velocity[0, -band:].mean()),
            "cylindrical_farfield_mean_speed": float(
                np.linalg.norm(velocity, axis=0)[
                    np.broadcast_to(
                        (
                            np.sqrt(
                                self.axis[:, None] ** 2
                                + self.axis[None, :] ** 2
                            )
                            >= float(self.cfg["farfield_radius"])
                            - max(self.dx, float(self.cfg["fringe_width"]) / 3.0)
                        )[None, :, :],
                        (self.n, self.n, self.n),
                    )
                ].mean()
            ),
        }


def naca_section(
    x01: np.ndarray,
    thickness: float,
    camber: float,
    position: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yc = np.where(
        x01 < position,
        camber / position**2 * (2.0 * position * x01 - x01**2),
        camber / (1.0 - position) ** 2
        * ((1.0 - 2.0 * position) + 2.0 * position * x01 - x01**2),
    )
    slope = np.where(
        x01 < position,
        2.0 * camber / position**2 * (position - x01),
        2.0 * camber / (1.0 - position) ** 2 * (position - x01),
    )
    angle = np.arctan(slope)
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(x01)
        - 0.1260 * x01
        - 0.3516 * x01**2
        + 0.2843 * x01**3
        - 0.1036 * x01**4
    )
    return (
        x01 - yt * np.sin(angle),
        yc + yt * np.cos(angle),
        x01 + yt * np.sin(angle),
        yc - yt * np.cos(angle),
    )


def blade_surfaces(
    cfg: dict[str, Any], azimuth: float
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.linspace(
        float(cfg["hub_radius"]), float(cfg["tip_radius"]), 24
    )
    span = (radial - float(cfg["hub_radius"])) / (
        float(cfg["tip_radius"]) - float(cfg["hub_radius"])
    )
    smooth_span = span**2 * (3.0 - 2.0 * span)
    beta = np.deg2rad(
        float(cfg["beta_root_deg"])
        + (
            float(cfg["beta_tip_deg"]) - float(cfg["beta_root_deg"])
        )
        * span**0.82
    )
    chord = float(cfg["chord_root"]) + (
        float(cfg["chord_tip"]) - float(cfg["chord_root"])
    ) * smooth_span
    chord *= 1.0 - 0.20 * span**10
    sweep = 0.18 * span**2 - 0.035 * span
    rake = -0.08 * span**2
    x01 = np.linspace(0.0, 1.0, 34)
    xu, yu, xl, yl = naca_section(
        x01,
        float(cfg["thickness_ratio"]),
        float(cfg["camber"]),
        float(cfg["camber_position"]),
    )
    er = np.array([0.0, math.cos(azimuth), math.sin(azimuth)])
    et = np.array([0.0, -math.sin(azimuth), math.cos(azimuth)])
    ex = np.array([1.0, 0.0, 0.0])
    upper = np.empty((len(radial), len(x01), 3))
    lower = np.empty_like(upper)
    stacking = float(cfg["stacking_fraction"])
    for index in range(len(radial)):
        dc = math.sin(beta[index]) * ex + math.cos(beta[index]) * et
        dn = math.cos(beta[index]) * ex - math.sin(beta[index]) * et
        center = radial[index] * er + sweep[index] * et + rake[index] * ex
        upper[index] = (
            center
            + chord[index] * (xu - stacking)[:, None] * dc
            + chord[index] * yu[:, None] * dn
        )
        lower[index] = (
            center
            + chord[index] * (xl - stacking)[:, None] * dc
            + chord[index] * yl[:, None] * dn
        )
    return upper, lower


def draw_rotor(ax: plt.Axes, cfg: dict[str, Any], theta: float) -> None:
    for blade in range(int(cfg["blade_count"])):
        azimuth = theta + 2.0 * math.pi * blade / int(cfg["blade_count"])
        upper, lower = blade_surfaces(cfg, azimuth)
        for surface, color in ((upper, "#d7e4ea"), (lower, "#8da8b8")):
            ax.plot_surface(
                surface[:, :, 0],
                surface[:, :, 1],
                surface[:, :, 2],
                color=color,
                linewidth=0.08,
                edgecolor=(0.1, 0.17, 0.22, 0.20),
                shade=True,
                alpha=1.0,
            )
    axial = np.linspace(
        -0.5 * float(cfg["hub_length"]),
        0.5 * float(cfg["hub_length"]),
        24,
    )
    angle = np.linspace(0.0, 2.0 * math.pi, 48)
    xx, aa = np.meshgrid(axial, angle, indexing="ij")
    radius = float(cfg["hub_radius"])
    ax.plot_surface(
        xx,
        radius * np.cos(aa),
        radius * np.sin(aa),
        color="#6f8290",
        linewidth=0.0,
        shade=True,
    )


def draw_domain(ax: plt.Axes, cfg: dict[str, Any]) -> None:
    xmin, xmax = float(cfg["xmin"]), float(cfg["xmax"])
    radius = float(cfg["farfield_radius"])
    angle = np.linspace(0.0, 2.0 * math.pi, 96)
    for endpoint, color in ((xmin, "#2879ad"), (xmax, "#b55f2b")):
        ax.plot(
            np.full_like(angle, endpoint),
            radius * np.cos(angle),
            radius * np.sin(angle),
            color=color,
            lw=0.9,
            alpha=0.72,
        )
    for azimuth in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False):
        ax.plot(
            [xmin, xmax],
            [radius * math.cos(azimuth)] * 2,
            [radius * math.sin(azimuth)] * 2,
            color="#8ca0ad",
            lw=0.48,
            alpha=0.42,
        )
    axial = np.linspace(xmin, xmax, 18)
    aa, xx = np.meshgrid(angle, axial, indexing="ij")
    ax.plot_surface(
        xx,
        radius * np.cos(aa),
        radius * np.sin(aa),
        color="#94a8b4",
        alpha=0.035,
        shade=False,
        linewidth=0.0,
    )
    ax.text(xmin, -radius, radius + 0.12, "INLET", color="#2879ad", fontsize=9)
    ax.text(xmax, -radius, radius + 0.12, "OUTLET", color="#b55f2b", fontsize=9)
    ax.text(
        0.0,
        radius + 0.18,
        radius,
        r"cylindrical far field  $R_\infty=2R_t$",
        color="#5e707b",
        fontsize=8,
        ha="center",
    )


def render_frame(
    cfg: dict[str, Any],
    axis: np.ndarray,
    velocity: np.ndarray,
    vorticity: np.ndarray,
    chi: np.ndarray,
    physical_time: float,
    rotor_angle: float,
    divergence: float,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(12.8, 7.2), dpi=100)
    figure.patch.set_facecolor("#f3f6f8")
    ax = figure.add_subplot(111, projection="3d")
    ax.set_position([0.055, 0.075, 0.89, 0.81])
    ax.set_facecolor("#f3f6f8")
    draw_domain(ax, cfg)

    mid = len(axis) // 2
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    axial = velocity[0, :, :, mid]
    normalized = np.clip((axial - 0.35) / 1.30, 0.0, 1.0)
    facecolors = colormaps["viridis"](normalized)
    facecolors[..., 3] = 0.36
    facecolors[np.abs(yy) > float(cfg["farfield_radius"]), 3] = 0.0
    ax.plot_surface(
        xx,
        yy,
        np.zeros_like(xx),
        facecolors=facecolors,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    magnitude = np.linalg.norm(vorticity, axis=0)
    yy_radial, zz_radial = np.meshgrid(axis, axis, indexing="ij")
    physical_cylinder = (
        yy_radial**2 + zz_radial**2
        <= float(cfg["farfield_radius"]) ** 2
    )
    fluid = (chi < 0.35) & physical_cylinder[None, :, :]
    usable = magnitude[fluid]
    threshold = float(np.percentile(usable, 97.3)) if usable.size else 0.0
    selected = (magnitude >= threshold) & fluid
    indices = np.argwhere(selected)
    if len(indices) > 5200:
        take = np.linspace(0, len(indices) - 1, 5200, dtype=np.int64)
        indices = indices[take]
    if len(indices):
        i, j, k = indices.T
        axial_vorticity = vorticity[0, i, j, k]
        color_scale = max(
            float(np.percentile(np.abs(axial_vorticity), 98.0)), 1.0e-6
        )
        colors = colormaps["coolwarm"](
            np.clip(0.5 + 0.5 * axial_vorticity / color_scale, 0.0, 1.0)
        )
        colors[:, 3] = 0.32
        sizes = 2.0 + 7.0 * np.clip(
            magnitude[i, j, k] / max(float(magnitude.max()), 1.0e-6),
            0.0,
            1.0,
        )
        ax.scatter(
            axis[i],
            axis[j],
            axis[k],
            c=colors,
            s=sizes,
            depthshade=False,
            linewidths=0.0,
        )

    draw_rotor(ax, cfg, rotor_angle)
    ax.set_xlim(float(cfg["xmin"]), float(cfg["xmax"]))
    radial_limit = 1.12 * float(cfg["farfield_radius"])
    ax.set_ylim(-radial_limit, radial_limit)
    ax.set_zlim(-radial_limit, radial_limit)
    ax.set_box_aspect(
        (
            (float(cfg["xmax"]) - float(cfg["xmin"]))
            / (2.0 * radial_limit),
            1.0,
            1.0,
        )
    )
    ax.view_init(elev=23.0, azim=-57.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(False)
    ax.set_title(
        "Rotating four-blade fan — independent 3D numerical GT preview\n"
        f"t = {physical_time:5.2f} s   Re = {float(cfg['reynolds']):.0f}   "
        f"|div u|∞ = {divergence:.2e}",
        fontsize=12,
        pad=12,
    )
    figure.text(
        0.02,
        0.025,
        "colored mid-plane: axial velocity   •   particles: high vorticity",
        color="#3b4e59",
        fontsize=9,
    )
    figure.savefig(output, facecolor=figure.get_facecolor())
    plt.close(figure)


def write_protocol(
    path: Path,
    cfg: dict[str, Any],
    elapsed_seconds: float,
    total_steps: int,
    max_divergence: float,
    boundary: dict[str, float],
) -> None:
    protocol = {
        "status": "independent numerical GT preview; not grid-converged",
        "frame": "inertial",
        "rotor_boundary": "true moving no-slip u_body = Omega cross r",
        "domain": "[-pi, pi]^3",
        "left_boundary": "uniform inlet through smooth fringe forcing",
        "right_boundary": "convective/sponge outlet through smooth fringe forcing",
        "lateral_boundary": (
            "rotation-invariant ambient cylindrical far field at "
            "R_infinity = 2 R_tip"
        ),
        "spatial_solver": "dealiased GPU pseudo-spectral projection",
        "time_integrator": "Heun RK2 with exact exponential Brinkman split",
        "geometry_parameters": ["beta(r)", "c(r)", "t/c"],
        "rotation_phase": "theta(t)=Omega*t is time state, not a learned parameter",
        "config": cfg,
        "elapsed_seconds": elapsed_seconds,
        "total_steps": total_steps,
        "max_divergence": max_divergence,
        "final_boundary_diagnostics": boundary,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)


def run(config_path: Path, run_dir: Path, smoke: bool) -> None:
    cfg = load_config(config_path)
    if smoke:
        cfg.update(
            {
                "resolution": 24,
                "warmup_time": 0.02,
                "video_duration": 0.08,
                "video_fps": 12,
                "cache_snapshots": 2,
                "dt_max": 0.01,
            }
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = run_dir / "frames"
    frames_dir.mkdir()
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(cfg, stream, indent=2)

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    solver = RotatingFanReference(cfg)
    started = time.perf_counter()
    total_steps = solver.advance_to(float(cfg["warmup_time"]))
    frame_count = round(float(cfg["video_duration"]) * int(cfg["video_fps"]))
    frame_times = np.arange(frame_count, dtype=np.float64) / int(cfg["video_fps"])
    cache_indices = set(
        np.rint(
            np.linspace(
                0, max(frame_count - 1, 0), int(cfg["cache_snapshots"])
            )
        )
        .astype(int)
        .tolist()
    )
    cache_velocity: list[np.ndarray] = []
    cache_time: list[float] = []
    cache_angle: list[float] = []
    cache_divergence: list[float] = []
    maximum_divergence = 0.0
    final_velocity: np.ndarray | None = None

    for frame, display_time in enumerate(frame_times):
        target = float(cfg["warmup_time"]) + float(display_time)
        total_steps += solver.advance_to(target)
        velocity, vorticity, divergence = solver.snapshot()
        final_velocity = velocity
        maximum_divergence = max(maximum_divergence, divergence)
        chi = solver.last_chi.detach().cpu().numpy().astype(np.float32)
        render_frame(
            cfg,
            solver.axis,
            velocity,
            vorticity,
            chi,
            float(display_time),
            solver.omega * solver.time,
            divergence,
            frames_dir / f"frame_{frame:04d}.png",
        )
        if frame in cache_indices:
            cache_velocity.append(velocity)
            cache_time.append(float(display_time))
            cache_angle.append(solver.omega * solver.time)
            cache_divergence.append(divergence)
        print(
            f"frame={frame + 1:03d}/{frame_count:03d} "
            f"t={display_time:.3f} solver_t={solver.time:.3f} "
            f"div={divergence:.3e}",
            flush=True,
        )

    cache_path = run_dir / "rotating_fan_gt_preview.h5"
    with h5py.File(cache_path, "w") as cache:
        cache.create_dataset(
            "velocity",
            data=np.stack(cache_velocity),
            chunks=(1, 3, solver.n, solver.n, solver.n),
            compression="lzf",
        )
        cache.create_dataset("time", data=np.asarray(cache_time))
        cache.create_dataset("angle", data=np.asarray(cache_angle))
        cache.create_dataset(
            "divergence_max", data=np.asarray(cache_divergence)
        )
        cache.create_dataset("axis", data=solver.axis)
        cache.attrs["frame"] = "inertial"
        cache.attrs["status"] = "numerical GT preview; not grid-converged"
        cache.attrs["rotation_rate"] = float(cfg["rotation_rate"])
        cache.attrs["warmup_angle"] = (
            float(cfg["rotation_rate"]) * float(cfg["warmup_time"])
        )
        cache.attrs["reynolds"] = float(cfg["reynolds"])
        cache.attrs["geometry_parameter_count"] = 3

    video_path = run_dir / "rotating_fan_gt_3s.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(int(cfg["video_fps"])),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    elapsed = time.perf_counter() - started
    assert final_velocity is not None
    boundary = solver.boundary_diagnostics(final_velocity)
    write_protocol(
        run_dir / "protocol.json",
        cfg,
        elapsed,
        total_steps,
        maximum_divergence,
        boundary,
    )
    print(video_path)
    print(cache_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.config.resolve(), args.run_dir.resolve(), args.smoke)


if __name__ == "__main__":
    main()
