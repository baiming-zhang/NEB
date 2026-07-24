"""Independent pseudo-spectral reference solver for the driven-wake case.

The solver advances incompressible Navier--Stokes dynamics without loading or
querying a learned basis.  A constant body force drives the triply periodic
flow, while the analytic obstacle is imposed by smooth Brinkman penalization.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .training import Config, DEVICE, obstacle_sdf_numpy


class PeriodicDrivenWake:
    """GPU pseudo-spectral LES with analytic obstacle penalization."""

    def __init__(
        self,
        cfg: Config,
        alpha: float,
        resolution: int,
        dt_max: float,
        reynolds: float = 500.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.resolution = resolution
        self.alpha = alpha
        self.dt_max = dt_max
        self.length = cfg.xmax - cfg.xmin
        self.dx = self.length / resolution
        self.nu = 1.0 / reynolds
        self.smagorinsky = 0.16
        self.device = DEVICE

        axis = np.linspace(
            cfg.xmin,
            cfg.xmax,
            resolution,
            endpoint=False,
            dtype=np.float32,
        )
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        sdf = obstacle_sdf_numpy(cfg, points, alpha).reshape(
            resolution, resolution, resolution
        )
        self.axis = axis
        self.sdf = sdf
        epsilon = 1.25 * self.dx
        self.chi = torch.sigmoid(
            torch.as_tensor(-sdf / epsilon, device=self.device)
        )

        frequency = (
            torch.fft.fftfreq(resolution, d=self.dx, device=self.device)
            * (2.0 * math.pi)
        )
        self.kx, self.ky, self.kz = torch.meshgrid(
            frequency, frequency, frequency, indexing="ij"
        )
        self.k2 = self.kx.square() + self.ky.square() + self.kz.square()
        cutoff = resolution // 3
        mode = torch.fft.fftfreq(
            resolution, d=1.0 / resolution, device=self.device
        )
        mx, my, mz = torch.meshgrid(mode, mode, mode, indexing="ij")
        self.dealias = (
            (mx.abs() <= cutoff)
            & (my.abs() <= cutoff)
            & (mz.abs() <= cutoff)
        )
        self.force = torch.zeros(
            (3, resolution, resolution, resolution),
            device=self.device,
            dtype=torch.float32,
        )
        self.force[0] = 1.0 - self.chi
        self.penalty_eta = 0.04
        self.uhat = self._initial_condition(x, y, z)

    @staticmethod
    def _fft(value: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(value, dim=(-3, -2, -1))

    @staticmethod
    def _ifft(value: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(value, dim=(-3, -2, -1)).real

    def _project(self, value: torch.Tensor) -> torch.Tensor:
        dot = self.kx * value[0] + self.ky * value[1] + self.kz * value[2]
        inverse = torch.where(self.k2 > 0, self.k2.reciprocal(), 0.0)
        result = value.clone()
        result[0] -= self.kx * dot * inverse
        result[1] -= self.ky * dot * inverse
        result[2] -= self.kz * dot * inverse
        return result * self.dealias

    def _initial_condition(self, x, y, z) -> torch.Tensor:
        amplitude = 0.08
        velocity = torch.as_tensor(
            np.stack(
                (
                    amplitude * (np.sin(z) + np.cos(y)),
                    amplitude * (np.sin(x) + np.cos(z)),
                    amplitude * (np.sin(y) + np.cos(x)),
                )
            ),
            device=self.device,
            dtype=torch.float32,
        )
        velocity *= 1.0 - self.chi
        return self._project(self._fft(velocity))

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

    def _rhs(self, uhat: torch.Tensor) -> torch.Tensor:
        velocity = self._ifft(uhat)
        vorticity = self._curl(uhat)
        rotational = torch.cross(velocity, vorticity, dim=0)
        resolved_rate = torch.sqrt(torch.mean(vorticity.square()))
        eddy_viscosity = (
            self.smagorinsky * self.dx
        ) ** 2 * resolved_rate
        nonlinear = self._project(self._fft(rotational + self.force))
        return nonlinear - (self.nu + eddy_viscosity) * self.k2 * uhat

    def _penalize(self, uhat: torch.Tensor, dt: float) -> torch.Tensor:
        velocity = self._ifft(uhat)
        velocity *= torch.exp(-dt * self.chi / self.penalty_eta)
        return self._project(self._fft(velocity))

    def step(self, remaining: float | None = None) -> float:
        speed = torch.linalg.vector_norm(self._ifft(self.uhat), dim=0)
        maximum = float(speed.max().cpu())
        dt = min(self.dt_max, 0.35 * self.dx / max(maximum, 1.0e-6))
        if remaining is not None:
            dt = min(dt, remaining)
        first = self._rhs(self.uhat)
        predictor = self._penalize(self.uhat + dt * first, dt)
        second = self._rhs(predictor)
        self.uhat = self._penalize(
            self.uhat + 0.5 * dt * (first + second), dt
        )
        return dt

    def velocity(self) -> np.ndarray:
        return (
            self._ifft(self.uhat)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def divergence_max(self) -> float:
        dot = (
            self.kx * self.uhat[0]
            + self.ky * self.uhat[1]
            + self.kz * self.uhat[2]
        )
        return float(self._ifft(1j * dot).abs().max().cpu())


def generate_reference(
    cfg: Config,
    alpha: float,
    resolution: int,
    final_time: float,
    dt_max: float,
    snapshots: int,
    reynolds: float = 500.0,
) -> dict[str, np.ndarray]:
    """Advance and return an independent reference trajectory."""
    solver = PeriodicDrivenWake(
        cfg, alpha, resolution, dt_max, reynolds=reynolds
    )
    targets = np.linspace(0.0, final_time, snapshots)
    fields: list[np.ndarray] = []
    times: list[float] = []
    divergence: list[float] = []
    current_time = 0.0
    next_snapshot = 0
    while next_snapshot < snapshots:
        if current_time + 1.0e-10 >= targets[next_snapshot]:
            fields.append(solver.velocity())
            times.append(current_time)
            divergence.append(solver.divergence_max())
            next_snapshot += 1
            continue
        current_time += solver.step(targets[next_snapshot] - current_time)
    return {
        "velocity": np.stack(fields),
        "time": np.asarray(times),
        "divergence_max": np.asarray(divergence),
        "axis": solver.axis,
        "sdf": solver.sdf,
        "alpha": np.asarray(alpha),
        "reynolds": np.asarray(reynolds),
    }


def save_reference(path: Path, reference: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)
