"""Generate one independent 128^3 Navier--Stokes ground-truth trajectory.

The solver is deliberately independent of both the learned EigenFluid basis and
the finite-difference Laplacian basis.  It advances the three-dimensional
incompressible Navier--Stokes equations in rotational form with a Fourier
pseudo-spectral projection, 2/3 de-aliasing, RK4, exact viscous damping, and
Strang-split Brinkman volume penalization for the rounded-box/metaball solids.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "train_validate.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "winner_k64_e2000.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "independent_dns_gt_128"


def load_training_module() -> Any:
    spec = importlib.util.spec_from_file_location("dns_gt_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = load_training_module()


@dataclass(frozen=True)
class DNSConfig:
    resolution: int = 128
    frames: int = 30
    fps: int = 10
    frame_dt: float = 0.04
    substeps_per_frame: int = 8
    viscosity: float = 2.5e-4
    penalty_time_scale: float = 1.0e-3
    penalty_smoothing_cells: float = 0.75
    geometry_token: int = 12_000_001
    initial_rms_speed: float = 0.32
    warmup_substeps: int = 4
    seed: int = 20_260_826
    isosurface_low_quantile: float = 0.90
    isosurface_high_quantile: float = 0.975


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def cell_center_geometry(
    cfg: Any, resolution: int, geometry: np.ndarray
) -> dict[str, np.ndarray]:
    coordinates = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    outer_level = training.outer_level_numpy(cfg, points)
    obstacle_level = training.obstacle_level_numpy(cfg, points, geometry)
    fluid = ((outer_level > 0.0) & (obstacle_level > 0.0)).reshape(
        resolution, resolution, resolution
    )
    obstacle = (obstacle_level <= 0.0).reshape(fluid.shape)
    return {
        "coordinates": coordinates,
        "fluid_mask": fluid,
        "obstacle_mask": obstacle,
        "outer_level": outer_level.reshape(fluid.shape).astype(np.float32),
        "obstacle_level": obstacle_level.reshape(fluid.shape).astype(np.float32),
    }


class SpectralOperators:
    def __init__(self, resolution: int, device: torch.device) -> None:
        self.resolution = resolution
        self.device = device
        self.spacing = 1.0 / resolution
        frequencies = (
            2.0
            * np.pi
            * torch.fft.fftfreq(resolution, d=self.spacing, device=device)
        )
        kz, ky, kx = torch.meshgrid(
            frequencies, frequencies, frequencies, indexing="ij"
        )
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.k2 = kx.square() + ky.square() + kz.square()
        self.safe_k2 = self.k2.clone()
        self.safe_k2[0, 0, 0] = 1.0
        integer_frequency = torch.fft.fftfreq(resolution, d=1.0 / resolution, device=device)
        mz, my, mx = torch.meshgrid(
            integer_frequency, integer_frequency, integer_frequency, indexing="ij"
        )
        cutoff = resolution // 3
        self.dealias = (
            (torch.abs(mx) <= cutoff)
            & (torch.abs(my) <= cutoff)
            & (torch.abs(mz) <= cutoff)
        )

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return (1, 2, 3)

    def fft(self, field: torch.Tensor) -> torch.Tensor:
        return torch.fft.fftn(field, dim=self.dimensions)

    def ifft_real(self, field_hat: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifftn(field_hat, dim=self.dimensions).real

    def project(self, field_hat: torch.Tensor) -> torch.Tensor:
        dot = (
            self.kx * field_hat[0]
            + self.ky * field_hat[1]
            + self.kz * field_hat[2]
        )
        projected = torch.stack(
            (
                field_hat[0] - self.kx * dot / self.safe_k2,
                field_hat[1] - self.ky * dot / self.safe_k2,
                field_hat[2] - self.kz * dot / self.safe_k2,
            ),
            dim=0,
        )
        projected = projected * self.dealias[None]
        projected[:, 0, 0, 0] = 0.0
        return projected

    def curl_hat(self, velocity_hat: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                1j * (self.ky * velocity_hat[2] - self.kz * velocity_hat[1]),
                1j * (self.kz * velocity_hat[0] - self.kx * velocity_hat[2]),
                1j * (self.kx * velocity_hat[1] - self.ky * velocity_hat[0]),
            ),
            dim=0,
        )

    def divergence_relative_l2(self, velocity_hat: torch.Tensor) -> float:
        divergence_hat = 1j * (
            self.kx * velocity_hat[0]
            + self.ky * velocity_hat[1]
            + self.kz * velocity_hat[2]
        )
        numerator = torch.linalg.vector_norm(divergence_hat)
        denominator = torch.linalg.vector_norm(
            torch.sqrt(self.k2)[None] * velocity_hat
        ).clamp_min(1.0e-20)
        return float((numerator / denominator).item())


def independent_initial_velocity(
    resolution: int, seed: int, device: torch.device
) -> torch.Tensor:
    coordinates = (
        torch.arange(resolution, device=device, dtype=torch.float32) + 0.5
    ) / resolution
    z, y, x = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    two_pi = 2.0 * np.pi
    taylor_green = torch.stack(
        (
            torch.sin(two_pi * x) * torch.cos(two_pi * y) * torch.cos(two_pi * z),
            -torch.cos(two_pi * x) * torch.sin(two_pi * y) * torch.cos(two_pi * z),
            torch.zeros_like(x),
        ),
        dim=0,
    )
    abc = torch.stack(
        (
            torch.sin(4.0 * np.pi * z) + 0.8 * torch.cos(4.0 * np.pi * y),
            0.9 * torch.sin(4.0 * np.pi * x) + torch.cos(4.0 * np.pi * z),
            0.8 * torch.sin(4.0 * np.pi * y) + 0.9 * torch.cos(4.0 * np.pi * x),
        ),
        dim=0,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    phases = 2.0 * np.pi * torch.rand(3, generator=generator, device=device)
    perturbation = torch.stack(
        (
            torch.sin(6.0 * np.pi * y + phases[0]) * torch.cos(4.0 * np.pi * z),
            torch.sin(6.0 * np.pi * z + phases[1]) * torch.cos(4.0 * np.pi * x),
            torch.sin(6.0 * np.pi * x + phases[2]) * torch.cos(4.0 * np.pi * y),
        ),
        dim=0,
    )
    return 0.68 * taylor_green + 0.26 * abc + 0.06 * perturbation


class PseudoSpectralDNS:
    def __init__(
        self,
        protocol: DNSConfig,
        solid_fraction: np.ndarray,
        fluid_mask: np.ndarray,
        device: torch.device,
    ) -> None:
        self.protocol = protocol
        self.device = device
        self.operators = SpectralOperators(protocol.resolution, device)
        self.solid_fraction = torch.as_tensor(
            solid_fraction, device=device, dtype=torch.float32
        )
        self.fluid_mask = torch.as_tensor(fluid_mask, device=device)
        self.dt = protocol.frame_dt / protocol.substeps_per_frame
        self.half_penalty = torch.exp(
            -0.5 * self.dt * self.solid_fraction / protocol.penalty_time_scale
        )
        self.viscous_factor = torch.exp(
            -0.5 * protocol.viscosity * self.operators.k2 * self.dt
        )

    def rhs_hat(self, velocity_hat: torch.Tensor) -> torch.Tensor:
        velocity = self.operators.ifft_real(velocity_hat)
        vorticity = self.operators.ifft_real(
            self.operators.curl_hat(velocity_hat)
        )
        nonlinear = torch.cross(velocity, vorticity, dim=0)
        return self.operators.project(self.operators.fft(nonlinear))

    def penalty_half_step(self, velocity_hat: torch.Tensor) -> torch.Tensor:
        physical = self.operators.ifft_real(velocity_hat)
        physical = physical * self.half_penalty[None]
        return self.operators.project(self.operators.fft(physical))

    def step(self, velocity_hat: torch.Tensor) -> torch.Tensor:
        velocity_hat = self.penalty_half_step(velocity_hat)
        velocity_hat = velocity_hat * self.viscous_factor[None]
        dt = self.dt
        k1 = self.rhs_hat(velocity_hat)
        k2 = self.rhs_hat(velocity_hat + 0.5 * dt * k1)
        k3 = self.rhs_hat(velocity_hat + 0.5 * dt * k2)
        k4 = self.rhs_hat(velocity_hat + dt * k3)
        velocity_hat = velocity_hat + (dt / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        velocity_hat = self.operators.project(velocity_hat)
        velocity_hat = velocity_hat * self.viscous_factor[None]
        return self.penalty_half_step(velocity_hat)

    def initialize(self) -> torch.Tensor:
        velocity = independent_initial_velocity(
            self.protocol.resolution, self.protocol.seed, self.device
        )
        velocity_hat = self.operators.project(self.operators.fft(velocity))
        for _ in range(8):
            velocity_hat = self.penalty_half_step(velocity_hat)
        physical = self.operators.ifft_real(velocity_hat)
        fluid_values = physical[:, self.fluid_mask]
        rms = torch.sqrt(torch.mean(torch.sum(fluid_values.square(), dim=0)))
        velocity_hat = velocity_hat * (self.protocol.initial_rms_speed / rms)
        for _ in range(self.protocol.warmup_substeps):
            velocity_hat = self.step(velocity_hat)
        return velocity_hat

    def fields(self, velocity_hat: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        velocity = self.operators.ifft_real(velocity_hat)
        vorticity = self.operators.ifft_real(
            self.operators.curl_hat(velocity_hat)
        )
        velocity = velocity.permute(1, 2, 3, 0).contiguous()
        vorticity = vorticity.permute(1, 2, 3, 0).contiguous()
        mask = self.fluid_mask[..., None]
        velocity = torch.where(mask, velocity, torch.zeros_like(velocity))
        vorticity = torch.where(mask, vorticity, torch.zeros_like(vorticity))
        return velocity.cpu().numpy(), vorticity.cpu().numpy()

    def diagnostics(
        self, frame: int, velocity_hat: torch.Tensor
    ) -> dict[str, float | int]:
        velocity = self.operators.ifft_real(velocity_hat)
        vorticity = self.operators.ifft_real(
            self.operators.curl_hat(velocity_hat)
        )
        fluid_velocity = velocity[:, self.fluid_mask]
        fluid_vorticity = vorticity[:, self.fluid_mask]
        solid_mask = ~self.fluid_mask
        solid_velocity = velocity[:, solid_mask]
        speed = torch.linalg.vector_norm(fluid_velocity, dim=0)
        omega = torch.linalg.vector_norm(fluid_vorticity, dim=0)
        return {
            "frame": frame,
            "time": frame * self.protocol.frame_dt,
            "kinetic_energy": float(0.5 * torch.mean(speed.square()).item()),
            "enstrophy": float(0.5 * torch.mean(omega.square()).item()),
            "rms_speed": float(torch.sqrt(torch.mean(speed.square())).item()),
            "max_speed": float(torch.max(speed).item()),
            "max_vorticity": float(torch.max(omega).item()),
            "vorticity_p95": float(torch.quantile(omega, 0.95).item()),
            "cfl": float(torch.max(speed).item() * self.dt / self.operators.spacing),
            "divergence_relative_l2": self.operators.divergence_relative_l2(
                velocity_hat
            ),
            "solid_rms_speed": float(
                torch.sqrt(torch.mean(solid_velocity.square())).item()
            ),
        }


def save_frame(
    output_dir: Path,
    frame: int,
    velocity: np.ndarray,
    vorticity: np.ndarray,
) -> None:
    atomic_npy(output_dir / "velocity" / f"frame_{frame:03d}.npy", velocity)
    atomic_npy(output_dir / "vorticity" / f"frame_{frame:03d}.npy", vorticity)


def run_dns(output_dir: Path, protocol: DNSConfig) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    solver_summary_path = output_dir / "solver_summary.json"
    if solver_summary_path.exists():
        existing = json.loads(solver_summary_path.read_text(encoding="utf-8"))
        if existing.get("complete") and existing.get("protocol") == asdict(protocol):
            log("reuse complete DNS trajectory")
            return existing

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = training.Config(**config)
    geometry = training.geometry_from_token(cfg, protocol.geometry_token)
    grid = cell_center_geometry(cfg, protocol.resolution, geometry)
    fluid_mask = grid["fluid_mask"]
    smooth_fluid = gaussian_filter(
        fluid_mask.astype(np.float32),
        sigma=protocol.penalty_smoothing_cells,
        mode="wrap",
    )
    solid_fraction = np.clip(1.0 - smooth_fluid, 0.0, 1.0).astype(np.float32)
    atomic_npy(output_dir / "coordinates.npy", grid["coordinates"])
    atomic_npy(output_dir / "fluid_mask.npy", fluid_mask)
    atomic_npy(output_dir / "obstacle_mask.npy", grid["obstacle_mask"])
    atomic_npy(output_dir / "solid_fraction.npy", solid_fraction)
    atomic_npy(output_dir / "geometry_xyzr.npy", geometry)

    device = training.DEVICE
    if device.type != "cuda" and protocol.resolution >= 128:
        raise RuntimeError("formal 128^3 DNS requires CUDA")
    solver = PseudoSpectralDNS(protocol, solid_fraction, fluid_mask, device)
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    diagnostics: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != asdict(protocol):
            raise ValueError("checkpoint protocol does not match requested DNS protocol")
        velocity_hat = checkpoint["velocity_hat"].to(device)
        next_frame = int(checkpoint["next_frame"])
        diagnostics = list(checkpoint["diagnostics"])
        log(f"resume DNS at frame {next_frame}/{protocol.frames}")
    else:
        log(
            f"initialize independent DNS {protocol.resolution}^3 fluid_nodes="
            f"{np.count_nonzero(fluid_mask)}"
        )
        velocity_hat = solver.initialize()
        velocity, vorticity = solver.fields(velocity_hat)
        save_frame(output_dir, 0, velocity, vorticity)
        diagnostics.append(solver.diagnostics(0, velocity_hat))
        next_frame = 1
        atomic_torch_save(
            checkpoint_path,
            {
                "protocol": asdict(protocol),
                "next_frame": next_frame,
                "velocity_hat": velocity_hat.cpu(),
                "diagnostics": diagnostics,
            },
        )
        log(f"saved GT frame 1/{protocol.frames}: {diagnostics[-1]}")

    started = time.perf_counter()
    for frame in range(next_frame, protocol.frames):
        frame_started = time.perf_counter()
        for _ in range(protocol.substeps_per_frame):
            velocity_hat = solver.step(velocity_hat)
        if not torch.isfinite(velocity_hat).all():
            raise FloatingPointError(f"non-finite DNS state at frame {frame}")
        row = solver.diagnostics(frame, velocity_hat)
        if float(row["cfl"]) > 1.0:
            raise FloatingPointError(
                f"DNS CFL exceeded one at frame {frame}: {row['cfl']:.3f}"
            )
        velocity, vorticity = solver.fields(velocity_hat)
        save_frame(output_dir, frame, velocity, vorticity)
        diagnostics.append(row)
        write_csv(output_dir / "diagnostics.csv", diagnostics)
        atomic_torch_save(
            checkpoint_path,
            {
                "protocol": asdict(protocol),
                "next_frame": frame + 1,
                "velocity_hat": velocity_hat.cpu(),
                "diagnostics": diagnostics,
            },
        )
        log(
            f"saved GT frame {frame + 1}/{protocol.frames} "
            f"E={row['kinetic_energy']:.6e} enstrophy={row['enstrophy']:.6e} "
            f"CFL={row['cfl']:.3f} div={row['divergence_relative_l2']:.2e} "
            f"seconds={time.perf_counter() - frame_started:.1f}"
        )

    if len(diagnostics) == 1:
        write_csv(output_dir / "diagnostics.csv", diagnostics)
    velocity_files = sorted((output_dir / "velocity").glob("frame_*.npy"))
    vorticity_files = sorted((output_dir / "vorticity").glob("frame_*.npy"))
    if len(velocity_files) != protocol.frames or len(vorticity_files) != protocol.frames:
        raise RuntimeError("DNS did not produce exactly the requested number of frames")
    summary = {
        "complete": True,
        "kind": "independent_numerical_ground_truth",
        "independent_of_modal_bases": True,
        "protocol": asdict(protocol),
        "grid": {
            "layout": "cell-centered periodic FFT lattice",
            "shape_zyx": [protocol.resolution] * 3,
            "spacing": 1.0 / protocol.resolution,
            "fluid_nodes": int(np.count_nonzero(fluid_mask)),
            "geometry_token": protocol.geometry_token,
            "sphere_parameters_xyzr": geometry.tolist(),
        },
        "solver": {
            "equations": "three-dimensional incompressible Navier-Stokes",
            "form": "rotational velocity form P(u cross curl(u)) + nu Laplacian(u)",
            "spatial_discretization": "Fourier pseudo-spectral with 2/3 de-aliasing",
            "time_integrator": "RK4 with exact half-step viscosity",
            "solid_boundary": "Strang-split smooth Brinkman volume penalization",
            "pressure": "Fourier Leray projection at every stage",
        },
        "data": {
            "velocity_directory": str((output_dir / "velocity").resolve()),
            "vorticity_directory": str((output_dir / "vorticity").resolve()),
            "array_layout": "z,y,x,component_xyz",
            "dtype": "float32",
            "frames": protocol.frames,
        },
        "diagnostics": diagnostics,
        "generation_wall_seconds_excluding_resume": time.perf_counter() - started,
    }
    atomic_json(solver_summary_path, summary)
    return summary


def isosurface_vertices_faces(
    volume: np.ndarray, level: float, resolution: int, step_size: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    vertices_zyx, faces, _, _ = marching_cubes(
        volume,
        level=level,
        spacing=(1.0 / resolution,) * 3,
        step_size=step_size,
        allow_degenerate=False,
    )
    vertices_xyz = vertices_zyx[:, (2, 1, 0)] + 0.5 / resolution
    return vertices_xyz, faces


def render_isosurface_frame(
    output_path: Path,
    vorticity_path: Path,
    fluid_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    low_level: float,
    high_level: float,
    frame: int,
    protocol: DNSConfig,
) -> dict[str, int]:
    vorticity = np.load(vorticity_path, mmap_mode="r")
    magnitude = np.linalg.norm(vorticity, axis=3).astype(np.float32)
    magnitude[~fluid_mask] = 0.0
    low_vertices, low_faces = isosurface_vertices_faces(
        magnitude, low_level, protocol.resolution
    )
    high_vertices, high_faces = isosurface_vertices_faces(
        magnitude, high_level, protocol.resolution
    )
    obstacle_vertices, obstacle_faces = isosurface_vertices_faces(
        obstacle_mask.astype(np.float32), 0.5, protocol.resolution
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )
    figure = plt.figure(figsize=(8.4, 7.2), facecolor="white")
    axis = figure.add_subplot(111, projection="3d")
    low_collection = Poly3DCollection(
        low_vertices[low_faces],
        facecolor="#3B4CC0",
        edgecolor="none",
        alpha=0.22,
        rasterized=True,
    )
    high_collection = Poly3DCollection(
        high_vertices[high_faces],
        facecolor="#B40426",
        edgecolor="none",
        alpha=0.78,
        rasterized=True,
    )
    obstacle_collection = Poly3DCollection(
        obstacle_vertices[obstacle_faces],
        facecolor="#9A9A9A",
        edgecolor="#666666",
        linewidth=0.08,
        alpha=0.18,
        rasterized=True,
    )
    axis.add_collection3d(low_collection)
    axis.add_collection3d(high_collection)
    axis.add_collection3d(obstacle_collection)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_zlim(0.0, 1.0)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.5, 1.0))
    axis.set_zticks((0.0, 0.5, 1.0))
    axis.view_init(elev=23.0, azim=-54.0)
    axis.set_title(
        "Independent 128³ DNS GT — 3D isovorticity\n"
        f"frame {frame + 1:02d}/{protocol.frames}   t={frame * protocol.frame_dt:.2f}"
    )
    axis.legend(
        handles=(
            Patch(color="#3B4CC0", alpha=0.35, label=f"|ω| = {low_level:.2f}"),
            Patch(color="#B40426", alpha=0.85, label=f"|ω| = {high_level:.2f}"),
            Patch(color="#9A9A9A", alpha=0.3, label="metaball solid"),
        ),
        loc="upper left",
        frameon=False,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    if frame in (0, protocol.frames - 1):
        figure.savefig(
            output_path.with_suffix(".pdf"),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    return {
        "low_vertices": len(low_vertices),
        "low_faces": len(low_faces),
        "high_vertices": len(high_vertices),
        "high_faces": len(high_faces),
    }


def ffprobe_video(path: Path, expected_frames: int) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    stream = payload["streams"][0]
    frames = int(stream["nb_read_frames"])
    if stream["codec_name"] != "h264" or frames != expected_frames:
        raise RuntimeError(f"video verification failed: {stream}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "average_frame_rate": stream["avg_frame_rate"],
        "frames": frames,
    }


def render_isosurfaces(output_dir: Path, protocol: DNSConfig) -> dict[str, Any]:
    render_summary_path = output_dir / "isosurface_summary.json"
    if render_summary_path.exists():
        existing = json.loads(render_summary_path.read_text(encoding="utf-8"))
        if existing.get("complete"):
            log("reuse complete isosurface rendering")
            return existing
    fluid_mask = np.load(output_dir / "fluid_mask.npy")
    obstacle_mask = np.load(output_dir / "obstacle_mask.npy")
    vorticity_paths = sorted((output_dir / "vorticity").glob("frame_*.npy"))
    if len(vorticity_paths) != protocol.frames:
        raise RuntimeError("cannot render before all vorticity frames exist")
    samples: list[np.ndarray] = []
    sampled_mask = fluid_mask[::4, ::4, ::4]
    for path in vorticity_paths:
        values = np.load(path, mmap_mode="r")[::4, ::4, ::4]
        samples.append(np.linalg.norm(values, axis=3)[sampled_mask])
    sample = np.concatenate(samples)
    low_level = float(np.quantile(sample, protocol.isosurface_low_quantile))
    high_level = float(np.quantile(sample, protocol.isosurface_high_quantile))
    if not 0.0 < low_level < high_level:
        raise FloatingPointError("invalid fixed isovorticity levels")
    frame_dir = output_dir / "isosurface_frames"
    mesh_counts: list[dict[str, Any]] = []
    for frame, path in enumerate(vorticity_paths):
        output_path = frame_dir / f"frame_{frame:03d}.png"
        counts = render_isosurface_frame(
            output_path,
            path,
            fluid_mask,
            obstacle_mask,
            low_level,
            high_level,
            frame,
            protocol,
        )
        mesh_counts.append({"frame": frame, **counts})
        log(f"rendered isovorticity frame {frame + 1}/{protocol.frames}")
    video_path = output_dir / "independent_dns_gt_isovorticity.mp4"
    ffmpeg_log_path = output_dir / "ffmpeg_isosurface.log"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(protocol.fps),
        "-i",
        str(frame_dir / "frame_%03d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    ffmpeg_log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed; inspect {ffmpeg_log_path}")
    verification = ffprobe_video(video_path, protocol.frames)
    summary = {
        "complete": True,
        "fixed_isovorticity_levels": {"low": low_level, "high": high_level},
        "colors": {"low": "#3B4CC0", "high": "#B40426"},
        "mesh_counts": mesh_counts,
        "png_frames": protocol.frames,
        "representative_pdf_frames": [0, protocol.frames - 1],
        "video": verification,
    }
    atomic_json(render_summary_path, summary)
    return summary


def verify_complete(output_dir: Path, protocol: DNSConfig) -> dict[str, Any]:
    velocity_paths = sorted((output_dir / "velocity").glob("frame_*.npy"))
    vorticity_paths = sorted((output_dir / "vorticity").glob("frame_*.npy"))
    png_paths = sorted((output_dir / "isosurface_frames").glob("frame_*.png"))
    expected_shape = (protocol.resolution,) * 3 + (3,)
    for path in velocity_paths + vorticity_paths:
        array = np.load(path, mmap_mode="r")
        if array.shape != expected_shape or array.dtype != np.float32:
            raise RuntimeError(f"invalid GT array {path}: {array.shape} {array.dtype}")
    if not (
        len(velocity_paths) == len(vorticity_paths) == len(png_paths) == protocol.frames
    ):
        raise RuntimeError("final GT frame counts are incomplete")
    solver_summary = json.loads(
        (output_dir / "solver_summary.json").read_text(encoding="utf-8")
    )
    render_summary = json.loads(
        (output_dir / "isosurface_summary.json").read_text(encoding="utf-8")
    )
    result = {
        "complete": True,
        "strict_independent_gt": True,
        "resolution": [protocol.resolution] * 3,
        "frames": protocol.frames,
        "velocity_frames": len(velocity_paths),
        "vorticity_frames": len(vorticity_paths),
        "isosurface_png_frames": len(png_paths),
        "solver_summary": str((output_dir / "solver_summary.json").resolve()),
        "isosurface_summary": str((output_dir / "isosurface_summary.json").resolve()),
        "video": render_summary["video"],
        "maximum_cfl": max(row["cfl"] for row in solver_summary["diagnostics"]),
        "maximum_divergence_relative_l2": max(
            row["divergence_relative_l2"] for row in solver_summary["diagnostics"]
        ),
    }
    atomic_json(output_dir / "completion.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--substeps-per-frame", type=int, default=8)
    parser.add_argument("--warmup-substeps", type=int, default=4)
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = DNSConfig(
        resolution=args.resolution,
        frames=args.frames,
        substeps_per_frame=args.substeps_per_frame,
        warmup_substeps=args.warmup_substeps,
    )
    output_dir = args.output_dir.resolve()
    started = time.perf_counter()
    run_dns(output_dir, protocol)
    if args.skip_render:
        return
    render_isosurfaces(output_dir, protocol)
    result = verify_complete(output_dir, protocol)
    result["total_wall_seconds"] = time.perf_counter() - started
    atomic_json(output_dir / "completion.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
