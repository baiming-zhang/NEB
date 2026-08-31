"""Animate a matched-grid 2-D Navier--Stokes/FOM versus Eigenfluid ROM test.

The reduced system follows De Witt, Lessig & Fiume (2012):

    dw_k/dt = w.T @ C_k @ w - nu * mu_k * w_k + f_k,

where ``mu_k`` are positive eigenvalues of ``-Delta`` (the paper uses the
negative Laplacian-eigenvalue convention).  Here ``f_k = 0``.  The nonlinear
structure coefficients are precomputed by Galerkin projection of vorticity
advection, RK4 advances the inviscid part, the paper's energy renormalization
is applied after that substep, and viscosity is integrated exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from compare_three_gt_eigenfluid_vs_laplacian import (
    eigenfluid_basis,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
    matched_grid_ritz,
    relative_l2,
    serializable_gt,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_connected_dirichlet_psi_basis import load_model
from metaball_runtime import fluid_mask, geometry_from_token


F32 = np.float32
F64 = np.float64


@dataclass
class BasisPackage:
    name: str
    streamfunction: np.ndarray
    velocity: np.ndarray
    vorticity: np.ndarray
    eigenvalues: np.ndarray
    velocity_mass: np.ndarray
    structure: np.ndarray
    diagnostics: dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_dirichlet_laplacian(
    valid: np.ndarray, spacing: float
) -> scipy.sparse.csr_matrix:
    """Five-point positive ``-Delta`` with zero extension off the fluid mask."""
    ny, nx = valid.shape
    indices = np.full((ny, nx), -1, dtype=np.int32)
    indices[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    inverse_h2 = 1.0 / (spacing * spacing)
    for iy, ix in np.argwhere(valid):
        row = int(indices[iy, ix])
        rows.append(row)
        columns.append(row)
        values.append(4.0 * inverse_h2)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and valid[jy, jx]:
                rows.append(row)
                columns.append(int(indices[jy, jx]))
                values.append(-inverse_h2)
    count = int(np.count_nonzero(valid))
    return scipy.sparse.csr_matrix(
        (values, (rows, columns)), shape=(count, count), dtype=F64
    )


def zero_extension_gradient(
    values: np.ndarray, valid: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values)
    squeeze = matrix.ndim == 1
    if squeeze:
        matrix = matrix[:, None]
    ny, nx = valid.shape
    grid = np.zeros((ny, nx, matrix.shape[1]), dtype=matrix.dtype)
    grid[valid] = matrix
    padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="constant")
    derivative_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * spacing)
    derivative_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * spacing)
    derivative_x = derivative_x[valid]
    derivative_y = derivative_y[valid]
    if squeeze:
        return derivative_x[:, 0], derivative_y[:, 0]
    return derivative_x, derivative_y


def make_basis_package(
    name: str,
    psi_l2_orthonormal: np.ndarray,
    eigenvalues: np.ndarray,
    valid: np.ndarray,
    spacing: float,
) -> BasisPackage:
    eigenvalues64 = np.asarray(eigenvalues, dtype=F64)
    if not np.all(np.isfinite(eigenvalues64)) or np.any(eigenvalues64 <= 0.0):
        raise FloatingPointError(f"{name}: non-positive Ritz eigenvalue")

    # If psi_k has unit L2 mass and -Delta psi_k = mu_k psi_k, then
    # S_k = psi_k/sqrt(mu_k) yields an energy-normalized velocity basis
    # U_k = curl S_k and the shared vorticity basis Omega_k = -Delta S_k.
    streamfunction = (
        np.asarray(psi_l2_orthonormal, dtype=F64)
        / np.sqrt(eigenvalues64)[None, :]
    )
    velocity, vorticity = matched_grid_observables(
        streamfunction.astype(F32), valid, spacing
    )
    velocity = velocity.astype(F64)
    vorticity = vorticity.astype(F64)
    weight = spacing * spacing
    velocity_mass = weight * (
        velocity[:, 0].T @ velocity[:, 0]
        + velocity[:, 1].T @ velocity[:, 1]
    )
    velocity_mass = 0.5 * (velocity_mass + velocity_mass.T)
    energy_mass = weight * (streamfunction.T @ vorticity)
    diagnostics: dict[str, Any] = {
        "eigenvalue_min": float(eigenvalues64[0]),
        "eigenvalue_max": float(eigenvalues64[-1]),
        "first16_eigenvalues": eigenvalues64[:16].tolist(),
        "velocity_mass_max_abs_from_identity": float(
            np.max(np.abs(velocity_mass - np.eye(len(eigenvalues64))))
        ),
        "streamfunction_energy_mass_max_abs_from_identity": float(
            np.max(np.abs(energy_mass - np.eye(len(eigenvalues64))))
        ),
        "velocity_mass_condition_number": float(np.linalg.cond(velocity_mass)),
    }
    return BasisPackage(
        name=name,
        streamfunction=streamfunction,
        velocity=velocity,
        vorticity=vorticity,
        eigenvalues=eigenvalues64,
        velocity_mass=velocity_mass,
        structure=np.empty((0, 0, 0), dtype=F64),
        diagnostics=diagnostics,
    )


def precompute_structure_coefficients(
    package: BasisPackage,
    valid: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project ``-U_i dot grad(Omega_j)`` into the vorticity basis."""
    started = time.perf_counter()
    grad_x, grad_y = zero_extension_gradient(
        package.vorticity.astype(F32), valid, spacing
    )
    omega_cp = cp.asarray(package.vorticity.astype(F32))
    velocity_cp = cp.asarray(package.velocity.astype(F32))
    grad_x_cp = cp.asarray(grad_x.astype(F32))
    grad_y_cp = cp.asarray(grad_y.astype(F32))
    modes = package.vorticity.shape[1]
    projected_cp = cp.empty((modes, modes, modes), dtype=cp.float32)
    weight = F32(spacing * spacing)
    for advecting in range(modes):
        advected = -(
            velocity_cp[:, 0, advecting, None] * grad_x_cp
            + velocity_cp[:, 1, advecting, None] * grad_y_cp
        )
        projected_cp[:, advecting, :] = weight * (omega_cp.T @ advected)
    cp.cuda.Stream.null.synchronize()
    projected = cp.asnumpy(projected_cp).astype(F64)
    omega_mass = (spacing * spacing) * (
        package.vorticity.T @ package.vorticity
    )
    omega_mass = 0.5 * (omega_mass + omega_mass.T)
    regularization = 1.0e-12 * float(np.trace(omega_mass) / modes)
    structure = scipy.linalg.solve(
        omega_mass + regularization * np.eye(modes),
        projected.reshape(modes, -1),
        assume_a="pos",
        check_finite=False,
    ).reshape(modes, modes, modes)
    rng = np.random.default_rng(20260825)
    defects: list[float] = []
    for _ in range(32):
        state = rng.normal(size=modes)
        state /= np.linalg.norm(state)
        nonlinear = np.einsum(
            "kij,i,j->k", structure, state, state, optimize=True
        )
        defects.append(float(state @ package.velocity_mass @ nonlinear))
    elapsed = time.perf_counter() - started
    del omega_cp, velocity_cp, grad_x_cp, grad_y_cp, projected_cp
    cp.get_default_memory_pool().free_all_blocks()
    return structure, {
        "precompute_seconds": elapsed,
        "vorticity_mass_condition_number": float(np.linalg.cond(omega_mass)),
        "vorticity_mass_regularization": regularization,
        "sample_energy_derivative_max_abs_before_renormalization": float(
            np.max(np.abs(defects))
        ),
    }


def project_initial_velocity(
    package: BasisPackage, target_velocity: np.ndarray, spacing: float
) -> np.ndarray:
    weight = spacing * spacing
    rhs = weight * (
        package.velocity[:, 0].T @ target_velocity[:, 0]
        + package.velocity[:, 1].T @ target_velocity[:, 1]
    )
    ridge = 1.0e-12 * float(np.trace(package.velocity_mass) / len(rhs))
    return scipy.linalg.solve(
        package.velocity_mass + ridge * np.eye(len(rhs)),
        rhs,
        assume_a="pos",
        check_finite=False,
    )


def rom_nonlinear(package: BasisPackage, state: np.ndarray) -> np.ndarray:
    return np.einsum(
        "kij,i,j->k", package.structure, state, state, optimize=True
    )


def advance_rom(
    package: BasisPackage, state: np.ndarray, dt: float, viscosity: float
) -> np.ndarray:
    """De Witt RK4 advection, energy correction, then exact viscosity."""
    initial_energy = float(state @ package.velocity_mass @ state)
    k1 = rom_nonlinear(package, state)
    k2 = rom_nonlinear(package, state + 0.5 * dt * k1)
    k3 = rom_nonlinear(package, state + 0.5 * dt * k2)
    k4 = rom_nonlinear(package, state + dt * k3)
    advected = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    advected_energy = float(advected @ package.velocity_mass @ advected)
    if initial_energy > 0.0 and advected_energy > 0.0:
        advected *= math.sqrt(initial_energy / advected_energy)
    # The paper's lambda_k is negative.  With mu_k = eig(-Delta) > 0,
    # exp(nu*lambda_k*dt) becomes exp(-nu*mu_k*dt).
    return advected * np.exp(-viscosity * package.eigenvalues * dt)


def reconstruct_rom(
    package: BasisPackage, state: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi = package.streamfunction @ state
    velocity = np.einsum(
        "ncm,m->nc", package.velocity, state, optimize=True
    )
    omega = package.vorticity @ state
    return psi, velocity, omega


class NavierStokesFOM:
    """Conventional vorticity--streamfunction finite-difference FOM."""

    def __init__(
        self,
        valid: np.ndarray,
        spacing: float,
        operator: scipy.sparse.csr_matrix,
        viscosity: float,
    ) -> None:
        self.valid = valid
        self.spacing = spacing
        self.operator = operator
        self.viscosity = viscosity
        self.poisson = scipy.sparse.linalg.factorized(operator.tocsc())

    def fields(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = np.asarray(self.poisson(omega), dtype=F64)
        velocity, _ = matched_grid_observables(
            psi.astype(F32), self.valid, self.spacing
        )
        return psi, velocity.astype(F64)

    def rhs(self, omega: np.ndarray) -> np.ndarray:
        _, velocity = self.fields(omega)
        grad_x, grad_y = zero_extension_gradient(
            omega, self.valid, self.spacing
        )
        advective = velocity[:, 0] * grad_x + velocity[:, 1] * grad_y
        conservative_x, _ = zero_extension_gradient(
            velocity[:, 0] * omega, self.valid, self.spacing
        )
        _, conservative_y = zero_extension_gradient(
            velocity[:, 1] * omega, self.valid, self.spacing
        )
        # Skew-symmetric centered advection is a standard kinetic-energy-
        # conservative spatial form on a uniform grid.
        advection = 0.5 * (advective + conservative_x + conservative_y)
        return -advection - self.viscosity * (self.operator @ omega)

    def advance(self, omega: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(omega)
        k2 = self.rhs(omega + 0.5 * dt * k1)
        k3 = self.rhs(omega + 0.5 * dt * k2)
        k4 = self.rhs(omega + dt * k3)
        result = omega + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("Navier--Stokes FOM became non-finite")
        return result


def kinetic_energy(velocity: np.ndarray, spacing: float) -> float:
    return 0.5 * spacing * spacing * float(np.sum(velocity * velocity))


def enstrophy(omega: np.ndarray, spacing: float) -> float:
    return 0.5 * spacing * spacing * float(np.sum(omega * omega))


def grid_values(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    grid = np.full(valid.shape, np.nan, dtype=F32)
    grid[valid] = np.asarray(values, dtype=F32)
    return grid


def frame_metrics(
    index: int,
    physical_time: float,
    spacing: float,
    gt_velocity: np.ndarray,
    gt_omega: np.ndarray,
    learned_velocity: np.ndarray,
    learned_omega: np.ndarray,
    laplacian_velocity: np.ndarray,
    laplacian_omega: np.ndarray,
) -> dict[str, float | int]:
    return {
        "frame": index,
        "time": physical_time,
        "gt_kinetic_energy": kinetic_energy(gt_velocity, spacing),
        "gt_enstrophy": enstrophy(gt_omega, spacing),
        "learned_velocity_relative_l2": relative_l2(
            learned_velocity, gt_velocity
        ),
        "learned_vorticity_relative_l2": relative_l2(learned_omega, gt_omega),
        "learned_kinetic_energy": kinetic_energy(learned_velocity, spacing),
        "learned_enstrophy": enstrophy(learned_omega, spacing),
        "laplacian_velocity_relative_l2": relative_l2(
            laplacian_velocity, gt_velocity
        ),
        "laplacian_vorticity_relative_l2": relative_l2(
            laplacian_omega, gt_omega
        ),
        "laplacian_kinetic_energy": kinetic_energy(
            laplacian_velocity, spacing
        ),
        "laplacian_enstrophy": enstrophy(laplacian_omega, spacing),
    }


def render_frames(
    output_dir: Path,
    coordinates: np.ndarray,
    valid: np.ndarray,
    stored: list[dict[str, Any]],
    metrics: list[dict[str, float | int]],
    fps: int,
) -> tuple[Path, float]:
    started = time.perf_counter()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 120,
            "savefig.dpi": 120,
        }
    )
    all_omega = np.concatenate(
        [
            np.concatenate(
                [frame["gt_omega"], frame["learned_omega"], frame["laplacian_omega"]]
            )
            for frame in stored
        ]
    )
    omega_limit = max(float(np.percentile(np.abs(all_omega), 99.5)), 1.0e-6)
    panel_titles = [
        r"Navier--Stokes GT ($128^2$)",
        r"Learned Eigenfluid ($K=64$)",
        r"Laplacian Eigenfluid ($K=64$)",
    ]
    method_keys = ["gt", "learned", "laplacian"]
    xgrid, ygrid = np.meshgrid(coordinates, coordinates, indexing="xy")
    stride = 8
    for frame_index, (frame, row) in enumerate(zip(stored, metrics)):
        fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.15), constrained_layout=False)
        fig.subplots_adjust(left=0.035, right=0.925, bottom=0.16, top=0.84, wspace=0.11)
        fig.suptitle(
            rf"2-D connected-metaball flow: full-order NS vs. De Witt Eigenfluid   $t={row['time']:.3f}$",
            y=0.955,
            fontsize=21,
        )
        last_image = None
        for column, (axis, key, title) in enumerate(
            zip(axes, method_keys, panel_titles)
        ):
            omega_grid = grid_values(frame[f"{key}_omega"], valid)
            velocity = frame[f"{key}_velocity"]
            u_grid = grid_values(velocity[:, 0], valid)
            v_grid = grid_values(velocity[:, 1], valid)
            axis.set_facecolor("#e6e6e6")
            last_image = axis.imshow(
                omega_grid,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                vmin=-omega_limit,
                vmax=omega_limit,
                interpolation="bilinear",
                rasterized=True,
            )
            axis.contour(
                xgrid,
                ygrid,
                valid.astype(float),
                levels=[0.5],
                colors=["#222222"],
                linewidths=1.0,
            )
            axis.quiver(
                xgrid[::stride, ::stride],
                ygrid[::stride, ::stride],
                u_grid[::stride, ::stride],
                v_grid[::stride, ::stride],
                color="#202020",
                alpha=0.50,
                width=0.0030,
                headwidth=3.2,
                scale=42.0,
                pivot="mid",
            )
            axis.set_title(title, pad=8)
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel(r"$x$")
            if column == 0:
                axis.set_ylabel(r"$y$")
                caption = (
                    rf"$E={row['gt_kinetic_energy']:.3e}$; "
                    rf"$Z={row['gt_enstrophy']:.3e}$"
                )
            else:
                prefix = key
                caption = (
                    rf"$\varepsilon_u={row[prefix + '_velocity_relative_l2']:.3f}$; "
                    rf"$\varepsilon_\omega={row[prefix + '_vorticity_relative_l2']:.3f}$; "
                    rf"$E/E_{{GT}}={row[prefix + '_kinetic_energy']/row['gt_kinetic_energy']:.3f}$"
                )
            axis.text(
                0.5,
                -0.16,
                caption,
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=14,
            )
        colorbar_axis = fig.add_axes([0.945, 0.215, 0.012, 0.55])
        colorbar = fig.colorbar(last_image, cax=colorbar_axis, orientation="vertical")
        colorbar.set_label(r"Vorticity $\omega$", labelpad=8)
        frame_path = frames_dir / f"frame_{frame_index:03d}.png"
        # Keep an identical even-pixel canvas for every H.264 frame.
        fig.savefig(frame_path, facecolor="white")
        plt.close(fig)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError("ffmpeg is required to encode MP4") from error
    video_path = output_dir / "ns_gt_vs_learned_and_laplacian_eigenfluid_128x128_20frames.mp4"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    subprocess.run(command, check=True)
    return video_path, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-token", type=int, default=8_009_482)
    parser.add_argument("--seed", type=int, default=20_260_825)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    parser.add_argument("--dt-max", type=float, default=1.0e-3)
    parser.add_argument("--cfl", type=float, default=0.28)
    parser.add_argument("--viscosity", type=float, default=2.5e-4)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--fps", type=int, default=5)
    args = parser.parse_args()
    if args.resolution != 128:
        raise ValueError("this protocol is fixed to a 128x128 grid")
    if args.modes != 64:
        raise ValueError("this protocol is fixed to K=64")
    if args.frames != 20:
        raise ValueError("this protocol is fixed to 20 output frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    metadata, cfg, model = load_model(args.checkpoint)
    if int(cfg.modes) != args.modes:
        raise ValueError(f"checkpoint contains K={cfg.modes}, expected K={args.modes}")
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    token = float(args.geometry_token)
    valid = fluid_mask(cfg, all_points, token).reshape(args.resolution, args.resolution)
    points = all_points[valid.ravel()]
    operator = positive_dirichlet_laplacian(valid, spacing)

    basis_started = time.perf_counter()
    learned_psi, _, _, learned_peeling, learned_area = eigenfluid_basis(
        cfg,
        model,
        points,
        token,
        args.peeling_points,
        args.seed + 701,
    )
    learned_psi, learned_values, learned_grid_ritz = matched_grid_ritz(
        learned_psi, valid, spacing
    )
    learned = make_basis_package(
        "learned_original_e2000",
        learned_psi,
        learned_values,
        valid,
        spacing,
    )
    laplacian_values, laplacian_psi, _, _ = finite_difference_laplacian(
        valid, spacing, args.modes
    )
    laplacian = make_basis_package(
        "matched_dirichlet_laplacian",
        laplacian_psi,
        laplacian_values,
        valid,
        spacing,
    )
    learned.structure, learned_structure = precompute_structure_coefficients(
        learned, valid, spacing
    )
    laplacian.structure, laplacian_structure = precompute_structure_coefficients(
        laplacian, valid, spacing
    )
    basis_seconds = time.perf_counter() - basis_started

    gt_parameters = generate_gt_parameters(
        cfg, token, args.seed + 100_003, args.vortices
    )
    initial_psi, _, _, gt_normalization = ground_truth_fields(
        cfg, points, token, gt_parameters
    )
    initial_velocity, _ = matched_grid_observables(initial_psi, valid, spacing)
    discrete_rms = math.sqrt(float(np.mean(np.sum(initial_velocity**2, axis=1))))
    initial_psi = initial_psi.astype(F64) / discrete_rms
    initial_omega = np.asarray(operator @ initial_psi, dtype=F64)
    fom = NavierStokesFOM(valid, spacing, operator, args.viscosity)
    _, initial_velocity = fom.fields(initial_omega)
    learned_state = project_initial_velocity(learned, initial_velocity, spacing)
    laplacian_state = project_initial_velocity(laplacian, initial_velocity, spacing)

    stored: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float | int]] = []
    physical_time = 0.0
    gt_omega = initial_omega.copy()
    internal_steps = 0
    cfl_field_seconds = 0.0
    fom_advance_seconds = 0.0
    learned_advance_seconds = 0.0
    laplacian_advance_seconds = 0.0
    simulation_started = time.perf_counter()
    output_times = np.arange(args.frames, dtype=F64) * args.frame_dt
    for frame_index, target_time in enumerate(output_times):
        while physical_time < target_time - 1.0e-13:
            step_timer = time.perf_counter()
            _, gt_velocity_now = fom.fields(gt_omega)
            cfl_field_seconds += time.perf_counter() - step_timer
            maximum_speed = max(
                float(np.max(np.linalg.norm(gt_velocity_now, axis=1))), 1.0e-12
            )
            dt = min(
                args.dt_max,
                args.cfl * spacing / maximum_speed,
                float(target_time - physical_time),
            )
            step_timer = time.perf_counter()
            gt_omega = fom.advance(gt_omega, dt)
            fom_advance_seconds += time.perf_counter() - step_timer
            step_timer = time.perf_counter()
            learned_state = advance_rom(
                learned, learned_state, dt, args.viscosity
            )
            learned_advance_seconds += time.perf_counter() - step_timer
            step_timer = time.perf_counter()
            laplacian_state = advance_rom(
                laplacian, laplacian_state, dt, args.viscosity
            )
            laplacian_advance_seconds += time.perf_counter() - step_timer
            physical_time += dt
            internal_steps += 1

        gt_psi, gt_velocity = fom.fields(gt_omega)
        learned_psi_frame, learned_velocity, learned_omega = reconstruct_rom(
            learned, learned_state
        )
        laplacian_psi_frame, laplacian_velocity, laplacian_omega = reconstruct_rom(
            laplacian, laplacian_state
        )
        stored.append(
            {
                "time": float(target_time),
                "gt_psi": gt_psi.astype(F32),
                "gt_velocity": gt_velocity.astype(F32),
                "gt_omega": gt_omega.astype(F32),
                "learned_psi": learned_psi_frame.astype(F32),
                "learned_velocity": learned_velocity.astype(F32),
                "learned_omega": learned_omega.astype(F32),
                "laplacian_psi": laplacian_psi_frame.astype(F32),
                "laplacian_velocity": laplacian_velocity.astype(F32),
                "laplacian_omega": laplacian_omega.astype(F32),
            }
        )
        metric_rows.append(
            frame_metrics(
                frame_index,
                float(target_time),
                spacing,
                gt_velocity,
                gt_omega,
                learned_velocity,
                learned_omega,
                laplacian_velocity,
                laplacian_omega,
            )
        )
        print(
            f"frame={frame_index:02d}/{args.frames - 1:02d} t={target_time:.3f} "
            f"learned_eu={metric_rows[-1]['learned_velocity_relative_l2']:.5f} "
            f"lap_eu={metric_rows[-1]['laplacian_velocity_relative_l2']:.5f}",
            flush=True,
        )
    simulation_seconds = time.perf_counter() - simulation_started

    fields_path = args.output_dir / "simulation_fields_20frames.npz"
    np.savez_compressed(
        fields_path,
        times=output_times,
        valid=valid,
        coordinates=coordinates,
        gt_psi=np.stack([frame["gt_psi"] for frame in stored]),
        gt_velocity=np.stack([frame["gt_velocity"] for frame in stored]),
        gt_omega=np.stack([frame["gt_omega"] for frame in stored]),
        learned_psi=np.stack([frame["learned_psi"] for frame in stored]),
        learned_velocity=np.stack([frame["learned_velocity"] for frame in stored]),
        learned_omega=np.stack([frame["learned_omega"] for frame in stored]),
        laplacian_psi=np.stack([frame["laplacian_psi"] for frame in stored]),
        laplacian_velocity=np.stack([frame["laplacian_velocity"] for frame in stored]),
        laplacian_omega=np.stack([frame["laplacian_omega"] for frame in stored]),
    )
    metrics_path = args.output_dir / "frame_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    video_path, render_seconds = render_frames(
        args.output_dir, coordinates, valid, stored, metric_rows, args.fps
    )
    total_seconds = time.perf_counter() - total_started
    summary = {
        "status": "complete",
        "method": {
            "reference": "De Witt, Lessig & Fiume, ACM TOG 31(1), 2012",
            "equation": "dw_k/dt = w^T C_k w - nu*mu_k*w_k + f_k",
            "positive_laplacian_convention": "mu_k = eig(-Delta) > 0",
            "force": "f_k = 0",
            "structure_coefficients": "vorticity-advection Galerkin projection",
            "rom_integrator": "RK4 inviscid + kinetic-energy renormalization + exact exponential viscosity",
            "fom": "vorticity-streamfunction NS; five-point Dirichlet -Delta; skew-centered advection; RK4",
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256(args.checkpoint),
            "epoch": int(metadata["epoch"]),
        },
        "configuration": {
            "resolution": args.resolution,
            "valid_nodes": int(np.count_nonzero(valid)),
            "modes": args.modes,
            "frames": args.frames,
            "frame_dt": args.frame_dt,
            "final_time": float(output_times[-1]),
            "dt_max": args.dt_max,
            "cfl": args.cfl,
            "viscosity": args.viscosity,
            "geometry_token": args.geometry_token,
            "geometry": geometry_from_token(cfg, token).tolist(),
            "seed": args.seed,
            "vortices": args.vortices,
            "fps": args.fps,
            "peeling_points": args.peeling_points,
        },
        "initial_condition": {
            "parameters": serializable_gt(gt_parameters),
            "analytic_velocity_normalization": gt_normalization,
            "discrete_velocity_rms_before_final_normalization": discrete_rms,
        },
        "basis": {
            "learned": {
                **learned.diagnostics,
                **learned_structure,
                "peeling": learned_peeling,
                "grid_ritz": learned_grid_ritz,
                "area": learned_area,
            },
            "laplacian": {**laplacian.diagnostics, **laplacian_structure},
        },
        "timing_seconds": {
            "basis_and_structure_precompute": basis_seconds,
            "time_integration": simulation_seconds,
            "fom_cfl_velocity_solves": cfl_field_seconds,
            "fom_rk4_advance": fom_advance_seconds,
            "learned_rom_rk4_advance": learned_advance_seconds,
            "laplacian_rom_rk4_advance": laplacian_advance_seconds,
            "render_and_encode": render_seconds,
            "total": total_seconds,
            "internal_steps": internal_steps,
        },
        "frame_metrics": metric_rows,
        "aggregate_metrics": {
            "learned_velocity_relative_l2_mean": float(
                np.mean([row["learned_velocity_relative_l2"] for row in metric_rows])
            ),
            "learned_velocity_relative_l2_final": float(
                metric_rows[-1]["learned_velocity_relative_l2"]
            ),
            "learned_vorticity_relative_l2_mean": float(
                np.mean([row["learned_vorticity_relative_l2"] for row in metric_rows])
            ),
            "learned_vorticity_relative_l2_final": float(
                metric_rows[-1]["learned_vorticity_relative_l2"]
            ),
            "laplacian_velocity_relative_l2_mean": float(
                np.mean([row["laplacian_velocity_relative_l2"] for row in metric_rows])
            ),
            "laplacian_velocity_relative_l2_final": float(
                metric_rows[-1]["laplacian_velocity_relative_l2"]
            ),
            "laplacian_vorticity_relative_l2_mean": float(
                np.mean([row["laplacian_vorticity_relative_l2"] for row in metric_rows])
            ),
            "laplacian_vorticity_relative_l2_final": float(
                metric_rows[-1]["laplacian_vorticity_relative_l2"]
            ),
        },
        "outputs": {
            "video": str(video_path.resolve()),
            "frames": str((args.output_dir / "frames").resolve()),
            "fields": str(fields_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
    }
    summary_path = args.output_dir / "simulation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate_metrics"], indent=2), flush=True)
    print(f"video={video_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
