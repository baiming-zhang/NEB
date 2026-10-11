





from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from _neb_paths import Path

import numpy as np
import torch
from scipy import ndimage

from .training import (
    Config,
    airfoil_level_numpy,
    inside_airfoil,
    trailing_edge,
)


@dataclass
class LBMSolverConfig:
    nx: int = 1001
    ny: int = 801
    reynolds: float = 1500.0
    lattice_inlet_speed: float = 0.08
    steps: int = 60000
    warmup_steps: int = 40000
    snapshot_every: int = 500
    collision_model: str = "trt"
    trt_magic_parameter: float = 3.0 / 16.0
    initial_perturbation_fraction: float = 0.01
    warmup_inlet_perturbation_fraction: float = 0.02
    warmup_inlet_perturbation_duration: float = 0.5
    warmup_inlet_perturbation_period_chords: float = 2.0
    require_separated_gt: bool = True
    device: str = "cuda"


CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
WEIGHTS = (4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36)
OPPOSITE = (0, 3, 4, 1, 2, 7, 8, 5, 6)


def equilibrium(
    rho: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    result = []
    speed2 = ux.square() + uy.square()
    for cx, cy, weight in zip(CX, CY, WEIGHTS):
        cu = 3.0 * (cx * ux + cy * uy)
        result.append(weight * rho * (1.0 + cu + 0.5 * cu.square() - 1.5 * speed2))
    return torch.stack(result)


def macroscopic(distributions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rho = distributions.sum(dim=0)
    safe_rho = torch.clamp(rho, min=1.0e-8)
    ux = sum(
        cx * distributions[index] for index, cx in enumerate(CX)
    ) / safe_rho
    uy = sum(
        cy * distributions[index] for index, cy in enumerate(CY)
    ) / safe_rho
    return rho, ux, uy


def separation_diagnostics(
    geometry: Config,
    beta: float,
    alpha_deg: float,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    fluid: np.ndarray,
    u: np.ndarray,
    vorticity: np.ndarray,
    inlet_speed: float,
) -> dict:

    points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    clearance = airfoil_level_numpy(
        geometry, points, beta, alpha_deg
    ).reshape(xgrid.shape)
    near_wall = fluid & (clearance > 0.0) & (
        clearance < 0.08 * geometry.chord
    )
    trailing_x, trailing_y = trailing_edge(geometry, alpha_deg)
    wake = (
        fluid
        & (xgrid > trailing_x)
        & (xgrid < trailing_x + 2.0 * geometry.chord)
        & (np.abs(ygrid - trailing_y) < 0.65 * geometry.chord)
    )
    frame_rows = []
    for frame in range(len(u)):
        wake_omega = vorticity[frame][wake]
        absolute_limit = max(
            float(np.percentile(np.abs(wake_omega), 92.0)), 1.0e-8
        )
        positive, positive_count = ndimage.label(
            wake
            & (vorticity[frame] > absolute_limit)
        )
        negative, negative_count = ndimage.label(
            wake
            & (vorticity[frame] < -absolute_limit)
        )

        def coherent_count(labels: np.ndarray, count: int) -> int:
            if count == 0:
                return 0
            sizes = np.bincount(labels.ravel())[1:]
            return int(np.sum(sizes >= 6))

        reverse = wake & (u[frame] < -0.01 * inlet_speed)
        frame_rows.append(
            {
                "near_wall_vorticity_p99": float(
                    np.percentile(
                        np.abs(vorticity[frame][near_wall]), 99.0
                    )
                ),
                "wake_vorticity_rms": float(
                    np.sqrt(np.mean(wake_omega**2))
                ),
                "wake_reverse_flow_fraction": float(
                    np.sum(reverse) / max(np.sum(wake), 1)
                ),
                "minimum_wake_u": float(np.min(u[frame][wake])),
                "positive_coherent_vortices": coherent_count(
                    positive, positive_count
                ),
                "negative_coherent_vortices": coherent_count(
                    negative, negative_count
                ),
            }
        )
    wake_history = vorticity[:, wake]
    temporal_wake_rms = float(
        np.sqrt(np.mean((wake_history - wake_history.mean(axis=0)) ** 2))
    )
    maximum_reverse = float(
        max(row["wake_reverse_flow_fraction"] for row in frame_rows)
    )
    maximum_vortices = int(
        max(
            row["positive_coherent_vortices"]
            + row["negative_coherent_vortices"]
            for row in frame_rows
        )
    )
    separation_expected = abs(alpha_deg) >= 8.0
    separation_quality_pass = (
        not separation_expected
        or (
            maximum_reverse >= 1.0e-3
            and maximum_vortices >= 3
            and temporal_wake_rms >= 2.0e-3
        )
    )
    return {
        "definition": {
            "near_wall_band_chords": 0.08,
            "wake_length_chords": 2.0,
            "wake_half_height_chords": 0.65,
            "reverse_flow_threshold_fraction": -0.01,
            "coherent_vortex_min_nodes": 6,
        },
        "frames": frame_rows,
        "temporal_wake_vorticity_rms": temporal_wake_rms,
        "maximum_reverse_flow_fraction": maximum_reverse,
        "maximum_coherent_vortex_count": maximum_vortices,
        "separation_expected": separation_expected,
        "separated_gt_quality_pass": separation_quality_pass,
        "separated_gt_quality_rule": (
            "|alpha|>=8 deg requires reverse fraction >=1e-3, "
            "at least 3 coherent wake vortices, and temporal wake "
            "vorticity RMS >=2e-3"
        ),
    }


def solve_reference(
    geometry: Config,
    solver: LBMSolverConfig,
    beta: float,
    alpha_deg: float,
    output: Path,
) -> Path:

    requested = torch.device(solver.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    x = np.linspace(geometry.xmin, geometry.xmax, solver.nx, dtype=np.float64)
    y = np.linspace(geometry.ymin, geometry.ymax, solver.ny, dtype=np.float64)
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    airfoil = inside_airfoil(geometry, points, beta, alpha_deg).reshape(
        solver.ny, solver.nx
    )
    solid_numpy = airfoil.copy()
    solid = torch.as_tensor(solid_numpy, device=requested)

    yn = torch.linspace(-1.0, 1.0, solver.ny, device=requested).reshape(-1, 1)
    inlet_profile = torch.full_like(yn, solver.lattice_inlet_speed)
    ux = inlet_profile.expand(solver.ny, solver.nx).clone()
    x_tensor = torch.as_tensor(xgrid, device=requested, dtype=ux.dtype)
    y_tensor = torch.as_tensor(ygrid, device=requested, dtype=ux.dtype)
    trailing_x, trailing_y = trailing_edge(geometry, alpha_deg)
    perturbation = (
        solver.initial_perturbation_fraction
        * solver.lattice_inlet_speed
        * torch.sin(
            2.0
            * torch.pi
            * (x_tensor - trailing_x)
            / (1.5 * geometry.chord)
        )
        * torch.exp(
            -(
                ((x_tensor - trailing_x) / (1.5 * geometry.chord)) ** 2
                + ((y_tensor - trailing_y) / (0.7 * geometry.chord)) ** 2
            )
        )
    )
    uy = perturbation
    ux[solid] = 0.0
    rho = torch.ones_like(ux)
    distributions = equilibrium(rho, ux, uy)

    chord_nodes = geometry.chord / (
        (geometry.xmax - geometry.xmin) / (solver.nx - 1)
    )
    viscosity = solver.lattice_inlet_speed * chord_nodes / solver.reynolds
    relaxation_time = 0.5 + 3.0 * viscosity
    if relaxation_time <= 0.5005:
        raise ValueError(
            f"unstable relaxation time {relaxation_time:.6f}; "
            "increase resolution/speed or lower Reynolds"
        )
    omega = 1.0 / relaxation_time
    if solver.collision_model == "trt":
        odd_relaxation_time = 0.5 + solver.trt_magic_parameter / (
            relaxation_time - 0.5
        )
        odd_omega = 1.0 / odd_relaxation_time
    elif solver.collision_model == "bgk":
        odd_relaxation_time = relaxation_time
        odd_omega = omega
    else:
        raise ValueError(
            f"collision_model must be 'trt' or 'bgk', got {solver.collision_model}"
        )
    snapshots_u: list[np.ndarray] = []
    snapshots_v: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    started = time.perf_counter()

    for step in range(1, solver.steps + 1):
        rho, ux, uy = macroscopic(distributions)
        ux[solid], uy[solid] = 0.0, 0.0
        equilibrium_fields = equilibrium(rho, ux, uy)
        if solver.collision_model == "trt":
            opposite_distributions = distributions[list(OPPOSITE)]
            opposite_equilibrium = equilibrium_fields[list(OPPOSITE)]
            even = 0.5 * (distributions + opposite_distributions)
            odd = 0.5 * (distributions - opposite_distributions)
            equilibrium_even = 0.5 * (
                equilibrium_fields + opposite_equilibrium
            )
            equilibrium_odd = 0.5 * (
                equilibrium_fields - opposite_equilibrium
            )
            post_collision = (
                distributions
                - omega * (even - equilibrium_even)
                - odd_omega * (odd - equilibrium_odd)
            )
        else:
            post_collision = distributions - omega * (
                distributions - equilibrium_fields
            )
        streamed = torch.empty_like(post_collision)
        for direction, (cx, cy) in enumerate(zip(CX, CY)):
            streamed[direction] = torch.roll(
                post_collision[direction], shifts=(cy, cx), dims=(0, 1)
            )



        for direction, (cx, cy) in enumerate(zip(CX, CY)):
            if direction == 0:
                continue
            destination_is_solid = torch.roll(
                solid, shifts=(-cy, -cx), dims=(0, 1)
            )
            boundary_fluid = destination_is_solid & ~solid
            streamed[OPPOSITE[direction], boundary_fluid] = post_collision[
                direction, boundary_fluid
            ]
        streamed[:, solid] = equilibrium(
            torch.ones_like(rho), torch.zeros_like(ux), torch.zeros_like(uy)
        )[:, solid]

        inlet_rho = torch.ones(solver.ny - 2, device=requested)
        inlet_u = inlet_profile[1:-1, 0]
        perturbation_stop = int(
            solver.warmup_steps
            * solver.warmup_inlet_perturbation_duration
        )
        if (
            step <= perturbation_stop
            and solver.warmup_inlet_perturbation_fraction > 0.0
        ):
            period_steps = max(
                1.0,
                solver.warmup_inlet_perturbation_period_chords
                * chord_nodes
                / solver.lattice_inlet_speed,
            )
            eta = 0.5 * (yn[1:-1, 0] + 1.0)
            inlet_v = (
                solver.warmup_inlet_perturbation_fraction
                * solver.lattice_inlet_speed
                * math.sin(2.0 * math.pi * step / period_steps)
                * torch.sin(2.0 * torch.pi * eta)
            )
        else:
            inlet_v = torch.zeros_like(inlet_u)
        streamed[:, 1:-1, 0] = equilibrium(
            inlet_rho, inlet_u, inlet_v
        )
        streamed[:, 1:-1, -1] = streamed[:, 1:-1, -2]
        farfield_rho = torch.ones(solver.nx, device=requested)
        farfield_u = torch.full(
            (solver.nx,), solver.lattice_inlet_speed, device=requested
        )
        farfield_v = torch.zeros_like(farfield_u)
        farfield_equilibrium = equilibrium(
            farfield_rho, farfield_u, farfield_v
        )
        streamed[:, 0, :] = farfield_equilibrium
        streamed[:, -1, :] = farfield_equilibrium
        distributions = streamed

        if step >= solver.warmup_steps and (
            (step - solver.warmup_steps) % solver.snapshot_every == 0
            or step == solver.steps
        ):
            _, current_u, current_v = macroscopic(distributions)
            current_u[solid], current_v[solid] = 0.0, 0.0
            snapshots_u.append(current_u.detach().cpu().numpy().astype(np.float32))
            snapshots_v.append(current_v.detach().cpu().numpy().astype(np.float32))
            snapshot_steps.append(step)
        if not torch.isfinite(distributions).all():
            raise RuntimeError(f"LBM became non-finite at step {step}")

    u = np.stack(snapshots_u)
    v = np.stack(snapshots_v)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    vorticity = np.gradient(v, dx, axis=2) - np.gradient(u, dy, axis=1)
    diagnostics = separation_diagnostics(
        geometry,
        beta,
        alpha_deg,
        xgrid,
        ygrid,
        ~solid_numpy,
        u,
        vorticity,
        solver.lattice_inlet_speed,
    )
    if solver.require_separated_gt and not diagnostics[
        "separated_gt_quality_pass"
    ]:
        raise RuntimeError(
            "reference remained monotone/under-resolved and failed separated "
            "GT quality checks; increase resolution/run time or adjust the "
            f"physical case. Diagnostics: {json.dumps(diagnostics)}"
        )
    metadata = {
        "solver": (
            f"D2Q9 {solver.collision_model.upper()} lattice Boltzmann "
            "Navier-Stokes"
        ),
        "purpose": "independent post-freeze reference only",
        "beta": beta,
        "alpha_deg": alpha_deg,
        "geometry": asdict(geometry),
        "solver_config": asdict(solver),
        "actual_device": str(requested),
        "chord_nodes": chord_nodes,
        "lattice_viscosity": viscosity,
        "relaxation_time": relaxation_time,
        "odd_relaxation_time": odd_relaxation_time,
        "elapsed_seconds": time.perf_counter() - started,
        "airflow": "uniform left-to-right atmospheric free stream",
        "boundary": (
            "no-slip airfoil; uniform free-stream top/bottom far field; "
            "open zero-gradient outlet"
        ),
        "initial_disturbance": (
            "deterministic localized cross-stream perturbation, "
            f"{solver.initial_perturbation_fraction:.3%} of inlet speed; "
            "initial condition only"
        ),
        "warmup_inlet_disturbance": (
            f"{solver.warmup_inlet_perturbation_fraction:.3%} cross-stream "
            f"gust for the first {solver.warmup_inlet_perturbation_duration:.1%} "
            "of warm-up only; zero during every saved GT frame"
        ),
        "separation_diagnostics": diagnostics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        steps=np.asarray(snapshot_steps, dtype=np.int64),
        u=u,
        v=v,
        vorticity=vorticity.astype(np.float32),
        airfoil=airfoil,
        solid=solid_numpy,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return output
