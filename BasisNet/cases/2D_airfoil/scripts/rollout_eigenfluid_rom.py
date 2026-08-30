"""Compare a coefficient-space Eigenfluid rollout with a TRT-LBM rollout.

Both trajectories start from the same velocity-optimal K-mode reconstruction.
The Navier--Stokes reference is advanced with the case's D2Q9 TRT solver.  The
ROM uses a dense-grid variational Galerkin contraction, retains the open
outlet total-pressure flux, and advances coefficients with an implicit
trapezoidal update plus exact modal viscous decay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from cupyx.scipy.sparse.linalg import LinearOperator, gmres
from scipy.interpolate import RegularGridInterpolator


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CASE_ROOT.parents[2]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402
from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    generalized_symmetric_eig,
    load_hardcoded_model,
    raw_fields,
    ritz_fluid_points,
)
from eigenfluid_sss.basisnet.airfoil2d.ns_reference import (  # noqa: E402
    CX,
    CY,
    OPPOSITE,
    equilibrium,
    macroscopic,
)
from eigenfluid_sss.basisnet.airfoil2d.rom_dynamics import (  # noqa: E402
    OpenBoundaryTerms,
    variational_advection_jvp,
    variational_advection_rhs,
)
from run_continuous_postprocess import encode_mp4  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return np.gradient(v, dx, axis=-1) - np.gradient(u, dy, axis=-2)


def relative_vector_error(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    predicted_u: np.ndarray,
    predicted_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    numerator = np.sum(
        (predicted_u[mask] - actual_u[mask]) ** 2
        + (predicted_v[mask] - actual_v[mask]) ** 2,
        dtype=np.float64,
    )
    denominator = np.sum(
        actual_u[mask] ** 2 + actual_v[mask] ** 2,
        dtype=np.float64,
    )
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def relative_scalar_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> float:
    numerator = np.sum((predicted[mask] - actual[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(actual[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


@torch.no_grad()
def rollout_trt_lbm(
    initial_u: np.ndarray,
    initial_v: np.ndarray,
    solid_numpy: np.ndarray,
    frames: int,
    steps_per_frame: int,
    inlet_speed: float,
    relaxation_time: float,
    odd_relaxation_time: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    solid = torch.as_tensor(solid_numpy, device=requested)
    ux = torch.as_tensor(initial_u, device=requested, dtype=torch.float32).clone()
    uy = torch.as_tensor(initial_v, device=requested, dtype=torch.float32).clone()
    ux[solid], uy[solid] = 0.0, 0.0
    rho = torch.ones_like(ux)
    distributions = equilibrium(rho, ux, uy)
    omega_even = 1.0 / relaxation_time
    omega_odd = 1.0 / odd_relaxation_time
    ny, nx = initial_u.shape
    inlet_rho = torch.ones(ny - 2, device=requested)
    inlet_u = torch.full((ny - 2,), inlet_speed, device=requested)
    inlet_v = torch.zeros_like(inlet_u)
    farfield_rho = torch.ones(nx, device=requested)
    farfield_u = torch.full((nx,), inlet_speed, device=requested)
    farfield_v = torch.zeros_like(farfield_u)
    farfield_equilibrium = equilibrium(farfield_rho, farfield_u, farfield_v)
    solid_equilibrium = equilibrium(
        torch.ones_like(ux), torch.zeros_like(ux), torch.zeros_like(uy)
    )
    destination_masks = []
    for direction, (cx, cy) in enumerate(zip(CX, CY)):
        if direction == 0:
            destination_masks.append(None)
        else:
            destination_masks.append(
                torch.roll(solid, shifts=(-cy, -cx), dims=(0, 1)) & ~solid
            )

    snapshots_u = np.empty((frames, ny, nx), dtype=np.float32)
    snapshots_v = np.empty_like(snapshots_u)
    started = time.perf_counter()
    total_steps = frames * steps_per_frame
    for step in range(1, total_steps + 1):
        rho, ux, uy = macroscopic(distributions)
        ux[solid], uy[solid] = 0.0, 0.0
        equilibrium_fields = equilibrium(rho, ux, uy)
        opposite_distributions = distributions[list(OPPOSITE)]
        opposite_equilibrium = equilibrium_fields[list(OPPOSITE)]
        even = 0.5 * (distributions + opposite_distributions)
        odd = 0.5 * (distributions - opposite_distributions)
        equilibrium_even = 0.5 * (equilibrium_fields + opposite_equilibrium)
        equilibrium_odd = 0.5 * (equilibrium_fields - opposite_equilibrium)
        post_collision = (
            distributions
            - omega_even * (even - equilibrium_even)
            - omega_odd * (odd - equilibrium_odd)
        )
        streamed = torch.empty_like(post_collision)
        for direction, (cx, cy) in enumerate(zip(CX, CY)):
            streamed[direction] = torch.roll(
                post_collision[direction], shifts=(cy, cx), dims=(0, 1)
            )
        for direction in range(1, len(CX)):
            boundary_fluid = destination_masks[direction]
            streamed[OPPOSITE[direction], boundary_fluid] = post_collision[
                direction, boundary_fluid
            ]
        streamed[:, solid] = solid_equilibrium[:, solid]
        streamed[:, 1:-1, 0] = equilibrium(inlet_rho, inlet_u, inlet_v)
        streamed[:, 1:-1, -1] = streamed[:, 1:-1, -2]
        streamed[:, 0, :] = farfield_equilibrium
        streamed[:, -1, :] = farfield_equilibrium
        distributions = streamed

        if step % steps_per_frame == 0:
            frame = step // steps_per_frame - 1
            _, current_u, current_v = macroscopic(distributions)
            current_u[solid], current_v[solid] = 0.0, 0.0
            snapshots_u[frame] = current_u.cpu().numpy()
            snapshots_v[frame] = current_v.cpu().numpy()
            print(
                f"[NS {frame + 1:02d}/{frames}] local_step={step}",
                flush=True,
            )
        if step % 100 == 0 and not bool(torch.isfinite(distributions).all()):
            raise RuntimeError(f"TRT-LBM became non-finite at local step {step}")
    return snapshots_u, snapshots_v, time.perf_counter() - started


def interpolate_grid_fields(
    x: np.ndarray,
    y: np.ndarray,
    fields: tuple[np.ndarray, ...],
    points: np.ndarray,
) -> tuple[np.ndarray, ...]:
    result = []
    query = np.column_stack((points[:, 1], points[:, 0]))
    for field in fields:
        interpolator = RegularGridInterpolator(
            (y, x), field, method="linear", bounds_error=True
        )
        result.append(interpolator(query).astype(np.float32))
    return tuple(result)


def full_grid_quadrature(
    gt_source: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return tensor-product trapezoidal weights on every fluid grid node."""

    x = gt_source["x"].astype(np.float64)
    y = gt_source["y"].astype(np.float64)
    solid = gt_source["solid"].astype(bool)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("the full-grid ROM quadrature needs at least a 2x2 grid")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if not np.allclose(np.diff(x), dx, rtol=5.0e-5, atol=5.0e-7) or not np.allclose(
        np.diff(y), dy, rtol=5.0e-5, atol=5.0e-7
    ):
        raise ValueError("the full-grid ROM quadrature requires a uniform grid")

    x_weights = np.ones(len(x), dtype=np.float64)
    y_weights = np.ones(len(y), dtype=np.float64)
    x_weights[[0, -1]] = 0.5
    y_weights[[0, -1]] = 0.5
    grid_weights = dy * dx * np.multiply.outer(y_weights, x_weights)
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    weights = grid_weights[fluid].astype(np.float32)

    fluid_flat = np.flatnonzero(fluid.ravel())
    outlet_rows = np.flatnonzero(fluid[:, -1])
    outlet_flat = outlet_rows * len(x) + (len(x) - 1)
    outlet_indices = np.searchsorted(fluid_flat, outlet_flat).astype(np.int64)
    if not np.array_equal(fluid_flat[outlet_indices], outlet_flat):
        raise RuntimeError("failed to map outlet nodes into the fluid-point ordering")
    outlet_weights = (dy * y_weights[outlet_rows]).astype(np.float32)
    outlet_points = np.column_stack(
        (np.full(len(outlet_rows), x[-1]), y[outlet_rows])
    ).astype(np.float32)
    return {
        "points": points,
        "weights": weights,
        "fluid": fluid,
        "outlet_indices": outlet_indices,
        "outlet_points": outlet_points,
        "outlet_rows": outlet_rows.astype(np.int64),
        "outlet_weights": outlet_weights,
    }


def accumulate_raw_operators(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    weights: np.ndarray,
    beta: float,
    alpha: float,
    field_batch: int,
    grid_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate full-grid raw mass and stiffness without storing raw fields."""

    raw_mass = np.zeros((cfg.modes, cfg.modes), dtype=np.float64)
    raw_stiffness = np.zeros_like(raw_mass)
    for start in range(0, len(points), grid_chunk):
        stop = min(start + grid_chunk, len(points))
        raw_u, raw_v, raw_omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha, field_batch
        )
        gpu_weight = cp.asarray(weights[start:stop])[:, None]
        if not (
            np.isfinite(raw_u).all()
            and np.isfinite(raw_v).all()
            and np.isfinite(raw_omega).all()
        ):
            raise FloatingPointError(
                f"non-finite raw basis field on full-grid chunk {start}:{stop}"
            )
        # Form the dense Gram operators in genuine FP64.  The archived MLP
        # output remains FP32/FP16-protocol data, but its quadrature sum must
        # not lose the smallest K1024 mass directions to FP32 accumulation.
        gpu_u = cp.asarray(raw_u, dtype=cp.float64)
        gpu_v = cp.asarray(raw_v, dtype=cp.float64)
        gpu_omega = cp.asarray(raw_omega, dtype=cp.float64)
        mass_chunk = gpu_u.T @ (gpu_weight * gpu_u)
        mass_chunk += gpu_v.T @ (gpu_weight * gpu_v)
        stiffness_chunk = gpu_omega.T @ (gpu_weight * gpu_omega)
        raw_mass += cp.asnumpy(mass_chunk).astype(np.float64)
        raw_stiffness += cp.asnumpy(stiffness_chunk).astype(np.float64)
        del gpu_weight, gpu_u, gpu_v, gpu_omega, mass_chunk, stiffness_chunk
        print(
            f"[ROM full-grid operators] {stop}/{len(points)} fluid nodes",
            flush=True,
        )
    cp.get_default_memory_pool().free_all_blocks()
    return raw_mass, raw_stiffness


def evaluate_full_grid_ritz_basis(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    weights: np.ndarray,
    rotation: np.ndarray,
    beta: float,
    alpha: float,
    field_batch: int,
    grid_chunk: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, float]:
    """Evaluate and verify the Ritz basis under the same dense-grid metric."""

    shape = (len(points), cfg.modes)
    basis_bytes = 3 * shape[0] * shape[1] * np.dtype(np.float32).itemsize
    free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
    reserve_bytes = 2 * 1024**3
    if basis_bytes + reserve_bytes > free_bytes:
        raise RuntimeError(
            "full-grid K-mode basis needs approximately "
            f"{basis_bytes / 1024**3:.2f} GiB plus a 2 GiB work reserve, but "
            f"only {free_bytes / 1024**3:.2f} GiB is free; another GPU job "
            "is probably active"
        )
    host_u = np.empty(shape, dtype=np.float32)
    host_v = np.empty(shape, dtype=np.float32)
    host_omega = np.empty(shape, dtype=np.float32)
    mass_check = np.zeros((cfg.modes, cfg.modes), dtype=np.float64)
    gpu_rotation = cp.asarray(rotation.astype(np.float32))
    for start in range(0, len(points), grid_chunk):
        stop = min(start + grid_chunk, len(points))
        raw_u, raw_v, raw_omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha, field_batch
        )
        basis_u = cp.asarray(raw_u) @ gpu_rotation
        basis_v = cp.asarray(raw_v) @ gpu_rotation
        basis_omega = cp.asarray(raw_omega) @ gpu_rotation
        host_u[start:stop] = cp.asnumpy(basis_u)
        host_v[start:stop] = cp.asnumpy(basis_v)
        host_omega[start:stop] = cp.asnumpy(basis_omega)
        gpu_weight = cp.asarray(weights[start:stop])[:, None]
        mass_chunk = basis_u.T @ (gpu_weight * basis_u)
        mass_chunk += basis_v.T @ (gpu_weight * basis_v)
        mass_check += cp.asnumpy(mass_chunk).astype(np.float64)
        del basis_u, basis_v, basis_omega, gpu_weight, mass_chunk
        print(
            f"[ROM full-grid Ritz basis] {stop}/{len(points)} fluid nodes",
            flush=True,
        )
    del gpu_rotation
    cp.get_default_memory_pool().free_all_blocks()
    orth_error = float(np.max(np.abs(mass_check - np.eye(cfg.modes))))
    gpu_u = cp.asarray(host_u)
    gpu_v = cp.asarray(host_v)
    gpu_omega = cp.asarray(host_omega)
    del host_u, host_v, host_omega, mass_check
    return gpu_u, gpu_v, gpu_omega, orth_error


class ImplicitTrapezoidalROM:
    def __init__(
        self,
        basis_u: cp.ndarray,
        basis_v: cp.ndarray,
        basis_omega: cp.ndarray,
        mean_u: cp.ndarray,
        mean_v: cp.ndarray,
        mean_omega: cp.ndarray,
        eigenvalues: cp.ndarray,
        quadrature_weight: Any,
        viscosity: float,
        newton_tolerance: float,
        gmres_tolerance: float,
        open_boundary: OpenBoundaryTerms | None = None,
    ) -> None:
        self.u = basis_u
        self.v = basis_v
        self.omega = basis_omega
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.mean_omega = mean_omega
        self.eigenvalues = eigenvalues
        self.weight = quadrature_weight
        self.viscosity = viscosity
        self.newton_tolerance = newton_tolerance
        self.gmres_tolerance = gmres_tolerance
        self.open_boundary = open_boundary
        self.linear_iterations = 0
        self.newton_iterations = 0
        self.step_splits = 0

    def rhs(self, coefficients: cp.ndarray) -> cp.ndarray:
        return variational_advection_rhs(
            cp,
            self.u,
            self.v,
            self.omega,
            coefficients,
            self.weight,
            self.mean_u,
            self.mean_v,
            self.mean_omega,
            self.open_boundary,
        )

    def jvp(self, coefficients: cp.ndarray, direction: cp.ndarray) -> cp.ndarray:
        return variational_advection_jvp(
            cp,
            self.u,
            self.v,
            self.omega,
            coefficients,
            direction,
            self.weight,
            self.mean_u,
            self.mean_v,
            self.mean_omega,
            self.open_boundary,
        )

    def trapezoidal_advection(self, start: cp.ndarray, dt: float) -> cp.ndarray:
        rhs_start = self.rhs(start)
        candidate = start + dt * rhs_start
        dimension = len(start)
        for _ in range(6):
            rhs_candidate = self.rhs(candidate)
            residual = candidate - start - 0.5 * dt * (rhs_start + rhs_candidate)
            residual_norm = float(cp.linalg.norm(residual).get())
            scale = 1.0 + float(cp.linalg.norm(candidate).get())
            if residual_norm <= self.newton_tolerance * scale:
                return candidate

            def matvec(direction: cp.ndarray) -> cp.ndarray:
                return direction - 0.5 * dt * self.jvp(candidate, direction)

            operator = LinearOperator(
                (dimension, dimension), matvec=matvec, dtype=cp.float32
            )
            iterations = [0]

            def callback(_value: Any) -> None:
                iterations[0] += 1

            correction, info = gmres(
                operator,
                -residual,
                rtol=self.gmres_tolerance,
                atol=0.0,
                restart=20,
                maxiter=8,
                callback=callback,
                callback_type="pr_norm",
            )
            self.linear_iterations += iterations[0]
            self.newton_iterations += 1
            if info != 0 or not bool(cp.isfinite(correction).all()):
                raise RuntimeError(f"GMRES failed with info={info}")
            candidate = candidate + correction
        raise RuntimeError("implicit trapezoidal Newton iteration did not converge")

    def step(self, coefficients: cp.ndarray, dt: float, depth: int = 0) -> cp.ndarray:
        half_decay = cp.exp(-0.5 * self.viscosity * self.eigenvalues * dt)
        try:
            value = coefficients * half_decay
            value = self.trapezoidal_advection(value, dt)
            value = value * half_decay
            if not bool(cp.isfinite(value).all()):
                raise RuntimeError("ROM coefficients became non-finite")
            return value
        except RuntimeError:
            if depth >= 5:
                raise
            self.step_splits += 1
            midpoint = self.step(coefficients, 0.5 * dt, depth + 1)
            return self.step(midpoint, 0.5 * dt, depth + 1)


def build_rom_and_rollout(
    checkpoint: Path,
    sample: dict[str, np.ndarray],
    gt_source: dict[str, np.ndarray],
    gt_metadata: dict[str, Any],
    frames: int,
    steps_per_frame: int,
    quadrature: int,
    field_batch: int,
    substeps_per_frame: int,
    integration: str,
    grid_chunk: int,
    outlet_static_pressure: float,
    grid_orthogonality_tolerance: float,
    zero_viscosity: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any, Any]:
    cfg, model, _ = load_hardcoded_model(checkpoint)
    beta = float(sample["beta"])
    alpha = float(sample["alpha_deg"])
    mean_u_grid = gt_source["u"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_v_grid = gt_source["v"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_omega_grid = (
        gt_source["vorticity"].mean(axis=0, dtype=np.float64).astype(np.float32)
    )

    grid_data: dict[str, np.ndarray] | None = None
    quadrature_seed: int | None = None
    if integration == "grid":
        grid_data = full_grid_quadrature(gt_source)
        points = grid_data["points"]
        weights = grid_data["weights"]
        estimated_basis_bytes = (
            3 * len(points) * cfg.modes * np.dtype(np.float32).itemsize
        )
        free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
        if estimated_basis_bytes + 2 * 1024**3 > free_bytes:
            raise RuntimeError(
                "full-grid ROM requires the GPU to be exclusive: "
                f"estimated basis {estimated_basis_bytes / 1024**3:.2f} GiB, "
                f"currently free {free_bytes / 1024**3:.2f} GiB"
            )
        print(
            f"[ROM basis] full {gt_source['solid'].shape[1]}x"
            f"{gt_source['solid'].shape[0]} grid: {len(points)} fluid nodes, "
            f"{cfg.modes} modes",
            flush=True,
        )
        raw_mass, raw_stiffness = accumulate_raw_operators(
            model,
            cfg,
            points,
            weights,
            beta,
            alpha,
            field_batch,
            grid_chunk,
        )
    elif integration == "sobol":
        quadrature_seed = int(cfg.seed + 188_200)
        points = ritz_fluid_points(
            cfg, quadrature, beta, alpha, quadrature_seed
        ).astype(np.float32)
        print(
            f"[ROM basis] Sobol quadrature: {quadrature} points, {cfg.modes} modes",
            flush=True,
        )
        raw_u, raw_v, raw_omega = raw_fields(
            model, cfg, points, beta, alpha, field_batch
        )
        from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import airfoil_area

        area = float(
            (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(cfg, beta)
        )
        weight = area / quadrature
        gpu_raw_u = cp.asarray(raw_u)
        gpu_raw_v = cp.asarray(raw_v)
        gpu_raw_omega = cp.asarray(raw_omega)
        raw_mass = cp.asnumpy(
            weight * (gpu_raw_u.T @ gpu_raw_u + gpu_raw_v.T @ gpu_raw_v)
        ).astype(np.float64)
        raw_stiffness = cp.asnumpy(weight * (gpu_raw_omega.T @ gpu_raw_omega)).astype(
            np.float64
        )
    else:
        raise ValueError(f"unknown ROM integration rule: {integration}")

    eigenvalues, rotation, rank = generalized_symmetric_eig(
        raw_stiffness,
        raw_mass,
        cfg.modes,
        sample_rows=(2 * len(points)),
    )
    if rank != cfg.modes or rotation.shape[1] != cfg.modes:
        raise RuntimeError(f"ROM quadrature rank is {rank}, expected {cfg.modes}")
    if integration == "grid":
        assert grid_data is not None
        basis_u, basis_v, basis_omega, orth_error = evaluate_full_grid_ritz_basis(
            model,
            cfg,
            points,
            grid_data["weights"],
            rotation,
            beta,
            alpha,
            field_batch,
            grid_chunk,
        )
        if orth_error > grid_orthogonality_tolerance:
            raise RuntimeError(
                "full-grid Ritz mass orthogonality error "
                f"{orth_error:.3e} exceeds {grid_orthogonality_tolerance:.3e}"
            )
        volume_weights: Any = cp.asarray(grid_data["weights"])
        fluid = grid_data["fluid"].astype(bool)
        mean_u = mean_u_grid[fluid]
        mean_v = mean_v_grid[fluid]
        mean_omega = mean_omega_grid[fluid]
        outlet_indices = cp.asarray(grid_data["outlet_indices"])
        outlet_basis_u = basis_u[outlet_indices]
        outlet_basis_v = basis_v[outlet_indices]
        outlet_rows = grid_data["outlet_rows"]
        outlet_mean_u = mean_u_grid[outlet_rows, -1]
        outlet_mean_v = mean_v_grid[outlet_rows, -1]
        outlet_weights = grid_data["outlet_weights"]
    else:
        gpu_rotation = cp.asarray(rotation.astype(np.float32))
        basis_u = gpu_raw_u @ gpu_rotation
        basis_v = gpu_raw_v @ gpu_rotation
        basis_omega = gpu_raw_omega @ gpu_rotation
        del gpu_raw_u, gpu_raw_v, gpu_raw_omega
        cp.get_default_memory_pool().free_all_blocks()
        mass_check = weight * (
            basis_u.astype(cp.float64).T @ basis_u.astype(cp.float64)
            + basis_v.astype(cp.float64).T @ basis_v.astype(cp.float64)
        )
        orth_error = float(
            cp.max(cp.abs(mass_check - cp.eye(cfg.modes, dtype=cp.float64))).get()
        )
        volume_weights = weight
        mean_u, mean_v, mean_omega = interpolate_grid_fields(
            gt_source["x"],
            gt_source["y"],
            (mean_u_grid, mean_v_grid, mean_omega_grid),
            points,
        )
        y = gt_source["y"].astype(np.float64)
        outlet_points = np.column_stack(
            (np.full(len(y), float(gt_source["x"][-1])), y)
        ).astype(np.float32)
        outlet_raw_u, outlet_raw_v, _ = raw_fields(
            model, cfg, outlet_points, beta, alpha, field_batch
        )
        outlet_basis_u = cp.asarray(outlet_raw_u) @ gpu_rotation
        outlet_basis_v = cp.asarray(outlet_raw_v) @ gpu_rotation
        outlet_mean_u = mean_u_grid[:, -1]
        outlet_mean_v = mean_v_grid[:, -1]
        outlet_weights = np.full(len(y), float(y[1] - y[0]), dtype=np.float32)
        outlet_weights[[0, -1]] *= 0.5

    raw_initial = sample["raw_modal_coefficients"].astype(np.float64)
    initial = rotation.T @ raw_mass @ raw_initial
    initial = cp.asarray(initial.astype(np.float32))
    open_boundary = OpenBoundaryTerms(
        basis_u=outlet_basis_u,
        basis_v=outlet_basis_v,
        basis_normal=outlet_basis_u,
        weights=cp.asarray(outlet_weights),
        static_pressure=cp.float32(outlet_static_pressure),
        mean_u=cp.asarray(outlet_mean_u),
        mean_v=cp.asarray(outlet_mean_v),
    )
    dx = float(gt_source["x"][1] - gt_source["x"][0])
    lattice_viscosity = float(gt_metadata["lattice_viscosity"])
    coordinate_viscosity = 0.0 if zero_viscosity else lattice_viscosity * dx
    output_dt = steps_per_frame * dx
    internal_dt = output_dt / substeps_per_frame
    integrator = ImplicitTrapezoidalROM(
        basis_u,
        basis_v,
        basis_omega,
        cp.asarray(mean_u),
        cp.asarray(mean_v),
        cp.asarray(mean_omega),
        cp.asarray(eigenvalues.astype(np.float32)),
        volume_weights,
        coordinate_viscosity,
        newton_tolerance=2.0e-6,
        gmres_tolerance=2.0e-5,
        open_boundary=open_boundary,
    )
    coefficient_frames = np.empty((frames + 1, cfg.modes), dtype=np.float32)
    coefficient_frames[0] = cp.asnumpy(initial)
    started = time.perf_counter()
    coefficients = initial
    for frame in range(frames):
        for _ in range(substeps_per_frame):
            coefficients = integrator.step(coefficients, internal_dt)
        coefficient_frames[frame + 1] = cp.asnumpy(coefficients)
        print(
            f"[ROM {frame + 1:02d}/{frames}] coefficient_norm="
            f"{float(cp.linalg.norm(coefficients).get()):.6e}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    diagnostics = {
        "integration_rule": integration,
        "quadrature": int(len(points)),
        "quadrature_seed": quadrature_seed,
        "grid_shape": list(gt_source["solid"].shape) if integration == "grid" else None,
        "volume_weight_sum": (
            float(np.sum(grid_data["weights"], dtype=np.float64))
            if grid_data is not None
            else float(weight * quadrature)
        ),
        "outlet_points": int(len(outlet_weights)),
        "outlet_static_pressure_gauge": float(outlet_static_pressure),
        "outlet_boundary_term": ("-integral_outlet (p + 0.5*|u|^2) Phi.normal ds"),
        "rank": int(rank),
        "mass_orthogonality_error": orth_error,
        "coordinate_dx_per_lattice_node": dx,
        "lattice_viscosity": lattice_viscosity,
        "coordinate_viscosity": coordinate_viscosity,
        "zero_viscosity": bool(zero_viscosity),
        "steps_per_frame": steps_per_frame,
        "coordinate_time_per_frame": output_dt,
        "substeps_per_frame": substeps_per_frame,
        "nominal_internal_dt": internal_dt,
        "newton_iterations": integrator.newton_iterations,
        "gmres_iterations": integrator.linear_iterations,
        "adaptive_step_splits": integrator.step_splits,
        "elapsed_seconds": elapsed,
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "initial_coefficient_norm": float(np.linalg.norm(coefficient_frames[0])),
        "final_coefficient_norm": float(np.linalg.norm(coefficient_frames[-1])),
    }

    del basis_u, basis_v, basis_omega, integrator, open_boundary
    cp.get_default_memory_pool().free_all_blocks()
    return coefficient_frames, rotation, diagnostics, cfg, model


def reconstruct_rom_frames(
    cfg: Any,
    model: Any,
    rotation: np.ndarray,
    coefficient_frames: np.ndarray,
    sample: dict[str, np.ndarray],
    gt_source: dict[str, np.ndarray],
    field_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    solid = sample["solid"].astype(bool)
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(sample["x"], sample["y"])
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    started = time.perf_counter()
    print(
        f"[ROM reconstruction] evaluating {cfg.modes} modes on {len(points)} points",
        flush=True,
    )
    raw_u, raw_v, raw_omega = raw_fields(
        model,
        cfg,
        points,
        float(sample["beta"]),
        float(sample["alpha_deg"]),
        field_batch,
    )
    raw_coefficients = rotation @ coefficient_frames[1:].T.astype(np.float64)
    raw_coefficients_gpu = cp.asarray(raw_coefficients.astype(np.float32))
    mean_u = gt_source["u"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_v = gt_source["v"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_omega = (
        gt_source["vorticity"].mean(axis=0, dtype=np.float64).astype(np.float32)
    )

    def reconstruct(
        raw: np.ndarray, mean: np.ndarray, coefficients_gpu: cp.ndarray
    ) -> np.ndarray:
        values = cp.asnumpy(cp.asarray(raw) @ coefficients_gpu)
        values += mean[fluid, None]
        result = np.zeros(
            (coefficient_frames.shape[0] - 1,) + solid.shape,
            dtype=np.float32,
        )
        result[:, fluid] = values.T
        return result

    rom_u = reconstruct(raw_u, mean_u, raw_coefficients_gpu)
    del raw_u
    cp.get_default_memory_pool().free_all_blocks()
    rom_v = reconstruct(raw_v, mean_v, raw_coefficients_gpu)
    del raw_v
    cp.get_default_memory_pool().free_all_blocks()
    rom_omega = reconstruct(raw_omega, mean_omega, raw_coefficients_gpu)
    del raw_omega, raw_coefficients_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return rom_u, rom_v, rom_omega, time.perf_counter() - started


def metric_rows(
    ns_u: np.ndarray,
    ns_v: np.ndarray,
    ns_omega: np.ndarray,
    rom_u: np.ndarray,
    rom_v: np.ndarray,
    rom_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    fluid: np.ndarray,
    wake: np.ndarray,
    steps_per_frame: int,
) -> list[dict[str, float | int]]:
    rows = []
    for index in range(len(ns_u)):
        row: dict[str, float | int] = {
            "frame": index + 1,
            "local_lbm_step": (index + 1) * steps_per_frame,
            "velocity_error": relative_vector_error(
                ns_u[index], ns_v[index], rom_u[index], rom_v[index], fluid
            ),
            "vorticity_error": relative_scalar_error(
                ns_omega[index], rom_omega[index], fluid
            ),
            "wake_velocity_error": relative_vector_error(
                ns_u[index], ns_v[index], rom_u[index], rom_v[index], wake
            ),
            "wake_vorticity_error": relative_scalar_error(
                ns_omega[index], rom_omega[index], wake
            ),
            "fluctuation_velocity_error": relative_vector_error(
                ns_u[index] - mean_u,
                ns_v[index] - mean_v,
                rom_u[index] - mean_u,
                rom_v[index] - mean_v,
                fluid,
            ),
            "fluctuation_vorticity_error": relative_scalar_error(
                ns_omega[index] - mean_omega,
                rom_omega[index] - mean_omega,
                fluid,
            ),
            "wake_fluctuation_velocity_error": relative_vector_error(
                ns_u[index] - mean_u,
                ns_v[index] - mean_v,
                rom_u[index] - mean_u,
                rom_v[index] - mean_v,
                wake,
            ),
            "wake_fluctuation_vorticity_error": relative_scalar_error(
                ns_omega[index] - mean_omega,
                rom_omega[index] - mean_omega,
                wake,
            ),
        }
        rows.append(row)
    return rows


def render_movies(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    ns_u: np.ndarray,
    ns_v: np.ndarray,
    ns_omega: np.ndarray,
    rom_u: np.ndarray,
    rom_v: np.ndarray,
    rom_omega: np.ndarray,
    metrics: list[dict[str, float | int]],
    fps: int,
    alpha_deg: float,
    beta: float,
) -> tuple[Path, Path, Path]:
    velocity_frames = output_dir / "frames_velocity"
    vorticity_frames = output_dir / "frames_vorticity"
    velocity_frames.mkdir(parents=True, exist_ok=True)
    vorticity_frames.mkdir(parents=True, exist_ok=True)
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    fluid = ~solid
    speed_values = np.concatenate(
        (
            np.hypot(ns_u[:, fluid], ns_v[:, fluid]).ravel()[::20],
            np.hypot(rom_u[:, fluid], rom_v[:, fluid]).ravel()[::20],
        )
    )
    speed_limit = float(np.percentile(speed_values, 99.5))
    omega_values = np.concatenate(
        (
            np.abs(ns_omega[:, fluid]).ravel()[::20],
            np.abs(rom_omega[:, fluid]).ravel()[::20],
        )
    )
    omega_limit = float(np.percentile(omega_values, 99.0))
    speed_error_limit = float(
        np.percentile(np.hypot(rom_u - ns_u, rom_v - ns_v)[:, fluid], 99.0)
    )
    omega_error_limit = float(
        np.percentile(np.abs(rom_omega - ns_omega)[:, fluid], 99.0)
    )
    mask = np.where(solid, 1.0, np.nan)

    def panel(
        axis: Any, values: np.ndarray, cmap: str, low: float, high: float, title: str
    ) -> None:
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=low,
            vmax=high,
            interpolation="bilinear",
        )
        axis.imshow(
            mask,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, fontsize=10)
        return image

    for index, row in enumerate(metrics):
        ns_speed = np.hypot(ns_u[index], ns_v[index])
        rom_speed = np.hypot(rom_u[index], rom_v[index])
        speed_error = np.hypot(rom_u[index] - ns_u[index], rom_v[index] - ns_v[index])
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
        panel(axes[0], ns_speed, "viridis", 0.0, speed_limit, "TRT-LBM / NS GT speed")
        panel(axes[1], rom_speed, "viridis", 0.0, speed_limit, "Eigenfluid ROM speed")
        panel(
            axes[2],
            speed_error,
            "magma",
            0.0,
            speed_error_limit,
            f"absolute speed error; relative L2={100 * float(row['velocity_error']):.2f}%",
        )
        figure.suptitle(
            f"alpha={alpha_deg:+.1f} deg, beta={beta:.2f} | "
            f"forecast {index + 1:02d}/{len(metrics)} | "
            f"LBM step +{int(row['local_lbm_step'])}",
            fontsize=13,
        )
        figure.savefig(velocity_frames / f"frame_{index:03d}.png", dpi=120)
        plt.close(figure)

        figure, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
        panel(
            axes[0],
            ns_omega[index],
            "RdBu_r",
            -omega_limit,
            omega_limit,
            "TRT-LBM / NS GT vorticity",
        )
        panel(
            axes[1],
            rom_omega[index],
            "RdBu_r",
            -omega_limit,
            omega_limit,
            "Eigenfluid ROM vorticity",
        )
        panel(
            axes[2],
            np.abs(rom_omega[index] - ns_omega[index]),
            "magma",
            0.0,
            omega_error_limit,
            f"absolute vorticity error; relative L2={100 * float(row['vorticity_error']):.2f}%",
        )
        figure.suptitle(
            f"alpha={alpha_deg:+.1f} deg, beta={beta:.2f} | "
            f"forecast {index + 1:02d}/{len(metrics)} | "
            f"LBM step +{int(row['local_lbm_step'])}",
            fontsize=13,
        )
        figure.savefig(vorticity_frames / f"frame_{index:03d}.png", dpi=120)
        plt.close(figure)

    velocity_movie = output_dir / "ns_vs_eigenfluid_velocity_10fps.mp4"
    vorticity_movie = output_dir / "ns_vs_eigenfluid_vorticity_10fps.mp4"
    encode_mp4(velocity_frames, velocity_movie, fps)
    encode_mp4(vorticity_frames, vorticity_movie, fps)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    frame_ids = [int(row["frame"]) for row in metrics]
    axes[0].plot(
        frame_ids, [100 * float(row["velocity_error"]) for row in metrics], label="full"
    )
    axes[0].plot(
        frame_ids,
        [100 * float(row["wake_velocity_error"]) for row in metrics],
        label="wake",
    )
    axes[0].set_title("Velocity relative L2 error")
    axes[1].plot(
        frame_ids,
        [100 * float(row["vorticity_error"]) for row in metrics],
        label="full",
    )
    axes[1].plot(
        frame_ids,
        [100 * float(row["wake_vorticity_error"]) for row in metrics],
        label="wake",
    )
    axes[1].set_title("Vorticity relative L2 error")
    for axis in axes:
        axis.set_xlabel("forecast frame")
        axis.set_ylabel("error (%)")
        axis.grid(alpha=0.25)
        axis.legend()
    error_curve = output_dir / "eigenfluid_rollout_error_curves.png"
    figure.savefig(error_curve, dpi=160)
    plt.close(figure)
    return velocity_movie, vorticity_movie, error_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--gt-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument(
        "--rom-integration",
        choices=("grid", "sobol"),
        default="grid",
        help=(
            "Use every fluid node of the GT grid by default; 'sobol' retains "
            "the earlier sampled operator for reproducibility only"
        ),
    )
    parser.add_argument("--rom-quadrature", type=int, default=8192)
    parser.add_argument("--rom-grid-chunk", type=int, default=8192)
    parser.add_argument("--outlet-static-pressure", type=float, default=0.0)
    parser.add_argument(
        "--zero-viscosity",
        action="store_true",
        help=(
            "Set the physical shear viscosity to exactly zero in both the "
            "TRT-LBM reference (tau_even=0.5) and the ROM modal decay. The "
            "TRT odd/ghost relaxation time is retained because it does not "
            "set the physical kinematic viscosity."
        ),
    )
    parser.add_argument(
        "--rom-grid-orthogonality-tolerance", type=float, default=5.0e-4
    )
    parser.add_argument("--rom-substeps-per-frame", type=int, default=5)
    parser.add_argument("--field-batch", type=int, default=32768)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.sample, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}
    with np.load(args.gt_source, allow_pickle=False) as archive:
        gt_source = {name: archive[name] for name in archive.files}
    gt_metadata = json.loads(str(gt_source["metadata"]))
    if abs(float(sample["alpha_deg"]) - float(gt_metadata["alpha_deg"])) > 1.0e-6:
        raise ValueError("sample and GT alpha values differ")
    if abs(float(sample["beta"]) - float(gt_metadata["beta"])) > 1.0e-6:
        raise ValueError("sample and GT beta values differ")
    if not np.array_equal(sample["solid"], gt_source["solid"]):
        raise ValueError("sample and GT solid masks differ")
    solver = gt_metadata["solver_config"]
    print("[stage 1/4] TRT-LBM Navier-Stokes reference rollout", flush=True)
    ns_relaxation_time = (
        0.5 if args.zero_viscosity else float(gt_metadata["relaxation_time"])
    )
    ns_u, ns_v, ns_elapsed = rollout_trt_lbm(
        sample["pinn_u"],
        sample["pinn_v"],
        sample["solid"].astype(bool),
        args.frames,
        args.steps_per_frame,
        float(solver["lattice_inlet_speed"]),
        ns_relaxation_time,
        float(gt_metadata["odd_relaxation_time"]),
        str(solver["device"]),
    )
    dx = float(sample["x"][1] - sample["x"][0])
    dy = float(sample["y"][1] - sample["y"][0])
    ns_omega = vorticity(ns_u, ns_v, dx, dy).astype(np.float32)
    torch.cuda.empty_cache()

    print("[stage 2/4] variational coefficient-space Eigenfluid rollout", flush=True)
    coefficients, rotation, rom_diagnostics, cfg, model = build_rom_and_rollout(
        args.checkpoint,
        sample,
        gt_source,
        gt_metadata,
        args.frames,
        args.steps_per_frame,
        args.rom_quadrature,
        args.field_batch,
        args.rom_substeps_per_frame,
        args.rom_integration,
        args.rom_grid_chunk,
        args.outlet_static_pressure,
        args.rom_grid_orthogonality_tolerance,
        args.zero_viscosity,
    )
    print("[stage 3/4] reconstructing ROM trajectory on the GT grid", flush=True)
    rom_u, rom_v, rom_omega, reconstruction_elapsed = reconstruct_rom_frames(
        cfg,
        model,
        rotation,
        coefficients,
        sample,
        gt_source,
        args.field_batch,
    )
    solid = sample["solid"].astype(bool)
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(sample["x"], sample["y"])
    xb, yb = body_numpy(
        cfg,
        np.column_stack((xgrid.ravel(), ygrid.ravel())),
        float(sample["alpha_deg"]),
    )
    wake = (
        fluid.ravel()
        & (xb >= cfg.chord)
        & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords))
        & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)
    ).reshape(solid.shape)
    mean_u = gt_source["u"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_v = gt_source["v"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_omega = (
        gt_source["vorticity"].mean(axis=0, dtype=np.float64).astype(np.float32)
    )
    metrics = metric_rows(
        ns_u,
        ns_v,
        ns_omega,
        rom_u,
        rom_v,
        rom_omega,
        mean_u,
        mean_v,
        mean_omega,
        fluid,
        wake,
        args.steps_per_frame,
    )
    fields_path = args.output_dir / "ns_vs_eigenfluid_rollout_fields.npz"
    np.savez_compressed(
        fields_path,
        x=sample["x"],
        y=sample["y"],
        solid=solid,
        ns_u=ns_u,
        ns_v=ns_v,
        ns_omega=ns_omega,
        eigenfluid_u=rom_u,
        eigenfluid_v=rom_v,
        eigenfluid_omega=rom_omega,
        coefficient_frames=coefficients,
        local_lbm_steps=np.arange(1, args.frames + 1, dtype=np.int64)
        * args.steps_per_frame,
    )

    print("[stage 4/4] rendering comparisons", flush=True)
    velocity_movie, vorticity_movie, error_curve = render_movies(
        args.output_dir,
        sample["x"],
        sample["y"],
        solid,
        ns_u,
        ns_v,
        ns_omega,
        rom_u,
        rom_v,
        rom_omega,
        metrics,
        args.fps,
        float(sample["alpha_deg"]),
        float(sample["beta"]),
    )
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "initial_reconstruction": str(args.sample.resolve()),
        "mean_flow_source": str(args.gt_source.resolve()),
        "beta": float(sample["beta"]),
        "alpha_deg": float(sample["alpha_deg"]),
        "frames": args.frames,
        "fps": args.fps,
        "steps_per_frame": args.steps_per_frame,
        "same_initial_velocity_for_both_solvers": True,
        "ns_initialization": (
            "D2Q9 equilibrium populations with rho=1 from the reconstructed "
            "velocity; no non-equilibrium stress was available in the velocity-only sample"
        ),
        "ns_solver": "D2Q9 TRT lattice Boltzmann Navier-Stokes",
        "zero_viscosity": bool(args.zero_viscosity),
        "ns_even_relaxation_time": ns_relaxation_time,
        "ns_odd_relaxation_time": float(gt_metadata["odd_relaxation_time"]),
        "ns_elapsed_seconds": ns_elapsed,
        "rom_equation": (
            "C[g,h,i]=integral curl(Phi_i) dot (Phi_g cross Phi_h) dOmega; "
            "da/dt=(C contracted with a) a minus the outlet total-pressure "
            "boundary integral, evaluated matrix-free"
        ),
        "rom_integrator": (
            "implicit trapezoidal Newton-GMRES for variational advection, "
            "Strang-split exact exp(-nu*lambda*dt) modal viscosity"
        ),
        "rom": rom_diagnostics,
        "full_grid_reconstruction_elapsed_seconds": reconstruction_elapsed,
        "metrics": metrics,
        "aggregate": {
            key: {
                "mean": float(np.mean([float(row[key]) for row in metrics])),
                "p95": float(np.percentile([float(row[key]) for row in metrics], 95.0)),
                "max": float(np.max([float(row[key]) for row in metrics])),
                "final": float(metrics[-1][key]),
            }
            for key in metrics[0]
            if key.endswith("_error")
        },
        "fields": str(fields_path.resolve()),
        "velocity_movie": str(velocity_movie.resolve()),
        "vorticity_movie": str(vorticity_movie.resolve()),
        "error_curve": str(error_curve.resolve()),
    }
    write_json(args.output_dir / "rollout_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
