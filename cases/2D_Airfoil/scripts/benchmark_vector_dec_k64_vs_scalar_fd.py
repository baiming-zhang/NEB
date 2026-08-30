#!/usr/bin/env python3
"""Fair K=64 scalar-FD versus divergence-free vector-DEC airfoil pilot.

The vector method uses edge-flux 1-forms and an exact coexact/nullspace
parameterization.  If ``G`` is the weighted edge incidence (rotated when
interpreted as velocity), its velocity mass is ``M = G.T @ G`` and the
curl--curl energy pulled back to streamfunction coordinates is ``K = M.T @ M``.
The generalized problem

    K psi = lambda M psi

is solved directly with block LOBPCG and an AMG-based inverse-biharmonic
preconditioner.  This is not a component-wise vector Laplacian or a divergence
penalty.  The discrete flux divergence of ``rot(grad(psi))`` vanishes by the
incidence identity, including on the masked Cartesian cell complex.

Both methods are generated from scratch on the same strided grid and use the
same mask, mixed boundary conditions, temporal mean, frame, projection points,
and error masks.  Flow-projection time is archived only as an audit field; the
headline timing comparison is basis generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

try:
    import psutil
except ImportError:  # pragma: no cover - only used by the deferred launcher
    psutil = None

try:
    import pyamg
except ImportError as exc:  # pragma: no cover - checked before a long run
    raise RuntimeError("pyamg is required for the vector-DEC pilot") from exc


METRIC_KEYS = (
    "velocity_error",
    "vorticity_error",
    "fluctuation_velocity_error",
    "fluctuation_vorticity_error",
    "wake_velocity_error",
    "wake_vorticity_error",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def update_state(path: Path, **updates: Any) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    payload["updated_at"] = now()
    atomic_json(path, payload)


def wait_for_pid(pid: int, poll_seconds: float, state_path: Path) -> None:
    if pid <= 0:
        return
    if psutil is None:
        raise RuntimeError("psutil is required when --wait-pid is used")
    announced = False
    while psutil.pid_exists(pid):
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.Error:
            break
        if not announced:
            print(f"[wait] deferring CPU/RAM-heavy eigensolves until PID {pid} exits", flush=True)
            announced = True
        update_state(state_path, status="waiting", wait_pid=pid)
        time.sleep(poll_seconds)


def build_edge_dec(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    stride: int,
) -> dict[str, Any]:
    """Build a compatible edge incidence and its scalar 0-form Laplacian."""
    started = time.perf_counter()
    if stride < 1 or (len(x) - 1) % stride or (len(y) - 1) % stride:
        raise ValueError("stride must be positive and divide both grid interval counts")
    xc = np.asarray(x[::stride], dtype=np.float64)
    yc = np.asarray(y[::stride], dtype=np.float64)
    solid_c = np.asarray(solid[::stride, ::stride], dtype=bool)
    dx, dy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    if not math.isclose(dx, dy, rel_tol=1.0e-5, abs_tol=1.0e-7):
        raise ValueError("the pilot requires an isotropic Cartesian grid")

    unknown = ~solid_c
    unknown = unknown.copy()
    unknown[:, 0] = False
    unknown[0, :] = False
    unknown[-1, :] = False
    # x=xmax remains unknown: omission of the exterior east edge is the same
    # natural outlet condition used by the existing scalar FD baseline.
    ny, nx = unknown.shape
    node_index = np.full((ny, nx), -1, dtype=np.int32)
    node_index[unknown] = np.arange(np.count_nonzero(unknown), dtype=np.int32)
    node_count = int(np.count_nonzero(unknown))

    edge_rows: list[int] = []
    edge_columns: list[int] = []
    edge_data: list[float] = []
    edge = 0
    inverse_h = 1.0 / dx
    # Each active-active edge appears once.  An active-to-Dirichlet/solid edge
    # is a one-entry incidence row; the missing east exterior edge is omitted.
    for iy, ix in np.argwhere(unknown):
        source = int(node_index[iy, ix])
        for jy, jx in ((iy + 1, ix), (iy, ix + 1)):
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            if unknown[jy, jx]:
                edge_rows.extend((edge, edge))
                edge_columns.extend((source, int(node_index[jy, jx])))
                edge_data.extend((inverse_h, -inverse_h))
                edge += 1
        for jy, jx in ((iy - 1, ix), (iy + 1, ix), (iy, ix - 1), (iy, ix + 1)):
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            if not unknown[jy, jx]:
                edge_rows.append(edge)
                edge_columns.append(source)
                edge_data.append(inverse_h)
                edge += 1

    incidence = scipy.sparse.csr_matrix(
        (edge_data, (edge_rows, edge_columns)), shape=(edge, node_count)
    )
    mass = (incidence.T @ incidence).tocsr()
    return {
        "x": xc,
        "y": yc,
        "solid": solid_c,
        "unknown": unknown,
        "node_index": node_index,
        "incidence": incidence,
        "mass": mass,
        "dx": dx,
        "dy": dy,
        "assembly_seconds": time.perf_counter() - started,
    }


def build_existing_scalar_fd(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    stride: int,
) -> dict[str, Any]:
    """Reproduce the existing scalar-FD operator and its measured assembly."""
    started = time.perf_counter()
    if stride < 1 or (len(x) - 1) % stride or (len(y) - 1) % stride:
        raise ValueError("stride must be positive and divide both grid interval counts")
    xc = np.asarray(x[::stride], dtype=np.float64)
    yc = np.asarray(y[::stride], dtype=np.float64)
    solid_c = np.asarray(solid[::stride, ::stride], dtype=bool)
    dx, dy = float(xc[1] - xc[0]), float(yc[1] - yc[0])
    if not math.isclose(dx, dy, rel_tol=1.0e-5, abs_tol=1.0e-7):
        raise ValueError("the pilot requires an isotropic Cartesian grid")
    unknown = ~solid_c
    unknown = unknown.copy()
    unknown[:, 0] = False
    unknown[0, :] = False
    unknown[-1, :] = False
    ny, nx = unknown.shape
    node_index = np.full((ny, nx), -1, dtype=np.int32)
    node_index[unknown] = np.arange(np.count_nonzero(unknown), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    inverse_h2 = 1.0 / (dx * dx)
    for iy, ix in np.argwhere(unknown):
        row = int(node_index[iy, ix])
        diagonal = 0.0
        for jy, jx in (
            (iy - 1, ix),
            (iy + 1, ix),
            (iy, ix - 1),
            (iy, ix + 1),
        ):
            if not (0 <= jy < ny and 0 <= jx < nx):
                continue
            diagonal += inverse_h2
            if unknown[jy, jx]:
                rows.append(row)
                columns.append(int(node_index[jy, jx]))
                data.append(-inverse_h2)
        rows.append(row)
        columns.append(row)
        data.append(diagonal)
    node_count = int(np.count_nonzero(unknown))
    operator = scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(node_count, node_count)
    )
    return {
        "x": xc,
        "y": yc,
        "solid": solid_c,
        "unknown": unknown,
        "node_index": node_index,
        "mass": operator,
        "dx": dx,
        "dy": dy,
        "assembly_seconds": time.perf_counter() - started,
    }


def scalar_fd_basis(dec: dict[str, Any], modes: int, seed: int) -> dict[str, Any]:
    operator = dec["mass"]
    rng = np.random.default_rng(seed)
    solve_started = time.perf_counter()
    values, vectors = scipy.sparse.linalg.eigsh(
        operator,
        k=modes,
        sigma=0.0,
        which="LM",
        v0=rng.standard_normal(operator.shape[0]),
        tol=2.0e-8,
        maxiter=20000,
    )
    solve_seconds = time.perf_counter() - solve_started
    post_started = time.perf_counter()
    order = np.argsort(values, kind="stable")
    values, vectors = values[order], vectors[:, order]
    dx, dy = dec["dx"], dec["dy"]
    vectors /= np.sqrt(np.sum(vectors * vectors, axis=0) * dx * dy)[None, :]
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(modes)])
    signs[signs == 0.0] = 1.0
    vectors *= signs

    ny, nx = dec["unknown"].shape
    psi = np.zeros((ny, nx, modes), dtype=np.float32)
    psi[dec["unknown"]] = vectors.astype(np.float32)
    gx = np.gradient(psi, dx, axis=1, edge_order=2).astype(np.float32)
    gy = np.gradient(psi, dy, axis=0, edge_order=2).astype(np.float32)
    gx[:, -1] = 0.0
    omega = psi * values.astype(np.float32)[None, None, :]
    residuals = np.linalg.norm(operator @ vectors - vectors * values, axis=0) / np.maximum(
        np.abs(values) * np.linalg.norm(vectors, axis=0), 1.0e-30
    )
    post_seconds = time.perf_counter() - post_started
    return {
        "method": "scalar_fd_streamfunction_laplacian",
        "psi": psi,
        "u": gy,
        "v": -gx,
        "omega": omega,
        "eigenvalues": values,
        "diagnostics": {
            "maximum_relative_eigen_residual": float(np.max(residuals)),
            "mass_orthogonality_max_abs": float(
                np.max(np.abs(dx * dy * (vectors.T @ vectors) - np.eye(modes)))
            ),
            "timing_seconds": {
                "operator_assembly": float(dec["assembly_seconds"]),
                "eigensolve": solve_seconds,
                "postprocess": post_seconds,
                "total_basis_generation": float(dec["assembly_seconds"])
                + solve_seconds
                + post_seconds,
            },
        },
    }


def _b_orthonormalize(x: np.ndarray, b: scipy.sparse.spmatrix) -> np.ndarray:
    gram = x.T @ (b @ x)
    gram = 0.5 * (gram + gram.T)
    chol = scipy.linalg.cholesky(gram, lower=False, check_finite=False)
    return scipy.linalg.solve_triangular(
        chol, x.T, lower=False, trans="T", check_finite=False
    ).T


def vector_dec_basis(
    dec: dict[str, Any],
    modes: int,
    seed: int,
    tolerance: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Solve curl--curl u=lambda u in the exact coexact DEC subspace."""
    mass = dec["mass"].astype(np.float64)
    node_count = mass.shape[0]
    if modes >= node_count - 1:
        raise ValueError(f"K={modes} is invalid for {node_count} potential DOFs")

    stiffness_started = time.perf_counter()
    stiffness = (mass @ mass).tocsr()
    stiffness_seconds = time.perf_counter() - stiffness_started

    amg_started = time.perf_counter()
    hierarchy = pyamg.smoothed_aggregation_solver(
        mass,
        symmetry="symmetric",
        strength="symmetric",
        max_coarse=256,
    )
    inverse_mass = hierarchy.aspreconditioner(cycle="V")

    def inverse_biharmonic_matmat(block: np.ndarray) -> np.ndarray:
        return inverse_mass @ (inverse_mass @ block)

    preconditioner = scipy.sparse.linalg.LinearOperator(
        shape=mass.shape,
        dtype=np.float64,
        matvec=lambda vector: inverse_biharmonic_matmat(vector.reshape(-1, 1)).ravel(),
        matmat=inverse_biharmonic_matmat,
    )
    amg_seconds = time.perf_counter() - amg_started

    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((node_count, modes))
    initial = _b_orthonormalize(initial, mass)
    solve_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values, vectors, lambda_history, residual_history = scipy.sparse.linalg.lobpcg(
            stiffness,
            initial,
            B=mass,
            M=preconditioner,
            largest=False,
            tol=tolerance,
            maxiter=max_iterations,
            verbosityLevel=0,
            retLambdaHistory=True,
            retResidualNormsHistory=True,
        )
    solve_seconds = time.perf_counter() - solve_started

    post_started = time.perf_counter()
    # Final Rayleigh--Ritz extraction in the returned subspace gives a stable
    # B-orthonormal basis even when LOBPCG exits after its best iteration.
    vectors = _b_orthonormalize(np.asarray(vectors, dtype=np.float64), mass)
    mass_vectors = mass @ vectors
    small_k = mass_vectors.T @ mass_vectors
    small_m = vectors.T @ mass_vectors
    values, rotation = scipy.linalg.eigh(
        0.5 * (small_k + small_k.T),
        0.5 * (small_m + small_m.T),
        check_finite=False,
    )
    order = np.argsort(values, kind="stable")[:modes]
    values, vectors = values[order], vectors @ rotation[:, order]
    mass_vectors = mass @ vectors
    stiffness_vectors = mass @ mass_vectors
    residuals = np.linalg.norm(stiffness_vectors - mass_vectors * values, axis=0) / np.maximum(
        np.linalg.norm(stiffness_vectors, axis=0)
        + np.abs(values) * np.linalg.norm(mass_vectors, axis=0),
        1.0e-30,
    )
    b_gram = vectors.T @ mass_vectors

    ny, nx = dec["unknown"].shape
    psi = np.zeros((ny, nx, modes), dtype=np.float64)
    psi[dec["unknown"]] = vectors
    dx, dy = dec["dx"], dec["dy"]
    # Edge-flux 1-forms.  Averaging adjacent staggered values is only the final
    # interpolation from DEC edges to the collocated NS output grid.
    u_edge = np.diff(psi, axis=0) / dy
    v_edge = -np.diff(psi, axis=1) / dx
    u = np.empty_like(psi)
    v = np.empty_like(psi)
    u[0] = u_edge[0]
    u[-1] = u_edge[-1]
    u[1:-1] = 0.5 * (u_edge[:-1] + u_edge[1:])
    v[:, 0] = v_edge[:, 0]
    v[:, -1] = 0.0  # exact natural outlet trace, matching scalar FD
    v[:, 1:-1] = 0.5 * (v_edge[:, :-1] + v_edge[:, 1:])
    omega = np.zeros_like(psi)
    omega[dec["unknown"]] = mass_vectors

    # Topological divergence diagnostic before nodal interpolation.
    divergence = (
        (u_edge[:, 1:] - u_edge[:, :-1]) / dx
        + (v_edge[1:, :] - v_edge[:-1, :]) / dy
    )
    flux_scale = max(float(np.max(np.abs(u_edge))), float(np.max(np.abs(v_edge))), 1.0e-30)
    post_seconds = time.perf_counter() - post_started
    residual_array = np.asarray(residual_history)
    return {
        "method": "vector_dec_edge_flux_curl_curl_exact_coexact",
        "psi": psi.astype(np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "omega": omega.astype(np.float32),
        "eigenvalues": values,
        "diagnostics": {
            "formulation": "K psi = lambda M psi; M=G^T G; K=M^T M",
            "velocity_dofs": int(dec["incidence"].shape[0]),
            "potential_dofs": int(node_count),
            "componentwise_vector_laplacian": False,
            "divergence_penalty": False,
            "exact_coexact_parameterization": True,
            "maximum_absolute_edge_flux_divergence": float(np.max(np.abs(divergence))),
            "maximum_scaled_edge_flux_divergence": float(
                np.max(np.abs(divergence)) * max(dx, dy) / flux_scale
            ),
            "mass_orthogonality_max_abs": float(np.max(np.abs(b_gram - np.eye(modes)))),
            "maximum_relative_generalized_eigen_residual": float(np.max(residuals)),
            "lobpcg_requested_tolerance": tolerance,
            "lobpcg_max_iterations": max_iterations,
            "lobpcg_history_length": int(len(lambda_history)),
            "lobpcg_last_max_absolute_residual": float(np.max(residual_array[-1])),
            "warnings": [str(item.message) for item in caught],
            "timing_seconds": {
                "edge_and_mass_assembly": float(dec["assembly_seconds"]),
                "curl_curl_stiffness_assembly": stiffness_seconds,
                "amg_preconditioner_setup": amg_seconds,
                "generalized_eigensolve": solve_seconds,
                "postprocess": post_seconds,
                "total_basis_generation": float(dec["assembly_seconds"])
                + stiffness_seconds
                + amg_seconds
                + solve_seconds
                + post_seconds,
            },
        },
    }


def interpolators(dec: dict[str, Any], basis: dict[str, Any]) -> tuple[Any, Any, Any]:
    axes = (dec["y"], dec["x"])
    options = {"bounds_error": True, "method": "linear"}
    return tuple(
        scipy.interpolate.RegularGridInterpolator(axes, basis[key], **options)
        for key in ("u", "v", "omega")
    )


def relative_velocity(
    target_u: np.ndarray,
    target_v: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum(
        (estimate_u[mask] - target_u[mask]) ** 2
        + (estimate_v[mask] - target_v[mask]) ** 2,
        dtype=np.float64,
    )
    denominator = np.sum(target_u[mask] ** 2 + target_v[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def relative_scalar(
    target: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum((estimate[mask] - target[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(target[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def velocity_projection(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    u64, v64 = basis_u.astype(np.float64), basis_v.astype(np.float64)
    tu64, tv64 = target_u.astype(np.float64), target_v.astype(np.float64)
    gram = u64.T @ u64 + v64.T @ v64
    right = u64.T @ tu64 + v64.T @ tv64
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
    retained = values > threshold
    coefficients = vectors[:, retained] @ ((vectors[:, retained].T @ right) / values[retained])
    residual = relative_velocity(
        target_u, target_v, basis_u @ coefficients, basis_v @ coefficients
    )
    return coefficients, int(np.count_nonzero(retained)), residual


def load_ground_truth(gt: Path, frame: int) -> dict[str, Any]:
    with np.load(gt, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
        metadata = json.loads(str(archive["metadata"]))
        steps = archive["steps"]
        if not 0 <= frame < len(steps):
            raise ValueError(f"frame {frame} is outside 0..{len(steps)-1}")
        actual_u = archive["u"][frame].astype(np.float32)
        actual_v = archive["v"][frame].astype(np.float32)
        actual_omega = archive["vorticity"][frame].astype(np.float32)
        mean_u = np.mean(archive["u"], axis=0, dtype=np.float64).astype(np.float32)
        mean_v = np.mean(archive["v"], axis=0, dtype=np.float64).astype(np.float32)
        mean_omega = np.mean(archive["vorticity"], axis=0, dtype=np.float64).astype(np.float32)
        step = int(steps[frame])
    return {
        "x": x,
        "y": y,
        "solid": solid,
        "metadata": metadata,
        "frame": frame,
        "step": step,
        "actual_u": actual_u,
        "actual_v": actual_v,
        "actual_omega": actual_omega,
        "mean_u": mean_u,
        "mean_v": mean_v,
        "mean_omega": mean_omega,
    }


def reconstruct(
    dec: dict[str, Any],
    basis: dict[str, Any],
    gt: dict[str, Any],
    projection_points: int,
    field_batch: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    fluid = ~gt["solid"]
    xgrid, ygrid = np.meshgrid(gt["x"], gt["y"])
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    query = np.column_stack((points[:, 1], points[:, 0]))
    target_u = gt["actual_u"][fluid] - gt["mean_u"][fluid]
    target_v = gt["actual_v"][fluid] - gt["mean_v"][fluid]
    target_omega = gt["actual_omega"][fluid] - gt["mean_omega"][fluid]
    subset = np.linspace(
        0, len(points) - 1, min(projection_points, len(points)), dtype=np.int64
    )
    ui, vi, wi = interpolators(dec, basis)
    basis_u = np.asarray(ui(query[subset]), dtype=np.float32)
    basis_v = np.asarray(vi(query[subset]), dtype=np.float32)
    coefficients, rank, projection_residual = velocity_projection(
        basis_u, basis_v, target_u[subset], target_v[subset]
    )
    del basis_u, basis_v

    fluctuation_u = np.empty(len(points), dtype=np.float32)
    fluctuation_v = np.empty(len(points), dtype=np.float32)
    fluctuation_omega = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        batch = query[start:stop]
        fluctuation_u[start:stop] = np.asarray(ui(batch), dtype=np.float32) @ coefficients
        fluctuation_v[start:stop] = np.asarray(vi(batch), dtype=np.float32) @ coefficients
        fluctuation_omega[start:stop] = np.asarray(wi(batch), dtype=np.float32) @ coefficients
    estimate_u = gt["mean_u"][fluid] + fluctuation_u
    estimate_v = gt["mean_v"][fluid] + fluctuation_v
    estimate_omega = gt["mean_omega"][fluid] + fluctuation_omega

    alpha = math.radians(-float(gt["metadata"]["alpha_deg"]))
    c, s = math.cos(alpha), math.sin(alpha)
    dx = points[:, 0] - 3.0
    dy = points[:, 1]
    xb = c * dx + s * dy + 0.5
    yb = -s * dx + c * dy
    wake = (xb >= 2.0) & (xb <= 8.0) & (np.abs(yb) <= 1.5)
    metrics = {
        "velocity_error": relative_velocity(
            gt["actual_u"][fluid], gt["actual_v"][fluid], estimate_u, estimate_v
        ),
        "vorticity_error": relative_scalar(gt["actual_omega"][fluid], estimate_omega),
        "fluctuation_velocity_error": relative_velocity(
            target_u, target_v, fluctuation_u, fluctuation_v
        ),
        "fluctuation_vorticity_error": relative_scalar(target_omega, fluctuation_omega),
        "wake_velocity_error": relative_velocity(
            gt["actual_u"][fluid], gt["actual_v"][fluid], estimate_u, estimate_v, wake
        ),
        "wake_vorticity_error": relative_scalar(
            gt["actual_omega"][fluid], estimate_omega, wake
        ),
    }

    def grid(values: np.ndarray) -> np.ndarray:
        result = np.zeros(fluid.shape, dtype=np.float32)
        result[fluid] = values
        return result

    return {
        "projection_rank": rank,
        "projection_relative_residual": projection_residual,
        "projection_points": int(len(subset)),
        "metrics": metrics,
        "evaluation_seconds": time.perf_counter() - started,
        "fields": {
            "u": grid(estimate_u),
            "v": grid(estimate_v),
            "omega": grid(estimate_omega),
        },
    }


def reference_metrics(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["laplacian"]["k64"]["metrics_by_frame"][0]
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def self_test() -> dict[str, float]:
    x = np.linspace(0.0, 1.0, 17)
    y = np.linspace(-0.5, 0.5, 17)
    solid = np.zeros((len(y), len(x)), dtype=bool)
    solid[7:10, 6:9] = True
    dec = build_edge_dec(x, y, solid, 1)
    scalar = build_existing_scalar_fd(x, y, solid, 1)
    mass = dec["mass"]
    factorization_error = scipy.sparse.linalg.norm(
        mass - dec["incidence"].T @ dec["incidence"]
    )
    baseline_matrix_error = scipy.sparse.linalg.norm(mass - scalar["mass"])
    dense_mass = mass.toarray()
    scalar_values = scipy.linalg.eigvalsh(dense_mass, check_finite=False)[:8]
    vector_values = scipy.linalg.eigvalsh(
        dense_mass @ dense_mass, dense_mass, check_finite=False
    )[:8]
    spectral_error = float(np.max(np.abs(scalar_values - vector_values)))
    iterative = vector_dec_basis(dec, 4, 9, 1.0e-7, 80)
    iterative_spectral_error = float(
        np.max(np.abs(iterative["eigenvalues"] - scalar_values[:4]))
    )
    rng = np.random.default_rng(8)
    psi = rng.standard_normal(dec["unknown"].shape)
    u_edge = np.diff(psi, axis=0) / dec["dy"]
    v_edge = -np.diff(psi, axis=1) / dec["dx"]
    divergence = (
        (u_edge[:, 1:] - u_edge[:, :-1]) / dec["dx"]
        + (v_edge[1:, :] - v_edge[:-1, :]) / dec["dy"]
    )
    divergence_error = float(np.max(np.abs(divergence)))
    if (
        factorization_error > 1.0e-10
        or baseline_matrix_error > 1.0e-10
        or spectral_error > 1.0e-7
        or iterative_spectral_error > 2.0e-5
        or divergence_error > 1.0e-10
    ):
        raise AssertionError(
            f"DEC self-test failed: factor={factorization_error}, "
            f"baseline={baseline_matrix_error}, spectrum={spectral_error}, "
            f"iterative={iterative_spectral_error}, divergence={divergence_error}"
        )
    return {
        "incidence_factorization_error": float(factorization_error),
        "existing_scalar_matrix_max_error": float(baseline_matrix_error),
        "generalized_spectrum_max_abs_error": spectral_error,
        "iterative_generalized_spectrum_max_abs_error": iterative_spectral_error,
        "rotated_gradient_divergence_max_abs": divergence_error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "method",
        "total_basis_generation_seconds",
        "eigensolve_seconds",
        "projection_rank",
        *METRIC_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--modes", type=int, default=64)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--lobpcg-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--lobpcg-max-iterations", type=int, default=300)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--wait-poll-seconds", type=float, default=30.0)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "state.json"

    test = self_test()
    print(json.dumps({"self_test": test}, indent=2), flush=True)
    if args.self_test_only:
        return
    wait_for_pid(args.wait_pid, args.wait_poll_seconds, state_path)
    update_state(state_path, status="loading_ground_truth", started_at=now())
    gt = load_ground_truth(args.gt, args.frame)

    update_state(state_path, status="assembling_scalar_fd")
    scalar_dec = build_existing_scalar_fd(
        gt["x"], gt["y"], gt["solid"], args.stride
    )
    update_state(state_path, status="solving_scalar_fd")
    scalar = scalar_fd_basis(scalar_dec, args.modes, args.seed + 1)
    update_state(
        state_path,
        status="evaluating_scalar_fd",
        scalar_timing=scalar["diagnostics"]["timing_seconds"],
    )
    scalar_reconstruction = reconstruct(
        scalar_dec, scalar, gt, args.projection_points, args.field_batch
    )
    del scalar["psi"], scalar["u"], scalar["v"], scalar["omega"]

    update_state(state_path, status="assembling_vector_dec")
    # Rebuild from raw geometry so both headline totals include independent,
    # from-scratch operator assembly; no scalar operator is reused.
    vector_dec = build_edge_dec(gt["x"], gt["y"], gt["solid"], args.stride)
    update_state(state_path, status="solving_vector_dec")
    vector = vector_dec_basis(
        vector_dec,
        args.modes,
        args.seed + 2,
        args.lobpcg_tolerance,
        args.lobpcg_max_iterations,
    )
    update_state(
        state_path,
        status="evaluating_vector_dec",
        vector_timing=vector["diagnostics"]["timing_seconds"],
    )
    vector_reconstruction = reconstruct(
        vector_dec, vector, gt, args.projection_points, args.field_batch
    )
    del vector["psi"], vector["u"], vector["v"], vector["omega"]

    reference = reference_metrics(args.reference_summary)
    scalar_metrics = scalar_reconstruction["metrics"]
    vector_metrics = vector_reconstruction["metrics"]
    scalar_time = scalar["diagnostics"]["timing_seconds"]["total_basis_generation"]
    vector_time = vector["diagnostics"]["timing_seconds"]["total_basis_generation"]
    comparison = {
        "vector_over_scalar_basis_generation_time": float(vector_time / scalar_time),
        "maximum_relative_eigenvalue_difference": float(
            np.max(
                np.abs(vector["eigenvalues"] - scalar["eigenvalues"])
                / np.maximum(np.abs(scalar["eigenvalues"]), 1.0e-30)
            )
        ),
        "vector_minus_scalar_metrics": {
            key: float(vector_metrics[key] - scalar_metrics[key]) for key in METRIC_KEYS
        },
        "vector_over_scalar_metrics": {
            key: float(vector_metrics[key] / max(scalar_metrics[key], 1.0e-30))
            for key in METRIC_KEYS
        },
        "fresh_scalar_minus_existing_fd_metrics": None
        if reference is None
        else {key: float(scalar_metrics[key] - reference[key]) for key in METRIC_KEYS},
    }
    result = {
        "complete": True,
        "completed_at": now(),
        "case_label": "up",
        "internal_gt_path": str(args.gt.resolve()),
        "frame": args.frame,
        "step": gt["step"],
        "modes": args.modes,
        "protocol": {
            "grid_shape_full": list(gt["solid"].shape),
            "grid_shape_basis": list(scalar_dec["unknown"].shape),
            "grid_stride": args.stride,
            "projection_points": args.projection_points,
            "same_projection_points": True,
            "same_41_frame_temporal_mean": True,
            "dirichlet_boundaries": ["airfoil", "inlet", "ymin", "ymax"],
            "natural_boundaries": ["outlet"],
            "headline_time_definition": "from-scratch basis generation only",
        },
        "self_test": test,
        "scalar_fd": {
            "eigenvalue_first": float(scalar["eigenvalues"][0]),
            "eigenvalue_last": float(scalar["eigenvalues"][-1]),
            "diagnostics": scalar["diagnostics"],
            "projection_rank": scalar_reconstruction["projection_rank"],
            "projection_relative_residual": scalar_reconstruction[
                "projection_relative_residual"
            ],
            "metrics": scalar_metrics,
            "evaluation_seconds_not_headline": scalar_reconstruction["evaluation_seconds"],
        },
        "vector_dec": {
            "eigenvalue_first": float(vector["eigenvalues"][0]),
            "eigenvalue_last": float(vector["eigenvalues"][-1]),
            "diagnostics": vector["diagnostics"],
            "projection_rank": vector_reconstruction["projection_rank"],
            "projection_relative_residual": vector_reconstruction[
                "projection_relative_residual"
            ],
            "metrics": vector_metrics,
            "evaluation_seconds_not_headline": vector_reconstruction["evaluation_seconds"],
        },
        "existing_fd_reference_metrics": reference,
        "comparison": comparison,
    }
    atomic_json(args.output_dir / "comparison.json", result)

    rows = []
    for name, method, reconstruction in (
        ("scalar_fd", scalar, scalar_reconstruction),
        ("vector_dec", vector, vector_reconstruction),
    ):
        timing = method["diagnostics"]["timing_seconds"]
        rows.append(
            {
                "method": name,
                "total_basis_generation_seconds": timing["total_basis_generation"],
                "eigensolve_seconds": timing.get(
                    "eigensolve", timing.get("generalized_eigensolve")
                ),
                "projection_rank": reconstruction["projection_rank"],
                **reconstruction["metrics"],
            }
        )
    write_csv(args.output_dir / "comparison.csv", rows)
    np.savez_compressed(
        args.output_dir / "reconstructions_stride2.npz",
        x=gt["x"][::2],
        y=gt["y"][::2],
        solid=gt["solid"][::2, ::2],
        actual_u=gt["actual_u"][::2, ::2],
        actual_v=gt["actual_v"][::2, ::2],
        actual_omega=gt["actual_omega"][::2, ::2],
        scalar_fd_u=scalar_reconstruction["fields"]["u"][::2, ::2],
        scalar_fd_v=scalar_reconstruction["fields"]["v"][::2, ::2],
        scalar_fd_omega=scalar_reconstruction["fields"]["omega"][::2, ::2],
        vector_dec_u=vector_reconstruction["fields"]["u"][::2, ::2],
        vector_dec_v=vector_reconstruction["fields"]["v"][::2, ::2],
        vector_dec_omega=vector_reconstruction["fields"]["omega"][::2, ::2],
    )
    update_state(
        state_path,
        status="complete",
        complete=True,
        comparison_json=str((args.output_dir / "comparison.json").resolve()),
    )
    print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
