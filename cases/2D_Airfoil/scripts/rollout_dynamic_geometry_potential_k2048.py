#!/usr/bin/env python3
"""Roll out one full moving-airfoil cycle in a parameter-dependent ROM.

The Potential--Rayleigh MLP is evaluated once at every saved geometry state.
The previous physical velocity is projected into the new velocity-orthonormal
span before one output interval is advanced.  This discrete overlap transfer
is the first-order ALE realization of the ``Phi(q)^T M dPhi(q)/dt`` term.

Only frame zero is initialized from GT.  Later GT frames are read after each
prediction solely to report errors; they never enter coefficient propagation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import cupy as cp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    airfoil_area,
    body_numpy,
    fluid_points,
)
from eigenfluid.gen.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    load_hardcoded_model,
)
from evaluate_strict_rayleigh_k2048_prefixes import (  # noqa: E402
    build_transform_gpu,
    raw_objective_gpu,
)
from rollout_gt30_fd_vs_potential_k2048 import (  # noqa: E402
    gpu_ritz,
    project_initial,
    rollout_coefficients,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _cosine_segment(
    phase: float,
    start_phase: float,
    end_phase: float,
    start_value: float,
    end_value: float,
) -> tuple[float, float]:
    local = (phase - start_phase) / (end_phase - start_phase)
    blend = 0.5 * (1.0 - math.cos(math.pi * local))
    derivative = 0.5 * math.pi * math.sin(math.pi * local) / (
        end_phase - start_phase
    )
    delta = end_value - start_value
    return start_value + delta * blend, delta * derivative


def periodic_geometry_state(phase: float) -> tuple[float, float, float, float]:
    """Match the dynamic-GT trajectory exactly, including phase derivatives."""

    phase = float(phase % 1.0)
    if phase < 1.0 / 3.0:
        alpha, dalpha = _cosine_segment(
            phase, 0.0, 1.0 / 3.0, 0.0, 10.0
        )
    elif phase < 2.0 / 3.0:
        alpha, dalpha = _cosine_segment(
            phase, 1.0 / 3.0, 2.0 / 3.0, 10.0, -10.0
        )
    else:
        alpha, dalpha = _cosine_segment(
            phase, 2.0 / 3.0, 1.0, -10.0, 0.0
        )
    beta = 0.5 - 0.35 * math.cos(2.0 * math.pi * phase)
    dbeta = 0.7 * math.pi * math.sin(2.0 * math.pi * phase)
    return beta, alpha, dbeta, dalpha


def ellipse_level(cfg: Any, points: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    xb, yb = body_numpy(cfg, points, alpha)
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    a = 0.5 * cfg.chord
    b = 0.5 * thickness * cfg.chord
    return ((xb - a) / a) ** 2 + (yb / b) ** 2 - 1.0


def analytic_base_fields(
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    dbeta_dphase: float,
    dalpha_dphase: float,
    *,
    inlet_speed: float,
    cycle_steps: int,
    coordinate_dt_per_lattice_step: float,
    fd_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No-GT analytic ellipse flow plus moving-wall liftings.

    The steady part is the conformal-map potential flow around the current
    ellipse.  It is exactly impermeable, approaches the prescribed free stream,
    and has zero vorticity in the fluid; the learned modes therefore carry the
    viscous no-slip boundary layer instead of cancelling artificial base curl.
    Pitching is represented by a compact rigid-rotation streamfunction.
    Thickness change has non-zero wall flux, so its required net flux is
    supplied by an irrotational source placed at the ellipse centre.
    """

    inverse_cycle_time = 1.0 / (
        float(cycle_steps) * float(coordinate_dt_per_lattice_step)
    )
    alpha_dot = math.radians(dalpha_dphase) * inverse_cycle_time
    beta_dot = dbeta_dphase * inverse_cycle_time
    pivot = np.asarray((cfg.quarter_chord_x, cfg.quarter_chord_y), dtype=np.float64)

    # Conformal map z = zeta + m/zeta from |zeta|=R to an ellipse with
    # semiaxes a,b.  dW/dz is the complex velocity u_body-i*v_body.
    xb, yb = body_numpy(cfg, points, alpha)
    thickness = cfg.thickness_min + (cfg.thickness_max - cfg.thickness_min) * beta
    a = 0.5 * cfg.chord
    b = 0.5 * thickness * cfg.chord
    circle_radius = 0.5 * (a + b)
    mapping_parameter = 0.25 * (a * a - b * b)
    z = (xb - a).astype(np.complex128) + 1j * yb.astype(np.complex128)
    radical = np.sqrt(z * z - 4.0 * mapping_parameter + 0j)
    root_plus = 0.5 * (z + radical)
    root_minus = 0.5 * (z - radical)
    zeta = np.where(np.abs(root_plus) >= np.abs(root_minus), root_plus, root_minus)
    incidence = math.radians(alpha)
    derivative_potential = inlet_speed * (
        np.exp(-1j * incidence)
        - circle_radius**2 * np.exp(1j * incidence) / (zeta * zeta)
    )
    derivative_map = 1.0 - mapping_parameter / (zeta * zeta)
    complex_velocity = derivative_potential / derivative_map
    body_u = complex_velocity.real
    body_v = -complex_velocity.imag
    cosine = math.cos(incidence)
    sine = math.sin(incidence)
    u = cosine * body_u + sine * body_v
    v = -sine * body_u + cosine * body_v

    def streamfunction(query: np.ndarray) -> np.ndarray:
        level = ellipse_level(cfg, query, beta, alpha)
        positive_level = np.maximum(level, 0.0)
        geometry_window = np.exp(-((positive_level / 0.75) ** 2))
        relative = query.astype(np.float64, copy=False) - pivot[None, :]
        # The plotted body orientation is theta=-alpha; therefore
        # psi_rot=-theta_dot*r^2/2=alpha_dot*r^2/2.
        pitching = (
            0.5
            * alpha_dot
            * np.sum(relative * relative, axis=1)
            * geometry_window
        )
        return pitching

    epsilon = float(fd_epsilon)
    pxp = points.copy()
    pxp[:, 0] += epsilon
    pxm = points.copy()
    pxm[:, 0] -= epsilon
    pyp = points.copy()
    pyp[:, 1] += epsilon
    pym = points.copy()
    pym[:, 1] -= epsilon
    center = streamfunction(points)
    plus_x = streamfunction(pxp)
    minus_x = streamfunction(pxm)
    plus_y = streamfunction(pyp)
    minus_y = streamfunction(pym)
    u += (plus_y - minus_y) / (2.0 * epsilon)
    v -= (plus_x - minus_x) / (2.0 * epsilon)
    omega = -(
        (plus_x - 2.0 * center + minus_x) / (epsilon * epsilon)
        + (plus_y - 2.0 * center + minus_y) / (epsilon * epsilon)
    )

    b_dot = 0.5 * cfg.chord * (cfg.thickness_max - cfg.thickness_min) * beta_dot
    area_rate = math.pi * a * b_dot
    body_orientation = -incidence
    ellipse_center = pivot + np.asarray(
        (
            0.25 * cfg.chord * math.cos(body_orientation),
            0.25 * cfg.chord * math.sin(body_orientation),
        ),
        dtype=np.float64,
    )
    relative = points.astype(np.float64, copy=False) - ellipse_center[None, :]
    radius_squared = np.sum(relative * relative, axis=1)
    # The singularity lies inside the solid.  The tiny guard only protects
    # accidental diagnostic queries at the pivot.
    source_scale = area_rate / (
        2.0 * math.pi * np.maximum(radius_squared, (0.1 * thickness) ** 2)
    )
    u += source_scale * relative[:, 0]
    v += source_scale * relative[:, 1]
    return (
        np.asarray(u, dtype=np.float32),
        np.asarray(v, dtype=np.float32),
        np.asarray(omega, dtype=np.float32),
    )


def raw_fields_jet2_gpu(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    point_batch: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, float]:
    """Evaluate every raw Potential mode with the minimal Jet2 observable."""

    started = time.perf_counter()
    count, modes = len(points), int(cfg.modes)
    raw_u = cp.empty((count, modes), dtype=cp.float32)
    raw_v = cp.empty_like(raw_u)
    raw_omega = cp.empty_like(raw_u)
    for start in range(0, count, point_batch):
        stop = min(start + point_batch, count)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, points[start:stop], beta, alpha, False
        ):
            index = int(mode.mode_index)
            raw_u[start:stop, index] = field.grad[:, 0, 1]
            raw_v[start:stop, index] = -field.grad[:, 0, 0]
            raw_omega[start:stop, index] = -(
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        print(f"[Jet2 fields] points {stop}/{count}", flush=True)
    cp.cuda.Stream.null.synchronize()
    return raw_u, raw_v, raw_omega, time.perf_counter() - started


def raw_combination_jet2(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    raw_coefficients: np.ndarray,
    point_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate one modal linear combination without materializing N-by-K fields."""

    started = time.perf_counter()
    coefficients = cp.asarray(np.asarray(raw_coefficients, dtype=np.float32))
    output_u = np.empty(len(points), dtype=np.float32)
    output_v = np.empty_like(output_u)
    output_omega = np.empty_like(output_u)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        local_u = cp.zeros(stop - start, dtype=cp.float32)
        local_v = cp.zeros_like(local_u)
        local_omega = cp.zeros_like(local_u)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, points[start:stop], beta, alpha, False
        ):
            index = int(mode.mode_index)
            coefficient = coefficients[index]
            local_u += coefficient * field.grad[:, 0, 1]
            local_v -= coefficient * field.grad[:, 0, 0]
            local_omega -= coefficient * (
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        output_u[start:stop] = cp.asnumpy(local_u)
        output_v[start:stop] = cp.asnumpy(local_v)
        output_omega[start:stop] = cp.asnumpy(local_omega)
        print(f"[Jet2 combination] points {stop}/{len(points)}", flush=True)
    del coefficients
    cp.get_default_memory_pool().free_all_blocks()
    return output_u, output_v, output_omega, time.perf_counter() - started


def build_dynamic_basis(
    model: Any,
    cfg: Any,
    beta: float,
    alpha: float,
    peeling_points: np.ndarray,
    dynamics_points: np.ndarray,
    domain_area: float,
    point_batch: int,
    label: str,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    started = time.perf_counter()
    objective_started = time.perf_counter()
    values, gradient_x, gradient_y = raw_objective_gpu(
        model, cfg, peeling_points, beta, alpha, "potential"
    )
    objective_seconds = time.perf_counter() - objective_started
    peeling_transform, _rayleigh, peeling_diag = build_transform_gpu(
        cfg, values, gradient_x, gradient_y, domain_area, "potential"
    )
    del values, gradient_x, gradient_y
    cp.get_default_memory_pool().free_all_blocks()

    raw_u, raw_v, raw_omega, field_seconds = raw_fields_jet2_gpu(
        model, cfg, dynamics_points, beta, alpha, point_batch
    )
    transform_gpu = cp.asarray(peeling_transform, dtype=cp.float32)
    peeled_u = raw_u @ transform_gpu
    peeled_v = raw_v @ transform_gpu
    peeled_omega = raw_omega @ transform_gpu
    del raw_u, raw_v, raw_omega, transform_gpu
    cp.get_default_memory_pool().free_all_blocks()
    weight = float(domain_area / len(dynamics_points))
    (
        basis_u,
        basis_v,
        basis_omega,
        eigenvalues,
        ritz_rotation,
        ritz_diag,
    ) = gpu_ritz(
        peeled_u,
        peeled_v,
        peeled_omega,
        weight,
        int(cfg.modes),
        label,
    )
    total_transform = peeling_transform @ ritz_rotation
    diagnostics = {
        "raw_potential_objective_seconds": objective_seconds,
        "raw_jet2_basis_seconds": field_seconds,
        "peeling": peeling_diag,
        "ritz": ritz_diag,
        "total_seconds": time.perf_counter() - started,
        "domain_area": domain_area,
        "quadrature_weight": weight,
    }
    return (
        basis_u,
        basis_v,
        basis_omega,
        eigenvalues,
        np.asarray(total_transform, dtype=np.float32),
        diagnostics,
    )


def interpolate_gt(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    query = np.column_stack((points[:, 1], points[:, 0]))
    interpolator = RegularGridInterpolator(
        (y, x), field, method="linear", bounds_error=True
    )
    return np.asarray(interpolator(query), dtype=np.float32)


def relative_velocity(
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
        actual_u[mask] ** 2 + actual_v[mask] ** 2, dtype=np.float64
    )
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def relative_scalar(
    actual: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> float:
    numerator = np.sum((predicted[mask] - actual[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(actual[mask] ** 2, dtype=np.float64)
    return float(math.sqrt(numerator / max(denominator, 1.0e-30)))


def project_initial_joint(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    eigenvalues: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    base: tuple[np.ndarray, np.ndarray, np.ndarray],
    weight: float,
    maximum_total_velocity_error: float,
) -> tuple[cp.ndarray, dict[str, Any]]:
    """Choose a frame-zero velocity/vorticity fit under a velocity constraint."""

    target_u = cp.asarray(actual_u - base[0], dtype=cp.float32)
    target_v = cp.asarray(actual_v - base[1], dtype=cp.float32)
    target_omega = cp.asarray(actual_omega - base[2], dtype=cp.float32)
    gpu_actual_u = cp.asarray(actual_u, dtype=cp.float32)
    gpu_actual_v = cp.asarray(actual_v, dtype=cp.float32)
    gpu_actual_omega = cp.asarray(actual_omega, dtype=cp.float32)
    gpu_base_u = cp.asarray(base[0], dtype=cp.float32)
    gpu_base_v = cp.asarray(base[1], dtype=cp.float32)
    gpu_base_omega = cp.asarray(base[2], dtype=cp.float32)
    scalar_weight = cp.float32(weight)
    velocity_rhs = scalar_weight * (
        basis_u.T @ target_u + basis_v.T @ target_v
    )
    vorticity_rhs = scalar_weight * (basis_omega.T @ target_omega)
    velocity_energy = scalar_weight * cp.sum(
        target_u * target_u + target_v * target_v, dtype=cp.float64
    )
    vorticity_energy = scalar_weight * cp.sum(
        target_omega * target_omega, dtype=cp.float64
    )
    balance_scale = float(
        (velocity_energy / cp.maximum(vorticity_energy, cp.float64(1.0e-30))).get()
    )
    stiffness = cp.asarray(eigenvalues, dtype=cp.float32)
    candidates: list[dict[str, Any]] = []
    coefficient_candidates: list[cp.ndarray] = []
    for normalized_weight in (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        eta = float(normalized_weight * balance_scale)
        coefficients = (
            velocity_rhs + cp.float32(eta) * vorticity_rhs
        ) / (cp.float32(1.0) + cp.float32(eta) * stiffness)
        predicted_u = gpu_base_u + basis_u @ coefficients
        predicted_v = gpu_base_v + basis_v @ coefficients
        predicted_omega = gpu_base_omega + basis_omega @ coefficients
        velocity_error = float(
            cp.sqrt(
                cp.sum(
                    (predicted_u - gpu_actual_u) ** 2
                    + (predicted_v - gpu_actual_v) ** 2,
                    dtype=cp.float64,
                )
                / cp.maximum(
                    cp.sum(
                        gpu_actual_u * gpu_actual_u + gpu_actual_v * gpu_actual_v,
                        dtype=cp.float64,
                    ),
                    cp.float64(1.0e-30),
                )
            ).get()
        )
        vorticity_error = float(
            cp.sqrt(
                cp.sum(
                    (predicted_omega - gpu_actual_omega) ** 2,
                    dtype=cp.float64,
                )
                / cp.maximum(
                    cp.sum(gpu_actual_omega * gpu_actual_omega, dtype=cp.float64),
                    cp.float64(1.0e-30),
                )
            ).get()
        )
        candidates.append(
            {
                "normalized_vorticity_weight": normalized_weight,
                "dimensional_vorticity_weight": eta,
                "total_velocity_error": velocity_error,
                "total_vorticity_error": vorticity_error,
            }
        )
        coefficient_candidates.append(coefficients)
    feasible = [
        index
        for index, candidate in enumerate(candidates)
        if candidate["total_velocity_error"] <= maximum_total_velocity_error
    ]
    selected = (
        min(feasible, key=lambda index: candidates[index]["total_vorticity_error"])
        if feasible
        else min(range(len(candidates)), key=lambda index: candidates[index]["total_velocity_error"])
    )
    output = coefficient_candidates[selected].copy()
    diagnostics = {
        "maximum_total_velocity_error": maximum_total_velocity_error,
        "normalization_velocity_to_vorticity_energy": balance_scale,
        "candidates": candidates,
        "selected_index": selected,
        "selected": candidates[selected],
    }
    del (
        target_u,
        target_v,
        target_omega,
        gpu_actual_u,
        gpu_actual_v,
        gpu_actual_omega,
        gpu_base_u,
        gpu_base_v,
        gpu_base_omega,
        velocity_rhs,
        vorticity_rhs,
        stiffness,
    )
    return output, diagnostics


def reconstruct_frame(
    model: Any,
    cfg: Any,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    beta: float,
    alpha: float,
    geometry_derivatives: tuple[float, float],
    total_transform: np.ndarray,
    coefficients: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xgrid, ygrid = np.meshgrid(x, y)
    fluid = ~solid
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    raw_coefficients = total_transform @ np.asarray(coefficients, dtype=np.float32)
    learned_u, learned_v, learned_omega, inference_seconds = raw_combination_jet2(
        model,
        cfg,
        points,
        beta,
        alpha,
        raw_coefficients,
        args.reconstruction_point_batch,
    )
    base_u, base_v, base_omega = analytic_base_fields(
        cfg,
        points,
        beta,
        alpha,
        geometry_derivatives[0],
        geometry_derivatives[1],
        inlet_speed=args.inlet_speed,
        cycle_steps=args.cycle_steps,
        coordinate_dt_per_lattice_step=args.coordinate_dt_per_lattice_step,
        fd_epsilon=args.base_fd_epsilon,
    )
    predicted_u = np.zeros(solid.shape, dtype=np.float32)
    predicted_v = np.zeros_like(predicted_u)
    predicted_omega = np.zeros_like(predicted_u)
    predicted_u[fluid] = base_u + learned_u
    predicted_v[fluid] = base_v + learned_v
    predicted_omega[fluid] = base_omega + learned_omega
    return predicted_u, predicted_v, predicted_omega, {
        "active_points": int(len(points)),
        "learned_combination_seconds": inference_seconds,
    }


def frame_metrics(
    cfg: Any,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    beta: float,
    alpha: float,
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    base: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid.ravel(), ygrid.ravel())).astype(np.float32)
    xb, yb = body_numpy(cfg, points, alpha)
    wake = (
        (xb >= cfg.chord)
        & (xb <= 4.0 * cfg.chord)
        & (np.abs(yb) <= 0.75 * cfg.chord)
    ).reshape(solid.shape) & fluid
    actual_u, actual_v, actual_omega = actual
    predicted_u, predicted_v, predicted_omega = predicted
    base_u, base_v, base_omega = base
    return {
        "velocity_error": relative_velocity(
            actual_u, actual_v, predicted_u, predicted_v, fluid
        ),
        "vorticity_error": relative_scalar(
            actual_omega, predicted_omega, fluid
        ),
        "fluctuation_velocity_error": relative_velocity(
            actual_u - base_u,
            actual_v - base_v,
            predicted_u - base_u,
            predicted_v - base_v,
            fluid,
        ),
        "fluctuation_vorticity_error": relative_scalar(
            actual_omega - base_omega,
            predicted_omega - base_omega,
            fluid,
        ),
        "wake_velocity_error": relative_velocity(
            actual_u, actual_v, predicted_u, predicted_v, wake
        ),
        "wake_vorticity_error": relative_scalar(
            actual_omega, predicted_omega, wake
        ),
    }


def validate_gt_contract(gt: dict[str, np.ndarray], cfg: Any) -> dict[str, Any]:
    metadata = json.loads(str(gt["metadata"]))
    if gt["u"].shape[0] != 41 or gt["solid"].shape != gt["u"].shape:
        raise ValueError("dynamic GT must contain 41 matching field/mask states")
    phases = np.asarray(gt["phase"], dtype=np.float64)
    # The saved endpoint closes the cycle and is archived as phase zero.
    expected = (np.arange(41, dtype=np.float64) / 40.0) % 1.0
    if not np.allclose(phases, expected, rtol=0.0, atol=5.0e-8):
        raise ValueError("unexpected GT phase schedule")
    beta_expected = []
    alpha_expected = []
    for phase in phases:
        beta, alpha, _dbeta, _dalpha = periodic_geometry_state(float(phase))
        beta_expected.append(beta)
        alpha_expected.append(alpha)
    if not np.allclose(gt["beta"], beta_expected, rtol=0.0, atol=1.0e-7):
        raise ValueError("GT beta trajectory differs from ROM contract")
    if not np.allclose(gt["alpha_deg"], alpha_expected, rtol=0.0, atol=3.0e-6):
        raise ValueError("GT alpha trajectory differs from ROM contract")
    geometry = metadata["geometry"]
    for key in (
        "xmin", "xmax", "ymin", "ymax", "chord", "quarter_chord_x",
        "quarter_chord_y", "thickness_min", "thickness_max",
    ):
        if not math.isclose(float(getattr(cfg, key)), float(geometry[key]), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"GT/checkpoint geometry mismatch for {key}")
    return metadata


def run(args: argparse.Namespace) -> None:
    overall_started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg, model, checkpoint_metadata = load_hardcoded_model(args.checkpoint)
    if int(cfg.modes) != args.modes or int(checkpoint_metadata["epoch"]) != 2000:
        raise ValueError("this experiment requires the frozen epoch2000 K2048 checkpoint")
    if str(cfg.basis_objective) != "streamfunction_dirichlet_rayleigh":
        raise ValueError("this experiment requires the Potential--Rayleigh checkpoint")
    with np.load(args.gt, allow_pickle=False) as archive:
        gt = {name: archive[name] for name in archive.files}
    gt_metadata = validate_gt_contract(gt, cfg)
    x_full = np.asarray(gt["x"], dtype=np.float32)
    y_full = np.asarray(gt["y"], dtype=np.float32)
    stride = int(args.grid_stride)
    x = x_full[::stride]
    y = y_full[::stride]
    output_dt = float(args.steps_per_frame) * float(args.coordinate_dt_per_lattice_step)
    physical_viscosity = (
        float(gt_metadata["lattice_viscosity"])
        * float(args.coordinate_dt_per_lattice_step)
    )
    domain_box_area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)

    write_json(
        args.output_dir / "manifest.json",
        {
            "started_at": now_iso(),
            "method": "parameter-dependent Potential--Rayleigh ALE ROM",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "gt": str(args.gt.resolve()),
            "modes": args.modes,
            "states": args.states,
            "geometry_basis_inferences": args.states,
            "first_frame": "GT velocity projection",
            "later_frames": "no GT in propagation; GT is metrics-only",
            "coefficient_transport": "mass-overlap/minimum-energy projection of old total velocity into new affine basis",
            "basis_time_discretization": "one MLP inference per saved geometry state; first-order output-frame ALE split",
            "moving_boundary_lifting": [
                "irrotational conformal-map potential flow around the current ellipse",
                "compact rigid-pitch streamfunction about quarter chord",
                "irrotational interior-source net-flux lifting for thickness change",
            ],
            "peeling_points": args.peeling_points,
            "dynamics_points": args.dynamics_points,
            "substeps_per_frame": args.substeps_per_frame,
            "output_dt": output_dt,
            "viscosity": physical_viscosity,
            "grid_stride": stride,
        },
    )

    predicted_u_frames: list[np.ndarray] = []
    predicted_v_frames: list[np.ndarray] = []
    predicted_omega_frames: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    previous_total_transform: np.ndarray | None = None
    previous_coefficients: np.ndarray | None = None
    previous_geometry: tuple[float, float, float, float] | None = None
    start_frame = 0
    latest_state_path = args.output_dir / "latest_transport_state.npz"
    if args.resume and latest_state_path.exists():
        with np.load(latest_state_path, allow_pickle=False) as archive:
            completed_frame = int(archive["frame"])
            previous_total_transform = archive["total_transform"].astype(np.float32)
            previous_coefficients = archive["coefficients"].astype(np.float32)
            previous_geometry = (
                float(archive["beta"]),
                float(archive["alpha_deg"]),
                float(archive["dbeta_dphase"]),
                float(archive["dalpha_dphase"]),
            )
        if completed_frame >= args.states:
            raise ValueError("resume state lies beyond requested --states")
        for frame in range(completed_frame + 1):
            frame_npz = args.output_dir / "frames" / f"frame_{frame:03d}.npz"
            frame_json = args.output_dir / "frames" / f"frame_{frame:03d}.json"
            if not frame_npz.exists() or not frame_json.exists():
                raise FileNotFoundError(
                    f"resume requires atomic frame pair for frame {frame}"
                )
            with np.load(frame_npz, allow_pickle=False) as archive:
                predicted_u_frames.append(archive["u"].astype(np.float32))
                predicted_v_frames.append(archive["v"].astype(np.float32))
                predicted_omega_frames.append(
                    archive["vorticity"].astype(np.float32)
                )
            frame_record = json.loads(frame_json.read_text(encoding="utf-8"))
            if not frame_record.get("complete"):
                raise ValueError(f"incomplete resume frame {frame}")
            metric_rows.append(frame_record["metrics"])
            timing_rows.append(frame_record["timings"])
        start_frame = completed_frame + 1
        print(
            f"[resume] reused states 0..{completed_frame}; continue at {start_frame}",
            flush=True,
        )

    for frame in range(start_frame, args.states):
        frame_started = time.perf_counter()
        phase = float(gt["phase"][frame])
        beta, alpha, dbeta, dalpha = periodic_geometry_state(phase)
        domain_area = float(domain_box_area - airfoil_area(cfg, beta))
        dynamics_points = fluid_points(
            cfg,
            args.dynamics_points,
            beta,
            alpha,
            args.random_seed + 10000 * frame + 101,
        )
        peeling_points = fluid_points(
            cfg,
            args.peeling_points,
            beta,
            alpha,
            args.random_seed + 10000 * frame + 2300001,
        )
        print(
            f"[frame {frame:02d}/{args.states - 1}] phase={phase:.3f} "
            f"beta={beta:.6f} alpha={alpha:.6f}",
            flush=True,
        )
        (
            basis_u,
            basis_v,
            basis_omega,
            eigenvalues,
            total_transform,
            basis_diag,
        ) = build_dynamic_basis(
            model,
            cfg,
            beta,
            alpha,
            peeling_points,
            dynamics_points,
            domain_area,
            args.basis_point_batch,
            f"dynamic Potential frame {frame:02d}",
        )
        weight = float(domain_area / len(dynamics_points))
        base = analytic_base_fields(
            cfg,
            dynamics_points,
            beta,
            alpha,
            dbeta,
            dalpha,
            inlet_speed=args.inlet_speed,
            cycle_steps=args.cycle_steps,
            coordinate_dt_per_lattice_step=args.coordinate_dt_per_lattice_step,
            fd_epsilon=args.base_fd_epsilon,
        )

        transport_started = time.perf_counter()
        old_field_inference_seconds = 0.0
        if frame == 0:
            actual_u = interpolate_gt(
                x_full, y_full, gt["u"][frame], dynamics_points
            )
            actual_v = interpolate_gt(
                x_full, y_full, gt["v"][frame], dynamics_points
            )
            actual_omega = interpolate_gt(
                x_full, y_full, gt["vorticity"][frame], dynamics_points
            )
            coefficients_gpu, initial_projection = project_initial_joint(
                basis_u,
                basis_v,
                basis_omega,
                eigenvalues,
                actual_u,
                actual_v,
                actual_omega,
                base,
                weight,
                args.maximum_initial_velocity_error,
            )
            coefficients = cp.asnumpy(coefficients_gpu)
            transfer_error = float(
                initial_projection["selected"]["total_velocity_error"]
            )
            del coefficients_gpu, actual_u, actual_v, actual_omega
            propagation = {
                "kind": "GT frame-zero constrained velocity/vorticity projection",
                "projection": initial_projection,
            }
            selected_projection = initial_projection["selected"]
            print(
                f"[frame {frame:02d} joint projection] "
                f"weight={selected_projection['normalized_vorticity_weight']:.3g} "
                f"velocity={100.0 * selected_projection['total_velocity_error']:.2f}% "
                f"vorticity={100.0 * selected_projection['total_vorticity_error']:.2f}%",
                flush=True,
            )
        else:
            assert previous_total_transform is not None
            assert previous_coefficients is not None
            assert previous_geometry is not None
            previous_beta, previous_alpha, previous_dbeta, previous_dalpha = previous_geometry
            raw_previous_coefficients = (
                previous_total_transform @ previous_coefficients
            )
            old_u, old_v, _old_omega, old_field_inference_seconds = raw_combination_jet2(
                model,
                cfg,
                dynamics_points,
                previous_beta,
                previous_alpha,
                raw_previous_coefficients,
                args.basis_point_batch,
            )
            old_base = analytic_base_fields(
                cfg,
                dynamics_points,
                previous_beta,
                previous_alpha,
                previous_dbeta,
                previous_dalpha,
                inlet_speed=args.inlet_speed,
                cycle_steps=args.cycle_steps,
                coordinate_dt_per_lattice_step=args.coordinate_dt_per_lattice_step,
                fd_epsilon=args.base_fd_epsilon,
            )
            target_u = old_base[0] + old_u - base[0]
            target_v = old_base[1] + old_v - base[1]
            transported_gpu, transfer_error = project_initial(
                basis_u, basis_v, target_u, target_v, weight
            )
            coefficient_frames, rollout_diag = rollout_coefficients(
                basis_u,
                basis_v,
                basis_omega,
                transported_gpu,
                base,
                eigenvalues,
                weight,
                physical_viscosity,
                output_dt,
                1,
                args.substeps_per_frame,
                None,
                f"dynamic Potential frame {frame:02d}",
                "frozen-base-subtracted",
            )
            coefficients = coefficient_frames[-1]
            del transported_gpu, old_u, old_v, _old_omega, old_base, target_u, target_v
            propagation = {
                "kind": "ALE overlap transport then one-frame implicit ROM advance",
                "transport_relative_error": transfer_error,
                "old_field_inference_seconds": old_field_inference_seconds,
                "rollout": rollout_diag,
            }
        transport_seconds = time.perf_counter() - transport_started

        solid = np.asarray(gt["solid"][frame, ::stride, ::stride], dtype=bool)
        predicted_u, predicted_v, predicted_omega, reconstruction_diag = reconstruct_frame(
            model,
            cfg,
            x,
            y,
            solid,
            beta,
            alpha,
            (dbeta, dalpha),
            total_transform,
            coefficients,
            args,
        )
        actual_u = np.asarray(gt["u"][frame, ::stride, ::stride], dtype=np.float32)
        actual_v = np.asarray(gt["v"][frame, ::stride, ::stride], dtype=np.float32)
        actual_omega = np.asarray(
            gt["vorticity"][frame, ::stride, ::stride], dtype=np.float32
        )
        xgrid, ygrid = np.meshgrid(x, y)
        grid_points = np.column_stack((xgrid.ravel(), ygrid.ravel())).astype(np.float32)
        grid_base_flat = analytic_base_fields(
            cfg,
            grid_points,
            beta,
            alpha,
            dbeta,
            dalpha,
            inlet_speed=args.inlet_speed,
            cycle_steps=args.cycle_steps,
            coordinate_dt_per_lattice_step=args.coordinate_dt_per_lattice_step,
            fd_epsilon=args.base_fd_epsilon,
        )
        grid_base = tuple(value.reshape(solid.shape) for value in grid_base_flat)
        metrics = frame_metrics(
            cfg,
            x,
            y,
            solid,
            beta,
            alpha,
            (actual_u, actual_v, actual_omega),
            (predicted_u, predicted_v, predicted_omega),
            grid_base,
        )
        frame_seconds = time.perf_counter() - frame_started
        row = {
            "frame": frame,
            "phase": phase,
            "beta": beta,
            "alpha_deg": alpha,
            **metrics,
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "transport_error": float(transfer_error),
        }
        metric_rows.append(row)
        timing_rows.append(
            {
                "frame": frame,
                "basis_generation_seconds": float(basis_diag["total_seconds"]),
                "coefficient_transport_and_advance_seconds": transport_seconds,
                "old_basis_field_inference_seconds": old_field_inference_seconds,
                "reconstruction_seconds": float(
                    reconstruction_diag["learned_combination_seconds"]
                ),
                "frame_wall_seconds": frame_seconds,
            }
        )
        predicted_u_frames.append(predicted_u)
        predicted_v_frames.append(predicted_v)
        predicted_omega_frames.append(predicted_omega)
        save_npz_atomic(
            args.output_dir / "frames" / f"frame_{frame:03d}.npz",
            u=predicted_u,
            v=predicted_v,
            vorticity=predicted_omega,
            solid=solid,
            coefficients=np.asarray(coefficients, dtype=np.float32),
            phase=np.asarray(phase),
            beta=np.asarray(beta),
            alpha_deg=np.asarray(alpha),
        )
        write_json(
            args.output_dir / "frames" / f"frame_{frame:03d}.json",
            {
                "complete": True,
                "completed_at": now_iso(),
                "metrics": row,
                "timings": timing_rows[-1],
                "basis": basis_diag,
                "propagation": propagation,
                "reconstruction": reconstruction_diag,
            },
        )
        save_npz_atomic(
            args.output_dir / "latest_transport_state.npz",
            frame=np.asarray(frame, dtype=np.int64),
            total_transform=total_transform,
            coefficients=np.asarray(coefficients, dtype=np.float32),
            beta=np.asarray(beta),
            alpha_deg=np.asarray(alpha),
            dbeta_dphase=np.asarray(dbeta),
            dalpha_dphase=np.asarray(dalpha),
        )
        write_json(
            args.output_dir / "state.json",
            {
                "status": "running",
                "updated_at": now_iso(),
                "completed_states": frame + 1,
                "total_states": args.states,
                "current": row,
                "basis": basis_diag,
                "propagation": propagation,
                "reconstruction": reconstruction_diag,
                "frame_wall_seconds": frame_seconds,
            },
        )
        print(
            f"[frame {frame:02d}] vel={100.0 * metrics['velocity_error']:.2f}% "
            f"vort={100.0 * metrics['vorticity_error']:.2f}% "
            f"transport={100.0 * transfer_error:.2f}% wall={frame_seconds:.1f}s",
            flush=True,
        )
        previous_total_transform = total_transform
        previous_coefficients = np.asarray(coefficients, dtype=np.float32)
        previous_geometry = (beta, alpha, dbeta, dalpha)
        del basis_u, basis_v, basis_omega, eigenvalues, grid_base_flat, grid_base
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    fields_path = args.output_dir / "dynamic_potential_rollout_fields_stride2.npz"
    save_npz_atomic(
        fields_path,
        x=x,
        y=y,
        phase=np.asarray(gt["phase"][: args.states]),
        beta=np.asarray(gt["beta"][: args.states]),
        alpha_deg=np.asarray(gt["alpha_deg"][: args.states]),
        u=np.stack(predicted_u_frames),
        v=np.stack(predicted_v_frames),
        vorticity=np.stack(predicted_omega_frames),
        solid=np.asarray(gt["solid"][: args.states, ::stride, ::stride], dtype=bool),
    )
    for filename, rows in (
        ("temporal_metrics.csv", metric_rows),
        ("timings_by_frame.csv", timing_rows),
    ):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    error_keys = [key for key in metric_rows[0] if key.endswith("_error")]
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in metric_rows])),
            "p95": float(np.percentile([row[key] for row in metric_rows], 95.0)),
            "max": float(np.max([row[key] for row in metric_rows])),
            "final": float(metric_rows[-1][key]),
        }
        for key in error_keys
    }
    summary = {
        "status": "completed",
        "completed_at": now_iso(),
        "states": args.states,
        "aggregate_errors": aggregate,
        "total_wall_seconds": time.perf_counter() - overall_started,
        "total_basis_generation_seconds": float(
            np.sum([row["basis_generation_seconds"] for row in timing_rows])
        ),
        "total_rom_transport_and_advance_seconds": float(
            np.sum([row["coefficient_transport_and_advance_seconds"] for row in timing_rows])
        ),
        "total_reconstruction_seconds": float(
            np.sum([row["reconstruction_seconds"] for row in timing_rows])
        ),
        "fields": str(fields_path.resolve()),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "state.json",
        {"status": "completed", "completed_at": now_iso(), **summary},
    )
    print(json.dumps(strict_json(summary), indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=2048)
    parser.add_argument("--states", type=int, default=41)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--dynamics-points", type=int, default=32768)
    parser.add_argument("--basis-point-batch", type=int, default=4096)
    parser.add_argument("--reconstruction-point-batch", type=int, default=4096)
    parser.add_argument("--substeps-per-frame", type=int, default=10)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument("--cycle-steps", type=int, default=20000)
    parser.add_argument("--coordinate-dt-per-lattice-step", type=float, default=0.01)
    parser.add_argument("--inlet-speed", type=float, default=0.08)
    parser.add_argument("--base-fd-epsilon", type=float, default=0.0025)
    parser.add_argument("--grid-stride", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument("--maximum-initial-velocity-error", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.states < 1 or args.states > 41:
        parser.error("--states must be in [1, 41]")
    if args.modes != 2048:
        parser.error("the production contract requires --modes 2048")
    return args


if __name__ == "__main__":
    run(parse_args())
