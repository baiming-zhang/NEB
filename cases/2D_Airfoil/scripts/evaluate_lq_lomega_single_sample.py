#!/usr/bin/env python3
"""Fair single-sample learned-basis reconstruction benchmark.

The script is intentionally run in a fresh Python process for each
formulation because the historical L_omega and current L_q checkpoints use
different versions of the airfoil backend.  Both processes derive their
basis and projection subsets from the same GT fluid-grid ordering.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


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
    denominator = np.sum(
        target_u[mask] ** 2 + target_v[mask] ** 2,
        dtype=np.float64,
    )
    return float(np.sqrt(numerator / max(float(denominator), 1.0e-30)))


def relative_scalar(
    target: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum((estimate[mask] - target[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(target[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(float(denominator), 1.0e-30)))


def load_case(path: Path, frame: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if frame < 0 or frame >= archive["u"].shape[0]:
            raise ValueError(f"frame {frame} is outside [0, {archive['u'].shape[0]})")
        return {
            "x": archive["x"].astype(np.float32),
            "y": archive["y"].astype(np.float32),
            "solid": archive["solid"].astype(bool),
            "metadata": json.loads(str(archive["metadata"])),
            "steps": archive["steps"].astype(np.int64),
            "actual_u": archive["u"][frame].astype(np.float32),
            "actual_v": archive["v"][frame].astype(np.float32),
            "actual_omega": archive["vorticity"][frame].astype(np.float32),
            "mean_u": np.mean(archive["u"], axis=0, dtype=np.float64).astype(np.float32),
            "mean_v": np.mean(archive["v"], axis=0, dtype=np.float64).astype(np.float32),
            "mean_omega": np.mean(archive["vorticity"], axis=0, dtype=np.float64).astype(np.float32),
            "snapshot_count": int(archive["u"].shape[0]),
        }


def load_model(checkpoint: Path, formulation: str) -> tuple[dict[str, Any], Any, Any]:
    from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: PLC0415
        HardcodedModel,
        cfg_namespace,
    )

    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    expected = (
        "streamfunction_dirichlet_rayleigh"
        if formulation == "lq"
        else "pure_vorticity_rayleigh"
    )
    if str(cfg.basis_objective) != expected:
        raise ValueError(f"expected {expected}, got {cfg.basis_objective}")
    if int(cfg.modes) != 1024:
        raise ValueError(f"expected K=1024, got {cfg.modes}")
    if formulation == "lomega":
        # This is the historical postprocessing schedule used for the wider
        # Jet3 network.  It changes memory scheduling only.
        cfg.mode_forward_batch = min(int(getattr(cfg, "mode_forward_batch", 1)), 64)
    return metadata, cfg, HardcodedModel(cfg, arrays)


def basis_components(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    formulation: str,
    point_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = int(cfg.modes)
    value = np.empty((len(points), modes), dtype=np.float32)
    gradient_x = np.empty_like(value)
    gradient_y = np.empty_like(value)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        for fallback, (network, field, _cache) in enumerate(
            model.forward_stream(cfg, points[start:stop], beta, alpha, False)
        ):
            index = int(getattr(network, "mode_index", fallback))
            if formulation == "lq":
                value[start:stop, index] = cp.asnumpy(field.value[:, 0])
                gradient_x[start:stop, index] = cp.asnumpy(field.grad[:, 0, 0])
                gradient_y[start:stop, index] = cp.asnumpy(field.grad[:, 0, 1])
            else:
                value[start:stop, index] = cp.asnumpy(
                    -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
                )
                gradient_x[start:stop, index] = cp.asnumpy(
                    -(field.third[:, 0, 0] + field.third[:, 0, 2])
                )
                gradient_y[start:stop, index] = cp.asnumpy(
                    -(field.third[:, 0, 1] + field.third[:, 0, 3])
                )
    return value, gradient_x, gradient_y


def raw_uvomega(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
    point_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    modes = int(cfg.modes)
    u = np.empty((len(points), modes), dtype=np.float32)
    v = np.empty_like(u)
    omega = np.empty_like(u)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        for fallback, (network, field, _cache) in enumerate(
            model.forward_stream(cfg, points[start:stop], beta, alpha, False)
        ):
            index = int(getattr(network, "mode_index", fallback))
            u[start:stop, index] = cp.asnumpy(field.grad[:, 0, 1])
            v[start:stop, index] = cp.asnumpy(-field.grad[:, 0, 0])
            omega[start:stop, index] = cp.asnumpy(
                -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
            )
    return u, v, omega


def peeled_transform(
    cfg: Any,
    value: np.ndarray,
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    domain_area: float,
    formulation: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    value_gpu = cp.asarray(value)
    gx_gpu = cp.asarray(gradient_x)
    gy_gpu = cp.asarray(gradient_y)
    count, modes = value.shape
    weight = cp.asarray(np.float32(domain_area / count), dtype=cp.float32)
    raw_mass = weight * cp.sum(value_gpu * value_gpu, axis=0)
    raw_stiffness = weight * cp.sum(
        gx_gpu * gx_gpu + gy_gpu * gy_gpu,
        axis=0,
    )
    raw_rayleigh = cp.asnumpy(raw_stiffness / raw_mass).astype(np.float64)
    if formulation == "lq":
        order = np.argsort(raw_rayleigh, kind="stable").tolist()
        ordering = "geometry_rayleigh"
    else:
        order = list(range(modes))
        ordering = "trained_causal_network_order"
    passes = int(getattr(cfg, "reorthogonalization_passes", 2))
    mass_floor = float(getattr(cfg, "mass_floor", 1.0e-14))
    basis = cp.empty_like(value_gpu)
    transform = cp.zeros((modes, modes), dtype=cp.float32)
    retained_networks: list[int] = []
    rejected: list[dict[str, float | int]] = []
    pre_normalization_masses: list[float] = []
    for network_index in order:
        rank = len(retained_networks)
        residual = value_gpu[:, network_index].copy()
        coefficients = cp.zeros(modes, dtype=cp.float32)
        coefficients[network_index] = 1.0
        if rank:
            previous = basis[:, :rank]
            previous_transform = transform[:, :rank]
            for _ in range(passes):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= mass_floor:
            rejected.append({"network_1based": network_index + 1, "mass": mass_value})
            continue
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis[:, rank] = residual * inverse_norm
        transform[:, rank] = coefficients * inverse_norm
        retained_networks.append(network_index)
        pre_normalization_masses.append(mass_value)
    rank = len(retained_networks)
    if rank == 0:
        raise FloatingPointError("all learned columns were unresolved")
    basis = basis[:, :rank]
    gram = weight * (basis.T @ basis)
    orthogonality = float(
        cp.max(cp.abs(gram - cp.eye(rank, dtype=cp.float32))).get()
    )
    result = cp.asnumpy(transform[:, :rank])
    diagnostics = {
        "peeled_rank": rank,
        "requested_modes": modes,
        "basis_metric": "streamfunction_l2" if formulation == "lq" else "vorticity_l2",
        "stiffness_metric": "velocity_l2" if formulation == "lq" else "grad_vorticity_l2",
        "causal_order": ordering,
        "reorthogonalization_passes": passes,
        "mass_floor": mass_floor,
        "peeled_orthogonality_max_abs": orthogonality,
        "minimum_retained_pre_normalization_mass": float(min(pre_normalization_masses)),
        "rejected_columns": rejected,
        "raw_rayleigh_min": float(np.nanmin(raw_rayleigh)),
        "raw_rayleigh_median": float(np.nanmedian(raw_rayleigh)),
        "raw_rayleigh_max": float(np.nanmax(raw_rayleigh)),
        "retained_networks_1based": [index + 1 for index in retained_networks],
    }
    del value_gpu, gx_gpu, gy_gpu, basis, gram, transform
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return result, diagnostics


def project_velocity_raw(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, int, float, dict[str, float]]:
    u64 = raw_u.astype(np.float64)
    v64 = raw_v.astype(np.float64)
    target_u64 = target_u.astype(np.float64)
    target_v64 = target_v.astype(np.float64)
    transform64 = transform.astype(np.float64)
    raw_gram = u64.T @ u64 + v64.T @ v64
    raw_right = u64.T @ target_u64 + v64.T @ target_v64
    gram = transform64.T @ raw_gram @ transform64
    right = transform64.T @ raw_right
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
    retained = values > threshold
    peeled_coefficients = vectors[:, retained] @ (
        (vectors[:, retained].T @ right) / values[retained]
    )
    raw_coefficients = transform64 @ peeled_coefficients
    fitted_u = raw_u @ raw_coefficients
    fitted_v = raw_v @ raw_coefficients
    residual = relative_velocity(target_u, target_v, fitted_u, fitted_v)
    condition = float(values[retained][-1] / values[retained][0])
    diagnostics = {
        "velocity_gram_threshold": threshold,
        "velocity_gram_retained_min": float(values[retained][0]),
        "velocity_gram_retained_max": float(values[retained][-1]),
        "velocity_gram_condition": condition,
    }
    return raw_coefficients.astype(np.float64), int(np.count_nonzero(retained)), residual, diagnostics


def as_grid(values: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    result = np.zeros(fluid.shape, dtype=np.float32)
    result[fluid] = values.astype(np.float32)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulation", choices=("lq", "lomega"), required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--basis-points", type=int, default=32768)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument(
        "--reconstruction-stride",
        type=int,
        default=1,
        help=(
            "Evaluate coefficients on the full-grid projection subset, then "
            "reconstruct only every Nth Cartesian grid point.  Stride 2 "
            "matches fields_stride2.npz from the K64--K2048 evaluation."
        ),
    )
    args = parser.parse_args()
    if args.reconstruction_stride < 1:
        parser.error("--reconstruction-stride must be positive")

    sys.path.insert(0, str(args.case_root / "src"))
    from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: PLC0415
        airfoil_area,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "complete.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("complete"):
            print(json.dumps(completion, indent=2), flush=True)
            return

    total_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    stage = time.perf_counter()
    case = load_case(args.gt, args.frame)
    metadata, cfg, model = load_model(args.checkpoint, args.formulation)
    stage_seconds["load_checkpoint_and_gt"] = time.perf_counter() - stage

    beta = float(case["metadata"]["beta"])
    alpha = float(case["metadata"]["alpha_deg"])
    full_fluid = ~case["solid"]
    full_xgrid, full_ygrid = np.meshgrid(case["x"], case["y"])
    full_points = np.column_stack(
        (full_xgrid[full_fluid], full_ygrid[full_fluid])
    ).astype(np.float32)
    basis_indices = np.linspace(
        0,
        len(full_points) - 1,
        min(args.basis_points, len(full_points)),
        dtype=np.int64,
    )
    projection_indices = np.linspace(
        0,
        len(full_points) - 1,
        min(args.projection_points, len(full_points)),
        dtype=np.int64,
    )
    full_target_u = (case["actual_u"] - case["mean_u"])[full_fluid]
    full_target_v = (case["actual_v"] - case["mean_v"])[full_fluid]

    stride = args.reconstruction_stride
    reconstruction_x = case["x"][::stride]
    reconstruction_y = case["y"][::stride]
    reconstruction_solid = case["solid"][::stride, ::stride]
    fluid = ~reconstruction_solid
    xgrid, ygrid = np.meshgrid(reconstruction_x, reconstruction_y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    actual_u_grid = case["actual_u"][::stride, ::stride]
    actual_v_grid = case["actual_v"][::stride, ::stride]
    actual_omega_grid = case["actual_omega"][::stride, ::stride]
    mean_u_grid = case["mean_u"][::stride, ::stride]
    mean_v_grid = case["mean_v"][::stride, ::stride]
    mean_omega_grid = case["mean_omega"][::stride, ::stride]
    actual_u = actual_u_grid[fluid]
    actual_v = actual_v_grid[fluid]
    actual_omega = actual_omega_grid[fluid]
    mean_u = mean_u_grid[fluid]
    mean_v = mean_v_grid[fluid]
    mean_omega = mean_omega_grid[fluid]
    target_u = actual_u - mean_u
    target_v = actual_v - mean_v

    print(f"[{args.formulation}] basis fields on {len(basis_indices)} common points", flush=True)
    stage = time.perf_counter()
    value, gx, gy = basis_components(
        model,
        cfg,
        full_points[basis_indices],
        beta,
        alpha,
        args.formulation,
        args.field_batch,
    )
    stage_seconds["basis_field_evaluation"] = time.perf_counter() - stage
    print(f"[{args.formulation}] two-pass formulation-specific peeling", flush=True)
    stage = time.perf_counter()
    domain_area = (
        (float(cfg.xmax) - float(cfg.xmin)) * (float(cfg.ymax) - float(cfg.ymin))
        - airfoil_area(cfg, beta)
    )
    transform, peeling = peeled_transform(cfg, value, gx, gy, domain_area, args.formulation)
    stage_seconds["basis_peeling"] = time.perf_counter() - stage
    del value, gx, gy
    gc.collect()

    print(f"[{args.formulation}] projection fields on {len(projection_indices)} common points", flush=True)
    stage = time.perf_counter()
    raw_u, raw_v, raw_omega = raw_uvomega(
        model,
        cfg,
        full_points[projection_indices],
        beta,
        alpha,
        args.field_batch,
    )
    stage_seconds["projection_field_evaluation"] = time.perf_counter() - stage
    stage = time.perf_counter()
    raw_coefficients, projection_rank, projection_residual, projection_diagnostics = project_velocity_raw(
        raw_u,
        raw_v,
        full_target_u[projection_indices],
        full_target_v[projection_indices],
        transform,
    )
    stage_seconds["velocity_projection_solve"] = time.perf_counter() - stage
    del raw_u, raw_v, raw_omega, transform
    gc.collect()

    print(
        f"[{args.formulation}] stride-{stride} reconstruction on "
        f"{len(points)} fluid points",
        flush=True,
    )
    stage = time.perf_counter()
    fluctuation_u = np.empty(len(points), dtype=np.float32)
    fluctuation_v = np.empty(len(points), dtype=np.float32)
    fluctuation_omega = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), args.field_batch):
        stop = min(start + args.field_batch, len(points))
        raw_u, raw_v, raw_omega = raw_uvomega(
            model,
            cfg,
            points[start:stop],
            beta,
            alpha,
            args.field_batch,
        )
        fluctuation_u[start:stop] = raw_u @ raw_coefficients
        fluctuation_v[start:stop] = raw_v @ raw_coefficients
        fluctuation_omega[start:stop] = raw_omega @ raw_coefficients
        del raw_u, raw_v, raw_omega
    estimate_u = mean_u + fluctuation_u
    estimate_v = mean_v + fluctuation_v
    estimate_omega = mean_omega + fluctuation_omega
    stage_seconds["full_grid_reconstruction"] = time.perf_counter() - stage

    # Define this mask directly from the shared GT coordinates, not through
    # either version of the backend.
    angle = math.radians(-alpha)
    cosine, sine = math.cos(angle), math.sin(angle)
    dx = points[:, 0] - float(cfg.quarter_chord_x)
    dy = points[:, 1] - float(cfg.quarter_chord_y)
    body_x = cosine * dx + sine * dy + 0.25 * float(cfg.chord)
    body_y = -sine * dx + cosine * dy
    wake = (
        (body_x >= float(cfg.chord))
        & (body_x <= 4.0 * float(cfg.chord))
        & (np.abs(body_y) <= 0.75 * float(cfg.chord))
    )
    if not np.any(wake):
        raise RuntimeError("shared wake mask is empty")
    fluctuation_target_omega = actual_omega - mean_omega
    metrics = {
        "velocity_error": relative_velocity(actual_u, actual_v, estimate_u, estimate_v),
        "vorticity_error": relative_scalar(actual_omega, estimate_omega),
        "fluctuation_velocity_error": relative_velocity(target_u, target_v, fluctuation_u, fluctuation_v),
        "fluctuation_vorticity_error": relative_scalar(fluctuation_target_omega, fluctuation_omega),
        "wake_velocity_error": relative_velocity(actual_u, actual_v, estimate_u, estimate_v, wake),
        "wake_vorticity_error": relative_scalar(actual_omega, estimate_omega, wake),
    }

    stage = time.perf_counter()
    archive_path = args.output_dir / "reconstruction.npz"
    np.savez_compressed(
        archive_path,
        x=reconstruction_x,
        y=reconstruction_y,
        solid=reconstruction_solid,
        frame=np.int32(args.frame),
        basis_indices=basis_indices,
        projection_indices=projection_indices,
        wake=as_grid(wake.astype(np.float32), fluid).astype(bool),
        actual_u=actual_u_grid,
        actual_v=actual_v_grid,
        actual_omega=actual_omega_grid,
        mean_u=mean_u_grid,
        mean_v=mean_v_grid,
        mean_omega=mean_omega_grid,
        estimate_u=as_grid(estimate_u, fluid),
        estimate_v=as_grid(estimate_v, fluid),
        estimate_omega=as_grid(estimate_omega, fluid),
        raw_coefficients=raw_coefficients.astype(np.float32),
    )
    stage_seconds["metrics_and_output"] = time.perf_counter() - stage
    total_seconds = time.perf_counter() - total_started
    result = {
        "complete": True,
        "completed_at": now(),
        "formulation": args.formulation,
        "objective": str(cfg.basis_objective),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(metadata.get("epoch", metadata.get("optimizer_step", -1))),
        "configured_epochs": int(getattr(cfg, "epochs", -1)),
        "gt": str(args.gt.resolve()),
        "beta": beta,
        "alpha_deg": alpha,
        "frame": args.frame,
        "step": int(case["steps"][args.frame]),
        "snapshot_count_for_mean": case["snapshot_count"],
        "source_full_fluid_points": int(len(full_points)),
        "reconstruction_grid_stride": int(stride),
        "fluid_points": int(len(points)),
        "basis_points": int(len(basis_indices)),
        "projection_points": int(len(projection_indices)),
        "field_batch": args.field_batch,
        "coefficient_fit": "velocity_l2",
        "wake_mask": "body_x in [1c,4c], |body_y| <= 0.75c",
        "wake_fluid_points": int(np.count_nonzero(wake)),
        "peeling": peeling,
        "projection_rank": projection_rank,
        "projection_relative_residual": projection_residual,
        "projection_diagnostics": projection_diagnostics,
        "metrics": metrics,
        "stage_wall_seconds": stage_seconds,
        "basis_preprocessing_wall_seconds": (
            stage_seconds["basis_field_evaluation"] + stage_seconds["basis_peeling"]
        ),
        "online_reconstruction_wall_seconds": (
            stage_seconds["projection_field_evaluation"]
            + stage_seconds["velocity_projection_solve"]
            + stage_seconds["full_grid_reconstruction"]
        ),
        "total_evaluation_wall_seconds": total_seconds,
        "archive": str(archive_path.resolve()),
        "network": {
            "hidden": int(cfg.hidden),
            "hidden_layers": int(cfg.hidden_layers),
            "spatial_derivative_order": int(cfg.spatial_derivative_order),
            "mode_forward_batch": int(getattr(cfg, "mode_forward_batch", 1)),
        },
    }
    write_json(completion_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
