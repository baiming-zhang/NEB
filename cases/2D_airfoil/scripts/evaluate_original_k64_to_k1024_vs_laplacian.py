#!/usr/bin/env python3
"""Evaluate a requested Original K subset and nested airfoil-aware FD Laplacian bases.

Three fixed unseen airfoil geometries are evaluated at the requested frames.
For every geometry, all learned bases use the same geometry-specific IID
peeling points and all ten reconstructions use the same velocity-projection
samples and temporal mean. One geometry-aware FD eigensolve at the largest
requested K is reused as nested bases for every requested K.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import scipy.interpolate


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    airfoil_area,
    body_numpy,
    cfg_namespace,
    fluid_points,
)
from evaluate_unseen_k64_vs_laplacian import (  # noqa: E402
    BLUE,
    RED,
    add_solid,
    as_grid,
    geometry_laplacian,
    peeled_transform,
    relative_scalar,
    relative_velocity,
    velocity_projection,
    write_json,
)


SUPPORTED_K_VALUES = (64, 128, 256, 512, 1024, 2048)
K_VALUES = SUPPORTED_K_VALUES
FRAMES = (30,)
CASES = (
    ("up", "gt_019.npz"),
    ("middle", "gt_015.npz"),
    ("down", "gt_014.npz"),
)
LEARNED_COLOR = BLUE
LAPLACIAN_COLOR = RED


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_model_any(checkpoint: Path) -> tuple[dict[str, Any], Any, HardcodedModel]:
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    if cfg.basis_objective != "streamfunction_dirichlet_rayleigh":
        raise ValueError(f"{checkpoint} has the wrong objective")
    if int(metadata["epoch"]) != 2000:
        raise ValueError(f"{checkpoint} is epoch {metadata['epoch']}, expected 2000")
    return metadata, cfg, HardcodedModel(cfg, arrays)


def validate_checkpoint_contract(checkpoints: dict[int, Path]) -> dict[int, dict[str, Any]]:
    configs: dict[int, dict[str, Any]] = {}
    base: dict[str, Any] | None = None
    for modes in K_VALUES:
        sidecar = json.loads(checkpoints[modes].with_suffix(".json").read_text(encoding="utf-8"))
        config = sidecar["config"]
        if int(sidecar["epoch"]) != 2000 or int(config["modes"]) != modes:
            raise ValueError(f"K{modes} checkpoint metadata mismatch")
        required = {
            "epochs": 2000,
            "hidden": 32,
            "hidden_layers": 4,
            "siren_first_omega0": 8.0,
            "siren_hidden_omega0": 30.0,
            "points_per_step": 4096,
            "base_learning_rate": 2.0e-4,
            "min_learning_rate": 1.0e-6,
            "grad_clip": 5.0,
            "endpoint_period": 20,
            "reorthogonalization_passes": 2,
            "basis_objective": "streamfunction_dirichlet_rayleigh",
            "causal_sorting": "geometry_rayleigh",
            "d4_symmetry_augmentation": False,
            "analytic_eigenmodes": False,
            "degenerate_block_direction_alignment": False,
            "rank_reporting": "peeled_only",
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in required.items()
            if config.get(key) != value
        }
        if mismatches:
            raise ValueError(f"K{modes} is not Original: {mismatches}")
        if base is None:
            base = config
        else:
            differences = {
                key
                for key in set(base) | set(config)
                if base.get(key) != config.get(key)
            }
            if not differences.issubset({"modes", "result_root"}):
                raise ValueError(f"K{modes} differs in {sorted(differences)}")
        configs[modes] = config
    return configs


def raw_psi_grad(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(points), int(cfg.modes))
    psi = np.empty(shape, dtype=np.float32)
    gx = np.empty(shape, dtype=np.float32)
    gy = np.empty(shape, dtype=np.float32)
    for network, field, _cache in model.forward_stream(cfg, points, beta, alpha, False):
        index = int(network.mode_index)
        psi[:, index] = cp.asnumpy(field.value[:, 0])
        gx[:, index] = cp.asnumpy(field.grad[:, 0, 0])
        gy[:, index] = cp.asnumpy(field.grad[:, 0, 1])
    return psi, gx, gy


def raw_uvomega(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(points), int(cfg.modes))
    u = np.empty(shape, dtype=np.float32)
    v = np.empty(shape, dtype=np.float32)
    omega = np.empty(shape, dtype=np.float32)
    for network, field, _cache in model.forward_stream(cfg, points, beta, alpha, False):
        index = int(network.mode_index)
        u[:, index] = cp.asnumpy(field.grad[:, 0, 1])
        v[:, index] = -cp.asnumpy(field.grad[:, 0, 0])
        omega[:, index] = cp.asnumpy(-(field.hess[:, 0, 0] + field.hess[:, 0, 2]))
    return u, v, omega


def learned_state(
    checkpoint: Path,
    state_dir: Path,
    beta: float,
    alpha: float,
    peeling_points: int,
    seed_offset: int,
) -> tuple[dict[str, Any], Any, HardcodedModel, np.ndarray, np.ndarray, dict[str, Any]]:
    metadata, cfg, model = load_model_any(checkpoint)
    modes = int(cfg.modes)
    cache = state_dir / f"learned_k{modes:04d}_state.npz"
    diagnostics_path = state_dir / f"learned_k{modes:04d}_state.json"
    if cache.exists() and diagnostics_path.exists():
        with np.load(cache, allow_pickle=False) as archive:
            transform = archive["transform"].astype(np.float32)
            rayleigh = archive["rayleigh"].astype(np.float64)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if transform.shape != (modes, modes) or len(rayleigh) != modes:
            raise ValueError(f"invalid cached K{modes} learned state")
        return metadata, cfg, model, transform, rayleigh, diagnostics
    integration = fluid_points(cfg, peeling_points, beta, alpha, int(cfg.seed) + seed_offset)
    psi, gx, gy = raw_psi_grad(model, cfg, integration, beta, alpha)
    area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(cfg, beta)
    transform, rayleigh, diagnostics = peeled_transform(cfg, psi, gx, gy, area)
    np.savez(cache, transform=transform, rayleigh=rayleigh)
    write_json(diagnostics_path, diagnostics)
    del integration, psi, gx, gy
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return metadata, cfg, model, transform, rayleigh, diagnostics


def load_case(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        steps = archive["steps"].astype(np.int64)
        if max(FRAMES) >= len(steps):
            raise ValueError(f"{path} has only {len(steps)} frames")
        result = {
            "x": archive["x"].astype(np.float32),
            "y": archive["y"].astype(np.float32),
            "solid": archive["solid"].astype(bool),
            "steps": steps,
            "metadata": json.loads(str(archive["metadata"])),
            "actual_u": np.stack([archive["u"][frame] for frame in FRAMES]).astype(np.float32),
            "actual_v": np.stack([archive["v"][frame] for frame in FRAMES]).astype(np.float32),
            "actual_omega": np.stack([archive["vorticity"][frame] for frame in FRAMES]).astype(np.float32),
            "mean_u": np.mean(archive["u"], axis=0, dtype=np.float64).astype(np.float32),
            "mean_v": np.mean(archive["v"], axis=0, dtype=np.float64).astype(np.float32),
            "mean_omega": np.mean(archive["vorticity"], axis=0, dtype=np.float64).astype(np.float32),
        }
    return result


def project_three_frames(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    targets_u: np.ndarray,
    targets_v: np.ndarray,
) -> tuple[np.ndarray, list[int], list[float]]:
    coefficients = []
    ranks = []
    residuals = []
    for frame_index in range(len(FRAMES)):
        coefficient, rank, residual = velocity_projection(
            basis_u,
            basis_v,
            targets_u[frame_index],
            targets_v[frame_index],
        )
        coefficients.append(coefficient)
        ranks.append(rank)
        residuals.append(residual)
    return np.column_stack(coefficients), ranks, residuals


def full_metrics(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    estimate_omega: np.ndarray,
    wake: np.ndarray,
) -> list[dict[str, float]]:
    results = []
    for index in range(len(FRAMES)):
        target_u = actual_u[index] - mean_u
        target_v = actual_v[index] - mean_v
        target_omega = actual_omega[index] - mean_omega
        results.append(
            {
                "velocity_error": relative_velocity(actual_u[index], actual_v[index], estimate_u[index], estimate_v[index]),
                "vorticity_error": relative_scalar(actual_omega[index], estimate_omega[index]),
                "fluctuation_velocity_error": relative_velocity(target_u, target_v, estimate_u[index] - mean_u, estimate_v[index] - mean_v),
                "fluctuation_vorticity_error": relative_scalar(target_omega, estimate_omega[index] - mean_omega),
                "wake_velocity_error": relative_velocity(actual_u[index], actual_v[index], estimate_u[index], estimate_v[index], wake),
                "wake_vorticity_error": relative_scalar(actual_omega[index], estimate_omega[index], wake),
            }
        )
    return results


def learned_reconstruction(
    checkpoint: Path,
    state_dir: Path,
    points: np.ndarray,
    subset: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    wake: np.ndarray,
    beta: float,
    alpha: float,
    peeling_points: int,
    field_batch: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    total_started = time.perf_counter_ns()
    state_started = time.perf_counter_ns()
    metadata, cfg, model, transform, rayleigh, peeling = learned_state(
        checkpoint, state_dir, beta, alpha, peeling_points, 2_300_001
    )
    state_seconds = (time.perf_counter_ns() - state_started) / 1.0e9
    modes = int(cfg.modes)
    projection_started = time.perf_counter_ns()
    _raw_psi, raw_gx, raw_gy = raw_psi_grad(model, cfg, points[subset], beta, alpha)
    basis_u = raw_gy @ transform
    basis_v = -raw_gx @ transform
    coefficients, ranks, residuals = project_three_frames(
        basis_u, basis_v, target_u[:, subset], target_v[:, subset]
    )
    raw_coefficients = transform @ coefficients
    del _raw_psi, raw_gx, raw_gy, basis_u, basis_v
    projection_seconds = (time.perf_counter_ns() - projection_started) / 1.0e9
    field_started = time.perf_counter_ns()
    fluctuations_u = np.empty((len(FRAMES), len(points)), dtype=np.float32)
    fluctuations_v = np.empty_like(fluctuations_u)
    fluctuations_omega = np.empty_like(fluctuations_u)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        raw_u, raw_v, raw_omega = raw_uvomega(model, cfg, points[start:stop], beta, alpha)
        fluctuations_u[:, start:stop] = (raw_u @ raw_coefficients).T
        fluctuations_v[:, start:stop] = (raw_v @ raw_coefficients).T
        fluctuations_omega[:, start:stop] = (raw_omega @ raw_coefficients).T
        del raw_u, raw_v, raw_omega
    estimate_u = mean_u[None, :] + fluctuations_u
    estimate_v = mean_v[None, :] + fluctuations_v
    estimate_omega = mean_omega[None, :] + fluctuations_omega
    metrics = full_metrics(
        actual_u,
        actual_v,
        actual_omega,
        mean_u,
        mean_v,
        mean_omega,
        estimate_u,
        estimate_v,
        estimate_omega,
        wake,
    )
    full_field_seconds = (time.perf_counter_ns() - field_started) / 1.0e9
    diagnostics = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "modes": modes,
        "projection_rank_by_frame": ranks,
        "projection_relative_residual_by_frame": residuals,
        "peeled_basis": peeling,
        "rayleigh_diagonal": rayleigh.tolist(),
        "metrics_by_frame": metrics,
        "timing_seconds": {
            "peeling_state": state_seconds,
            "projection": projection_seconds,
            "full_field_and_metrics": full_field_seconds,
            "reconstruction": projection_seconds + full_field_seconds,
        },
    }
    fields = {"u": estimate_u, "v": estimate_v, "omega": estimate_omega, "coefficients": coefficients}
    cleanup_started = time.perf_counter_ns()
    del model, transform, raw_coefficients, fluctuations_u, fluctuations_v, fluctuations_omega
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    diagnostics["timing_seconds"]["cleanup"] = (
        time.perf_counter_ns() - cleanup_started
    ) / 1.0e9
    diagnostics["timing_seconds"]["total_wall"] = (
        time.perf_counter_ns() - total_started
    ) / 1.0e9
    return fields, diagnostics


def laplacian_cache(
    cache_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    stride: int,
    seed: int,
) -> dict[str, Any]:
    max_modes = max(K_VALUES)
    cache_dir.mkdir(parents=True, exist_ok=True)
    required = [
        cache_dir / "x.npy",
        cache_dir / "y.npy",
        cache_dir / "solid.npy",
        cache_dir / "psi.npy",
        cache_dir / "eigenvalues.npy",
        cache_dir / "diagnostics.json",
    ]
    fresh_gx: np.ndarray | None = None
    fresh_gy: np.ndarray | None = None
    cache_is_usable = all(path.exists() for path in required)
    if cache_is_usable:
        xc = np.load(required[0])
        yc = np.load(required[1])
        solid_coarse = np.load(required[2]).astype(bool)
        psi = np.load(required[3], mmap_mode="r")
        eigenvalues = np.load(required[4])
        diagnostics = json.loads(required[5].read_text(encoding="utf-8"))
        cache_is_usable = psi.shape[-1] >= max_modes and len(eigenvalues) >= max_modes
    if not cache_is_usable:
        if "psi" in locals():
            del psi, eigenvalues
            gc.collect()
        basis = geometry_laplacian(x, y, solid, max_modes, stride, seed)
        xc, yc = basis["x"], basis["y"]
        solid_coarse = basis["solid"]
        psi = basis["psi"]
        eigenvalues = basis["eigenvalues"]
        diagnostics = basis["diagnostics"]
        fresh_gx = basis["gx"]
        fresh_gy = basis["gy"]
        np.save(required[0], xc)
        np.save(required[1], yc)
        np.save(required[2], solid_coarse)
        np.save(required[3], psi)
        np.save(required[4], eigenvalues)
        write_json(required[5], diagnostics)
        del basis
    elif psi.shape[-1] > max_modes:
        psi = psi[..., :max_modes]
        eigenvalues = eigenvalues[:max_modes]
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])
    gx = fresh_gx if fresh_gx is not None else np.gradient(
        psi, dx, axis=1, edge_order=2
    ).astype(np.float32, copy=False)
    gy = fresh_gy if fresh_gy is not None else np.gradient(
        psi, dy, axis=0, edge_order=2
    ).astype(np.float32, copy=False)
    gx[:, -1] = 0.0
    return {
        "x": xc,
        "y": yc,
        "solid": solid_coarse,
        "psi": psi,
        "gx": gx,
        "gy": gy,
        "eigenvalues": eigenvalues,
        "diagnostics": diagnostics,
    }


def laplacian_interpolators(basis: dict[str, Any]) -> tuple[Any, Any, Any]:
    axes = (basis["y"], basis["x"])
    options = {"bounds_error": True, "method": "linear"}
    return (
        scipy.interpolate.RegularGridInterpolator(axes, basis["psi"], **options),
        scipy.interpolate.RegularGridInterpolator(axes, basis["gx"], **options),
        scipy.interpolate.RegularGridInterpolator(axes, basis["gy"], **options),
    )


def laplacian_fields(points: np.ndarray, interpolators: tuple[Any, Any, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.column_stack((points[:, 1], points[:, 0]))
    psi = np.asarray(interpolators[0](query), dtype=np.float32)
    gx = np.asarray(interpolators[1](query), dtype=np.float32)
    gy = np.asarray(interpolators[2](query), dtype=np.float32)
    return gy, -gx, psi


def laplacian_reconstructions(
    cache_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    points: np.ndarray,
    subset: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    wake: np.ndarray,
    stride: int,
    field_batch: int,
    seed: int,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, Any]]]:
    basis = laplacian_cache(cache_dir, x, y, solid, stride, seed)
    interpolators = laplacian_interpolators(basis)
    subset_u, subset_v, subset_psi = laplacian_fields(points[subset], interpolators)
    coefficients: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for modes in K_VALUES:
        coeff, ranks, residuals = project_three_frames(
            subset_u[:, :modes],
            subset_v[:, :modes],
            target_u[:, subset],
            target_v[:, subset],
        )
        coefficients[modes] = coeff
        diagnostics[modes] = {
            "modes": modes,
            "projection_rank_by_frame": ranks,
            "projection_relative_residual_by_frame": residuals,
            "first_eigenvalue": float(basis["eigenvalues"][0]),
            "last_eigenvalue": float(basis["eigenvalues"][modes - 1]),
            "eigensolve": basis["diagnostics"],
        }
    del subset_u, subset_v, subset_psi
    fluctuations = {
        modes: {
            "u": np.empty((len(FRAMES), len(points)), dtype=np.float32),
            "v": np.empty((len(FRAMES), len(points)), dtype=np.float32),
            "omega": np.empty((len(FRAMES), len(points)), dtype=np.float32),
        }
        for modes in K_VALUES
    }
    eigenvalues = basis["eigenvalues"].astype(np.float32)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        batch_u, batch_v, batch_psi = laplacian_fields(points[start:stop], interpolators)
        for modes in K_VALUES:
            coeff = coefficients[modes]
            fluctuations[modes]["u"][:, start:stop] = (batch_u[:, :modes] @ coeff).T
            fluctuations[modes]["v"][:, start:stop] = (batch_v[:, :modes] @ coeff).T
            fluctuations[modes]["omega"][:, start:stop] = (
                (batch_psi[:, :modes] * eigenvalues[None, :modes]) @ coeff
            ).T
        del batch_u, batch_v, batch_psi
    fields: dict[int, dict[str, np.ndarray]] = {}
    for modes in K_VALUES:
        estimate_u = mean_u[None, :] + fluctuations[modes]["u"]
        estimate_v = mean_v[None, :] + fluctuations[modes]["v"]
        estimate_omega = mean_omega[None, :] + fluctuations[modes]["omega"]
        diagnostics[modes]["metrics_by_frame"] = full_metrics(
            actual_u,
            actual_v,
            actual_omega,
            mean_u,
            mean_v,
            mean_omega,
            estimate_u,
            estimate_v,
            estimate_omega,
            wake,
        )
        fields[modes] = {
            "u": estimate_u,
            "v": estimate_v,
            "omega": estimate_omega,
            "coefficients": coefficients[modes],
        }
    del basis, interpolators, fluctuations
    gc.collect()
    return fields, diagnostics


def grid_three(values: np.ndarray, fluid: np.ndarray, stride: int) -> np.ndarray:
    return np.stack([as_grid(values[index], fluid)[::stride, ::stride] for index in range(len(FRAMES))])


def save_geometry_archive(
    path: Path,
    case: dict[str, Any],
    fluid: np.ndarray,
    learned: dict[int, dict[str, np.ndarray]],
    laplacian: dict[int, dict[str, np.ndarray]],
    stride: int,
) -> None:
    payload: dict[str, Any] = {
        "x": case["x"][::stride],
        "y": case["y"][::stride],
        "solid": case["solid"][::stride, ::stride],
        "frames": np.asarray(FRAMES, dtype=np.int32),
        "actual_u": case["actual_u"][:, ::stride, ::stride],
        "actual_v": case["actual_v"][:, ::stride, ::stride],
        "actual_omega": case["actual_omega"][:, ::stride, ::stride],
    }
    for modes in K_VALUES:
        for name in ("u", "v", "omega"):
            payload[f"learned_k{modes}_{name}"] = grid_three(learned[modes][name], fluid, stride)
            payload[f"laplacian_k{modes}_{name}"] = grid_three(laplacian[modes][name], fluid, stride)
        payload[f"learned_k{modes}_coefficients"] = learned[modes]["coefficients"]
        payload[f"laplacian_k{modes}_coefficients"] = laplacian[modes]["coefficients"]
    np.savez_compressed(path, **payload)


def save_geometry_archive_with_laplacian_reference(
    path: Path,
    case: dict[str, Any],
    fluid: np.ndarray,
    learned: dict[int, dict[str, np.ndarray]],
    stride: int,
    reference_archive_path: Path,
) -> None:
    payload: dict[str, Any] = {
        "x": case["x"][::stride],
        "y": case["y"][::stride],
        "solid": case["solid"][::stride, ::stride],
        "frames": np.asarray(FRAMES, dtype=np.int32),
        "actual_u": case["actual_u"][:, ::stride, ::stride],
        "actual_v": case["actual_v"][:, ::stride, ::stride],
        "actual_omega": case["actual_omega"][:, ::stride, ::stride],
    }
    for modes in K_VALUES:
        for name in ("u", "v", "omega"):
            payload[f"learned_k{modes}_{name}"] = grid_three(
                learned[modes][name], fluid, stride
            )
        payload[f"learned_k{modes}_coefficients"] = learned[modes]["coefficients"]

    with np.load(reference_archive_path, allow_pickle=False) as reference:
        for name in ("x", "y", "solid", "frames", "actual_u", "actual_v", "actual_omega"):
            if name not in reference.files or not np.array_equal(payload[name], reference[name]):
                raise ValueError(
                    f"Laplacian reference {reference_archive_path} does not match {name}"
                )
        for modes in K_VALUES:
            for name in ("u", "v", "omega", "coefficients"):
                key = f"laplacian_k{modes}_{name}"
                if key not in reference.files:
                    raise ValueError(f"Laplacian reference is missing {key}")
                payload[key] = reference[key]
    np.savez_compressed(path, **payload)


def geometry_evaluation(
    orientation: str,
    gt_path: Path,
    output_dir: Path,
    checkpoints: dict[int, Path],
    peeling_points: int,
    projection_points: int,
    field_batch: int,
    laplacian_stride: int,
    archive_stride: int,
    random_seed: int,
    laplacian_cache_root: Path,
    laplacian_reference_root: Path | None,
    require_learned_timing: bool,
) -> tuple[Path, Path]:
    archive_path = output_dir / "fields_stride2.npz"
    summary_path = output_dir / "summary.json"
    if archive_path.exists() and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_modes = {f"k{modes}" for modes in K_VALUES}
            summary_complete = (
                bool(summary.get("complete"))
                and tuple(summary.get("frames", ())) == tuple(FRAMES)
                and expected_modes.issubset(summary.get("learned", {}))
                and expected_modes.issubset(summary.get("laplacian", {}))
            )
            if require_learned_timing:
                summary_complete = summary_complete and all(
                    "timing_seconds" in summary["learned"][f"k{modes}"]
                    for modes in K_VALUES
                )
            required_archive_keys = {"x", "y", "solid", "frames", "actual_u", "actual_v", "actual_omega"}
            for modes in K_VALUES:
                for method in ("learned", "laplacian"):
                    required_archive_keys.update(
                        {
                            f"{method}_k{modes}_u",
                            f"{method}_k{modes}_v",
                            f"{method}_k{modes}_omega",
                            f"{method}_k{modes}_coefficients",
                        }
                    )
            with np.load(archive_path, allow_pickle=False) as archive:
                archive_complete = (
                    required_archive_keys.issubset(archive.files)
                    and tuple(int(frame) for frame in archive["frames"]) == tuple(FRAMES)
                )
            if summary_complete and archive_complete:
                return archive_path, summary_path
            print(
                f"  [{orientation}] refresh stale geometry cache for "
                f"K={','.join(str(modes) for modes in K_VALUES)}",
                flush=True,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"  [{orientation}] refresh invalid geometry cache: {error}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "basis_state"
    state_dir.mkdir(exist_ok=True)
    case = load_case(gt_path)
    beta = float(case["metadata"]["beta"])
    alpha = float(case["metadata"]["alpha_deg"])
    fluid = ~case["solid"]
    xgrid, ygrid = np.meshgrid(case["x"], case["y"])
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    actual_u = case["actual_u"][:, fluid]
    actual_v = case["actual_v"][:, fluid]
    actual_omega = case["actual_omega"][:, fluid]
    mean_u = case["mean_u"][fluid]
    mean_v = case["mean_v"][fluid]
    mean_omega = case["mean_omega"][fluid]
    target_u = actual_u - mean_u[None, :]
    target_v = actual_v - mean_v[None, :]
    subset = np.linspace(0, len(points) - 1, min(projection_points, len(points)), dtype=np.int64)
    _, cfg64, model64 = load_model_any(checkpoints[64])
    xb, yb = body_numpy(cfg64, points, alpha)
    del model64
    cp.get_default_memory_pool().free_all_blocks()
    wake = (xb >= cfg64.chord) & (xb <= 4.0 * cfg64.chord) & (np.abs(yb) <= 0.75 * cfg64.chord)
    learned_fields: dict[int, dict[str, np.ndarray]] = {}
    learned_diagnostics: dict[int, dict[str, Any]] = {}
    for modes in K_VALUES:
        print(f"  [{orientation}] learned K{modes}", flush=True)
        fields, diagnostics = learned_reconstruction(
            checkpoints[modes],
            state_dir,
            points,
            subset,
            target_u,
            target_v,
            actual_u,
            actual_v,
            actual_omega,
            mean_u,
            mean_v,
            mean_omega,
            wake,
            beta,
            alpha,
            peeling_points,
            field_batch,
        )
        learned_fields[modes] = fields
        learned_diagnostics[modes] = diagnostics
    lap_fields: dict[int, dict[str, np.ndarray]] | None = None
    if laplacian_reference_root is None:
        print(
            f"  [{orientation}] nested geometry-aware FD "
            f"{','.join(f'K{modes}' for modes in K_VALUES)}",
            flush=True,
        )
        lap_fields, lap_diagnostics = laplacian_reconstructions(
            laplacian_cache_root / orientation,
            case["x"],
            case["y"],
            case["solid"],
            points,
            subset,
            target_u,
            target_v,
            actual_u,
            actual_v,
            actual_omega,
            mean_u,
            mean_v,
            mean_omega,
            wake,
            laplacian_stride,
            field_batch,
            random_seed + 2_400_001,
        )
        save_geometry_archive(
            archive_path, case, fluid, learned_fields, lap_fields, archive_stride
        )
    else:
        reference_dir = laplacian_reference_root / orientation
        reference_summary_path = reference_dir / "summary.json"
        reference_archive_path = reference_dir / "fields_stride2.npz"
        print(
            f"  [{orientation}] reuse frozen FD reconstruction and metrics from "
            f"{reference_dir}",
            flush=True,
        )
        reference_summary = json.loads(reference_summary_path.read_text(encoding="utf-8"))
        if (
            reference_summary.get("orientation") != orientation
            or tuple(reference_summary.get("frames", ())) != tuple(FRAMES)
            or not math.isclose(float(reference_summary["beta"]), beta, rel_tol=0.0, abs_tol=0.0)
            or not math.isclose(float(reference_summary["alpha_deg"]), alpha, rel_tol=0.0, abs_tol=0.0)
            or int(reference_summary.get("projection_points", -1)) != int(len(subset))
        ):
            raise ValueError(f"Laplacian reference metadata mismatch for {orientation}")
        lap_diagnostics = {
            modes: reference_summary["laplacian"][f"k{modes}"]
            for modes in K_VALUES
        }
        save_geometry_archive_with_laplacian_reference(
            archive_path,
            case,
            fluid,
            learned_fields,
            archive_stride,
            reference_archive_path,
        )
    summary = {
        "complete": True,
        "orientation": orientation,
        "gt": str(gt_path.resolve()),
        "beta": beta,
        "alpha_deg": alpha,
        "frames": list(FRAMES),
        "steps": [int(case["steps"][frame]) for frame in FRAMES],
        "fluid_points": int(len(points)),
        "projection_points": int(len(subset)),
        "learned": {f"k{modes}": learned_diagnostics[modes] for modes in K_VALUES},
        "laplacian": {f"k{modes}": lap_diagnostics[modes] for modes in K_VALUES},
        "archive": str(archive_path.resolve()),
    }
    write_json(summary_path, summary)
    del learned_fields, lap_fields, learned_diagnostics, lap_diagnostics, points, xgrid, ygrid, case
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return archive_path, summary_path


def panel_figure(
    archive: Any,
    summary: dict[str, Any],
    frame_index: int,
    field: str,
) -> plt.Figure:
    x = archive["x"]
    y = archive["y"]
    solid = archive["solid"].astype(bool)
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    actual_u = archive["actual_u"][frame_index]
    actual_v = archive["actual_v"][frame_index]
    actual_omega = archive["actual_omega"][frame_index]
    def display(method: str, modes: int | None) -> np.ndarray:
        if method == "gt":
            if field == "speed":
                return np.hypot(actual_u, actual_v)
            return actual_omega
        prefix = f"{method}_k{modes}"
        if field == "speed":
            return np.hypot(archive[f"{prefix}_u"][frame_index], archive[f"{prefix}_v"][frame_index])
        return archive[f"{prefix}_omega"][frame_index]
    panels = [display("gt", None)]
    for method in ("learned", "laplacian"):
        panels.extend(display(method, modes) for modes in K_VALUES)
    if field == "vorticity":
        limit = max(float(np.percentile(np.abs(item[~solid]), 99.4)) for item in panels)
        cmap, vmin, vmax = "RdBu_r", -limit, limit
    else:
        vmin = min(float(np.percentile(item[~solid], 0.5)) for item in panels)
        vmax = max(float(np.percentile(item[~solid], 99.5)) for item in panels)
        cmap = "magma"
    columns = 1 + len(K_VALUES)
    figure, axes = plt.subplots(
        2,
        columns,
        figsize=(3.33 * columns, 6.8),
        constrained_layout=False,
        squeeze=False,
    )
    image = None
    for row, method in enumerate(("learned", "laplacian")):
        for column in range(columns):
            modes = None if column == 0 else K_VALUES[column - 1]
            values = display("gt" if column == 0 else method, modes)
            image = axes[row, column].imshow(
                np.ma.masked_where(solid, values),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
            )
            add_solid(axes[row, column], solid, extent)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            axes[row, column].set_aspect("equal")
            if column == 0:
                title = "Ground truth"
            else:
                metrics = summary[method][f"k{modes}"]["metrics_by_frame"][frame_index]
                error_name = "velocity_error" if field == "speed" else "vorticity_error"
                title = f"K={modes}\nL2={100.0 * metrics[error_name]:.2f}%"
            axes[row, column].set_title(title, fontsize=9.0, pad=2.0)
        axes[row, 0].set_ylabel(
            "Original learned" if row == 0 else "Airfoil FD Laplacian",
            fontsize=10.0,
            weight="bold",
            color=LEARNED_COLOR if row == 0 else LAPLACIAN_COLOR,
        )
    assert image is not None
    figure.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.035, pad=0.065).set_label(field)
    frame = int(archive["frames"][frame_index])
    figure.suptitle(
        f"{summary['orientation'].capitalize()} unseen airfoil, frame {frame}: {field} reconstruction\n"
        rf"$\beta={summary['beta']:.6f}$, $\alpha={summary['alpha_deg']:+.5f}^\circ$; same mean, samples, and K",
        fontsize=12.0,
        y=0.985,
    )
    figure.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.13, wspace=0.06, hspace=0.16)
    return figure


def render_outputs(output_dir: Path, geometry_outputs: list[tuple[Path, Path]]) -> dict[str, Any]:
    range_label = f"k{min(K_VALUES)}_to_k{max(K_VALUES)}"
    panel_label = f"2x{1 + len(K_VALUES)}"
    vorticity_pdf = output_dir / f"original_vs_airfoil_laplacian_{range_label}_vorticity.pdf"
    speed_pdf = output_dir / f"original_vs_airfoil_laplacian_{range_label}_speed.pdf"
    pages = []
    with PdfPages(vorticity_pdf) as vort_document, PdfPages(speed_pdf) as speed_document:
        for archive_path, summary_path in geometry_outputs:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with np.load(archive_path, allow_pickle=False) as archive:
                for frame_index, frame in enumerate(FRAMES):
                    page = {"orientation": summary["orientation"], "frame": frame}
                    for field, document in (("vorticity", vort_document), ("speed", speed_document)):
                        figure = panel_figure(archive, summary, frame_index, field)
                        document.savefig(figure, bbox_inches="tight", pad_inches=0.04)
                        png = output_dir / f"{summary['orientation']}_frame{frame:02d}_{field}_{panel_label}.png"
                        figure.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.04)
                        plt.close(figure)
                        page[f"{field}_png"] = str(png.resolve())
                    pages.append(page)
    return {
        "vorticity_pdf": str(vorticity_pdf.resolve()),
        "speed_pdf": str(speed_pdf.resolve()),
        "pages": pages,
    }


def write_metrics_csv(output: Path, geometry_outputs: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    rows = []
    metric_names = (
        "velocity_error",
        "vorticity_error",
        "fluctuation_velocity_error",
        "fluctuation_vorticity_error",
        "wake_velocity_error",
        "wake_vorticity_error",
    )
    for _archive, summary_path in geometry_outputs:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for method in ("learned", "laplacian"):
            for modes in K_VALUES:
                data = summary[method][f"k{modes}"]
                for index, frame in enumerate(FRAMES):
                    row = {
                        "orientation": summary["orientation"],
                        "beta": summary["beta"],
                        "alpha_deg": summary["alpha_deg"],
                        "frame": frame,
                        "method": method,
                        "modes": modes,
                        **data["metrics_by_frame"][index],
                        "projection_rank": data["projection_rank_by_frame"][index],
                        "projection_relative_residual": data["projection_relative_residual_by_frame"][index],
                    }
                    rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as stream:
        fields = (
            "orientation",
            "beta",
            "alpha_deg",
            "frame",
            "method",
            "modes",
            *metric_names,
            "projection_rank",
            "projection_relative_residual",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_reconstruction_timing_csv(
    output: Path, geometry_outputs: list[tuple[Path, Path]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _archive_path, summary_path in geometry_outputs:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for modes in K_VALUES:
            timing = summary["learned"][f"k{modes}"].get("timing_seconds")
            if timing is None:
                raise ValueError(
                    f"missing learned reconstruction timing for {summary['orientation']} K{modes}"
                )
            rows.append(
                {
                    "orientation": summary["orientation"],
                    "frame": FRAMES[0] if len(FRAMES) == 1 else ";".join(map(str, FRAMES)),
                    "modes": modes,
                    "peeling_state_seconds": timing["peeling_state"],
                    "projection_seconds": timing["projection"],
                    "full_field_and_metrics_seconds": timing["full_field_and_metrics"],
                    "reconstruction_seconds": timing["reconstruction"],
                    "cleanup_seconds": timing["cleanup"],
                    "total_wall_seconds": timing["total_wall"],
                }
            )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    global K_VALUES, FRAMES
    parser = argparse.ArgumentParser(description=__doc__)
    for modes in SUPPORTED_K_VALUES:
        parser.add_argument(f"--k{modes}-checkpoint", type=Path)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        choices=SUPPORTED_K_VALUES,
        default=list(SUPPORTED_K_VALUES),
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=[30],
        help="zero-based temporal frame indices to evaluate (default: 30)",
    )
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument("--laplacian-stride", type=int, default=2)
    parser.add_argument("--archive-stride", type=int, default=2)
    parser.add_argument(
        "--laplacian-cache-root",
        type=Path,
        help=(
            "shared per-geometry FD cache root; defaults to a stable sibling of "
            "the staged evaluation directories"
        ),
    )
    parser.add_argument(
        "--laplacian-reference-root",
        type=Path,
        help=(
            "completed evaluation root whose identical FD fields, coefficients, "
            "diagnostics, and metrics are reused without any FD interpolation or projection"
        ),
    )
    parser.add_argument(
        "--require-learned-timing",
        action="store_true",
        help="recompute learned geometry outputs when per-K timing is absent",
    )
    args = parser.parse_args()
    requested = tuple(args.k_values)
    if requested != tuple(sorted(set(requested))) or not requested or requested[0] != 64:
        parser.error("--k-values must be unique, increasing, and start with 64")
    K_VALUES = requested
    requested_frames = tuple(args.frames)
    if (
        requested_frames != tuple(sorted(set(requested_frames)))
        or not requested_frames
        or min(requested_frames) < 0
    ):
        parser.error("--frames must be unique, increasing, non-negative indices")
    FRAMES = requested_frames
    laplacian_cache_root = args.laplacian_cache_root
    if laplacian_cache_root is None:
        laplacian_cache_root = args.output_dir.parent / "shared_geometry_laplacian_cache"
    laplacian_cache_root.mkdir(parents=True, exist_ok=True)
    if args.laplacian_reference_root is not None:
        args.laplacian_reference_root = args.laplacian_reference_root.resolve()
    missing = [modes for modes in K_VALUES if getattr(args, f"k{modes}_checkpoint") is None]
    if missing:
        parser.error(f"missing checkpoint arguments for K={missing}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    checkpoints = {modes: getattr(args, f"k{modes}_checkpoint") for modes in K_VALUES}
    configs = validate_checkpoint_contract(checkpoints)
    protocol = {
        "k_values": list(K_VALUES),
        "frames": list(FRAMES),
        "peeling_points": args.peeling_points,
        "projection_points": args.projection_points,
        "same_projection_points_all_methods": True,
        "same_41_frame_temporal_mean_all_methods": True,
        "learned": "geometry-Rayleigh causal sorting plus two-pass MGS; peeled basis only",
        "laplacian": (
            "frozen reference reconstruction reused byte-for-byte"
            if args.laplacian_reference_root is not None
            else f"one geometry-aware five-point FD K{max(K_VALUES)} eigensolve nested to every requested K"
        ),
        "laplacian_reference_root": (
            str(args.laplacian_reference_root)
            if args.laplacian_reference_root is not None
            else None
        ),
        "laplacian_stride": args.laplacian_stride,
        "dirichlet_boundaries": ["airfoil", "inlet", "ymin", "ymax"],
        "natural_boundaries": ["outlet"],
        "d4": False,
        "analytic_eigenmodes": False,
        "degenerate_block_direction_alignment": False,
        "raw_rank_reported": False,
    }
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "started_at": now(),
            "random_seed": args.random_seed,
            "checkpoints": {f"k{modes}": str(path.resolve()) for modes, path in checkpoints.items()},
            "configs": {f"k{modes}": configs[modes] for modes in K_VALUES},
            "cases": [{"orientation": label, "gt": str((args.gt_root / name).resolve())} for label, name in CASES],
            "protocol": protocol,
        },
    )
    geometry_outputs = []
    for index, (orientation, filename) in enumerate(CASES, start=1):
        print(f"[{index}/3] {orientation} geometry, frames {FRAMES}", flush=True)
        geometry_outputs.append(
            geometry_evaluation(
                orientation,
                args.gt_root / filename,
                args.output_dir / orientation,
                checkpoints,
                args.peeling_points,
                args.projection_points,
                args.field_batch,
                args.laplacian_stride,
                args.archive_stride,
                args.random_seed + 1000 * index,
                laplacian_cache_root,
                args.laplacian_reference_root,
                args.require_learned_timing,
            )
        )
    print(f"[4/5] render 2x{1 + len(K_VALUES)} PNG and multi-page PDF products", flush=True)
    figures = render_outputs(args.output_dir, geometry_outputs)
    print("[5/5] write metrics and completion marker", flush=True)
    metrics_path = args.output_dir / "reconstruction_metrics.csv"
    rows = write_metrics_csv(metrics_path, geometry_outputs)
    timing_path: Path | None = None
    timing_rows: list[dict[str, Any]] = []
    if args.require_learned_timing:
        timing_path = args.output_dir / "reconstruction_timing.csv"
        timing_rows = write_reconstruction_timing_csv(timing_path, geometry_outputs)
    summary = {
        "complete": True,
        "completed_at": now(),
        "evaluation_wall_seconds": (time.perf_counter_ns() - started) / 1.0e9,
        "protocol": protocol,
        "checkpoints": {f"k{modes}": str(path.resolve()) for modes, path in checkpoints.items()},
        "geometry_summaries": [str(path.resolve()) for _archive, path in geometry_outputs],
        "metric_rows": len(rows),
        "metrics_csv": str(metrics_path.resolve()),
        "timing_rows": len(timing_rows),
        "timing_csv": str(timing_path.resolve()) if timing_path is not None else None,
        "figures": figures,
    }
    write_json(args.output_dir / "comparison_complete.json", summary)
    print(json.dumps({"complete": True, "wall_seconds": summary["evaluation_wall_seconds"], "figures": figures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
