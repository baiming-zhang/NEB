#!/usr/bin/env python3
"""Evaluate strict Potential/Velocity/Vorticity K2048 nested prefixes.

For each unseen geometry and each objective, this program builds exactly one
geometry-specific K2048 causal order and two-pass MGS transform on the same IID
fluid points.  The evaluated K=64,128,256,512,1024,2048 spaces are literal
column prefixes of that one transform.  All objectives use the same temporal
mean, velocity-projection nodes, GT frame, spatial grid, and downstream error
definitions.  Method/geometry products are atomic and restartable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    airfoil_area,
    body_numpy,
    cfg_namespace,
    fluid_points,
)
from evaluate_unseen_k64_vs_laplacian import (  # noqa: E402
    relative_scalar,
    relative_velocity,
)


PREFIXES = (64, 128, 256, 512, 1024, 2048)
OBJECTIVES = ("potential", "velocity", "vorticity")
CASES = (
    ("up", "gt_019.npz"),
    ("middle", "gt_015.npz"),
    ("down", "gt_014.npz"),
)
FRAME = 30


def now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint: Path, objective: str):
    sidecar = checkpoint.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    aliases = {
        "streamfunction_dirichlet_rayleigh": "potential",
        "velocity_rayleigh": "velocity",
        "pure_vorticity_rayleigh": "vorticity",
    }
    actual = aliases.get(str(cfg.basis_objective))
    expected_jet = {"potential": 1, "velocity": 2, "vorticity": 3}[objective]
    if actual != objective:
        raise ValueError(f"{checkpoint}: objective {actual!r} != {objective!r}")
    if int(metadata.get("epoch", -1)) != 2000 or int(cfg.modes) != 2048:
        raise ValueError(f"{checkpoint}: strict evaluation requires epoch2000 K2048")
    if int(cfg.spatial_derivative_order) != expected_jet:
        raise ValueError(f"{checkpoint}: expected Jet{expected_jet}")
    if int(cfg.reorthogonalization_passes) != 2:
        raise ValueError(f"{checkpoint}: expected two-pass MGS")
    if str(cfg.sampling_mode) != "uniform_random":
        raise ValueError(f"{checkpoint}: expected PCG64 uniform_random training")
    return metadata, cfg, HardcodedModel(cfg, arrays)


def validate_common_contract(checkpoints: dict[str, Path]) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {}
    epoch0_hashes: dict[str, str | None] = {}
    for objective, checkpoint in checkpoints.items():
        sidecar = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
        configs[objective] = sidecar["config"]
        epoch0_hashes[objective] = sidecar.get("initial_archive_sha256")
        metadata, _cfg, model = load_model(checkpoint, objective)
        del metadata, model
        cp.get_default_memory_pool().free_all_blocks()
    common_keys = (
        "modes", "epochs", "hidden", "hidden_layers",
        "siren_first_omega0", "siren_hidden_omega0", "points_per_step",
        "sampling_mode", "reorthogonalization_passes", "causal_sorting",
        "optimizer", "base_learning_rate", "min_learning_rate", "grad_clip",
        "validate_every", "early_stopping_enabled", "seed",
    )
    reference = configs["potential"]

    def contract_value(config: dict[str, Any], key: str) -> Any:
        # The frozen Potential checkpoint predates the explicit optimizer field.
        # Its training protocol and checkpoint arrays are Adam, which was the
        # hardcoded backend default.  Normalize that legacy omission without
        # relaxing any other part of the paired contract.
        if key == "optimizer":
            return config.get(key, "adam")
        return config.get(key)

    mismatches: dict[str, Any] = {}
    for objective in OBJECTIVES[1:]:
        current = configs[objective]
        difference = {
            key: (contract_value(reference, key), contract_value(current, key))
            for key in common_keys
            if contract_value(reference, key) != contract_value(current, key)
        }
        if difference:
            mismatches[objective] = difference
    if mismatches:
        raise ValueError(f"strict common configuration mismatch: {mismatches}")
    return {"configs": configs, "epoch0_hashes": epoch0_hashes}


def load_case(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        u_all = archive["u"].astype(np.float32)
        v_all = archive["v"].astype(np.float32)
        omega_all = archive["vorticity"].astype(np.float32)
        return {
            "x": archive["x"].astype(np.float32),
            "y": archive["y"].astype(np.float32),
            "solid": archive["solid"].astype(bool),
            "steps": archive["steps"].astype(np.int64),
            "metadata": metadata,
            "actual_u": u_all[FRAME],
            "actual_v": v_all[FRAME],
            "actual_omega": omega_all[FRAME],
            "mean_u": np.mean(u_all, axis=0, dtype=np.float64).astype(np.float32),
            "mean_v": np.mean(v_all, axis=0, dtype=np.float64).astype(np.float32),
            "mean_omega": np.mean(omega_all, axis=0, dtype=np.float64).astype(np.float32),
        }


def raw_objective_gpu(
    model: HardcodedModel,
    cfg: SimpleNamespace,
    points: np.ndarray,
    beta: float,
    alpha: float,
    objective: str,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return flattened objective values and their x/y gradients on CUDA."""

    count, modes = len(points), int(cfg.modes)
    if objective == "potential":
        value = cp.empty((count, modes), dtype=cp.float32)
        gx = cp.empty_like(value)
        gy = cp.empty_like(value)
        iterator = model.forward_stream_jet1(cfg, points, beta, alpha, False)
        for mode, field, _cache in iterator:
            index = int(mode.mode_index)
            value[:, index] = field.value[:, 0]
            gx[:, index] = field.grad[:, 0, 0]
            gy[:, index] = field.grad[:, 0, 1]
        return value, gx, gy

    if objective == "velocity":
        u = cp.empty((count, modes), dtype=cp.float32)
        v = cp.empty_like(u)
        ux = cp.empty_like(u)
        uy = cp.empty_like(u)
        vx = cp.empty_like(u)
        vy = cp.empty_like(u)
        iterator = model.forward_stream_jet2(cfg, points, beta, alpha, False)
        for mode, field, _cache in iterator:
            index = int(mode.mode_index)
            u[:, index] = field.grad[:, 0, 1]
            v[:, index] = -field.grad[:, 0, 0]
            ux[:, index] = field.hess[:, 0, 1]
            uy[:, index] = field.hess[:, 0, 2]
            vx[:, index] = -field.hess[:, 0, 0]
            vy[:, index] = -field.hess[:, 0, 1]
        return cp.concatenate((u, v), axis=0), cp.concatenate((ux, vx), axis=0), cp.concatenate((uy, vy), axis=0)

    omega = cp.empty((count, modes), dtype=cp.float32)
    omega_x = cp.empty_like(omega)
    omega_y = cp.empty_like(omega)
    iterator = model.forward_stream(cfg, points, beta, alpha, False)
    for mode, field, _cache in iterator:
        index = int(mode.mode_index)
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        omega_x[:, index] = -(field.third[:, 0, 0] + field.third[:, 0, 2])
        omega_y[:, index] = -(field.third[:, 0, 1] + field.third[:, 0, 3])
    return omega, omega_x, omega_y


def build_transform_gpu(
    cfg: SimpleNamespace,
    values: cp.ndarray,
    gradient_x: cp.ndarray,
    gradient_y: cp.ndarray,
    domain_area: float,
    objective: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Stable GPU causal sort followed by exactly two MGS passes."""

    started = time.perf_counter()
    rows, modes = values.shape
    # Values are stacked by objective component.  Gradient arrays use the same
    # stacking, hence component_count = value rows / physical point count.
    # The caller supplies IID points; recover their count from the objective.
    component_count = 2 if objective == "velocity" else 1
    physical_points = rows // component_count
    if physical_points <= 0 or rows != physical_points * component_count:
        raise ValueError("invalid objective field stacking")
    weight = cp.asarray(np.float32(domain_area / physical_points), dtype=cp.float32)
    candidate_mass = weight * cp.sum(values * values, axis=0)
    candidate_stiffness = weight * cp.sum(
        gradient_x * gradient_x + gradient_y * gradient_y, axis=0
    )
    sorting_rayleigh = candidate_stiffness / candidate_mass
    order = cp.argsort(sorting_rayleigh, kind="stable")
    sorted_values = values[:, order]
    basis = cp.empty_like(sorted_values)
    transform_sorted = cp.zeros((modes, modes), dtype=cp.float32)
    pre_mass = cp.empty(modes, dtype=cp.float32)
    for causal_rank in range(modes):
        residual = sorted_values[:, causal_rank].copy()
        coefficients = cp.zeros(modes, dtype=cp.float32)
        coefficients[causal_rank] = 1.0
        if causal_rank:
            previous = basis[:, :causal_rank]
            previous_transform = transform_sorted[:, :causal_rank]
            for _ in range(2):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        pre_mass[causal_rank] = mass
        inverse_norm = cp.reciprocal(cp.sqrt(cp.maximum(mass, np.float32(cfg.mass_floor))))
        basis[:, causal_rank] = residual * inverse_norm
        transform_sorted[:, causal_rank] = coefficients * inverse_norm
    transform_network = cp.zeros_like(transform_sorted)
    transform_network[order, :] = transform_sorted
    stiffness_gram = weight * (
        gradient_x.T @ gradient_x + gradient_y.T @ gradient_y
    )
    peeled_rayleigh = cp.sum(
        transform_network * (stiffness_gram @ transform_network), axis=0
    )
    gram = weight * (basis.T @ basis)
    raw_mass_sorted = candidate_mass[order]
    relative_mass = pre_mass / raw_mass_sorted
    cp.cuda.Stream.null.synchronize()
    transform = cp.asnumpy(transform_network)
    rayleigh = cp.asnumpy(peeled_rayleigh).astype(np.float64)
    order_cpu = cp.asnumpy(order).astype(np.int64)
    pre_mass_cpu = cp.asnumpy(pre_mass).astype(np.float64)
    relative_mass_cpu = cp.asnumpy(relative_mass).astype(np.float64)
    sorting_cpu = cp.asnumpy(sorting_rayleigh).astype(np.float64)
    orth = float(cp.max(cp.abs(gram - cp.eye(modes, dtype=cp.float32))).get())
    if (
        not np.all(np.isfinite(transform))
        or not np.all(np.isfinite(rayleigh))
        or np.any(pre_mass_cpu <= float(cfg.mass_floor))
        or np.any(
            relative_mass_cpu
            <= float(getattr(cfg, "peeling_relative_mass_floor", 1.0e-12))
        )
    ):
        raise FloatingPointError(f"{objective} causal MGS produced an invalid state")
    diagnostics = {
        "objective": objective,
        "modes": modes,
        "peeled_rank": modes,
        "causal_sort_device": "gpu",
        "mgs_device": "gpu",
        "reorthogonalization_passes": 2,
        "optimizer_host_transfers_per_step": 0,
        "causal_order_1based": (order_cpu + 1).tolist(),
        "sorting_rayleigh_by_network": sorting_cpu.tolist(),
        "peeled_rayleigh_diagonal": rayleigh.tolist(),
        "peeled_orthogonality_max_abs": orth,
        "minimum_pre_normalization_mass": float(np.min(pre_mass_cpu)),
        "minimum_relative_mass": float(np.min(relative_mass_cpu)),
        "seconds": time.perf_counter() - started,
    }
    return transform.astype(np.float32), rayleigh, diagnostics


def state_for_objective(
    checkpoint: Path,
    objective: str,
    state_dir: Path,
    points: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[Any, HardcodedModel, np.ndarray, dict[str, Any]]:
    state_npz = state_dir / f"{objective}_k2048_prefix_state.npz"
    state_json = state_dir / f"{objective}_k2048_prefix_state.json"
    metadata, cfg, model = load_model(checkpoint, objective)
    if state_npz.exists() and state_json.exists():
        with np.load(state_npz, allow_pickle=False) as archive:
            transform = archive["transform_network_to_causal"].astype(np.float32)
        diagnostics = json.loads(state_json.read_text(encoding="utf-8"))
        if transform.shape != (2048, 2048) or diagnostics.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError(f"stale strict state {state_npz}")
        return cfg, model, transform, diagnostics
    started = time.perf_counter()
    values, gx, gy = raw_objective_gpu(model, cfg, points, beta, alpha, objective)
    field_seconds = time.perf_counter() - started
    domain_area = (
        (float(cfg.xmax) - float(cfg.xmin))
        * (float(cfg.ymax) - float(cfg.ymin))
        - airfoil_area(cfg, beta)
    )
    transform, rayleigh, diagnostics = build_transform_gpu(
        cfg, values, gx, gy, domain_area, objective
    )
    diagnostics.update(
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_epoch=int(metadata["epoch"]),
        beta=beta,
        alpha_deg=alpha,
        peeling_points=int(len(points)),
        raw_objective_field_seconds=field_seconds,
        prefixes=list(PREFIXES),
        completed_at=now_iso(),
    )
    state_dir.mkdir(parents=True, exist_ok=True)
    temporary = state_npz.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            transform_network_to_causal=transform,
            peeled_rayleigh=rayleigh,
        )
    temporary.replace(state_npz)
    write_json(state_json, diagnostics)
    del values, gx, gy
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return cfg, model, transform, diagnostics


def raw_uvomega_gpu(
    model: HardcodedModel,
    cfg: SimpleNamespace,
    points: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    count, modes = len(points), int(cfg.modes)
    u = cp.empty((count, modes), dtype=cp.float32)
    v = cp.empty_like(u)
    omega = cp.empty_like(u)
    for mode, field, _cache in model.forward_stream(cfg, points, beta, alpha, False):
        index = int(mode.mode_index)
        u[:, index] = field.grad[:, 0, 1]
        v[:, index] = -field.grad[:, 0, 0]
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
    return u, v, omega


def solve_nested_projection(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    target_u_gpu = cp.asarray(target_u, dtype=cp.float32)
    target_v_gpu = cp.asarray(target_v, dtype=cp.float32)
    gram = cp.asnumpy(basis_u.T @ basis_u + basis_v.T @ basis_v).astype(np.float64)
    right = cp.asnumpy(basis_u.T @ target_u_gpu + basis_v.T @ target_v_gpu).astype(np.float64)
    coefficients: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    denominator = float(np.sqrt(np.sum(target_u.astype(np.float64) ** 2 + target_v.astype(np.float64) ** 2)))
    for modes in PREFIXES:
        started = time.perf_counter()
        current = 0.5 * (gram[:modes, :modes] + gram[:modes, :modes].T)
        values, vectors = np.linalg.eigh(current)
        threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
        retained = values > threshold
        coefficient = vectors[:, retained] @ (
            (vectors[:, retained].T @ right[:modes]) / values[retained]
        )
        fitted_u = cp.asnumpy(basis_u[:, :modes] @ cp.asarray(coefficient, dtype=cp.float32))
        fitted_v = cp.asnumpy(basis_v[:, :modes] @ cp.asarray(coefficient, dtype=cp.float32))
        residual = float(
            np.sqrt(np.sum((fitted_u - target_u) ** 2 + (fitted_v - target_v) ** 2, dtype=np.float64))
            / max(denominator, 1.0e-30)
        )
        coefficients[modes] = coefficient.astype(np.float32)
        diagnostics[modes] = {
            "projection_rank": int(np.count_nonzero(retained)),
            "projection_relative_residual": residual,
            "projection_solve_seconds": time.perf_counter() - started,
            "gram_min_retained_eigenvalue": float(np.min(values[retained])),
            "gram_max_eigenvalue": float(values[-1]),
        }
    return coefficients, diagnostics


def six_metrics(
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
) -> dict[str, float]:
    return {
        "velocity_error": relative_velocity(actual_u, actual_v, estimate_u, estimate_v),
        "vorticity_error": relative_scalar(actual_omega, estimate_omega),
        "fluctuation_velocity_error": relative_velocity(actual_u - mean_u, actual_v - mean_v, estimate_u - mean_u, estimate_v - mean_v),
        "fluctuation_vorticity_error": relative_scalar(actual_omega - mean_omega, estimate_omega - mean_omega),
        "wake_velocity_error": relative_velocity(actual_u, actual_v, estimate_u, estimate_v, wake),
        "wake_vorticity_error": relative_scalar(actual_omega, estimate_omega, wake),
    }


def evaluate_objective(
    checkpoint: Path,
    objective: str,
    output_dir: Path,
    peeling_points: np.ndarray,
    points: np.ndarray,
    projection_indices: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    wake: np.ndarray,
    beta: float,
    alpha: float,
    field_batch: int,
) -> dict[str, Any]:
    result_npz = output_dir / f"{objective}_reconstruction.npz"
    result_json = output_dir / f"{objective}_summary.json"
    if result_npz.exists() and result_json.exists():
        summary = json.loads(result_json.read_text(encoding="utf-8"))
        if summary.get("complete") and summary.get("checkpoint_sha256") == file_sha256(checkpoint):
            return summary
    total_started = time.perf_counter()
    cfg, model, transform, state = state_for_objective(
        checkpoint, objective, output_dir / "basis_state", peeling_points, beta, alpha
    )
    transform_gpu = cp.asarray(transform)
    projection_started = time.perf_counter()
    raw_u, raw_v, raw_omega = raw_uvomega_gpu(
        model, cfg, points[projection_indices], beta, alpha
    )
    basis_u = raw_u @ transform_gpu
    basis_v = raw_v @ transform_gpu
    target_u = actual_u - mean_u
    target_v = actual_v - mean_v
    coefficients, projection = solve_nested_projection(
        basis_u,
        basis_v,
        target_u[projection_indices],
        target_v[projection_indices],
    )
    raw_coefficients = cp.zeros((2048, len(PREFIXES)), dtype=cp.float32)
    for column, modes in enumerate(PREFIXES):
        raw_coefficients[:, column] = (
            transform_gpu[:, :modes] @ cp.asarray(coefficients[modes])
        )
    projection_seconds = time.perf_counter() - projection_started
    del raw_u, raw_v, raw_omega, basis_u, basis_v
    cp.get_default_memory_pool().free_all_blocks()

    field_started = time.perf_counter()
    fluctuations_u = np.empty((len(PREFIXES), len(points)), dtype=np.float32)
    fluctuations_v = np.empty_like(fluctuations_u)
    fluctuations_omega = np.empty_like(fluctuations_u)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        raw_u, raw_v, raw_omega = raw_uvomega_gpu(
            model, cfg, points[start:stop], beta, alpha
        )
        fluctuations_u[:, start:stop] = cp.asnumpy(raw_u @ raw_coefficients).T
        fluctuations_v[:, start:stop] = cp.asnumpy(raw_v @ raw_coefficients).T
        fluctuations_omega[:, start:stop] = cp.asnumpy(raw_omega @ raw_coefficients).T
        del raw_u, raw_v, raw_omega
    estimates_u = fluctuations_u + mean_u[None, :]
    estimates_v = fluctuations_v + mean_v[None, :]
    estimates_omega = fluctuations_omega + mean_omega[None, :]
    metrics: dict[str, Any] = {}
    for index, modes in enumerate(PREFIXES):
        metrics[f"k{modes}"] = {
            **projection[modes],
            "metrics": six_metrics(
                actual_u, actual_v, actual_omega,
                mean_u, mean_v, mean_omega,
                estimates_u[index], estimates_v[index], estimates_omega[index], wake,
            ),
        }
    field_seconds = time.perf_counter() - field_started
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = result_npz.with_suffix(".npz.tmp")
    raw_coefficients_cpu = cp.asnumpy(raw_coefficients)
    causal_coefficients = np.zeros((2048, len(PREFIXES)), dtype=np.float32)
    for column, modes in enumerate(PREFIXES):
        causal_coefficients[:modes, column] = coefficients[modes]
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            prefixes=np.asarray(PREFIXES, dtype=np.int64),
            estimate_u=estimates_u,
            estimate_v=estimates_v,
            estimate_omega=estimates_omega,
            causal_coefficients=causal_coefficients,
            raw_network_coefficients=raw_coefficients_cpu,
        )
    temporary.replace(result_npz)
    summary = {
        "complete": True,
        "objective": objective,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "prefixes": list(PREFIXES),
        "basis_state": state,
        "results": metrics,
        "timing_seconds": {
            "basis_generation": float(state.get("raw_objective_field_seconds", 0.0)) + float(state.get("seconds", 0.0)),
            "projection_all_prefixes": projection_seconds,
            "full_field_and_metrics_all_prefixes": field_seconds,
            "total_wall": time.perf_counter() - total_started,
        },
        "completed_at": now_iso(),
    }
    write_json(result_json, summary)
    del model, transform_gpu, raw_coefficients, fluctuations_u, fluctuations_v, fluctuations_omega, estimates_u, estimates_v, estimates_omega
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for objective in OBJECTIVES:
        parser.add_argument(f"--{objective}-checkpoint", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument("--random-seed", type=int, default=20260824)
    args = parser.parse_args()
    checkpoints = {
        objective: getattr(args, f"{objective}_checkpoint").resolve()
        for objective in OBJECTIVES
    }
    contract = validate_common_contract(checkpoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {
        "status": "running",
        "started_at": now_iso(),
        "checkpoints": {
            objective: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for objective, path in checkpoints.items()
        },
        "contract": contract,
        "protocol": {
            "full_modes": 2048,
            "prefixes": list(PREFIXES),
            "one_transform_per_objective_geometry": True,
            "independent_prefix_peeling": False,
            "peeling_points": args.peeling_points,
            "projection_points": args.projection_points,
            "same_points_all_objectives": True,
            "frame": FRAME,
            "same_41_frame_mean": True,
            "projection_field": "velocity fluctuation",
            "mgs": "two-pass causal objective-metric MGS on GPU",
        },
    }
    write_json(args.output_dir / "evaluation_manifest.json", manifest)
    summaries: list[dict[str, Any]] = []
    for geometry_index, (orientation, filename) in enumerate(CASES, start=1):
        case = load_case(args.gt_root / filename)
        beta = float(case["metadata"]["beta"])
        alpha = float(case["metadata"]["alpha_deg"])
        fluid = ~case["solid"]
        xgrid, ygrid = np.meshgrid(case["x"], case["y"])
        points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
        actual_u = case["actual_u"][fluid]
        actual_v = case["actual_v"][fluid]
        actual_omega = case["actual_omega"][fluid]
        mean_u = case["mean_u"][fluid]
        mean_v = case["mean_v"][fluid]
        mean_omega = case["mean_omega"][fluid]
        subset = np.linspace(
            0, len(points) - 1, min(args.projection_points, len(points)), dtype=np.int64
        )
        _, cfg, temporary_model = load_model(checkpoints["potential"], "potential")
        peeling = fluid_points(
            cfg,
            args.peeling_points,
            beta,
            alpha,
            args.random_seed + 1000 * geometry_index + 2_300_001,
        )
        xb, yb = body_numpy(cfg, points, alpha)
        wake = (
            (xb >= cfg.chord)
            & (xb <= 4.0 * cfg.chord)
            & (np.abs(yb) <= 0.75 * cfg.chord)
        )
        del temporary_model
        cp.get_default_memory_pool().free_all_blocks()
        geometry_dir = args.output_dir / orientation
        geometry_dir.mkdir(parents=True, exist_ok=True)
        geometry_summary: dict[str, Any] = {
            "orientation": orientation,
            "gt": str((args.gt_root / filename).resolve()),
            "beta": beta,
            "alpha_deg": alpha,
            "frame": FRAME,
            "step": int(case["steps"][FRAME]),
            "fluid_points": int(len(points)),
            "projection_points": int(len(subset)),
            "objectives": {},
        }
        for objective in OBJECTIVES:
            write_json(
                args.output_dir / "evaluation_state.json",
                {
                    "status": "running",
                    "orientation": orientation,
                    "geometry_index": geometry_index,
                    "objective": objective,
                    "updated_at": now_iso(),
                },
            )
            print(f"[{geometry_index}/3] {orientation}: {objective} K2048 nested prefixes", flush=True)
            geometry_summary["objectives"][objective] = evaluate_objective(
                checkpoints[objective], objective, geometry_dir, peeling, points, subset,
                actual_u, actual_v, actual_omega,
                mean_u, mean_v, mean_omega, wake,
                beta, alpha, args.field_batch,
            )
            write_json(geometry_dir / "summary.json", geometry_summary)
        summaries.append(geometry_summary)
        del case, points, peeling, actual_u, actual_v, actual_omega, mean_u, mean_v, mean_omega
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    rows: list[dict[str, Any]] = []
    for geometry in summaries:
        for objective in OBJECTIVES:
            result = geometry["objectives"][objective]
            for modes in PREFIXES:
                item = result["results"][f"k{modes}"]
                rows.append(
                    {
                        "orientation": geometry["orientation"],
                        "objective": objective,
                        "modes": modes,
                        **item["metrics"],
                        "projection_rank": item["projection_rank"],
                        "projection_relative_residual": item["projection_relative_residual"],
                        "basis_generation_seconds": result["timing_seconds"]["basis_generation"],
                        "objective_geometry_total_seconds": result["timing_seconds"]["total_wall"],
                    }
                )
    csv_path = args.output_dir / "strict_prefix_reconstruction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completion = {
        "complete": True,
        "completed_at": now_iso(),
        "wall_seconds": time.perf_counter() - started,
        "rows": len(rows),
        "metrics_csv": str(csv_path.resolve()),
        "geometry_summaries": [
            str((args.output_dir / geometry["orientation"] / "summary.json").resolve())
            for geometry in summaries
        ],
    }
    write_json(args.output_dir / "comparison_complete.json", completion)
    write_json(args.output_dir / "evaluation_state.json", {"status": "complete", **completion})
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
