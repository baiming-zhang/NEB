#!/usr/bin/env python3
"""Independently project every GT frame into the frozen Potential K2048 span.

This is a kinematic reconstruction diagnostic.  Every frame receives its own
least-squares coefficients; no coefficient ODE, temporal recurrence, or ROM
time integration is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from evaluate_strict_rayleigh_k2048_prefixes import (  # noqa: E402
    load_model,
    raw_uvomega_gpu,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def relative_l2(actual: tuple[np.ndarray, ...], predicted: tuple[np.ndarray, ...], fluid: np.ndarray) -> float:
    numerator = sum(
        float(np.sum((p[:, fluid].astype(np.float64) - a[:, fluid]) ** 2))
        for a, p in zip(actual, predicted, strict=True)
    )
    denominator = sum(
        float(np.sum(a[:, fluid].astype(np.float64) ** 2)) for a in actual
    )
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=31)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=4096)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    state_path = args.output_root / "state.json"
    write_json(state_path, {"status": "running", "stage": "load", "started_at": now_iso()})

    with np.load(args.gt, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        x = archive["x"][::2].astype(np.float32)
        y = archive["y"][::2].astype(np.float32)
        solid = archive["solid"][::2, ::2].astype(bool)
        steps = archive["steps"][: args.frames].astype(np.int64)
        all_u = archive["u"][:, ::2, ::2].astype(np.float32)
        all_v = archive["v"][:, ::2, ::2].astype(np.float32)
        all_omega = archive["vorticity"][:, ::2, ::2].astype(np.float32)
    if args.frames < 1 or args.frames > len(all_u):
        raise ValueError(f"frames must be in [1, {len(all_u)}]")
    actual_u = all_u[: args.frames]
    actual_v = all_v[: args.frames]
    actual_omega = all_omega[: args.frames]
    mean_u = np.mean(all_u, axis=0, dtype=np.float64).astype(np.float32)
    mean_v = np.mean(all_v, axis=0, dtype=np.float64).astype(np.float32)
    mean_omega = np.mean(all_omega, axis=0, dtype=np.float64).astype(np.float32)
    del all_u, all_v, all_omega

    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    subset = np.linspace(
        0,
        len(points) - 1,
        min(args.projection_points, len(points)),
        dtype=np.int64,
    )
    beta = float(metadata["beta"])
    alpha = float(metadata["alpha_deg"])
    _checkpoint_metadata, cfg, model = load_model(args.checkpoint, "potential")
    with np.load(args.transform, allow_pickle=False) as archive:
        transform = archive["transform_network_to_causal"].astype(np.float32)
    if transform.shape != (2048, 2048):
        raise ValueError(f"unexpected transform shape {transform.shape}")

    # Evaluate the raw velocity basis only once on the common projection set.
    projection_started = time.perf_counter()
    write_json(state_path, {"status": "running", "stage": "projection_basis", "updated_at": now_iso()})
    count = len(subset)
    raw_u = cp.empty((count, 2048), dtype=cp.float32)
    raw_v = cp.empty_like(raw_u)
    for mode, field, _cache in model.forward_stream_jet1(
        cfg, points[subset], beta, alpha, False
    ):
        index = int(mode.mode_index)
        raw_u[:, index] = field.grad[:, 0, 1]
        raw_v[:, index] = -field.grad[:, 0, 0]

    target_u = np.ascontiguousarray(
        (actual_u[:, fluid] - mean_u[fluid][None, :])[:, subset].T
    )
    target_v = np.ascontiguousarray(
        (actual_v[:, fluid] - mean_v[fluid][None, :])[:, subset].T
    )
    target_u_gpu = cp.asarray(target_u)
    target_v_gpu = cp.asarray(target_v)
    transform_gpu = cp.asarray(transform)

    # Materialize the causal basis before forming normal equations.  Forming a
    # raw-network Gram and applying the highly conditioned triangular transform
    # afterwards loses numerical rank in FP32, even though it is algebraically
    # equivalent in exact arithmetic.
    basis_u = raw_u @ transform_gpu
    basis_v = raw_v @ transform_gpu
    gram_gpu = basis_u.T @ basis_u + basis_v.T @ basis_v
    right_gpu = basis_u.T @ target_u_gpu + basis_v.T @ target_v_gpu
    gram = cp.asnumpy(gram_gpu).astype(np.float64)
    right = cp.asnumpy(right_gpu).astype(np.float64)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    threshold = max(float(eigenvalues[-1]) * 1.0e-10, 1.0e-14)
    retained = eigenvalues > threshold
    coefficients = eigenvectors[:, retained] @ (
        (eigenvectors[:, retained].T @ right) / eigenvalues[retained, None]
    )
    coefficients = coefficients.astype(np.float32)
    raw_coefficients = transform_gpu @ cp.asarray(coefficients)
    fitted_u = cp.asnumpy(basis_u @ cp.asarray(coefficients))
    fitted_v = cp.asnumpy(basis_v @ cp.asarray(coefficients))
    target_norm = np.sqrt(
        np.sum(target_u.astype(np.float64) ** 2 + target_v.astype(np.float64) ** 2, axis=0)
    )
    projection_residual = np.sqrt(
        np.sum((fitted_u - target_u) ** 2 + (fitted_v - target_v) ** 2, axis=0, dtype=np.float64)
    ) / np.maximum(target_norm, np.finfo(np.float64).tiny)
    projection_seconds = time.perf_counter() - projection_started
    del raw_u, raw_v, basis_u, basis_v, gram_gpu, right_gpu, target_u_gpu, target_v_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # One full-grid basis evaluation reconstructs all requested frames at once.
    field_started = time.perf_counter()
    reconstructed_u = np.empty((args.frames, len(points)), dtype=np.float32)
    reconstructed_v = np.empty_like(reconstructed_u)
    reconstructed_omega = np.empty_like(reconstructed_u)
    batches = (len(points) + args.field_batch - 1) // args.field_batch
    for batch_index, start in enumerate(range(0, len(points), args.field_batch), start=1):
        stop = min(start + args.field_batch, len(points))
        write_json(
            state_path,
            {
                "status": "running",
                "stage": "full_grid_fields",
                "batch": batch_index,
                "batches": batches,
                "updated_at": now_iso(),
            },
        )
        raw_u, raw_v, raw_omega = raw_uvomega_gpu(
            model, cfg, points[start:stop], beta, alpha
        )
        reconstructed_u[:, start:stop] = cp.asnumpy(raw_u @ raw_coefficients).T
        reconstructed_v[:, start:stop] = cp.asnumpy(raw_v @ raw_coefficients).T
        reconstructed_omega[:, start:stop] = cp.asnumpy(raw_omega @ raw_coefficients).T
        del raw_u, raw_v, raw_omega
        if batch_index == 1 or batch_index % 10 == 0 or batch_index == batches:
            print(f"[field] batch {batch_index}/{batches}", flush=True)
    field_seconds = time.perf_counter() - field_started

    predicted_u = np.broadcast_to(mean_u, actual_u.shape).copy()
    predicted_v = np.broadcast_to(mean_v, actual_v.shape).copy()
    predicted_omega = np.broadcast_to(mean_omega, actual_omega.shape).copy()
    predicted_u[:, fluid] += reconstructed_u
    predicted_v[:, fluid] += reconstructed_v
    predicted_omega[:, fluid] += reconstructed_omega
    for values in (actual_u, actual_v, actual_omega, predicted_u, predicted_v, predicted_omega):
        values[:, solid] = 0.0

    rows = []
    for frame in range(args.frames):
        velocity_error = relative_l2(
            (actual_u[frame : frame + 1], actual_v[frame : frame + 1]),
            (predicted_u[frame : frame + 1], predicted_v[frame : frame + 1]),
            fluid,
        )
        vorticity_error = relative_l2(
            (actual_omega[frame : frame + 1],),
            (predicted_omega[frame : frame + 1],),
            fluid,
        )
        rows.append(
            {
                "frame": frame,
                "lbm_step": int(steps[frame]),
                "velocity_relative_l2": velocity_error,
                "vorticity_relative_l2": vorticity_error,
                "projection_residual": float(projection_residual[frame]),
            }
        )

    fields_path = args.output_root / "potential_direct_projection_fields_stride2.npz"
    temporary = fields_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            x=x,
            y=y,
            solid=solid,
            actual_u=actual_u,
            actual_v=actual_v,
            actual_omega=actual_omega,
            predicted_u=predicted_u,
            predicted_v=predicted_v,
            predicted_omega=predicted_omega,
            coefficient_frames=coefficients.T,
            steps=steps,
        )
    temporary.replace(fields_path)
    metrics_path = args.output_root / "per_frame_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "complete": True,
        "completed_at": now_iso(),
        "method": "Potential-Rayleigh",
        "modes": 2048,
        "operation": "independent velocity-fluctuation projection for every GT frame",
        "temporal_rollout": False,
        "frames": args.frames,
        "beta": beta,
        "alpha_deg": alpha,
        "coordinate_time_per_frame": 5.0,
        "projection_points": int(len(subset)),
        "projection_rank": int(np.count_nonzero(retained)),
        "gram_min_retained_eigenvalue": float(np.min(eigenvalues[retained])),
        "gram_max_eigenvalue": float(eigenvalues[-1]),
        "mean_velocity_error": float(np.mean([row["velocity_relative_l2"] for row in rows])),
        "mean_vorticity_error": float(np.mean([row["vorticity_relative_l2"] for row in rows])),
        "inputs": {
            "checkpoint": str(args.checkpoint.resolve()),
            "transform": str(args.transform.resolve()),
            "gt": str(args.gt.resolve()),
        },
        "outputs": {
            "fields": str(fields_path.resolve()),
            "metrics": str(metrics_path.resolve()),
        },
        "timing_seconds": {
            "projection_all_frames": projection_seconds,
            "full_grid_all_frames": field_seconds,
            "total": time.perf_counter() - started,
        },
    }
    summary_path = args.output_root / "summary.json"
    write_json(summary_path, summary)
    write_json(state_path, {"status": "complete", **summary})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
