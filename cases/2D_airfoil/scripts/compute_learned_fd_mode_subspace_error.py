#!/usr/bin/env python3
"""Measure directed FD-mode reconstruction error in learned velocity spans.

For each geometry and matched dimension K, every one of the first K
geometry-aware FD-Laplacian velocity modes is reconstructed by least squares
in the K-dimensional peeled learned span.  The reported "subspace error" is
the arithmetic mean of the K per-mode relative L2 residuals, as requested.
Both bases are evaluated on the same deterministic set of fluid-grid points.
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
import scipy.interpolate


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from evaluate_original_k64_to_k1024_vs_laplacian import load_model_any  # noqa: E402


K_VALUES = (64, 128, 256, 512, 1024, 2048)
ORIENTATIONS = ("up", "middle", "down")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def projection_points(gt_path: Path, count: int) -> np.ndarray:
    with np.load(gt_path, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
    xgrid, ygrid = np.meshgrid(x, y)
    fluid = ~solid
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    subset = np.linspace(0, len(points) - 1, min(count, len(points)), dtype=np.int64)
    result = points[subset]
    if len(result) != count:
        raise ValueError(f"requested {count} points but geometry has only {len(points)} fluid nodes")
    return result


def fd_velocity_on_points(
    cache_dir: Path,
    points: np.ndarray,
    modes: int,
    batch_modes: int,
) -> tuple[cp.ndarray, cp.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    x = np.load(cache_dir / "x.npy").astype(np.float32)
    y = np.load(cache_dir / "y.npy").astype(np.float32)
    psi = np.load(cache_dir / "psi.npy", mmap_mode="r")
    if psi.shape != (len(y), len(x), 2048) or modes > psi.shape[-1]:
        raise ValueError(f"invalid FD cache shape {psi.shape} at {cache_dir}")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    query = np.column_stack((points[:, 1], points[:, 0]))
    options = {"bounds_error": True, "method": "linear"}
    basis_u = cp.empty((len(points), modes), dtype=cp.float32)
    basis_v = cp.empty_like(basis_u)
    for start in range(0, modes, batch_modes):
        stop = min(start + batch_modes, modes)
        block = np.asarray(psi[..., start:stop], dtype=np.float32)
        gx = np.gradient(block, dx, axis=1, edge_order=2).astype(np.float32, copy=False)
        gy = np.gradient(block, dy, axis=0, edge_order=2).astype(np.float32, copy=False)
        gx[:, -1] = 0.0
        interp_gx = scipy.interpolate.RegularGridInterpolator((y, x), gx, **options)
        interp_gy = scipy.interpolate.RegularGridInterpolator((y, x), gy, **options)
        host_u = np.asarray(interp_gy(query), dtype=np.float32)
        host_v = -np.asarray(interp_gx(query), dtype=np.float32)
        basis_u[:, start:stop] = cp.asarray(host_u)
        basis_v[:, start:stop] = cp.asarray(host_v)
        print(f"    FD modes {start + 1}-{stop}/{modes}", flush=True)
        del block, gx, gy, interp_gx, interp_gy, host_u, host_v
        gc.collect()
    norms = cp.sqrt(cp.sum(basis_u * basis_u + basis_v * basis_v, axis=0))
    if bool(cp.any(~cp.isfinite(norms)).get()) or float(cp.min(norms).get()) <= 0.0:
        raise ValueError("non-finite or zero FD velocity-mode norm")
    basis_u /= norms[None, :]
    basis_v /= norms[None, :]
    diagnostics = {
        "seconds": time.perf_counter() - started,
        "minimum_pre_normalization_norm": float(cp.min(norms).get()),
        "maximum_pre_normalization_norm": float(cp.max(norms).get()),
        "cache": str(cache_dir.resolve()),
        "interpolation": "finite-difference gradients followed by bilinear interpolation",
    }
    return basis_u, basis_v, diagnostics


def learned_velocity_on_points(
    checkpoint: Path,
    state_path: Path,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
) -> tuple[cp.ndarray, cp.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    metadata, cfg, model = load_model_any(checkpoint)
    modes = int(cfg.modes)
    with np.load(state_path, allow_pickle=False) as archive:
        transform = archive["transform"].astype(np.float32)
    if transform.shape != (modes, modes):
        raise ValueError(f"invalid peeled transform {transform.shape} for K={modes}")
    raw_gx = cp.empty((len(points), modes), dtype=cp.float32)
    raw_gy = cp.empty_like(raw_gx)
    for network, field, _cache in model.forward_stream(cfg, points, beta, alpha_deg, False):
        index = int(network.mode_index)
        raw_gx[:, index] = field.grad[:, 0, 0]
        raw_gy[:, index] = field.grad[:, 0, 1]
    transform_gpu = cp.asarray(transform)
    basis_u = raw_gy @ transform_gpu
    basis_v = -(raw_gx @ transform_gpu)
    del raw_gx, raw_gy, transform_gpu, transform, model
    gc.collect()
    diagnostics = {
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "spatial_derivative_order": int(metadata["config"]["spatial_derivative_order"]),
        "peeled_transform": str(state_path.resolve()),
    }
    return basis_u, basis_v, diagnostics


def directed_fd_mode_errors(
    learned_u: cp.ndarray,
    learned_v: cp.ndarray,
    fd_u: cp.ndarray,
    fd_v: cp.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    modes = learned_u.shape[1]
    learned_norms = cp.sqrt(cp.sum(learned_u * learned_u + learned_v * learned_v, axis=0))
    if float(cp.min(learned_norms).get()) <= 0.0:
        raise ValueError("zero learned velocity-mode norm")
    learned_u = learned_u / learned_norms[None, :]
    learned_v = learned_v / learned_norms[None, :]
    gram = learned_u.T @ learned_u + learned_v.T @ learned_v
    gram = cp.float32(0.5) * (gram + gram.T)
    cross = learned_u.T @ fd_u + learned_v.T @ fd_v
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    largest = float(eigenvalues[-1].get())
    tolerance = float(largest * modes * np.finfo(np.float32).eps * 10.0)
    valid = eigenvalues > tolerance
    rank = int(cp.count_nonzero(valid).get())
    coordinates = eigenvectors[:, valid].T @ cross
    captured = cp.sum((coordinates * coordinates) / eigenvalues[valid, None], axis=0)
    captured = cp.clip(captured, 0.0, 1.0)
    errors = cp.sqrt(cp.maximum(0.0, 1.0 - captured))
    direct_max_abs_difference = None
    if modes <= 128:
        coefficients = eigenvectors[:, valid] @ (coordinates / eigenvalues[valid, None])
        residual_u = learned_u @ coefficients - fd_u
        residual_v = learned_v @ coefficients - fd_v
        direct = cp.sqrt(cp.sum(residual_u * residual_u + residual_v * residual_v, axis=0))
        direct_max_abs_difference = float(cp.max(cp.abs(direct - errors)).get())
        del coefficients, residual_u, residual_v, direct
    errors_host = cp.asnumpy(errors).astype(np.float64)
    diagnostics = {
        "seconds": time.perf_counter() - started,
        "learned_sample_rank": rank,
        "requested_rank": modes,
        "gram_minimum_eigenvalue": float(eigenvalues[0].get()),
        "gram_maximum_eigenvalue": largest,
        "gram_condition_number": float(largest / max(float(eigenvalues[0].get()), np.finfo(np.float32).tiny)),
        "rank_tolerance": tolerance,
        "direct_formula_max_abs_difference": direct_max_abs_difference,
    }
    if rank != modes:
        raise ValueError(f"learned sample rank {rank} is smaller than requested K={modes}")
    if not np.all(np.isfinite(errors_host)):
        raise ValueError("non-finite per-mode reconstruction error")
    return errors_host, diagnostics


def aggregate(
    output_root: Path,
    orientations: tuple[str, ...],
    modes_values: tuple[int, ...],
    laplacian_cache_root: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_mode_rows: list[dict[str, Any]] = []
    for orientation in orientations:
        for modes in modes_values:
            path = output_root / orientation / f"k{modes:04d}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            errors = np.asarray(payload["per_mode_relative_errors"], dtype=np.float64)
            eigenvalues = np.asarray(
                np.load(laplacian_cache_root / orientation / "eigenvalues.npy")[:modes],
                dtype=np.float64,
            )
            if len(errors) != modes or len(eigenvalues) != modes:
                raise ValueError(f"incomplete per-mode data for {orientation} K={modes}")
            if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0.0):
                raise ValueError(f"invalid FD eigenvalues for {orientation} K={modes}")
            inverse_eigenvalue_weights = 1.0 / eigenvalues
            inverse_eigenvalue_weighted_mean = float(
                np.sum(inverse_eigenvalue_weights * errors) / np.sum(inverse_eigenvalue_weights)
            )
            inverse_eigenvalue_weighted_rms = float(
                np.sqrt(
                    np.sum(inverse_eigenvalue_weights * errors * errors)
                    / np.sum(inverse_eigenvalue_weights)
                )
            )
            payload["inverse_eigenvalue_weighted_mean_error"] = inverse_eigenvalue_weighted_mean
            payload["inverse_eigenvalue_weighted_rms_error"] = inverse_eigenvalue_weighted_rms
            payload["weighting"] = {
                "primary_metric": "inverse_eigenvalue_weighted_rms_error",
                "weights": "w_i = 1 / lambda_i",
                "normalization": "sum_i w_i",
            }
            write_json(path, payload)
            rows.append(
                {
                    "orientation": orientation,
                    "modes": modes,
                    "mean_error": payload["mean_error"],
                    "median_error": payload["median_error"],
                    "rms_error": payload["rms_error"],
                    "p95_error": payload["p95_error"],
                    "maximum_error": payload["maximum_error"],
                    "inverse_eigenvalue_weighted_mean_error": inverse_eigenvalue_weighted_mean,
                    "inverse_eigenvalue_weighted_rms_error": inverse_eigenvalue_weighted_rms,
                    "sample_points": payload["sample_points"],
                }
            )
            per_mode_rows.extend(
                {
                    "orientation": orientation,
                    "modes": modes,
                    "fd_mode_index_1based": index + 1,
                    "fd_eigenvalue": float(eigenvalues[index]),
                    "inverse_eigenvalue_weight": float(inverse_eigenvalue_weights[index]),
                    "relative_error": error,
                }
                for index, error in enumerate(payload["per_mode_relative_errors"])
            )
    with (output_root / "subspace_error_by_geometry_and_k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["orientation", "modes"])
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "per_fd_mode_reconstruction_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(per_mode_rows[0]) if per_mode_rows else ["orientation", "modes", "fd_mode_index_1based", "relative_error"],
        )
        writer.writeheader()
        writer.writerows(per_mode_rows)
    summary_rows: list[dict[str, Any]] = []
    for modes in modes_values:
        matching = [row for row in rows if row["modes"] == modes]
        if len(matching) == len(orientations):
            summary_rows.append(
                {
                    "modes": modes,
                    "inverse_eigenvalue_weighted_rms_mean_over_geometries": float(
                        np.mean([row["inverse_eigenvalue_weighted_rms_error"] for row in matching])
                    ),
                    "inverse_eigenvalue_weighted_rms_by_geometry": {
                        row["orientation"]: row["inverse_eigenvalue_weighted_rms_error"] for row in matching
                    },
                    "arithmetic_mean_over_geometries": float(
                        np.mean([row["mean_error"] for row in matching])
                    ),
                    "arithmetic_mean_by_geometry": {
                        row["orientation"]: row["mean_error"] for row in matching
                    },
                }
            )
    summary = {
        "definition": "inverse-FD-eigenvalue-weighted RMS over the first K FD-Laplacian velocity modes of each mode's relative L2 least-squares residual in the matched K-dimensional learned velocity span",
        "formula": "sqrt(sum_i ((1/lambda_i) * e_i^2) / sum_i (1/lambda_i))",
        "unweighted_statistics_retained": True,
        "direction": "FD-Laplacian modes reconstructed by learned modes",
        "complete": len(rows) == len(orientations) * len(modes_values),
        "orientations": list(orientations),
        "modes": list(modes_values),
        "rows": summary_rows,
    }
    write_json(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    default_sequence = CASE_ROOT / "results" / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=default_sequence / "evaluation_three_geometries_frame30")
    parser.add_argument(
        "--laplacian-cache-root",
        type=Path,
        default=CASE_ROOT / "results" / "siren_original_k64_to_k1024_e2000_sequence" / "shared_geometry_laplacian_cache_k2048",
    )
    parser.add_argument("--output-root", type=Path, default=default_sequence / "fd_mode_subspace_error_three_geometries")
    parser.add_argument("--orientations", nargs="+", choices=ORIENTATIONS, default=list(ORIENTATIONS))
    parser.add_argument("--modes", nargs="+", type=int, choices=K_VALUES, default=list(K_VALUES))
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--fd-batch-modes", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orientations = tuple(args.orientations)
    modes_values = tuple(sorted(set(args.modes)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.evaluation_root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    checkpoints = {modes: Path(manifest["checkpoints"][f"k{modes}"]) for modes in modes_values}
    state = {
        "started_at": now(),
        "status": "running",
        "orientations": list(orientations),
        "modes": list(modes_values),
        "projection_points": args.projection_points,
    }
    write_json(args.output_root / "state.json", state)
    total_started = time.perf_counter()
    for orientation in orientations:
        completed = {
            modes
            for modes in modes_values
            if (args.output_root / orientation / f"k{modes:04d}.json").exists()
        }
        remaining = [modes for modes in modes_values if modes not in completed]
        if not remaining:
            print(f"[{orientation}] all requested K values already complete", flush=True)
            continue
        geometry_started = time.perf_counter()
        summary = json.loads((args.evaluation_root / orientation / "summary.json").read_text(encoding="utf-8"))
        gt_path = Path(summary["gt"])
        beta = float(summary["beta"])
        alpha_deg = float(summary["alpha_deg"])
        points = projection_points(gt_path, args.projection_points)
        max_modes = max(remaining)
        print(f"[{orientation}] {len(points)} shared points; build FD K={max_modes}", flush=True)
        fd_u, fd_v, fd_diagnostics = fd_velocity_on_points(
            args.laplacian_cache_root / orientation,
            points,
            max_modes,
            args.fd_batch_modes,
        )
        for modes in remaining:
            state.update({"orientation": orientation, "current_modes": modes, "updated_at": now()})
            write_json(args.output_root / "state.json", state)
            print(f"[{orientation}] reconstruct first {modes} FD modes with learned K={modes}", flush=True)
            learned_u, learned_v, learned_diagnostics = learned_velocity_on_points(
                checkpoints[modes],
                args.evaluation_root / orientation / "basis_state" / f"learned_k{modes:04d}_state.npz",
                points,
                beta,
                alpha_deg,
            )
            errors, solve_diagnostics = directed_fd_mode_errors(
                learned_u,
                learned_v,
                fd_u[:, :modes],
                fd_v[:, :modes],
            )
            payload = {
                "complete": True,
                "orientation": orientation,
                "modes": modes,
                "beta": beta,
                "alpha_deg": alpha_deg,
                "sample_points": len(points),
                "mean_error": float(np.mean(errors)),
                "median_error": float(np.median(errors)),
                "rms_error": float(math.sqrt(float(np.mean(errors * errors)))),
                "p95_error": float(np.percentile(errors, 95.0)),
                "maximum_error": float(np.max(errors)),
                "per_mode_relative_errors": errors.tolist(),
                "fd_diagnostics": fd_diagnostics,
                "learned_diagnostics": learned_diagnostics,
                "solve_diagnostics": solve_diagnostics,
                "completed_at": now(),
            }
            write_json(args.output_root / orientation / f"k{modes:04d}.json", payload)
            print(
                f"[{orientation}] K={modes} mean={100.0 * payload['mean_error']:.4f}% "
                f"median={100.0 * payload['median_error']:.4f}% "
                f"max={100.0 * payload['maximum_error']:.4f}%",
                flush=True,
            )
            del learned_u, learned_v, errors
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        del fd_u, fd_v, points
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        print(f"[{orientation}] wall={time.perf_counter() - geometry_started:.1f}s", flush=True)
    final_summary = aggregate(args.output_root, orientations, modes_values, args.laplacian_cache_root)
    state.update(
        {
            "status": "complete" if final_summary["complete"] else "partial",
            "completed_at": now(),
            "total_wall_seconds": time.perf_counter() - total_started,
        }
    )
    write_json(args.output_root / "state.json", state)
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
