#!/usr/bin/env python3
"""Project the initial airfoil vorticity fluctuation into three K=2048 spans."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np

from rollout_gt30_fd_vs_potential_k2048 import (
    LABELS,
    build_masked_cartesian_complex,
    gpu_ritz,
    load_hardcoded_model,
    load_npz,
    mimetic_fields_from_psi,
    potential_peeled_psi_on_active_grid,
    sha256,
    write_json,
)


METHODS = ("potential", "velocity", "vorticity")


def weighted_relative(residual: cp.ndarray, target: cp.ndarray, weight: cp.ndarray) -> float:
    numerator = cp.sum(weight * residual * residual, dtype=cp.float64)
    denominator = cp.sum(weight * target * target, dtype=cp.float64)
    return float(cp.sqrt(numerator / cp.maximum(denominator, cp.float64(1.0e-30))).get())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--laplacian-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--orientation", default="up", choices=("up", "middle", "down"))
    parser.add_argument("--modes", type=int, default=2048)
    parser.add_argument("--psi-point-batch", type=int, default=8192)
    parser.add_argument("--fd-batch-modes", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = json.loads((args.evaluation_root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((args.evaluation_root / args.orientation / "summary.json").read_text(encoding="utf-8"))
    gt = load_npz(Path(summary["gt"]))
    beta = float(summary["beta"])
    alpha = float(summary["alpha_deg"])
    cache = args.laplacian_cache_root / args.orientation
    x = np.load(cache / "x.npy").astype(np.float64)
    y = np.load(cache / "y.npy").astype(np.float64)
    solid = np.load(cache / "solid.npy").astype(bool)
    complex_ = build_masked_cartesian_complex(x, y, solid)

    stride = 2
    archived_mean_omega = gt["vorticity"].mean(axis=0, dtype=np.float64)[::stride, ::stride].astype(np.float32)
    initial_omega = gt["vorticity"][0, ::stride, ::stride].astype(np.float32)
    target_fluctuation = initial_omega[complex_.unknown] - archived_mean_omega[complex_.unknown]
    target_total = initial_omega[complex_.unknown]
    weight_gpu = cp.asarray(complex_.weights, dtype=cp.float32)
    target_gpu = cp.asarray(target_fluctuation, dtype=cp.float32)
    total_gpu = cp.asarray(target_total, dtype=cp.float32)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        method_started = time.perf_counter()
        checkpoint = Path(manifest["checkpoints"][method]["path"])
        peeled_state = args.evaluation_root / args.orientation / "basis_state" / f"{method}_k2048_prefix_state.npz"
        with np.load(peeled_state, allow_pickle=False) as archive:
            transform = archive["transform_network_to_causal"].astype(np.float32)
        cfg, model, checkpoint_metadata = load_hardcoded_model(checkpoint)
        if int(cfg.modes) != args.modes or transform.shape != (args.modes, args.modes):
            raise ValueError(f"invalid K for {method}")

        workspace = args.output_root / f"{method}_psi_active.tmp.f32"
        active_psi, psi_diag = potential_peeled_psi_on_active_grid(
            model, cfg, complex_, beta, alpha, transform,
            args.psi_point_batch, workspace, method,
        )
        raw_u, raw_v, raw_omega, chain_diag = mimetic_fields_from_psi(
            complex_, args.modes, args.fd_batch_modes,
            LABELS[method].replace("--", "-"), active_psi=active_psi,
        )
        basis_u, basis_v, basis_omega, eigenvalues, _rotation, ritz_diag = gpu_ritz(
            raw_u, raw_v, raw_omega, complex_.weights, args.modes,
            LABELS[method].replace("--", "-"),
        )
        del basis_u, basis_v
        eigen_gpu = cp.asarray(eigenvalues, dtype=cp.float64)
        right = basis_omega.T @ (weight_gpu * target_gpu)
        coefficients = right.astype(cp.float64) / cp.maximum(eigen_gpu, cp.float64(1.0e-30))
        estimate_fluctuation = basis_omega @ coefficients.astype(cp.float32)
        residual = estimate_fluctuation - target_gpu
        fluctuation_error = weighted_relative(residual, target_gpu, weight_gpu)
        total_error = weighted_relative(residual, total_gpu, weight_gpu)
        coefficient_norm = float(cp.linalg.norm(coefficients).get())
        finite = all(math.isfinite(v) for v in (fluctuation_error, total_error, coefficient_norm))
        if not finite:
            raise FloatingPointError(f"non-finite vorticity projection for {method}")
        row = {
            "method": method,
            "label": LABELS[method],
            "modes": args.modes,
            "orientation": args.orientation,
            "projection_field": "archived GT vorticity fluctuation at initial frame",
            "mean_restored_for_total_metric": True,
            "initial_total_vorticity_relative_l2": total_error,
            "initial_fluctuation_vorticity_relative_l2": fluctuation_error,
            "coefficient_l2": coefficient_norm,
            "rank": int(ritz_diag["rank"]),
            "mass_orthogonality_max_abs_fp32": float(ritz_diag["mass_orthogonality_max_abs_fp32"]),
            "ritz_eigenvalue_min": float(eigenvalues[0]),
            "ritz_eigenvalue_max": float(eigenvalues[-1]),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
            "psi_seconds": float(psi_diag["seconds"]),
            "mimetic_fields_seconds": float(chain_diag["seconds"]),
            "ritz_seconds": float(ritz_diag["seconds"]),
            "total_seconds": time.perf_counter() - method_started,
        }
        rows.append(row)
        write_json(args.output_root / f"{method}.json", {"complete": True, **row})
        print(json.dumps(row, indent=2), flush=True)

        del basis_omega, right, coefficients, estimate_fluctuation, residual
        del cfg, model, active_psi
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        if workspace.exists():
            workspace.unlink()

    csv_path = args.output_root / "initial_vorticity_projection_k2048.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.output_root / "summary.json",
        {
            "complete": True,
            "orientation": args.orientation,
            "modes": args.modes,
            "frame": 0,
            "projection": "weighted L2 least squares of archived GT vorticity fluctuation",
            "error": "weighted relative L2 after restoring the shared 41-frame mean",
            "rows": rows,
            "total_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
