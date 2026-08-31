"""Evaluate geometry-aware FD-Laplacian K=1024 on ten unseen metaballs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from compare_three_gt_eigenfluid_vs_laplacian import (
    fit_velocity,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
    serializable_gt,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_velocity_k1024_three_unseen import common_velocity_mass_whitening
from metaball_runtime import backend, fluid_mask, geometry_from_token, install_backend


F32 = np.float32
CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    CASE_ROOT
    / "configs/k1024_metaball_connected_siren_velocity_rayleigh_omega160_onepass_2000.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-start", type=int, default=4_000_001)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--modes", type=int, default=1024)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.cases != 10:
        raise ValueError("this audit is fixed to ten unseen cases")
    if args.modes != 1024:
        raise ValueError("this audit is fixed to K=1024")
    if args.resolution != 128:
        raise ValueError("this audit is fixed to a 128 x 128 grid")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for case_index in range(args.cases):
        token_integer = args.token_start + case_index
        token = float(token_integer)
        case_started = time.perf_counter()
        valid = fluid_mask(cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        points = all_points[valid.ravel()]
        if len(points) <= args.modes + 1:
            raise ValueError(
                f"token {token_integer} has only {len(points)} fluid nodes"
            )

        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]
        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(
            cfg, token, gt_seed, args.vortices
        )
        gt_psi, _, _, gt_normalization = ground_truth_fields(
            cfg, points, token, gt_parameters
        )
        gt_velocity, gt_omega = matched_grid_observables(
            gt_psi, valid, spacing
        )

        eigenvalues, psi, raw_omega, raw_velocity = finite_difference_laplacian(
            valid, spacing, args.modes
        )
        del raw_omega, raw_velocity
        psi, velocity, omega, whitening = common_velocity_mass_whitening(
            psi, valid, spacing
        )
        reconstruction = fit_velocity(
            psi,
            velocity,
            omega,
            gt_psi,
            gt_velocity,
            gt_omega,
            fit_indices,
            test_indices,
            args.ridge_relative,
        )
        elapsed = time.perf_counter() - case_started
        metrics = reconstruction["metrics"]
        row = {
            "case": case_index + 1,
            "geometry_token": token_integer,
            "fluid_nodes": int(len(points)),
            "fit_nodes": int(len(fit_indices)),
            "test_nodes": int(len(test_indices)),
            "test_psi_relative_l2": metrics["test_psi_relative_l2"],
            "test_velocity_relative_l2": metrics["test_velocity_relative_l2"],
            "test_vorticity_relative_l2": metrics["test_vorticity_relative_l2"],
            "effective_rank": metrics["fit_velocity_gram_effective_rank"],
            "velocity_gram_condition_number": metrics[
                "fit_velocity_gram_condition_number"
            ],
            "eigenvalue_min": float(eigenvalues[0]),
            "eigenvalue_max": float(eigenvalues[-1]),
            "seconds": elapsed,
        }
        rows.append(row)
        cases.append(
            {
                **row,
                "geometry_parameters": geometry_from_token(cfg, token).tolist(),
                "gt_seed": gt_seed,
                "gt_velocity_rms_before_normalization": gt_normalization,
                "gt_parameters": serializable_gt(gt_parameters),
                "ridge": metrics["ridge"],
                "coefficient_l2": metrics["coefficient_l2"],
                "velocity_mass_whitening": whitening,
            }
        )
        print(json.dumps(row), flush=True)
        del psi, velocity, omega, reconstruction
        gc.collect()

    metric_keys = (
        "test_psi_relative_l2",
        "test_velocity_relative_l2",
        "test_vorticity_relative_l2",
    )
    statistics = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
            "p95": float(np.percentile([row[key] for row in rows], 95.0)),
            "max": float(np.max([row[key] for row in rows])),
            "worst_case": int(max(rows, key=lambda row: row[key])["case"]),
        }
        for key in metric_keys
    }
    summary = {
        "status": "complete",
        "case": "2D connected three-circle smooth metaball",
        "geometry_status": "ten fixed unseen cases",
        "geometry_tokens": [row["geometry_token"] for row in rows],
        "method": "geometry-aware five-point Dirichlet FD Laplacian",
        "modes": args.modes,
        "grid_resolution": args.resolution,
        "boundary_condition": "psi=0 on every boundary component",
        "velocity_operator": "zero-extended centered difference",
        "vorticity_operator": "five-point positive -Delta",
        "fit": "velocity least squares on 70% of valid nodes",
        "test": "relative L2 on disjoint held-out 30% valid nodes",
        "statistics": statistics,
        "all_effective_ranks": [row["effective_rank"] for row in rows],
        "total_seconds": time.perf_counter() - total_started,
        "cases": cases,
    }
    csv_path = args.output_dir / "fdlaplacian_k1024_ten_unseen_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "fdlaplacian_k1024_ten_unseen_summary.json"
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    print(json.dumps({"summary": str(json_path), "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
