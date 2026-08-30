"""Evaluate one velocity-Rayleigh checkpoint on three fixed unseen cases."""

from __future__ import annotations

import argparse
import csv
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
from evaluate_connected_dirichlet_psi_basis import load_model
from evaluate_velocity_k1024_three_unseen import (
    common_velocity_mass_whitening,
    learned_span,
)
from metaball_runtime import fluid_mask, geometry_from_token


F32 = np.float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[4_000_001, 4_000_002, 4_000_003]
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    parser.add_argument(
        "--learned-only",
        action="store_true",
        help="Skip the geometry-aware FD Laplacian baseline when it is already cached.",
    )
    args = parser.parse_args()
    if args.resolution != 128:
        raise ValueError("the strict comparison is fixed to 128 x 128")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, model = load_model(args.checkpoint)
    if str(cfg.basis_objective) != "velocity_rayleigh":
        raise ValueError("expected a velocity-Rayleigh checkpoint")
    if int(cfg.reorthogonalization_passes) != 1:
        raise ValueError("expected strict single-pass peeling")
    epoch = int(metadata["epoch"])
    modes = int(cfg.modes)
    learned_label = f"Learned-e{epoch}"

    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    total_started = time.perf_counter()

    for case_index, token_integer in enumerate(args.tokens):
        case_started = time.perf_counter()
        token = float(token_integer)
        valid = fluid_mask(cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        points = all_points[valid.ravel()]
        if len(points) <= modes + 1:
            raise ValueError(
                f"token {token_integer} has only {len(points)} nodes for K={modes}"
            )
        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]

        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(cfg, token, gt_seed, args.vortices)
        gt_psi, _, _, gt_normalization = ground_truth_fields(
            cfg, points, token, gt_parameters
        )
        gt_velocity, gt_omega = matched_grid_observables(gt_psi, valid, spacing)

        learned_started = time.perf_counter()
        learned_psi, learned_velocity, learned_omega, learned_diagnostics = (
            learned_span(
                cfg,
                model,
                points,
                valid,
                spacing,
                token,
                args.peeling_points,
                args.seed + 200_003 * (case_index + 1),
            )
        )
        learned_reconstruction = fit_velocity(
            learned_psi,
            learned_velocity,
            learned_omega,
            gt_psi,
            gt_velocity,
            gt_omega,
            fit_indices,
            test_indices,
            args.ridge_relative,
        )
        learned_result = {
            "metrics": learned_reconstruction["metrics"],
            "diagnostics": learned_diagnostics,
            "seconds": time.perf_counter() - learned_started,
        }

        results = {learned_label: learned_result}
        if not args.learned_only:
            laplacian_started = time.perf_counter()
            laplacian_values, laplacian_psi, _, _ = finite_difference_laplacian(
                valid, spacing, modes
            )
            (
                laplacian_psi,
                laplacian_velocity,
                laplacian_omega,
                laplacian_whitening,
            ) = common_velocity_mass_whitening(laplacian_psi, valid, spacing)
            laplacian_reconstruction = fit_velocity(
                laplacian_psi,
                laplacian_velocity,
                laplacian_omega,
                gt_psi,
                gt_velocity,
                gt_omega,
                fit_indices,
                test_indices,
                args.ridge_relative,
            )
            results["Laplacian"] = {
                "metrics": laplacian_reconstruction["metrics"],
                "eigenvalue_min": float(laplacian_values[0]),
                "eigenvalue_max": float(laplacian_values[-1]),
                "discrete_velocity_mass_whitening": laplacian_whitening,
                "seconds": time.perf_counter() - laplacian_started,
            }
        for label, result in results.items():
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "method": label,
                    **result["metrics"],
                }
            )
        cases.append(
            {
                "case": case_index + 1,
                "geometry_token": token_integer,
                "geometry_parameters": geometry_from_token(cfg, token).tolist(),
                "fluid_nodes": int(len(points)),
                "fit_nodes": int(len(fit_indices)),
                "held_out_test_nodes": int(len(test_indices)),
                "gt_seed": gt_seed,
                "gt_velocity_rms_before_normalization": gt_normalization,
                "gt_parameters": serializable_gt(gt_parameters),
                "results": results,
                "seconds": time.perf_counter() - case_started,
            }
        )
        print(
            json.dumps(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "test_errors": {
                        label: result["metrics"] for label, result in results.items()
                    },
                }
            ),
            flush=True,
        )

    metric_keys = (
        "test_psi_relative_l2",
        "test_velocity_relative_l2",
        "test_vorticity_relative_l2",
    )
    means = {
        label: {
            key: float(
                np.mean([case["results"][label]["metrics"][key] for case in cases])
            )
            for key in metric_keys
        }
        for label in results
    }
    wins = None
    if not args.learned_only:
        wins = {
            key: int(
                sum(
                    case["results"][learned_label]["metrics"][key]
                    < case["results"]["Laplacian"]["metrics"][key]
                    for case in cases
                )
            )
            for key in metric_keys
        }
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": epoch,
        "modes": modes,
        "geometry_tokens": args.tokens,
        "grid_resolution": args.resolution,
        "peeling_points": args.peeling_points,
        "learned_only": args.learned_only,
        "fit": "velocity least squares on 70% of valid nodes",
        "test": "relative L2 on disjoint held-out 30% valid nodes",
        "fairness_contract": {
            "same_geometry_mask_gt_split_and_mode_count": True,
            "same_boundary_condition": "psi=0 on every boundary component",
            "same_velocity_operator": "zero-extended centered difference",
            "same_vorticity_operator": "five-point positive -Delta",
            "same_invertible_velocity_mass_whitening": True,
            "learned_peeling_passes": 1,
        },
        "mean_test_errors": means,
        "learned_wins_vs_laplacian_out_of_three": wins,
        "cases": cases,
        "total_seconds": time.perf_counter() - total_started,
    }
    csv_path = args.output_dir / "three_unseen_reconstruction_metrics.csv"
    json_path = args.output_dir / "three_unseen_reconstruction_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary = json_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(json_path)
    print(json.dumps({"summary": str(json_path), "means": means}, indent=2), flush=True)


if __name__ == "__main__":
    main()
