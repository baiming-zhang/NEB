"""List the first 16 Eigenfluid and FD-Laplacian eigenvalues per geometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cupy as cp
import numpy as np

from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_connected_dirichlet_psi_basis import (
    load_model,
    numerical_ritz_transform,
    peeled_transform,
    raw_fields,
)
from metaball_runtime import excluded_area, fluid_mask, metaball_fluid_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[4_000_001, 4_000_002, 4_000_003]
    )
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, model = load_model(args.checkpoint)
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=np.float32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    box_area = float((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin))

    rows: list[dict] = []
    cases: list[dict] = []
    for case_index, token_integer in enumerate(args.tokens):
        token = float(token_integer)
        reference = metaball_fluid_points(
            cfg,
            args.peeling_points,
            token,
            0.0,
            args.seed + 200_003 * (case_index + 1) + 701,
        )
        value, gradient = raw_fields(cfg, model, reference, token)
        fluid_area = box_area - excluded_area(cfg, token)
        peeling, peeling_diagnostics = peeled_transform(
            cfg, value, gradient, fluid_area
        )
        total, eigenfluid_values, ritz_diagnostics = numerical_ritz_transform(
            value, gradient, peeling, fluid_area
        )
        del value, gradient, peeling, total
        cp.get_default_memory_pool().free_all_blocks()

        valid = fluid_mask(cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        laplacian_values, laplacian_psi, laplacian_omega, laplacian_velocity = (
            finite_difference_laplacian(valid, spacing, 16)
        )
        del laplacian_psi, laplacian_omega, laplacian_velocity

        eigenfluid_first = eigenfluid_values[:16].astype(float)
        laplacian_first = laplacian_values[:16].astype(float)
        for mode, (eigenfluid, laplacian) in enumerate(
            zip(eigenfluid_first, laplacian_first), start=1
        ):
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "mode": mode,
                    "eigenfluid_eigenvalue": float(eigenfluid),
                    "laplacian_eigenvalue": float(laplacian),
                    "relative_difference": float(
                        abs(eigenfluid - laplacian) / laplacian
                    ),
                }
            )
        cases.append(
            {
                "case": case_index + 1,
                "geometry_token": token_integer,
                "eigenfluid_first16": eigenfluid_first.tolist(),
                "laplacian_first16": laplacian_first.tolist(),
                "relative_difference_first16": [
                    float(abs(left - right) / right)
                    for left, right in zip(eigenfluid_first, laplacian_first)
                ],
                "eigenfluid_peeling": {
                    "peeled_rank": int(peeling_diagnostics["peeled_rank"]),
                    "peeled_orthogonality_max_abs": float(
                        peeling_diagnostics["peeled_orthogonality_max_abs"]
                    ),
                    "causal_order_1based": peeling_diagnostics[
                        "causal_order_1based"
                    ],
                },
                "ritz_mass_orthogonality_max_abs": float(
                    ritz_diagnostics["ritz_mass_orthogonality_max_abs"]
                ),
            }
        )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "modes_reported": 16,
        "eigenfluid_definition": (
            "geometry-Rayleigh causal sorting, two-pass streamfunction peeling, "
            "then numerical generalized Ritz ordering"
        ),
        "laplacian_definition": (
            "geometry-specific five-point finite-difference Dirichlet "
            f"streamfunction Laplacian on a {args.resolution}x{args.resolution} grid"
        ),
        "cases": cases,
    }
    csv_path = args.output_dir / "first16_eigenvalues.csv"
    json_path = args.output_dir / "first16_eigenvalues.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
