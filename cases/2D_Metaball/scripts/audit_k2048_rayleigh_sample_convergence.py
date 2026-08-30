"""Audit K=2048 peeled velocity-Rayleigh convergence versus point count."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from evaluate_connected_dirichlet_psi_basis import load_model
from evaluate_velocity_k1024_three_unseen import _raw_velocity_data
from metaball_runtime import backend, excluded_area, install_backend, metaball_fluid_points


F32 = np.float32
CASE_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = CASE_ROOT / (
    "results/siren_k2048_connected_velocity_rayleigh_omega224_onepass_e2000"
)
RUN_DIR = RESULT_ROOT / "run_001"
DEFAULT_VALIDATION = RESULT_ROOT / "inputs/fixed_validation_iid_uniform_v1.npz"
DEFAULT_CHECKPOINTS = [
    RUN_DIR / "checkpoints/hardcoded_best_model.npz",
    RUN_DIR / "checkpoints/hardcoded_epoch0500.npz",
    RUN_DIR / "checkpoints/hardcoded_epoch1000.npz",
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def peel_prefix(
    cfg: Any,
    full_gradient: cp.ndarray,
    full_hessian: cp.ndarray,
    count: int,
    fluid_area: float,
) -> dict[str, Any]:
    """Evaluate the algebraically equivalent mass-Gram peeled basis.

    In exact arithmetic, MGS of the dynamically sorted velocity columns gives
    the inverse-transpose Cholesky transform of their mass Gram matrix.  Using
    that identity avoids O(N K^2) sequential tall-matrix updates while keeping
    the sampled mass, stiffness, causal order, and resulting Rayleigh values.
    The 4096-point results are checked against the production MGS history.
    """
    modes = int(cfg.modes)
    gradient = full_gradient[: 2 * count]
    hessian = full_hessian[: 3 * count]
    weight = cp.asarray(F32(fluid_area / count), dtype=backend.F32)
    hessian_metric = cp.tile(
        cp.asarray((1.0, 2.0, 1.0), dtype=backend.F32), count
    )
    mass_gram = weight * (gradient.T @ gradient)
    mass_gram = F32(0.5) * (mass_gram + mass_gram.T)
    weighted_hessian = hessian_metric[:, None] * hessian
    stiffness_gram = weight * (hessian.T @ weighted_hessian)
    stiffness_gram = F32(0.5) * (stiffness_gram + stiffness_gram.T)
    raw_mass = cp.diag(mass_gram)
    raw_stiffness = cp.diag(stiffness_gram)
    raw_rayleigh = raw_stiffness / raw_mass
    order = cp.argsort(raw_rayleigh, kind="stable")
    sorted_mass = mass_gram[order][:, order]
    sorted_stiffness = stiffness_gram[order][:, order]
    lower = cp.linalg.cholesky(sorted_mass)
    transform = cp.linalg.solve(
        lower.T,
        cp.eye(modes, dtype=backend.F32),
    )
    transformed_mass = transform.T @ sorted_mass @ transform
    stiffness_times_transform = sorted_stiffness @ transform
    peeled_rayleigh = cp.sum(
        transform * stiffness_times_transform, axis=0, dtype=backend.F32
    ) / cp.diag(transformed_mass)
    relative_masses = (cp.diag(lower) ** F32(2.0)) / cp.diag(sorted_mass)
    orth_error = cp.max(
        cp.abs(transformed_mass - cp.eye(modes, dtype=backend.F32))
    )
    packed = cp.asnumpy(
        cp.stack(
            (
                relative_masses,
                peeled_rayleigh,
                raw_rayleigh,
                order.astype(backend.F32),
            )
        )
    )
    relative, rayleigh, raw, order_float = packed
    order_host = order_float.astype(np.int64)
    rejected = (
        ~np.isfinite(relative)
        | ~np.isfinite(rayleigh)
        | (relative < float(cfg.peeling_relative_mass_floor))
    )
    orth_error_host = float(orth_error.get())
    if not np.isfinite(orth_error_host) or orth_error_host > float(
        cfg.peeling_overlap_tolerance
    ):
        rejected[:] = True
    first_rejected = int(np.flatnonzero(rejected)[0] + 1) if np.any(rejected) else None
    diagnostics = {
        "points": count,
        "peeled_rank": modes if first_rejected is None else first_rejected - 1,
        "accepted_under_protocol": first_rejected is None,
        "first_rejected_causal_rank": first_rejected,
        "min_relative_residual_mass": float(np.min(relative)),
        "max_post_peel_overlap": orth_error_host,
        "rayleigh_mean": float(np.mean(rayleigh, dtype=np.float64)),
        "rayleigh_median": float(np.median(rayleigh)),
        "rayleigh_p95": float(np.percentile(rayleigh, 95.0)),
        "rayleigh_min": float(np.min(rayleigh)),
        "rayleigh_max": float(np.max(rayleigh)),
        "raw_rayleigh_mean": float(np.mean(raw, dtype=np.float64)),
        "causal_order_1based": (order_host + 1).tolist(),
    }
    del mass_gram, stiffness_gram, weighted_hessian
    del sorted_mass, sorted_stiffness, lower, transform, transformed_mass
    del stiffness_times_transform, relative_masses, peeled_rayleigh, packed
    cp.get_default_memory_pool().free_all_blocks()
    return diagnostics


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3, default=DEFAULT_CHECKPOINTS
    )
    parser.add_argument("--validation-pack", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--point-counts", type=int, nargs=4, default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--extension-seed", type=int, default=30260823)
    args = parser.parse_args()
    if args.point_counts != [4096, 8192, 16384, 32768]:
        raise ValueError("audit point counts are fixed to 4096/8192/16384/32768")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = now()

    metadata0 = json.loads(
        args.checkpoints[0].with_suffix(".json").read_text(encoding="utf-8")
    )
    install_backend(metadata0["config"])
    cfg0 = backend.cfg_namespace(metadata0["config"])
    with np.load(args.validation_pack, allow_pickle=False) as archive:
        validation_points = np.asarray(archive["validation_points"], dtype=F32)
        validation_tokens = np.asarray(archive["validation_beta"], dtype=F32)
        validation_alphas = np.asarray(archive["validation_alpha"], dtype=F32)
    history_path = RUN_DIR / "data/hardcoded_training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    production_validation = {
        int(row["epoch"]): float(row["validation_loss"])
        for row in history
        if row.get("validation_loss") is not None
    }
    if validation_points.shape[1] != 4096:
        raise ValueError(f"expected 4096 fixed prefix points, got {validation_points.shape}")
    max_points = args.point_counts[-1]
    nested_points: list[np.ndarray] = []
    for geometry_index, (prefix, token, alpha) in enumerate(
        zip(validation_points, validation_tokens, validation_alphas)
    ):
        extension = metaball_fluid_points(
            cfg0,
            max_points - len(prefix),
            float(token),
            float(alpha),
            args.extension_seed + 1009 * geometry_index,
        )
        nested_points.append(np.concatenate((prefix, extension), axis=0).astype(F32))

    detail_rows: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    for checkpoint in args.checkpoints:
        metadata, cfg, model = load_model(checkpoint)
        epoch = int(metadata["epoch"])
        if int(cfg.modes) != 2048 or str(cfg.basis_objective) != "velocity_rayleigh":
            raise ValueError(f"invalid audit checkpoint {checkpoint}")
        if int(cfg.reorthogonalization_passes) != 1:
            raise ValueError("audit requires strict one-pass peeling")
        checkpoint_rows: list[dict[str, Any]] = []
        checkpoint_started = time.perf_counter()
        for geometry_index, (points, token, alpha) in enumerate(
            zip(nested_points, validation_tokens, validation_alphas)
        ):
            raw_started = time.perf_counter()
            full_gradient, full_hessian = _raw_velocity_data(
                cfg, model, points, float(token)
            )
            raw_seconds = time.perf_counter() - raw_started
            # The streamed Jet3 forward can leave large temporary blocks in
            # CuPy's cache.  Only the compact gradient/Hessian matrices are
            # needed by the prefix audits, so release cached temporaries before
            # allocating sorted and peeled workspaces.
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            fluid_area = float(
                (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
                - excluded_area(cfg, float(token))
            )
            print(
                json.dumps(
                    {
                        "stage": "raw_fields_ready",
                        "epoch": epoch,
                        "geometry": geometry_index + 1,
                        "token": float(token),
                        "max_points": max_points,
                        "seconds": raw_seconds,
                    }
                ),
                flush=True,
            )
            for point_count in args.point_counts:
                peel_started = time.perf_counter()
                diagnostics = peel_prefix(
                    cfg, full_gradient, full_hessian, point_count, fluid_area
                )
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                row = {
                    "checkpoint": checkpoint.name,
                    "epoch": epoch,
                    "geometry_index": geometry_index + 1,
                    "geometry_token": float(token),
                    "alpha": float(alpha),
                    "points": point_count,
                    "peel_seconds": time.perf_counter() - peel_started,
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if key != "causal_order_1based"
                    },
                }
                detail_rows.append(row)
                checkpoint_rows.append(row)
                print(json.dumps(row), flush=True)
            del full_gradient, full_hessian
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        for point_count in args.point_counts:
            selected = [row for row in checkpoint_rows if row["points"] == point_count]
            aggregate = {
                "checkpoint": checkpoint.name,
                "epoch": epoch,
                "points": point_count,
                "validation_geometry_count": len(selected),
                "validation_loss": float(
                    np.mean([row["rayleigh_mean"] for row in selected])
                ),
                "validation_loss_p95": float(
                    np.percentile([row["rayleigh_mean"] for row in selected], 95.0)
                ),
                "validation_loss_max": float(
                    np.max([row["rayleigh_mean"] for row in selected])
                ),
                "minimum_peeled_rank": int(
                    min(row["peeled_rank"] for row in selected)
                ),
                "all_geometries_accepted": bool(
                    all(row["accepted_under_protocol"] for row in selected)
                ),
                "minimum_relative_residual_mass": float(
                    min(row["min_relative_residual_mass"] for row in selected)
                ),
                "maximum_post_peel_overlap": float(
                    max(row["max_post_peel_overlap"] for row in selected)
                ),
            }
            if point_count == 4096:
                reference = production_validation[epoch]
                aggregate["production_mgs_validation_loss"] = reference
                aggregate["relative_difference_from_production_mgs"] = float(
                    abs(aggregate["validation_loss"] - reference) / reference
                )
            aggregates.append(aggregate)
            print(json.dumps({"stage": "aggregate", **aggregate}), flush=True)
        checkpoint_records.append(
            {
                "checkpoint": str(checkpoint.resolve()),
                "epoch": epoch,
                "seconds": time.perf_counter() - checkpoint_started,
            }
        )
        del model
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    rankings = {
        str(point_count): [
            {
                "epoch": int(row["epoch"]),
                "validation_loss": float(row["validation_loss"]),
            }
            for row in sorted(
                [row for row in aggregates if row["points"] == point_count],
                key=lambda row: row["validation_loss"],
            )
        ]
        for point_count in args.point_counts
    }
    rank_orders = [[item["epoch"] for item in rankings[str(n)]] for n in args.point_counts]
    summary = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": now(),
        "total_seconds": time.perf_counter() - started,
        "objective": "peeled velocity-Rayleigh",
        "audit_orthogonalization": (
            "inverse-transpose mass-Gram Cholesky, algebraically equivalent "
            "to full-rank dynamically sorted MGS"
        ),
        "modes": 2048,
        "peeling_passes": 1,
        "geometry_tokens": [float(value) for value in validation_tokens],
        "point_counts": args.point_counts,
        "nested_sampling": (
            "the original fixed 4096 validation points are the prefix; "
            "larger sets append deterministic IID physical-fluid points"
        ),
        "checkpoints": checkpoint_records,
        "aggregates": aggregates,
        "rankings": rankings,
        "ranking_stable_across_point_counts": all(
            order == rank_orders[0] for order in rank_orders[1:]
        ),
        "maximum_4096_relative_difference_from_production_mgs": float(
            max(
                row["relative_difference_from_production_mgs"]
                for row in aggregates
                if row["points"] == 4096
            )
        ),
    }
    detail_path = args.output_dir / "rayleigh_sample_convergence_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    aggregate_path = args.output_dir / "rayleigh_sample_convergence_aggregate.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    summary_path = args.output_dir / "rayleigh_sample_convergence_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
