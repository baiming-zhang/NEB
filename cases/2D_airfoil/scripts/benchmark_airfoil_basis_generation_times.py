#!/usr/bin/env python3
"""Benchmark learned peeling inference against fresh FD-Laplacian generation.

This benchmark intentionally performs no flow-field projection or reconstruction.
For each fixed unseen geometry and requested K, it measures (1) one cold learned
Jet1/GPU inference-plus-peeling pass on the configured IID fluid points and (2)
one independent geometry-aware finite-difference operator assembly/eigensolve.
Completed method/K records are resumable and are never recomputed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    fluid_points,
    peel_step,
)
from evaluate_original_k64_to_k1024_vs_laplacian import (  # noqa: E402
    SUPPORTED_K_VALUES,
    load_model_any,
)
from evaluate_unseen_k64_vs_laplacian import (  # noqa: E402
    geometry_laplacian,
    write_json,
)


CASES = (
    ("up", "gt_019.npz"),
    ("middle", "gt_015.npz"),
    ("down", "gt_014.npz"),
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def load_geometry(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "x": archive["x"].astype(np.float32),
            "y": archive["y"].astype(np.float32),
            "solid": archive["solid"].astype(bool),
            "metadata": json.loads(str(archive["metadata"])),
        }


def synchronize_gpu() -> None:
    cp.cuda.Stream.null.synchronize()


def benchmark_learned(
    checkpoint: Path,
    beta: float,
    alpha: float,
    peeling_points: int,
    seed_offset: int,
) -> dict[str, Any]:
    method_started = time.perf_counter_ns()
    load_started = time.perf_counter_ns()
    metadata, cfg, model = load_model_any(checkpoint)
    checkpoint_load_seconds = (time.perf_counter_ns() - load_started) / 1.0e9

    sampling_started = time.perf_counter_ns()
    points = fluid_points(
        cfg,
        peeling_points,
        beta,
        alpha,
        int(cfg.seed) + seed_offset,
    )
    sampling_seconds = (time.perf_counter_ns() - sampling_started) / 1.0e9

    synchronize_gpu()
    peeling_started = time.perf_counter_ns()
    diagnostics = peel_step(
        cfg,
        model,
        points,
        beta,
        alpha,
        training=False,
        full_diagnostics=True,
    )
    synchronize_gpu()
    peeling_inference_seconds = (
        time.perf_counter_ns() - peeling_started
    ) / 1.0e9
    total_seconds = (time.perf_counter_ns() - method_started) / 1.0e9

    result = {
        "complete": True,
        "completed_at": now(),
        "method": "learned_jet1_gpu_inference_plus_geometry_rayleigh_peeling",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "modes": int(cfg.modes),
        "peeling_points": int(len(points)),
        "timing_seconds": {
            "checkpoint_load": checkpoint_load_seconds,
            "iid_point_sampling": sampling_seconds,
            "peeling_inference": peeling_inference_seconds,
            "total_with_load_and_sampling": total_seconds,
        },
        "diagnostics": {
            "peeled_rank": int(diagnostics["peeled_rank"]),
            "orth_error": float(diagnostics["orth_error"]),
            "spatial_jet_order_used": int(diagnostics["spatial_jet_order_used"]),
            "causal_sort_device": diagnostics["causal_sort_device"],
            "diagnostic_host_transfers_per_step": int(
                diagnostics["diagnostic_host_transfers_per_step"]
            ),
            "optimizer_host_transfers_per_step": int(
                diagnostics["optimizer_host_transfers_per_step"]
            ),
        },
    }
    del diagnostics, model, points
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()
    return result


def benchmark_laplacian(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    modes: int,
    stride: int,
    seed: int,
) -> dict[str, Any]:
    basis = geometry_laplacian(x, y, solid, modes, stride, seed)
    diagnostics = basis["diagnostics"]
    result = {
        "complete": True,
        "completed_at": now(),
        "method": "fresh_geometry_aware_five_point_fd_laplacian",
        "modes": modes,
        "timing_seconds": diagnostics["timing_seconds"],
        "diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key != "timing_seconds"
        },
    }
    del basis
    gc.collect()
    return result


def completed_record(path: Path, modes: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if bool(record.get("complete")) and int(record.get("modes", -1)) == modes:
        return record
    return None


def write_summary(output_dir: Path, modes_values: tuple[int, ...]) -> Path:
    rows: list[dict[str, Any]] = []
    for orientation, _filename in CASES:
        for modes in modes_values:
            learned_path = output_dir / orientation / f"k{modes:04d}_learned.json"
            laplacian_path = (
                output_dir / orientation / f"k{modes:04d}_laplacian.json"
            )
            learned = completed_record(learned_path, modes)
            laplacian = completed_record(laplacian_path, modes)
            if learned is None or laplacian is None:
                continue
            learned_seconds = float(
                learned["timing_seconds"]["peeling_inference"]
            )
            laplacian_seconds = float(
                laplacian["timing_seconds"]["total_basis_generation"]
            )
            rows.append(
                {
                    "orientation": orientation,
                    "modes": modes,
                    "learned_peeling_inference_seconds": learned_seconds,
                    "learned_checkpoint_load_seconds": learned["timing_seconds"]["checkpoint_load"],
                    "learned_iid_point_sampling_seconds": learned["timing_seconds"]["iid_point_sampling"],
                    "learned_total_with_load_sampling_seconds": learned["timing_seconds"]["total_with_load_and_sampling"],
                    "laplacian_operator_assembly_seconds": laplacian["timing_seconds"]["operator_assembly"],
                    "laplacian_eigensolve_seconds": laplacian["timing_seconds"]["eigensolve"],
                    "laplacian_postprocess_seconds": laplacian["timing_seconds"]["postprocess"],
                    "laplacian_total_basis_generation_seconds": laplacian_seconds,
                    "laplacian_over_learned_speedup": laplacian_seconds / learned_seconds,
                    "learned_peeled_rank": learned["diagnostics"]["peeled_rank"],
                    "learned_orth_error": learned["diagnostics"]["orth_error"],
                    "laplacian_unknown_nodes": laplacian["diagnostics"]["unknown_nodes"],
                    "laplacian_max_relative_eigen_residual": laplacian["diagnostics"]["maximum_relative_eigen_residual"],
                }
            )
    csv_path = output_dir / "basis_generation_timing.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        output_dir / "basis_generation_timing_summary.json",
        {
            "updated_at": now(),
            "complete": len(rows) == len(CASES) * len(modes_values),
            "expected_rows": len(CASES) * len(modes_values),
            "completed_rows": len(rows),
            "csv": str(csv_path.resolve()),
            "rows": rows,
        },
    )
    return csv_path


def main() -> None:
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
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--laplacian-stride", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=20260824)
    args = parser.parse_args()

    modes_values = tuple(args.k_values)
    if modes_values != tuple(sorted(set(modes_values))):
        parser.error("--k-values must be unique and increasing")
    checkpoints = {
        modes: getattr(args, f"k{modes}_checkpoint") for modes in modes_values
    }
    missing = [modes for modes, path in checkpoints.items() if path is None]
    if missing:
        parser.error(f"missing checkpoint arguments for K={missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "benchmark_manifest.json",
        {
            "started_at": now(),
            "python": sys.version,
            "platform": platform.platform(),
            "k_values": list(modes_values),
            "cases": [label for label, _filename in CASES],
            "peeling_points": args.peeling_points,
            "laplacian_stride": args.laplacian_stride,
            "learned_definition": (
                "one cold Jet1 GPU forward, stable GPU geometry-Rayleigh sort, "
                "and two-pass GPU MGS; checkpoint load and IID sampling reported separately"
            ),
            "laplacian_definition": (
                "fresh per-K operator assembly, independent shift-invert eigsh, "
                "and basis postprocessing; no cache reuse"
            ),
            "no_flow_projection_or_reconstruction": True,
            "checkpoints": {
                f"k{modes}": str(path.resolve())
                for modes, path in checkpoints.items()
            },
        },
    )

    for case_index, (orientation, filename) in enumerate(CASES, start=1):
        geometry = load_geometry(args.gt_root / filename)
        beta = float(geometry["metadata"]["beta"])
        alpha = float(geometry["metadata"]["alpha_deg"])
        case_dir = args.output_dir / orientation
        case_dir.mkdir(exist_ok=True)
        for modes in modes_values:
            learned_path = case_dir / f"k{modes:04d}_learned.json"
            laplacian_path = case_dir / f"k{modes:04d}_laplacian.json"
            if completed_record(learned_path, modes) is None:
                print(
                    f"[{orientation}] learned peeling inference K{modes}",
                    flush=True,
                )
                learned = benchmark_learned(
                    checkpoints[modes],
                    beta,
                    alpha,
                    args.peeling_points,
                    2_300_001 + 10_000 * case_index,
                )
                learned.update(
                    {
                        "orientation": orientation,
                        "beta": beta,
                        "alpha_deg": alpha,
                    }
                )
                write_json(learned_path, learned)
            else:
                print(f"[{orientation}] skip completed learned K{modes}", flush=True)

            if completed_record(laplacian_path, modes) is None:
                print(
                    f"[{orientation}] fresh FD-Laplacian generation K{modes}",
                    flush=True,
                )
                laplacian = benchmark_laplacian(
                    geometry["x"],
                    geometry["y"],
                    geometry["solid"],
                    modes,
                    args.laplacian_stride,
                    args.random_seed + 1000 * case_index + modes,
                )
                laplacian.update(
                    {
                        "orientation": orientation,
                        "beta": beta,
                        "alpha_deg": alpha,
                    }
                )
                write_json(laplacian_path, laplacian)
            else:
                print(f"[{orientation}] skip completed FD K{modes}", flush=True)
            write_summary(args.output_dir, modes_values)
        del geometry
        gc.collect()

    csv_path = write_summary(args.output_dir, modes_values)
    print(
        json.dumps(
            {
                "complete": True,
                "completed_at": now(),
                "csv": str(csv_path.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
