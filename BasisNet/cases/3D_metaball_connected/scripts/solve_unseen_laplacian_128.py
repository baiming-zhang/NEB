"""Solve and cache the unseen-geometry 128^3 Dirichlet Laplacian modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyamg
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))
import evaluate_unseen_vs_laplacian as base  # noqa: E402


CONFIG_PATH = CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_500.json"
DEFAULT_OUTPUT = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction_128"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def initial_subspace(
    cfg: Any,
    geometry: np.ndarray,
    high_points: np.ndarray,
    scalar_modes: int,
    low_resolution: int,
) -> np.ndarray:
    print(f"build low-resolution initial subspace: {low_resolution}^3", flush=True)
    low_grid = base.build_grid(cfg, low_resolution, geometry)
    low_matrix, _ = base.assemble_dirichlet_laplacian(
        low_grid["mask"], low_grid["spacing"]
    )
    _, low_vectors, _ = base.solve_scalar_laplacian(
        low_matrix,
        scalar_modes,
        low_grid["spacing"],
        1.0e-8,
    )
    query_zyx = high_points[:, (2, 1, 0)].astype(np.float64)
    axes = (low_grid["coordinates"],) * 3
    initial = np.empty((len(high_points), scalar_modes), dtype=np.float32)
    for mode in range(scalar_modes):
        volume = np.zeros(low_grid["shape"], dtype=np.float64)
        volume[low_grid["mask"]] = low_vectors[:, mode]
        interpolator = RegularGridInterpolator(
            axes,
            volume,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        initial[:, mode] = interpolator(query_zyx).astype(np.float32)
        print(f"initial subspace prolonged: {mode + 1}/{scalar_modes}", flush=True)
    rng = np.random.default_rng(cfg.seed + 128_000_003)
    rms = np.sqrt(np.mean(initial**2, axis=0, keepdims=True)).clip(min=1.0e-8)
    initial += 1.0e-4 * rms * rng.standard_normal(initial.shape, dtype=np.float32)
    return initial


def relative_residuals(
    matrix: Any, eigenvalues: np.ndarray, eigenvectors: np.ndarray
) -> np.ndarray:
    residual = matrix @ eigenvectors - eigenvectors * eigenvalues[None, :]
    numerator = np.linalg.norm(residual, axis=0)
    denominator = np.linalg.norm(eigenvectors * eigenvalues[None, :], axis=0)
    return numerator / np.maximum(denominator, 1.0e-20)


def solve(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    eigenvector_path = output / f"laplacian_scalar_eigenvectors_{args.resolution}.npy"
    metadata_path = output / f"laplacian_scalar_metadata_{args.resolution}.json"
    if metadata_path.exists() and eigenvector_path.exists() and not args.force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("complete"):
            print(f"reuse complete cache: {metadata_path}", flush=True)
            return metadata

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = base.training.Config(**config)
    base.training.validate_contract(cfg)
    geometry = base.training.geometry_from_token(cfg, args.geometry_token)
    grid = base.build_grid(cfg, args.resolution, geometry)
    spacing = grid["spacing"]
    started = time.perf_counter()
    print(
        f"assemble 128^3 geometry token={args.geometry_token} "
        f"fluid_nodes={len(grid['points'])} spacing={spacing:.9f}",
        flush=True,
    )
    matrix, _ = base.assemble_dirichlet_laplacian(grid["mask"], spacing)
    matrix = matrix.astype(np.float32)
    matrix *= np.float32(spacing**2)
    print(
        f"matrix ready shape={matrix.shape} nnz={matrix.nnz} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )

    partial_path = output / f"laplacian_scalar_partial_{args.resolution}.npy"
    if partial_path.exists() and not args.force:
        initial = np.load(partial_path).astype(np.float32, copy=False)
        print(f"resume LOBPCG subspace from {partial_path}", flush=True)
    else:
        initial = initial_subspace(
            cfg,
            geometry,
            grid["points"],
            args.scalar_modes,
            args.low_resolution,
        )
        np.save(partial_path, initial)

    print("build smoothed-aggregation multigrid preconditioner", flush=True)
    multigrid_started = time.perf_counter()
    multigrid = pyamg.smoothed_aggregation_solver(
        matrix,
        symmetry="symmetric",
        max_coarse=2000,
    )
    preconditioner = multigrid.aspreconditioner(cycle="V")
    print(
        f"multigrid ready levels={len(multigrid.levels)} "
        f"operator_complexity={multigrid.operator_complexity():.3f} "
        f"seconds={time.perf_counter() - multigrid_started:.1f}",
        flush=True,
    )

    best_values: np.ndarray | None = None
    best_vectors: np.ndarray | None = None
    best_residuals: np.ndarray | None = None
    cycle_history: list[dict[str, Any]] = []
    for cycle in range(1, args.refinement_cycles + 1):
        cycle_started = time.perf_counter()
        print(
            f"LOBPCG cycle={cycle}/{args.refinement_cycles} maxiter={args.maxiter_per_cycle}",
            flush=True,
        )
        eigenvalues, eigenvectors, residual_history = spla.lobpcg(
            matrix,
            initial,
            M=preconditioner,
            largest=False,
            tol=args.absolute_tolerance,
            maxiter=args.maxiter_per_cycle,
            retResidualNormsHistory=True,
            restartControl=20,
            verbosityLevel=1,
        )
        order = np.argsort(eigenvalues, kind="stable")
        eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float32)
        residuals = relative_residuals(matrix, eigenvalues, eigenvectors)
        elapsed = time.perf_counter() - cycle_started
        record = {
            "cycle": cycle,
            "iterations": len(residual_history) - 1,
            "seconds": elapsed,
            "relative_residual_min": float(np.min(residuals)),
            "relative_residual_mean": float(np.mean(residuals)),
            "relative_residual_max": float(np.max(residuals)),
        }
        cycle_history.append(record)
        print(json.dumps(record), flush=True)
        best_values, best_vectors, best_residuals = eigenvalues, eigenvectors, residuals
        np.save(partial_path, eigenvectors)
        atomic_json(
            metadata_path.with_name(metadata_path.stem + "_partial.json"),
            {
                "complete": False,
                "resolution": args.resolution,
                "geometry_token": args.geometry_token,
                "fluid_node_count": len(grid["points"]),
                "cycle_history": cycle_history,
            },
        )
        if float(np.max(residuals)) <= args.relative_residual_target:
            break
        initial = eigenvectors

    if best_values is None or best_vectors is None or best_residuals is None:
        raise RuntimeError("LOBPCG did not return an eigenspace")
    physical_values = best_values / spacing**2
    physical_vectors = best_vectors.astype(np.float64)
    physical_vectors /= np.sqrt(
        np.sum(physical_vectors**2, axis=0, keepdims=True) * spacing**3
    )
    physical_vectors = physical_vectors.astype(np.float32)
    np.save(eigenvector_path, physical_vectors)
    metadata = {
        "complete": bool(np.max(best_residuals) <= args.relative_residual_target),
        "method": "smoothed-aggregation-preconditioned block LOBPCG",
        "matrix_scaling": "h^2 times seven-point Dirichlet -Delta",
        "resolution": args.resolution,
        "grid_shape": [args.resolution] * 3,
        "spacing": spacing,
        "geometry_token": args.geometry_token,
        "sphere_parameters_xyzr": geometry.tolist(),
        "fluid_node_count": len(grid["points"]),
        "fluid_fraction": float(len(grid["points"]) / args.resolution**3),
        "scalar_modes": args.scalar_modes,
        "eigenvalues": physical_values.tolist(),
        "relative_residuals": best_residuals.tolist(),
        "relative_residual_max": float(np.max(best_residuals)),
        "relative_residual_target": args.relative_residual_target,
        "cycle_history": cycle_history,
        "wall_seconds": float(time.perf_counter() - started),
        "eigenvectors": str(eigenvector_path),
    }
    atomic_json(metadata_path, metadata)
    if not metadata["complete"]:
        raise RuntimeError(
            f"128^3 eigensolve residual target not met: {metadata['relative_residual_max']:.3e}"
        )
    partial_path.unlink(missing_ok=True)
    metadata_path.with_name(metadata_path.stem + "_partial.json").unlink(missing_ok=True)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-token", type=int, default=12_000_001)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--scalar-modes", type=int, default=22)
    parser.add_argument("--low-resolution", type=int, default=34)
    parser.add_argument("--maxiter-per-cycle", type=int, default=50)
    parser.add_argument("--refinement-cycles", type=int, default=3)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--relative-residual-target", type=float, default=3.0e-4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    solve(parse_args())


if __name__ == "__main__":
    main()
