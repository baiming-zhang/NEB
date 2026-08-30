"""Strict full-grid K1024 Laplacian-eigenfluid rollout for the airfoil case.

The rectangular streamfunction eigenmodes are re-orthogonalized in the actual
fluid-domain mass metric.  Dynamics use the same full-grid variational
Galerkin operator and implicit trapezoidal integrator as the Eigenfluid
rollout, so the comparison isolates the spatial basis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

from evaluate_laplacian_baseline import laplacian_mode_table, mode_fields
from rollout_eigenfluid_rom import (
    ImplicitTrapezoidalROM,
    full_grid_quadrature,
    metric_rows,
    write_json,
)

import sys

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402
from eigenfluid.gen.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    generalized_symmetric_eig,
)
from eigenfluid.gen.airfoil2d.rom_dynamics import OpenBoundaryTerms  # noqa: E402


def accumulate_operators(
    points: np.ndarray,
    weights: np.ndarray,
    bounds: tuple[float, float, float, float],
    modes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(modes[0])
    mass = np.zeros((count, count), dtype=np.float64)
    stiffness = np.zeros_like(mass)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        u, v, omega = mode_fields(points[start:stop], bounds, modes)
        w = cp.asarray(weights[start:stop])[:, None]
        gu, gv, go = cp.asarray(u, dtype=cp.float64), cp.asarray(v, dtype=cp.float64), cp.asarray(omega, dtype=cp.float64)
        mass += cp.asnumpy(gu.T @ (w * gu) + gv.T @ (w * gv)).astype(np.float64)
        stiffness += cp.asnumpy(go.T @ (w * go)).astype(np.float64)
        del w, gu, gv, go
        print(f"[Laplacian operators] {stop}/{len(points)} fluid nodes", flush=True)
    cp.get_default_memory_pool().free_all_blocks()
    return mass, stiffness


def evaluate_basis(
    points: np.ndarray,
    weights: np.ndarray,
    bounds: tuple[float, float, float, float],
    modes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rotation: np.ndarray,
    chunk: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, float]:
    count = rotation.shape[1]
    shape = (len(points), count)
    basis_bytes = 3 * int(np.prod(shape)) * np.dtype(np.float32).itemsize
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    if basis_bytes + 2 * 1024**3 > free_bytes:
        raise RuntimeError(
            f"Laplacian basis requires {basis_bytes / 1024**3:.2f} GiB plus "
            f"2 GiB reserve, but only {free_bytes / 1024**3:.2f} GiB is free"
        )
    hu = np.empty(shape, dtype=np.float32)
    hv = np.empty(shape, dtype=np.float32)
    ho = np.empty(shape, dtype=np.float32)
    check = np.zeros((count, count), dtype=np.float64)
    gr = cp.asarray(rotation.astype(np.float32))
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        ru, rv, ro = mode_fields(points[start:stop], bounds, modes)
        gu = cp.asarray(ru, dtype=cp.float32) @ gr
        gv = cp.asarray(rv, dtype=cp.float32) @ gr
        go = cp.asarray(ro, dtype=cp.float32) @ gr
        hu[start:stop], hv[start:stop], ho[start:stop] = cp.asnumpy(gu), cp.asnumpy(gv), cp.asnumpy(go)
        w = cp.asarray(weights[start:stop])[:, None]
        gu64 = gu.astype(cp.float64)
        gv64 = gv.astype(cp.float64)
        w64 = w.astype(cp.float64)
        check += cp.asnumpy(
            gu64.T @ (w64 * gu64) + gv64.T @ (w64 * gv64)
        ).astype(np.float64)
        del gu, gv, go, w, gu64, gv64, w64
        print(f"[Laplacian Ritz basis] {stop}/{len(points)} fluid nodes", flush=True)
    del gr
    cp.get_default_memory_pool().free_all_blocks()
    orthogonality = float(np.max(np.abs(check - np.eye(count))))
    return cp.asarray(hu), cp.asarray(hv), cp.asarray(ho), orthogonality


def reconstruct(
    basis: cp.ndarray,
    coefficients: np.ndarray,
    mean: np.ndarray,
    fluid: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    values = cp.asnumpy(basis @ cp.asarray(coefficients[1:].T.astype(np.float32)))
    values += mean[fluid, None]
    result = np.zeros((len(coefficients) - 1,) + shape, dtype=np.float32)
    result[:, fluid] = values.T
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--gt-source", type=Path, required=True)
    parser.add_argument("--reference-fields", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument("--substeps-per-frame", type=int, default=10)
    parser.add_argument("--grid-chunk", type=int, default=8192)
    parser.add_argument("--orthogonality-tolerance", type=float, default=5.0e-4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.sample, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}
    with np.load(args.gt_source, allow_pickle=False) as archive:
        gt_source = {name: archive[name] for name in archive.files}
    with np.load(args.reference_fields, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(gt_source["metadata"]))
    cfg = type("Config", (), json.loads(args.airfoil_config.read_text(encoding="utf-8")))()
    if len(reference["ns_u"]) != args.frames:
        raise ValueError("reference trajectory frame count differs")
    if not np.array_equal(reference["solid"], sample["solid"]):
        raise ValueError("reference and sample masks differ")

    grid = full_grid_quadrature(gt_source)
    points, weights = grid["points"], grid["weights"]
    bounds = (float(sample["x"][0]), float(sample["x"][-1]), float(sample["y"][0]), float(sample["y"][-1]))
    modes = laplacian_mode_table(args.modes, *bounds)
    print(f"[stage 1/3] full-grid Laplacian operators: {len(points)} nodes, {args.modes} modes", flush=True)
    raw_mass, raw_stiffness = accumulate_operators(points, weights, bounds, modes, args.grid_chunk)
    eigenvalues, rotation, rank = generalized_symmetric_eig(
        raw_stiffness,
        raw_mass,
        args.modes,
        sample_rows=(2 * len(points)),
    )
    if rank != args.modes:
        raise RuntimeError(f"Laplacian numerical rank {rank}, expected {args.modes}")
    basis_u, basis_v, basis_omega, orth_error = evaluate_basis(points, weights, bounds, modes, rotation, args.grid_chunk)
    if orth_error > args.orthogonality_tolerance:
        raise RuntimeError(f"Laplacian mass orthogonality {orth_error:.3e} exceeds tolerance")

    solid = sample["solid"].astype(bool)
    fluid = ~solid
    mean_u = gt_source["u"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_v = gt_source["v"].mean(axis=0, dtype=np.float64).astype(np.float32)
    mean_omega = gt_source["vorticity"].mean(axis=0, dtype=np.float64).astype(np.float32)
    target_u = cp.asarray(sample["pinn_u"][fluid] - mean_u[fluid])
    target_v = cp.asarray(sample["pinn_v"][fluid] - mean_v[fluid])
    w = cp.asarray(weights)
    initial = basis_u.T @ (w * target_u) + basis_v.T @ (w * target_v)
    projection_u = cp.asnumpy(basis_u @ initial)
    projection_v = cp.asnumpy(basis_v @ initial)
    numerator = np.sum((projection_u - cp.asnumpy(target_u)) ** 2 + (projection_v - cp.asnumpy(target_v)) ** 2, dtype=np.float64)
    denominator = np.sum(cp.asnumpy(target_u) ** 2 + cp.asnumpy(target_v) ** 2, dtype=np.float64)
    initial_projection_error = float(np.sqrt(numerator / max(denominator, 1.0e-30)))
    del target_u, target_v, projection_u, projection_v

    outlet_indices = cp.asarray(grid["outlet_indices"])
    rows = grid["outlet_rows"]
    open_boundary = OpenBoundaryTerms(
        basis_u=basis_u[outlet_indices],
        basis_v=basis_v[outlet_indices],
        basis_normal=basis_u[outlet_indices],
        weights=cp.asarray(grid["outlet_weights"]),
        static_pressure=cp.float32(0.0),
        mean_u=cp.asarray(mean_u[rows, -1]),
        mean_v=cp.asarray(mean_v[rows, -1]),
    )
    dx = float(sample["x"][1] - sample["x"][0])
    coordinate_viscosity = float(metadata["lattice_viscosity"]) * dx
    output_dt = args.steps_per_frame * dx
    internal_dt = output_dt / args.substeps_per_frame
    integrator = ImplicitTrapezoidalROM(
        basis_u,
        basis_v,
        basis_omega,
        cp.asarray(mean_u[fluid]),
        cp.asarray(mean_v[fluid]),
        cp.asarray(mean_omega[fluid]),
        cp.asarray(eigenvalues.astype(np.float32)),
        w,
        coordinate_viscosity,
        newton_tolerance=2.0e-6,
        gmres_tolerance=2.0e-5,
        open_boundary=open_boundary,
    )
    coefficients = np.empty((args.frames + 1, args.modes), dtype=np.float32)
    coefficients[0] = cp.asnumpy(initial)
    state = initial
    started = time.perf_counter()
    print("[stage 2/3] implicit variational rollout", flush=True)
    for frame in range(args.frames):
        for _ in range(args.substeps_per_frame):
            state = integrator.step(state, internal_dt)
        coefficients[frame + 1] = cp.asnumpy(state)
        print(f"[Laplacian ROM {frame + 1:02d}/{args.frames}] coefficient_norm={float(cp.linalg.norm(state).get()):.6e}", flush=True)
    elapsed = time.perf_counter() - started

    print("[stage 3/3] full-grid reconstruction and metrics", flush=True)
    lap_u = reconstruct(basis_u, coefficients, mean_u, fluid, solid.shape)
    lap_v = reconstruct(basis_v, coefficients, mean_v, fluid, solid.shape)
    lap_omega = reconstruct(basis_omega, coefficients, mean_omega, fluid, solid.shape)
    xgrid, ygrid = np.meshgrid(sample["x"], sample["y"])
    xb, yb = body_numpy(cfg, np.column_stack((xgrid.ravel(), ygrid.ravel())), float(sample["alpha_deg"]))
    wake = (fluid.ravel() & (xb >= cfg.chord) & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords)) & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)).reshape(solid.shape)
    metrics = metric_rows(reference["ns_u"], reference["ns_v"], reference["ns_omega"], lap_u, lap_v, lap_omega, mean_u, mean_v, mean_omega, fluid, wake, args.steps_per_frame)
    fields = args.output_dir / "ns_vs_laplacian_rollout_fields.npz"
    np.savez_compressed(fields, x=sample["x"], y=sample["y"], solid=solid, ns_u=reference["ns_u"], ns_v=reference["ns_v"], ns_omega=reference["ns_omega"], laplacian_u=lap_u, laplacian_v=lap_v, laplacian_omega=lap_omega, coefficient_frames=coefficients, mode_m=modes[0].astype(np.int32), mode_n=modes[1].astype(np.int32), raw_eigenvalues=modes[2], ritz_eigenvalues=eigenvalues)
    summary: dict[str, Any] = {
        "status": "complete",
        "method": "rectangular Dirichlet streamfunction Laplacian modes, full-fluid-domain reorthogonalization",
        "modes": args.modes,
        "rank": int(rank),
        "mass_orthogonality_error": orth_error,
        "initial_fluctuation_projection_relative_l2": initial_projection_error,
        "frames": args.frames,
        "steps_per_frame": args.steps_per_frame,
        "substeps_per_frame": args.substeps_per_frame,
        "coordinate_time_per_frame": output_dt,
        "coordinate_viscosity": coordinate_viscosity,
        "rom_integrator": "same implicit-trapezoidal Newton-GMRES and exact modal viscous split as Eigenfluid",
        "newton_iterations": integrator.newton_iterations,
        "gmres_iterations": integrator.linear_iterations,
        "adaptive_step_splits": integrator.step_splits,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "aggregate": {key: {"mean": float(np.mean([float(row[key]) for row in metrics])), "p95": float(np.percentile([float(row[key]) for row in metrics], 95.0)), "max": float(np.max([float(row[key]) for row in metrics])), "final": float(metrics[-1][key])} for key in metrics[0] if key.endswith("_error")},
        "fields": str(fields.resolve()),
    }
    write_json(args.output_dir / "rollout_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
