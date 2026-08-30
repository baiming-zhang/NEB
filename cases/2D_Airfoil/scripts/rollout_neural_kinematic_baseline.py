"""Airfoil rollout with the SIGGRAPH Asia 2025 camera-ready Eq. (8)."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc

from rollout_eigenfluid_rom import metric_rows, write_json
from train_neural_kinematic_baseline import NeuralKinematicBases

import sys

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
from eigenfluid.gen.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402


def normalize(points: np.ndarray) -> np.ndarray:
    return np.column_stack((points[:, 0] / 5.0 - 1.0, points[:, 1] / 4.0)).astype(np.float32)


def evaluate_basis(model: NeuralKinematicBases, points: np.ndarray, batch: int) -> np.ndarray:
    parts = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(points), batch):
            inputs = torch.as_tensor(normalize(points[start : start + batch]), device=device)
            parts.append(model(inputs).cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def evaluate_basis_with_vorticity(
    model: NeuralKinematicBases, points: np.ndarray, batch: int
) -> tuple[np.ndarray, np.ndarray]:
    fields, curls = [], []
    device = next(model.parameters()).device
    for start in range(0, len(points), batch):
        inputs = torch.as_tensor(normalize(points[start : start + batch]), device=device)
        inputs.requires_grad_(True)
        values = model(inputs)
        curl_modes = []
        for mode in range(model.bases):
            grad_v = torch.autograd.grad(values[:, mode, 1].sum(), inputs, retain_graph=True)[0][:, 0] / 5.0
            grad_u = torch.autograd.grad(values[:, mode, 0].sum(), inputs, retain_graph=True)[0][:, 1] / 4.0
            curl_modes.append(grad_v - grad_u)
        fields.append(values.detach().cpu().numpy().astype(np.float32))
        curls.append(torch.stack(curl_modes, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(fields), np.concatenate(curls)


def project_to_domain(points: np.ndarray, beta: float, alpha_deg: float) -> np.ndarray:
    result = points.copy()
    result[:, 0] = np.clip(result[:, 0], 0.0, 10.0)
    result[:, 1] = np.clip(result[:, 1], -4.0, 4.0)
    angle = math.radians(-alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    dx, dy = result[:, 0] - 3.0, result[:, 1]
    xb = c * dx + s * dy + 0.5
    yb = -s * dx + c * dy
    thickness = 0.08 + (0.18 - 0.08) * beta
    qx, qy = xb - 1.0, yb
    radius = np.sqrt(qx * qx + (qy / thickness) ** 2)
    inside = radius < 1.0
    scale = 1.0 / np.maximum(radius[inside], 1.0e-12)
    xb[inside] = 1.0 + qx[inside] * scale
    yb[inside] = qy[inside] * scale
    # body -> world, the inverse of body_numpy
    angle = math.radians(-alpha_deg)
    c, s = math.cos(angle), math.sin(angle)
    bx, by = xb - 0.5, yb
    result[:, 0] = c * bx - s * by + 3.0
    result[:, 1] = s * bx + c * by
    return result


def integration_points(sample: dict[str, np.ndarray], count: int, seed: int) -> np.ndarray:
    x, y = sample["x"], sample["y"]
    sampler = qmc.Sobol(2, scramble=True, seed=seed)
    pieces, remaining = [], count
    solid = sample["solid"].astype(bool)
    while remaining:
        draw = sampler.random(max(remaining * 2, 4096))
        candidates = np.column_stack((10.0 * draw[:, 0], -4.0 + 8.0 * draw[:, 1]))
        ix = np.clip(np.rint((candidates[:, 0] - x[0]) / (x[1] - x[0])).astype(int), 0, len(x) - 1)
        iy = np.clip(np.rint((candidates[:, 1] - y[0]) / (y[1] - y[0])).astype(int), 0, len(y) - 1)
        accepted = candidates[~solid[iy, ix]][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces).astype(np.float32)


def relative_initial_error(target_u: np.ndarray, target_v: np.ndarray, pred_u: np.ndarray, pred_v: np.ndarray) -> float:
    return float(np.sqrt(np.sum((pred_u - target_u) ** 2 + (pred_v - target_v) ** 2, dtype=np.float64) / max(np.sum(target_u**2 + target_v**2, dtype=np.float64), 1.0e-30)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--reference-fields", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument("--substeps-per-frame", type=int, default=10)
    parser.add_argument("--integration-points", type=int, default=250000)
    parser.add_argument("--field-batch", type=int, default=32768)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.sample, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}
    with np.load(args.reference_fields, allow_pickle=False) as archive:
        reference = {name: archive[name] for name in archive.files}
    cfg = type("Config", (), json.loads(args.airfoil_config.read_text(encoding="utf-8")))()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = NeuralKinematicBases().cuda().eval()
    model.load_state_dict(state["model"])
    beta, alpha = float(sample["beta"]), float(sample["alpha_deg"])
    solid, fluid = sample["solid"].astype(bool), ~sample["solid"].astype(bool)
    ygrid, xgrid = np.meshgrid(sample["y"], sample["x"], indexing="ij")
    all_points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)

    started = time.perf_counter()
    print(f"[stage 1/3] Eq. (8) integration basis at {args.integration_points} points", flush=True)
    points = integration_points(sample, args.integration_points, 20250813)
    basis = evaluate_basis(model, points, args.field_batch)
    matrix = basis.transpose(0, 2, 1).reshape(-1, model.bases).astype(np.float64)
    mass = matrix.T @ matrix
    mass = 0.5 * (mass + mass.T)
    mass_values = np.linalg.eigvalsh(mass)
    mass_floor = np.finfo(np.float64).eps * matrix.shape[0] * mass_values[-1]
    rank = int(np.sum(mass_values > mass_floor))
    if rank != model.bases:
        raise RuntimeError(f"paper-basis mass rank {rank}, expected {model.bases}")
    target_u = np.interp(points[:, 0], sample["x"], np.zeros_like(sample["x"]))  # allocate only
    ix = np.clip(np.rint((points[:, 0] - sample["x"][0]) / (sample["x"][1] - sample["x"][0])).astype(int), 0, len(sample["x"]) - 1)
    iy = np.clip(np.rint((points[:, 1] - sample["y"][0]) / (sample["y"][1] - sample["y"][0])).astype(int), 0, len(sample["y"]) - 1)
    target_u = sample["pinn_u"][iy, ix].astype(np.float64)
    target_v = sample["pinn_v"][iy, ix].astype(np.float64)
    rhs = matrix.T @ np.column_stack((target_u, target_v)).reshape(-1)
    coefficient = np.linalg.solve(mass, rhs)
    fitted = np.einsum("nki,k->ni", basis, coefficient, optimize=True).astype(np.float32)
    initial_projection_error = relative_initial_error(target_u, target_v, fitted[:, 0], fitted[:, 1])

    coefficients = np.empty((args.frames + 1, model.bases), dtype=np.float64)
    coefficients[0] = coefficient
    coordinate_dt = args.steps_per_frame * float(sample["x"][1] - sample["x"][0]) / args.substeps_per_frame
    print(f"[stage 2/3] camera-ready semi-Lagrangian Eq. (8), dt={coordinate_dt:.6g}", flush=True)
    rollout_started = time.perf_counter()
    for frame in range(args.frames):
        for _ in range(args.substeps_per_frame):
            velocity = np.einsum("nki,k->ni", basis, coefficient, optimize=True)
            origins = project_to_domain(points - velocity * coordinate_dt, beta, alpha)
            origin_basis = evaluate_basis(model, origins, args.field_batch)
            advected = np.einsum("nki,k->ni", origin_basis, coefficient, optimize=True)
            rhs = matrix.T @ advected.reshape(-1).astype(np.float64)
            coefficient = np.linalg.solve(mass, rhs)
            if not np.all(np.isfinite(coefficient)):
                raise FloatingPointError("non-finite Eq. (8) coefficient")
        coefficients[frame + 1] = coefficient
        print(f"[paper Eq.8 {frame + 1:02d}/{args.frames}] coefficient_norm={np.linalg.norm(coefficient):.6e}", flush=True)
    rollout_elapsed = time.perf_counter() - rollout_started

    print("[stage 3/3] full-grid basis, vorticity, reconstruction, metrics", flush=True)
    full_basis, full_curl = evaluate_basis_with_vorticity(model, all_points, args.field_batch)
    paper_u = np.zeros((args.frames,) + solid.shape, dtype=np.float32)
    paper_v = np.zeros_like(paper_u)
    paper_omega = np.zeros_like(paper_u)
    fields = np.einsum("nki,tk->tni", full_basis, coefficients[1:], optimize=True).astype(np.float32)
    curls = np.einsum("nk,tk->tn", full_curl, coefficients[1:], optimize=True).astype(np.float32)
    paper_u[:, fluid], paper_v[:, fluid], paper_omega[:, fluid] = fields[:, :, 0], fields[:, :, 1], curls
    xb, yb = body_numpy(cfg, all_points.astype(np.float64), alpha)
    wake_flat = (xb >= cfg.chord) & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords)) & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)
    wake = np.zeros_like(fluid)
    wake[fluid] = wake_flat
    zeros = np.zeros_like(solid, dtype=np.float32)
    metrics = metric_rows(
        reference["ns_u"][: args.frames],
        reference["ns_v"][: args.frames],
        reference["ns_omega"][: args.frames],
        paper_u,
        paper_v,
        paper_omega,
        zeros,
        zeros,
        zeros,
        fluid,
        wake,
        args.steps_per_frame,
    )
    fields_path = args.output_dir / "ns_vs_paper_camera_ready_rollout_fields.npz"
    np.savez_compressed(fields_path, x=sample["x"], y=sample["y"], solid=solid, ns_u=reference["ns_u"][: args.frames], ns_v=reference["ns_v"][: args.frames], ns_omega=reference["ns_omega"][: args.frames], paper_u=paper_u, paper_v=paper_v, paper_omega=paper_omega, coefficient_frames=coefficients)
    summary = {
        "status": "complete",
        "method": "Neural Kinematic Bases for Fluids SIGGRAPH Asia 2025 camera-ready Eq. (8) reproduction",
        "official_checkpoint": False,
        "reason": "author repository publishes no source code or model weights",
        "fixed_airfoil_adaptation": True,
        "bases": model.bases,
        "integration_points": args.integration_points,
        "rank": rank,
        "mass_condition_number": float(mass_values[-1] / mass_values[0]),
        "initial_full_velocity_projection_relative_l2": initial_projection_error,
        "frames": args.frames,
        "substeps_per_frame": args.substeps_per_frame,
        "coordinate_dt": coordinate_dt,
        "equation": "camera-ready Eq. (8): backward trace, closest-domain projection, copy origin velocity, least-squares projection",
        "viscosity_or_inlet_forcing": False,
        "rollout_elapsed_seconds": rollout_elapsed,
        "total_elapsed_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "aggregate": {key: {"mean": float(np.mean([float(row[key]) for row in metrics])), "p95": float(np.percentile([float(row[key]) for row in metrics], 95.0)), "max": float(np.max([float(row[key]) for row in metrics])), "final": float(metrics[-1][key])} for key in metrics[0] if key.endswith("_error")},
        "fields": str(fields_path.resolve()),
    }
    write_json(args.output_dir / "rollout_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
