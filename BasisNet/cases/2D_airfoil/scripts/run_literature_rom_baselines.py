"""Train and roll out POD--Galerkin, exact DMD, and quadratic OpInf ROMs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402
from eigenfluid_sss.basisnet.airfoil2d.literature_rom import (  # noqa: E402
    backward_error_rank,
    evaluate_quadratic_opinf,
    fit_exact_dmd,
    fit_quadratic_opinf,
)
from eigenfluid_sss.basisnet.airfoil2d.rom_dynamics import (  # noqa: E402
    OpenBoundaryTerms,
)
from rollout_eigenfluid_rom import (  # noqa: E402
    ImplicitTrapezoidalROM,
    metric_rows,
    vorticity,
    write_json,
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def trapezoidal_weights(x: np.ndarray, y: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    wx, wy = np.ones(len(x)), np.ones(len(y))
    wx[[0, -1]], wy[[0, -1]] = 0.5, 0.5
    return ((x[1] - x[0]) * (y[1] - y[0]) * np.multiply.outer(wy, wx))[fluid]


def weighted_pod(
    training: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    solid = training["solid"].astype(bool)
    fluid = ~solid
    u, v = training["u"][:, fluid].astype(np.float64), training["v"][:, fluid].astype(np.float64)
    mean_u, mean_v = u.mean(axis=0), v.mean(axis=0)
    weights = trapezoidal_weights(training["x"], training["y"], fluid)
    sqrt_weights = np.sqrt(np.concatenate((weights, weights)))
    snapshots = np.column_stack((u - mean_u, v - mean_v))
    weighted = snapshots * sqrt_weights[None, :]
    temporal_u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
    rank = backward_error_rank(singular, *weighted.shape)
    coefficients = temporal_u[:, :rank] * singular[:rank]
    modes = (weighted.T @ temporal_u[:, :rank]) / singular[:rank]
    modes /= sqrt_weights[:, None]
    phi_u, phi_v = modes[: len(weights)], modes[len(weights) :]
    captured = float(np.sum(singular[:rank] ** 2) / np.sum(singular**2))
    return phi_u, phi_v, coefficients, weights, {
        "snapshot_count": len(snapshots),
        "float64_backward_error_rank": rank,
        "captured_snapshot_energy": captured,
        "mass_orthogonality_defect": float(
            np.max(np.abs(modes.T @ (sqrt_weights[:, None] ** 2 * modes) - np.eye(rank)))
        ),
    }


def grid_vorticity_modes(
    phi_u: np.ndarray,
    phi_v: np.ndarray,
    fluid: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    result = np.empty_like(phi_u)
    for mode in range(phi_u.shape[1]):
        u = np.zeros(fluid.shape, dtype=np.float64)
        v = np.zeros_like(u)
        u[fluid], v[fluid] = phi_u[:, mode], phi_v[:, mode]
        result[:, mode] = vorticity(u, v, dx, dy)[fluid]
    return result


def project_initial(
    sample: dict[str, np.ndarray],
    phi_u: np.ndarray,
    phi_v: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fluid = ~sample["solid"].astype(bool)
    du = sample["pinn_u"][fluid].astype(np.float64) - mean_u
    dv = sample["pinn_v"][fluid].astype(np.float64) - mean_v
    return phi_u.T @ (weights * du) + phi_v.T @ (weights * dv)


def reconstruct(
    coefficients: np.ndarray,
    modes: np.ndarray,
    mean: np.ndarray,
    fluid: np.ndarray,
) -> np.ndarray:
    fields = coefficients @ modes.T + mean[None, :]
    result = np.zeros((len(coefficients),) + fluid.shape, dtype=np.float32)
    result[:, fluid] = fields.astype(np.float32)
    return result


def rk4(rhs, state: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def advance_dmd(operator: np.ndarray, initial: np.ndarray, frames: int) -> np.ndarray:
    result = np.empty((frames + 1, len(initial)), dtype=np.float64)
    result[0] = initial
    for frame in range(frames):
        result[frame + 1] = operator @ result[frame]
    return result


def advance_opinf(
    operator: np.ndarray,
    initial: np.ndarray,
    frames: int,
    frame_dt: float,
    substeps: int,
) -> np.ndarray:
    result = np.empty((frames + 1, len(initial)), dtype=np.float64)
    result[0] = initial
    dt = frame_dt / substeps
    for frame in range(frames):
        current = result[frame]
        for _ in range(substeps):
            current = rk4(lambda state: evaluate_quadratic_opinf(operator, state), current, dt)
            if not np.all(np.isfinite(current)):
                raise FloatingPointError("quadratic Operator Inference rollout became non-finite")
        result[frame + 1] = current
    return result


def select_regularized_opinf(
    train_coefficients: np.ndarray,
    sample_dt: float,
    substeps: int,
) -> tuple[np.ndarray, int, dict[str, float | int | list[dict[str, float]]]]:
    """McQuarrie--Huang--Willcox training-trajectory regularization search."""

    candidates = np.logspace(-12, 2, 29)
    records: list[dict[str, float]] = []
    bound = 1.05 * float(np.max(np.abs(train_coefficients)))
    best: tuple[float, np.ndarray, int, dict] | None = None
    for regularization in candidates:
        operator, order, diagnostics = fit_quadratic_opinf(
            train_coefficients,
            sample_dt,
            regularization_linear=float(regularization),
            regularization_quadratic=float(regularization),
        )
        try:
            rollout = advance_opinf(
                operator,
                train_coefficients[0, :order],
                len(train_coefficients) - 1,
                sample_dt,
                substeps,
            )
            finite = bool(np.all(np.isfinite(rollout)))
            maximum = float(np.max(np.abs(rollout))) if finite else float("inf")
            error = (
                float(
                    np.linalg.norm(rollout - train_coefficients[:, :order])
                    / max(np.linalg.norm(train_coefficients[:, :order]), 1.0e-30)
                )
                if finite and maximum <= bound
                else float("inf")
            )
        except (FloatingPointError, OverflowError):
            maximum, error = float("inf"), float("inf")
        records.append(
            {
                "regularization": float(regularization),
                "training_rollout_relative_error": error,
                "maximum_absolute_state": maximum,
            }
        )
        if np.isfinite(error) and (best is None or error < best[0]):
            best = error, operator, order, diagnostics
    if best is None:
        raise RuntimeError("regularized Operator Inference found no bounded training rollout")
    _, operator, order, diagnostics = best
    diagnostics["selection"] = (
        "regularization minimizes integrated training-trajectory error subject "
        "to the McQuarrie-Huang-Willcox state bound"
    )
    diagnostics["state_bound"] = bound
    diagnostics["regularization_candidates"] = records
    return operator, order, diagnostics


def advance_pod_galerkin(
    phi_u: np.ndarray,
    phi_v: np.ndarray,
    phi_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    weights: np.ndarray,
    training: dict[str, np.ndarray],
    initial: np.ndarray,
    frames: int,
    frame_dt: float,
    substeps: int,
    viscosity: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    fluid = ~training["solid"].astype(bool)
    rows = np.flatnonzero(fluid[:, -1])
    fluid_flat = np.flatnonzero(fluid.ravel())
    outlet_flat = rows * fluid.shape[1] + (fluid.shape[1] - 1)
    outlet_indices = np.searchsorted(fluid_flat, outlet_flat)
    dy = float(training["y"][1] - training["y"][0])
    outlet_weights = np.full(len(rows), dy)
    outlet_weights[[0, -1]] *= 0.5
    stiffness = phi_omega.T @ (weights[:, None] * phi_omega)
    eigenvalues = np.diag(stiffness)
    open_boundary = OpenBoundaryTerms(
        basis_u=cp.asarray(phi_u[outlet_indices], dtype=cp.float64),
        basis_v=cp.asarray(phi_v[outlet_indices], dtype=cp.float64),
        basis_normal=cp.asarray(phi_u[outlet_indices], dtype=cp.float64),
        weights=cp.asarray(outlet_weights, dtype=cp.float64),
        static_pressure=cp.float64(0.0),
        mean_u=cp.asarray(mean_u[outlet_indices], dtype=cp.float64),
        mean_v=cp.asarray(mean_v[outlet_indices], dtype=cp.float64),
    )
    integrator = ImplicitTrapezoidalROM(
        cp.asarray(phi_u, dtype=cp.float64),
        cp.asarray(phi_v, dtype=cp.float64),
        cp.asarray(phi_omega, dtype=cp.float64),
        cp.asarray(mean_u, dtype=cp.float64),
        cp.asarray(mean_v, dtype=cp.float64),
        cp.asarray(mean_omega, dtype=cp.float64),
        cp.asarray(eigenvalues, dtype=cp.float64),
        cp.asarray(weights, dtype=cp.float64),
        viscosity,
        newton_tolerance=2.0e-8,
        gmres_tolerance=2.0e-8,
        open_boundary=open_boundary,
    )
    result = np.empty((frames + 1, len(initial)), dtype=np.float64)
    current = cp.asarray(initial, dtype=cp.float64)
    result[0] = initial
    dt = frame_dt / substeps
    for frame in range(frames):
        for _ in range(substeps):
            current = integrator.step(current, dt)
        result[frame + 1] = cp.asnumpy(current)
        print(f"[POD-Galerkin {frame + 1:02d}/{frames}] norm={np.linalg.norm(result[frame + 1]):.6e}", flush=True)
    diagnostics = {
        "newton_iterations": integrator.newton_iterations,
        "gmres_iterations": integrator.linear_iterations,
        "adaptive_step_splits": integrator.step_splits,
        "viscosity": viscosity,
        "substeps_per_frame": substeps,
    }
    del integrator, open_boundary, current
    cp.get_default_memory_pool().free_all_blocks()
    return result, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-gt", type=Path, required=True)
    parser.add_argument("--initial-sample", type=Path, required=True)
    parser.add_argument("--reference-fields", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument("--pod-substeps-per-frame", type=int, default=10)
    parser.add_argument("--opinf-substeps-per-frame", type=int, default=50)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed_paths = {
        name: args.output_dir / name / "rollout_summary.json"
        for name in ("pod_galerkin", "exact_dmd", "quadratic_opinf")
    }
    if args.reuse_existing and all(path.exists() for path in completed_paths.values()):
        protocol = {
            "status": "complete",
            "elapsed_seconds": 0.0,
            "reused_complete_outputs": True,
            "common_training_data": "41 D2Q9 TRT-LBM GT snapshots at 500-step spacing",
            "common_evaluation": "independent 30-frame TRT-LBM trajectory initialized from the same epoch-5000 reconstructed velocity",
            "rank_policy": "float64 spectral backward-error bound only",
            "methods": {
                "pod_galerkin": "Noack et al. POD-Galerkin projection of incompressible Navier-Stokes with outlet total-pressure term",
                "exact_dmd": "Schmid exact projected DMD one-step map",
                "quadratic_opinf": "McQuarrie-Huang-Willcox regularized continuous quadratic Operator Inference",
            },
        }
        write_json(args.output_dir / "protocol.json", protocol)
        summaries = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in completed_paths.items()
        }
        print(
            json.dumps(
                {name: value["aggregate"] for name, value in summaries.items()},
                indent=2,
            ),
            flush=True,
        )
        return
    training, sample, reference = load(args.training_gt), load(args.initial_sample), load(args.reference_fields)
    metadata = json.loads(str(training["metadata"]))
    cfg = SimpleNamespace(**json.loads(args.airfoil_config.read_text(encoding="utf-8")))
    fluid = ~training["solid"].astype(bool)
    if not np.array_equal(fluid, ~sample["solid"].astype(bool)):
        raise RuntimeError("training and rollout fluid masks differ")
    dx, dy = float(training["x"][1] - training["x"][0]), float(training["y"][1] - training["y"][0])
    frame_dt = args.steps_per_frame * dx

    phi_u, phi_v, train_coefficients, weights, pod_diagnostics = weighted_pod(training)
    rank = phi_u.shape[1]
    print(f"[training] POD float64 rank={rank}/{len(training['u'])}", flush=True)
    mean_u = training["u"][:, fluid].mean(axis=0, dtype=np.float64)
    mean_v = training["v"][:, fluid].mean(axis=0, dtype=np.float64)
    mean_u_grid, mean_v_grid = np.zeros(fluid.shape), np.zeros(fluid.shape)
    mean_u_grid[fluid], mean_v_grid[fluid] = mean_u, mean_v
    mean_omega_grid = vorticity(mean_u_grid, mean_v_grid, dx, dy)
    mean_omega = mean_omega_grid[fluid]
    phi_omega = grid_vorticity_modes(phi_u, phi_v, fluid, dx, dy)
    initial = project_initial(sample, phi_u, phi_v, mean_u, mean_v, weights)

    dmd_operator, dmd_diagnostics = fit_exact_dmd(train_coefficients)
    dmd_coefficients = advance_dmd(dmd_operator, initial, args.frames)[1:]
    opinf_operator, opinf_order, opinf_diagnostics = select_regularized_opinf(
        train_coefficients, frame_dt, args.opinf_substeps_per_frame
    )
    opinf_coefficients = advance_opinf(opinf_operator, initial[:opinf_order], args.frames, frame_dt, args.opinf_substeps_per_frame)[1:]
    lattice_viscosity = float(metadata["lattice_viscosity"])
    # POD--Galerkin's viscous operator is dense in the energy-optimal POD
    # coordinates.  Orthogonally diagonalize it inside the identical POD
    # subspace so that the exact exponential viscous split remains valid.
    pod_stiffness = phi_omega.T @ (weights[:, None] * phi_omega)
    pod_eigenvalues, pod_rotation = np.linalg.eigh(
        0.5 * (pod_stiffness + pod_stiffness.T)
    )
    if pod_eigenvalues[0] < -1.0e-10 * max(pod_eigenvalues[-1], 1.0):
        raise RuntimeError("POD stiffness operator is not positive semidefinite")
    pod_eigenvalues = np.maximum(pod_eigenvalues, 0.0)
    pod_phi_u = phi_u @ pod_rotation
    pod_phi_v = phi_v @ pod_rotation
    pod_phi_omega = phi_omega @ pod_rotation
    pod_initial = pod_rotation.T @ initial
    pod_coefficients, pod_integrator = advance_pod_galerkin(
        pod_phi_u, pod_phi_v, pod_phi_omega, mean_u, mean_v, mean_omega, weights, training,
        pod_initial, args.frames, frame_dt, args.pod_substeps_per_frame, lattice_viscosity * dx,
    )
    pod_coefficients = pod_coefficients[1:]
    pod_integrator.update(
        {
            "viscous_operator_diagonalized_within_pod_subspace": True,
            "viscous_eigenvalue_min": float(pod_eigenvalues[0]),
            "viscous_eigenvalue_max": float(pod_eigenvalues[-1]),
        }
    )

    xgrid, ygrid = np.meshgrid(training["x"], training["y"])
    flat_points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    xb, yb = body_numpy(cfg, flat_points, float(sample["alpha_deg"]))
    wake = (fluid.ravel() & (xb >= cfg.chord) & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords)) & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)).reshape(fluid.shape)
    ns_u, ns_v, ns_omega = reference["ns_u"][: args.frames], reference["ns_v"][: args.frames], reference["ns_omega"][: args.frames]

    methods = {
        "pod_galerkin": (pod_coefficients, pod_phi_u, pod_phi_v, pod_phi_omega, {**pod_diagnostics, **pod_integrator}),
        "exact_dmd": (dmd_coefficients, phi_u, phi_v, phi_omega, dmd_diagnostics),
        "quadratic_opinf": (opinf_coefficients, phi_u[:, :opinf_order], phi_v[:, :opinf_order], phi_omega[:, :opinf_order], opinf_diagnostics),
    }
    summaries = {}
    for name, (coefficients, u_modes, v_modes, omega_modes, diagnostics) in methods.items():
        pred_u = reconstruct(coefficients, u_modes, mean_u, fluid)
        pred_v = reconstruct(coefficients, v_modes, mean_v, fluid)
        pred_omega = reconstruct(coefficients, omega_modes, mean_omega, fluid)
        metrics = metric_rows(ns_u, ns_v, ns_omega, pred_u, pred_v, pred_omega, mean_u_grid, mean_v_grid, mean_omega_grid, fluid, wake, args.steps_per_frame)
        method_dir = args.output_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(method_dir / "rollout_fields.npz", x=training["x"], y=training["y"], solid=training["solid"], ns_u=ns_u, ns_v=ns_v, ns_omega=ns_omega, predicted_u=pred_u, predicted_v=pred_v, predicted_omega=pred_omega, coefficients=coefficients)
        summary = {
            "status": "complete",
            "method": name,
            "training_snapshot_count": len(training["u"]),
            "training_gt": str(args.training_gt.resolve()),
            "evaluation_reference": str(args.reference_fields.resolve()),
            "same_initial_velocity": True,
            "pod": pod_diagnostics,
            "diagnostics": diagnostics,
            "metrics": metrics,
            "aggregate": {key: {"mean": float(np.mean([row[key] for row in metrics])), "p95": float(np.percentile([row[key] for row in metrics], 95)), "max": float(np.max([row[key] for row in metrics])), "final": float(metrics[-1][key])} for key in metrics[0] if key.endswith("_error")},
        }
        write_json(method_dir / "rollout_summary.json", summary)
        summaries[name] = summary
        print(f"[{name}] mean velocity={summary['aggregate']['velocity_error']['mean']:.6e}", flush=True)

    protocol = {
        "status": "complete",
        "elapsed_seconds": time.perf_counter() - started,
        "common_training_data": "41 D2Q9 TRT-LBM GT snapshots at 500-step spacing",
        "common_evaluation": "independent 30-frame TRT-LBM trajectory initialized from the same epoch-5000 reconstructed velocity",
        "rank_policy": "float64 spectral backward-error bound only",
        "methods": {
            "pod_galerkin": "Noack et al. POD-Galerkin projection of incompressible Navier-Stokes with outlet total-pressure term",
            "exact_dmd": "Schmid exact projected DMD one-step map",
            "quadratic_opinf": "McQuarrie-Huang-Willcox regularized continuous quadratic Operator Inference",
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps({name: value["aggregate"] for name, value in summaries.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
