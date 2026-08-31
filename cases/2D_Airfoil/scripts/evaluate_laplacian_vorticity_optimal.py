"""Evaluate full-grid vorticity-optimal coefficients for Laplacian-256 modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eigenfluid.gen.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402
from evaluate_laplacian_baseline import (  # noqa: E402
    METRIC_NAMES,
    aggregate,
    laplacian_mode_table,
    mode_fields,
    scalar_relative_error,
    velocity_relative_error,
    write_json,
)


def evaluate_case(
    gt_path: Path,
    source_sample: Path,
    output_sample: Path,
    cfg: SimpleNamespace,
    mode_count: int,
    field_batch: int,
) -> dict:
    with np.load(source_sample, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
    x = source["x"].astype(np.float64)
    y = source["y"].astype(np.float64)
    solid = source["solid"].astype(bool)
    fluid = ~solid
    frame = int(source["frame"])
    beta = float(source["beta"])
    alpha = float(source["alpha_deg"])
    xgrid, ygrid = np.meshgrid(x, y)
    all_points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    points = all_points[fluid.ravel()]

    with np.load(gt_path, allow_pickle=False) as gt:
        u_all = gt["u"]
        mean_u = np.mean(u_all, axis=0, dtype=np.float64)[fluid]
        actual_u = u_all[frame][fluid].astype(np.float64)
        del u_all
        v_all = gt["v"]
        mean_v = np.mean(v_all, axis=0, dtype=np.float64)[fluid]
        actual_v = v_all[frame][fluid].astype(np.float64)
        del v_all
        omega_all = gt["vorticity"]
        mean_omega = np.mean(omega_all, axis=0, dtype=np.float64)[fluid]
        actual_omega = omega_all[frame][fluid].astype(np.float64)
        del omega_all

    bounds = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    modes = laplacian_mode_table(mode_count, *bounds)
    target_omega = actual_omega - mean_omega

    # On the full tensor grid, distinct sine modes are discretely orthogonal.
    # Remove the solid contribution to obtain the exact fluid-grid Gram matrix.
    amplitude = modes[3] * modes[2]
    full_diagonal = (
        0.25 * (len(x) - 1) * (len(y) - 1) * amplitude**2
    )
    solid_points = all_points[solid.ravel()]
    _, _, solid_omega = mode_fields(solid_points, bounds, modes)
    gram = np.diag(full_diagonal) - solid_omega.T @ solid_omega
    del solid_omega, solid_points, all_points

    rhs = np.zeros(mode_count, dtype=np.float64)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        _, _, basis_omega = mode_fields(points[start:stop], bounds, modes)
        rhs += basis_omega.T @ target_omega[start:stop]
    values, vectors = np.linalg.eigh(gram)
    threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
    retained = values > threshold
    coefficients = vectors[:, retained] @ (
        (vectors[:, retained].T @ rhs) / values[retained]
    )
    del gram, rhs, vectors

    reconstructed_u = np.empty(len(points), dtype=np.float32)
    reconstructed_v = np.empty(len(points), dtype=np.float32)
    reconstructed_omega = np.empty(len(points), dtype=np.float32)
    residual_numerator = 0.0
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        basis_u, basis_v, basis_omega = mode_fields(
            points[start:stop], bounds, modes
        )
        omega_fluctuation = basis_omega @ coefficients
        reconstructed_u[start:stop] = (
            mean_u[start:stop] + basis_u @ coefficients
        ).astype(np.float32)
        reconstructed_v[start:stop] = (
            mean_v[start:stop] + basis_v @ coefficients
        ).astype(np.float32)
        reconstructed_omega[start:stop] = (
            mean_omega[start:stop] + omega_fluctuation
        ).astype(np.float32)
        residual_numerator += float(
            np.sum(
                (omega_fluctuation - target_omega[start:stop]) ** 2,
                dtype=np.float64,
            )
        )
    projection_residual = float(
        np.sqrt(
            residual_numerator
            / max(float(np.sum(target_omega**2, dtype=np.float64)), 1.0e-30)
        )
    )

    fluctuation_u = actual_u - mean_u
    fluctuation_v = actual_v - mean_v
    reconstructed_fluctuation_u = reconstructed_u - mean_u
    reconstructed_fluctuation_v = reconstructed_v - mean_v
    reconstructed_fluctuation_omega = reconstructed_omega - mean_omega
    xb, yb = body_numpy(cfg, points, alpha)
    wake = (
        (xb >= cfg.chord)
        & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords))
        & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)
    )
    metrics = {
        "velocity_error": velocity_relative_error(
            actual_u, actual_v, reconstructed_u, reconstructed_v
        ),
        "vorticity_error": scalar_relative_error(
            actual_omega, reconstructed_omega
        ),
        "fluctuation_velocity_error": velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            reconstructed_fluctuation_u,
            reconstructed_fluctuation_v,
        ),
        "fluctuation_vorticity_error": scalar_relative_error(
            target_omega, reconstructed_fluctuation_omega
        ),
        "wake_velocity_error": velocity_relative_error(
            actual_u, actual_v, reconstructed_u, reconstructed_v, wake
        ),
        "wake_vorticity_error": scalar_relative_error(
            actual_omega, reconstructed_omega, wake
        ),
        "wake_fluctuation_velocity_error": velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            reconstructed_fluctuation_u,
            reconstructed_fluctuation_v,
            wake,
        ),
        "wake_fluctuation_vorticity_error": scalar_relative_error(
            target_omega, reconstructed_fluctuation_omega, wake
        ),
    }

    def grid(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(solid, dtype=np.float32)
        result[fluid] = values
        return result

    output_sample.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_sample,
        laplacian_vorticity_opt_u=grid(reconstructed_u),
        laplacian_vorticity_opt_v=grid(reconstructed_v),
        laplacian_vorticity_opt_omega=grid(reconstructed_omega),
        coefficients=coefficients,
        projection_rank=np.int32(np.count_nonzero(retained)),
        projection_points=np.int32(len(points)),
        projection_relative_residual=np.float64(projection_residual),
        **{name: np.float64(value) for name, value in metrics.items()},
    )
    return {
        "case": int(output_sample.stem.split("_")[-1]),
        "beta": beta,
        "alpha_deg": alpha,
        "frame": frame,
        "projection_rank": int(np.count_nonzero(retained)),
        "projection_points": int(len(points)),
        "projection_relative_residual": projection_residual,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=256)
    parser.add_argument("--field-batch", type=int, default=32768)
    args = parser.parse_args()

    source_dir = args.run_dir / "postprocess"
    output_dir = args.run_dir / "postprocess_laplacian_vorticity_optimal"
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        **json.loads(args.airfoil_config.read_text(encoding="utf-8"))
    )
    cases = json.loads(
        (source_dir / "gt_cache_reuse_manifest.json").read_text(encoding="utf-8")
    )["cases"]
    metrics = []
    write_json(
        output_dir / "pipeline_state.json",
        {"status": "running", "completed_cases": 0, "total_cases": len(cases)},
    )
    for position, case in enumerate(cases):
        index = int(case["index"])
        source_sample = source_dir / "samples" / f"sample_{index:03d}.npz"
        output_sample = sample_dir / f"sample_{index:03d}.npz"
        print(
            f"[vorticity-optimal {position + 1:03d}/{len(cases)}] "
            f"beta={float(case['beta']):.4f} "
            f"alpha={float(case['alpha_deg']):+.3f}",
            flush=True,
        )
        result = evaluate_case(
            Path(case["path"]),
            source_sample,
            output_sample,
            cfg,
            args.modes,
            args.field_batch,
        )
        with np.load(source_sample, allow_pickle=False) as source:
            result.update(
                {
                    f"pinn_{name}": float(source[f"vorticity_opt_{name}"])
                    for name in METRIC_NAMES
                }
            )
        metrics.append(result)
        write_json(
            output_dir / "pipeline_state.json",
            {
                "status": "running",
                "completed_cases": position + 1,
                "total_cases": len(cases),
                "last_case": index,
            },
        )

    summary = {
        "status": "complete",
        "objective": "full-fluid-grid fluctuation-vorticity L2",
        "mode_count": args.modes,
        "coefficient_semantics": (
            "one coefficient vector reconstructs velocity and vorticity"
        ),
        "metrics": metrics,
        "aggregate_metrics": {
            "pinn_vorticity_optimal": aggregate(metrics, prefix="pinn_"),
            "laplacian_vorticity_optimal": aggregate(metrics),
        },
    }
    write_json(output_dir / "vorticity_optimal_complete.json", summary)
    write_json(
        output_dir / "pipeline_state.json",
        {"status": "complete", "completed_cases": len(cases), "total_cases": len(cases)},
    )
    print(json.dumps(summary["aggregate_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
