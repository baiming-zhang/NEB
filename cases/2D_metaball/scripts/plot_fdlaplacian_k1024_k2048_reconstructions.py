"""Render matched K=1024 and K=2048 FD-Laplacian reconstructions."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_three_gt_eigenfluid_vs_laplacian import (
    fit_velocity,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
)
from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian
from evaluate_velocity_k1024_three_unseen import common_velocity_mass_whitening
from metaball_runtime import backend, fluid_mask, install_backend


F32 = np.float32
CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    CASE_ROOT
    / "configs/k2048_metaball_connected_siren_velocity_rayleigh_omega224_onepass_2000.json"
)


def masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(valid.shape, np.nan, dtype=F32)
    result[valid] = np.asarray(values, dtype=F32)
    return result


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def colormaps() -> tuple[Any, Any, Any]:
    diverging = plt.get_cmap("RdBu_r").copy()
    sequential = plt.get_cmap("Blues").copy()
    error = plt.get_cmap("magma").copy()
    for cmap in (diverging, sequential, error):
        cmap.set_bad("#E6E6E6")
    return diverging, sequential, error


def limits(case: dict[str, Any]) -> dict[str, float]:
    gt_omega = case["gt_omega_grid"]
    gt_speed = case["gt_speed_grid"]
    omega_errors = [
        np.abs(case["results"][m]["omega_grid"] - gt_omega) for m in (1024, 2048)
    ]
    speed_errors = [
        np.abs(case["results"][m]["speed_grid"] - gt_speed) for m in (1024, 2048)
    ]
    return {
        "omega": max(float(np.nanpercentile(np.abs(gt_omega), 99.0)), 1.0e-8),
        "speed": max(float(np.nanpercentile(gt_speed, 99.0)), 1.0e-8),
        "omega_error": max(
            *(float(np.nanpercentile(value, 99.0)) for value in omega_errors),
            1.0e-8,
        ),
        "speed_error": max(
            *(float(np.nanpercentile(value, 99.0)) for value in speed_errors),
            1.0e-8,
        ),
    }


def render_method(
    output_stem: Path, cases: list[dict[str, Any]], modes: int
) -> None:
    diverging, sequential, error_cmap = colormaps()
    fig, axes = plt.subplots(
        len(cases), 6, figsize=(18.2, 8.8), constrained_layout=True
    )
    headers = (
        r"Ground truth $\omega$",
        rf"FD Laplacian $K={modes}$",
        r"Absolute error $|\Delta\omega|$",
        r"Ground truth $\|\mathbf{{u}}\|$",
        rf"FD Laplacian $K={modes}$",
        r"Absolute error $|\Delta\mathbf{u}|$",
    )
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontweight="bold", pad=7)

    for row, case in enumerate(cases):
        result = case["results"][modes]
        scale = case["limits"]
        omega_error = np.abs(result["omega_grid"] - case["gt_omega_grid"])
        speed_error = np.abs(result["speed_grid"] - case["gt_speed_grid"])
        panels = (
            (case["gt_omega_grid"], diverging, -scale["omega"], scale["omega"]),
            (result["omega_grid"], diverging, -scale["omega"], scale["omega"]),
            (omega_error, error_cmap, 0.0, scale["omega_error"]),
            (case["gt_speed_grid"], sequential, 0.0, scale["speed"]),
            (result["speed_grid"], sequential, 0.0, scale["speed"]),
            (speed_error, error_cmap, 0.0, scale["speed_error"]),
        )
        for column, (field, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if column in (2, 5):
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
        axes[row, 0].set_ylabel(
            f"Unseen {row + 1}\nToken {case['token']}",
            fontweight="bold",
            rotation=90,
        )
        metrics = result["metrics"]
        axes[row, 1].text(
            0.5,
            0.025,
            rf"$e_\omega={metrics['test_vorticity_relative_l2']:.3f}$",
            transform=axes[row, 1].transAxes,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.2},
        )
        axes[row, 4].text(
            0.5,
            0.025,
            rf"$e_u={metrics['test_velocity_relative_l2']:.3f}$",
            transform=axes[row, 4].transAxes,
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.2},
        )

    fig.suptitle(
        rf"Geometry-aware FD Laplacian reconstruction, $K={modes}$",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_comparison(output_stem: Path, cases: list[dict[str, Any]]) -> None:
    diverging, sequential, _ = colormaps()
    fig, axes = plt.subplots(
        len(cases), 6, figsize=(17.6, 8.8), constrained_layout=True
    )
    headers = (
        r"Ground truth $\omega$",
        r"FD $K=1024$: $\omega$",
        r"FD $K=2048$: $\omega$",
        r"Ground truth $\|\mathbf{u}\|$",
        r"FD $K=1024$: $\|\mathbf{u}\|$",
        r"FD $K=2048$: $\|\mathbf{u}\|$",
    )
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontweight="bold", pad=7)
    for row, case in enumerate(cases):
        scale = case["limits"]
        panels = (
            (case["gt_omega_grid"], diverging, -scale["omega"], scale["omega"]),
            (case["results"][1024]["omega_grid"], diverging, -scale["omega"], scale["omega"]),
            (case["results"][2048]["omega_grid"], diverging, -scale["omega"], scale["omega"]),
            (case["gt_speed_grid"], sequential, 0.0, scale["speed"]),
            (case["results"][1024]["speed_grid"], sequential, 0.0, scale["speed"]),
            (case["results"][2048]["speed_grid"], sequential, 0.0, scale["speed"]),
        )
        for column, (field, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row, column]
            axis.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(
            f"Unseen {row + 1}\nToken {case['token']}",
            fontweight="bold",
        )
        for column, modes in ((1, 1024), (2, 2048)):
            metric = case["results"][modes]["metrics"]["test_vorticity_relative_l2"]
            axes[row, column].text(
                0.5, 0.025, rf"$e_\omega={metric:.3f}$",
                transform=axes[row, column].transAxes, ha="center", va="bottom",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.2},
            )
        for column, modes in ((4, 1024), (5, 2048)):
            metric = case["results"][modes]["metrics"]["test_velocity_relative_l2"]
            axes[row, column].text(
                0.5, 0.025, rf"$e_u={metric:.3f}$",
                transform=axes[row, column].transAxes, ha="center", va="bottom",
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.2},
            )
    fig.suptitle(
        "Geometry-aware FD Laplacian reconstruction: K=1024 versus K=2048",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[4_000_001, 4_000_002, 4_000_003]
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.resolution != 128:
        raise ValueError("this comparison is fixed to a 128 x 128 grid")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case_index, token_integer in enumerate(args.tokens):
        token = float(token_integer)
        valid = fluid_mask(cfg, all_points, token).reshape(args.resolution, args.resolution)
        points = all_points[valid.ravel()]
        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]
        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(cfg, token, gt_seed, args.vortices)
        gt_psi, _, _, _ = ground_truth_fields(cfg, points, token, gt_parameters)
        gt_velocity, gt_omega = matched_grid_observables(gt_psi, valid, spacing)
        case: dict[str, Any] = {
            "token": token_integer,
            "valid": valid,
            "gt_omega_grid": masked(gt_omega, valid),
            "gt_speed_grid": masked(np.linalg.norm(gt_velocity, axis=1), valid),
            "results": {},
        }

        for modes in (1024, 2048):
            method_started = time.perf_counter()
            eigenvalues, psi, raw_omega, raw_velocity = finite_difference_laplacian(
                valid, spacing, modes
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
            elapsed = time.perf_counter() - method_started
            case["results"][modes] = {
                "omega_grid": masked(reconstruction["omega"], valid),
                "speed_grid": masked(
                    np.linalg.norm(reconstruction["velocity"], axis=1), valid
                ),
                "metrics": reconstruction["metrics"],
                "eigenvalue_min": float(eigenvalues[0]),
                "eigenvalue_max": float(eigenvalues[-1]),
                "whitening": whitening,
                "seconds": elapsed,
            }
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "modes": modes,
                    **reconstruction["metrics"],
                    "eigenvalue_min": float(eigenvalues[0]),
                    "eigenvalue_max": float(eigenvalues[-1]),
                    "seconds": elapsed,
                }
            )
            print(
                json.dumps(
                    {
                        "case": case_index + 1,
                        "token": token_integer,
                        "modes": modes,
                        "metrics": reconstruction["metrics"],
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
            del psi, velocity, omega, reconstruction
            gc.collect()
        case["limits"] = limits(case)
        cases.append(case)

    style()
    render_method(args.output_dir / "fd_laplacian_k1024_three_unseen", cases, 1024)
    render_method(args.output_dir / "fd_laplacian_k2048_three_unseen", cases, 2048)
    render_comparison(
        args.output_dir / "fd_laplacian_k1024_vs_k2048_three_unseen", cases
    )

    csv_path = args.output_dir / "fd_laplacian_k1024_vs_k2048_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    means = {
        str(modes): {
            key: float(np.mean([row[key] for row in rows if row["modes"] == modes]))
            for key in (
                "test_psi_relative_l2",
                "test_velocity_relative_l2",
                "test_vorticity_relative_l2",
            )
        }
        for modes in (1024, 2048)
    }
    summary = {
        "status": "complete",
        "case": "2D connected three-circle smooth metaball",
        "geometry_tokens": args.tokens,
        "resolution": args.resolution,
        "methods": ["geometry-aware FD Laplacian K=1024", "geometry-aware FD Laplacian K=2048"],
        "same_gt_mask_split_and_operators": True,
        "fit": "velocity least squares on 70% of valid nodes",
        "test": "relative L2 on disjoint held-out 30% valid nodes",
        "mean_test_errors": means,
        "total_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    (args.output_dir / "fd_laplacian_k1024_vs_k2048_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
