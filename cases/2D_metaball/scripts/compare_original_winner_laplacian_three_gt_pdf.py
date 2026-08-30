"""Strict three-GT comparison of original/winner EigenFluid and Laplacian.

The numerical protocol intentionally reuses the matched-grid operators from
``compare_three_gt_eigenfluid_vs_laplacian.py``.  The only differences among
the three reconstructed spans are therefore the two learned checkpoints and
the geometry-specific finite-difference Dirichlet Laplacian basis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from compare_three_gt_eigenfluid_vs_laplacian import (
    eigenfluid_basis,
    finite_difference_laplacian,
    fit_velocity,
    generate_gt_parameters,
    ground_truth_fields,
    matched_grid_observables,
    matched_grid_ritz,
    serializable_gt,
)
from evaluate_connected_dirichlet_psi_basis import load_model
from metaball_runtime import fluid_mask, geometry_from_token


F32 = np.float32
BLUE = "#3C5488"
RED = "#E64B35"
GRAY = "#6F6F6F"
LIGHT_BLUE = "#4DBBD5"
INK = "#202124"
PALE = "#F3F5F7"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
            "figure.titlesize": 22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def assert_fair_geometry(original_cfg: Any, winner_cfg: Any) -> None:
    keys = (
        "modes",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "geometry_mode",
        "circle_count",
        "circle_center_min",
        "circle_center_max",
        "circle_radius_min",
        "circle_radius_max",
        "metaball_connectivity",
        "metaball_minimum_overlap",
        "metaball_outer_clearance",
        "outer_half_extent",
        "outer_superellipse_power",
        "metaball_softmin_tau",
        "outer_envelope_transition",
        "obstacle_envelope_transition",
        "envelope_mode",
        "basis_objective",
    )
    mismatches = {
        key: (getattr(original_cfg, key), getattr(winner_cfg, key))
        for key in keys
        if getattr(original_cfg, key) != getattr(winner_cfg, key)
    }
    if mismatches:
        raise ValueError(f"checkpoint geometry/problem mismatch: {mismatches}")


def grid_field(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(valid.shape, np.nan, dtype=np.float32)
    output[valid] = np.asarray(values, dtype=np.float32)
    return output


def grid_vector(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.full(valid.shape, np.nan, dtype=np.float32)
    v = np.full(valid.shape, np.nan, dtype=np.float32)
    u[valid] = np.asarray(values[:, 0], dtype=np.float32)
    v[valid] = np.asarray(values[:, 1], dtype=np.float32)
    return u, v


def robust_symmetric_limit(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = np.concatenate(
        [np.abs(array[np.isfinite(array)]).ravel() for array in arrays]
    )
    return max(float(np.percentile(finite, percentile)), 1.0e-8)


def robust_positive_limit(arrays: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])
    return max(float(np.percentile(finite, percentile)), 1.0e-8)


def method_metric(result: dict[str, Any], field: str) -> float:
    key = {
        "psi": "test_psi_relative_l2",
        "speed": "test_velocity_relative_l2",
        "omega": "test_vorticity_relative_l2",
    }[field]
    return float(result["metrics"][key])


def add_footer(fig: plt.Figure, page: int, note: str) -> None:
    fig.text(0.035, 0.018, note, ha="left", va="bottom", fontsize=9.5, color=GRAY)
    fig.text(0.965, 0.018, str(page), ha="right", va="bottom", fontsize=10, color=GRAY)


def summary_page(
    means: dict[str, dict[str, float]],
    cases: list[dict[str, Any]],
    original_epoch: int,
    winner_epoch: int,
    tokens: list[int],
    resolution: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(
        0.045,
        0.935,
        "Three unseen connected-metaball flows",
        fontsize=25,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.045,
        0.885,
        "Strict K=64 reconstruction: original-2000 vs winner-2000 vs Laplacian",
        fontsize=16,
        color=GRAY,
        ha="left",
    )

    ax_bar = fig.add_axes([0.055, 0.18, 0.49, 0.60])
    metrics = ["Streamfunction", "Velocity", "Vorticity"]
    keys = [
        "test_psi_relative_l2",
        "test_velocity_relative_l2",
        "test_vorticity_relative_l2",
    ]
    methods = ["Original-2000", "Winner-2000", "Laplacian"]
    colors = [BLUE, RED, GRAY]
    x = np.arange(len(metrics), dtype=float)
    width = 0.23
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = [means[method][key] for key in keys]
        bars = ax_bar.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=method,
            color=color,
            edgecolor="white",
            linewidth=0.7,
        )
        ax_bar.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax_bar.set_xticks(x, metrics)
    ax_bar.set_ylabel("Mean held-out relative $L^2$ error")
    ax_bar.set_ylim(0.0, max(means["Laplacian"][key] for key in keys) * 1.20)
    ax_bar.grid(axis="y", color="#D9DDE2", linewidth=0.8, alpha=0.9)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.12))

    ax_info = fig.add_axes([0.59, 0.18, 0.36, 0.64])
    ax_info.axis("off")
    winner_counts = {}
    for key in keys:
        winner_counts[key] = sum(
            min(
                case["results"]["Original-2000"]["metrics"][key],
                case["results"]["Winner-2000"]["metrics"][key],
            )
            < case["results"]["Laplacian"]["metrics"][key]
            for case in cases
        )
    winner_vs_original = {
        key: sum(
            case["results"]["Winner-2000"]["metrics"][key]
            < case["results"]["Original-2000"]["metrics"][key]
            for case in cases
        )
        for key in keys
    }
    info_lines = [
        ("Checkpoints", True),
        (f"Original best epoch: {original_epoch}", False),
        (f"Winner best epoch: {winner_epoch}", False),
        ("", False),
        ("Evaluation contract", True),
        (f"Grid: {resolution} x {resolution}; K=64", False),
        ("Fit/test nodes: 70% / 30%", False),
        ("Fit objective: matched-grid velocity", False),
        ("Boundary: psi=0 on every component", False),
        ("Same zero-extension FD operators", False),
        ("", False),
        ("Case-level outcomes", True),
        (
            "Winner beats Original: "
            + ", ".join(
                f"{label} {winner_vs_original[key]}/3"
                for label, key in zip(("psi", "u", "omega"), keys)
            ),
            False,
        ),
        (
            "Best learned beats Laplacian: "
            + ", ".join(
                f"{label} {winner_counts[key]}/3"
                for label, key in zip(("psi", "u", "omega"), keys)
            ),
            False,
        ),
        ("", False),
        ("Random unseen tokens", True),
        (", ".join(str(token) for token in tokens), False),
    ]
    y = 0.98
    for text, heading in info_lines:
        ax_info.text(
            0.0,
            y,
            text,
            transform=ax_info.transAxes,
            fontsize=15 if heading else 12.5,
            fontweight="bold" if heading else "normal",
            color=INK if heading else GRAY,
            va="top",
            wrap=True,
        )
        y -= 0.066 if heading else 0.046

    add_footer(
        fig,
        1,
        "GT: seven Gaussian vortical structures under the hard Dirichlet envelope. "
        "Metrics use disjoint held-out grid nodes.",
    )
    return fig


def case_page(
    case: dict[str, Any],
    xx: np.ndarray,
    yy: np.ndarray,
    page: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    grid = fig.add_gridspec(
        3,
        5,
        left=0.055,
        right=0.965,
        bottom=0.075,
        top=0.855,
        width_ratios=[1, 1, 1, 1, 0.045],
        hspace=0.25,
        wspace=0.14,
    )
    fig.text(
        0.055,
        0.945,
        f"Case {case['case']}  |  geometry token {case['geometry_token']}",
        fontsize=23,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.055,
        0.897,
        f"Fluid nodes {case['fluid_nodes']:,}  |  fit {case['fit_nodes']:,}  |  held-out test {case['test_nodes']:,}",
        fontsize=13.5,
        color=GRAY,
        ha="left",
    )

    method_names = ["GT", "Original-2000", "Winner-2000", "Laplacian"]
    row_fields = ["psi", "speed", "omega"]
    row_labels = [r"Streamfunction $\psi$", r"Velocity magnitude $|\mathbf{u}|$", r"Vorticity $\omega$"]
    arrays: dict[str, dict[str, np.ndarray]] = case["plot_fields"]
    valid = case["valid"]

    psi_limit = robust_symmetric_limit([arrays[name]["psi"] for name in method_names])
    omega_limit = robust_symmetric_limit([arrays[name]["omega"] for name in method_names], 98.5)
    speed_limit = robust_positive_limit([arrays[name]["speed"] for name in method_names], 99.0)
    limits = {
        "psi": (-psi_limit, psi_limit, "RdBu_r"),
        "speed": (0.0, speed_limit, "magma"),
        "omega": (-omega_limit, omega_limit, "RdBu_r"),
    }

    for row, (field, row_label) in enumerate(zip(row_fields, row_labels)):
        mappable = None
        for column, method in enumerate(method_names):
            ax = fig.add_subplot(grid[row, column])
            vmin, vmax, cmap = limits[field]
            mappable = ax.pcolormesh(
                xx,
                yy,
                arrays[method][field],
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.contour(xx, yy, valid.astype(float), levels=[0.5], colors="black", linewidths=0.7)
            if field == "speed":
                step = max(1, len(xx) // 18)
                u = arrays[method]["u"]
                v = arrays[method]["v"]
                ax.quiver(
                    xx[::step, ::step],
                    yy[::step, ::step],
                    np.nan_to_num(u[::step, ::step]),
                    np.nan_to_num(v[::step, ::step]),
                    color="white",
                    alpha=0.70,
                    scale=25.0,
                    width=0.003,
                    headwidth=3.0,
                )
            if row == 0:
                color = {"GT": INK, "Original-2000": BLUE, "Winner-2000": RED, "Laplacian": GRAY}[method]
                ax.set_title(method, fontweight="bold", color=color, pad=9)
            if column == 0:
                ax.set_ylabel(row_label, fontweight="bold", labelpad=8)
            if method != "GT":
                error = method_metric(case["results"][method], field)
                ax.text(
                    0.03,
                    0.04,
                    f"test rel. $L^2$ = {error:.3f}",
                    transform=ax.transAxes,
                    fontsize=10.5,
                    color=INK,
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.2},
                )
            ax.set_aspect("equal")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xticks([0.0, 0.5, 1.0] if row == 2 else [])
            ax.set_yticks([0.0, 0.5, 1.0] if column == 0 else [])
        cax = fig.add_subplot(grid[row, 4])
        colorbar = fig.colorbar(mappable, cax=cax)
        colorbar.ax.tick_params(labelsize=9)

    add_footer(
        fig,
        page,
        "Black contour: common fluid boundary. All displayed observables use the same zero-extended finite-difference operators.",
    )
    return fig


def save_case_arrays(
    output_path: Path,
    case: dict[str, Any],
    coordinates: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "coordinates": coordinates,
        "valid_mask": case["valid"],
        "fit_indices": case["fit_indices"],
        "test_indices": case["test_indices"],
    }
    for method, fields in case["plot_fields"].items():
        prefix = method.lower().replace("-", "_")
        for name, value in fields.items():
            arrays[f"{prefix}_{name}"] = value
    np.savez_compressed(output_path, **arrays)


def render_saved_evaluation(args: argparse.Namespace) -> None:
    summary_path = args.output_dir / "reconstruction_summary.json"
    metrics_path = args.output_dir / "reconstruction_metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("saved evaluation summary/metrics are missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if list(summary["geometry_tokens"]) != list(args.tokens):
        raise ValueError("saved geometry tokens do not match the requested tokens")
    if int(summary["grid_resolution"]) != int(args.resolution):
        raise ValueError("saved grid resolution does not match the request")

    arrays_dir = args.output_dir / "case_arrays"
    pages_dir = args.output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    coordinates: np.ndarray | None = None
    cases: list[dict[str, Any]] = []
    prefixes = {
        "GT": "gt",
        "Original-2000": "original_2000",
        "Winner-2000": "winner_2000",
        "Laplacian": "laplacian",
    }
    for case_summary in summary["cases"]:
        case_index = int(case_summary["case"])
        token = int(case_summary["geometry_token"])
        array_path = arrays_dir / f"case_{case_index:02d}_token_{token}.npz"
        with np.load(array_path, allow_pickle=False) as archive:
            current_coordinates = np.asarray(archive["coordinates"], dtype=F32)
            if coordinates is None:
                coordinates = current_coordinates
            elif not np.array_equal(coordinates, current_coordinates):
                raise ValueError("saved case coordinate grids differ")
            plot_fields = {
                method: {
                    field: np.asarray(archive[f"{prefix}_{field}"], dtype=F32)
                    for field in ("psi", "u", "v", "speed", "omega")
                }
                for method, prefix in prefixes.items()
            }
            case = {
                **case_summary,
                "valid": np.asarray(archive["valid_mask"], dtype=bool),
                "fit_indices": np.asarray(archive["fit_indices"], dtype=np.int64),
                "test_indices": np.asarray(archive["test_indices"], dtype=np.int64),
                "plot_fields": plot_fields,
                "results": {
                    method: {"metrics": metrics}
                    for method, metrics in case_summary["metrics"].items()
                },
            }
            cases.append(case)

    if coordinates is None:
        raise ValueError("saved evaluation contains no cases")
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    figures: list[plt.Figure] = [
        summary_page(
            summary["mean_test_errors"],
            cases,
            int(summary["original_checkpoint_epoch"]),
            int(summary["winner_checkpoint_epoch"]),
            list(summary["geometry_tokens"]),
            int(summary["grid_resolution"]),
        )
    ]
    figures.extend(
        case_page(case, xx, yy, index + 2) for index, case in enumerate(cases)
    )
    pdf_path = args.output_dir / "metaball_k64_three_random_gt_reconstruction_comparison.pdf"
    with PdfPages(pdf_path, metadata={"Title": "K=64 metaball reconstruction comparison"}) as pdf:
        for index, figure in enumerate(figures, start=1):
            figure.savefig(
                pages_dir / f"page_{index:02d}.png",
                dpi=220,
                facecolor="white",
                bbox_inches=None,
            )
            pdf.savefig(figure, dpi=220, facecolor="white", bbox_inches=None)
            plt.close(figure)
    print(
        json.dumps(
            {
                "pdf": str(pdf_path.resolve()),
                "metrics_csv": str(metrics_path.resolve()),
                "summary_json": str(summary_path.resolve()),
                "mean_test_errors": summary["mean_test_errors"],
                "reused_saved_evaluation": True,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--winner-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs=3, default=[8_919_975, 6_226_191, 8_009_482]
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--resolution", type=int, default=320)
    parser.add_argument("--peeling-points", type=int, default=16384)
    parser.add_argument("--vortices", type=int, default=7)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-10)
    parser.add_argument("--reuse-evaluation", action="store_true")
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_evaluation:
        render_saved_evaluation(args)
        return
    pages_dir = args.output_dir / "pages"
    arrays_dir = args.output_dir / "case_arrays"
    pages_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    original_metadata, original_cfg, original_model = load_model(args.original_checkpoint)
    winner_metadata, winner_cfg, winner_model = load_model(args.winner_checkpoint)
    assert_fair_geometry(original_cfg, winner_cfg)
    if int(original_cfg.modes) != 64:
        raise ValueError("this report requires K=64 checkpoints")

    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for case_index, token_integer in enumerate(args.tokens):
        token = float(token_integer)
        original_geometry = geometry_from_token(original_cfg, token)
        winner_geometry = geometry_from_token(winner_cfg, token)
        if not np.array_equal(original_geometry, winner_geometry):
            raise RuntimeError("geometry differs between learned checkpoints")

        valid = fluid_mask(winner_cfg, all_points, token).reshape(
            args.resolution, args.resolution
        )
        points = all_points[valid.ravel()]
        if len(points) < int(winner_cfg.modes) + 1024:
            raise ValueError(f"insufficient fluid nodes for token {token_integer}")
        split_rng = np.random.default_rng(args.seed + 1009 * case_index)
        permutation = split_rng.permutation(len(points))
        fit_count = int(0.70 * len(points))
        fit_indices = permutation[:fit_count]
        test_indices = permutation[fit_count:]

        gt_seed = args.seed + 100_003 * (case_index + 1)
        gt_parameters = generate_gt_parameters(
            winner_cfg, token, gt_seed, args.vortices
        )
        gt_psi, _, _, gt_normalization = ground_truth_fields(
            winner_cfg, points, token, gt_parameters
        )
        gt_velocity, gt_omega = matched_grid_observables(gt_psi, valid, spacing)

        results: dict[str, dict[str, Any]] = {}
        learned_diagnostics: dict[str, Any] = {}
        for method, cfg, model, offset in (
            ("Original-2000", original_cfg, original_model, 0),
            ("Winner-2000", winner_cfg, winner_model, 10_000_000),
        ):
            learned_psi, _, _, peeling, area = eigenfluid_basis(
                cfg,
                model,
                points,
                token,
                args.peeling_points,
                args.seed + 200_003 * (case_index + 1) + offset,
            )
            learned_psi, learned_eigenvalues, grid_ritz = matched_grid_ritz(
                learned_psi, valid, spacing
            )
            learned_velocity, learned_omega = matched_grid_observables(
                learned_psi, valid, spacing
            )
            results[method] = fit_velocity(
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
            learned_diagnostics[method] = {
                "peeling": peeling,
                "matched_grid_ritz": grid_ritz,
                "first16_eigenvalues": [
                    float(value) for value in learned_eigenvalues[:16]
                ],
                "fluid_area": float(area["fluid_area"]),
            }

        laplacian_values, lap_psi, lap_omega_legacy, lap_velocity_legacy = (
            finite_difference_laplacian(valid, spacing, int(winner_cfg.modes))
        )
        lap_velocity, lap_omega = matched_grid_observables(lap_psi, valid, spacing)
        velocity_operator_relative = float(
            np.max(np.abs(lap_velocity - lap_velocity_legacy))
            / max(float(np.max(np.abs(lap_velocity_legacy))), 1.0e-30)
        )
        omega_operator_relative = float(
            np.linalg.norm(
                lap_omega.astype(np.float64) - lap_omega_legacy.astype(np.float64)
            )
            / max(float(np.linalg.norm(lap_omega_legacy.astype(np.float64))), 1.0e-30)
        )
        if velocity_operator_relative > 2.0e-6 or omega_operator_relative > 2.0e-4:
            raise RuntimeError("matched Laplacian observable audit failed")
        results["Laplacian"] = fit_velocity(
            lap_psi,
            lap_velocity,
            lap_omega,
            gt_psi,
            gt_velocity,
            gt_omega,
            fit_indices,
            test_indices,
            args.ridge_relative,
        )

        gt_u, gt_v = grid_vector(gt_velocity, valid)
        plot_fields: dict[str, dict[str, np.ndarray]] = {
            "GT": {
                "psi": grid_field(gt_psi, valid),
                "u": gt_u,
                "v": gt_v,
                "speed": grid_field(np.linalg.norm(gt_velocity, axis=1), valid),
                "omega": grid_field(gt_omega, valid),
            }
        }
        for method, result in results.items():
            u, v = grid_vector(result["velocity"], valid)
            plot_fields[method] = {
                "psi": grid_field(result["psi"], valid),
                "u": u,
                "v": v,
                "speed": grid_field(np.linalg.norm(result["velocity"], axis=1), valid),
                "omega": grid_field(result["omega"], valid),
            }

        case = {
            "case": case_index + 1,
            "geometry_token": token_integer,
            "geometry_parameters": original_geometry.tolist(),
            "fluid_nodes": int(len(points)),
            "fit_nodes": int(len(fit_indices)),
            "test_nodes": int(len(test_indices)),
            "gt_seed": gt_seed,
            "gt_velocity_rms_before_normalization": gt_normalization,
            "gt_parameters": serializable_gt(gt_parameters),
            "valid": valid,
            "fit_indices": fit_indices,
            "test_indices": test_indices,
            "plot_fields": plot_fields,
            "results": results,
            "learned_diagnostics": learned_diagnostics,
            "laplacian": {
                "first16_eigenvalues": [float(value) for value in laplacian_values[:16]],
                "velocity_operator_relative_max": velocity_operator_relative,
                "vorticity_operator_relative_l2": omega_operator_relative,
            },
        }
        cases.append(case)
        save_case_arrays(
            arrays_dir / f"case_{case_index + 1:02d}_token_{token_integer}.npz",
            case,
            coordinates,
        )
        for method, result in results.items():
            rows.append(
                {
                    "case": case_index + 1,
                    "geometry_token": token_integer,
                    "method": method,
                    **result["metrics"],
                }
            )

    metric_keys = (
        "test_psi_relative_l2",
        "test_velocity_relative_l2",
        "test_vorticity_relative_l2",
    )
    means = {
        method: {
            key: float(
                np.mean(
                    [case["results"][method]["metrics"][key] for case in cases]
                )
            )
            for key in metric_keys
        }
        for method in ("Original-2000", "Winner-2000", "Laplacian")
    }

    csv_path = args.output_dir / "reconstruction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_cases = []
    for case in cases:
        summary_cases.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"valid", "fit_indices", "test_indices", "plot_fields", "results"}
            }
        )
        summary_cases[-1]["metrics"] = {
            method: result["metrics"] for method, result in case["results"].items()
        }
    summary = {
        "comparison_status": "strict_matched_grid_fairness",
        "random_seed": args.seed,
        "geometry_tokens": args.tokens,
        "grid_resolution": args.resolution,
        "modes": 64,
        "peeling_points": args.peeling_points,
        "ridge_relative": args.ridge_relative,
        "original_checkpoint": str(args.original_checkpoint.resolve()),
        "original_checkpoint_epoch": int(original_metadata["epoch"]),
        "winner_checkpoint": str(args.winner_checkpoint.resolve()),
        "winner_checkpoint_epoch": int(winner_metadata["epoch"]),
        "mean_test_errors": means,
        "fairness_contract": {
            "same_random_geometry_and_gt": True,
            "same_valid_grid_nodes": True,
            "same_fit_test_split": True,
            "same_K": 64,
            "same_ridge": args.ridge_relative,
            "same_boundary_condition": "psi=0 on every boundary component",
            "same_velocity_operator": "zero-extended centered difference",
            "same_vorticity_operator": "five-point positive -Delta",
            "learned_postprocess": "invertible matched-grid generalized Ritz rotation only",
            "coefficient_fit": "velocity least squares on 70% of valid nodes",
            "reported_error": "relative L2 on disjoint held-out 30% nodes",
        },
        "cases": summary_cases,
    }
    json_path = args.output_dir / "reconstruction_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pdf_path = args.output_dir / "metaball_k64_three_random_gt_reconstruction_comparison.pdf"
    figures: list[plt.Figure] = [
        summary_page(
            means,
            cases,
            int(original_metadata["epoch"]),
            int(winner_metadata["epoch"]),
            list(args.tokens),
            args.resolution,
        )
    ]
    figures.extend(
        case_page(case, xx, yy, index + 2) for index, case in enumerate(cases)
    )
    with PdfPages(pdf_path, metadata={"Title": "K=64 metaball reconstruction comparison"}) as pdf:
        for index, figure in enumerate(figures, start=1):
            figure.savefig(
                pages_dir / f"page_{index:02d}.png",
                dpi=220,
                facecolor="white",
                bbox_inches=None,
            )
            pdf.savefig(figure, dpi=220, facecolor="white", bbox_inches=None)
            plt.close(figure)

    print(
        json.dumps(
            {
                "pdf": str(pdf_path.resolve()),
                "metrics_csv": str(csv_path.resolve()),
                "summary_json": str(json_path.resolve()),
                "mean_test_errors": means,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
