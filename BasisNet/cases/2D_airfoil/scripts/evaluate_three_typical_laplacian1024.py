from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from evaluate_laplacian_baseline import (  # noqa: E402
    laplacian_mode_table,
    mode_fields,
    reconstruct_laplacian,
    velocity_panel,
)


VELOCITY_METRICS = (
    "velocity_error",
    "fluctuation_velocity_error",
    "wake_velocity_error",
    "wake_fluctuation_velocity_error",
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def baseline_complete(path: Path, mode_count: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if len(archive["eigenvalues"]) != mode_count:
                return False
            if len(archive["coefficients"]) != mode_count:
                return False
            values = [float(archive[name]) for name in VELOCITY_METRICS]
            return all(math.isfinite(value) for value in values)
    except (OSError, ValueError, KeyError):
        return False


def render_comparison(
    source_path: Path,
    baseline_path: Path,
    output: Path,
    mode_count: int,
) -> None:
    source = load_npz(source_path)
    lap = load_npz(baseline_path)
    x, y = source["x"], source["y"]
    solid = source["solid"].astype(bool)
    pinn_du = source["pinn_u"] - source["actual_u"]
    pinn_dv = source["pinn_v"] - source["actual_v"]
    lap_du = lap["laplacian_u"] - source["actual_u"]
    lap_dv = lap["laplacian_v"] - source["actual_v"]
    speed_limit = max(
        float(np.percentile(np.hypot(source["actual_u"], source["actual_v"])[~solid], 99.5)),
        1.0e-8,
    )
    errors = np.concatenate(
        (
            np.hypot(pinn_du, pinn_dv)[~solid][::8],
            np.hypot(lap_du, lap_dv)[~solid][::8],
        )
    )
    error_limit = max(float(np.percentile(errors, 99.0)), 1.0e-8)
    figure, axes = plt.subplots(
        3, 2, figsize=(15.5, 8.8), constrained_layout=True
    )
    velocity_panel(
        axes[0, 0],
        x,
        y,
        source["actual_u"],
        source["actual_v"],
        solid,
        "GT velocity magnitude + streamlines",
        speed_limit,
    )
    axes[0, 1].axis("off")
    rows = [
        ("Total velocity", "velocity_error"),
        ("Fluctuation velocity", "fluctuation_velocity_error"),
        ("Wake velocity", "wake_velocity_error"),
        ("Wake fluctuation velocity", "wake_fluctuation_velocity_error"),
    ]
    text = ["Velocity-L2 reconstruction error", "", "metric                 PINN       Laplacian"]
    for label, name in rows:
        text.append(
            f"{label:<23} {100*float(source[name]):6.2f}%    "
            f"{100*float(lap[name]):6.2f}%"
        )
    text.extend(
        (
            "",
            f"Laplacian modes generated: {mode_count}",
            f"Laplacian projection rank: {int(lap['projection_rank'])}",
            f"Projection sample points: {int(lap['projection_points'])}",
            "Both methods use the same GT temporal mean and frame.",
        )
    )
    axes[0, 1].text(
        0.04,
        0.96,
        "\n".join(text),
        transform=axes[0, 1].transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
    )
    velocity_panel(
        axes[1, 0],
        x,
        y,
        source["pinn_u"],
        source["pinn_v"],
        solid,
        f"PINN-K1024 velocity reconstruction, L2={100*float(source['velocity_error']):.2f}%",
        speed_limit,
    )
    velocity_panel(
        axes[1, 1],
        x,
        y,
        pinn_du,
        pinn_dv,
        solid,
        "PINN absolute velocity-vector error",
        error_limit,
        cmap="magma",
    )
    velocity_panel(
        axes[2, 0],
        x,
        y,
        lap["laplacian_u"],
        lap["laplacian_v"],
        solid,
        f"Laplacian-K{mode_count} velocity reconstruction, L2={100*float(lap['velocity_error']):.2f}%",
        speed_limit,
    )
    velocity_panel(
        axes[2, 1],
        x,
        y,
        lap_du,
        lap_dv,
        solid,
        "Laplacian absolute velocity-vector error",
        error_limit,
        cmap="magma",
    )
    figure.suptitle(
        rf"PINN vs rectangular Laplacian: $\beta={float(source['beta']):.3f}$, "
        rf"$\alpha={float(source['alpha_deg']):+.2f}^\circ$",
        fontsize=13,
    )
    figure.savefig(output, dpi=130, facecolor="white")
    plt.close(figure)


def render_lowest_six(
    source_path: Path,
    output: Path,
    bounds: tuple[float, float, float, float],
) -> list[float]:
    source = load_npz(source_path)
    x = source["x"].astype(np.float64)[::2]
    y = source["y"].astype(np.float64)[::2]
    solid = source["solid"].astype(bool)[::2, ::2]
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    modes = laplacian_mode_table(6, *bounds)
    u, v, _ = mode_fields(points, bounds, modes)
    u = u.reshape(len(y), len(x), 6)
    v = v.reshape(len(y), len(x), 6)
    limits = [
        max(float(np.percentile(np.hypot(u[:, :, i], v[:, :, i])[~solid], 99.5)), 1.0e-8)
        for i in range(6)
    ]
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 8.4), constrained_layout=True)
    for index, axis in enumerate(axes.ravel()):
        velocity_panel(
            axis,
            x,
            y,
            u[:, :, index],
            v[:, :, index],
            solid,
            f"Laplacian mode {index + 1}, lambda={modes[2][index]:.6f}",
            limits[index],
        )
    figure.suptitle(
        rf"Six lowest rectangular Laplacian modes, $\beta={float(source['beta']):.3f}$, "
        rf"$\alpha={float(source['alpha_deg']):+.2f}^\circ$",
        fontsize=13,
    )
    figure.savefig(output, dpi=140, facecolor="white")
    plt.close(figure)
    return [float(value) for value in modes[2]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Laplacian-K1024 and PINN-K1024 velocity reconstructions on three airfoil cases"
    )
    parser.add_argument("--pinn-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=1024)
    parser.add_argument("--projection-points", type=int, default=32768)
    parser.add_argument("--field-batch", type=int, default=32768)
    args = parser.parse_args()

    pinn_root = args.pinn_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(
        **json.loads(args.airfoil_config.read_text(encoding="utf-8"))
    )
    bounds = (cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax)
    source_summary = json.loads(
        (pinn_root / "three_case_summary.json").read_text(encoding="utf-8")
    )
    state_path = output_root / "pipeline_state.json"
    results = []
    for index, case in enumerate(source_summary["cases"], start=1):
        source_path = Path(case["sample"])
        gt_path = Path(case["gt"])
        label = source_path.parent.name
        case_dir = output_root / label
        case_dir.mkdir(exist_ok=True)
        baseline_path = case_dir / "laplacian1024_reconstruction.npz"
        write_json(
            state_path,
            {
                "stage": "laplacian_velocity_projection",
                "completed_cases": index - 1,
                "total_cases": 3,
                "current_case": label,
                "beta": case["beta"],
                "alpha_deg": case["alpha_deg"],
            },
        )
        if not baseline_complete(baseline_path, args.modes):
            print(
                f"[laplacian {index}/3] beta={float(case['beta']):.3f} "
                f"alpha={float(case['alpha_deg']):+.2f}",
                flush=True,
            )
            lap_metrics = reconstruct_laplacian(
                gt_path,
                source_path,
                baseline_path,
                cfg,
                args.modes,
                args.projection_points,
                args.field_batch,
            )
        else:
            lap = load_npz(baseline_path)
            lap_metrics = {
                "beta": float(case["beta"]),
                "alpha_deg": float(case["alpha_deg"]),
                "frame": int(case["selected_frame"]),
                "projection_rank": int(lap["projection_rank"]),
                "projection_points": int(lap["projection_points"]),
                "projection_relative_residual": float(
                    lap["projection_relative_residual"]
                ),
                **{name: float(lap[name]) for name in VELOCITY_METRICS},
            }
        if not baseline_complete(baseline_path, args.modes):
            raise RuntimeError(f"invalid Laplacian output for {label}")

        comparison = case_dir / "pinn_vs_laplacian1024_velocity.png"
        modes_figure = case_dir / "laplacian_lowest6.png"
        render_comparison(source_path, baseline_path, comparison, args.modes)
        first_six = render_lowest_six(source_path, modes_figure, bounds)
        pinn_metrics = {
            name: float(case["metrics"][name]) for name in VELOCITY_METRICS
        }
        result = {
            "label": label,
            "beta": float(case["beta"]),
            "alpha_deg": float(case["alpha_deg"]),
            "selected_frame": int(case["selected_frame"]),
            "snapshot_count": int(case["snapshot_count"]),
            "mode_count": args.modes,
            "first_six_eigenvalues": first_six,
            "laplacian_projection_rank": int(lap_metrics["projection_rank"]),
            "laplacian_projection_points": int(lap_metrics["projection_points"]),
            "laplacian_projection_relative_residual": float(
                lap_metrics["projection_relative_residual"]
            ),
            "pinn": pinn_metrics,
            "laplacian": {
                name: float(lap_metrics[name]) for name in VELOCITY_METRICS
            },
            "laplacian_to_pinn_error_ratio": {
                name: float(lap_metrics[name]) / max(pinn_metrics[name], 1.0e-30)
                for name in VELOCITY_METRICS
            },
            "comparison_figure": str(comparison),
            "laplacian_lowest6_figure": str(modes_figure),
            "laplacian_sample": str(baseline_path),
        }
        write_json(case_dir / "comparison_summary.json", result)
        results.append(result)
        write_json(
            state_path,
            {
                "stage": "laplacian_velocity_projection",
                "completed_cases": index,
                "total_cases": 3,
                "last_case": label,
                "beta": case["beta"],
                "alpha_deg": case["alpha_deg"],
            },
        )

    _, _, all_eigenvalues, _ = laplacian_mode_table(args.modes, *bounds)
    summary = {
        "status": "complete",
        "definition": {
            "name": "rectangular_dirichlet_laplacian_streamfunction",
            "mode_count": args.modes,
            "streamfunction": "sin(m*pi*(x-xmin)/Lx)*sin(n*pi*(y-ymin)/Ly)",
            "velocity": "(d psi/dy, -d psi/dx)",
            "ordering": "ascending Laplacian eigenvalue",
            "coefficient_fit": "velocity_l2",
            "projection_points": args.projection_points,
            "mean_field": "same temporal GT mean as PINN",
            "airfoil_boundary": "not enforced; solid points excluded from fit and metrics",
        },
        "generated_eigenvalue_min": float(all_eigenvalues[0]),
        "generated_eigenvalue_max": float(all_eigenvalues[-1]),
        "cases": results,
    }
    summary_path = output_root / "laplacian1024_vs_pinn_summary.json"
    write_json(summary_path, summary)
    write_json(
        state_path,
        {
            "stage": "complete",
            "completed_cases": 3,
            "total_cases": 3,
            "summary": str(summary_path),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
