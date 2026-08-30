"""Compare cached Lq and newly evaluated Lomega on the same stride-2 sample."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "velocity_error",
    "vorticity_error",
    "fluctuation_velocity_error",
    "fluctuation_vorticity_error",
    "wake_velocity_error",
    "wake_vorticity_error",
)


def relative_scalar(actual: np.ndarray, estimate: np.ndarray, mask: np.ndarray) -> float:
    delta = estimate[mask].astype(np.float64) - actual[mask].astype(np.float64)
    denominator = np.linalg.norm(actual[mask].astype(np.float64))
    return float(np.linalg.norm(delta) / denominator)


def relative_velocity(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    mask: np.ndarray,
) -> float:
    du = estimate_u[mask].astype(np.float64) - actual_u[mask].astype(np.float64)
    dv = estimate_v[mask].astype(np.float64) - actual_v[mask].astype(np.float64)
    au = actual_u[mask].astype(np.float64)
    av = actual_v[mask].astype(np.float64)
    return float(np.sqrt(np.sum(du * du + dv * dv) / np.sum(au * au + av * av)))


def metrics(
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    mean_omega: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    estimate_omega: np.ndarray,
    fluid: np.ndarray,
    wake: np.ndarray,
) -> dict[str, float]:
    fluctuation_u = estimate_u - mean_u
    fluctuation_v = estimate_v - mean_v
    fluctuation_omega = estimate_omega - mean_omega
    target_u = actual_u - mean_u
    target_v = actual_v - mean_v
    target_omega = actual_omega - mean_omega
    return {
        "velocity_error": relative_velocity(actual_u, actual_v, estimate_u, estimate_v, fluid),
        "vorticity_error": relative_scalar(actual_omega, estimate_omega, fluid),
        "fluctuation_velocity_error": relative_velocity(
            target_u, target_v, fluctuation_u, fluctuation_v, fluid
        ),
        "fluctuation_vorticity_error": relative_scalar(
            target_omega, fluctuation_omega, fluid
        ),
        "wake_velocity_error": relative_velocity(
            actual_u, actual_v, estimate_u, estimate_v, wake
        ),
        "wake_vorticity_error": relative_scalar(actual_omega, estimate_omega, wake),
    }


def strict_training_seconds(path: Path) -> tuple[int, float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid training history: {path}")
    if "training_elapsed_seconds" in rows[-1]:
        return int(rows[-1]["epoch"]), float(rows[-1]["training_elapsed_seconds"])
    total = 0.0
    previous = 0.0
    for row in rows:
        current = float(row["seconds"])
        if current < previous:
            total += previous
        previous = current
    total += previous
    return int(rows[-1]["epoch"]), total


def assert_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right, equal_nan=True):
        difference = float(np.nanmax(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        raise ValueError(f"fairness check failed for {name}: max_abs={difference}")


def masked(field: np.ndarray, solid: np.ndarray) -> np.ndarray:
    result = field.astype(np.float64).copy()
    result[solid] = np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lq-fields", type=Path, required=True)
    parser.add_argument("--lomega-archive", type=Path, required=True)
    parser.add_argument("--lomega-complete", type=Path, required=True)
    parser.add_argument("--lq-training-history", type=Path, required=True)
    parser.add_argument("--lomega-training-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lomega_complete = json.loads(args.lomega_complete.read_text(encoding="utf-8"))
    if not lomega_complete.get("complete"):
        raise ValueError("Lomega evaluation is not complete")
    with np.load(args.lq_fields, allow_pickle=False) as source:
        lq = {name: source[name] for name in source.files}
    with np.load(args.lomega_archive, allow_pickle=False) as source:
        lomega = {name: source[name] for name in source.files}

    assert_equal("x", lq["x"], lomega["x"])
    assert_equal("y", lq["y"], lomega["y"])
    assert_equal("solid", lq["solid"], lomega["solid"])
    assert int(lq["frames"][0]) == int(lomega["frame"]) == 30
    for name in ("actual_u", "actual_v", "actual_omega"):
        assert_equal(name, lq[name][0], lomega[name])

    actual_u = lomega["actual_u"]
    actual_v = lomega["actual_v"]
    actual_omega = lomega["actual_omega"]
    mean_u = lomega["mean_u"]
    mean_v = lomega["mean_v"]
    mean_omega = lomega["mean_omega"]
    solid = lomega["solid"].astype(bool)
    fluid = ~solid
    wake = lomega["wake"].astype(bool) & fluid
    estimates = {
        "Lq": (
            lq["learned_k1024_u"][0],
            lq["learned_k1024_v"][0],
            lq["learned_k1024_omega"][0],
        ),
        "Lomega": (
            lomega["estimate_u"],
            lomega["estimate_v"],
            lomega["estimate_omega"],
        ),
    }
    results = {
        method: metrics(
            actual_u,
            actual_v,
            actual_omega,
            mean_u,
            mean_v,
            mean_omega,
            estimate[0],
            estimate[1],
            estimate[2],
            fluid,
            wake,
        )
        for method, estimate in estimates.items()
    }
    if not all(np.isfinite(value) for row in results.values() for value in row.values()):
        raise ValueError("non-finite reconstruction metric")

    metrics_csv = args.output_dir / "reconstruction_metrics_stride2.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("metric", "lq", "lomega", "lomega_minus_lq", "lomega_relative_change_percent"),
        )
        writer.writeheader()
        for name in METRICS:
            lq_value = results["Lq"][name]
            lomega_value = results["Lomega"][name]
            writer.writerow(
                {
                    "metric": name,
                    "lq": lq_value,
                    "lomega": lomega_value,
                    "lomega_minus_lq": lomega_value - lq_value,
                    "lomega_relative_change_percent": 100.0 * (lomega_value / lq_value - 1.0),
                }
            )

    lq_steps, lq_training = strict_training_seconds(args.lq_training_history)
    lomega_steps, lomega_training = strict_training_seconds(args.lomega_training_history)
    timing_csv = args.output_dir / "timing_comparison.csv"
    with timing_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "method",
                "training_steps",
                "strict_training_seconds",
                "strict_training_hours",
                "seconds_per_training_step",
                "comparison_basis_preprocessing_seconds",
                "comparison_projection_and_reconstruction_seconds",
                "comparison_total_evaluation_seconds",
                "comparison_note",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "Lq",
                "training_steps": lq_steps,
                "strict_training_seconds": lq_training,
                "strict_training_hours": lq_training / 3600.0,
                "seconds_per_training_step": lq_training / lq_steps,
                "comparison_basis_preprocessing_seconds": 0.0,
                "comparison_projection_and_reconstruction_seconds": 0.0,
                "comparison_total_evaluation_seconds": 0.0,
                "comparison_note": "reused the existing K1024 fields_stride2 archive; no Lq model evaluation",
            }
        )
        writer.writerow(
            {
                "method": "Lomega",
                "training_steps": lomega_steps,
                "strict_training_seconds": lomega_training,
                "strict_training_hours": lomega_training / 3600.0,
                "seconds_per_training_step": lomega_training / lomega_steps,
                "comparison_basis_preprocessing_seconds": lomega_complete[
                    "basis_preprocessing_wall_seconds"
                ],
                "comparison_projection_and_reconstruction_seconds": lomega_complete[
                    "online_reconstruction_wall_seconds"
                ],
                "comparison_total_evaluation_seconds": lomega_complete[
                    "total_evaluation_wall_seconds"
                ],
                "comparison_note": "fresh Lomega evaluation on shared projection points and cached stride-2 grid",
            }
        )

    speed_actual = np.sqrt(actual_u**2 + actual_v**2)
    fluctuation_actual_u = actual_u - mean_u
    fluctuation_actual_v = actual_v - mean_v
    fields: list[tuple[str, list[np.ndarray], str]] = []
    total_speed = [speed_actual]
    total_omega = [actual_omega]
    fluctuation_speed = [np.sqrt(fluctuation_actual_u**2 + fluctuation_actual_v**2)]
    fluctuation_omega = [actual_omega - mean_omega]
    for method in ("Lq", "Lomega"):
        u, v, omega = estimates[method]
        total_speed.append(np.sqrt(u**2 + v**2))
        total_omega.append(omega)
        fluctuation_speed.append(np.sqrt((u - mean_u) ** 2 + (v - mean_v) ** 2))
        fluctuation_omega.append(omega - mean_omega)
    fields.extend(
        (
            ("Speed", total_speed, "viridis"),
            ("Vorticity", total_omega, "RdBu_r"),
            ("Fluctuation speed", fluctuation_speed, "viridis"),
            ("Fluctuation vorticity", fluctuation_omega, "RdBu_r"),
        )
    )
    extent = [float(lomega["x"][0]), float(lomega["x"][-1]), float(lomega["y"][0]), float(lomega["y"][-1])]
    fig, axes = plt.subplots(4, 3, figsize=(12.6, 10.2), constrained_layout=True)
    columns = ("Ground truth", r"$L_q$", r"$L_\omega$")
    for row, (label, row_fields, cmap) in enumerate(fields):
        arrays = [masked(value, solid) for value in row_fields]
        if "Vorticity" in label or "vorticity" in label:
            limit = max(float(np.nanpercentile(np.abs(value), 99.5)) for value in arrays)
            vmin, vmax = -limit, limit
        else:
            vmin = 0.0
            vmax = max(float(np.nanpercentile(value, 99.5)) for value in arrays)
        image = None
        for column, (axis, value) in enumerate(zip(axes[row], arrays, strict=True)):
            axis.set_facecolor("#303030")
            image = axis.imshow(
                value,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            if row == 0:
                axis.set_title(columns[column], fontsize=11)
            if column == 0:
                axis.set_ylabel(label, fontsize=10)
            if row == 3:
                axis.set_xlabel("x")
            axis.set_xticks((0, 5, 10))
            axis.set_yticks((-4, 0, 4))
        fig.colorbar(image, ax=axes[row].tolist(), shrink=0.82, pad=0.012)
    fig.suptitle(
        r"Airfoil, up geometry, frame 30, $K=1024$; identical projection and stride-2 grids",
        fontsize=12,
    )
    figure_path = args.output_dir / "lq_vs_lomega_k1024_up_frame30.png"
    fig.savefig(figure_path, dpi=260, facecolor="white")
    plt.close(fig)

    summary: dict[str, Any] = {
        "complete": True,
        "fairness_checks_passed": True,
        "single_variable_formulation_ablation": False,
        "limitation": (
            "The historical Lomega model differs from Lq in width, precision, "
            "optimizer schedule, and training budget; evaluation data and projection are shared."
        ),
        "sample": {"orientation": "up", "frame": 30, "modes": 1024},
        "grid": {
            "shape": list(solid.shape),
            "grid_points": int(solid.size),
            "fluid_points": int(np.count_nonzero(fluid)),
            "wake_points": int(np.count_nonzero(wake)),
            "stride": 2,
            "projection_points": int(lomega_complete["projection_points"]),
            "peeling_points": int(lomega_complete["basis_points"]),
        },
        "metrics": results,
        "training": {
            "Lq": {"steps": lq_steps, "seconds": lq_training},
            "Lomega": {"steps": lomega_steps, "seconds": lomega_training},
        },
        "lomega_evaluation": lomega_complete,
        "outputs": {
            "metrics_csv": str(metrics_csv.resolve()),
            "timing_csv": str(timing_csv.resolve()),
            "figure_png": str(figure_path.resolve()),
        },
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
