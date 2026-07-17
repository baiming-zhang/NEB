"""NS/PINN/error movie evaluated directly at 101 alpha values.

The three rows show the independent NS reference, the PINN K=96
reconstruction, and the absolute velocity error. Every alpha performs an
independent PINN forward/Ritz extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import cKDTree

from eigenfluid_sss.basisnet.karman2d.ns_reference import (
    ScatteredDifferentiator,
    upstream_extension,
)
from eigenfluid_sss.basisnet.karman2d.postprocess import (
    basis_fields,
    compute_ritz_basis,
    encode_mp4,
    load_model,
    quadrature_weights,
    relative_scalar_error,
    relative_vector_error,
    triangulation,
    weighted_projection,
)


FIELD_KEYS = (
    "actual_u",
    "actual_v",
    "actual_psi",
    "actual_omega",
    "pinn_u",
    "pinn_v",
    "pinn_psi",
    "pinn_omega",
)


def geometry(ax, cylinder_x: float, radius: float) -> None:
    ax.add_patch(
        Rectangle((0, -2), 10, 4, fill=False, edgecolor="black", linewidth=0.55)
    )
    ax.add_patch(
        Circle(
            (cylinder_x, 0),
            radius,
            facecolor="#303030",
            edgecolor="black",
            linewidth=0.5,
            zorder=20,
        )
    )
    ax.set(xlim=(0, 10), ylim=(-2, 2))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def phase_snapshot(actual, ids, upstream, phase: float):
    count = int(actual["phase_frames"])
    position = (phase % 1.0) * (count - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, count - 1)
    fraction = position - lower

    def blend(key, upstream_key):
        source = (1.0 - fraction) * actual[key][ids, lower] + fraction * actual[key][
            ids, upper
        ]
        return np.r_[source, upstream[upstream_key]]

    return {
        "actual_u": blend("phase_u", "u"),
        "actual_v": blend("phase_v", "v"),
        "actual_psi": blend("phase_psi", "psi"),
        "actual_omega": blend("phase_omega", "omega"),
    }


def grid_field(triangulation, points, values, xgrid, ygrid, nearest):
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    gridded = np.ma.filled(interpolator(xgrid, ygrid), np.nan)
    missing = ~np.isfinite(gridded)
    if np.any(missing):
        flat_values = values[nearest]
        gridded.ravel()[missing.ravel()] = flat_values[missing.ravel()]
    return gridded.astype(np.float32)


def build_sample(
    model,
    cfg,
    actual,
    alpha: float,
    phase: float,
    coefficients: np.ndarray,
    rank: int,
    xgrid,
    ygrid,
):
    if rank < cfg.modes:
        raise RuntimeError(f"alpha={alpha:.2f}: rank={rank} < modes={cfg.modes}")

    cx = cfg.centre_left + (cfg.centre_right - cfg.centre_left) * alpha
    sx = actual["xr"].astype(float) + cx
    sy = actual["yr"].astype(float)
    keep = (
        (sx >= 0)
        & (sx <= 10)
        & (sy >= -2)
        & (sy <= 2)
        & ((sx - cx) ** 2 + sy**2 >= (0.995 * cfg.radius) ** 2)
    )
    ids = np.flatnonzero(keep)
    source = np.c_[sx[keep], sy[keep]]
    xmin = max(float(source[:, 0].min()), 0)
    upstream = upstream_extension(xmin, 202607143, 900)
    points = np.vstack((source, np.c_[upstream["x"], upstream["y"]])).astype(np.float32)
    weights = quadrature_weights(points)

    basis_psi, basis_u, basis_v = basis_fields(model, points, alpha, coefficients)
    lift_u = np.r_[actual["mean_u"][ids], upstream["u"]]
    lift_v = np.r_[actual["mean_v"][ids], upstream["v"]]
    lift_psi = np.r_[actual["mean_psi"][ids], upstream["psi"]]
    fields = phase_snapshot(actual, ids, upstream, phase)
    modal_coefficients = weighted_projection(
        fields["actual_u"] - lift_u,
        fields["actual_v"] - lift_v,
        basis_u,
        basis_v,
        weights,
    )
    fields["pinn_u"] = lift_u + basis_u @ modal_coefficients
    fields["pinn_v"] = lift_v + basis_v @ modal_coefficients
    fields["pinn_psi"] = lift_psi + basis_psi @ modal_coefficients
    differentiator = ScatteredDifferentiator(points)
    grad_u = differentiator.gradient(fields["pinn_u"])
    grad_v = differentiator.gradient(fields["pinn_v"])
    fields["pinn_omega"] = grad_v[:, 0] - grad_u[:, 1]

    velocity_error = relative_vector_error(
        fields["actual_u"],
        fields["actual_v"],
        fields["pinn_u"],
        fields["pinn_v"],
        weights,
    )
    vorticity_error = relative_scalar_error(
        fields["actual_omega"], fields["pinn_omega"], weights
    )

    mesh = triangulation(points, cx, cfg.radius)
    flat_grid = np.c_[xgrid.ravel(), ygrid.ravel()]
    nearest = cKDTree(points).query(flat_grid, k=1, workers=-1)[1]
    gridded = {
        key: grid_field(
            mesh,
            points,
            fields[key],
            xgrid,
            ygrid,
            nearest,
        )
        for key in FIELD_KEYS
    }
    gridded.update(
        {
            "alpha": np.float32(alpha),
            "phase": np.float32(phase),
            "cylinder_x": np.float32(cx),
            "velocity_error": np.float32(velocity_error),
            "vorticity_error": np.float32(vorticity_error),
            "rank": np.int32(rank),
            "modal_coefficients": modal_coefficients.astype(np.float32),
        }
    )
    return gridded


def relative_grid_errors(payload, xgrid, ygrid, cx, radius):
    mask = (xgrid - cx) ** 2 + ygrid**2 >= (1.01 * radius) ** 2
    du = payload["actual_u"] - payload["pinn_u"]
    dv = payload["actual_v"] - payload["pinn_v"]
    numerator = np.sum((du[mask] ** 2 + dv[mask] ** 2))
    denominator = np.sum(
        payload["actual_u"][mask] ** 2 + payload["actual_v"][mask] ** 2
    )
    velocity = float(np.sqrt(numerator / max(denominator, 1e-24)))
    domega = payload["actual_omega"] - payload["pinn_omega"]
    vort_num = np.sum(domega[mask] ** 2)
    vort_den = np.sum(payload["actual_omega"][mask] ** 2)
    vorticity = float(np.sqrt(vort_num / max(vort_den, 1e-24)))
    return velocity, vorticity


def vorticity_panel(ax, xgrid, ygrid, omega, psi, cx, radius, title, limit):
    ax.contourf(
        xgrid,
        ygrid,
        omega,
        levels=np.linspace(-limit, limit, 41),
        cmap="RdBu_r",
        extend="both",
        antialiased=True,
    )
    fluid = (xgrid - cx) ** 2 + ygrid**2 > (1.02 * radius) ** 2
    valid_psi = psi[fluid]
    lo, hi = np.percentile(valid_psi, [2, 98])
    if hi > lo:
        ax.contour(
            xgrid,
            ygrid,
            psi,
            levels=np.linspace(lo, hi, 22),
            colors="#8bd3ee",
            linewidths=0.48,
            alpha=0.78,
        )
    geometry(ax, cx, radius)
    ax.set_title(title, fontsize=9, pad=2)


def error_panel(ax, xgrid, ygrid, error, cx, radius, title, limit):
    ax.contourf(
        xgrid,
        ygrid,
        error,
        levels=np.linspace(0, limit, 35),
        cmap="magma",
        extend="max",
        antialiased=True,
    )
    geometry(ax, cx, radius)
    ax.set_title(title, fontsize=9, pad=2)


def render_frame(
    payload,
    xgrid,
    ygrid,
    cx,
    radius,
    alpha,
    phase,
    cycle_position,
    velocity_error,
    vorticity_error,
    vorticity_limit,
    error_limit,
    output,
):
    fig = plt.figure(figsize=(11.8, 7.4), dpi=120)
    grid = fig.add_gridspec(3, 2, width_ratios=(1, 0.023), wspace=0.035, hspace=0.22)
    axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    vorticity_panel(
        axes[0],
        xgrid,
        ygrid,
        payload["actual_omega"],
        payload["actual_psi"],
        cx,
        radius,
        "Independent Re=100 Navier--Stokes reference",
        vorticity_limit,
    )
    vorticity_panel(
        axes[1],
        xgrid,
        ygrid,
        payload["pinn_omega"],
        payload["pinn_psi"],
        cx,
        radius,
        f"SSS/PINN 96-mode reconstruction  (velocity err={100 * velocity_error:.2f}%)",
        vorticity_limit,
    )
    error = np.hypot(
        payload["pinn_u"] - payload["actual_u"],
        payload["pinn_v"] - payload["actual_v"],
    )
    error_panel(
        axes[2],
        xgrid,
        ygrid,
        error,
        cx,
        radius,
        f"Absolute velocity error  (vorticity err={100 * vorticity_error:.2f}%)",
        error_limit,
    )

    vort_map = ScalarMappable(
        norm=Normalize(-vorticity_limit, vorticity_limit), cmap="RdBu_r"
    )
    err_map = ScalarMappable(norm=Normalize(0, error_limit), cmap="magma")
    for row in (0, 1):
        cax = fig.add_subplot(grid[row, 1])
        fig.colorbar(vort_map, cax=cax, label="Vorticity")
        cax.tick_params(labelsize=7)
    cax = fig.add_subplot(grid[2, 1])
    fig.colorbar(err_map, cax=cax, label="Absolute velocity error")
    cax.tick_params(labelsize=7)

    fig.suptitle(
        rf"2D Karman wake, $\alpha={alpha:.3f}$, "
        rf"phase={phase:.3f}, cycle={cycle_position:.2f}, "
        "velocity-minimum coefficients",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(left=0.025, right=0.965, bottom=0.025, top=0.93)
    fig.savefig(output, facecolor="white", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha-samples", type=int, default=101)
    parser.add_argument("--phase-cycles", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--grid-nx", type=int, default=501)
    parser.add_argument("--grid-ny", type=int, default=201)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    basis_dir = args.output_dir / "basis"
    frames_dir = args.output_dir / "frames"
    samples_dir.mkdir(exist_ok=True)
    basis_dir.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    cfg, model, checkpoint_payload = load_model(args.checkpoint)
    actual = np.load(args.source_cache)
    x = np.linspace(0, 10, args.grid_nx)
    y = np.linspace(-2, 2, args.grid_ny)
    xgrid, ygrid = np.meshgrid(x, y)
    alphas = np.linspace(0, 1, args.alpha_samples)

    sample_paths = []
    sample_rows = []
    for index, alpha_value in enumerate(alphas):
        alpha = float(alpha_value)
        phase_position = args.phase_cycles * alpha
        phase = phase_position % 1.0
        path = samples_dir / f"sample_{index:03d}.npz"
        if not path.exists():
            atag = f"alpha_{alpha:.2f}".replace(".", "p")
            basis_path = basis_dir / f"{atag}.npz"
            if basis_path.exists():
                basis = np.load(basis_path)
                coefficients = basis["coefficients"]
                rank = int(basis["rank"])
            else:
                eigenvalues, coefficients, rank = compute_ritz_basis(
                    model, cfg, alpha, quadrature=512
                )
                np.savez_compressed(
                    basis_path,
                    alpha=alpha,
                    eigenvalues=eigenvalues,
                    coefficients=coefficients,
                    rank=rank,
                )
            print(
                f"[sample {index + 1:03d}/{len(alphas)}] "
                f"alpha={alpha:.2f}, phase={phase:.3f}",
                flush=True,
            )
            payload = build_sample(
                model,
                cfg,
                actual,
                alpha,
                phase,
                coefficients,
                rank,
                xgrid,
                ygrid,
            )
            np.savez_compressed(path, **payload)
            torch.cuda.empty_cache()
        saved = np.load(path)
        sample_paths.append(path)
        sample_rows.append(
            {
                "sample": index,
                "alpha": alpha,
                "phase": phase,
                "rank": int(saved["rank"]),
                "velocity_error": float(saved["velocity_error"]),
                "vorticity_error": float(saved["vorticity_error"]),
            }
        )

    samples = [np.load(path) for path in sample_paths]
    vorticity_limit = max(
        float(
            np.percentile(
                np.abs(
                    np.concatenate(
                        [
                            sample[key].ravel()
                            for sample in samples
                            for key in ("actual_omega", "pinn_omega")
                        ]
                    )
                ),
                99,
            )
        ),
        1e-8,
    )
    error_limit = max(
        float(
            np.percentile(
                np.concatenate(
                    [
                        np.hypot(
                            sample["pinn_u"] - sample["actual_u"],
                            sample["pinn_v"] - sample["actual_v"],
                        ).ravel()
                        for sample in samples
                    ]
                ),
                99,
            )
        ),
        1e-8,
    )

    frame_rows = []
    for frame, (alpha_value, payload) in enumerate(zip(alphas, samples)):
        alpha = float(alpha_value)
        cx = cfg.centre_left + (cfg.centre_right - cfg.centre_left) * alpha
        cycle_position = args.phase_cycles * alpha
        phase = cycle_position % 1.0
        velocity_error, vorticity_error = relative_grid_errors(
            payload, xgrid, ygrid, cx, cfg.radius
        )
        render_frame(
            payload,
            xgrid,
            ygrid,
            cx,
            cfg.radius,
            alpha,
            phase,
            cycle_position,
            velocity_error,
            vorticity_error,
            vorticity_limit,
            error_limit,
            frames_dir / f"frame_{frame:03d}.png",
        )
        frame_rows.append(
            {
                "frame": frame,
                "alpha": alpha,
                "phase": phase,
                "cycle_position": cycle_position,
                "velocity_error": velocity_error,
                "vorticity_error": vorticity_error,
                "rank": int(payload["rank"]),
            }
        )
        if frame % 10 == 0 or frame == args.alpha_samples - 1:
            print(
                f"[frame {frame + 1:03d}/{args.alpha_samples}] "
                f"alpha={alpha:.3f}, phase={phase:.3f}",
                flush=True,
            )

    movie = args.output_dir / "ns_vs_pinn_k96_continuous_alpha_flow.mp4"
    encode_mp4(frames_dir, movie, args.fps)
    with (args.output_dir / "alpha_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sample_rows[0].keys())
        writer.writeheader()
        writer.writerows(sample_rows)
    with (args.output_dir / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(frame_rows)

    summary = {
        "method": "BasisNet K=96 reconstruction with an NS reference",
        "layout": [
            "independent Re=100 NS reference",
            "PINN K=96 reconstruction",
            "absolute velocity error",
        ],
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "alpha_start": 0.0,
        "alpha_stop": 1.0,
        "alpha_samples": args.alpha_samples,
        "rendered_frames": args.alpha_samples,
        "rendered_alpha_step": 1.0 / (args.alpha_samples - 1),
        "phase_cycles": args.phase_cycles,
        "fps": args.fps,
        "duration_seconds": args.alpha_samples / args.fps,
        "minimum_rank": min(row["rank"] for row in frame_rows),
        "movie": str(movie),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
