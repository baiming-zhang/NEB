"""Compare a 256-mode rectangular Laplacian baseline with PINN reconstructions."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import body_numpy  # noqa: E402


METRIC_NAMES = (
    "velocity_error",
    "vorticity_error",
    "fluctuation_velocity_error",
    "fluctuation_vorticity_error",
    "wake_velocity_error",
    "wake_vorticity_error",
    "wake_fluctuation_velocity_error",
    "wake_fluctuation_vorticity_error",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def laplacian_mode_table(
    count: int, xmin: float, xmax: float, ymin: float, ymax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the lowest rectangular Dirichlet-Laplacian streamfunction modes."""
    lx, ly = xmax - xmin, ymax - ymin
    upper = max(32, int(math.ceil(math.sqrt(count) * 3)))
    candidates = []
    for m in range(1, upper + 1):
        for n in range(1, upper + 1):
            eigenvalue = (m * math.pi / lx) ** 2 + (n * math.pi / ly) ** 2
            candidates.append((eigenvalue, m, n))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = candidates[:count]
    eigenvalues = np.asarray([item[0] for item in selected], dtype=np.float64)
    m = np.asarray([item[1] for item in selected], dtype=np.float64)
    n = np.asarray([item[2] for item in selected], dtype=np.float64)
    # This scaling makes the continuous full-box velocity L2 norm equal to one.
    scales = np.sqrt(4.0 / (lx * ly * eigenvalues))
    return m, n, eigenvalues, scales


def mode_fields(
    points: np.ndarray,
    bounds: tuple[float, float, float, float],
    modes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate divergence-free velocities and vorticities from streamfunctions."""
    xmin, xmax, ymin, ymax = bounds
    lx, ly = xmax - xmin, ymax - ymin
    m, n, eigenvalues, scales = modes
    ax = np.pi * (points[:, 0:1] - xmin) / lx
    ay = np.pi * (points[:, 1:2] - ymin) / ly
    sin_x = np.sin(ax * m[None, :])
    cos_x = np.cos(ax * m[None, :])
    sin_y = np.sin(ay * n[None, :])
    cos_y = np.cos(ay * n[None, :])
    u = sin_x * cos_y * (scales * n * np.pi / ly)[None, :]
    v = -cos_x * sin_y * (scales * m * np.pi / lx)[None, :]
    omega = sin_x * sin_y * (scales * eigenvalues)[None, :]
    return u, v, omega


def deterministic_subset(count: int, target: int) -> np.ndarray:
    if count <= target:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, target, dtype=np.int64)


def velocity_relative_error(
    target_u: np.ndarray,
    target_v: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum(
        (estimate_u[mask] - target_u[mask]) ** 2
        + (estimate_v[mask] - target_v[mask]) ** 2,
        dtype=np.float64,
    )
    denominator = np.sum(
        target_u[mask] ** 2 + target_v[mask] ** 2, dtype=np.float64
    )
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def scalar_relative_error(
    target: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum((estimate[mask] - target[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(target[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def reconstruct_laplacian(
    gt_path: Path,
    source_sample: Path,
    output_sample: Path,
    cfg: SimpleNamespace,
    mode_count: int,
    projection_points: int,
    field_batch: int,
) -> dict:
    with np.load(source_sample, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}
    x = sample["x"].astype(np.float64)
    y = sample["y"].astype(np.float64)
    solid = sample["solid"].astype(bool)
    fluid = ~solid
    beta = float(sample["beta"])
    alpha = float(sample["alpha_deg"])
    frame = int(sample["frame"])
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid]))

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
    subset = deterministic_subset(len(points), projection_points)
    basis_u, basis_v, _ = mode_fields(points[subset], bounds, modes)
    target_u = actual_u[subset] - mean_u[subset]
    target_v = actual_v[subset] - mean_v[subset]
    gram = basis_u.T @ basis_u + basis_v.T @ basis_v
    rhs = basis_u.T @ target_u + basis_v.T @ target_v
    values, vectors = np.linalg.eigh(gram)
    threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
    retained = values > threshold
    coefficients = vectors[:, retained] @ (
        (vectors[:, retained].T @ rhs) / values[retained]
    )
    fitted_u = basis_u @ coefficients
    fitted_v = basis_v @ coefficients
    projection_residual = velocity_relative_error(
        target_u, target_v, fitted_u, fitted_v
    )
    del basis_u, basis_v, fitted_u, fitted_v, gram, rhs, vectors

    lap_u = np.empty(len(points), dtype=np.float32)
    lap_v = np.empty(len(points), dtype=np.float32)
    lap_omega = np.empty(len(points), dtype=np.float32)
    for start in range(0, len(points), field_batch):
        stop = min(start + field_batch, len(points))
        batch_u, batch_v, batch_omega = mode_fields(
            points[start:stop], bounds, modes
        )
        lap_u[start:stop] = (mean_u[start:stop] + batch_u @ coefficients).astype(
            np.float32
        )
        lap_v[start:stop] = (mean_v[start:stop] + batch_v @ coefficients).astype(
            np.float32
        )
        lap_omega[start:stop] = (
            mean_omega[start:stop] + batch_omega @ coefficients
        ).astype(np.float32)

    fluctuation_u = actual_u - mean_u
    fluctuation_v = actual_v - mean_v
    fluctuation_omega = actual_omega - mean_omega
    lap_fluctuation_u = lap_u - mean_u
    lap_fluctuation_v = lap_v - mean_v
    lap_fluctuation_omega = lap_omega - mean_omega
    xb, yb = body_numpy(cfg, points, alpha)
    wake = (
        (xb >= cfg.chord)
        & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords))
        & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)
    )
    metrics = {
        "velocity_error": velocity_relative_error(
            actual_u, actual_v, lap_u, lap_v
        ),
        "vorticity_error": scalar_relative_error(actual_omega, lap_omega),
        "fluctuation_velocity_error": velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            lap_fluctuation_u,
            lap_fluctuation_v,
        ),
        "fluctuation_vorticity_error": scalar_relative_error(
            fluctuation_omega, lap_fluctuation_omega
        ),
        "wake_velocity_error": velocity_relative_error(
            actual_u, actual_v, lap_u, lap_v, wake
        ),
        "wake_vorticity_error": scalar_relative_error(
            actual_omega, lap_omega, wake
        ),
        "wake_fluctuation_velocity_error": velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            lap_fluctuation_u,
            lap_fluctuation_v,
            wake,
        ),
        "wake_fluctuation_vorticity_error": scalar_relative_error(
            fluctuation_omega, lap_fluctuation_omega, wake
        ),
    }

    def grid(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(solid, dtype=np.float32)
        result[fluid] = values
        return result

    output_sample.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_sample,
        laplacian_u=grid(lap_u),
        laplacian_v=grid(lap_v),
        laplacian_omega=grid(lap_omega),
        coefficients=coefficients.astype(np.float64),
        mode_m=modes[0].astype(np.int32),
        mode_n=modes[1].astype(np.int32),
        eigenvalues=modes[2],
        projection_rank=np.int32(np.count_nonzero(retained)),
        projection_points=np.int32(len(subset)),
        projection_relative_residual=np.float64(projection_residual),
        **{name: np.float64(value) for name, value in metrics.items()},
    )
    return {
        "beta": beta,
        "alpha_deg": alpha,
        "frame": frame,
        "projection_rank": int(np.count_nonzero(retained)),
        "projection_points": int(len(subset)),
        "projection_relative_residual": projection_residual,
        **metrics,
    }


def geometry(axis, x: np.ndarray, y: np.ndarray, solid: np.ndarray) -> None:
    xgrid, ygrid = np.meshgrid(x, y)
    axis.contourf(
        xgrid,
        ygrid,
        solid.astype(float),
        levels=(0.5, 1.5),
        colors=("#24272d",),
        zorder=20,
    )
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(float(y[0]), float(y[-1]))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def velocity_panel(axis, x, y, u, v, solid, title, limit, cmap="viridis") -> None:
    speed = np.ma.array(np.hypot(u, v), mask=solid)
    axis.pcolormesh(x, y, speed, shading="auto", cmap=cmap, vmin=0.0, vmax=limit)
    stride = max(1, len(x) // 140)
    axis.streamplot(
        x[::stride],
        y[::stride],
        np.ma.array(u, mask=solid)[::stride, ::stride],
        np.ma.array(v, mask=solid)[::stride, ::stride],
        color="white",
        density=1.0,
        linewidth=0.3,
        arrowsize=0.4,
    )
    geometry(axis, x, y, solid)
    axis.set_title(title, fontsize=8, pad=2)


def vorticity_panel(
    axis, x, y, omega, solid, title, limit, absolute=False
) -> None:
    field = np.ma.array(np.abs(omega) if absolute else omega, mask=solid)
    axis.pcolormesh(
        x,
        y,
        field,
        shading="auto",
        cmap="magma" if absolute else "RdBu_r",
        vmin=0.0 if absolute else -limit,
        vmax=limit,
    )
    geometry(axis, x, y, solid)
    axis.set_title(title, fontsize=8, pad=2)


def render_frame(
    source_sample: Path, baseline_sample: Path, limits: dict, output: Path
) -> None:
    with np.load(source_sample, allow_pickle=False) as source:
        sample = {name: source[name] for name in source.files}
    with np.load(baseline_sample, allow_pickle=False) as baseline:
        lap = {name: baseline[name] for name in baseline.files}
    x, y, solid = sample["x"], sample["y"], sample["solid"].astype(bool)
    pinn_du = sample["pinn_u"] - sample["actual_u"]
    pinn_dv = sample["pinn_v"] - sample["actual_v"]
    pinn_dw = sample["pinn_omega"] - sample["actual_omega"]
    lap_du = lap["laplacian_u"] - sample["actual_u"]
    lap_dv = lap["laplacian_v"] - sample["actual_v"]
    lap_dw = lap["laplacian_omega"] - sample["actual_omega"]
    figure, axes = plt.subplots(5, 2, figsize=(15.5, 13.2), constrained_layout=True)
    velocity_panel(
        axes[0, 0],
        x,
        y,
        sample["actual_u"],
        sample["actual_v"],
        solid,
        "GT velocity magnitude + streamlines",
        limits["speed"],
    )
    vorticity_panel(
        axes[0, 1],
        x,
        y,
        sample["actual_omega"],
        solid,
        "GT vorticity",
        limits["vorticity"],
    )
    velocity_panel(
        axes[1, 0],
        x,
        y,
        sample["pinn_u"],
        sample["pinn_v"],
        solid,
        f"PINN-256 reconstruction, velocity L2={100*float(sample['velocity_error']):.2f}%",
        limits["speed"],
    )
    vorticity_panel(
        axes[1, 1],
        x,
        y,
        sample["pinn_omega"],
        solid,
        f"PINN-256 vorticity, L2={100*float(sample['vorticity_error']):.2f}%",
        limits["vorticity"],
    )
    velocity_panel(
        axes[2, 0],
        x,
        y,
        pinn_du,
        pinn_dv,
        solid,
        "PINN absolute velocity-vector error",
        limits["velocity_error"],
        cmap="magma",
    )
    vorticity_panel(
        axes[2, 1],
        x,
        y,
        pinn_dw,
        solid,
        "PINN absolute vorticity error",
        limits["vorticity_error"],
        absolute=True,
    )
    velocity_panel(
        axes[3, 0],
        x,
        y,
        lap["laplacian_u"],
        lap["laplacian_v"],
        solid,
        f"Laplacian-256 reconstruction, velocity L2={100*float(lap['velocity_error']):.2f}%",
        limits["speed"],
    )
    vorticity_panel(
        axes[3, 1],
        x,
        y,
        lap["laplacian_omega"],
        solid,
        f"Laplacian-256 vorticity, L2={100*float(lap['vorticity_error']):.2f}%",
        limits["vorticity"],
    )
    velocity_panel(
        axes[4, 0],
        x,
        y,
        lap_du,
        lap_dv,
        solid,
        "Laplacian absolute velocity-vector error",
        limits["velocity_error"],
        cmap="magma",
    )
    vorticity_panel(
        axes[4, 1],
        x,
        y,
        lap_dw,
        solid,
        "Laplacian absolute vorticity error",
        limits["vorticity_error"],
        absolute=True,
    )
    figure.suptitle(
        rf"PINN vs rectangular Laplacian baseline: "
        rf"$\beta={float(sample['beta']):.3f}$, "
        rf"$\alpha={float(sample['alpha_deg']):+.2f}^\circ$",
        fontsize=12,
    )
    figure.savefig(output, dpi=110, facecolor="white")
    plt.close(figure)


def sampled_limit(
    source_paths: list[Path], baseline_paths: list[Path]
) -> dict[str, float]:
    actual_speed, actual_omega = [], []
    velocity_errors, omega_errors = [], []
    for source_path, baseline_path in zip(source_paths, baseline_paths):
        with np.load(source_path, allow_pickle=False) as source:
            stride = max(1, source["actual_u"].size // 100_000)
            actual_speed.append(
                np.hypot(source["actual_u"], source["actual_v"]).ravel()[::stride]
            )
            actual_omega.append(np.abs(source["actual_omega"]).ravel()[::stride])
            velocity_errors.append(
                np.hypot(
                    source["pinn_u"] - source["actual_u"],
                    source["pinn_v"] - source["actual_v"],
                ).ravel()[::stride]
            )
            omega_errors.append(
                np.abs(source["pinn_omega"] - source["actual_omega"]).ravel()[
                    ::stride
                ]
            )
            with np.load(baseline_path, allow_pickle=False) as baseline:
                velocity_errors.append(
                    np.hypot(
                        baseline["laplacian_u"] - source["actual_u"],
                        baseline["laplacian_v"] - source["actual_v"],
                    ).ravel()[::stride]
                )
                omega_errors.append(
                    np.abs(
                        baseline["laplacian_omega"] - source["actual_omega"]
                    ).ravel()[::stride]
                )
    return {
        "speed": max(float(np.percentile(np.concatenate(actual_speed), 99.5)), 1e-8),
        "vorticity": max(
            float(np.percentile(np.concatenate(actual_omega), 99.0)), 1e-8
        ),
        "velocity_error": max(
            float(np.percentile(np.concatenate(velocity_errors), 99.0)), 1e-8
        ),
        "vorticity_error": max(
            float(np.percentile(np.concatenate(omega_errors), 99.0)), 1e-8
        ),
    }


def aggregate(metrics: list[dict], prefix: str = "") -> dict:
    result = {}
    for name in METRIC_NAMES:
        key = prefix + name
        values = np.asarray([float(item[key]) for item in metrics])
        maximum = int(np.argmax(values))
        result[name] = {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95.0)),
            "max": float(values[maximum]),
            "max_case": maximum,
        }
    return result


def encode_mp4(
    frames: Path,
    output: Path,
    frame_duration_seconds: float,
    output_fps: int,
) -> None:
    input_fps = 1.0 / frame_duration_seconds
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{input_fps:.12g}",
            "-i",
            str(frames / "frame_%03d.png"),
            "-vf",
            "scale=960:816:force_original_aspect_ratio=decrease,"
            "pad=960:816:(ow-iw)/2:(oh-ih)/2:color=white,"
            f"setsar=1,tpad=stop_mode=clone:stop_duration="
            f"{frame_duration_seconds:.12g},"
            f"minterpolate=fps={output_fps}:mi_mode=blend",
            "-c:v",
            "libx264",
            "-crf",
            "24",
            "-maxrate",
            "3M",
            "-bufsize",
            "6M",
            "-tune",
            "animation",
            "-profile:v",
            "baseline",
            "-level:v",
            "3.2",
            "-refs",
            "1",
            "-coder",
            "0",
            "-bf",
            "0",
            "-g",
            str(output_fps),
            "-keyint_min",
            str(output_fps),
            "-sc_threshold",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--airfoil-config", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=256)
    parser.add_argument("--projection-points", type=int, default=16384)
    parser.add_argument("--field-batch", type=int, default=32768)
    parser.add_argument("--frame-duration", type=float, default=0.3)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--limit-cases", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    source_dir = args.run_dir / "postprocess"
    output_dir = args.run_dir / "postprocess_laplacian_baseline"
    baseline_dir = output_dir / "samples"
    frame_dir = output_dir / "frames"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.airfoil_config.read_text(encoding="utf-8"))
    cfg = SimpleNamespace(**config)
    manifest = json.loads(
        (source_dir / "gt_cache_reuse_manifest.json").read_text(encoding="utf-8")
    )
    cases = manifest["cases"]
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    source_paths, baseline_paths, metrics = [], [], []
    state = {"status": "running", "stage": "reconstruction", "total_cases": len(cases)}
    write_json(output_dir / "pipeline_state.json", state)
    for local_index, case in enumerate(cases):
        index = int(case["index"])
        source_path = source_dir / "samples" / f"sample_{index:03d}.npz"
        baseline_path = baseline_dir / f"sample_{index:03d}.npz"
        if not baseline_path.exists():
            print(
                f"[laplacian {local_index + 1:03d}/{len(cases)}] "
                f"beta={float(case['beta']):.4f} "
                f"alpha={float(case['alpha_deg']):+.3f}",
                flush=True,
            )
            result = reconstruct_laplacian(
                Path(case["path"]),
                source_path,
                baseline_path,
                cfg,
                args.modes,
                args.projection_points,
                args.field_batch,
            )
        else:
            with np.load(source_path, allow_pickle=False) as source:
                result = {
                    "beta": float(source["beta"]),
                    "alpha_deg": float(source["alpha_deg"]),
                    "frame": int(source["frame"]),
                }
            with np.load(baseline_path, allow_pickle=False) as baseline:
                result.update(
                    {
                        "projection_rank": int(baseline["projection_rank"]),
                        "projection_points": int(baseline["projection_points"]),
                        "projection_relative_residual": float(
                            baseline["projection_relative_residual"]
                        ),
                        **{
                            name: float(baseline[name])
                            for name in METRIC_NAMES
                        },
                    }
                )
        with np.load(source_path, allow_pickle=False) as source:
            result.update(
                {f"pinn_{name}": float(source[name]) for name in METRIC_NAMES}
            )
        result["case"] = index
        metrics.append(result)
        source_paths.append(source_path)
        baseline_paths.append(baseline_path)
        state.update({"completed_cases": local_index + 1, "last_case": index})
        write_json(output_dir / "pipeline_state.json", state)

    limits = sampled_limit(source_paths, baseline_paths)
    write_json(output_dir / "render_limits.json", limits)
    state["stage"] = "rendering"
    write_json(output_dir / "pipeline_state.json", state)
    for local_index, (source_path, baseline_path) in enumerate(
        zip(source_paths, baseline_paths)
    ):
        index = int(metrics[local_index]["case"])
        frame_path = frame_dir / f"frame_{index:03d}.png"
        print(f"[render {local_index + 1:03d}/{len(cases)}]", flush=True)
        render_frame(source_path, baseline_path, limits, frame_path)

    movie = output_dir / "continuous_gt_pinn_laplacian256_error.mp4"
    if args.limit_cases is None:
        encode_mp4(
            frame_dir,
            movie,
            frame_duration_seconds=args.frame_duration,
            output_fps=args.output_fps,
        )
    m, n, eigenvalues, _ = laplacian_mode_table(
        args.modes, cfg.xmin, cfg.xmax, cfg.ymin, cfg.ymax
    )
    summary = {
        "status": "complete",
        "definition": {
            "name": "rectangular_dirichlet_laplacian_streamfunction",
            "mode_count": args.modes,
            "streamfunction": "sin(m*pi*(x-xmin)/Lx)*sin(n*pi*(y-ymin)/Ly)",
            "velocity": "(d psi/dy, -d psi/dx)",
            "vorticity": "-laplacian(psi)=lambda*psi",
            "ordering": "ascending Laplacian eigenvalue",
            "projection": "sampled velocity least squares on temporal fluctuations",
            "projection_points": args.projection_points,
            "mean_field": "same temporal GT mean used by the PINN evaluation",
            "outer_boundary": "constant streamfunction (zero normal perturbation)",
            "airfoil_boundary": "not enforced; solid points excluded from fit and metrics",
            "mode_indices": [
                {"m": int(mi), "n": int(ni), "eigenvalue": float(value)}
                for mi, ni, value in zip(m, n, eigenvalues)
            ],
        },
        "movie": str(movie.resolve()) if args.limit_cases is None else None,
        "fps": args.output_fps,
        "source_frame_duration_seconds": args.frame_duration,
        "transition": "linear_frame_blend",
        "layout": [
            "GT",
            "PINN-256 velocity-optimal reconstruction",
            "PINN absolute error",
            "Laplacian-256 velocity-optimal reconstruction",
            "Laplacian absolute error",
        ],
        "metrics": metrics,
        "aggregate_metrics": {
            "pinn": aggregate(metrics, prefix="pinn_"),
            "laplacian": aggregate(metrics),
        },
        "render_limits": limits,
    }
    write_json(output_dir / "laplacian_baseline_complete.json", summary)
    state.update({"status": "complete", "stage": "complete"})
    write_json(output_dir / "pipeline_state.json", state)
    print(json.dumps(summary["aggregate_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
