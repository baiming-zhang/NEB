#!/usr/bin/env python3
"""Reconstruct vortex-merger frame 0 and advance it with the de Witt ROM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
)
from eigenfluid.gen.vortex_merger2d.witt_rom import (  # noqa: E402
    ImplicitTrapezoidalWittROM,
    generalized_symmetric_eig,
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_gt(path: Path, frame_limit: int | None) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        frames = len(handle["time"]) if frame_limit is None else frame_limit
        if not 2 <= frames <= len(handle["time"]):
            raise ValueError("frame limit must retain at least two available GT frames")
        gt: dict[str, np.ndarray | float] = {
            "time": handle["time"][:frames].astype(np.float64),
            "x": handle["x"][:].astype(np.float64),
            "y": handle["y"][:].astype(np.float64),
            "u": handle["velocity_u"][:frames].astype(np.float32),
            "v": handle["velocity_v"][:frames].astype(np.float32),
            "omega": handle["vorticity"][:frames].astype(np.float32),
            "viscosity": float(handle.attrs["kinematic_viscosity"]),
        }
    shape = np.asarray(gt["u"]).shape
    if shape[1:] != (128, 128):
        raise ValueError(f"expected 128x128 GT, received {shape[1:]}")
    times = np.asarray(gt["time"])
    if not np.allclose(np.diff(times), np.diff(times)[0], rtol=1e-10, atol=1e-12):
        raise ValueError("de Witt rollout requires uniformly spaced GT frames")
    return gt


def load_model(checkpoint: Path) -> tuple[dict[str, Any], Any, HardcodedModel]:
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = metadata["config"]
    required = {
        "geometry_mode": "free_box",
        "envelope_mode": "free_identity",
        "input_encoding": "raw_xy",
        "input_features": 2,
        "basis_objective": "potential_rayleigh",
        "spatial_derivative_order": 1,
        "causal_sorting": "geometry_rayleigh",
        "reorthogonalization_passes": 2,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint protocol mismatch: {mismatches}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(config)
    return metadata, cfg, HardcodedModel(cfg, arrays)


def evaluate_raw_fields(
    model: HardcodedModel,
    cfg: Any,
    points: np.ndarray,
    point_batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(points), int(cfg.modes))
    raw_u = np.empty(shape, dtype=np.float32)
    raw_v = np.empty(shape, dtype=np.float32)
    raw_omega = np.empty(shape, dtype=np.float32)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        device_points = cp.asarray(points[start:stop], dtype=cp.float32)
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, device_points, 0.0, 0.0, False
        ):
            index = int(mode.mode_index)
            raw_u[start:stop, index] = cp.asnumpy(field.grad[:, 0, 1])
            raw_v[start:stop, index] = cp.asnumpy(-field.grad[:, 0, 0])
            raw_omega[start:stop, index] = cp.asnumpy(
                -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
            )
        print(f"[basis Jet2] points {stop}/{len(points)}", flush=True)
        del device_points
    return raw_u, raw_v, raw_omega


def ritz_velocity_basis(
    raw_u: np.ndarray,
    raw_v: np.ndarray,
    raw_omega: np.ndarray,
    area_weight: float,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    gpu_u64 = cp.asarray(raw_u, dtype=cp.float64)
    gpu_v64 = cp.asarray(raw_v, dtype=cp.float64)
    gpu_omega64 = cp.asarray(raw_omega, dtype=cp.float64)
    weight64 = cp.float64(area_weight)
    mass = cp.asnumpy(
        weight64 * (gpu_u64.T @ gpu_u64 + gpu_v64.T @ gpu_v64)
    )
    stiffness = cp.asnumpy(weight64 * (gpu_omega64.T @ gpu_omega64))
    del gpu_u64, gpu_v64, gpu_omega64
    cp.get_default_memory_pool().free_all_blocks()
    modes = raw_u.shape[1]
    eigenvalues, rotation, resolved_rank = generalized_symmetric_eig(
        stiffness, mass, modes, sample_rows=2 * len(raw_u)
    )
    gpu_rotation = cp.asarray(rotation.astype(np.float32))
    basis_u = cp.asarray(raw_u) @ gpu_rotation
    basis_v = cp.asarray(raw_v) @ gpu_rotation
    basis_omega = cp.asarray(raw_omega) @ gpu_rotation
    gram = cp.float32(area_weight) * (
        basis_u.T @ basis_u + basis_v.T @ basis_v
    )
    orthogonality = float(
        cp.max(cp.abs(gram - cp.eye(modes, dtype=cp.float32))).get()
    )
    diagnostics = {
        "resolved_raw_velocity_mass_rank": resolved_rank,
        "ritz_modes": modes,
        "velocity_orthogonality_max_abs": orthogonality,
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "seconds": time.perf_counter() - started,
    }
    return basis_u, basis_v, basis_omega, eigenvalues, rotation, diagnostics


def project_initial_velocity(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    weight: float,
) -> tuple[cp.ndarray, float]:
    gpu_u = cp.asarray(target_u.ravel(), dtype=cp.float32)
    gpu_v = cp.asarray(target_v.ravel(), dtype=cp.float32)
    coefficients = cp.float32(weight) * (
        basis_u.T @ gpu_u + basis_v.T @ gpu_v
    )
    reconstructed_u = basis_u @ coefficients
    reconstructed_v = basis_v @ coefficients
    mse = float(
        cp.mean(
            cp.concatenate(
                ((reconstructed_u - gpu_u) ** 2, (reconstructed_v - gpu_v) ** 2)
            )
        ).get()
    )
    return coefficients, mse


def rollout_coefficients(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    eigenvalues: np.ndarray,
    initial: cp.ndarray,
    times: np.ndarray,
    weight: float,
    viscosity: float,
    substeps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    integrator = ImplicitTrapezoidalWittROM(
        basis_u,
        basis_v,
        basis_omega,
        cp.asarray(eigenvalues.astype(np.float32)),
        weight,
        viscosity,
    )
    values = np.empty((len(times), len(initial)), dtype=np.float32)
    values[0] = cp.asnumpy(initial)
    state = initial
    frame_dt = float(times[1] - times[0])
    internal_dt = frame_dt / substeps
    per_frame: list[float] = []
    started = time.perf_counter()
    for frame in range(1, len(times)):
        frame_started = time.perf_counter()
        for _ in range(substeps):
            state = integrator.step(state, internal_dt)
        values[frame] = cp.asnumpy(state)
        per_frame.append(time.perf_counter() - frame_started)
        print(
            f"[de Witt {frame:02d}/{len(times) - 1}] "
            f"t={times[frame]:.6f} norm={float(cp.linalg.norm(state).get()):.6e} "
            f"wall={per_frame[-1]:.2f}s",
            flush=True,
        )
    return values, {
        "coefficient_ode": "de Witt--Lessig--Fiume vorticity-Galerkin",
        "time_integrator": "implicit trapezoidal advection with exact viscous Strang split",
        "frames": len(times),
        "output_dt": frame_dt,
        "substeps_per_output": substeps,
        "internal_dt": internal_dt,
        "newton_iterations": integrator.newton_iterations,
        "gmres_iterations": integrator.linear_iterations,
        "adaptive_step_splits": integrator.step_splits,
        "per_frame_seconds": per_frame,
        "seconds": time.perf_counter() - started,
    }


def reconstruct(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    coefficients: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gpu_coefficients = cp.asarray(coefficients.T, dtype=cp.float32)
    shape = (len(coefficients), grid_size, grid_size)
    predicted_u = cp.asnumpy(basis_u @ gpu_coefficients).T.reshape(shape)
    predicted_v = cp.asnumpy(basis_v @ gpu_coefficients).T.reshape(shape)
    predicted_omega = cp.asnumpy(basis_omega @ gpu_coefficients).T.reshape(shape)
    return predicted_u, predicted_v, predicted_omega


def mse_rows(
    times: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    predicted_u: np.ndarray,
    predicted_v: np.ndarray,
    predicted_omega: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for frame, time_value in enumerate(times):
        delta_u = predicted_u[frame] - actual_u[frame]
        delta_v = predicted_v[frame] - actual_v[frame]
        delta_omega = predicted_omega[frame] - actual_omega[frame]
        rows.append(
            {
                "frame": frame,
                "time": float(time_value),
                "velocity_mse": float(
                    np.mean(np.stack((delta_u, delta_v)) ** 2, dtype=np.float64)
                ),
                "u_mse": float(np.mean(delta_u**2, dtype=np.float64)),
                "v_mse": float(np.mean(delta_v**2, dtype=np.float64)),
                "vorticity_mse": float(np.mean(delta_omega**2, dtype=np.float64)),
            }
        )
    return rows


def aggregate_mse(rows: list[dict[str, float | int]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "definition": "plain elementwise mean squared error; no normalization or relative error",
        "examples": len(rows),
    }
    for key in ("velocity_mse", "u_mse", "v_mse", "vorticity_mse"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "first_frame": float(values[0]),
            "last_frame": float(values[-1]),
            "mean_all_frames": float(np.mean(values)),
        }
    return result


def draw_streamlines(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    omega: np.ndarray,
    norm: Normalize,
    title: str,
) -> Any:
    levels = np.linspace(norm.vmin, norm.vmax, 41)
    background = axis.contourf(
        x, y, omega, levels=levels, cmap="RdBu_r", norm=norm, extend="both", alpha=0.72
    )
    axis.streamplot(
        x,
        y,
        u,
        v,
        density=1.35,
        color="#17202A",
        linewidth=0.58,
        arrowsize=0.72,
        minlength=0.08,
        maxlength=5.0,
        integration_direction="both",
    )
    axis.set_title(title, fontsize=10.5)
    axis.set_aspect("equal")
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(float(y[0]), float(y[-1]))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    return background


def render_initial_final(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: list[dict[str, float | int]],
    modes: int,
    width: int,
) -> tuple[Path, Path]:
    au, av, ao = actual
    pu, pv, po = predicted
    scale = max(
        float(np.percentile(np.abs(np.concatenate((ao[[0, -1]], po[[0, -1]]))), 99.5)),
        1e-12,
    )
    norm = Normalize(-scale, scale)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.2), constrained_layout=True)
    images = []
    for column, frame in enumerate((0, len(times) - 1)):
        images.append(
            draw_streamlines(
                axes[0, column],
                x,
                y,
                au[frame],
                av[frame],
                ao[frame],
                norm,
                f"GT velocity streamlines · frame {frame} · t={times[frame]:.3f}",
            )
        )
        images.append(
            draw_streamlines(
                axes[1, column],
                x,
                y,
                pu[frame],
                pv[frame],
                po[frame],
                norm,
                f"de Witt ROM · velocity MSE={float(rows[frame]['velocity_mse']):.3e}",
            )
        )
    colorbar = figure.colorbar(images[0], ax=axes, shrink=0.84, pad=0.025)
    colorbar.set_label(r"signed vorticity $\omega$ (streamlines show velocity direction)")
    figure.suptitle(
        f"2D vortex merger · raw x,y · free boundary · K={modes} · width={width}",
        fontsize=14,
    )
    png = output_dir / "velocity_streamlines_first_last.png"
    pdf = output_dir / "velocity_streamlines_first_last.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def render_mse_curve(
    output_dir: Path,
    times: np.ndarray,
    rows: list[dict[str, float | int]],
    aggregate: dict[str, Any],
) -> tuple[Path, Path]:
    velocity = np.asarray([row["velocity_mse"] for row in rows], dtype=np.float64)
    vorticity = np.asarray([row["vorticity_mse"] for row in rows], dtype=np.float64)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    for axis, values, color, label, key in (
        (axes[0], velocity, "#2166AC", "velocity MSE", "velocity_mse"),
        (axes[1], vorticity, "#B2182B", "vorticity MSE", "vorticity_mse"),
    ):
        axis.plot(times, values, color=color, linewidth=2.0)
        axis.scatter(times[[0, -1]], values[[0, -1]], color=color, s=28, zorder=3)
        stats = aggregate[key]
        axis.axhline(
            stats["mean_all_frames"], color="#4D4D4D", linestyle="--", linewidth=1.1
        )
        axis.set_title(
            f"{label}\nfirst={stats['first_frame']:.3e} · "
            f"last={stats['last_frame']:.3e} · mean={stats['mean_all_frames']:.3e}"
        )
        axis.set_xlabel("time")
        axis.set_ylabel("plain MSE")
        axis.grid(True, color="#D9D9D9", linewidth=0.6)
        axis.set_facecolor("#FAFAFA")
    png = output_dir / "mse_all_gt_frames.png"
    pdf = output_dir / "mse_all_gt_frames.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def encode_video(
    output_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: list[dict[str, float | int]],
    fps: int,
) -> Path:
    frames_dir = output_dir / "velocity_streamline_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    au, av, ao = actual
    pu, pv, po = predicted
    scale = max(
        float(np.percentile(np.abs(np.concatenate((ao, po))), 99.5)), 1e-12
    )
    norm = Normalize(-scale, scale)
    for frame in range(len(times)):
        figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), constrained_layout=True)
        image = draw_streamlines(
            axes[0], x, y, au[frame], av[frame], ao[frame], norm, "GT velocity"
        )
        draw_streamlines(
            axes[1],
            x,
            y,
            pu[frame],
            pv[frame],
            po[frame],
            norm,
            f"de Witt ROM · MSE={float(rows[frame]['velocity_mse']):.3e}",
        )
        colorbar = figure.colorbar(image, ax=axes, shrink=0.83, pad=0.025)
        colorbar.set_label(r"signed vorticity $\omega$")
        figure.suptitle(f"2D vortex merger velocity streamlines · t={times[frame]:.3f}")
        figure.savefig(frames_dir / f"frame_{frame:03d}.png", dpi=150)
        plt.close(figure)
    movie = output_dir / "gt_vs_witt_velocity_streamlines.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(movie),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    return movie


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--substeps-per-output", type=int, default=10)
    parser.add_argument("--point-batch", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    gt_path = args.gt.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata, cfg, model = load_model(checkpoint)
    gt = load_gt(gt_path, args.frame_limit)
    x = np.asarray(gt["x"])
    y = np.asarray(gt["y"])
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    area = (float(cfg.xmax) - float(cfg.xmin)) * (
        float(cfg.ymax) - float(cfg.ymin)
    )
    weight = area / len(points)
    raw_u, raw_v, raw_omega = evaluate_raw_fields(
        model, cfg, points, args.point_batch
    )
    basis_u, basis_v, basis_omega, eigenvalues, rotation, ritz = ritz_velocity_basis(
        raw_u, raw_v, raw_omega, weight
    )
    del raw_u, raw_v, raw_omega, model
    initial, initial_projection_mse = project_initial_velocity(
        basis_u,
        basis_v,
        np.asarray(gt["u"])[0],
        np.asarray(gt["v"])[0],
        weight,
    )
    coefficients, rollout = rollout_coefficients(
        basis_u,
        basis_v,
        basis_omega,
        eigenvalues,
        initial,
        np.asarray(gt["time"]),
        weight,
        float(gt["viscosity"]),
        args.substeps_per_output,
    )
    predicted = reconstruct(
        basis_u, basis_v, basis_omega, coefficients, len(x)
    )
    actual = (
        np.asarray(gt["u"]),
        np.asarray(gt["v"]),
        np.asarray(gt["omega"]),
    )
    rows = mse_rows(np.asarray(gt["time"]), *actual, *predicted)
    aggregate = aggregate_mse(rows)
    metrics_csv = output_dir / "mse_all_gt_frames.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields_path = output_dir / "gt_vs_witt_fields.npz"
    np.savez_compressed(
        fields_path,
        time=np.asarray(gt["time"]),
        x=x,
        y=y,
        actual_u=actual[0],
        actual_v=actual[1],
        actual_vorticity=actual[2],
        predicted_u=predicted[0],
        predicted_v=predicted[1],
        predicted_vorticity=predicted[2],
        coefficients=coefficients,
        ritz_eigenvalues=eigenvalues,
        raw_to_ritz=rotation,
    )
    streamline_png, streamline_pdf = render_initial_final(
        output_dir,
        x,
        y,
        np.asarray(gt["time"]),
        actual,
        predicted,
        rows,
        int(cfg.modes),
        int(cfg.hidden),
    )
    mse_png, mse_pdf = render_mse_curve(
        output_dir, np.asarray(gt["time"]), rows, aggregate
    )
    movie = None
    if not args.skip_video:
        movie = encode_video(
            output_dir,
            x,
            y,
            np.asarray(gt["time"]),
            actual,
            predicted,
            rows,
            args.fps,
        )
    summary = {
        "complete": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]),
        "gt": str(gt_path),
        "gt_sha256": sha256(gt_path),
        "architecture": {
            "modes": int(cfg.modes),
            "hidden_width": int(cfg.hidden),
            "hidden_layers": int(cfg.hidden_layers),
            "total_affine_layers": int(cfg.hidden_layers) + 1,
        },
        "protocol": {
            "initial_condition": "GT frame 0 velocity projected into the learned K-mode Ritz span",
            "time_advance": "de Witt--Lessig--Fiume vorticity-Galerkin",
            "boundary": "free on four sides; no envelope and no periodic seam constraint",
            "input": "physical raw x,y only; no positional/Fourier encoding",
            "basis_derivatives": "analytic Jet2 velocity and vorticity from the Jet1-trained potential",
            "mse": aggregate["definition"],
            "velocity_visualization": "streamlines with arrows; signed vorticity background; no speed-magnitude plot",
        },
        "initial_projection_velocity_mse": initial_projection_mse,
        "ritz": ritz,
        "rollout": rollout,
        "mse": aggregate,
        "files": {
            "metrics_csv": str(metrics_csv),
            "fields_npz": str(fields_path),
            "streamlines_png": str(streamline_png),
            "streamlines_pdf": str(streamline_pdf),
            "mse_png": str(mse_png),
            "mse_pdf": str(mse_pdf),
            "video_mp4": str(movie) if movie is not None else None,
        },
        "total_seconds": time.perf_counter() - started,
        "completed_at": now(),
    }
    write_json(output_dir / "witt_rollout_summary.json", summary)
    print(json.dumps(strict_json({"mse": aggregate, "files": summary["files"]}), indent=2))


if __name__ == "__main__":
    main()
