#!/usr/bin/env python3
"""Render a reproducible five-method airfoil temporal comparison video."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from plot_airfoil_reconstruction_vector import (
    PKU_RED,
    WHITE,
    ZJU_BLUE,
    add_airfoil,
    airfoil_vertices,
    configure_typography,
    extend_under_airfoil,
    force_white_canvas,
    nearest_fluid_indices,
    style_panel,
)


METHODS = (
    ("gt", "Ground truth"),
    ("potential", "Potential-Rayleigh"),
    ("vorticity", "Vorticity-Rayleigh"),
    ("velocity", "Velocity-Rayleigh"),
    ("fd_laplacian", "FD-Laplacian"),
)
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "white_to_zju_blue",
    (WHITE, "#DCE8F5", "#7FA4CB", ZJU_BLUE),
    N=512,
)
VORTICITY_CMAP = LinearSegmentedColormap.from_list(
    "zju_blue_white_pku_red_video",
    (ZJU_BLUE, "#7397C4", WHITE, "#CA7F88", PKU_RED),
    N=512,
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--potential-fields", type=Path, required=True)
    parser.add_argument("--vorticity-fields", type=Path, required=True)
    parser.add_argument("--velocity-fields", type=Path, required=True)
    parser.add_argument("--fd-fields", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--case-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=1024)
    parser.add_argument("--source-fps", type=float, default=5.0)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    return parser.parse_args()


def load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "x",
            "y",
            "solid",
            "actual_u",
            "actual_v",
            "actual_omega",
            "predicted_u",
            "predicted_v",
            "predicted_omega",
            "coefficient_frames",
        }
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} misses {sorted(missing)}")
        return {name: archive[name] for name in required}


def relative_l2(
    actual: tuple[np.ndarray, ...], predicted: tuple[np.ndarray, ...], fluid: np.ndarray
) -> float:
    numerator = sum(
        float(np.sum((p[fluid].astype(np.float64) - a[fluid]) ** 2))
        for a, p in zip(actual, predicted, strict=True)
    )
    denominator = sum(
        float(np.sum(a[fluid].astype(np.float64) ** 2)) for a in actual
    )
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny)))


def validate_and_collect(
    inputs: dict[str, Path], modes: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    archives = {method: load_fields(path) for method, path in inputs.items()}
    reference = archives["potential"]
    expected_shape = (31, 401, 501)
    for method, archive in archives.items():
        if archive["actual_u"].shape != expected_shape:
            raise ValueError(f"{method}: invalid temporal shape {archive['actual_u'].shape}")
        for key in ("x", "y", "solid", "actual_u", "actual_v", "actual_omega"):
            if not np.array_equal(archive[key], reference[key]):
                raise ValueError(f"{method}: shared array {key} differs")
        for key in ("predicted_u", "predicted_v", "predicted_omega"):
            if archive[key].shape != expected_shape or not np.isfinite(archive[key]).all():
                raise ValueError(f"{method}: invalid or non-finite {key}")
        if archive["coefficient_frames"].shape != (31, modes):
            raise ValueError(
                f"{method}: coefficient shape {archive['coefficient_frames'].shape} "
                f"does not match K={modes}"
            )
    actual = {
        "u": reference["actual_u"],
        "v": reference["actual_v"],
        "omega": reference["actual_omega"],
        "x": reference["x"],
        "y": reference["y"],
        "solid": reference["solid"].astype(bool),
    }
    predicted = {
        method: {
            "u": archive["predicted_u"],
            "v": archive["predicted_v"],
            "omega": archive["predicted_omega"],
        }
        for method, archive in archives.items()
    }
    return actual, predicted


def render_frame(
    output: Path,
    frame: int,
    modes: int,
    actual: dict[str, np.ndarray],
    predicted: dict[str, dict[str, np.ndarray]],
    airfoil: np.ndarray,
    speed_limit: float,
    omega_limit: float,
    save_pdf: Path | None = None,
) -> dict[str, dict[str, float]]:
    x, y = actual["x"], actual["y"]
    solid = actual["solid"]
    fluid = ~solid
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    fluid_indices = nearest_fluid_indices(solid)
    figure = plt.figure(figsize=(16.0, 6.75), dpi=160, facecolor=WHITE)
    force_white_canvas(figure)
    grid = figure.add_gridspec(
        2,
        6,
        width_ratios=(1, 1, 1, 1, 1, 0.045),
        left=0.050,
        right=0.975,
        top=0.865,
        bottom=0.075,
        wspace=0.055,
        hspace=0.205,
    )
    figure.suptitle(
        rf"Upper airfoil  $\cdot$  $K={modes}$  $\cdot$  frame {frame:02d}/30  "
        rf"$\cdot$  $t={5.0 * frame:.1f}$",
        fontsize=18,
        fontweight="normal",
        y=0.955,
    )
    figure.text(0.012, 0.655, "Velocity\nmagnitude", ha="left", va="center", fontsize=14)
    figure.text(0.012, 0.265, "Vorticity", ha="left", va="center", fontsize=14)

    frame_errors: dict[str, dict[str, float]] = {}
    speed_image = None
    omega_image = None
    for column, (method, label) in enumerate(METHODS):
        if method == "gt":
            u = actual["u"][frame]
            v = actual["v"][frame]
            omega = actual["omega"][frame]
            velocity_error = 0.0
            omega_error = 0.0
        else:
            source = predicted[method]
            u, v, omega = source["u"][frame], source["v"][frame], source["omega"][frame]
            velocity_error = relative_l2(
                (actual["u"][frame], actual["v"][frame]), (u, v), fluid
            )
            omega_error = relative_l2((actual["omega"][frame],), (omega,), fluid)
        frame_errors[method] = {
            "velocity_error": velocity_error,
            "vorticity_error": omega_error,
        }
        fields = ((np.hypot(u, v), "speed"), (omega, "omega"))
        for row, (values, field) in enumerate(fields):
            axis = figure.add_subplot(grid[row, column])
            if field == "speed":
                image = axis.imshow(
                    extend_under_airfoil(values, solid, fluid_indices),
                    origin="lower",
                    extent=extent,
                    cmap=SPEED_CMAP,
                    vmin=0.0,
                    vmax=speed_limit,
                    interpolation="bilinear",
                    rasterized=True,
                )
                speed_image = image
                error = velocity_error
            else:
                image = axis.imshow(
                    extend_under_airfoil(values, solid, fluid_indices),
                    origin="lower",
                    extent=extent,
                    cmap=VORTICITY_CMAP,
                    vmin=-omega_limit,
                    vmax=omega_limit,
                    interpolation="bilinear",
                    rasterized=True,
                )
                omega_image = image
                error = omega_error
            add_airfoil(axis, airfoil)
            style_panel(axis)
            if row == 0:
                axis.set_title(label, fontsize=14, fontweight="normal", pad=5.0)
            axis.set_xlabel(
                "Reference" if method == "gt" else rf"$\epsilon={100.0 * error:.2f}\%$",
                fontsize=11.5,
                labelpad=3.0,
                fontweight="normal",
            )

    if speed_image is None or omega_image is None:
        raise RuntimeError("frame renderer did not create both fields")
    speed_bar = figure.add_subplot(grid[0, 5])
    omega_bar = figure.add_subplot(grid[1, 5])
    speed_cbar = figure.colorbar(speed_image, cax=speed_bar, orientation="vertical")
    omega_cbar = figure.colorbar(omega_image, cax=omega_bar, orientation="vertical")
    speed_cbar.set_label(r"$|\mathbf{u}|$", fontsize=13, labelpad=5)
    omega_cbar.set_label(r"$\omega$", fontsize=13, labelpad=5)
    for colorbar in (speed_cbar, omega_cbar):
        colorbar.ax.tick_params(labelsize=10, length=2.5, width=0.55, pad=2)
        colorbar.outline.set_linewidth(0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=WHITE, edgecolor=WHITE)
    if save_pdf is not None:
        figure.savefig(save_pdf, facecolor=WHITE, edgecolor=WHITE)
    plt.close(figure)
    return frame_errors


def main() -> None:
    args = parse_args()
    configure_typography()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "potential": args.potential_fields.resolve(),
        "vorticity": args.vorticity_fields.resolve(),
        "velocity": args.velocity_fields.resolve(),
        "fd_laplacian": args.fd_fields.resolve(),
    }
    started = time.perf_counter()
    actual, predicted = validate_and_collect(inputs, args.modes)
    summary = json.loads(args.case_summary.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")
    )
    geometry = evaluation_manifest["configs"]["k64"]
    airfoil = airfoil_vertices(
        geometry,
        float(summary["beta"]),
        float(summary["alpha_deg"]),
        samples=2048,
    )
    fluid = ~actual["solid"]
    gt_speed = np.hypot(actual["u"], actual["v"])
    speed_limit = float(np.ceil(100.0 * np.max(gt_speed[:, fluid])) / 100.0)
    omega_limit = float(
        np.ceil(100.0 * np.percentile(np.abs(actual["omega"][:, fluid]), 99.9))
        / 100.0
    )
    frame_errors: list[dict[str, Any]] = []
    render_started = time.perf_counter()
    preview_png = args.output_root / "airfoil_up_five_method_preview.png"
    preview_pdf = args.output_root / "airfoil_up_five_method_preview.pdf"
    for frame in range(31):
        output = frames_dir / f"frame_{frame:03d}.png"
        errors = render_frame(
            output,
            frame,
            args.modes,
            actual,
            predicted,
            airfoil,
            speed_limit,
            omega_limit,
            save_pdf=preview_pdf if frame == 15 else None,
        )
        if frame == 15:
            shutil.copy2(output, preview_png)
        frame_errors.append({"frame": frame, "time": 5.0 * frame, "methods": errors})
        print(f"[video] rendered frame {frame:02d}/30", flush=True)
    render_seconds = time.perf_counter() - render_started

    ffmpeg = (
        str(args.ffmpeg.resolve())
        if args.ffmpeg is not None
        else shutil.which("ffmpeg")
    )
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg executable was not found")
    video = args.output_root / f"airfoil_up_gt_potential_vorticity_velocity_fd_k{args.modes}.mp4"
    encode_started = time.perf_counter()
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.source_fps),
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-vf",
        f"fps={args.output_fps},format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        str(video),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    encode_seconds = time.perf_counter() - encode_started
    manifest = {
        "complete": True,
        "completed_at": now(),
        "title": "Upper-airfoil temporal ROM comparison",
        "methods": [label for _method, label in METHODS],
        "modes": int(args.modes),
        "rayleigh_definition": "||grad(F_k)||_h^2 / ||F_k||_h^2",
        "rayleigh_observables": {
            "Potential-Rayleigh": "F_k=A_k",
            "Velocity-Rayleigh": "F_k=u_k",
            "Vorticity-Rayleigh": "F_k=omega_k",
        },
        "geometry": {
            "label": "Upper",
            "beta": float(summary["beta"]),
            "alpha_deg": float(summary["alpha_deg"]),
        },
        "grid_shape": [401, 501],
        "frames": 31,
        "frame_indices": [0, 30],
        "time_per_frame": 5.0,
        "no_temporal_interpolation": True,
        "source_fps": float(args.source_fps),
        "output_fps": int(args.output_fps),
        "color_limits": {
            "velocity_magnitude": [0.0, speed_limit],
            "vorticity": [-omega_limit, omega_limit],
            "source": "frozen once from all 31 GT frames",
        },
        "error_definition": (
            "per-frame relative discrete L2 on the shared fluid mask; velocity "
            "uses both components and vorticity uses the scalar field"
        ),
        "inputs": {
            method: {"path": str(path), "sha256": sha256(path)}
            for method, path in inputs.items()
        },
        "frame_errors": frame_errors,
        "outputs": {
            "mp4": str(video.resolve()),
            "mp4_sha256": sha256(video),
            "preview_png": str(preview_png.resolve()),
            "preview_pdf": str(preview_pdf.resolve()),
            "frames": str(frames_dir.resolve()),
        },
        "encoding_command": command,
        "ffmpeg_stderr_tail": completed.stderr[-4000:],
        "timings": {
            "render_seconds": render_seconds,
            "encode_seconds": encode_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    manifest_path = args.output_root / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["outputs"], indent=2), flush=True)


if __name__ == "__main__":
    main()
