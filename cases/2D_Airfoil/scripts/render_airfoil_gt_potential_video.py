#!/usr/bin/env python3
"""Render a GT-versus-Potential airfoil temporal comparison video.

This renderer performs no ROM inference or time integration.  It consumes a
completed ``potential_rollout_fields_stride2.npz`` archive and writes 31 PNG
frames, an H.264 MP4, and a middle-time PNG/PDF preview.
"""

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


METHODS = (("gt", "Ground truth"), ("potential", "Potential-Rayleigh"))
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "white_to_zju_blue_gt_potential",
    (WHITE, "#DCE8F5", "#7FA4CB", ZJU_BLUE),
    N=512,
)
VORTICITY_CMAP = LinearSegmentedColormap.from_list(
    "zju_blue_white_pku_red_gt_potential",
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
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--case-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=2048)
    parser.add_argument("--case-label", default="Upper airfoil")
    parser.add_argument("--source-fps", type=float, default=5.0)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--ffmpeg", type=Path, default=None)
    return parser.parse_args()


def load_and_validate(path: Path, modes: int) -> dict[str, np.ndarray]:
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
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} misses {sorted(missing)}")
        fields = {name: archive[name] for name in required}
    expected_shape = (31, 401, 501)
    for name in (
        "actual_u",
        "actual_v",
        "actual_omega",
        "predicted_u",
        "predicted_v",
        "predicted_omega",
    ):
        values = fields[name]
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError(f"{name}: expected finite {expected_shape}, got {values.shape}")
    if fields["coefficient_frames"].shape != (31, modes):
        raise ValueError(
            "coefficient_frames: expected "
            f"{(31, modes)}, got {fields['coefficient_frames'].shape}"
        )
    if fields["solid"].shape != expected_shape[1:]:
        raise ValueError(f"solid: unexpected shape {fields['solid'].shape}")
    return fields


def relative_l2(
    actual: tuple[np.ndarray, ...],
    predicted: tuple[np.ndarray, ...],
    fluid: np.ndarray,
) -> float:
    numerator = sum(
        float(np.sum((p[fluid].astype(np.float64) - a[fluid]) ** 2))
        for a, p in zip(actual, predicted, strict=True)
    )
    denominator = sum(
        float(np.sum(a[fluid].astype(np.float64) ** 2)) for a in actual
    )
    return float(np.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny)))


def render_frame(
    output: Path,
    frame: int,
    modes: int,
    case_label: str,
    coordinate_time: float,
    fields: dict[str, np.ndarray],
    airfoil: np.ndarray,
    speed_limit: float,
    omega_limit: float,
    fluid_indices: tuple[np.ndarray, ...],
    save_pdf: Path | None = None,
) -> dict[str, float]:
    x = fields["x"]
    y = fields["y"]
    solid = fields["solid"].astype(bool)
    fluid = ~solid
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    velocity_error = relative_l2(
        (fields["actual_u"][frame], fields["actual_v"][frame]),
        (fields["predicted_u"][frame], fields["predicted_v"][frame]),
        fluid,
    )
    vorticity_error = relative_l2(
        (fields["actual_omega"][frame],),
        (fields["predicted_omega"][frame],),
        fluid,
    )

    figure = plt.figure(figsize=(12.0, 6.75), dpi=160, facecolor=WHITE)
    force_white_canvas(figure)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        left=0.075,
        right=0.945,
        top=0.855,
        bottom=0.075,
        wspace=0.055,
        hspace=0.205,
    )
    figure.suptitle(
        rf"{case_label}  $\cdot$  Potential-Rayleigh  $\cdot$  "
        rf"$K={modes}$  $\cdot$  $t={coordinate_time:.1f}$",
        fontsize=18,
        fontweight="normal",
        y=0.950,
    )
    figure.text(0.015, 0.650, "Velocity\nmagnitude", ha="left", va="center", fontsize=14)
    figure.text(0.015, 0.260, "Vorticity", ha="left", va="center", fontsize=14)

    speed_image = None
    omega_image = None
    for column, (method, label) in enumerate(METHODS):
        if method == "gt":
            u = fields["actual_u"][frame]
            v = fields["actual_v"][frame]
            omega = fields["actual_omega"][frame]
            labels = ("Reference", "Reference")
        else:
            u = fields["predicted_u"][frame]
            v = fields["predicted_v"][frame]
            omega = fields["predicted_omega"][frame]
            labels = (
                rf"$\epsilon_u={100.0 * velocity_error:.2f}\%$",
                rf"$\epsilon_\omega={100.0 * vorticity_error:.2f}\%$",
            )
        values_and_limits = (
            (np.hypot(u, v), SPEED_CMAP, 0.0, speed_limit),
            (omega, VORTICITY_CMAP, -omega_limit, omega_limit),
        )
        for row, (values, cmap, vmin, vmax) in enumerate(values_and_limits):
            axis = figure.add_subplot(grid[row, column])
            image = axis.imshow(
                extend_under_airfoil(values, solid, fluid_indices),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bilinear",
                rasterized=True,
            )
            add_airfoil(axis, airfoil)
            style_panel(axis)
            if row == 0:
                axis.set_title(label, fontsize=15, fontweight="normal", pad=5.0)
            axis.set_xlabel(labels[row], fontsize=12, labelpad=3.0, fontweight="normal")
            if row == 0:
                speed_image = image
            else:
                omega_image = image

    if speed_image is None or omega_image is None:
        raise RuntimeError("frame renderer did not create both fields")
    for row, (image, label) in enumerate(
        ((speed_image, r"$|\mathbf{u}|$"), (omega_image, r"$\omega$"))
    ):
        bar_axis = figure.add_subplot(grid[row, 2])
        colorbar = figure.colorbar(image, cax=bar_axis, orientation="vertical")
        colorbar.set_label(label, fontsize=13, labelpad=5)
        colorbar.ax.tick_params(labelsize=10, length=2.5, width=0.55, pad=2)
        colorbar.outline.set_linewidth(0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=WHITE, edgecolor=WHITE)
    if save_pdf is not None:
        figure.savefig(save_pdf, facecolor=WHITE, edgecolor=WHITE)
    plt.close(figure)
    return {
        "velocity_error": velocity_error,
        "vorticity_error": vorticity_error,
    }


def main() -> None:
    args = parse_args()
    configure_typography()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    source = args.potential_fields.resolve()
    fields = load_and_validate(source, args.modes)
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
    time_per_frame = float(summary.get("coordinate_time_per_frame", 5.0))
    solid = fields["solid"].astype(bool)
    fluid = ~solid
    fluid_indices = nearest_fluid_indices(solid)
    gt_speed = np.hypot(fields["actual_u"], fields["actual_v"])
    speed_limit = float(np.ceil(100.0 * np.max(gt_speed[:, fluid])) / 100.0)
    omega_limit = float(
        np.ceil(100.0 * np.percentile(np.abs(fields["actual_omega"][:, fluid]), 99.9))
        / 100.0
    )

    preview_png = args.output_root / f"airfoil_up_gt_potential_k{args.modes}_preview.png"
    preview_pdf = args.output_root / f"airfoil_up_gt_potential_k{args.modes}_preview.pdf"
    frame_errors: list[dict[str, Any]] = []
    render_started = time.perf_counter()
    for frame in range(31):
        output = frames_dir / f"frame_{frame:03d}.png"
        errors = render_frame(
            output,
            frame,
            args.modes,
            args.case_label,
            time_per_frame * frame,
            fields,
            airfoil,
            speed_limit,
            omega_limit,
            fluid_indices,
            save_pdf=preview_pdf if frame == 15 else None,
        )
        if frame == 15:
            shutil.copy2(output, preview_png)
        frame_errors.append(
            {"frame": frame, "time": time_per_frame * frame, **errors}
        )
        print(f"[video] rendered frame {frame:02d}/30", flush=True)
    render_seconds = time.perf_counter() - render_started

    ffmpeg = str(args.ffmpeg.resolve()) if args.ffmpeg is not None else shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg executable was not found")
    video = args.output_root / f"airfoil_up_gt_potential_k{args.modes}.mp4"
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
    encode_started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    encode_seconds = time.perf_counter() - encode_started
    manifest = {
        "complete": True,
        "completed_at": now(),
        "title": "Upper-airfoil GT versus Potential-Rayleigh temporal comparison",
        "methods": [label for _method, label in METHODS],
        "modes": int(args.modes),
        "geometry": {
            "label": args.case_label,
            "beta": float(summary["beta"]),
            "alpha_deg": float(summary["alpha_deg"]),
        },
        "grid_shape": [401, 501],
        "frames": 31,
        "time_per_frame": time_per_frame,
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
        "input": {"path": str(source), "sha256": sha256(source)},
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
