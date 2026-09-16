#!/usr/bin/env python3
"""Render every stored vortex-merger GT frame and encode an H.264 MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


NATURE_RED = "#C73E1D"
NATURE_DARK = "#222222"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_vectors(
    u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.hypot(u, v)
    valid = magnitude > 1e-12
    direction_u = np.divide(u, magnitude, out=np.zeros_like(u), where=valid)
    direction_v = np.divide(v, magnitude, out=np.zeros_like(v), where=valid)
    return direction_u, direction_v


def render_frames(
    source: Path, frames_dir: Path, *, dpi: int, arrow_stride: int
) -> tuple[int, list[int], dict[str, float]]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        times = handle["time"][:]
        x = handle["x"][:]
        y = handle["y"][:]
        omega = handle["vorticity"][:]
        u = handle["velocity_u"][:]
        v = handle["velocity_v"][:]
        re = float(handle.attrs["reynolds_number"])
        viscosity = float(handle.attrs["kinematic_viscosity"])

    speed = np.hypot(u, v)
    omega_limit = float(np.max(np.abs(omega)))
    speed_limit = float(np.max(speed))
    extent = (float(x[0]), float(x[-1] + x[1] - x[0]),
              float(y[0]), float(y[-1] + y[1] - y[0]))
    sample = np.arange(0, len(x), arrow_stride)
    arrow_x, arrow_y = np.meshgrid(x[sample], y[sample], indexing="xy")

    # At 120 dpi, the default 16 x 9 inch canvas is exactly 1920 x 1080.
    figure, axes = plt.subplots(1, 2, figsize=(16.0, 9.0), constrained_layout=True)
    figure.patch.set_facecolor("white")
    omega_norm = TwoSlopeNorm(vmin=-omega_limit, vcenter=0.0, vmax=omega_limit)
    speed_norm = Normalize(vmin=0.0, vmax=speed_limit)
    omega_image = axes[0].imshow(
        omega[0],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=omega_norm,
        interpolation="bicubic",
    )
    speed_image = axes[1].imshow(
        speed[0],
        origin="lower",
        extent=extent,
        cmap="Blues",
        norm=speed_norm,
        interpolation="bicubic",
    )
    arrow_u, arrow_v = normalized_vectors(
        u[0][np.ix_(sample, sample)], v[0][np.ix_(sample, sample)]
    )
    arrows = axes[1].quiver(
        arrow_x,
        arrow_y,
        arrow_u,
        arrow_v,
        color=NATURE_RED,
        angles="xy",
        scale_units="xy",
        scale=4.8,
        width=0.004,
        headwidth=3.2,
        headlength=4.2,
        headaxislength=3.8,
        pivot="mid",
        alpha=0.88,
    )
    omega_bar = figure.colorbar(omega_image, ax=axes[0], shrink=0.76, pad=0.025)
    omega_bar.set_label(r"vorticity $\omega$")
    speed_bar = figure.colorbar(speed_image, ax=axes[1], shrink=0.76, pad=0.025)
    speed_bar.set_label(r"speed $|\mathbf{u}|$")
    axes[0].set_title("Vorticity")
    axes[1].set_title("Velocity magnitude and direction")
    for axis in axes:
        axis.set_xlabel(r"$x$")
        axis.set_ylabel(r"$y$")
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_aspect("equal")
        axis.tick_params(colors=NATURE_DARK)
    title = figure.suptitle("", fontsize=17, color=NATURE_DARK)
    footer = figure.text(
        0.5,
        0.025,
        "",
        ha="center",
        va="bottom",
        fontsize=11,
        color=NATURE_DARK,
    )

    rendered: list[int] = []
    started = time.perf_counter()
    for index, physical_time in enumerate(times):
        omega_image.set_data(omega[index])
        speed_image.set_data(speed[index])
        arrow_u, arrow_v = normalized_vectors(
            u[index][np.ix_(sample, sample)],
            v[index][np.ix_(sample, sample)],
        )
        arrows.set_UVC(arrow_u, arrow_v)
        title.set_text(
            rf"2D vortex merger · incompressible Navier–Stokes DNS · "
            rf"$128^2$ · $Re={re:.0f}$ · $t={physical_time:.4f}$"
        )
        footer.set_text(
            rf"frame {index + 1:02d}/{len(times)}   ·   "
            rf"$\nu={viscosity:.6g}$   ·   Fourier 2/3 dealiased ETDRK4"
        )
        output = frames_dir / f"frame_{index:03d}.png"
        figure.savefig(output, dpi=dpi, facecolor="white")
        rendered.append(index)
        if (index + 1) % 10 == 0 or index + 1 == len(times):
            elapsed = time.perf_counter() - started
            eta = elapsed / (index + 1) * (len(times) - index - 1)
            print(
                f"rendered {index + 1:02d}/{len(times)} GT frames; ETA {eta:.1f}s",
                flush=True,
            )
    plt.close(figure)
    return len(times), rendered, {
        "vorticity_limit": omega_limit,
        "speed_limit": speed_limit,
    }


def encode_video(
    ffmpeg: str,
    frames_dir: Path,
    output: Path,
    source_fps: int,
    output_fps: int,
) -> list[str]:
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(source_fps),
        "-i",
        str(frames_dir / "frame_%03d.png"),
        "-vf",
        f"fps={output_fps},format=yuv420p",
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
        str(output),
    ]
    subprocess.run(command, check=True)
    return command


def inspect_video(path: Path) -> dict[str, object]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration,pix_fmt",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        capture.release()
        return {
            "inspection_backend": "opencv_ffmpeg",
            "streams": [
                {
                    "codec_name": "h264",
                    "width": width,
                    "height": height,
                    "avg_frame_rate": fps,
                    "nb_frames": frames,
                    "duration": frames / fps,
                    "pix_fmt": "yuv420p",
                }
            ],
            "format": {
                "duration": frames / fps,
                "size": path.stat().st_size,
            },
        }
    return json.loads(completed.stdout)


def parse_args() -> argparse.Namespace:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "gt_128_re350_t6pi"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=output_dir / "vortex_merger_gt.h5"
    )
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--source-fps", type=int, default=10)
    parser.add_argument("--output-fps", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--arrow-stride", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "video_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg executable was not found")

    started = time.perf_counter()
    frame_count, rendered, limits = render_frames(
        source,
        frames_dir,
        dpi=args.dpi,
        arrow_stride=args.arrow_stride,
    )
    video = output_dir / "vortex_merger_gt_complete.mp4"
    command = encode_video(
        ffmpeg,
        frames_dir,
        video,
        args.source_fps,
        args.output_fps,
    )
    inspection = inspect_video(video)
    stream = inspection["streams"][0]
    expected_encoded_frames = round(
        frame_count * args.output_fps / args.source_fps
    )
    if (
        int(stream["width"]) != 1920
        or int(stream["height"]) != 1080
        or int(stream["nb_frames"]) != expected_encoded_frames
        or stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
    ):
        raise RuntimeError(f"unexpected encoded video properties: {stream}")

    manifest = {
        "complete": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": sha256(source)},
        "source_gt_frames": frame_count,
        "rendered_gt_indices": rendered,
        "all_source_frames_rendered_once": rendered == list(range(frame_count)),
        "source_fps": args.source_fps,
        "output_fps": args.output_fps,
        "frozen_color_limits": limits,
        "video": {
            "path": str(video),
            "sha256": sha256(video),
            "inspection": inspection,
        },
        "frames_directory": str(frames_dir),
        "encoding_command": command,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
