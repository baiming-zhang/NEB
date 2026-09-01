"""Render the joint NS GT and four K=2048 de Witt ROMs as a 3x5 slice video.

Every ROM column reports the velocity and common-FD-curl vorticity relative
L2 errors for the currently displayed frame.  This is rendering-only: it does
not project or advance any field.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT_RESULT = ROOT / "results" / "joint_fd_lap_potential_physical_ns_gt_k2048_full128"
RESULT = ROOT / "results" / "joint_fd_lap_potential_physical_ns_gt30_dewitt_four_method_k2048_3d_full128"
OUTPUT = RESULT / "output" / "video_three_slices_with_errors"
FRAMES_DIR = OUTPUT / "frames_png"
PNG_DIR = RESULT / "output" / "png"
PDF_DIR = RESULT / "output" / "pdf"

GRID = 128
MID = GRID // 2
FRAMES = 30
FPS = 6
METHODS = OrderedDict(
    (
        ("strict_fd_lap_rr", "FD-Laplacian"),
        ("potential_rayleigh", "Potential"),
        ("velocity_rayleigh", "Velocity"),
        ("vorticity_rayleigh", "Vorticity"),
    )
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_metrics(method: str) -> list[dict[str, float]]:
    path = RESULT / "methods" / method / "trajectory_metrics.csv"
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != FRAMES:
        raise RuntimeError(f"{method} has {len(rows)} metric rows, expected {FRAMES}")
    return [
        {
            "frame": int(row["frame"]),
            "velocity": float(row["velocity_relative_l2"]),
            "vorticity": float(row["vorticity_relative_l2_common_fd_curl"]),
        }
        for row in rows
    ]


def full_field_from_fluid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.zeros((GRID**3, 3), dtype=np.float32)
    result[indices] = np.asarray(values, dtype=np.float32)
    return result.reshape(GRID, GRID, GRID, 3)


def slice_rows(field: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(field[MID, :, :, 0]).T,
        np.asarray(field[:, MID, :, 1]).T,
        np.asarray(field[:, :, MID, 2]).T,
    ]


def main() -> None:
    started = time.perf_counter()
    for directory in (OUTPUT, FRAMES_DIR, PNG_DIR, PDF_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    completion = json.loads((RESULT / "completion.json").read_text(encoding="utf-8"))
    if not completion.get("complete"):
        raise RuntimeError("the four-method temporal comparison is incomplete")
    indices = np.fromfile(GT_RESULT / "gt" / "fluid_indices.u32", dtype="<u4")
    if indices.size != 1_599_754:
        raise RuntimeError(f"unexpected fluid-node count: {indices.size}")

    trajectories = {
        method: np.load(
            RESULT / "methods" / method / "vorticity_trajectory_fluid_common_fd.npy",
            mmap_mode="r",
        )
        for method in METHODS
    }
    metrics = {method: load_metrics(method) for method in METHODS}

    # Match the previous three-slice presentation: one symmetric color scale
    # from the GT central slices over the entire 30-frame trajectory.
    gt_samples = []
    for frame in range(FRAMES):
        gt = np.load(
            GT_RESULT / "gt" / "vorticity_common_fd" / f"frame_{frame:03d}.npy",
            mmap_mode="r",
        )
        gt_samples.extend(np.abs(row).reshape(-1) for row in slice_rows(gt))
    common_limit = float(np.quantile(np.concatenate(gt_samples), 0.995))

    row_labels = [
        r"$x=0.5$: $\omega_x(y,z)$",
        r"$y=0.5$: $\omega_y(x,z)$",
        r"$z=0.5$: $\omega_z(x,y)$",
    ]
    timing_rows = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.7,
        }
    )

    for frame in range(FRAMES):
        frame_started = time.perf_counter()
        gt = np.load(
            GT_RESULT / "gt" / "vorticity_common_fd" / f"frame_{frame:03d}.npy",
            mmap_mode="r",
        )
        fields = [
            gt,
            *[
                full_field_from_fluid(trajectories[method][frame], indices)
                for method in METHODS
            ],
        ]
        titles = ["GT\nreference"]
        for method, label in METHODS.items():
            row = metrics[method][frame]
            titles.append(
                f"{label}\n$e_u$={100.0 * row['velocity']:.2f}%  |  "
                f"$e_\\omega$={100.0 * row['vorticity']:.2f}%"
            )

        figure, axes = plt.subplots(3, 5, figsize=(15.2, 8.8), constrained_layout=False)
        figure.subplots_adjust(
            left=0.055,
            right=0.93,
            bottom=0.07,
            top=0.865,
            wspace=0.07,
            hspace=0.15,
        )
        figure.suptitle(
            f"Physical NS GT and K=2048 de Witt ROMs | frame {frame:02d}/29 | t={frame * 0.01:.2f}",
            fontsize=15,
            fontweight="bold",
            y=0.975,
        )
        figure.text(
            0.5,
            0.93,
            "Three central normal-vorticity slices; per-frame velocity and vorticity relative L2 errors",
            ha="center",
            fontsize=9,
        )
        image = None
        for column, (title, field) in enumerate(zip(titles, fields)):
            axes[0, column].set_title(
                title,
                fontsize=8.3,
                fontweight="bold" if column == 0 else "normal",
                pad=5,
                linespacing=1.25,
            )
            for row, data in enumerate(slice_rows(field)):
                axis = axes[row, column]
                image = axis.imshow(
                    np.clip(data, -common_limit, common_limit),
                    origin="lower",
                    extent=(0, 1, 0, 1),
                    cmap="RdBu_r",
                    vmin=-common_limit,
                    vmax=common_limit,
                    interpolation="bilinear",
                )
                if column == 0:
                    axis.set_ylabel(row_labels[row], fontsize=8)
                else:
                    axis.set_yticklabels([])
                if row == 2:
                    axis.set_xlabel("normalized coordinate", fontsize=7)
                else:
                    axis.set_xticklabels([])
                axis.tick_params(labelsize=6, length=2)
                axis.set_aspect("equal")
        colorbar_axis = figure.add_axes([0.945, 0.15, 0.012, 0.66])
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("normal vorticity", fontsize=8)
        colorbar.ax.tick_params(labelsize=7)
        figure.text(
            0.055,
            0.025,
            f"Fixed symmetric range: +/-{common_limit:.3f} (GT central-slice 99.5th percentile over all 30 frames).",
            fontsize=7.5,
        )
        png = FRAMES_DIR / f"frame_{frame:03d}.png"
        figure.savefig(png, dpi=120, facecolor="white")
        if frame in (0, FRAMES - 1):
            figure.savefig(
                PNG_DIR / f"frame_{frame:03d}_five_columns_three_slices_with_errors.png",
                dpi=300,
                facecolor="white",
            )
            figure.savefig(
                PDF_DIR / f"frame_{frame:03d}_five_columns_three_slices_with_errors.pdf",
                facecolor="white",
            )
        plt.close(figure)
        seconds = time.perf_counter() - frame_started
        timing_rows.append(
            {"frame": frame, "physical_time": frame * 0.01, "render_seconds": seconds}
        )
        print(f"render annotated three-slice frame {frame + 1:02d}/{FRAMES}", flush=True)

    with (OUTPUT / "frame_render_timing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    video = OUTPUT / "joint_ns_gt_four_k2048_roms_three_slices_per_frame_errors_30frames.mp4"
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    encoded = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAMES_DIR / "frame_%03d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "17",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (OUTPUT / "ffmpeg.log").write_text(encoded.stdout + encoded.stderr, encoding="utf-8")
    if encoded.returncode or not video.exists() or video.stat().st_size == 0:
        raise RuntimeError("ffmpeg encoding failed")
    probed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,width,height,r_frame_rate,duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(probed.stdout)["streams"][0]
    encoded_frames = int(probe["nb_read_frames"])
    if encoded_frames != FRAMES:
        raise RuntimeError(f"encoded frame count is {encoded_frames}, expected {FRAMES}")
    state = {
        "complete": True,
        "video": str(video.resolve()),
        "frames_png": str(FRAMES_DIR.resolve()),
        "frame0_png": str((PNG_DIR / "frame_000_five_columns_three_slices_with_errors.png").resolve()),
        "frame0_pdf": str((PDF_DIR / "frame_000_five_columns_three_slices_with_errors.pdf").resolve()),
        "final_png": str((PNG_DIR / "frame_029_five_columns_three_slices_with_errors.png").resolve()),
        "final_pdf": str((PDF_DIR / "frame_029_five_columns_three_slices_with_errors.pdf").resolve()),
        "layout": "3 rows (x/y/z central normal-vorticity slices) x 5 columns (GT + four methods)",
        "annotations": "each ROM title shows the current frame velocity and common-FD-curl vorticity relative L2 errors",
        "common_symmetric_limit": common_limit,
        "fps": FPS,
        "encoded_frames": encoded_frames,
        "ffprobe": probe,
        "render_seconds": time.perf_counter() - started,
    }
    atomic_json(OUTPUT / "completion.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
