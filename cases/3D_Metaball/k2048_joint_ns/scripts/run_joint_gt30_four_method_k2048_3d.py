"""Project the joint physical GT at frame zero, advance four K=2048 ROMs,
and render a common five-column 3-D isovorticity animation.

The ground truth trajectory is never re-projected after frame zero.  Frames
1--29 are the already completed 128^3 incompressible Navier--Stokes DNS, while
each ROM is advanced independently with the de Witt implicit trapezoidal
vorticity--Galerkin update used by the previous K=2048 comparison.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
CACHE_RESULT = ROOT / "results" / "fem_stokes_lowpass_gt30_dewitt2012_four_method_k2048_three_slices_full128"
BASIS_RESULT = ROOT / "results" / "gt30_dewitt2012_four_method_k2048_three_slices_full128"
CORRECTED_STRICT = ROOT / "results" / "fd_lap_strict_global_ritz_A2048_k2048_full128"
BASE_SCRIPT = ROOT / "scripts" / "run_gt30_five_method_dewitt2012.py"

K = 2048
FRAMES = 30
FPS = 6


def load_module():
    spec = importlib.util.spec_from_file_location("joint_gt30_k2048_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_module()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_file(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def configure_base() -> dict:
    manifest = json.loads((BASIS_RESULT / "basis_manifest.json").read_text(encoding="utf-8"))
    manifest["methods"]["strict_fd_lap_rr"] = {
        "basis": str((CORRECTED_STRICT / "basis_chunks").resolve()),
        "gram": str((CORRECTED_STRICT / "system" / "retained_velocity_gram_unweighted.npz").resolve()),
        "construction": "2048 scalar shapes x 3 polarizations = 6144 candidates; global Ritz retains 2048",
    }
    colors = {
        "strict_fd_lap_rr": "#B40426",
        "potential_rayleigh": "#3B4CC0",
        "velocity_rayleigh": "#762A83",
        "vorticity_rayleigh": "#168B98",
    }
    labels = {
        "strict_fd_lap_rr": "FD-Laplacian",
        "potential_rayleigh": "Potential",
        "velocity_rayleigh": "Velocity",
        "vorticity_rayleigh": "Vorticity",
    }
    base.RESULT = RESULT
    base.GT_RESULT = GT_RESULT
    base.GT_SOURCE = GT_RESULT / "source"
    base.K = K
    base.METHODS = OrderedDict(
        (
            method,
            {
                "label": labels[method],
                "color": colors[method],
                "basis": Path(settings["basis"]),
                "gram": Path(settings["gram"]),
            },
        )
        for method, settings in manifest["methods"].items()
    )
    return manifest


def reuse_operator_cache() -> dict:
    """Hard-link immutable quadrature/Ritz caches from the prior identical operators."""
    linked_bytes = 0
    linked_files = 0
    sources = []
    for source in (CACHE_RESULT / "quadrature").glob("*"):
        if source.is_file():
            sources.append((source, RESULT / "quadrature" / source.name))
    for method in base.METHODS:
        old_root = CACHE_RESULT / "methods" / method
        new_root = RESULT / "methods" / method
        for source in (old_root / "quadrature").glob("*"):
            if source.is_file():
                sources.append((source, new_root / "quadrature" / source.name))
        sources.append((old_root / "generalized_ritz.npz", new_root / "generalized_ritz.npz"))
    for source, destination in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        link_file(source, destination)
        linked_bytes += source.stat().st_size
        linked_files += 1
    state = {
        "complete": True,
        "source": str(CACHE_RESULT.resolve()),
        "reason": "same four bases, geometry, 32768 quadrature points, curl operator, and generalized Ritz systems; only GT frame-0 projection changes",
        "files": linked_files,
        "logical_bytes": linked_bytes,
        "storage": "NTFS hard links when supported, copy fallback otherwise",
    }
    atomic_json(RESULT / "operator_cache_reuse.json", state)
    return state


def render_3d_video(indices: np.ndarray) -> dict:
    started = time.perf_counter()
    video_dir = RESULT / "output" / "video_3d"
    frames_dir = video_dir / "frames_png"
    pdf_dir = RESULT / "output" / "pdf"
    png_dir = RESULT / "output" / "png"
    for directory in (video_dir, frames_dir, pdf_dir, png_dir):
        directory.mkdir(parents=True, exist_ok=True)

    method_ids = list(base.METHODS)
    trajectories = {
        method: np.load(
            RESULT / "methods" / method / "vorticity_trajectory_fluid_common_fd.npy",
            mmap_mode="r",
        )
        for method in method_ids
    }
    gt_final = np.load(
        GT_RESULT / "gt" / "vorticity_common_fd" / f"frame_{FRAMES - 1:03d}.npy",
        mmap_mode="r",
    )
    level = float(np.quantile(np.linalg.norm(np.asarray(gt_final).reshape(-1, 3)[indices], axis=1), 0.965))
    labels = ["GT", *[base.METHODS[m]["label"] for m in method_ids]]
    colors = ["#303030", *[base.METHODS[m]["color"] for m in method_ids]]
    timing_rows: list[dict] = []
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    for frame in range(FRAMES):
        frame_started = time.perf_counter()
        fields = [
            np.load(
                GT_RESULT / "gt" / "vorticity_common_fd" / f"frame_{frame:03d}.npy",
                mmap_mode="r",
            ),
            *[
                base.full_field_from_fluid(trajectories[method][frame], indices)
                for method in method_ids
            ],
        ]
        figure = plt.figure(figsize=(13.2, 3.25), facecolor="white")
        figure.suptitle(
            f"Physical NS GT and K=2048 de Witt ROMs | frame {frame:02d}/29 | t={frame * base.FRAME_DT:.2f}",
            y=0.98,
            fontsize=11,
            fontweight="bold",
        )
        left, right, bottom, top, gap = 0.015, 0.985, 0.12, 0.87, 0.007
        width = (right - left - 4 * gap) / 5
        for column, (label, color, field) in enumerate(zip(labels, colors, fields)):
            axis = figure.add_axes(
                [left + column * (width + gap), bottom, width, top - bottom],
                projection="3d",
            )
            base.draw_isosurface(axis, np.asarray(field), level, color)
            axis.set_title(label, fontsize=9, pad=0, fontweight="bold" if column == 0 else "normal")
            axis.text2D(
                0.0,
                0.98,
                chr(ord("a") + column),
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
            )
        figure.text(
            0.5,
            0.035,
            rf"Common $|\omega|={level:.4f}$ (GT frame-29 q96.5); 128$^3$ common second-order FD curl. GT: NS DNS; ROMs: frame-0 projection then implicit de Witt advance.",
            ha="center",
            fontsize=7.4,
        )
        png = frames_dir / f"frame_{frame:03d}.png"
        figure.savefig(png, dpi=150, bbox_inches="tight", pad_inches=0.04)
        if frame in (0, FRAMES - 1):
            figure.savefig(
                pdf_dir / f"frame_{frame:03d}_gt_plus_four_methods_3d_isovorticity.pdf",
                bbox_inches="tight",
                pad_inches=0.04,
            )
            figure.savefig(
                png_dir / f"frame_{frame:03d}_gt_plus_four_methods_3d_isovorticity.png",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.04,
            )
        plt.close(figure)
        timing_rows.append(
            {
                "frame": frame,
                "physical_time": frame * base.FRAME_DT,
                "render_seconds": time.perf_counter() - frame_started,
            }
        )
        print(f"render 3-D five-column frame {frame + 1:02d}/{FRAMES}", flush=True)

    timing_csv = video_dir / "frame_render_timing.csv"
    with timing_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
        writer.writeheader()
        writer.writerows(timing_rows)

    video = video_dir / "joint_physical_ns_gt_plus_four_k2048_dewitt_roms_3d_30frames.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    encoded = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%03d.png"),
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
    (video_dir / "ffmpeg.log").write_text(encoded.stdout + encoded.stderr, encoding="utf-8")
    if encoded.returncode or not video.exists() or video.stat().st_size == 0:
        raise RuntimeError("ffmpeg encoding failed")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for frame verification")
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
    return {
        "complete": True,
        "video": str(video.resolve()),
        "frames_png": str(frames_dir.resolve()),
        "frame0_png": str((png_dir / "frame_000_gt_plus_four_methods_3d_isovorticity.png").resolve()),
        "frame0_pdf": str((pdf_dir / "frame_000_gt_plus_four_methods_3d_isovorticity.pdf").resolve()),
        "final_png": str((png_dir / "frame_029_gt_plus_four_methods_3d_isovorticity.png").resolve()),
        "final_pdf": str((pdf_dir / "frame_029_gt_plus_four_methods_3d_isovorticity.pdf").resolve()),
        "layout": "1 row x 5 columns: GT, FD-Laplacian, Potential, Velocity, Vorticity",
        "common_isovorticity_level": level,
        "fps": FPS,
        "encoded_frames": encoded_frames,
        "ffprobe": probe,
        "render_seconds": time.perf_counter() - started,
    }


def main() -> None:
    started = time.perf_counter()
    gt_completion = json.loads((GT_RESULT / "completion.json").read_text(encoding="utf-8"))
    if not gt_completion.get("complete"):
        raise RuntimeError("joint physical NS GT is incomplete")
    velocity_frames = sorted((GT_RESULT / "gt" / "velocity").glob("frame_*.npy"))
    vorticity_frames = sorted((GT_RESULT / "gt" / "vorticity_common_fd").glob("frame_*.npy"))
    if len(velocity_frames) != FRAMES or len(vorticity_frames) != FRAMES:
        raise RuntimeError("GT must contain exactly 30 velocity and 30 vorticity frames")

    configure_base()
    RESULT.mkdir(parents=True, exist_ok=True)
    cache = reuse_operator_cache()
    indices = base.fluid_indices()
    mask = base.fluid_mask(indices)
    gt_frame0 = np.load(GT_RESULT / "gt" / "velocity" / "frame_000.npy", mmap_mode="r")
    gt_frame0_fluid = np.asarray(gt_frame0.reshape(-1, 3)[indices], dtype=np.float32)

    method_states = {}
    all_metrics = {}
    for method in base.METHODS:
        print(f"run K=2048 frame-0 projection + implicit ROM: {method}", flush=True)
        rows, state = base.run_method(method, indices, mask, gt_frame0_fluid)
        method_states[method] = state
        all_metrics[method] = rows

    render = render_3d_video(indices)
    payload = {
        "complete": True,
        "experiment": "joint FD-Lap/Potential-informed physical NS GT with four K=2048 implicit de Witt ROMs and five-column 3-D rendering",
        "benchmark_independence": False,
        "frames": FRAMES,
        "grid": [base.GRID, base.GRID, base.GRID],
        "modes_per_method": K,
        "projection_policy": "only GT frame 0 is projected into each ROM",
        "ground_truth_evolution": "precomputed independent 128^3 incompressible NS DNS for frames 1-29",
        "rom_evolution": "exact viscous half steps around implicit trapezoidal de Witt vorticity-Galerkin advection",
        "protocol": {
            "frame_dt": base.FRAME_DT,
            "substeps_per_frame": base.SUBSTEPS_PER_FRAME,
            "internal_dt": base.DT,
            "viscosity": base.VISCOSITY,
            "quadrature_points": base.QUADRATURE_POINTS,
        },
        "gt_source": str((GT_RESULT / "completion.json").resolve()),
        "basis_manifest": str((BASIS_RESULT / "basis_manifest.json").resolve()),
        "strict_global_ritz": str((CORRECTED_STRICT / "completion.json").resolve()),
        "operator_cache": cache,
        "methods": method_states,
        "render": render,
        "observed_wall_seconds": time.perf_counter() - started,
    }
    atomic_json(RESULT / "completion.json", payload)
    print(json.dumps({"complete": True, "video": render["video"], "encoded_frames": render["encoded_frames"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
