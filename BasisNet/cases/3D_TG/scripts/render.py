"""Render periodic NS truth, PINN reconstruction, and error as 3D vortices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes


def spectral_vorticity(velocity: np.ndarray) -> np.ndarray:
    n = velocity.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    velocity_hat = np.fft.fftn(velocity, axes=(0, 1, 2))
    omega_hat = 1j * np.stack(
        (
            ky * velocity_hat[..., 2] - kz * velocity_hat[..., 1],
            kz * velocity_hat[..., 0] - kx * velocity_hat[..., 2],
            kx * velocity_hat[..., 1] - ky * velocity_hat[..., 0],
        ),
        axis=-1,
    )
    return np.fft.ifftn(omega_hat, axes=(0, 1, 2)).real


def draw_cube(ax, lower: float, upper: float) -> None:
    corners = {
        (x, y, z): np.asarray((x, y, z))
        for x in (lower, upper)
        for y in (lower, upper)
        for z in (lower, upper)
    }
    for corner, start in corners.items():
        for axis in range(3):
            if corner[axis] != lower:
                continue
            end_key = list(corner)
            end_key[axis] = upper
            end = corners[tuple(end_key)]
            ax.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color="#666666",
                linewidth=0.7,
                zorder=2,
            )


def setup_axis(ax, title: str, lower: float, upper: float) -> None:
    padding = 0.08 * (upper - lower)
    ax.set(
        xlim=(lower - padding, upper + padding),
        ylim=(lower - padding, upper + padding),
        zlim=(lower - padding, upper + padding),
        title=title,
    )
    ax.set_box_aspect((1, 1, 1), zoom=1.08)
    ax.set_proj_type("persp", focal_length=0.95)
    ax.view_init(elev=23, azim=-43)
    ax.set_axis_off()
    draw_cube(ax, lower, upper)


def colored_floor(
    ax,
    values: np.ndarray,
    coordinates: np.ndarray,
    z_value: float,
    cmap,
    norm: Normalize,
) -> None:
    xgrid, ygrid = np.meshgrid(coordinates, coordinates, indexing="ij")
    colors = cmap(norm(values))
    colors[..., 3] = 0.76
    ax.plot_surface(
        xgrid,
        ygrid,
        np.full_like(xgrid, z_value),
        facecolors=colors,
        edgecolor="none",
        shade=False,
        antialiased=False,
        rcount=len(coordinates),
        ccount=len(coordinates),
        zorder=1,
    )


def vortex_surface(
    ax,
    omega: np.ndarray,
    coordinates: np.ndarray,
    level: float,
    color_limit: float,
) -> int:
    magnitude = np.linalg.norm(omega, axis=-1)
    spacing = float(coordinates[1] - coordinates[0])
    if float(magnitude.max()) <= level:
        return 0
    vertices, faces, _, _ = marching_cubes(
        magnitude,
        level=level,
        spacing=(spacing, spacing, spacing),
        allow_degenerate=False,
    )
    vertices += coordinates[0]
    indices = np.clip(
        np.rint((vertices - coordinates[0]) / spacing).astype(int),
        0,
        len(coordinates) - 1,
    )
    vertex_color = omega[
        indices[:, 0], indices[:, 1], indices[:, 2], 2
    ]
    face_color = np.mean(vertex_color[faces], axis=1)
    cmap = matplotlib.colormaps["RdBu_r"]
    norm = Normalize(-color_limit, color_limit)
    collection = Poly3DCollection(
        vertices[faces],
        facecolors=cmap(norm(face_color)),
        edgecolors="none",
        linewidths=0,
        alpha=0.92,
        zorder=3,
    )
    ax.add_collection3d(collection)
    return len(faces)


def error_surface(
    ax,
    error: np.ndarray,
    coordinates: np.ndarray,
    level: float,
    limit: float,
) -> int:
    spacing = float(coordinates[1] - coordinates[0])
    if float(error.max()) <= level:
        return 0
    vertices, faces, _, _ = marching_cubes(
        error,
        level=level,
        spacing=(spacing, spacing, spacing),
        allow_degenerate=False,
    )
    vertices += coordinates[0]
    collection = Poly3DCollection(
        vertices[faces],
        facecolors=matplotlib.colormaps["magma"](
            Normalize(0.0, limit)(level)
        ),
        edgecolors="none",
        linewidths=0,
        alpha=0.74,
        zorder=3,
    )
    ax.add_collection3d(collection)
    return len(faces)


def render_frame(
    output: Path,
    truth: np.ndarray,
    reconstructed: np.ndarray,
    truth_omega: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    reynolds: float,
    modes: int,
    case_label: str,
    vortex_level: float,
    vorticity_limit: float,
    error_level: float,
    error_limit: float,
) -> dict[str, float | int]:
    reconstructed_omega = spectral_vorticity(reconstructed)
    error = np.linalg.norm(reconstructed - truth, axis=-1)
    relative_error = float(
        np.sqrt(
            np.sum((reconstructed - truth) ** 2)
            / np.maximum(np.sum(truth**2), 1.0e-24)
        )
    )
    lower = float(coordinates[0])
    upper = float(coordinates[-1] + coordinates[1] - coordinates[0])
    mid = len(coordinates) // 2
    vorticity_norm = Normalize(-vorticity_limit, vorticity_limit)
    error_norm = Normalize(0.0, error_limit)
    vorticity_cmap = matplotlib.colormaps["RdBu_r"]
    error_cmap = matplotlib.colormaps["magma"]

    fig = plt.figure(figsize=(18.0, 6.3), dpi=120)
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=(1, 1, 0.035, 1, 0.035),
        wspace=0.03,
    )
    truth_axis = fig.add_subplot(grid[0, 0], projection="3d")
    pinn_axis = fig.add_subplot(grid[0, 1], projection="3d")
    error_axis = fig.add_subplot(grid[0, 3], projection="3d")
    setup_axis(truth_axis, f"Re={reynolds:g} Navier–Stokes truth", lower, upper)
    setup_axis(pinn_axis, f"{modes}-mode divergence-free PINN", lower, upper)
    setup_axis(error_axis, "Absolute velocity error", lower, upper)
    colored_floor(
        truth_axis,
        truth_omega[:, :, mid, 2],
        coordinates,
        lower,
        vorticity_cmap,
        vorticity_norm,
    )
    colored_floor(
        pinn_axis,
        reconstructed_omega[:, :, mid, 2],
        coordinates,
        lower,
        vorticity_cmap,
        vorticity_norm,
    )
    colored_floor(
        error_axis,
        error[:, :, mid],
        coordinates,
        lower,
        error_cmap,
        error_norm,
    )
    truth_faces = vortex_surface(
        truth_axis,
        truth_omega,
        coordinates,
        vortex_level,
        vorticity_limit,
    )
    pinn_faces = vortex_surface(
        pinn_axis,
        reconstructed_omega,
        coordinates,
        vortex_level,
        vorticity_limit,
    )
    error_faces = error_surface(
        error_axis,
        error,
        coordinates,
        error_level,
        error_limit,
    )
    vorticity_axis = fig.add_subplot(grid[0, 2])
    fig.colorbar(
        cm.ScalarMappable(norm=vorticity_norm, cmap="RdBu_r"),
        cax=vorticity_axis,
        label=r"signed vorticity $\omega_z$",
    )
    error_color_axis = fig.add_subplot(grid[0, 4])
    fig.colorbar(
        cm.ScalarMappable(norm=error_norm, cmap="magma"),
        cax=error_color_axis,
        label=r"$\|\mathbf{u}_{PINN}-\mathbf{u}_{NS}\|_2$",
    )
    fig.suptitle(
        rf"{case_label},  $t={time_value:.2f}$,  "
        rf"relative velocity error={100.0 * relative_error:.2f}%",
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.975, bottom=0.02, top=0.91)
    fig.savefig(output, facecolor="white")
    plt.close(fig)
    return {
        "time": time_value,
        "relative_velocity_error": relative_error,
        "truth_faces": truth_faces,
        "pinn_faces": pinn_faces,
        "error_faces": error_faces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vortex-level", type=float, default=2.5)
    parser.add_argument("--vorticity-limit", type=float, default=3.5)
    parser.add_argument("--error-level", type=float, default=0.2)
    parser.add_argument("--error-limit", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--case-label", default="Periodic closed vortex")
    parser.add_argument(
        "--movie-name", default="closed_vortex_ns_vs_pinn_k256.mp4"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    rows = []
    with (
        h5py.File(args.source, "r") as source,
        h5py.File(args.reconstruction, "r") as reconstruction,
    ):
        coordinates = source["coordinates/x"][:]
        times = source["time"][:]
        reynolds = float(source.attrs["reynolds_number"])
        modes = int(reconstruction.attrs["modes"])
        for index, time_value in enumerate(times):
            row = render_frame(
                frame_dir / f"frame_{index:03d}.png",
                source["fields/velocity"][index],
                reconstruction["fields/velocity"][index],
                source["fields/vorticity"][index],
                coordinates,
                float(time_value),
                reynolds,
                modes,
                args.case_label,
                args.vortex_level,
                args.vorticity_limit,
                args.error_level,
                args.error_limit,
            )
            rows.append(row)
            print(
                f"[frame {index + 1:02d}/{len(times)}] "
                f"error={row['relative_velocity_error']:.4f}",
                flush=True,
            )
    movie = args.output_dir / args.movie_name
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    summary = {
        "source": str(args.source),
        "reconstruction": str(args.reconstruction),
        "frames": len(rows),
        "fps": args.fps,
        "playback_seconds": len(rows) / args.fps,
        "reynolds": reynolds,
        "modes": modes,
        "vortex_level": args.vortex_level,
        "vorticity_colormap": "RdBu_r",
        "error_colormap": "magma",
        "movie": str(movie),
        "metrics": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
