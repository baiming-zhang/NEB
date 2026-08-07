"""Render the first six fan modes as an ABC-style Python MP4.

The learned basis is evaluated once at theta=0.  Every video frame then uses
the exact rotation operator around the fan axis, so theta remains a known
state rather than a learned geometry parameter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from render_first_six_isovorticity import (
    CASE_ROOT,
    InferenceEngine,
    compute_modes,
    vorticity_field,
)
from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (
    parameter_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "checkpoints"
            / "best_inference.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "visualization_first_six_python"
        ),
    )
    parser.add_argument("--movie-name", default="first-six-modes-vorticity.mp4")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--step-size", type=int, default=3)
    return parser.parse_args()


def rotate_x(points: np.ndarray, theta: float) -> np.ndarray:
    """Apply the exact active rotation around the axial x direction."""
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rotated = np.array(points, copy=True)
    rotated[..., 1] = cosine * points[..., 1] - sine * points[..., 2]
    rotated[..., 2] = sine * points[..., 1] + cosine * points[..., 2]
    return rotated


def structured_faces(nu: int, nv: int, reverse: bool = False) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            b = (i + 1) * nv + j
            c = (i + 1) * nv + j + 1
            d = i * nv + j + 1
            if reverse:
                faces.extend(((a, c, b), (a, d, c)))
            else:
                faces.extend(((a, b, c), (a, c, d)))
    return np.asarray(faces, dtype=np.int32)


def naca_section(
    x01: np.ndarray,
    thickness: float,
    camber: float,
    position: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yc = np.where(
        x01 < position,
        camber / position**2 * (2.0 * position * x01 - x01**2),
        camber
        / (1.0 - position) ** 2
        * ((1.0 - 2.0 * position) + 2.0 * position * x01 - x01**2),
    )
    slope = np.where(
        x01 < position,
        2.0 * camber / position**2 * (position - x01),
        2.0 * camber / (1.0 - position) ** 2 * (position - x01),
    )
    angle = np.arctan(slope)
    yt = 5.0 * thickness * (
        0.2969 * np.sqrt(x01)
        - 0.1260 * x01
        - 0.3516 * x01**2
        + 0.2843 * x01**3
        - 0.1036 * x01**4
    )
    return (
        x01 - yt * np.sin(angle),
        yc + yt * np.cos(angle),
        x01 + yt * np.sin(angle),
        yc - yt * np.cos(angle),
    )


def blade_surfaces(
    cfg,
    beta_tip_deg: float,
    chord_tip: float,
    thickness_ratio: float,
    azimuth: float,
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.linspace(cfg.hub_radius, cfg.tip_radius, 18)
    span = (radial - cfg.hub_radius) / (cfg.tip_radius - cfg.hub_radius)
    smooth_span = span**2 * (3.0 - 2.0 * span)
    beta = np.deg2rad(
        cfg.beta_root_deg
        + (beta_tip_deg - cfg.beta_root_deg) * span**0.82
    )
    chord = cfg.chord_root + (chord_tip - cfg.chord_root) * smooth_span
    chord *= 1.0 - 0.20 * span**10
    sweep = 0.18 * span**2 - 0.035 * span
    rake = -0.08 * span**2
    x01 = np.linspace(0.0, 1.0, 26)
    xu, yu, xl, yl = naca_section(
        x01,
        thickness_ratio,
        cfg.camber,
        cfg.camber_position,
    )
    er = np.asarray((0.0, math.cos(azimuth), math.sin(azimuth)))
    et = np.asarray((0.0, -math.sin(azimuth), math.cos(azimuth)))
    ex = np.asarray((1.0, 0.0, 0.0))
    upper = np.empty((len(radial), len(x01), 3))
    lower = np.empty_like(upper)
    for index in range(len(radial)):
        chord_direction = math.sin(beta[index]) * ex + math.cos(beta[index]) * et
        normal_direction = math.cos(beta[index]) * ex - math.sin(beta[index]) * et
        center = radial[index] * er + sweep[index] * et + rake[index] * ex
        upper[index] = (
            center
            + chord[index]
            * (xu - cfg.stacking_fraction)[:, None]
            * chord_direction
            + chord[index] * yu[:, None] * normal_direction
        )
        lower[index] = (
            center
            + chord[index]
            * (xl - cfg.stacking_fraction)[:, None]
            * chord_direction
            + chord[index] * yl[:, None] * normal_direction
        )
    return upper, lower


def rotor_mesh(cfg, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta_tip, chord_tip, thickness_ratio = parameter_values(cfg, parameters)
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for blade in range(cfg.blade_count):
        angle = 2.0 * math.pi * blade / cfg.blade_count
        upper, lower = blade_surfaces(
            cfg,
            float(beta_tip),
            float(chord_tip),
            float(thickness_ratio),
            angle,
        )
        for surface, reverse in ((upper, False), (lower, True)):
            flat = surface.reshape(-1, 3)
            local_faces = structured_faces(
                surface.shape[0],
                surface.shape[1],
                reverse=reverse,
            )
            vertices.append(flat)
            faces.append(local_faces + offset)
            offset += len(flat)

    axial = np.linspace(-0.5 * cfg.hub_length, 0.5 * cfg.hub_length, 18)
    angle = np.linspace(0.0, 2.0 * math.pi, 36)
    xx, aa = np.meshgrid(axial, angle, indexing="ij")
    hub = np.column_stack(
        (
            xx.ravel(),
            cfg.hub_radius * np.cos(aa).ravel(),
            cfg.hub_radius * np.sin(aa).ravel(),
        )
    )
    vertices.append(hub)
    faces.append(structured_faces(len(axial), len(angle)) + offset)
    return np.concatenate(vertices), np.concatenate(faces)


def extract_mode_mesh(
    omega: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    quantile: float,
    step_size: int,
) -> dict[str, np.ndarray | float]:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    radius = np.sqrt(yy * yy + zz * zz)
    interior = (radius < 1.78) & (xx > x[2]) & (xx < x[-3])
    magnitude = np.linalg.norm(omega, axis=-1)
    values = magnitude[interior & np.isfinite(magnitude)]
    level = float(np.quantile(values, quantile))
    volume = np.where(interior, magnitude, 0.0)
    spacing = np.asarray(
        (x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        dtype=np.float64,
    )
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=level,
        spacing=tuple(float(value) for value in spacing),
        step_size=step_size,
        allow_degenerate=False,
    )
    origin = np.asarray((x[0], y[0], z[0]))
    vertices += origin
    indices = np.rint((vertices - origin) / spacing).astype(np.int64)
    indices[:, 0] = np.clip(indices[:, 0], 0, len(x) - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, len(y) - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, len(z) - 1)
    axial = omega[indices[:, 0], indices[:, 1], indices[:, 2], 0]
    face_axial = np.mean(axial[faces], axis=1)
    color_limit = max(float(np.quantile(np.abs(face_axial), 0.98)), 1.0e-6)
    return {
        "vertices": vertices,
        "faces": faces.astype(np.int32),
        "face_axial_normalized": np.clip(face_axial / color_limit, -1.0, 1.0),
        "level": level,
        "color_limit": color_limit,
    }


def draw_cylinder(ax, xmin: float, xmax: float, radius: float) -> None:
    angle = np.linspace(0.0, 2.0 * math.pi, 96)
    for endpoint in (xmin, xmax):
        ax.plot(
            np.full_like(angle, endpoint),
            radius * np.cos(angle),
            radius * np.sin(angle),
            color="#666666",
            linewidth=0.65,
            alpha=0.72,
            zorder=1,
        )
    for azimuth in np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False):
        ax.plot(
            (xmin, xmax),
            (radius * math.cos(azimuth),) * 2,
            (radius * math.sin(azimuth),) * 2,
            color="#777777",
            linewidth=0.55,
            alpha=0.52,
            zorder=1,
        )


def setup_axis(ax, cfg, title: str) -> None:
    radial_limit = 1.08 * cfg.farfield_radius
    axial_padding = 0.04 * (cfg.xmax - cfg.xmin)
    ax.set_xlim(cfg.xmin - axial_padding, cfg.xmax + axial_padding)
    ax.set_ylim(-radial_limit, radial_limit)
    ax.set_zlim(-radial_limit, radial_limit)
    ax.set_box_aspect(
        ((cfg.xmax - cfg.xmin) / (2.0 * radial_limit), 1.0, 1.0),
        zoom=1.15,
    )
    ax.set_proj_type("persp", focal_length=0.95)
    ax.view_init(elev=23.0, azim=-43.0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5, pad=2)
    draw_cylinder(ax, cfg.xmin, cfg.xmax, cfg.farfield_radius)


def render_frame(
    output: Path,
    meshes: list[dict[str, np.ndarray | float]],
    rotor_vertices: np.ndarray,
    rotor_faces: np.ndarray,
    rayleigh: np.ndarray,
    cfg,
    theta: float,
    frame_index: int,
    frame_count: int,
) -> None:
    figure = plt.figure(figsize=(16.0, 9.0), dpi=120, facecolor="white")
    axes = [
        figure.add_subplot(2, 3, index + 1, projection="3d")
        for index in range(6)
    ]
    cmap = matplotlib.colormaps["RdBu_r"]
    rotated_rotor = rotate_x(rotor_vertices, theta)

    for mode_index, (ax, mesh) in enumerate(zip(axes, meshes)):
        setup_axis(
            ax,
            cfg,
            rf"Mode {mode_index + 1}   $\lambda={rayleigh[mode_index]:.3f}$",
        )
        vertices = rotate_x(np.asarray(mesh["vertices"]), theta)
        faces = np.asarray(mesh["faces"])
        colors = cmap(
            Normalize(-1.0, 1.0)(
                np.asarray(mesh["face_axial_normalized"])
            )
        )
        colors[:, 3] = 0.92
        vortex = Poly3DCollection(
            vertices[faces],
            facecolors=colors,
            edgecolors="none",
            linewidths=0.0,
            alpha=0.92,
            zorder=3,
        )
        ax.add_collection3d(vortex)
        rotor = Poly3DCollection(
            rotated_rotor[rotor_faces],
            facecolors="#8d9499",
            edgecolors=(0.30, 0.32, 0.34, 0.22),
            linewidths=0.08,
            alpha=1.0,
            zorder=4,
        )
        ax.add_collection3d(rotor)
        ax.text2D(
            0.02,
            0.01,
            (
                rf"$|\omega|_{{iso}}={float(mesh['level']):.3f}$"
                "\n"
                rf"$|\omega_x|_{{98}}={float(mesh['color_limit']):.3f}$"
            ),
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555555",
        )

    color_axis = figure.add_axes((0.927, 0.18, 0.012, 0.62))
    colorbar = figure.colorbar(
        cm.ScalarMappable(norm=Normalize(-1.0, 1.0), cmap="RdBu_r"),
        cax=color_axis,
    )
    colorbar.set_label(
        r"signed axial vorticity  $\omega_x/|\omega_x|_{98}$",
        fontsize=9,
    )
    colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        (
            "Four-blade parametric fan — first six energy-ordered modes\n"
            rf"exact rotation state  $\theta={theta:.3f}$ rad"
        ),
        fontsize=14,
        y=0.985,
    )
    beta_tip, chord_tip, thickness = parameter_values(
        cfg,
        np.asarray((0.5, 0.5, 0.5)),
    )
    figure.text(
        0.5,
        0.013,
        (
            rf"$\beta_{{tip}}={float(beta_tip):.1f}^\circ$   "
            rf"$c_{{tip}}={float(chord_tip):.2f}R$   "
            rf"$t/c={float(thickness):.2f}$   |   "
            rf"frame {frame_index + 1}/{frame_count}"
        ),
        ha="center",
        fontsize=9,
        color="#4f4f4f",
    )
    figure.subplots_adjust(
        left=0.012,
        right=0.91,
        bottom=0.045,
        top=0.91,
        wspace=-0.04,
        hspace=0.02,
    )
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def encode_mp4(frame_dir: Path, movie: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
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


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0 or args.fps <= 0:
        raise ValueError("duration and fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    started = time.perf_counter()

    engine = InferenceEngine(args.checkpoint)
    x, y, z, velocity, rayleigh = compute_modes(
        engine,
        args.nx,
        args.nr,
        args.chunk,
    )
    omega = vorticity_field(velocity, x, y, z)
    meshes = [
        extract_mode_mesh(
            omega[:, :, :, mode_index, :],
            x,
            y,
            z,
            args.quantile,
            args.step_size,
        )
        for mode_index in range(6)
    ]
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    rotor_vertices, rotor_faces = rotor_mesh(engine.cfg, parameters)

    frame_count = max(1, round(args.duration * args.fps))
    # Fourfold rotor symmetry makes [0, pi/2) a seamless physical loop.
    angles = np.linspace(0.0, 0.5 * math.pi, frame_count, endpoint=False)
    for frame_index, theta in enumerate(angles):
        render_frame(
            frame_dir / f"frame_{frame_index:03d}.png",
            meshes,
            rotor_vertices,
            rotor_faces,
            rayleigh,
            engine.cfg,
            float(theta),
            frame_index,
            frame_count,
        )
        print(
            f"[frame {frame_index + 1:02d}/{frame_count}] "
            f"theta={theta:.4f}",
            flush=True,
        )

    movie = args.output_dir / args.movie_name
    encode_mp4(frame_dir, movie, args.fps)
    summary = {
        "renderer": "Python/Matplotlib/skimage/ffmpeg",
        "reference_style": "3D_periodic_free oscillatory ABC",
        "checkpoint": str(args.checkpoint.resolve()),
        "movie": str(movie.resolve()),
        "frames": frame_count,
        "fps": args.fps,
        "duration_seconds": frame_count / args.fps,
        "rotation": "exact active rotation about x over [0, pi/2)",
        "vortex_surface": "vorticity magnitude",
        "surface_quantile": args.quantile,
        "surface_coloring": "signed axial vorticity omega_x, RdBu_r",
        "surface_alpha": 0.92,
        "rayleigh": [float(value) for value in rayleigh],
        "levels": [float(mesh["level"]) for mesh in meshes],
        "color_limits": [float(mesh["color_limit"]) for mesh in meshes],
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
