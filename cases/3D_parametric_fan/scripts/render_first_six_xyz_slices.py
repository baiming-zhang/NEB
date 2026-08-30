"""Render the first six energy-ordered modes on orthogonal vorticity slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from render_first_six_isovorticity import (  # noqa: E402
    InferenceEngine,
    fluid_points,
    peel_and_adjoint,
    raw_fields,
    rotor_distance,
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
            / "first_six_xyz_slices"
        ),
    )
    parser.add_argument("--nx", type=int, default=80)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=4096)
    return parser.parse_args()


def slice_geometry(
    cfg: object,
    parameters: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    plane: str,
) -> tuple[np.ndarray, np.ndarray]:
    aa, bb = np.meshgrid(first, second, indexing="ij")
    if plane == "x":
        points = np.column_stack(
            (np.zeros(aa.size), aa.ravel(), bb.ravel())
        )
        physical = aa * aa + bb * bb <= cfg.farfield_radius**2
    elif plane == "y":
        points = np.column_stack(
            (aa.ravel(), np.zeros(aa.size), bb.ravel())
        )
        physical = np.abs(bb) <= cfg.farfield_radius
    else:
        points = np.column_stack(
            (aa.ravel(), bb.ravel(), np.zeros(aa.size))
        )
        physical = np.abs(bb) <= cfg.farfield_radius
    obstacle = (
        np.asarray(rotor_distance(cfg, points, parameters)).reshape(aa.shape)
        <= 0.0
    )
    return physical, obstacle


def compute_modes_and_vorticity(
    engine: InferenceEngine,
    nx: int,
    nr: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = engine.cfg
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    quadrature = fluid_points(
        cfg,
        cfg.validation_points,
        parameters,
        cfg.seed + 9_000_000,
    )
    quadrature_raw = raw_fields(
        cfg,
        engine.model,
        quadrature,
        parameters,
    )
    peeled = peel_and_adjoint(
        cfg,
        quadrature_raw,
        len(quadrature),
        need_adjoint=False,
        need_coefficients=True,
    )
    energy_order = np.argsort(
        peeled["mode_energies"],
        kind="stable",
    )[:6]
    coefficients = peeled["coefficient_matrix"][:, energy_order]
    rayleigh = np.asarray(peeled["mode_energies"])[energy_order]

    x = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    y = np.linspace(
        -cfg.farfield_radius,
        cfg.farfield_radius,
        nr,
        dtype=np.float32,
    )
    z = y.copy()
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    physical = np.hypot(points[:, 1], points[:, 2]) <= cfg.farfield_radius
    indices = np.flatnonzero(physical)
    velocity = np.zeros((len(points), 6, 3), dtype=np.float32)
    vorticity = np.zeros_like(velocity)
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        evaluated = raw_fields(
            cfg,
            engine.model,
            points[selected],
            parameters,
        )
        mode_velocity = evaluated["velocity"] @ coefficients
        mode_gradient = evaluated["gradient"] @ coefficients
        mode_velocity = (
            cp.asnumpy(mode_velocity)
            .reshape(3, len(selected), 6)
            .transpose(1, 2, 0)
        )
        mode_gradient = (
            cp.asnumpy(mode_gradient)
            .reshape(3, 3, len(selected), 6)
            .transpose(2, 3, 0, 1)
        )
        mode_vorticity = np.empty_like(mode_velocity)
        mode_vorticity[:, :, 0] = (
            mode_gradient[:, :, 2, 1] - mode_gradient[:, :, 1, 2]
        )
        mode_vorticity[:, :, 1] = (
            mode_gradient[:, :, 0, 2] - mode_gradient[:, :, 2, 0]
        )
        mode_vorticity[:, :, 2] = (
            mode_gradient[:, :, 1, 0] - mode_gradient[:, :, 0, 1]
        )
        velocity[selected] = mode_velocity.astype(np.float32, copy=False)
        vorticity[selected] = mode_vorticity.astype(np.float32, copy=False)
    shape = (nx, nr, nr, 6, 3)
    return (
        x,
        y,
        z,
        velocity.reshape(shape),
        vorticity.reshape(shape),
        rayleigh,
    )


def draw_slice(
    axis: plt.Axes,
    first: np.ndarray,
    second: np.ndarray,
    values: np.ndarray,
    physical: np.ndarray,
    obstacle: np.ndarray,
    limit: float,
    labels: tuple[str, str],
    title: str,
) -> object:
    plotted = np.where(physical & ~obstacle, values, np.nan)
    image = axis.pcolormesh(
        first,
        second,
        plotted.T,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        rasterized=True,
    )
    axis.contourf(
        first,
        second,
        obstacle.T.astype(float),
        levels=[0.5, 1.5],
        colors=["#777777"],
        antialiased=True,
    )
    axis.contour(
        first,
        second,
        obstacle.T.astype(float),
        levels=[0.5],
        colors=["#444444"],
        linewidths=0.8,
    )
    axis.set_aspect("equal")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title, color="#17365d", fontsize=11)
    axis.tick_params(labelsize=8, colors="#364f6b")
    for spine in axis.spines.values():
        spine.set_color("#8aa4bd")
        spine.set_linewidth(0.7)
    return image


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(args.checkpoint)
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    x, y, z, velocity, omega, rayleigh = compute_modes_and_vorticity(
        engine,
        args.nx,
        args.nr,
        args.chunk,
    )
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z)))
    geometries = {
        "x": slice_geometry(engine.cfg, parameters, y, z, "x"),
        "y": slice_geometry(engine.cfg, parameters, x, z, "y"),
        "z": slice_geometry(engine.cfg, parameters, x, y, "z"),
    }

    all_rows: list[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, tuple[str, str]]]] = []
    for mode in range(6):
        panels = [
            (
                y,
                z,
                omega[ix, :, :, mode, 0],
                *geometries["x"],
                r"$x=0:\ \omega_x$",
                ("y", "z"),
            ),
            (
                x,
                z,
                omega[:, iy, :, mode, 1],
                *geometries["y"],
                r"$y=0:\ \omega_y$",
                ("x", "z"),
            ),
            (
                x,
                y,
                omega[:, :, iz, mode, 2],
                *geometries["z"],
                r"$z=0:\ \omega_z$",
                ("x", "y"),
            ),
        ]
        all_rows.append(panels)
        finite = np.concatenate(
            [
                np.abs(values[physical & ~obstacle])
                for _, _, values, physical, obstacle, _, _ in panels
            ]
        )
        limit = max(float(np.quantile(finite, 0.99)), 1.0e-8)
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(13.2, 4.0),
            constrained_layout=True,
            facecolor="white",
        )
        last_image = None
        for axis, panel in zip(axes, panels):
            first, second, values, physical, obstacle, title, labels = panel
            last_image = draw_slice(
                axis,
                first,
                second,
                values,
                physical,
                obstacle,
                limit,
                labels,
                title,
            )
        figure.colorbar(
            last_image,
            ax=axes,
            shrink=0.82,
            pad=0.018,
            label=r"signed normal vorticity",
        )
        figure.suptitle(
            rf"Mode {mode + 1}   $\lambda={rayleigh[mode]:.4f}$",
            color="#17365d",
            fontsize=15,
            fontweight="semibold",
        )
        figure.savefig(
            args.output_dir / f"mode_{mode + 1:02d}_xyz_slices.png",
            dpi=240,
            facecolor="white",
        )
        plt.close(figure)

    figure, axes = plt.subplots(
        6,
        3,
        figsize=(12.8, 21.0),
        constrained_layout=True,
        facecolor="white",
    )
    for mode, panels in enumerate(all_rows):
        finite = np.concatenate(
            [
                np.abs(values[physical & ~obstacle])
                for _, _, values, physical, obstacle, _, _ in panels
            ]
        )
        limit = max(float(np.quantile(finite, 0.99)), 1.0e-8)
        for column, panel in enumerate(panels):
            first, second, values, physical, obstacle, title, labels = panel
            image = draw_slice(
                axes[mode, column],
                first,
                second,
                values,
                physical,
                obstacle,
                limit,
                labels,
                (
                    title
                    if mode
                    else title
                ),
            )
            if column == 2:
                figure.colorbar(
                    image,
                    ax=axes[mode, :],
                    shrink=0.72,
                    pad=0.012,
                )
        axes[mode, 0].text(
            -0.34,
            0.5,
            f"Mode {mode + 1}\n"
            + rf"$\lambda={rayleigh[mode]:.3f}$",
            transform=axes[mode, 0].transAxes,
            ha="center",
            va="center",
            color="#17365d",
            fontsize=11,
            fontweight="semibold",
        )
    figure.suptitle(
        "First six PINN modes — orthogonal signed-vorticity slices",
        color="#17365d",
        fontsize=17,
        fontweight="semibold",
    )
    figure.savefig(
        args.output_dir / "first_six_modes_xyz_slices.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)
    np.savez_compressed(
        args.output_dir / "first_six_xyz_slice_data.npz",
        x=x,
        y=y,
        z=z,
        rayleigh=rayleigh,
        velocity=velocity,
        vorticity=omega,
    )
    print(
        "rayleigh="
        + ",".join(f"{value:.8f}" for value in rayleigh),
        flush=True,
    )


if __name__ == "__main__":
    main()
