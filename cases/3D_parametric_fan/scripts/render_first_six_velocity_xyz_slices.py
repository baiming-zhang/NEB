"""Render the first six energy-ordered velocity modes on orthogonal slices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from render_first_six_xyz_slices import (  # noqa: E402
    InferenceEngine,
    draw_slice,
    slice_geometry,
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
        "--data",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "first_six_xyz_slices"
            / "first_six_xyz_slice_data.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "first_six_velocity_xyz_slices"
        ),
    )
    return parser.parse_args()


def mode_panels(
    velocity: np.ndarray,
    mode: int,
    ix: int,
    iy: int,
    iz: int,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    geometries: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        str,
        tuple[str, str],
    ]
]:
    return [
        (
            y,
            z,
            velocity[ix, :, :, mode, 0],
            *geometries["x"],
            r"$x=0:\ u_x$",
            ("y", "z"),
        ),
        (
            x,
            z,
            velocity[:, iy, :, mode, 1],
            *geometries["y"],
            r"$y=0:\ u_y$",
            ("x", "z"),
        ),
        (
            x,
            y,
            velocity[:, :, iz, mode, 2],
            *geometries["z"],
            r"$z=0:\ u_z$",
            ("x", "y"),
        ),
    ]


def color_limit(
    panels: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            str,
            tuple[str, str],
        ]
    ],
) -> float:
    finite = np.concatenate(
        [
            np.abs(values[physical & ~obstacle])
            for _, _, values, physical, obstacle, _, _ in panels
        ]
    )
    return max(float(np.quantile(finite, 0.99)), 1.0e-8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = InferenceEngine(args.checkpoint)
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    with np.load(args.data, allow_pickle=False) as archive:
        x = archive["x"]
        y = archive["y"]
        z = archive["z"]
        rayleigh = archive["rayleigh"]
        velocity = archive["velocity"]
    ix = int(np.argmin(np.abs(x)))
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z)))
    geometries = {
        "x": slice_geometry(engine.cfg, parameters, y, z, "x"),
        "y": slice_geometry(engine.cfg, parameters, x, z, "y"),
        "z": slice_geometry(engine.cfg, parameters, x, y, "z"),
    }
    rows = [
        mode_panels(
            velocity,
            mode,
            ix,
            iy,
            iz,
            x,
            y,
            z,
            geometries,
        )
        for mode in range(6)
    ]

    for mode, panels in enumerate(rows):
        limit = color_limit(panels)
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(13.2, 4.0),
            constrained_layout=True,
            facecolor="white",
        )
        image = None
        for axis, panel in zip(axes, panels):
            first, second, values, physical, obstacle, title, labels = panel
            image = draw_slice(
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
            image,
            ax=axes,
            shrink=0.82,
            pad=0.018,
            label="signed velocity component",
        )
        figure.suptitle(
            rf"Mode {mode + 1}   $\lambda={rayleigh[mode]:.4f}$",
            color="#17365d",
            fontsize=15,
            fontweight="semibold",
        )
        figure.savefig(
            args.output_dir / f"mode_{mode + 1:02d}_velocity_xyz_slices.png",
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
    for mode, panels in enumerate(rows):
        limit = color_limit(panels)
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
                title,
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
        "First six PINN modes — orthogonal signed-velocity slices",
        color="#17365d",
        fontsize=17,
        fontweight="semibold",
    )
    figure.savefig(
        args.output_dir / "first_six_modes_velocity_xyz_slices.png",
        dpi=220,
        facecolor="white",
    )
    plt.close(figure)
    print(
        "rayleigh="
        + ",".join(f"{value:.8f}" for value in rayleigh),
        flush=True,
    )


if __name__ == "__main__":
    main()
