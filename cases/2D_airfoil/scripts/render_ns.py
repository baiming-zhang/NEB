"""Render speed, vorticity, and streamlines from an independent NS cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--view-xmin", type=float)
    parser.add_argument("--view-xmax", type=float)
    parser.add_argument("--view-ymin", type=float)
    parser.add_argument("--view-ymax", type=float)
    args = parser.parse_args()
    cache = np.load(args.input)
    metadata = json.loads(str(cache["metadata"]))
    frame = args.frame % len(cache["u"])
    # Reconstruct exact uniform float64 axes. Matplotlib's streamplot checks
    # uniformity more strictly than float32 linspace coordinates can satisfy.
    x = np.linspace(
        float(cache["x"][0]), float(cache["x"][-1]), len(cache["x"])
    )
    y = np.linspace(
        float(cache["y"][0]), float(cache["y"][-1]), len(cache["y"])
    )
    u = np.ma.array(cache["u"][frame], mask=cache["solid"])
    v = np.ma.array(cache["v"][frame], mask=cache["solid"])
    omega = np.ma.array(cache["vorticity"][frame], mask=cache["solid"])
    speed = np.sqrt(u**2 + v**2)
    airfoil = cache["airfoil"].astype(bool)
    xgrid, ygrid = np.meshgrid(x, y)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13.0, 6.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    speed_limit = max(float(np.percentile(speed.compressed(), 99.5)), 1.0e-6)
    image = axes[0].pcolormesh(
        x, y, speed, shading="auto", cmap="viridis", vmin=0.0, vmax=speed_limit
    )
    stride = max(1, len(x) // 150)
    axes[0].streamplot(
        x[::stride],
        y[::stride],
        np.asarray(u)[::stride, ::stride],
        np.asarray(v)[::stride, ::stride],
        color="white",
        density=1.25,
        linewidth=0.45,
        arrowsize=0.55,
    )
    figure.colorbar(image, ax=axes[0], label=r"$|\mathbf{u}|$")
    limit = max(float(np.percentile(np.abs(omega.compressed()), 99.0)), 1.0e-6)
    image = axes[1].pcolormesh(
        x,
        y,
        omega,
        shading="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    figure.colorbar(image, ax=axes[1], label=r"$\omega_z$")
    for axis, title in zip(
        axes,
        ("Left-to-right velocity and streamlines", "Spanwise vorticity"),
    ):
        axis.contourf(
            xgrid,
            ygrid,
            airfoil.astype(float),
            levels=(0.5, 1.5),
            colors=("#20242b",),
            zorder=10,
        )
        axis.set_aspect("equal")
        axis.set_xlim(
            float(x[0]) if args.view_xmin is None else args.view_xmin,
            float(x[-1]) if args.view_xmax is None else args.view_xmax,
        )
        axis.set_ylim(
            float(y[0]) if args.view_ymin is None else args.view_ymin,
            float(y[-1]) if args.view_ymax is None else args.view_ymax,
        )
        axis.set_ylabel("y")
        axis.set_title(title)
    axes[-1].set_xlabel("x")
    figure.suptitle(
        rf"2D NACA airfoil: $\beta={metadata['beta']:.3f}$, "
        rf"$\alpha={metadata['alpha_deg']:+.1f}^\circ$, "
        rf"$Re={metadata['solver_config']['reynolds']:g}$"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
