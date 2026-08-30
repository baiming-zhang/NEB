#!/usr/bin/env python3
"""Render publication-ready airfoil reconstruction panels from saved arrays.

The script performs no reconstruction or projection.  It reads the completed
``fields_stride2.npz`` archives and writes direct Matplotlib PDF figures whose
typography and annotations remain vector objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.ndimage import distance_transform_edt


K_VALUES = (64, 128, 256, 512, 1024, 2048)
CASES = (("up", "Upper"), ("middle", "Middle"), ("down", "Lower"))
WHITE = "#FFFFFF"
CHARCOAL = "#202020"
ZJU_BLUE = "#003F88"
PKU_RED = "#8B0012"
POTENTIAL_COLOR = ZJU_BLUE
LAPLACIAN_COLOR = PKU_RED
VORTICITY_CMAP = LinearSegmentedColormap.from_list(
    "zju_blue_white_pku_red",
    (
        (0.00, ZJU_BLUE),
        (0.38, "#7397C4"),
        (0.47, WHITE),
        (0.53, WHITE),
        (0.62, "#CA7F88"),
        (1.00, PKU_RED),
    ),
    N=512,
)


def configure_typography() -> None:
    font_path = Path(r"C:\Windows\Fonts\times.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def force_white_canvas(figure: plt.Figure) -> None:
    """Paint an explicit opaque white PDF/PNG canvas."""
    figure.patch.set_facecolor(WHITE)
    figure.patch.set_alpha(1.0)
    figure.add_artist(
        Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=figure.transFigure,
            facecolor=WHITE,
            edgecolor="none",
            linewidth=0.0,
            zorder=-10_000,
            clip_on=False,
        )
    )


def airfoil_vertices(
    geometry: dict[str, float], beta: float, alpha_deg: float, samples: int = 1440
) -> np.ndarray:
    """Return the analytic ellipse boundary in physical coordinates.

    The evaluation mask is sampled on the CFD grid.  Plotting that Boolean
    array produces a staircase.  This path instead uses the exact continuous
    geometry used during training and evaluation, rotated about quarter chord.
    """
    chord = float(geometry["chord"])
    quarter_x = float(geometry["quarter_chord_x"])
    quarter_y = float(geometry["quarter_chord_y"])
    thickness = float(geometry["thickness_min"]) + (
        float(geometry["thickness_max"]) - float(geometry["thickness_min"])
    ) * beta
    theta = np.linspace(0.0, 2.0 * np.pi, samples + 1, dtype=np.float64)
    xb = 0.5 * chord + 0.5 * chord * np.cos(theta)
    yb = 0.5 * thickness * chord * np.sin(theta)

    # Matches body_coordinates_numpy: positive pitch is clockwise in the plot.
    angle = np.deg2rad(-alpha_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    body_dx = xb - 0.25 * chord
    dx = cosine * body_dx - sine * yb
    dy = sine * body_dx + cosine * yb
    return np.column_stack((quarter_x + dx, quarter_y + dy))


def add_airfoil(axis: plt.Axes, vertices: np.ndarray) -> None:
    axis.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=CHARCOAL,
            edgecolor="black",
            linewidth=0.50,
            antialiased=True,
            joinstyle="round",
            rasterized=False,
            zorder=20,
        )
    )


def nearest_fluid_indices(solid: np.ndarray) -> tuple[np.ndarray, ...]:
    """Indices used only to extend raster colors underneath the vector body."""
    indices = distance_transform_edt(
        solid, return_distances=False, return_indices=True
    )
    return tuple(indices)


def extend_under_airfoil(
    values: np.ndarray,
    solid: np.ndarray,
    fluid_indices: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Remove the raster-mask halo without changing any visible fluid sample."""
    display = np.array(values, copy=True)
    nearest = values[fluid_indices]
    display[solid] = nearest[solid]
    return display


def style_panel(axis: plt.Axes) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    axis.set_facecolor(WHITE)
    axis.patch.set_alpha(1.0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.45)


def field_values(archive: np.lib.npyio.NpzFile, method: str, modes: int | None, field: str) -> np.ndarray:
    if method == "reference":
        u = archive["actual_u"][0]
        v = archive["actual_v"][0]
        omega = archive["actual_omega"][0]
    else:
        prefix = f"{method}_k{modes}"
        u = archive[f"{prefix}_u"][0]
        v = archive[f"{prefix}_v"][0]
        omega = archive[f"{prefix}_omega"][0]
    return np.hypot(u, v) if field == "speed" else omega


def color_limits(archive: np.lib.npyio.NpzFile, solid: np.ndarray, field: str) -> tuple[float, float]:
    arrays: list[np.ndarray] = [field_values(archive, "reference", None, field)]
    for method in ("learned", "laplacian"):
        arrays.extend(field_values(archive, method, modes, field) for modes in K_VALUES)
    if field == "vorticity":
        limit = max(float(np.percentile(np.abs(values[~solid]), 99.4)) for values in arrays)
        return -limit, limit
    return (
        min(float(np.percentile(values[~solid], 0.5)) for values in arrays),
        max(float(np.percentile(values[~solid], 99.5)) for values in arrays),
    )


def build_figure(
    archive_path: Path,
    summary_path: Path,
    case_label: str,
    geometry: dict[str, float],
) -> plt.Figure:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    beta = float(summary["beta"])
    alpha = float(summary["alpha_deg"])
    with np.load(archive_path) as archive:
        x = archive["x"]
        y = archive["y"]
        solid = archive["solid"].astype(bool)
        fluid_indices = nearest_fluid_indices(solid)
        airfoil = airfoil_vertices(geometry, beta, alpha)
        extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
        limits = {
            "speed": color_limits(archive, solid, "speed"),
            "vorticity": color_limits(archive, solid, "vorticity"),
        }

        figure, axes = plt.subplots(
            4, 7, figsize=(7.25, 4.42), squeeze=False, facecolor=WHITE
        )
        force_white_canvas(figure)
        figure.subplots_adjust(left=0.105, right=0.985, top=0.885, bottom=0.175, wspace=0.035, hspace=0.12)
        rows = (
            ("learned", "speed", "Potential-Rayleigh\nspeed"),
            ("laplacian", "speed", "FD-Laplacian\nspeed"),
            ("learned", "vorticity", "Potential-Rayleigh\nvorticity"),
            ("laplacian", "vorticity", "FD-Laplacian\nvorticity"),
        )
        images: dict[str, matplotlib.image.AxesImage] = {}
        for row, (method, field, row_label) in enumerate(rows):
            for column in range(7):
                modes = None if column == 0 else K_VALUES[column - 1]
                values = field_values(archive, "reference" if column == 0 else method, modes, field)
                vmin, vmax = limits[field]
                image = axes[row, column].imshow(
                    extend_under_airfoil(values, solid, fluid_indices),
                    origin="lower",
                    extent=extent,
                    cmap=VORTICITY_CMAP if field == "vorticity" else "magma",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                    rasterized=True,
                )
                images[field] = image
                add_airfoil(axes[row, column], airfoil)
                style_panel(axes[row, column])
                if row == 0:
                    axes[row, column].set_title("Reference" if column == 0 else f"$K={modes}$", fontsize=8.0, pad=2.5)
            axes[row, 0].set_ylabel(row_label, fontsize=7.5, labelpad=6.0, rotation=0, ha="right", va="center")

        figure.suptitle(
            rf"{case_label}: $\beta={beta:.2f}$, $\alpha={alpha:.2f}^\circ$",
            fontsize=10.0,
            y=0.965,
        )
        speed_bar = figure.add_axes((0.175, 0.065, 0.285, 0.017))
        vort_bar = figure.add_axes((0.625, 0.065, 0.285, 0.017))
        speed_cbar = figure.colorbar(images["speed"], cax=speed_bar, orientation="horizontal")
        vort_cbar = figure.colorbar(images["vorticity"], cax=vort_bar, orientation="horizontal")
        speed_cbar.set_label("Speed magnitude", fontsize=7.5, labelpad=1.5)
        vort_cbar.set_label("Vorticity", fontsize=7.5, labelpad=1.5)
        for colorbar in (speed_cbar, vort_cbar):
            colorbar.ax.tick_params(labelsize=6.5, length=2.0, width=0.5, pad=1.0)
            colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            colorbar.outline.set_linewidth(0.45)
        return figure


def build_combined_figure(
    evaluation_root: Path, geometry: dict[str, float]
) -> plt.Figure:
    figure = plt.figure(figsize=(14.5, 6.55), facecolor=WHITE)
    force_white_canvas(figure)
    outer = figure.add_gridspec(3, 1, left=0.055, right=0.995, top=0.985, bottom=0.035, hspace=0.24)
    for block, (case_name, case_label) in enumerate(CASES):
        archive_path = evaluation_root / case_name / "fields_stride2.npz"
        summary_path = evaluation_root / case_name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        beta = float(summary["beta"])
        alpha = float(summary["alpha_deg"])
        grid = outer[block].subgridspec(
            4,
            15,
            height_ratios=(0.45, 1.0, 1.0, 0.14),
            width_ratios=(1, 1, 1, 1, 1, 1, 1, 0.16, 1, 1, 1, 1, 1, 1, 1),
            hspace=0.12,
            wspace=0.045,
        )
        heading = figure.add_subplot(grid[0, :])
        heading.set_axis_off()
        heading.text(
            0.5,
            0.80,
            rf"{case_label}: $\beta={beta:.2f}$, $\alpha={alpha:.2f}^\circ$",
            ha="center",
            va="center",
            fontsize=8.2,
        )

        with np.load(archive_path) as archive:
            x = archive["x"]
            y = archive["y"]
            solid = archive["solid"].astype(bool)
            fluid_indices = nearest_fluid_indices(solid)
            airfoil = airfoil_vertices(geometry, beta, alpha)
            extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
            limits = {
                "speed": color_limits(archive, solid, "speed"),
                "vorticity": color_limits(archive, solid, "vorticity"),
            }
            images: dict[str, matplotlib.image.AxesImage] = {}
            for method_row, (method, method_label, method_color) in enumerate(
                (
                    ("learned", "Potential-Rayleigh", POTENTIAL_COLOR),
                    ("laplacian", "FD-Laplacian", LAPLACIAN_COLOR),
                )
            ):
                row = 1 + method_row
                for field, offset in (("speed", 0), ("vorticity", 8)):
                    for column in range(7):
                        modes = None if column == 0 else K_VALUES[column - 1]
                        values = field_values(archive, "reference" if column == 0 else method, modes, field)
                        vmin, vmax = limits[field]
                        axis = figure.add_subplot(grid[row, offset + column])
                        image = axis.imshow(
                            extend_under_airfoil(values, solid, fluid_indices),
                            origin="lower",
                            extent=extent,
                            cmap=VORTICITY_CMAP if field == "vorticity" else "magma",
                            vmin=vmin,
                            vmax=vmax,
                            interpolation="bilinear",
                            rasterized=True,
                        )
                        images[field] = image
                        add_airfoil(axis, airfoil)
                        style_panel(axis)
                        if method_row == 0:
                            axis.set_title(
                                "Reference" if column == 0 else f"$K={modes}$",
                                fontsize=5.6,
                                pad=1.2,
                            )
                        if column == 0:
                            axis.set_ylabel(
                                method_label,
                                fontsize=5.7,
                                color=method_color,
                                labelpad=3.0,
                            )

            speed_bar = figure.add_subplot(grid[3, 2:5])
            vort_bar = figure.add_subplot(grid[3, 10:13])
            speed_cbar = figure.colorbar(images["speed"], cax=speed_bar, orientation="horizontal")
            vort_cbar = figure.colorbar(images["vorticity"], cax=vort_bar, orientation="horizontal")
            speed_cbar.set_label("Speed magnitude", fontsize=5.1, labelpad=0.8)
            vort_cbar.set_label("Vorticity", fontsize=5.1, labelpad=0.8)
            for colorbar in (speed_cbar, vort_cbar):
                colorbar.ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=0.7)
                colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                colorbar.outline.set_linewidth(0.4)
    return figure


def build_field_figure(
    evaluation_root: Path, field: str, geometry: dict[str, float]
) -> plt.Figure:
    figure = plt.figure(figsize=(7.25, 6.15), facecolor=WHITE)
    force_white_canvas(figure)
    outer = figure.add_gridspec(3, 1, left=0.095, right=0.995, top=0.985, bottom=0.035, hspace=0.24)
    for block, (case_name, case_label) in enumerate(CASES):
        archive_path = evaluation_root / case_name / "fields_stride2.npz"
        summary_path = evaluation_root / case_name / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        beta = float(summary["beta"])
        alpha = float(summary["alpha_deg"])
        grid = outer[block].subgridspec(
            4,
            7,
            height_ratios=(0.38, 1.0, 1.0, 0.13),
            hspace=0.12,
            wspace=0.045,
        )
        heading = figure.add_subplot(grid[0, :])
        heading.set_axis_off()
        heading.text(
            0.5,
            0.74,
            rf"{case_label}: $\beta={beta:.2f}$, $\alpha={alpha:.2f}^\circ$",
            ha="center",
            va="center",
            fontsize=8.4,
        )

        with np.load(archive_path) as archive:
            x = archive["x"]
            y = archive["y"]
            solid = archive["solid"].astype(bool)
            fluid_indices = nearest_fluid_indices(solid)
            airfoil = airfoil_vertices(geometry, beta, alpha)
            extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
            vmin, vmax = color_limits(archive, solid, field)
            image: matplotlib.image.AxesImage | None = None
            for method_row, (method, method_label, method_color) in enumerate(
                (
                    ("learned", "Potential-Rayleigh", POTENTIAL_COLOR),
                    ("laplacian", "FD-Laplacian", LAPLACIAN_COLOR),
                )
            ):
                row = 1 + method_row
                for column in range(7):
                    modes = None if column == 0 else K_VALUES[column - 1]
                    values = field_values(archive, "reference" if column == 0 else method, modes, field)
                    axis = figure.add_subplot(grid[row, column])
                    image = axis.imshow(
                        extend_under_airfoil(values, solid, fluid_indices),
                        origin="lower",
                        extent=extent,
                        cmap=VORTICITY_CMAP if field == "vorticity" else "magma",
                        vmin=vmin,
                        vmax=vmax,
                        interpolation="bilinear",
                        rasterized=True,
                    )
                    add_airfoil(axis, airfoil)
                    style_panel(axis)
                    if method_row == 0:
                        axis.set_title(
                            "Reference" if column == 0 else f"$K={modes}$",
                            fontsize=6.2,
                            pad=1.3,
                        )
                    if column == 0:
                        axis.set_ylabel(
                            method_label,
                            fontsize=6.2,
                            color=method_color,
                            labelpad=3.5,
                        )
            assert image is not None
            colorbar_axis = figure.add_subplot(grid[3, 2:5])
            colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
            colorbar.set_label("Speed magnitude" if field == "speed" else "Vorticity", fontsize=5.7, labelpad=0.8)
            colorbar.ax.tick_params(labelsize=5.0, length=1.5, width=0.4, pad=0.7)
            colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            colorbar.outline.set_linewidth(0.4)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    configure_typography()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.evaluation_root / "evaluation_manifest.json").read_text(encoding="utf-8")
    )
    geometry = manifest["configs"]["k64"]
    for case_name, case_label in CASES:
        archive_path = args.evaluation_root / case_name / "fields_stride2.npz"
        summary_path = args.evaluation_root / case_name / "summary.json"
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        figure = build_figure(archive_path, summary_path, case_label, geometry)
        stem = args.output_root / f"airfoil_{case_name}_reconstruction_vector"
        figure.savefig(
            stem.with_suffix(".pdf"),
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor=WHITE,
            edgecolor=WHITE,
            transparent=False,
        )
        figure.savefig(
            stem.with_suffix(".png"),
            dpi=320,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor=WHITE,
            edgecolor=WHITE,
            transparent=False,
        )
        plt.close(figure)
        print(stem.with_suffix(".pdf"))

    combined = build_combined_figure(args.evaluation_root, geometry)
    combined_stem = args.output_root / "airfoil_reconstruction_vector"
    combined.savefig(
        combined_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "Airfoil reconstruction comparison",
            "Author": "Baiming Zhang",
            "Creator": "Matplotlib",
        },
        facecolor=WHITE,
        edgecolor=WHITE,
        transparent=False,
    )
    combined.savefig(
        combined_stem.with_suffix(".png"),
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor=WHITE,
        edgecolor=WHITE,
        transparent=False,
    )
    plt.close(combined)
    print(combined_stem.with_suffix(".pdf"))

    for field in ("speed", "vorticity"):
        field_figure = build_field_figure(args.evaluation_root, field, geometry)
        field_stem = args.output_root / f"airfoil_{field}_reconstruction_vector"
        field_figure.savefig(
            field_stem.with_suffix(".pdf"),
            metadata={
                "Title": f"Airfoil {field} reconstruction comparison",
                "Author": "Baiming Zhang",
                "Creator": "Matplotlib",
            },
            facecolor=WHITE,
            edgecolor=WHITE,
            transparent=False,
        )
        field_figure.savefig(
            field_stem.with_suffix(".png"),
            dpi=300,
            facecolor=WHITE,
            edgecolor=WHITE,
            transparent=False,
        )
        plt.close(field_figure)
        print(field_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
