"""Render the three-parameter fan obstacle proposed for the 3D Eigenfluid case.

The rotor axis is x, matching the streamwise direction of 3D_parametric_wake.
Each blade is a radial loft of symmetric NACA 00xx sections.  The three shape
parameters are spanwise pitch beta(r), spanwise chord c(r), and thickness t.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CASE_ROOT / "figures" / "fan_obstacle_concept.png"


@dataclass(frozen=True)
class FanGeometry:
    blade_count: int = 4
    hub_radius: float = 0.24
    hub_length: float = 0.52
    tip_radius: float = 1.0
    beta_root_deg: float = 48.0
    beta_tip_deg: float = 22.0
    chord_root: float = 0.46
    chord_tip: float = 0.22
    thickness_ratio: float = 0.12


def naca_symmetric_half_thickness(x01: np.ndarray, thickness: float) -> np.ndarray:
    """Closed-trailing-edge NACA 00xx half-thickness, normalized by chord."""
    return 5.0 * thickness * (
        0.2969 * np.sqrt(np.clip(x01, 0.0, 1.0))
        - 0.1260 * x01
        - 0.3516 * x01**2
        + 0.2843 * x01**3
        - 0.1036 * x01**4
    )


def blade_surfaces(
    geometry: FanGeometry,
    azimuth: float,
    radial_samples: int = 42,
    chord_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Return upper and lower surface grids for one blade."""
    radial = np.linspace(geometry.hub_radius, geometry.tip_radius, radial_samples)
    span = (radial - geometry.hub_radius) / (
        geometry.tip_radius - geometry.hub_radius
    )
    beta = np.deg2rad(
        geometry.beta_root_deg
        + span * (geometry.beta_tip_deg - geometry.beta_root_deg)
    )
    chord = geometry.chord_root + span * (
        geometry.chord_tip - geometry.chord_root
    )

    x01 = np.linspace(0.0, 1.0, chord_samples)
    chord_coordinate = x01 - 0.5
    half_thickness = naca_symmetric_half_thickness(
        x01, geometry.thickness_ratio
    )

    e_radial = np.array([0.0, math.cos(azimuth), math.sin(azimuth)])
    e_tangent = np.array([0.0, -math.sin(azimuth), math.cos(azimuth)])
    e_axial = np.array([1.0, 0.0, 0.0])

    upper = np.empty((radial_samples, chord_samples, 3), dtype=np.float64)
    lower = np.empty_like(upper)
    for index, (radius, pitch, local_chord) in enumerate(
        zip(radial, beta, chord, strict=True)
    ):
        chord_direction = math.cos(pitch) * e_tangent + math.sin(pitch) * e_axial
        normal_direction = -math.sin(pitch) * e_tangent + math.cos(pitch) * e_axial
        center = radius * e_radial
        camber_line = (
            center[None, :]
            + local_chord * chord_coordinate[:, None] * chord_direction[None, :]
        )
        offset = (
            local_chord * half_thickness[:, None] * normal_direction[None, :]
        )
        upper[index] = camber_line + offset
        lower[index] = camber_line - offset
    return upper, lower


def add_surface(ax: plt.Axes, surface: np.ndarray, color: str, alpha: float) -> None:
    ax.plot_surface(
        surface[:, :, 0],
        surface[:, :, 1],
        surface[:, :, 2],
        color=color,
        linewidth=0.15,
        edgecolor=(0.13, 0.20, 0.28, 0.20),
        antialiased=True,
        shade=True,
        alpha=alpha,
    )


def add_blade_edges(ax: plt.Axes, upper: np.ndarray, lower: np.ndarray) -> None:
    for row in (0, -1):
        ring = np.concatenate(
            (upper[row], lower[row, ::-1], upper[row, :1]), axis=0
        )
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#16324a", lw=0.75)
    for column in (0, -1):
        rim = np.concatenate(
            (
                upper[:, column],
                lower[::-1, column],
                upper[:1, column],
            ),
            axis=0,
        )
        ax.plot(rim[:, 0], rim[:, 1], rim[:, 2], color="#16324a", lw=0.75)


def add_hub(ax: plt.Axes, geometry: FanGeometry) -> None:
    axial = np.linspace(-geometry.hub_length / 2.0, geometry.hub_length / 2.0, 30)
    angle = np.linspace(0.0, 2.0 * math.pi, 72)
    axial_grid, angle_grid = np.meshgrid(axial, angle, indexing="ij")
    y = geometry.hub_radius * np.cos(angle_grid)
    z = geometry.hub_radius * np.sin(angle_grid)
    ax.plot_surface(
        axial_grid,
        y,
        z,
        color="#6d7d8b",
        linewidth=0.0,
        antialiased=True,
        shade=True,
    )
    for endpoint in (-geometry.hub_length / 2.0, geometry.hub_length / 2.0):
        vertices = [
            list(
                zip(
                    np.full_like(angle, endpoint),
                    geometry.hub_radius * np.cos(angle),
                    geometry.hub_radius * np.sin(angle),
                    strict=True,
                )
            )
        ]
        ax.add_collection3d(
            Poly3DCollection(
                vertices,
                facecolor="#7e8d99",
                edgecolor="#263746",
                linewidth=0.5,
            )
        )


def add_domain_hint(ax: plt.Axes) -> None:
    """Draw a small wireframe cue for the [-pi, pi]^3 periodic domain."""
    limit = math.pi
    corners = np.array(
        [
            [x, y, z]
            for x in (-limit, limit)
            for y in (-limit, limit)
            for z in (-limit, limit)
        ],
        dtype=np.float64,
    )
    edges = []
    for first in range(8):
        for second in range(first + 1, 8):
            if np.count_nonzero(corners[first] != corners[second]) == 1:
                edges.append((corners[first], corners[second]))
    for first, second in edges:
        ax.plot(
            [first[0], second[0]],
            [first[1], second[1]],
            [first[2], second[2]],
            color="#778899",
            lw=0.45,
            alpha=0.32,
        )


def render_fan(
    ax: plt.Axes,
    geometry: FanGeometry,
    title: str,
    subtitle: str,
    show_domain: bool = False,
) -> None:
    for blade_index in range(geometry.blade_count):
        azimuth = 2.0 * math.pi * blade_index / geometry.blade_count
        upper, lower = blade_surfaces(geometry, azimuth)
        add_surface(ax, upper, "#55a8d4", 0.97)
        add_surface(ax, lower, "#2f78a5", 0.97)
        add_blade_edges(ax, upper, lower)
    add_hub(ax, geometry)

    if show_domain:
        add_domain_hint(ax)
        bound = math.pi
    else:
        bound = 1.12

    ax.quiver(
        -0.75,
        0.0,
        0.0,
        0.45,
        0.0,
        0.0,
        color="#c24536",
        arrow_length_ratio=0.16,
        linewidth=1.5,
    )
    ax.text(-0.78, -0.07, -0.06, "axis / flow  x", color="#9f2f25", fontsize=8)

    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_zlim(-bound, bound)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=24.0, azim=-52.0)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=8)
    ax.set_xlabel("x", labelpad=-2)
    ax.set_ylabel("y", labelpad=-2)
    ax.set_zlabel("z", labelpad=-2)
    ax.tick_params(labelsize=7, pad=-1)
    ax.grid(True, alpha=0.18)


def main() -> None:
    nominal = FanGeometry()
    variants = [
        (
            nominal,
            "Nominal three-blade obstacle",
            r"$R_t=1,\ \beta:48^\circ\to22^\circ,\ c:0.46\to0.22,\ t=0.12$",
            False,
        ),
        (
            replace(nominal, beta_tip_deg=8.0),
            r"Parameter 1: pitch / twist  $\beta(r)$",
            r"stronger spanwise twist: $48^\circ\to8^\circ$",
            False,
        ),
        (
            replace(nominal, chord_tip=0.38),
            r"Parameter 2: chord  $c(r)$",
            r"wider tip: $0.46\to0.38$",
            False,
        ),
        (
            replace(nominal, thickness_ratio=0.20),
            r"Parameter 3: thickness  $t$",
            r"NACA thickness ratio: $0.12\to0.20$",
            False,
        ),
    ]

    figure = plt.figure(figsize=(15.5, 12.5), dpi=180, constrained_layout=True)
    figure.patch.set_facecolor("#f4f7fa")
    figure.suptitle(
        "Expected 3D Eigenfluid obstacle: parameterized turbine / fan rotor",
        fontsize=17,
        fontweight="bold",
    )
    for index, (geometry, title, subtitle, show_domain) in enumerate(variants, 1):
        axis = figure.add_subplot(2, 2, index, projection="3d")
        axis.set_facecolor("#f7f9fb")
        render_fan(axis, geometry, title, subtitle, show_domain)

    figure.text(
        0.5,
        0.012,
        r"Computational domain follows the existing 3D case: "
        r"$[-\pi,\pi]^3$; rotor axis and imposed mean flow are aligned with $x$.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#334455",
    )
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(DEFAULT_OUTPUT, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
