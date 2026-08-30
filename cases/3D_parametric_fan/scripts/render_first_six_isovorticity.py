"""Render six vortex-magnitude surfaces colored by signed axial vorticity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cupy as cp
import numpy as np
from skimage.measure import marching_cubes

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from serve_interactive import InferenceEngine  # noqa: E402
from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (  # noqa: E402
    fluid_points,
    peel_and_adjoint,
    raw_fields,
    raw_velocity,
    rotor_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nr", type=int, default=48)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--chunk", type=int, default=4096)
    return parser.parse_args()


def project(point: np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in point)
    return (
        165.0 + 40.0 * x + 13.0 * y,
        128.0 - 28.0 * z + 8.0 * y - 3.5 * x,
        0.35 * x + 0.65 * y + z,
    )


def cylinder_paths() -> str:
    rings: list[list[tuple[float, float, float]]] = []
    pieces: list[str] = []
    for x in (-math.pi, math.pi):
        ring = [
            project(
                np.asarray(
                    [x, 2.0 * math.cos(angle), 2.0 * math.sin(angle)]
                )
            )
            for angle in (
                2.0 * math.pi * index / 48
                for index in range(49)
            )
        ]
        rings.append(ring)
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{px:.1f},{py:.1f}"
            for index, (px, py, _) in enumerate(ring)
        )
        pieces.append(f'<path class="iso-domain" d="{path}"/>')
    for index in (0, 12, 24, 36):
        left = rings[0][index]
        right = rings[1][index]
        pieces.append(
            '<path class="iso-domain-edge" '
            f'd="M{left[0]:.1f},{left[1]:.1f} '
            f'L{right[0]:.1f},{right[1]:.1f}"/>'
        )
    return "".join(pieces)


def rotor_paths() -> str:
    paths: list[str] = []
    for blade in range(4):
        angle = blade * math.pi / 2.0
        tangent = np.asarray((-math.sin(angle), math.cos(angle)))
        radial = np.asarray((math.cos(angle), math.sin(angle)))
        polygon: list[tuple[float, float, float]] = []
        for radius, half_width in ((0.25, 0.14), (1.0, 0.10)):
            for sign in (-1.0, 1.0):
                yz = radial * radius + tangent * half_width * sign
                polygon.append(project(np.asarray([0.0, yz[0], yz[1]])))
        polygon = [polygon[0], polygon[1], polygon[3], polygon[2]]
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{px:.1f},{py:.1f}"
            for index, (px, py, _) in enumerate(polygon)
        )
        paths.append(f'<path class="iso-blade" d="{path} Z"/>')
    hub = project(np.zeros(3))
    paths.append(
        f'<ellipse class="iso-hub" cx="{hub[0]:.1f}" cy="{hub[1]:.1f}" '
        'rx="10.5" ry="12.5"/>'
    )
    return "".join(paths)


def compute_modes(
    engine: InferenceEngine,
    nx: int,
    nr: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = engine.cfg
    parameters = np.asarray((0.5, 0.5, 0.5), dtype=np.float32)
    quadrature = fluid_points(
        cfg,
        cfg.validation_points,
        parameters,
        cfg.seed + 9_000_000,
    )
    raw = raw_fields(cfg, engine.model, quadrature, parameters)
    peeled = peel_and_adjoint(
        cfg,
        raw,
        len(quadrature),
        need_adjoint=False,
        need_coefficients=True,
    )
    energy_order = np.argsort(peeled["mode_energies"], kind="stable")[:6]
    coefficient_matrix = peeled["coefficient_matrix"][:, energy_order]
    rayleigh = np.asarray(peeled["mode_energies"])[energy_order]

    x = np.linspace(cfg.xmin, cfg.xmax, nx, dtype=np.float32)
    y = np.linspace(-cfg.farfield_radius, cfg.farfield_radius, nr, dtype=np.float32)
    z = np.linspace(-cfg.farfield_radius, cfg.farfield_radius, nr, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    all_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    radial = np.sqrt(all_points[:, 1] ** 2 + all_points[:, 2] ** 2)
    obstacle_distance = np.asarray(
        rotor_distance(cfg, all_points, parameters)
    )
    valid = (radial < 1.92) & (obstacle_distance > 0.07)
    valid_indices = np.flatnonzero(valid)
    velocity = np.zeros((len(all_points), 6, 3), dtype=np.float32)
    for start in range(0, len(valid_indices), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        points = all_points[indices]
        raw_chunk = raw_velocity(cfg, engine.model, points, parameters)
        modes = raw_chunk @ coefficient_matrix
        modes = cp.asnumpy(modes).reshape(3, len(points), 6).transpose(1, 2, 0)
        velocity[indices] = modes.astype(np.float32, copy=False)
    velocity = velocity.reshape(nx, nr, nr, 6, 3)
    return x, y, z, velocity, rayleigh


def vorticity_field(
    velocity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    result = np.empty(velocity.shape, dtype=np.float32)
    for mode in range(velocity.shape[3]):
        u = velocity[:, :, :, mode, 0]
        v = velocity[:, :, :, mode, 1]
        w = velocity[:, :, :, mode, 2]
        du_dx, du_dy, du_dz = np.gradient(u, x, y, z, edge_order=2)
        dv_dx, dv_dy, dv_dz = np.gradient(v, x, y, z, edge_order=2)
        dw_dx, dw_dy, dw_dz = np.gradient(w, x, y, z, edge_order=2)
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        result[:, :, :, mode, 0] = omega_x
        result[:, :, :, mode, 1] = omega_y
        result[:, :, :, mode, 2] = omega_z
    return result


def render_panel(
    omega: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rank: int,
    rayleigh: float,
    quantile: float,
) -> str:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    radius = np.sqrt(yy * yy + zz * zz)
    interior = (
        (radius < 1.78)
        & (xx > x[2])
        & (xx < x[-3])
    )
    magnitude = np.linalg.norm(omega, axis=-1)
    finite_values = magnitude[interior & np.isfinite(magnitude)]
    level = float(np.quantile(finite_values, quantile))
    volume = np.where(interior, magnitude, 0.0)
    vertices, faces, normals, _ = marching_cubes(
        volume,
        level=level,
        spacing=(
            float(x[1] - x[0]),
            float(y[1] - y[0]),
            float(z[1] - z[0]),
        ),
        step_size=3,
        allow_degenerate=False,
    )
    vertices += np.asarray((x[0], y[0], z[0]))
    spacing = np.asarray(
        (x[1] - x[0], y[1] - y[0], z[1] - z[0]),
        dtype=np.float64,
    )
    vertex_indices = np.rint(
        (vertices - np.asarray((x[0], y[0], z[0]))) / spacing
    ).astype(np.int64)
    vertex_indices[:, 0] = np.clip(vertex_indices[:, 0], 0, len(x) - 1)
    vertex_indices[:, 1] = np.clip(vertex_indices[:, 1], 0, len(y) - 1)
    vertex_indices[:, 2] = np.clip(vertex_indices[:, 2], 0, len(z) - 1)
    vertex_axial = omega[
        vertex_indices[:, 0],
        vertex_indices[:, 1],
        vertex_indices[:, 2],
        0,
    ]
    color_values = np.abs(omega[:, :, :, 0][interior])
    color_limit = max(float(np.quantile(color_values, 0.98)), 1.0e-9)
    triangles: list[tuple[float, str]] = []
    for face in faces:
        triangle = vertices[face]
        signed_value = float(np.mean(vertex_axial[face]))
        normalized = float(np.clip(signed_value / color_limit, -1.0, 1.0))
        if normalized <= -0.125:
            color_class = f"iso-negative-{min(4, max(1, int(math.ceil(-4.0 * normalized))))}"
        elif normalized >= 0.125:
            color_class = f"iso-positive-{min(4, max(1, int(math.ceil(4.0 * normalized))))}"
        else:
            color_class = "iso-neutral"
        projected = [project(vertex) for vertex in triangle]
        depth = sum(item[2] for item in projected) / 3.0
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{px:.1f},{py:.1f}"
            for index, (px, py, _) in enumerate(projected)
        )
        triangles.append(
            (
                depth,
                f'<path class="iso-surface {color_class}" '
                f'd="{path} Z"/>',
            )
        )
    triangles.sort()
    return f"""
<svg class="iso-panel" viewBox="0 0 330 240" role="img"
     aria-label="Mode {rank} vortex surface colored by signed axial vorticity">
  <title>Mode {rank} vorticity isosurface</title>
  <desc>Vorticity-magnitude isosurface colored blue for negative axial
  vorticity and red for positive axial vorticity.</desc>
  <text class="iso-label" x="10" y="18">Mode {rank}</text>
  <text class="iso-lambda" x="320" y="18" text-anchor="end">λ = {rayleigh:.3f}</text>
  {cylinder_paths()}
  {''.join(path for _, path in triangles)}
  {rotor_paths()}
  <text class="iso-level" x="10" y="231">|ω| = {level:.3f} · |ωₓ|lim = {color_limit:.3f}</text>
</svg>"""


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(args.checkpoint.resolve())
    x, y, z, velocity, rayleigh = compute_modes(
        engine,
        args.nx,
        args.nr,
        args.chunk,
    )
    vorticity = vorticity_field(velocity, x, y, z)
    panels = "".join(
        render_panel(
            vorticity[:, :, :, mode, :],
            x,
            y,
            z,
            mode + 1,
            float(rayleigh[mode]),
            args.quantile,
        )
        for mode in range(6)
    )
    fragment = f"""<div id="fan-six-isovorticity">
  <div class="iso-legend" aria-label="Signed axial vorticity legend">
    <span><i class="iso-swatch iso-negative-4"></i>ωₓ &lt; 0</span>
    <span><i class="iso-swatch iso-positive-4"></i>ωₓ &gt; 0</span>
  </div>
  <div class="iso-grid">{panels}</div>
</div>
<style>
  #fan-six-isovorticity {{
    width: 100%;
    color: var(--foreground);
  }}
  #fan-six-isovorticity .iso-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }}
  #fan-six-isovorticity .iso-legend {{
    display: flex;
    justify-content: center;
    gap: 18px;
    margin-bottom: 8px;
    color: var(--muted-foreground);
  }}
  #fan-six-isovorticity .iso-legend span {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  #fan-six-isovorticity .iso-swatch {{
    display: inline-block;
    width: 18px;
    height: 8px;
  }}
  #fan-six-isovorticity .iso-panel {{
    display: block;
    width: 100%;
    height: auto;
    overflow: visible;
  }}
  #fan-six-isovorticity .iso-domain,
  #fan-six-isovorticity .iso-domain-edge {{
    fill: none;
    stroke: var(--border);
    stroke-width: 1;
  }}
  #fan-six-isovorticity .iso-surface {{
    opacity: .92;
    stroke-width: .18;
    stroke-linejoin: round;
  }}
  #fan-six-isovorticity .iso-negative-1 {{
    fill: color-mix(in srgb, var(--viz-series-1) 35%, var(--muted));
    stroke: color-mix(in srgb, var(--viz-series-1) 35%, var(--muted));
  }}
  #fan-six-isovorticity .iso-negative-2 {{
    fill: color-mix(in srgb, var(--viz-series-1) 55%, var(--muted));
    stroke: color-mix(in srgb, var(--viz-series-1) 55%, var(--muted));
  }}
  #fan-six-isovorticity .iso-negative-3 {{
    fill: color-mix(in srgb, var(--viz-series-1) 75%, var(--muted));
    stroke: color-mix(in srgb, var(--viz-series-1) 75%, var(--muted));
  }}
  #fan-six-isovorticity .iso-negative-4 {{
    fill: var(--viz-series-1);
    stroke: var(--viz-series-1);
  }}
  #fan-six-isovorticity .iso-positive-1 {{
    fill: color-mix(in srgb, var(--destructive) 35%, var(--muted));
    stroke: color-mix(in srgb, var(--destructive) 35%, var(--muted));
  }}
  #fan-six-isovorticity .iso-positive-2 {{
    fill: color-mix(in srgb, var(--destructive) 55%, var(--muted));
    stroke: color-mix(in srgb, var(--destructive) 55%, var(--muted));
  }}
  #fan-six-isovorticity .iso-positive-3 {{
    fill: color-mix(in srgb, var(--destructive) 75%, var(--muted));
    stroke: color-mix(in srgb, var(--destructive) 75%, var(--muted));
  }}
  #fan-six-isovorticity .iso-positive-4 {{
    fill: var(--destructive);
    stroke: var(--destructive);
  }}
  #fan-six-isovorticity .iso-neutral {{
    fill: var(--muted);
    stroke: var(--muted);
  }}
  #fan-six-isovorticity .iso-blade,
  #fan-six-isovorticity .iso-hub {{
    fill: var(--muted);
    stroke: var(--muted-foreground);
    stroke-width: .8;
  }}
  #fan-six-isovorticity .iso-label,
  #fan-six-isovorticity .iso-lambda {{
    fill: var(--foreground);
    font-weight: 500;
  }}
  #fan-six-isovorticity .iso-level {{
    fill: var(--muted-foreground);
  }}
  @media (max-width: 640px) {{
    #fan-six-isovorticity .iso-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
  }}
  @media (max-width: 420px) {{
    #fan-six-isovorticity .iso-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")


if __name__ == "__main__":
    main()
