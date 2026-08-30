"""Render the first six trained fan modes as a self-contained SVG fragment."""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def project(point: list[float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        165.0 + 40.0 * x + 13.0 * y,
        126.0 - 28.0 * z + 8.0 * y - 3.5 * x,
        0.35 * x + 0.65 * y + z,
    )


def cylinder_paths() -> str:
    pieces: list[str] = []
    rings: list[list[tuple[float, float, float]]] = []
    for x in (-math.pi, math.pi):
        ring = [
            project([x, 2.0 * math.cos(a), 2.0 * math.sin(a)])
            for a in (2.0 * math.pi * index / 48 for index in range(49))
        ]
        rings.append(ring)
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{px:.1f},{py:.1f}"
            for index, (px, py, _) in enumerate(ring)
        )
        pieces.append(f'<path class="m6-domain" d="{path}"/>')
    for index in (0, 12, 24, 36):
        left = rings[0][index]
        right = rings[1][index]
        pieces.append(
            '<path class="m6-domain-edge" '
            f'd="M{left[0]:.1f},{left[1]:.1f} '
            f'L{right[0]:.1f},{right[1]:.1f}"/>'
        )
    return "".join(pieces)


def rotor_paths() -> str:
    blades: list[tuple[float, str]] = []
    for blade in range(4):
        angle = blade * math.pi / 2.0
        tangent = (-math.sin(angle), math.cos(angle))
        radial = (math.cos(angle), math.sin(angle))
        polygon: list[tuple[float, float, float]] = []
        for radius, width in ((0.25, 0.14), (1.0, 0.10)):
            for sign in (-1.0, 1.0):
                y = radial[0] * radius + tangent[0] * width * sign
                z = radial[1] * radius + tangent[1] * width * sign
                polygon.append(project([0.0, y, z]))
        polygon = [polygon[0], polygon[1], polygon[3], polygon[2]]
        depth = sum(point[2] for point in polygon) / len(polygon)
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{px:.1f},{py:.1f}"
            for index, (px, py, _) in enumerate(polygon)
        ) + " Z"
        blades.append((depth, f'<path class="m6-blade" d="{path}"/>'))
    hub = project([0.0, 0.0, 0.0])
    blades.sort()
    return (
        "".join(path for _, path in blades)
        + f'<ellipse class="m6-hub" cx="{hub[0]:.1f}" cy="{hub[1]:.1f}" '
        'rx="10.5" ry="12.5"/>'
    )


def mode_svg(payload: dict[str, object], panel_index: int) -> str:
    points = payload["points"]
    vectors = payload["vectors"]
    arrows: list[tuple[float, str]] = []
    for point, vector in list(zip(points, vectors))[::2]:
        magnitude = math.sqrt(sum(component * component for component in vector))
        if not math.isfinite(magnitude) or magnitude < 0.035:
            continue
        factor = min(0.34, 0.34 / max(magnitude, 1.0))
        finish = [
            point[axis] + factor * vector[axis]
            for axis in range(3)
        ]
        start_screen = project(point)
        finish_screen = project(finish)
        opacity = min(0.92, 0.22 + 0.38 * magnitude)
        width = min(1.7, 0.65 + 0.35 * magnitude)
        arrows.append(
            (
                start_screen[2],
                '<path class="m6-vector" '
                f'd="M{start_screen[0]:.1f},{start_screen[1]:.1f} '
                f'L{finish_screen[0]:.1f},{finish_screen[1]:.1f}" '
                f'opacity="{opacity:.2f}" stroke-width="{width:.2f}" '
                f'marker-end="url(#m6-arrow-{panel_index})"/>',
            )
        )
    arrows.sort()
    rank = int(payload["rank"])
    rayleigh = float(payload["rayleigh"])
    scale = float(payload["vector_scale_p92"])
    return f"""
<svg class="m6-panel" viewBox="0 0 330 235" role="img"
     aria-label="Mode {rank}, Rayleigh value {rayleigh:.3f}">
  <title>Mode {rank}</title>
  <desc>Four-blade fan velocity basis field in the cylindrical domain.</desc>
  <defs>
    <marker id="m6-arrow-{panel_index}" markerWidth="5" markerHeight="5"
            refX="4.2" refY="2.5" orient="auto">
      <path class="m6-arrow-head" d="M0,0 L5,2.5 L0,5 Z"/>
    </marker>
  </defs>
  <text class="m6-mode-label" x="10" y="18">Mode {rank}</text>
  <text class="m6-lambda" x="320" y="18" text-anchor="end">λ = {rayleigh:.3f}</text>
  {cylinder_paths()}
  {''.join(path for _, path in arrows)}
  {rotor_paths()}
  <text class="m6-scale" x="10" y="226">p92 |u| = {scale:.4f}</text>
</svg>"""


def main() -> None:
    args = parse_args()
    payloads: list[dict[str, object]] = []
    for rank in range(1, 7):
        query = urllib.parse.urlencode(
            {
                "beta": 23.0,
                "chord": 0.28,
                "thickness": 0.14,
                "rank": rank,
            }
        )
        with urllib.request.urlopen(
            f"{args.base_url}/api/mode?{query}",
            timeout=30,
        ) as response:
            payloads.append(json.load(response))
    panels = "".join(
        mode_svg(payload, index)
        for index, payload in enumerate(payloads, start=1)
    )
    fragment = f"""<div id="fan-first-six-modes">
  <div class="m6-grid">{panels}</div>
</div>
<style>
  #fan-first-six-modes {{
    width: 100%;
    color: var(--foreground);
  }}
  #fan-first-six-modes .m6-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }}
  #fan-first-six-modes .m6-panel {{
    display: block;
    width: 100%;
    height: auto;
    overflow: visible;
  }}
  #fan-first-six-modes .m6-domain {{
    fill: none;
    stroke: var(--border);
    stroke-width: 1.1;
  }}
  #fan-first-six-modes .m6-domain-edge {{
    fill: none;
    stroke: var(--border);
    stroke-width: .8;
  }}
  #fan-first-six-modes .m6-vector {{
    fill: none;
    stroke: var(--viz-series-1);
    stroke-linecap: round;
  }}
  #fan-first-six-modes .m6-arrow-head {{
    fill: var(--viz-series-1);
  }}
  #fan-first-six-modes .m6-blade,
  #fan-first-six-modes .m6-hub {{
    fill: var(--muted);
    stroke: var(--muted-foreground);
    stroke-width: .8;
  }}
  #fan-first-six-modes .m6-mode-label,
  #fan-first-six-modes .m6-lambda {{
    fill: var(--foreground);
    font-weight: 500;
  }}
  #fan-first-six-modes .m6-scale {{
    fill: var(--muted-foreground);
  }}
  @media (max-width: 640px) {{
    #fan-first-six-modes .m6-grid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
  }}
  @media (max-width: 420px) {{
    #fan-first-six-modes .m6-grid {{
      grid-template-columns: 1fr;
    }}
  }}
</style>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment, encoding="utf-8")


if __name__ == "__main__":
    main()
