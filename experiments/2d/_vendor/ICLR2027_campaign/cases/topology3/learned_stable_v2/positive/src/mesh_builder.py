"""Conforming triangular meshes for the topology-family fluid domains."""
from __future__ import annotations

from dataclasses import dataclass

import meshpy.triangle as triangle
import numpy as np

from geometry import ObstacleSpec, fluid_mask, get_obstacles, obstacle_boundary, outer_boundary


@dataclass
class DomainMesh:
    xy: np.ndarray
    tri: np.ndarray
    boundary_edges: np.ndarray
    boundary_edge_labels: np.ndarray
    boundary_nodes: tuple[np.ndarray, ...]
    case_name: str
    obstacles: tuple[ObstacleSpec, ...]
    tau: float
    max_area: float


def build_domain_mesh(
    case_name: str,
    tau: float,
    max_area: float,
    outer_points: int,
    obstacle_points: int,
    min_angle: float = 28.0,
) -> DomainMesh:
    outer = outer_boundary(tau, outer_points)
    specs = get_obstacles(case_name)
    obstacles = [obstacle_boundary(spec, obstacle_points) for spec in specs]
    points: list[list[float]] = []
    facets: list[list[int]] = []
    markers: list[int] = []

    def append_loop(loop: np.ndarray, marker: int, reverse: bool) -> None:
        if reverse:
            loop = loop[::-1].copy()
        start = len(points)
        points.extend(loop.tolist())
        for index in range(len(loop)):
            facets.append([start + index, start + (index + 1) % len(loop)])
            markers.append(marker)

    append_loop(outer, 1, reverse=False)
    for index, obstacle in enumerate(obstacles):
        append_loop(obstacle, index + 2, reverse=True)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_facets(facets, facet_markers=markers)
    info.set_holes([spec.anchor.tolist() for spec in specs])
    built = triangle.build(
        info,
        max_volume=max_area,
        min_angle=min_angle,
        allow_boundary_steiner=False,
        allow_volume_steiner=True,
        quality_meshing=True,
    )
    xy = np.asarray(built.points, dtype=np.float64)
    tri = np.asarray(built.elements, dtype=np.int32)
    edge = np.asarray(built.facets, dtype=np.int32)
    labels = np.asarray(built.facet_markers, dtype=np.int32) - 1
    if edge.ndim != 2 or edge.shape[1] != 2 or len(edge) != len(labels):
        raise RuntimeError("MeshPy did not retain marked boundary facets")

    points_tri = xy[tri]
    det = ((points_tri[:, 1, 0] - points_tri[:, 0, 0])
           * (points_tri[:, 2, 1] - points_tri[:, 0, 1])
           - (points_tri[:, 2, 0] - points_tri[:, 0, 0])
           * (points_tri[:, 1, 1] - points_tri[:, 0, 1]))
    negative = det < 0.0
    tri[negative, 1], tri[negative, 2] = tri[negative, 2].copy(), tri[negative, 1].copy()
    points_tri = xy[tri]
    det = ((points_tri[:, 1, 0] - points_tri[:, 0, 0])
           * (points_tri[:, 2, 1] - points_tri[:, 0, 1])
           - (points_tri[:, 2, 0] - points_tri[:, 0, 0])
           * (points_tri[:, 1, 1] - points_tri[:, 0, 1]))
    if np.min(det) <= 0.0:
        raise RuntimeError("mesh contains non-positive triangles")
    centroids = np.mean(points_tri, axis=1)
    if not bool(np.all(fluid_mask(centroids, case_name, tau))):
        raise RuntimeError("mesh contains triangle centroids outside the analytic fluid domain")
    expected = set(range(len(specs) + 1))
    if np.min(labels) < 0 or np.max(labels) > len(specs) or set(np.unique(labels)) != expected:
        raise RuntimeError(f"unexpected boundary markers: {np.unique(labels).tolist()}")
    boundary_nodes = tuple(np.unique(edge[labels == component]) for component in range(len(specs) + 1))
    if any(len(nodes) == 0 for nodes in boundary_nodes):
        raise RuntimeError("empty boundary component")
    return DomainMesh(
        xy=xy,
        tri=tri,
        boundary_edges=edge,
        boundary_edge_labels=labels,
        boundary_nodes=boundary_nodes,
        case_name=case_name,
        obstacles=specs,
        tau=float(tau),
        max_area=float(max_area),
    )


def mesh_diagnostics(mesh: DomainMesh) -> dict:
    points = mesh.xy[mesh.tri]
    edges = np.stack((points[:, 1] - points[:, 0], points[:, 2] - points[:, 1],
                      points[:, 0] - points[:, 2]), axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    det = ((points[:, 1, 0] - points[:, 0, 0]) * (points[:, 2, 1] - points[:, 0, 1])
           - (points[:, 2, 0] - points[:, 0, 0]) * (points[:, 1, 1] - points[:, 0, 1]))
    area = 0.5 * det
    min_altitude = np.min(2.0 * area / np.max(lengths, axis=1))
    boundary_counts = [int(np.count_nonzero(mesh.boundary_edge_labels == label))
                       for label in range(len(mesh.boundary_nodes))]
    return {
        "vertices": int(len(mesh.xy)),
        "triangles": int(len(mesh.tri)),
        "boundary_edges_per_component": boundary_counts,
        "area": float(np.sum(area)),
        "triangle_area_min": float(np.min(area)),
        "triangle_area_max": float(np.max(area)),
        "minimum_altitude": float(min_altitude),
        "all_triangle_centroids_in_fluid": bool(np.all(
            fluid_mask(np.mean(points, axis=1), mesh.case_name, mesh.tau))),
    }
