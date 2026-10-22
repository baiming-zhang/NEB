"""Geometry/configuration compatibility for DNS and analytic CUDA networks.

Neural training lives in experiments/metaball/train.py. This module deliberately
retains only the unchanged geometry and sampling definitions used by DNS.
"""









from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import time
from dataclasses import dataclass, fields
from datetime import datetime
from _neb_paths import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
NATURE_BLUE = "#3B6FB6"
NATURE_RED = "#D1495B"
NATURE_DARK = "#243447"


def wall_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def replace_with_retry(temporary: Path, path: Path) -> None:

    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.02)


@dataclass(frozen=True)
class Config:
    seed: int
    modes: int
    epochs: int
    points_per_step: int
    validation_points: int
    validation_geometries: int
    validate_every: int
    checkpoint_every: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    outer_half_extent: float
    outer_superellipsoid_power: int
    sphere_count: int
    sphere_center_min: float
    sphere_center_max: float
    sphere_radius_min: float
    sphere_radius_max: float
    metaball_softmin_tau: float
    minimum_overlap: float
    outer_clearance: float
    geometry_max_attempts: int
    outer_envelope_transition: float
    obstacle_envelope_transition: float
    input_features: int
    coordinate_normalization: str
    hidden: int
    hidden_layers: int
    siren_first_omega0: float
    siren_hidden_omega0: float
    learning_rate: float
    min_learning_rate: float
    lr_scheduler: str
    grad_clip: float
    mass_floor: float
    reorthogonalization_passes: int
    causal_sorting: str
    sampling_mode: str
    basis_objective: str
    symmetry_augmentation: str
    analytic_eigenmodes: bool
    degenerate_block_alignment: bool
    report_raw_rank: bool
    dtype: str
    result_root: str


def load_config(path: Path, overrides: dict[str, int | None]) -> Config:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = {field.name for field in fields(Config)}
    unknown = sorted(set(payload) - valid)
    if unknown:
        raise ValueError("unknown config fields: " + ", ".join(unknown))
    for key, value in overrides.items():
        if value is not None:
            payload[key] = value
    cfg = Config(**payload)
    validate_contract(cfg)
    return cfg


def validate_contract(cfg: Config) -> None:
    checks = {
        "three spheres": cfg.sphere_count == 3,
        "15 raw inputs": cfg.input_features == 15,
        "centered-unit normalization": cfg.coordinate_normalization == "centered_unit",
        "positive hidden-layer count": cfg.hidden_layers >= 1,
        "two MGS passes": cfg.reorthogonalization_passes == 2,
        "geometry Rayleigh sorting": cfg.causal_sorting == "geometry_rayleigh",
        "IID PCG64 sampling": cfg.sampling_mode == "iid_pcg64_uniform_fluid_volume",
        "Dirichlet objective": cfg.basis_objective == "antisymmetric_potential_dirichlet_rayleigh",
        "no symmetry augmentation": cfg.symmetry_augmentation == "none",
        "no analytic modes": not cfg.analytic_eigenmodes,
        "no degeneracy alignment": not cfg.degenerate_block_alignment,
        "no raw rank reporting": not cfg.report_raw_rank,
        "FP32": cfg.dtype == "float32",
        "cosine scheduler": cfg.lr_scheduler == "cosine",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("contract violation: " + ", ".join(failed))
    if cfg.modes < 1 or cfg.points_per_step < cfg.modes:
        raise ValueError("quadrature must contain at least as many points as modes")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    replace_with_retry(temporary, path)


def append_log(run_dir: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with (run_dir / "train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def geometry_from_token(cfg: Config, token: int) -> np.ndarray:

    rng = np.random.default_rng(cfg.seed + 104729 * int(token))
    for _ in range(cfg.geometry_max_attempts):
        sample = rng.random((3, 4))
        geometry = np.empty((3, 4), dtype=np.float64)
        geometry[:, :3] = cfg.sphere_center_min + sample[:, :3] * (
            cfg.sphere_center_max - cfg.sphere_center_min
        )
        geometry[:, 3] = cfg.sphere_radius_min + sample[:, 3] * (
            cfg.sphere_radius_max - cfg.sphere_radius_min
        )
        if not overlap_graph_connected(geometry, cfg.minimum_overlap):
            continue
        if not spheres_inside_rounded_cube(cfg, geometry):
            continue
        return geometry.astype(np.float32)
    raise RuntimeError(
        f"could not sample connected three-sphere geometry for token {token}"
    )


def overlap_graph_connected(geometry: np.ndarray, minimum_overlap: float) -> bool:
    adjacency = [set() for _ in range(3)]
    for left in range(3):
        for right in range(left + 1, 3):
            distance = float(np.linalg.norm(geometry[left, :3] - geometry[right, :3]))
            overlap = float(geometry[left, 3] + geometry[right, 3] - distance)
            if overlap >= minimum_overlap:
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached, frontier = {0}, [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    return len(reached) == 3


def sphere_directions(count: int = 256) -> np.ndarray:
    index = np.arange(count, dtype=np.float64)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((radius * np.cos(golden * index), radius * np.sin(golden * index), z))


SPHERE_DIRECTIONS = sphere_directions()


def spheres_inside_rounded_cube(cfg: Config, geometry: np.ndarray) -> bool:
    half = cfg.outer_half_extent - cfg.outer_clearance
    if half <= 0.0:
        return False
    for sphere in geometry:
        boundary = sphere[:3] + sphere[3] * SPHERE_DIRECTIONS
        normalized = np.abs((boundary - 0.5) / half)
        if np.any(np.sum(normalized**cfg.outer_superellipsoid_power, axis=1) >= 1.0):
            return False
    return True


def outer_level_numpy(cfg: Config, points: np.ndarray) -> np.ndarray:
    normalized = (points - 0.5) / cfg.outer_half_extent
    return 1.0 - np.sum(normalized**cfg.outer_superellipsoid_power, axis=1)


def obstacle_level_numpy(cfg: Config, points: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - geometry[None, :, :3], axis=2)
    distances -= geometry[None, :, 3]
    minimum = distances.min(axis=1, keepdims=True)
    tau = cfg.metaball_softmin_tau
    return minimum[:, 0] - tau * np.log(np.exp(-(distances - minimum) / tau).sum(axis=1))


def fluid_mask(cfg: Config, points: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    return (outer_level_numpy(cfg, points) > 0.0) & (
        obstacle_level_numpy(cfg, points, geometry) > 0.0
    )


def iid_fluid_points(cfg: Config, count: int, geometry: np.ndarray, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    pieces: list[np.ndarray] = []
    remaining = int(count)
    while remaining:
        candidates = rng.random((max(2 * remaining, 256), 3), dtype=np.float32)
        accepted = candidates[fluid_mask(cfg, candidates, geometry)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
    return np.concatenate(pieces).astype(np.float32, copy=False)


def approximate_fluid_volume(cfg: Config, geometry: np.ndarray, token: int) -> float:
    rng = np.random.default_rng(cfg.seed + 3_000_003 + int(token))
    points = rng.random((65536, 3), dtype=np.float32)
    return float(np.mean(fluid_mask(cfg, points, geometry), dtype=np.float64))
