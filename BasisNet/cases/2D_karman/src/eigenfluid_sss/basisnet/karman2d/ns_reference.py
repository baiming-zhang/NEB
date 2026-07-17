from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

SOURCE_DOI = "10.5281/zenodo.18600528"
SOURCE_CENTRE = (0.2, 0.2)
LENGTH_SCALE = 9.0
MEAN_START_INDEX = 400
PHASE_START_INDEX = 767
PHASE_END_INDEX = 800


def canonical_coordinates(x: np.ndarray, y: np.ndarray):
    xr = LENGTH_SCALE * (x - SOURCE_CENTRE[0])
    signed = y - SOURCE_CENTRE[1]
    yr = LENGTH_SCALE * signed
    lower = signed < 0.0
    q = np.clip(((-signed[lower]) - 0.15) / 0.05, 0.0, 1.0)
    yr[lower] -= 0.2 * q * q * (3.0 - 2.0 * q)
    q = np.clip((signed[~lower] - 0.15) / 0.06, 0.0, 1.0)
    yr[~lower] += 0.11 * q * q * (3.0 - 2.0 * q)
    return xr, yr


def inlet_profile(y: np.ndarray):
    u = 1.5 * np.maximum(1.0 - (0.5 * y) ** 2, 0.0)
    v = np.zeros_like(y)
    psi = 1.5 * (y - y**3 / 12.0)
    omega = 0.75 * y
    return u, v, psi, omega


class ScatteredDifferentiator:
    def __init__(self, points: np.ndarray, neighbours: int = 28):
        distance, index = cKDTree(points).query(points, k=neighbours + 1)
        self.index = index[:, 1:]
        distance = distance[:, 1:]
        delta = points[self.index] - points[:, None, :]
        scale = np.maximum(distance[:, -1:], 1.0e-12)
        self.weight = np.exp(-4.0 * (distance / scale) ** 2)
        self.dx, self.dy = delta[:, :, 0], delta[:, :, 1]
        a = np.sum(self.weight * self.dx**2, axis=1)
        b = np.sum(self.weight * self.dx * self.dy, axis=1)
        c = np.sum(self.weight * self.dy**2, axis=1)
        determinant = np.maximum(a * c - b * b, 1.0e-20)
        self.ixx, self.ixy = c / determinant, -b / determinant
        self.iyy = a / determinant

    def gradient(self, values: np.ndarray) -> np.ndarray:
        difference = values[self.index] - values[:, None]
        rx = np.sum(self.weight * self.dx * difference, axis=1)
        ry = np.sum(self.weight * self.dy * difference, axis=1)
        return np.column_stack(
            (
                self.ixx * rx + self.ixy * ry,
                self.ixy * rx + self.iyy * ry,
            )
        )


class StreamfunctionIntegrator:
    def __init__(self, points: np.ndarray, neighbours: int = 8):
        _, adjacency = cKDTree(points).query(points, k=neighbours + 1)
        left = np.repeat(np.arange(len(points)), neighbours)
        right = adjacency[:, 1:].reshape(-1)
        keep = left < right
        self.left, self.right = left[keep], right[keep]
        delta = points[self.right] - points[self.left]
        self.dx, self.dy = delta[:, 0], delta[:, 1]
        rows = np.repeat(np.arange(len(self.left)), 2)
        columns = np.column_stack((self.left, self.right)).reshape(-1)
        values = np.tile(np.array([-1.0, 1.0]), len(self.left))
        matrix = sparse.coo_matrix(
            (values, (rows, columns)),
            shape=(len(self.left) + 1, len(points)),
        ).tolil()
        anchor = int(
            np.argmin(
                (points[:, 0] - points[:, 0].min()) ** 2 + (points[:, 1] + 1.95) ** 2
            )
        )
        matrix[len(self.left), anchor] = 1.0
        self.matrix = matrix.tocsr()

    def integrate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        target = (
            0.5 * (u[self.left] + u[self.right]) * self.dy
            - 0.5 * (v[self.left] + v[self.right]) * self.dx
        )
        rhs = np.concatenate((target, [0.0]))
        return lsqr(
            self.matrix,
            rhs,
            atol=1.0e-11,
            btol=1.0e-11,
            iter_lim=2400,
        )[0]


def align_streamfunction(xr, yr, psi):
    near = xr < xr.min() + 0.035
    shift = float(np.median(psi[near] - inlet_profile(yr[near])[2]))
    return psi - shift


def build_cache(source: Path, output: Path, phase_frames: int = 17) -> Path:
    with h5py.File(source, "r") as handle:
        x = handle["coordinates/x"][:]
        y = handle["coordinates/y"][:]
        time = handle["time"][:]
        indices = np.rint(
            np.linspace(PHASE_START_INDEX, PHASE_END_INDEX, phase_frames)
        ).astype(int)
        phase_u = handle["fields/field_1"][:, indices]
        phase_v = handle["fields/field_2"][:, indices]
        mean_u = np.mean(handle["fields/field_1"][:, MEAN_START_INDEX:], axis=1)
        mean_v = np.mean(handle["fields/field_2"][:, MEAN_START_INDEX:], axis=1)

    xr, yr = canonical_coordinates(x, y)
    points = np.column_stack((xr, yr))
    differentiator = ScatteredDifferentiator(points)
    integrator = StreamfunctionIntegrator(points)
    mean_grad_u = differentiator.gradient(mean_u)
    mean_grad_v = differentiator.gradient(mean_v)
    mean_omega = mean_grad_v[:, 0] - mean_grad_u[:, 1]
    mean_psi = align_streamfunction(xr, yr, integrator.integrate(mean_u, mean_v))
    phase_omega, phase_psi = [], []
    for frame in range(phase_frames):
        grad_u = differentiator.gradient(phase_u[:, frame])
        grad_v = differentiator.gradient(phase_v[:, frame])
        phase_omega.append(grad_v[:, 0] - grad_u[:, 1])
        phase_psi.append(
            align_streamfunction(
                xr,
                yr,
                integrator.integrate(phase_u[:, frame], phase_v[:, frame]),
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    cache = output / "actual_canonical_cache.npz"
    np.savez_compressed(
        cache,
        xr=xr.astype(np.float32),
        yr=yr.astype(np.float32),
        phase_indices=indices,
        phase_times=time[indices].astype(np.float32),
        phase_u=phase_u.astype(np.float32),
        phase_v=phase_v.astype(np.float32),
        phase_psi=np.asarray(phase_psi, np.float32).T,
        phase_omega=np.asarray(phase_omega, np.float32).T,
        mean_u=mean_u.astype(np.float32),
        mean_v=mean_v.astype(np.float32),
        mean_psi=mean_psi.astype(np.float32),
        mean_omega=mean_omega.astype(np.float32),
        phase_frames=phase_frames,
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_doi": SOURCE_DOI,
                "reynolds_number": 100,
                "phase_indices": indices.tolist(),
                "phase_frames": phase_frames,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache


def upstream_extension(xmax: float, seed: int, count: int):
    if xmax <= 1.0e-8:
        empty = np.empty(0, dtype=np.float32)
        return {key: empty for key in ("x", "y", "u", "v", "psi", "omega")}
    unit = (
        torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
        .draw(count)
        .cpu()
        .numpy()
    )
    x, y = xmax * unit[:, 0], -2.0 + 4.0 * unit[:, 1]
    boundary_y = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
    boundary_x = xmax * (np.arange(100, dtype=np.float32) + 0.5) / 100.0
    x = np.concatenate((x, np.zeros_like(boundary_y), boundary_x, boundary_x))
    y = np.concatenate(
        (
            y,
            boundary_y,
            np.full_like(boundary_x, -2.0),
            np.full_like(boundary_x, 2.0),
        )
    )
    u, v, psi, omega = inlet_profile(y)
    return {
        key: value.astype(np.float32)
        for key, value in zip(
            ("x", "y", "u", "v", "psi", "omega"),
            (x, y, u, v, psi, omega),
        )
    }
