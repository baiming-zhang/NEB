"""Complete GT reconstruction and basis audit for the optimized 3D fan run.

This script is framework-free: the trained SIREN modes, analytic spatial
derivatives, SSS peeling, and all-mode projections use the handwritten
CuPy/NumPy backend.  The independent rotating-fan Navier--Stokes reference is
never used during basis training.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cupy as cp
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.fan3d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
    fluid_points,
)
from eigenfluid.gen.fan3d.optimized_backend import (  # noqa: E402
    OptimizedFanModel,
    moving_wall_lifting_fields,
    peel_and_adjoint,
    raw_fields,
    raw_velocity,
)


BLUE = "#3B6FB6"
LIGHT_BLUE = "#9ECAE1"
RED = "#D1495B"
LIGHT_RED = "#F4A3A8"
DARK_BLUE = "#17365D"
GREY = "#777777"
LIGHT_GREY = "#D9E2EC"
BLACK = "#222222"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    run_root = (
        CASE_ROOT
        / "results"
        / "k1024_manual_n4096_analytic_fp16fwd_fp32bwd_5000_run008"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=run_root / "checkpoints" / "best_inference.npz",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "gt_rotating_stable_re500_nominal_n96"
            / "stable_fan_reference.h5"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=run_root / "complete_evaluation",
    )
    parser.add_argument("--fit-points", type=int, default=16384)
    parser.add_argument("--validation-points", type=int, default=8192)
    parser.add_argument("--generalization-points", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=1024)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def save_pair(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=260, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#9AA7B2",
            "axes.linewidth": 0.8,
            "axes.titlecolor": DARK_BLUE,
            "axes.labelcolor": BLACK,
            "xtick.color": "#4A5560",
            "ytick.color": "#4A5560",
            "grid.color": "#DCE6EF",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


class EvaluationEngine:
    def __init__(self, checkpoint: Path) -> None:
        metadata = json.loads(
            checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )
        config = metadata["config"]
        if not bool(config.get("optimized_backend", False)):
            raise ValueError("checkpoint is not from the optimized analytic backend")
        if int(config["modes"]) != 1024:
            raise ValueError("complete evaluation expects the requested 1024 modes")
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        layer_count = int(config["hidden_layers"]) + 1
        parameter_shapes: list[tuple[int, ...]] = []
        for layer in range(int(config["hidden_layers"])):
            parameter_shapes.extend(
                (arrays[f"l{layer}.weight"].shape, arrays[f"l{layer}.bias"].shape)
            )
        parameter_shapes.extend(
            (arrays["output.weight"].shape, arrays["output.bias"].shape)
        )
        for index, shape in enumerate(parameter_shapes):
            arrays[f"adam.p{index:02d}.exp_avg"] = np.zeros(shape, np.float32)
            arrays[f"adam.p{index:02d}.exp_avg_sq"] = np.zeros(shape, np.float32)
        arrays["adam.steps"] = np.zeros(
            (int(config["modes"]), 2 * layer_count), dtype=np.int64
        )
        self.metadata = metadata
        self.cfg = cfg_namespace(config)
        self.model = OptimizedFanModel(self.cfg, arrays)

    def lifting_fields(self, points: np.ndarray) -> dict[str, np.ndarray]:
        if bool(getattr(self.cfg, "moving_wall_lifting", False)):
            fields = moving_wall_lifting_fields(self.cfg, points)
            return {
                key: cp.asnumpy(value).astype(np.float64, copy=False)
                for key, value in fields.items()
            }
        velocity = np.zeros((len(points), 3), dtype=np.float64)
        velocity[:, 0] = 1.0
        return {
            "velocity": velocity,
            "gradient": np.zeros((len(points), 3, 3), dtype=np.float64),
        }

    def reference_residual(
        self, reference: "ReferenceField", points: np.ndarray
    ) -> np.ndarray:
        return reference.interpolate(reference.velocity, points) - self.lifting_fields(
            points
        )["velocity"]

    def normalize(self, beta: float, chord: float, thickness: float) -> np.ndarray:
        cfg = self.cfg
        return np.asarray(
            [
                (beta - cfg.beta_tip_min_deg)
                / (cfg.beta_tip_max_deg - cfg.beta_tip_min_deg),
                (chord - cfg.chord_tip_min)
                / (cfg.chord_tip_max - cfg.chord_tip_min),
                (thickness - cfg.thickness_min)
                / (cfg.thickness_max - cfg.thickness_min),
            ],
            dtype=np.float32,
        )


class ReferenceField:
    def __init__(self, path: Path) -> None:
        with h5py.File(path, "r") as archive:
            self.axis = archive["axis"][:].astype(np.float64)
            self.velocity = archive["velocity"][:].astype(np.float64)
            self.vorticity = archive["vorticity"][:].astype(np.float64)
            self.rotor_mask = archive["rotor_mask"][:].astype(np.float64)
            self.attributes = {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in archive.attrs.items()
            }
        self.dx = float(self.axis[1] - self.axis[0])
        self.xmin = float(self.axis[0])

    def interpolate(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        coordinates = ((points - self.xmin) / self.dx).T
        return np.column_stack(
            [
                map_coordinates(
                    field[component],
                    coordinates,
                    order=1,
                    mode="grid-wrap",
                    prefilter=False,
                )
                for component in range(field.shape[0])
            ]
        )

    def disturbance(self, points: np.ndarray) -> np.ndarray:
        values = self.interpolate(self.velocity, points)
        values[:, 0] -= 1.0
        return values


def component_major(values: np.ndarray) -> np.ndarray:
    return values.T.reshape(-1)


def relative_error(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(
        np.linalg.norm((approximation - reference).ravel())
        / max(np.linalg.norm(reference.ravel()), 1.0e-30)
    )


def rotate_quarter(points_or_vectors: np.ndarray) -> np.ndarray:
    rotated = points_or_vectors.copy()
    rotated[:, 1] = -points_or_vectors[:, 2]
    rotated[:, 2] = points_or_vectors[:, 1]
    return rotated


def load_gt_protocol(reference: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = reference.parent
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    sensitivity = json.loads(
        (root / "grid_sensitivity.json").read_text(encoding="utf-8")
    )
    return protocol, sensitivity


def fit_raw_projection(
    engine: EvaluationEngine,
    points: np.ndarray,
    target: np.ndarray,
    parameters: np.ndarray,
    chunk: int,
) -> tuple[cp.ndarray, cp.ndarray, float, dict[str, Any]]:
    modes = engine.cfg.modes
    target_vector = component_major(target)
    gram = cp.zeros((modes, modes), dtype=cp.float64)
    right = cp.zeros(modes, dtype=cp.float64)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        basis = raw_velocity(
            engine.cfg, engine.model, points[start:stop], parameters
        )
        target_chunk = np.concatenate(
            [
                target_vector[component * len(points) + start : component * len(points) + stop]
                for component in range(3)
            ]
        )
        target_device = cp.asarray(target_chunk, dtype=cp.float64)
        gram += basis.T @ basis
        right += basis.T @ target_device
        print(f"[fit {stop:05d}/{len(points):05d}]", flush=True)
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    threshold = cp.maximum(eigenvalues[-1], engine.cfg.mass_floor) * 1.0e-8
    keep = eigenvalues > threshold
    coefficients = eigenvectors[:, keep] @ (
        (eigenvectors[:, keep].T @ right) / eigenvalues[keep]
    )
    target_mass = float(np.dot(target_vector, target_vector))
    residual = (
        target_mass
        - 2.0 * float((coefficients @ right).get())
        + float((coefficients @ gram @ coefficients).get())
    )
    diagnostics = {
        "effective_rank": int(cp.count_nonzero(keep).get()),
        "relative_rank_threshold": 1.0e-8,
        "gram_largest_eigenvalue": float(eigenvalues[-1].get()),
        "gram_smallest_retained_eigenvalue": float(eigenvalues[keep][0].get()),
        "gram_condition_retained": float(
            (eigenvalues[-1] / eigenvalues[keep][0]).get()
        ),
        "fit_disturbance_relative_l2": math.sqrt(
            max(residual, 0.0) / max(target_mass, 1.0e-30)
        ),
    }
    return gram, right, target_mass, diagnostics


def peel_basis(
    engine: EvaluationEngine, parameters: np.ndarray
) -> tuple[cp.ndarray, np.ndarray, dict[str, Any]]:
    points = fluid_points(
        engine.cfg,
        engine.cfg.validation_points,
        parameters,
        engine.cfg.seed + 9_000_000,
    )
    fields = raw_fields(engine.cfg, engine.model, points, parameters)
    peeled = peel_and_adjoint(
        engine.cfg,
        fields,
        len(points),
        need_adjoint=False,
        need_coefficients=True,
        diagnostics=True,
    )
    coefficients = peeled["coefficient_matrix"]
    energies = np.asarray(peeled["mode_energies"], dtype=np.float64)
    diagnostics = {
        "points": len(points),
        "orthogonality_error": float(peeled["orth_error"]),
        "effective_raw_rank": int(peeled["effective_raw_rank"]),
        "mean_rayleigh_loss_scaled": float(peeled["loss"]),
        "min_rayleigh": float(np.min(energies)),
        "max_rayleigh": float(np.max(energies)),
    }
    del fields, peeled
    cp.get_default_memory_pool().free_all_blocks()
    return coefficients, energies, diagnostics


def convergence_projections(
    gram: cp.ndarray,
    right: cp.ndarray,
    target_mass: float,
    peeling: cp.ndarray,
    mode_counts: list[int],
) -> tuple[cp.ndarray, cp.ndarray, list[dict[str, Any]]]:
    peeled_gram = peeling.T @ gram @ peeling
    peeled_gram = 0.5 * (peeled_gram + peeled_gram.T)
    peeled_right = peeling.T @ right
    raw_columns: list[cp.ndarray] = []
    peeled_columns: list[cp.ndarray] = []
    rows: list[dict[str, Any]] = []
    for count in mode_counts:
        local_gram = peeled_gram[:count, :count]
        local_right = peeled_right[:count]
        eigenvalues, eigenvectors = cp.linalg.eigh(local_gram)
        threshold = cp.maximum(eigenvalues[-1], 1.0e-30) * 1.0e-8
        keep = eigenvalues > threshold
        local = eigenvectors[:, keep] @ (
            (eigenvectors[:, keep].T @ local_right) / eigenvalues[keep]
        )
        peeled_column = cp.zeros(peeling.shape[1], dtype=cp.float64)
        peeled_column[:count] = local
        raw_column = peeling @ peeled_column
        residual = (
            target_mass
            - 2.0 * float((raw_column @ right).get())
            + float((raw_column @ gram @ raw_column).get())
        )
        rows.append(
            {
                "modes": count,
                "effective_rank": int(cp.count_nonzero(keep).get()),
                "fit_disturbance_relative_l2": math.sqrt(
                    max(residual, 0.0) / max(target_mass, 1.0e-30)
                ),
            }
        )
        raw_columns.append(raw_column)
        peeled_columns.append(peeled_column)
    return (
        cp.stack(raw_columns, axis=1),
        cp.stack(peeled_columns, axis=1),
        rows,
    )


def evaluate_projection_family(
    engine: EvaluationEngine,
    points: np.ndarray,
    target: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reconstructions = np.empty(
        (len(points), coefficients.shape[1], 3), dtype=np.float64
    )
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        basis = raw_velocity(
            engine.cfg, engine.model, points[start:stop], parameters
        )
        values = (
            cp.asnumpy(basis @ coefficients)
            .reshape(3, stop - start, coefficients.shape[1])
            .transpose(1, 2, 0)
        )
        reconstructions[start:stop] = values
        print(f"[validate {stop:05d}/{len(points):05d}]", flush=True)
    difference = reconstructions - target[:, None, :]
    disturbance_errors = np.sqrt(
        np.sum(difference * difference, axis=(0, 2))
        / np.maximum(np.sum(target * target), 1.0e-30)
    )
    lifting = engine.lifting_fields(points)["velocity"]
    target_full = target + lifting
    reconstruction_full = reconstructions + lifting[:, None, :]
    full_errors = np.sqrt(
        np.sum(
            (reconstruction_full - target_full[:, None, :]) ** 2,
            axis=(0, 2),
        )
        / np.maximum(np.sum(target_full * target_full), 1.0e-30)
    )
    return reconstructions, disturbance_errors, full_errors


def region_metrics(
    engine: EvaluationEngine,
    points: np.ndarray,
    target: np.ndarray,
    reconstruction: np.ndarray,
    cfg: Any,
) -> dict[str, Any]:
    x = points[:, 0]
    radius = np.hypot(points[:, 1], points[:, 2])
    masks = {
        "all_fluid": np.ones(len(points), dtype=bool),
        "near_rotor": (np.abs(x) < 0.75) & (radius < 1.30),
        "near_wake": (x >= 0.30) & (x < 1.50) & (radius < 1.25),
        "far_wake": (x >= 1.50) & (x < cfg.xmax - 0.35) & (radius < 1.25),
        "core": (np.abs(x) < cfg.xmax - 0.72)
        & (radius < cfg.farfield_radius - 0.25),
        "outer_field": radius >= 1.40,
    }
    lifting = engine.lifting_fields(points)["velocity"]
    target_full = target + lifting
    reconstruction_full = reconstruction + lifting
    output: dict[str, Any] = {}
    for label, mask in masks.items():
        output[label] = {
            "points": int(np.count_nonzero(mask)),
            "disturbance_relative_l2": relative_error(
                target[mask], reconstruction[mask]
            ),
            "full_velocity_relative_l2": relative_error(
                target_full[mask], reconstruction_full[mask]
            ),
        }
    return output


def evaluate_raw_coefficients(
    engine: EvaluationEngine,
    points: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    chunk: int,
    need_gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    count = coefficients.shape[1]
    velocity = np.empty((len(points), count, 3), dtype=np.float64)
    gradient = (
        np.empty((len(points), count, 3, 3), dtype=np.float64)
        if need_gradient
        else None
    )
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        if need_gradient:
            fields = raw_fields(
                engine.cfg, engine.model, points[start:stop], parameters
            )
            basis_velocity = fields["velocity"]
            basis_gradient = fields["gradient"]
        else:
            fields = None
            basis_velocity = raw_velocity(
                engine.cfg, engine.model, points[start:stop], parameters
            )
            basis_gradient = None
        velocity[start:stop] = (
            cp.asnumpy(basis_velocity @ coefficients)
            .reshape(3, stop - start, count)
            .transpose(1, 2, 0)
        )
        if need_gradient and gradient is not None and basis_gradient is not None:
            gradient[start:stop] = (
                cp.asnumpy(basis_gradient @ coefficients)
                .reshape(3, 3, stop - start, count)
                .transpose(2, 3, 0, 1)
            )
        del fields, basis_velocity, basis_gradient
    return velocity, gradient


def reconstruction_physics(
    engine: EvaluationEngine,
    points: np.ndarray,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    validation_reconstruction: np.ndarray,
    reference: ReferenceField,
    chunk: int,
) -> dict[str, Any]:
    _, gradient = evaluate_raw_coefficients(
        engine, points, parameters, coefficients, chunk, need_gradient=True
    )
    if gradient is None:
        raise RuntimeError("gradient evaluation unexpectedly missing")
    lifting = engine.lifting_fields(points)
    gradient = gradient[:, 0] + lifting["gradient"]
    divergence = np.trace(gradient, axis1=1, axis2=2)
    divergence_ratio = float(
        np.linalg.norm(divergence)
        / max(np.linalg.norm(gradient.ravel()), 1.0e-30)
    )
    rotated_points = rotate_quarter(points)
    rotated_velocity, _ = evaluate_raw_coefficients(
        engine,
        rotated_points,
        parameters,
        coefficients,
        chunk,
        need_gradient=False,
    )
    reconstruction_total = validation_reconstruction + lifting["velocity"]
    rotated_lifting = engine.lifting_fields(rotated_points)["velocity"]
    rotated_total = rotated_velocity[:, 0] + rotated_lifting
    expected_reconstruction = rotate_quarter(reconstruction_total)
    reconstruction_c4 = relative_error(
        expected_reconstruction, rotated_total
    )
    gt_here = reference.interpolate(reference.velocity, points)
    gt_rotated = reference.interpolate(reference.velocity, rotated_points)
    gt_c4 = relative_error(rotate_quarter(gt_here), gt_rotated)

    rng = np.random.default_rng(engine.cfg.seed + 44_000_000)
    boundary_count = 2048
    radius = engine.cfg.farfield_radius * np.sqrt(rng.random(boundary_count))
    angle = 2.0 * math.pi * rng.random(boundary_count)
    inlet = np.column_stack(
        (
            np.full(boundary_count, engine.cfg.xmin),
            radius * np.cos(angle),
            radius * np.sin(angle),
        )
    ).astype(np.float32)
    axial = rng.uniform(
        engine.cfg.xmin, engine.cfg.xmax, boundary_count
    )
    lateral_angle = 2.0 * math.pi * rng.random(boundary_count)
    lateral = np.column_stack(
        (
            axial,
            engine.cfg.farfield_radius * np.cos(lateral_angle),
            engine.cfg.farfield_radius * np.sin(lateral_angle),
        )
    ).astype(np.float32)
    boundary = np.concatenate((inlet, lateral))
    boundary_velocity, _ = evaluate_raw_coefficients(
        engine, boundary, parameters, coefficients, chunk, need_gradient=False
    )
    return {
        "divergence_l2_over_gradient_l2": divergence_ratio,
        "reconstruction_c4_relative_l2": reconstruction_c4,
        "gt_c4_relative_l2": gt_c4,
        "inlet_disturbance_rms": float(
            np.sqrt(np.mean(boundary_velocity[:boundary_count, 0] ** 2))
        ),
        "lateral_farfield_disturbance_rms": float(
            np.sqrt(np.mean(boundary_velocity[boundary_count:, 0] ** 2))
        ),
    }


def plane_definitions(reference: ReferenceField) -> list[dict[str, Any]]:
    axis = reference.axis.astype(np.float32)
    middle = int(np.argmin(np.abs(axis)))
    center = float(axis[middle])
    off_axis_index = middle + 1
    off_axis = float(axis[off_axis_index])
    first, second = np.meshgrid(axis, axis, indexing="ij")
    return [
        {
            "label": r"$x=0$ (rotor plane)",
            "first": axis,
            "second": axis,
            "first_label": "y",
            "second_label": "z",
            "points": np.column_stack(
                (np.full(first.size, center), first.ravel(), second.ravel())
            ).astype(np.float32),
            "gt_velocity": reference.velocity[0, middle, :, :],
            "gt_vorticity": reference.vorticity[0, middle, :, :],
            "obstacle": reference.rotor_mask[middle, :, :] > 0.5,
            "normal_component": 0,
        },
        {
            "label": rf"$y=\Delta$ (meridional, $\Delta={off_axis:.3f}$)",
            "first": axis,
            "second": axis,
            "first_label": "x",
            "second_label": "z",
            "points": np.column_stack(
                (first.ravel(), np.full(first.size, off_axis), second.ravel())
            ).astype(np.float32),
            "gt_velocity": reference.velocity[0, :, off_axis_index, :],
            "gt_vorticity": reference.vorticity[1, :, off_axis_index, :],
            "obstacle": reference.rotor_mask[:, off_axis_index, :] > 0.5,
            "normal_component": 1,
        },
        {
            "label": rf"$z=\Delta$ (meridional, $\Delta={off_axis:.3f}$)",
            "first": axis,
            "second": axis,
            "first_label": "x",
            "second_label": "y",
            "points": np.column_stack(
                (first.ravel(), second.ravel(), np.full(first.size, off_axis))
            ).astype(np.float32),
            "gt_velocity": reference.velocity[0, :, :, off_axis_index],
            "gt_vorticity": reference.vorticity[2, :, :, off_axis_index],
            "obstacle": reference.rotor_mask[:, :, off_axis_index] > 0.5,
            "normal_component": 2,
        },
    ]


def vorticity_from_gradient(gradient: np.ndarray) -> np.ndarray:
    output = np.empty(gradient.shape[:-2] + (3,), dtype=gradient.dtype)
    output[..., 0] = gradient[..., 2, 1] - gradient[..., 1, 2]
    output[..., 1] = gradient[..., 0, 2] - gradient[..., 2, 0]
    output[..., 2] = gradient[..., 1, 0] - gradient[..., 0, 1]
    return output


def plot_comparison_planes(
    engine: EvaluationEngine,
    reference: ReferenceField,
    parameters: np.ndarray,
    coefficients: cp.ndarray,
    chunk: int,
    output_dir: Path,
) -> dict[str, Any]:
    planes = plane_definitions(reference)
    velocity_rows: list[list[np.ndarray]] = []
    vorticity_rows: list[list[np.ndarray]] = []
    metrics: dict[str, Any] = {}
    for plane in planes:
        velocity, gradient = evaluate_raw_coefficients(
            engine,
            plane["points"],
            parameters,
            coefficients,
            chunk,
            need_gradient=True,
        )
        if gradient is None:
            raise RuntimeError("slice gradient missing")
        lifting = engine.lifting_fields(plane["points"])
        shape = (len(plane["first"]), len(plane["second"]))
        predicted_velocity = (
            velocity[:, 0, 0] + lifting["velocity"][:, 0]
        ).reshape(shape)
        predicted_vorticity = vorticity_from_gradient(
            gradient[:, 0] + lifting["gradient"]
        )[
            :, plane["normal_component"]
        ].reshape(shape)
        physical = (
            np.hypot(plane["points"][:, 1], plane["points"][:, 2])
            <= engine.cfg.farfield_radius
        ).reshape(shape)
        mask = physical & ~plane["obstacle"]
        velocity_rows.append(
            [
                np.where(mask, plane["gt_velocity"], np.nan),
                np.where(mask, predicted_velocity, np.nan),
                np.where(mask, predicted_velocity - plane["gt_velocity"], np.nan),
            ]
        )
        vorticity_rows.append(
            [
                np.where(mask, plane["gt_vorticity"], np.nan),
                np.where(mask, predicted_vorticity, np.nan),
                np.where(mask, predicted_vorticity - plane["gt_vorticity"], np.nan),
            ]
        )
        metrics[plane["label"]] = {
            "axial_velocity_relative_l2": relative_error(
                plane["gt_velocity"][mask], predicted_velocity[mask]
            ),
            "normal_vorticity_relative_l2": relative_error(
                plane["gt_vorticity"][mask], predicted_vorticity[mask]
            ),
        }

    for quantity, rows, stem, symbol in (
        (
            "Axial velocity",
            velocity_rows,
            "gt_reconstruction_velocity_xyz_slices",
            r"$u_x$",
        ),
        (
            "Plane-normal vorticity",
            vorticity_rows,
            "gt_reconstruction_vorticity_xyz_slices",
            r"$\omega_n$",
        ),
    ):
        figure, axes = plt.subplots(
            3, 3, figsize=(12.4, 10.0), constrained_layout=True, facecolor="white"
        )
        column_titles = ["Numerical GT", "1024-mode reconstruction", "Error"]
        for row_index, (plane, fields) in enumerate(zip(planes, rows)):
            shared_limit = max(
                float(np.nanquantile(np.abs(fields[0]), 0.99)),
                float(np.nanquantile(np.abs(fields[1]), 0.99)),
                1.0e-8,
            )
            error_limit = max(
                float(np.nanquantile(np.abs(fields[2]), 0.99)), 1.0e-8
            )
            for column_index, field in enumerate(fields):
                axis = axes[row_index, column_index]
                limit = shared_limit if column_index < 2 else error_limit
                image = axis.pcolormesh(
                    plane["first"],
                    plane["second"],
                    field.T,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                obstacle = np.ma.masked_where(~plane["obstacle"], plane["obstacle"])
                axis.pcolormesh(
                    plane["first"],
                    plane["second"],
                    obstacle.T,
                    shading="auto",
                    cmap=matplotlib.colors.ListedColormap([GREY]),
                    vmin=0.0,
                    vmax=1.0,
                )
                axis.set_aspect("equal")
                axis.set_xlabel(plane["first_label"])
                axis.set_ylabel(plane["second_label"])
                if row_index == 0:
                    axis.set_title(column_titles[column_index])
                if column_index == 0:
                    axis.text(
                        -0.22,
                        0.5,
                        plane["label"],
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        color=DARK_BLUE,
                    )
                figure.colorbar(image, ax=axis, shrink=0.78, pad=0.02)
        figure.suptitle(f"{quantity}: GT, reconstruction, and error ({symbol})", color=DARK_BLUE)
        save_pair(figure, output_dir / stem)
    return metrics


def plot_first_six_modes(
    engine: EvaluationEngine,
    reference: ReferenceField,
    parameters: np.ndarray,
    peeling: cp.ndarray,
    energies: np.ndarray,
    chunk: int,
    output_dir: Path,
) -> dict[str, Any]:
    selected = np.argsort(energies, kind="stable")[:6]
    coefficients = peeling[:, selected]
    selected_energies = energies[selected]
    planes = plane_definitions(reference)
    velocity_data: list[np.ndarray] = []
    vorticity_data: list[np.ndarray] = []
    for plane in planes:
        velocity, gradient = evaluate_raw_coefficients(
            engine,
            plane["points"],
            parameters,
            coefficients,
            chunk,
            need_gradient=True,
        )
        if gradient is None:
            raise RuntimeError("mode slice gradient missing")
        shape = (len(plane["first"]), len(plane["second"]), 6)
        component = plane["normal_component"]
        velocity_data.append(
            velocity[:, :, component].reshape(shape)
        )
        omega = vorticity_from_gradient(gradient)
        vorticity_data.append(omega[:, :, component].reshape(shape))

    for quantity, data, stem, symbol in (
        ("Velocity modes", velocity_data, "first_six_velocity_xyz_slices", "u_n"),
        (
            "Vorticity modes",
            vorticity_data,
            "first_six_vorticity_xyz_slices",
            r"\omega_n",
        ),
    ):
        figure, axes = plt.subplots(
            6, 3, figsize=(11.6, 17.2), constrained_layout=True, facecolor="white"
        )
        for mode in range(6):
            finite_values = []
            for plane, values in zip(planes, data):
                shape = values[:, :, mode].shape
                physical = (
                    np.hypot(plane["points"][:, 1], plane["points"][:, 2])
                    <= engine.cfg.farfield_radius
                ).reshape(shape)
                finite_values.append(
                    np.abs(values[:, :, mode][physical & ~plane["obstacle"]])
                )
            limit = max(float(np.quantile(np.concatenate(finite_values), 0.99)), 1e-8)
            for column, (plane, values) in enumerate(zip(planes, data)):
                axis = axes[mode, column]
                field = values[:, :, mode]
                shape = field.shape
                physical = (
                    np.hypot(plane["points"][:, 1], plane["points"][:, 2])
                    <= engine.cfg.farfield_radius
                ).reshape(shape)
                field = np.where(physical & ~plane["obstacle"], field, np.nan)
                image = axis.pcolormesh(
                    plane["first"],
                    plane["second"],
                    field.T,
                    shading="auto",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                obstacle = np.ma.masked_where(~plane["obstacle"], plane["obstacle"])
                axis.pcolormesh(
                    plane["first"],
                    plane["second"],
                    obstacle.T,
                    shading="auto",
                    cmap=matplotlib.colors.ListedColormap([GREY]),
                    vmin=0.0,
                    vmax=1.0,
                )
                axis.set_aspect("equal")
                axis.set_xlabel(plane["first_label"])
                axis.set_ylabel(plane["second_label"])
                if mode == 0:
                    axis.set_title(plane["label"])
                if column == 0:
                    axis.text(
                        -0.30,
                        0.5,
                        rf"Mode {mode + 1}\n$\lambda={selected_energies[mode]:.3g}$",
                        transform=axis.transAxes,
                        rotation=90,
                        va="center",
                        ha="center",
                        color=DARK_BLUE,
                    )
                figure.colorbar(image, ax=axis, shrink=0.72, pad=0.015)
        figure.suptitle(f"First six energy-ordered {quantity.lower()} (${symbol}$)", color=DARK_BLUE)
        save_pair(figure, output_dir / stem)

    np.savez_compressed(
        output_dir / "first_six_mode_slice_data.npz",
        selected_columns=selected,
        rayleigh=selected_energies,
        velocity_x_plane=velocity_data[0],
        velocity_y_plane=velocity_data[1],
        velocity_z_plane=velocity_data[2],
        vorticity_x_plane=vorticity_data[0],
        vorticity_y_plane=vorticity_data[1],
        vorticity_z_plane=vorticity_data[2],
        axis=reference.axis,
    )
    return {
        "selected_peeled_columns_zero_based": selected.tolist(),
        "rayleigh_values": selected_energies.tolist(),
    }


def parameter_design() -> list[tuple[str, np.ndarray]]:
    output = [("center", np.asarray([0.5, 0.5, 0.5], np.float32))]
    for beta in (0.0, 1.0):
        for chord in (0.0, 1.0):
            for thickness in (0.0, 1.0):
                output.append(
                    (
                        f"corner_{int(beta)}{int(chord)}{int(thickness)}",
                        np.asarray([beta, chord, thickness], np.float32),
                    )
                )
    return output


def evaluate_parameter_domain(
    engine: EvaluationEngine,
    point_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (label, parameters) in enumerate(parameter_design()):
        points = fluid_points(
            engine.cfg,
            point_count,
            parameters,
            engine.cfg.seed + 51_000_000 + index,
        )
        fields = raw_fields(engine.cfg, engine.model, points, parameters)
        result = peel_and_adjoint(
            engine.cfg,
            fields,
            len(points),
            need_adjoint=False,
            need_coefficients=False,
            diagnostics=True,
        )
        rows.append(
            {
                "case": label,
                "beta_normalized": float(parameters[0]),
                "chord_normalized": float(parameters[1]),
                "thickness_normalized": float(parameters[2]),
                "loss": float(result["loss"]),
                "orthogonality_error": float(result["orth_error"]),
                "effective_rank": int(result["effective_raw_rank"]),
                "min_rayleigh": float(result["min_energy"]),
                "max_rayleigh": float(result["max_energy"]),
            }
        )
        del fields, result
        cp.get_default_memory_pool().free_all_blocks()
        print(f"[parameter {index + 1:02d}/09] {label}", flush=True)
    return rows


def plot_training(history: list[dict[str, Any]], output_dir: Path) -> None:
    epochs = np.asarray([row["epoch"] for row in history])
    train = np.asarray([row["train_loss"] for row in history], dtype=float)
    validation_rows = [
        row for row in history if math.isfinite(float(row["validation_loss"]))
    ]
    validation_epochs = np.asarray([row["epoch"] for row in validation_rows])
    validation = np.asarray(
        [row["validation_loss"] for row in validation_rows], dtype=float
    )
    learning_rate = np.asarray([row["learning_rate"] for row in history])
    window = 31
    rolling = np.asarray(
        [
            np.median(train[max(0, index - window + 1) : index + 1])
            for index in range(len(train))
        ]
    )
    figure, left = plt.subplots(figsize=(8.3, 4.7), constrained_layout=True)
    left.semilogy(epochs, train, color=LIGHT_BLUE, linewidth=0.7, alpha=0.45, label="Train")
    left.semilogy(epochs, rolling, color=BLUE, linewidth=1.7, label="Train rolling median")
    left.semilogy(
        validation_epochs,
        validation,
        color=RED,
        linewidth=1.5,
        marker="o",
        markersize=2.6,
        label="Validation",
    )
    best_index = int(np.argmin(validation))
    left.scatter(
        [validation_epochs[best_index]],
        [validation[best_index]],
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        s=38,
        zorder=5,
    )
    left.annotate(
        f"best {validation[best_index]:.4f} @ {validation_epochs[best_index]}",
        (validation_epochs[best_index], validation[best_index]),
        xytext=(8, 10),
        textcoords="offset points",
        color=DARK_BLUE,
    )
    left.set_xlabel("Epoch")
    left.set_ylabel("Rayleigh loss")
    left.grid(True, which="both", alpha=0.75)
    right = left.twinx()
    right.step(epochs, learning_rate, where="post", color="#6A51A3", linewidth=1.2)
    right.set_yscale("log")
    right.set_ylabel("Learning rate", color="#6A51A3")
    handles, labels = left.get_legend_handles_labels()
    left.legend(handles, labels, loc="upper right")
    left.set_title("Training convergence and adaptive learning rate")
    save_pair(figure, output_dir / "training_convergence")


def plot_projection_convergence(
    convergence: list[dict[str, Any]], output_dir: Path
) -> None:
    modes = np.asarray([row["modes"] for row in convergence])
    fit = np.asarray([row["fit_disturbance_relative_l2"] for row in convergence])
    validation = np.asarray(
        [row["validation_disturbance_relative_l2"] for row in convergence]
    )
    full = np.asarray([row["validation_full_velocity_relative_l2"] for row in convergence])
    figure, axis = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    axis.semilogx(modes, fit, "o-", color=LIGHT_BLUE, label="Fit disturbance")
    axis.semilogx(modes, validation, "o-", color=RED, label="Validation disturbance")
    axis.semilogx(modes, full, "s-", color=BLUE, label="Validation full velocity")
    axis.set_xlabel("Number of leading SSS modes")
    axis.set_ylabel("Relative $L_2$ error")
    axis.set_xticks(modes)
    axis.set_xticklabels([str(value) for value in modes], rotation=35)
    axis.grid(True, which="both")
    axis.legend()
    axis.set_title("Stable rotating-fan reconstruction convergence")
    save_pair(figure, output_dir / "projection_convergence")


def plot_modal_coefficients(
    peeled_coefficients: np.ndarray,
    energies: np.ndarray,
    output_dir: Path,
) -> None:
    values = peeled_coefficients
    cumulative = np.cumsum(values * values) / max(float(np.sum(values * values)), 1e-30)
    ranks = np.arange(1, len(values) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    axes[0].semilogy(ranks, np.maximum(np.abs(values), 1e-16), color=BLUE, linewidth=1.0)
    axes[0].set_xlabel("SSS rank")
    axes[0].set_ylabel(r"$|a_k|$")
    axes[0].grid(True, which="both")
    axes[0].set_title("Projection coefficient magnitude")
    axes[1].plot(ranks, cumulative, color=RED, linewidth=1.6, label="Coefficient energy")
    axes[1].plot(
        ranks,
        np.cumsum(1.0 / np.maximum(energies, 1e-12))
        / np.sum(1.0 / np.maximum(energies, 1e-12)),
        color=LIGHT_BLUE,
        linewidth=1.2,
        label=r"Cumulative $1/\lambda_k$",
    )
    axes[1].set_xlabel("SSS rank")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_title("Cumulative modal content")
    save_pair(figure, output_dir / "modal_coefficients")


def plot_generalization(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [row["case"].replace("corner_", "") for row in rows]
    losses = np.asarray([row["loss"] for row in rows])
    figure, axis = plt.subplots(figsize=(8.1, 4.4), constrained_layout=True)
    colors = [BLUE] + [LIGHT_BLUE] * (len(rows) - 1)
    bars = axis.bar(
        np.arange(len(rows)), losses, color=colors, edgecolor="white", linewidth=0.7
    )
    axis.set_xticks(np.arange(len(rows)), labels)
    axis.set_ylabel("Scaled Rayleigh loss")
    axis.set_xlabel(r"Normalized parameter corner $(\beta,c,t)$")
    axis.set_yscale("log")
    maximum = int(np.argmax(losses))
    axis.annotate(
        f"{losses[maximum]:.3g}",
        (bars[maximum].get_x() + bars[maximum].get_width() / 2.0, losses[maximum]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        color=DARK_BLUE,
    )
    axis.grid(True, axis="y")
    axis.set_title("Independent parameter-domain basis audit")
    save_pair(figure, output_dir / "parameter_domain_audit")


def plot_summary(
    metrics: dict[str, Any],
    convergence: list[dict[str, Any]],
    generalization: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    modes = np.asarray([row["modes"] for row in convergence])
    axes[0, 0].semilogx(
        modes,
        [row["validation_disturbance_relative_l2"] for row in convergence],
        "o-",
        color=RED,
        label="Disturbance",
    )
    axes[0, 0].semilogx(
        modes,
        [row["validation_full_velocity_relative_l2"] for row in convergence],
        "s-",
        color=BLUE,
        label="Full velocity",
    )
    axes[0, 0].set_title("Reconstruction convergence")
    axes[0, 0].set_xlabel("Modes")
    axes[0, 0].set_ylabel("Relative $L_2$ error")
    axes[0, 0].grid(True, which="both")
    axes[0, 0].legend()

    region_names = ["near_rotor", "near_wake", "far_wake", "outer_field"]
    region_values = [
        metrics["reconstruction"]["regions"][name]["disturbance_relative_l2"]
        for name in region_names
    ]
    axes[0, 1].bar(
        np.arange(len(region_names)), region_values, color=[RED, LIGHT_RED, LIGHT_BLUE, BLUE]
    )
    axes[0, 1].set_xticks(np.arange(len(region_names)), ["Rotor", "Wake", "Far wake", "Outer"])
    axes[0, 1].set_ylabel("Disturbance relative $L_2$")
    axes[0, 1].set_title("Spatial error distribution")
    axes[0, 1].grid(True, axis="y")

    c4_values = [
        metrics["physics"]["gt_c4_relative_l2"],
        metrics["physics"]["reconstruction_c4_relative_l2"],
    ]
    c4_bars = axes[1, 0].bar([0, 1], c4_values, color=[GREY, BLUE])
    axes[1, 0].set_xticks([0, 1], ["GT", "Reconstruction"])
    axes[1, 0].set_ylabel("C4 relative defect")
    axes[1, 0].set_yscale("log")
    axes[1, 0].bar_label(c4_bars, labels=[f"{value:.2e}" for value in c4_values], padding=3)
    axes[1, 0].set_title("Quarter-turn symmetry")
    axes[1, 0].grid(True, axis="y")

    losses = np.asarray([row["loss"] for row in generalization])
    parameter_bars = axes[1, 1].bar(
        np.arange(len(losses)), losses, color=[BLUE] + [LIGHT_BLUE] * 8
    )
    axes[1, 1].set_xticks(np.arange(len(losses)), ["C"] + [str(i) for i in range(8)])
    axes[1, 1].set_ylabel("Scaled Rayleigh loss")
    axes[1, 1].set_yscale("log")
    maximum = int(np.argmax(losses))
    axes[1, 1].annotate(
        f"{losses[maximum]:.3g}",
        (
            parameter_bars[maximum].get_x()
            + parameter_bars[maximum].get_width() / 2.0,
            losses[maximum],
        ),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        color=DARK_BLUE,
    )
    axes[1, 1].set_title("Center and eight parameter corners")
    axes[1, 1].grid(True, axis="y")
    figure.suptitle("3D fan Eigenfluid: complete reconstruction audit", color=DARK_BLUE)
    save_pair(figure, output_dir / "evaluation_summary")


def write_report(metrics: dict[str, Any], output_dir: Path) -> None:
    rec = metrics["reconstruction"]
    physics = metrics["physics"]
    gt = metrics["ground_truth"]
    text = f"""# 3D parametric fan: complete reconstruction evaluation

## Checkpoint and reference

- Best checkpoint epoch: {metrics['checkpoint']['epoch']}
- Best validation loss: {metrics['training']['best_validation_loss']:.8g}
- Independent GT: {gt['solver']}, Re={gt['reynolds']:.0f}, grid {gt['resolution']}^3
- GT phase-periodic residual: {gt['periodic_relative_l2']:.6g}
- GT grid-independent at 5%: {gt['grid_independent_at_five_percent']}

## All-1024-mode projection

- Fit disturbance relative L2: {rec['fit_disturbance_relative_l2']:.6g}
- Validation disturbance relative L2: {rec['validation_disturbance_relative_l2']:.6g}
- Validation full-velocity relative L2: {rec['validation_full_velocity_relative_l2']:.6g}
- Near-rotor disturbance relative L2: {rec['regions']['near_rotor']['disturbance_relative_l2']:.6g}
- Near-wake disturbance relative L2: {rec['regions']['near_wake']['disturbance_relative_l2']:.6g}
- Far-wake disturbance relative L2: {rec['regions']['far_wake']['disturbance_relative_l2']:.6g}

## Physical diagnostics

- Divergence L2 / gradient L2: {physics['divergence_l2_over_gradient_l2']:.6g}
- GT C4 symmetry defect: {physics['gt_c4_relative_l2']:.6g}
- Reconstruction C4 symmetry defect: {physics['reconstruction_c4_relative_l2']:.6g}
- Inlet disturbance RMS: {physics['inlet_disturbance_rms']:.6g}
- Lateral far-field disturbance RMS: {physics['lateral_farfield_disturbance_rms']:.6g}

## Interpretation

The trained span is divergence-free and satisfies the homogeneous disturbance
boundary construction, but the numerical GT uses a genuinely moving no-slip
rotor.  Only the uniform inlet was lifted before projection; the non-homogeneous
rotor-wall velocity was not lifted.  Therefore the near-rotor mismatch is a
boundary-data incompatibility as well as a finite-dimensional approximation
error.  In addition, the 96^3 GT is phase-periodic but not grid-independent, so
the reported projection errors should not be interpreted below the documented
GT discretization uncertainty.
"""
    (output_dir / "evaluation_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_plotting()
    # Evaluation is restart-safe: completed plots are overwritten atomically by
    # Matplotlib and numerical archives are written only after all audits pass.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    engine = EvaluationEngine(args.checkpoint.resolve())
    reference = ReferenceField(args.reference.resolve())
    gt_protocol, gt_grid = load_gt_protocol(args.reference.resolve())
    parameters = engine.normalize(23.0, 0.28, 0.14)

    run_root = args.checkpoint.resolve().parents[1]
    history = json.loads(
        (run_root / "data" / "training_history.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (run_root / "completion.json").read_text(encoding="utf-8")
    )
    plot_training(history, args.output_dir)

    fit_points = fluid_points(
        engine.cfg,
        args.fit_points,
        parameters,
        engine.cfg.seed + 31_000_000,
    )
    fit_target = engine.reference_residual(reference, fit_points)
    gram, right, target_mass, fit_diagnostics = fit_raw_projection(
        engine, fit_points, fit_target, parameters, args.chunk
    )
    peeling, energies, peeling_diagnostics = peel_basis(engine, parameters)
    mode_counts = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024]
    raw_family, peeled_family, convergence = convergence_projections(
        gram, right, target_mass, peeling, mode_counts
    )

    validation_points = fluid_points(
        engine.cfg,
        args.validation_points,
        parameters,
        engine.cfg.seed + 32_000_000,
    )
    validation_target = engine.reference_residual(reference, validation_points)
    validation_reconstruction, disturbance_errors, full_errors = (
        evaluate_projection_family(
            engine,
            validation_points,
            validation_target,
            parameters,
            raw_family,
            args.chunk,
        )
    )
    for index, row in enumerate(convergence):
        row["validation_disturbance_relative_l2"] = float(disturbance_errors[index])
        row["validation_full_velocity_relative_l2"] = float(full_errors[index])
    final_raw = raw_family[:, -1:]
    final_peeled = peeled_family[:, -1]
    final_reconstruction = validation_reconstruction[:, -1]
    regions = region_metrics(
        engine,
        validation_points,
        validation_target,
        final_reconstruction,
        engine.cfg,
    )
    physics = reconstruction_physics(
        engine,
        validation_points,
        parameters,
        final_raw,
        final_reconstruction,
        reference,
        args.chunk,
    )
    slice_metrics = plot_comparison_planes(
        engine,
        reference,
        parameters,
        final_raw,
        args.chunk,
        args.output_dir,
    )
    mode_metrics = plot_first_six_modes(
        engine,
        reference,
        parameters,
        peeling,
        energies,
        args.chunk,
        args.output_dir,
    )
    generalization = evaluate_parameter_domain(
        engine, args.generalization_points
    )

    plot_projection_convergence(convergence, args.output_dir)
    plot_modal_coefficients(cp.asnumpy(final_peeled), energies, args.output_dir)
    plot_generalization(generalization, args.output_dir)
    with (args.output_dir / "projection_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(convergence[0].keys()))
        writer.writeheader()
        writer.writerows(convergence)
    with (args.output_dir / "parameter_domain_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(generalization[0].keys()))
        writer.writeheader()
        writer.writerows(generalization)

    transform_residual = float(
        (
            cp.linalg.norm(peeling @ final_peeled - final_raw[:, 0])
            / cp.maximum(cp.linalg.norm(final_raw[:, 0]), 1e-30)
        ).get()
    )
    validation_rows = [
        row for row in history if math.isfinite(float(row["validation_loss"]))
    ]
    learning_rate_events = []
    previous_lr = float(history[0]["learning_rate"])
    for row in history[1:]:
        value = float(row["learning_rate"])
        if value != previous_lr:
            learning_rate_events.append(
                {"epoch": int(row["epoch"]), "learning_rate": value}
            )
            previous_lr = value
    fine_comparison = gt_grid["comparisons"][-1]["regions"]
    metrics = {
        "status": "completed",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": int(engine.metadata["epoch"]),
            "modes": int(engine.cfg.modes),
            "backend": "manual CuPy analytic jet; FP16 forward, FP32 reverse, FP64 SSS",
            "rotation_prior": getattr(engine.cfg, "symmetry_prior", "none"),
        },
        "training": {
            "epochs_completed": int(completion["epochs_completed"]),
            "stopped_early": bool(completion["stopped_early"]),
            "best_validation_loss": float(completion["best_validation"]),
            "best_validation_epoch": int(completion["best_validation_epoch"]),
            "final_validation_loss": float(validation_rows[-1]["validation_loss"]),
            "learning_rate_events": learning_rate_events,
            "wall_seconds": float(completion["wall_seconds"]),
        },
        "ground_truth": {
            "path": str(args.reference.resolve()),
            "solver": gt_protocol["spatial_solver"],
            "equations": gt_protocol["equations"],
            "reynolds": 500.0,
            "resolution": len(reference.axis),
            "moving_wall": gt_protocol["rotor_boundary"],
            "periodic_relative_l2": float(gt_protocol["final_periodic_relative_l2"]),
            "maximum_divergence": float(gt_protocol["maximum_divergence"]),
            "grid_independent_at_five_percent": bool(
                gt_grid["grid_independent_at_five_percent"]
            ),
            "n80_to_n96_core_full_velocity_relative_l2": float(
                fine_comparison["core"]["full_velocity_relative_l2"]
            ),
            "n80_to_n96_core_disturbance_relative_l2": float(
                fine_comparison["core"]["disturbance_relative_l2"]
            ),
        },
        "parameters_physical": {
            "beta_tip_deg": 23.0,
            "chord_tip": 0.28,
            "thickness_ratio": 0.14,
        },
        "parameters_normalized": parameters.tolist(),
        "reconstruction": {
            "projection_target": (
                "u_GT - u_moving_wall_lift"
                if bool(getattr(engine.cfg, "moving_wall_lifting", False))
                else "u_GT - U_infinity e_x"
            ),
            "fit_points": len(fit_points),
            "validation_points": len(validation_points),
            **fit_diagnostics,
            "validation_disturbance_relative_l2": float(disturbance_errors[-1]),
            "validation_full_velocity_relative_l2": float(full_errors[-1]),
            "regions": regions,
            "slices": slice_metrics,
            "convergence": convergence,
            "raw_to_peeled_transform_relative_residual": transform_residual,
        },
        "basis": {
            "peeling": peeling_diagnostics,
            "first_six": mode_metrics,
            "parameter_domain": {
                "points_per_case": args.generalization_points,
                "cases": generalization,
                "loss_mean": float(np.mean([row["loss"] for row in generalization])),
                "loss_max": float(np.max([row["loss"] for row in generalization])),
                "max_orthogonality_error": float(
                    np.max([row["orthogonality_error"] for row in generalization])
                ),
                "minimum_effective_rank": int(
                    np.min([row["effective_rank"] for row in generalization])
                ),
            },
        },
        "physics": physics,
        "interpretation": {
            "boundary_compatibility": (
                "exact analytic moving-wall/inflow lifting is subtracted before "
                "projection; learned residual modes have homogeneous rotor and "
                "far-field data"
                if bool(getattr(engine.cfg, "moving_wall_lifting", False))
                else "only the uniform inflow lifting was applied"
            ),
            "gt_limitation": (
                "phase-periodic reference is not grid-independent at the five-percent level"
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    plot_summary(metrics, convergence, generalization, args.output_dir)
    np.savez_compressed(
        args.output_dir / "projection_coefficients.npz",
        raw_coefficients=cp.asnumpy(final_raw[:, 0]),
        peeled_coefficients=cp.asnumpy(final_peeled),
        peeling_matrix=cp.asnumpy(peeling),
        rayleigh_energies=energies,
        parameters_normalized=parameters,
    )
    np.savez_compressed(
        args.output_dir / "validation_reconstruction.npz",
        points=validation_points,
        gt_disturbance=validation_target,
        reconstruction_disturbance=final_reconstruction,
        lifting_velocity=engine.lifting_fields(validation_points)["velocity"],
    )
    atomic_json(args.output_dir / "metrics.json", metrics)
    write_report(metrics, args.output_dir)
    print(json.dumps(metrics, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
