"""Fair numerical reconstruction comparison on one unseen 3D metaball.

The learned and finite-difference Laplacian bases use the same geometry, fluid
mask, vector-potential Dirichlet condition, grid quadrature, target, and
train/test split.  Only the geometry-conditioned, twice-MGS-peeled learned
basis is reported; raw-basis ranks and raw-basis metrics are intentionally not
serialized.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
TRAINING_SCRIPT = CASE_ROOT / "scripts" / "train_validate.py"
DEFAULT_CHECKPOINT = (
    CASE_ROOT
    / "results"
    / "k64_siren_dirichlet_e500"
    / "run_001"
    / "checkpoints"
    / "final.pt"
)
DEFAULT_OUTPUT = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction"
COMPONENT_NAMES = ("A12", "A13", "A23")


def load_training_module():
    spec = importlib.util.spec_from_file_location("connected_metaball_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training implementation from {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = load_training_module()


@dataclass(frozen=True)
class EvaluationConfig:
    geometry_token: int = 12_000_001
    grid_resolution: int = 34
    target_seed: int = 20_260_824
    fit_fraction: float = 0.70
    ridge_relative: float = 1.0e-8
    learned_batch_size: int = 4096
    laplacian_tolerance: float = 1.0e-7
    reconstruction_prefixes: tuple[int, ...] = (4, 8, 16, 32, 64)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_grid(cfg: Any, resolution: int, geometry: np.ndarray) -> dict[str, Any]:
    coordinates = np.linspace(0.0, 1.0, resolution, dtype=np.float64)
    z_grid, y_grid, x_grid = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )
    all_points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    mask_flat = training.fluid_mask(cfg, all_points, geometry)
    mask = mask_flat.reshape((resolution, resolution, resolution))
    valid_points = all_points[mask_flat].astype(np.float32)
    return {
        "coordinates": coordinates,
        "spacing": float(coordinates[1] - coordinates[0]),
        "shape": mask.shape,
        "mask": mask,
        "mask_flat": mask_flat,
        "points": valid_points,
    }


def assemble_dirichlet_laplacian(mask: np.ndarray, spacing: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """Seven-point -Delta with zero values on every missing neighbor."""
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[mask] = np.arange(np.count_nonzero(mask), dtype=np.int64)
    count = int(np.count_nonzero(mask))
    rows: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    cols: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    data: list[np.ndarray] = [np.full(count, 6.0 / spacing**2, dtype=np.float64)]
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        pair_mask = mask[tuple(lower)] & mask[tuple(upper)]
        left = index[tuple(lower)][pair_mask]
        right = index[tuple(upper)][pair_mask]
        coefficient = np.full(len(left), -1.0 / spacing**2, dtype=np.float64)
        rows.extend((left, right))
        cols.extend((right, left))
        data.extend((coefficient, coefficient.copy()))
    matrix = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(count, count),
    ).tocsr()
    return matrix, index


def solve_scalar_laplacian(
    matrix: sp.csr_matrix,
    scalar_modes: int,
    spacing: float,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    started = time.perf_counter()
    eigenvalues, eigenvectors = spla.eigsh(
        matrix,
        k=scalar_modes,
        sigma=0.0,
        which="LM",
        tol=tolerance,
        maxiter=20_000,
    )
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eigenvectors /= np.sqrt(
        np.sum(eigenvectors * eigenvectors, axis=0, keepdims=True) * spacing**3
    )
    residuals = np.linalg.norm(
        matrix @ eigenvectors - eigenvectors * eigenvalues[None, :], axis=0
    ) / np.maximum(np.linalg.norm(eigenvectors * eigenvalues[None, :], axis=0), 1.0e-15)
    print(
        f"laplacian solved: nodes={matrix.shape[0]} scalar_modes={scalar_modes} "
        f"seconds={time.perf_counter() - started:.1f} max_residual={residuals.max():.3e}",
        flush=True,
    )
    return eigenvalues, eigenvectors, residuals


def load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location=training.DEVICE, weights_only=False)
    cfg = training.Config(**payload["config"])
    training.validate_contract(cfg)
    if int(payload["epoch"]) != cfg.epochs:
        raise ValueError(
            f"checkpoint is epoch {payload['epoch']}, but the requested campaign has {cfg.epochs} epochs"
        )
    model = training.BasisModel(cfg).to(training.DEVICE)
    model.load_state_dict(payload["model"])
    model.eval()
    return cfg, model, payload


def evaluate_raw_learned_basis(
    cfg: Any,
    model: Any,
    points: np.ndarray,
    geometry: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(points)
    raw_potential = np.empty((count, 3, cfg.modes), dtype=np.float32)
    raw_gradient = np.empty((count, 3, 3, cfg.modes), dtype=np.float32)
    geometry_tensor = torch.as_tensor(
        geometry, device=training.DEVICE, dtype=training.DTYPE
    )
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            point_tensor = torch.as_tensor(
                points[start:stop], device=training.DEVICE, dtype=training.DTYPE
            )
            features, feature_gradient = training.normalized_features(
                cfg, point_tensor, geometry_tensor
            )
            envelope, envelope_gradient = training.envelope_and_gradient(
                cfg, point_tensor, geometry_tensor
            )
            for mode in range(cfg.modes):
                potential, gradient = training.mode_field(
                    cfg,
                    model,
                    mode,
                    features,
                    feature_gradient,
                    envelope,
                    envelope_gradient,
                )
                raw_potential[start:stop, :, mode] = (
                    potential.reshape(stop - start, 3).cpu().numpy()
                )
                raw_gradient[start:stop, :, :, mode] = (
                    gradient.reshape(stop - start, 3, 3).cpu().numpy()
                )
            print(f"learned basis evaluated: {stop}/{count} grid nodes", flush=True)
    return raw_potential.astype(np.float64), raw_gradient.astype(np.float64)


def peel_learned_basis(
    raw_potential: np.ndarray,
    raw_gradient: np.ndarray,
    weight: float,
    passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Stable Rayleigh sorting and two-pass MGS on the shared grid."""
    count, _, modes = raw_potential.shape
    potential_matrix = raw_potential.reshape(count * 3, modes)
    gradient_matrix = raw_gradient.reshape(count * 9, modes)
    sorting_rayleigh = np.sum(gradient_matrix**2, axis=0) / np.maximum(
        np.sum(potential_matrix**2, axis=0), 1.0e-30
    )
    order = np.argsort(sorting_rayleigh, kind="stable")

    basis = np.empty_like(potential_matrix)
    transform = np.zeros((modes, modes), dtype=np.float64)
    residual_masses = np.empty(modes, dtype=np.float64)
    for rank, source_mode in enumerate(order):
        residual = potential_matrix[:, source_mode].copy()
        coefficients = np.zeros(modes, dtype=np.float64)
        coefficients[source_mode] = 1.0
        if rank:
            previous_basis = basis[:, :rank]
            previous_transform = transform[:, :rank]
            for _ in range(passes):
                projection = weight * (previous_basis.T @ residual)
                residual -= previous_basis @ projection
                coefficients -= previous_transform @ projection
        mass = weight * float(residual @ residual)
        if not np.isfinite(mass) or mass <= 1.0e-14:
            raise FloatingPointError(f"unresolved peeled learned mode at rank {rank + 1}")
        scale = 1.0 / np.sqrt(mass)
        basis[:, rank] = residual * scale
        transform[:, rank] = coefficients * scale
        residual_masses[rank] = mass

    peeled_potential = (potential_matrix @ transform).reshape(count, 3, modes)
    peeled_gradient = (gradient_matrix @ transform).reshape(count, 3, 3, modes)
    gram = weight * (basis.T @ basis)
    gram_eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    threshold = np.finfo(np.float64).eps * max(basis.shape) * max(
        abs(float(gram_eigenvalues[-1])), 1.0e-14
    )
    peeled_rank = int(np.count_nonzero(gram_eigenvalues > threshold))
    rayleigh = np.sum(peeled_gradient.reshape(count * 9, modes) ** 2, axis=0) / np.maximum(
        np.sum(peeled_potential.reshape(count * 3, modes) ** 2, axis=0), 1.0e-30
    )
    metrics = {
        "reporting_scope": "peeled_basis_only",
        "peeled_effective_rank": peeled_rank,
        "peeled_orthogonality_max_abs": float(
            np.max(np.abs(gram - np.eye(modes, dtype=np.float64)))
        ),
        "peeled_gram_eigenvalue_min": float(gram_eigenvalues[0]),
        "peeled_gram_eigenvalue_max": float(gram_eigenvalues[-1]),
        "peeled_residual_mass_min": float(np.min(residual_masses)),
        "peeled_residual_mass_max": float(np.max(residual_masses)),
        "causal_order_1based": [int(index) + 1 for index in order],
        "reorthogonalization_passes": passes,
    }
    return peeled_potential, peeled_gradient, rayleigh, metrics


def scalar_gradient_on_mask(
    values: np.ndarray,
    mask: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Use the same zero-Dirichlet extension as the discrete Laplacian."""
    scalar_modes = values.shape[1]
    gradient = np.empty((values.shape[0], 3, scalar_modes), dtype=np.float64)
    for mode in range(scalar_modes):
        volume = np.zeros(mask.shape, dtype=np.float64)
        volume[mask] = values[:, mode]
        derivative_z, derivative_y, derivative_x = np.gradient(
            volume, spacing, edge_order=2
        )
        gradient[:, 0, mode] = derivative_x[mask]
        gradient[:, 1, mode] = derivative_y[mask]
        gradient[:, 2, mode] = derivative_z[mask]
    return gradient


def vector_potential_gradient_on_mask(
    potential: np.ndarray,
    mask: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Apply one common zero-extension difference operator to every component/mode."""
    if potential.ndim == 2:
        potential = potential[:, :, None]
        squeeze_mode = True
    elif potential.ndim == 3:
        squeeze_mode = False
    else:
        raise ValueError("potential must have shape [nodes,3] or [nodes,3,modes]")
    count, components, modes = potential.shape
    if components != 3:
        raise ValueError("the vector potential must have three independent components")
    merged = potential.reshape(count, components * modes)
    derivative_first = scalar_gradient_on_mask(merged, mask, spacing)
    gradient = derivative_first.reshape(count, 3, components, modes).transpose(0, 2, 1, 3)
    return gradient[:, :, :, 0] if squeeze_mode else gradient


def expand_laplacian_basis(
    scalar_values: np.ndarray,
    scalar_gradient: np.ndarray,
    scalar_eigenvalues: np.ndarray,
    modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    count = scalar_values.shape[0]
    potential = np.zeros((count, 3, modes), dtype=np.float64)
    gradient = np.zeros((count, 3, 3, modes), dtype=np.float64)
    rayleigh = np.empty(modes, dtype=np.float64)
    labels: list[dict[str, Any]] = []
    for mode in range(modes):
        scalar_mode = mode // 3
        component = mode % 3
        potential[:, component, mode] = scalar_values[:, scalar_mode]
        gradient[:, component, :, mode] = scalar_gradient[:, :, scalar_mode]
        rayleigh[mode] = scalar_eigenvalues[scalar_mode]
        labels.append(
            {
                "mode_1based": mode + 1,
                "scalar_mode_1based": scalar_mode + 1,
                "component": COMPONENT_NAMES[component],
                "rayleigh": float(rayleigh[mode]),
            }
        )
    return potential, gradient, rayleigh, labels


def target_potential_and_gradient(
    cfg: Any,
    points: np.ndarray,
    geometry: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Independent smooth target: random quadratic polynomial times the envelope."""
    rng = np.random.default_rng(seed)
    centered = 2.0 * points.astype(np.float64) - 1.0
    x, y, z = centered.T
    features = np.column_stack(
        (
            np.ones(len(points)),
            x,
            y,
            z,
            x * y,
            x * z,
            y * z,
            x * x,
            y * y,
            z * z,
        )
    )
    feature_gradient = np.zeros((len(points), 10, 3), dtype=np.float64)
    feature_gradient[:, 1, 0] = 2.0
    feature_gradient[:, 2, 1] = 2.0
    feature_gradient[:, 3, 2] = 2.0
    feature_gradient[:, 4, 0] = 2.0 * y
    feature_gradient[:, 4, 1] = 2.0 * x
    feature_gradient[:, 5, 0] = 2.0 * z
    feature_gradient[:, 5, 2] = 2.0 * x
    feature_gradient[:, 6, 1] = 2.0 * z
    feature_gradient[:, 6, 2] = 2.0 * y
    feature_gradient[:, 7, 0] = 4.0 * x
    feature_gradient[:, 8, 1] = 4.0 * y
    feature_gradient[:, 9, 2] = 4.0 * z
    degree_scale = np.asarray([1.0, 0.7, 0.7, 0.7, 0.4, 0.4, 0.4, 0.25, 0.25, 0.25])
    coefficients = rng.normal(size=(10, 3)) * degree_scale[:, None]
    raw = features @ coefficients
    raw_gradient = np.einsum("ntd,tc->ncd", feature_gradient, coefficients)

    point_tensor = torch.as_tensor(points, device=training.DEVICE, dtype=training.DTYPE)
    geometry_tensor = torch.as_tensor(
        geometry, device=training.DEVICE, dtype=training.DTYPE
    )
    with torch.inference_mode():
        envelope, envelope_gradient = training.envelope_and_gradient(
            cfg, point_tensor, geometry_tensor
        )
    envelope_numpy = envelope.cpu().numpy().astype(np.float64)
    envelope_gradient_numpy = envelope_gradient.cpu().numpy().astype(np.float64)
    potential = envelope_numpy[:, None] * raw
    gradient = (
        envelope_numpy[:, None, None] * raw_gradient
        + raw[:, :, None] * envelope_gradient_numpy[:, None, :]
    )
    specification = {
        "kind": "independent_enveloped_random_quadratic_vector_potential",
        "seed": seed,
        "polynomial_degree": 2,
        "raw_feature_count": 10,
        "uses_same_hard_dirichlet_envelope_as_learned_basis": True,
        "constructed_without_learned_or_laplacian_modes": True,
    }
    return potential, gradient, specification


def velocity_from_potential_gradient(gradient: np.ndarray) -> np.ndarray:
    """Divergence of antisymmetric A=(A12,A13,A23); supports an optional mode axis."""
    velocity = np.empty((gradient.shape[0], 3, *gradient.shape[3:]), dtype=gradient.dtype)
    velocity[:, 0] = gradient[:, 0, 1] + gradient[:, 1, 2]
    velocity[:, 1] = -gradient[:, 0, 0] + gradient[:, 2, 2]
    velocity[:, 2] = -gradient[:, 1, 0] - gradient[:, 2, 1]
    return velocity


def relative_l2(reference: np.ndarray, approximation: np.ndarray) -> float:
    numerator = float(np.sum((approximation - reference) ** 2))
    denominator = float(np.sum(reference**2))
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def fit_reconstruction(
    potential_basis: np.ndarray,
    velocity_basis: np.ndarray,
    target_potential: np.ndarray,
    target_velocity: np.ndarray,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    modes: int,
    ridge_relative: float,
) -> dict[str, Any]:
    coefficients, ridge = ridge_fit_coefficients(
        potential_basis,
        target_potential,
        fit_indices,
        modes,
        ridge_relative,
    )
    fit_potential = np.einsum(
        "nck,k->nc", potential_basis[fit_indices, :, :modes], coefficients
    )
    test_potential = np.einsum(
        "nck,k->nc", potential_basis[test_indices, :, :modes], coefficients
    )
    fit_velocity = np.einsum(
        "nck,k->nc", velocity_basis[fit_indices, :, :modes], coefficients
    )
    test_velocity = np.einsum(
        "nck,k->nc", velocity_basis[test_indices, :, :modes], coefficients
    )
    return {
        "modes": modes,
        "ridge_absolute": ridge,
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "fit_potential_relative_l2": relative_l2(
            target_potential[fit_indices], fit_potential
        ),
        "test_potential_relative_l2": relative_l2(
            target_potential[test_indices], test_potential
        ),
        "fit_velocity_relative_l2": relative_l2(
            target_velocity[fit_indices], fit_velocity
        ),
        "test_velocity_relative_l2": relative_l2(
            target_velocity[test_indices], test_velocity
        ),
    }


def ridge_fit_coefficients(
    potential_basis: np.ndarray,
    target_potential: np.ndarray,
    fit_indices: np.ndarray,
    modes: int,
    ridge_relative: float,
) -> tuple[np.ndarray, float]:
    design = potential_basis[fit_indices, :, :modes].reshape(-1, modes)
    target = target_potential[fit_indices].reshape(-1)
    normal = design.T @ design
    ridge = ridge_relative * float(np.trace(normal)) / modes
    coefficients = np.linalg.solve(
        normal + ridge * np.eye(modes, dtype=np.float64), design.T @ target
    )
    return coefficients, ridge


def laplacian_orthogonality(
    potential: np.ndarray, weight: float
) -> dict[str, float | int]:
    matrix = potential.reshape(-1, potential.shape[-1])
    gram = weight * (matrix.T @ matrix)
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    threshold = np.finfo(np.float64).eps * max(matrix.shape) * max(
        abs(float(eigenvalues[-1])), 1.0e-14
    )
    return {
        "effective_rank": int(np.count_nonzero(eigenvalues > threshold)),
        "orthogonality_max_abs": float(
            np.max(np.abs(gram - np.eye(potential.shape[-1])))
        ),
        "gram_eigenvalue_min": float(eigenvalues[0]),
        "gram_eigenvalue_max": float(eigenvalues[-1]),
    }


def write_reconstruction_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def evaluate(checkpoint: Path, output_dir: Path, evaluation: EvaluationConfig) -> dict[str, Any]:
    wall_started = time.perf_counter()
    cfg, model, checkpoint_payload = load_model(checkpoint)
    if cfg.modes != 64:
        raise ValueError(f"expected K=64 checkpoint, got K={cfg.modes}")
    if evaluation.geometry_token <= cfg.epochs or (
        10_000_000 <= evaluation.geometry_token < 10_000_000 + cfg.validation_geometries
    ):
        raise ValueError("geometry token overlaps the training or validation campaign")

    geometry = training.geometry_from_token(cfg, evaluation.geometry_token)
    grid = build_grid(cfg, evaluation.grid_resolution, geometry)
    points = grid["points"]
    spacing = grid["spacing"]
    quadrature_weight = spacing**3
    print(
        f"unseen geometry token={evaluation.geometry_token} resolution={evaluation.grid_resolution} "
        f"fluid_nodes={len(points)} device={training.DEVICE}",
        flush=True,
    )

    laplacian, _ = assemble_dirichlet_laplacian(grid["mask"], spacing)
    scalar_mode_count = int(np.ceil(cfg.modes / 3))
    scalar_eigenvalues, scalar_eigenvectors, scalar_residuals = solve_scalar_laplacian(
        laplacian,
        scalar_mode_count,
        spacing,
        evaluation.laplacian_tolerance,
    )
    scalar_gradient = scalar_gradient_on_mask(
        scalar_eigenvectors, grid["mask"], spacing
    )
    lap_potential, lap_gradient, lap_rayleigh, lap_labels = expand_laplacian_basis(
        scalar_eigenvectors,
        scalar_gradient,
        scalar_eigenvalues,
        cfg.modes,
    )

    raw_potential, raw_gradient = evaluate_raw_learned_basis(
        cfg,
        model,
        points,
        geometry,
        evaluation.learned_batch_size,
    )
    learned_potential, learned_gradient, learned_rayleigh, learned_metrics = (
        peel_learned_basis(
            raw_potential,
            raw_gradient,
            quadrature_weight,
            cfg.reorthogonalization_passes,
        )
    )
    del raw_potential, raw_gradient

    target_potential, _, target_specification = target_potential_and_gradient(
        cfg, points, geometry, evaluation.target_seed
    )
    target_discrete_gradient = vector_potential_gradient_on_mask(
        target_potential, grid["mask"], spacing
    )
    learned_discrete_gradient = vector_potential_gradient_on_mask(
        learned_potential, grid["mask"], spacing
    )
    lap_discrete_gradient = vector_potential_gradient_on_mask(
        lap_potential, grid["mask"], spacing
    )
    target_velocity = velocity_from_potential_gradient(target_discrete_gradient)
    learned_velocity = velocity_from_potential_gradient(learned_discrete_gradient)
    lap_velocity = velocity_from_potential_gradient(lap_discrete_gradient)

    rng = np.random.default_rng(evaluation.target_seed + 1)
    permutation = rng.permutation(len(points))
    fit_count = int(round(evaluation.fit_fraction * len(points)))
    fit_indices = permutation[:fit_count]
    test_indices = permutation[fit_count:]
    curve: list[dict[str, Any]] = []
    for modes in evaluation.reconstruction_prefixes:
        learned_row = fit_reconstruction(
            learned_potential,
            learned_velocity,
            target_potential,
            target_velocity,
            fit_indices,
            test_indices,
            modes,
            evaluation.ridge_relative,
        )
        lap_row = fit_reconstruction(
            lap_potential,
            lap_velocity,
            target_potential,
            target_velocity,
            fit_indices,
            test_indices,
            modes,
            evaluation.ridge_relative,
        )
        curve.append({"basis": "learned_peeled", **learned_row})
        curve.append({"basis": "numerical_dirichlet_laplacian", **lap_row})
        print(
            f"K={modes:2d} test potential learned={learned_row['test_potential_relative_l2']:.6f} "
            f"lap={lap_row['test_potential_relative_l2']:.6f}; "
            f"velocity learned={learned_row['test_velocity_relative_l2']:.6f} "
            f"lap={lap_row['test_velocity_relative_l2']:.6f}",
            flush=True,
        )

    learned_k64 = next(
        row for row in curve if row["basis"] == "learned_peeled" and row["modes"] == 64
    )
    lap_k64 = next(
        row
        for row in curve
        if row["basis"] == "numerical_dirichlet_laplacian" and row["modes"] == 64
    )
    result = {
        "complete": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "device": str(training.DEVICE),
        "evaluation": asdict(evaluation),
        "fairness_contract": {
            "same_unseen_geometry": True,
            "same_outer_boundary": "rounded cube level set from training config",
            "same_obstacle_boundary": "three-sphere soft-min level set, tau=0.02",
            "same_boundary_condition": "homogeneous Dirichlet on A12, A13, A23",
            "same_valid_grid_nodes": True,
            "same_grid_quadrature": True,
            "same_target": True,
            "same_fit_test_split": True,
            "same_mode_count": cfg.modes,
            "same_reconstruction_velocity_operator": (
                "centered finite difference of the zero-Dirichlet-extended vector potential"
            ),
            "laplacian_boundary_discretization": (
                "seven-point finite difference on the identical fluid mask; every missing "
                "neighbor is a zero-Dirichlet neighbor"
            ),
            "learned_peeling_quadrature": "the identical valid grid nodes and h^3 weights",
            "degenerate_block_direction_alignment": False,
        },
        "geometry": {
            "token": evaluation.geometry_token,
            "sphere_parameters_xyzr": geometry.tolist(),
            "connected": bool(training.overlap_graph_connected(geometry, cfg.minimum_overlap)),
        },
        "grid": {
            "resolution_per_axis": evaluation.grid_resolution,
            "spacing": spacing,
            "fluid_node_count": len(points),
            "fluid_mask_fraction": float(len(points) / evaluation.grid_resolution**3),
            "quadrature_weight": quadrature_weight,
            "note": "curved boundaries are represented by the reported finite grid mask",
        },
        "target": target_specification,
        "reconstruction_velocity_operator": (
            "common centered finite difference on the shared grid with zero Dirichlet extension"
        ),
        "split": {
            "fit_nodes": len(fit_indices),
            "test_nodes": len(test_indices),
            "fit_fraction_realized": float(len(fit_indices) / len(points)),
        },
        "learned_peeled_basis": {
            **learned_metrics,
            "rayleigh_first16": learned_rayleigh[:16].tolist(),
            "rayleigh_k64_min": float(np.min(learned_rayleigh)),
            "rayleigh_k64_mean": float(np.mean(learned_rayleigh)),
            "rayleigh_k64_max": float(np.max(learned_rayleigh)),
        },
        "numerical_dirichlet_laplacian_basis": {
            **laplacian_orthogonality(lap_potential, quadrature_weight),
            "scalar_modes_solved": scalar_mode_count,
            "scalar_eigenvalues": scalar_eigenvalues.tolist(),
            "scalar_relative_residual_max": float(np.max(scalar_residuals)),
            "expanded_mode_metadata_first16": lap_labels[:16],
            "rayleigh_first16": lap_rayleigh[:16].tolist(),
            "rayleigh_k64_min": float(np.min(lap_rayleigh)),
            "rayleigh_k64_mean": float(np.mean(lap_rayleigh)),
            "rayleigh_k64_max": float(np.max(lap_rayleigh)),
        },
        "reconstruction_curve": curve,
        "k64_reconstruction": {
            "learned_peeled": learned_k64,
            "numerical_dirichlet_laplacian": lap_k64,
            "test_error_ratio_learned_over_laplacian": {
                "potential": float(
                    learned_k64["test_potential_relative_l2"]
                    / lap_k64["test_potential_relative_l2"]
                ),
                "velocity": float(
                    learned_k64["test_velocity_relative_l2"]
                    / lap_k64["test_velocity_relative_l2"]
                ),
            },
        },
        "reporting_scope": "peeled_basis_only",
        "wall_seconds": float(time.perf_counter() - wall_started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    learned_coefficients, _ = ridge_fit_coefficients(
        learned_potential,
        target_potential,
        fit_indices,
        cfg.modes,
        evaluation.ridge_relative,
    )
    lap_coefficients, _ = ridge_fit_coefficients(
        lap_potential,
        target_potential,
        fit_indices,
        cfg.modes,
        evaluation.ridge_relative,
    )
    np.savez_compressed(
        output_dir / "reconstruction_fields.npz",
        coordinates=grid["coordinates"],
        mask=grid["mask"],
        geometry=geometry,
        target_potential=target_potential,
        learned_potential=np.einsum(
            "nck,k->nc", learned_potential, learned_coefficients
        ),
        laplacian_potential=np.einsum(
            "nck,k->nc", lap_potential, lap_coefficients
        ),
        target_velocity=target_velocity,
        learned_velocity=np.einsum(
            "nck,k->nc", learned_velocity, learned_coefficients
        ),
        laplacian_velocity=np.einsum(
            "nck,k->nc", lap_velocity, lap_coefficients
        ),
    )
    atomic_json(output_dir / "reconstruction_metrics.json", result)
    write_reconstruction_csv(output_dir / "reconstruction_curve.csv", curve)
    atomic_json(
        output_dir / "first16_rayleigh.json",
        {
            "learned_peeled": learned_rayleigh[:16].tolist(),
            "numerical_dirichlet_laplacian": lap_labels[:16],
            "reporting_scope": "peeled_basis_only",
        },
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--geometry-token", type=int, default=12_000_001)
    parser.add_argument("--grid-resolution", type=int, default=34)
    parser.add_argument("--target-seed", type=int, default=20_260_824)
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fit_fraction < 1.0:
        raise ValueError("fit-fraction must be strictly between zero and one")
    if args.grid_resolution < 12:
        raise ValueError("grid-resolution must be at least 12")
    evaluation = EvaluationConfig(
        geometry_token=args.geometry_token,
        grid_resolution=args.grid_resolution,
        target_seed=args.target_seed,
        fit_fraction=args.fit_fraction,
        learned_batch_size=args.batch_size,
    )
    result = evaluate(args.checkpoint.resolve(), args.output_dir.resolve(), evaluation)
    print(json.dumps(result["k64_reconstruction"], indent=2), flush=True)
    print(f"metrics: {(args.output_dir.resolve() / 'reconstruction_metrics.json')}", flush=True)


if __name__ == "__main__":
    main()
