"""Evaluate a generic peeled-Ritz K=64 Dirichlet streamfunction basis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from square_runtime import backend, install_backend, square_fluid_points


BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#252525"
F32 = np.float32


def load_model(checkpoint: Path):
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    install_backend(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = backend.cfg_namespace(config)
    return metadata, cfg, backend.HardcodedModel(cfg, arrays)


def raw_fields(
    cfg: SimpleNamespace, model: Any, points: np.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    count, modes = len(points), int(cfg.modes)
    value = cp.empty((count, modes), dtype=backend.F32)
    gradient = cp.empty((count, 2, modes), dtype=backend.F32)
    omega = cp.empty((count, modes), dtype=backend.F32)
    velocity = cp.empty((count, 2, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, 0.0, 0.0, False)
    ):
        value[:, index] = field.value[:, 0]
        gradient[:, :, index] = field.grad[:, 0]
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        velocity[:, 0, index] = field.grad[:, 0, 1]
        velocity[:, 1, index] = -field.grad[:, 0, 0]
    return value, gradient, omega, velocity


def peeled_transform(
    cfg: SimpleNamespace,
    network_value: cp.ndarray,
    network_gradient: cp.ndarray,
) -> tuple[cp.ndarray, dict[str, Any]]:
    count, modes = network_value.shape
    weight = cp.asarray(np.float32(1.0 / count), dtype=backend.F32)
    basis = cp.empty_like(network_value)
    transform = cp.zeros((modes, modes), dtype=backend.F32)
    masses: list[float] = []
    passes = max(1, int(getattr(cfg, "reorthogonalization_passes", 2)))
    candidate_mass = weight * cp.sum(network_value * network_value, axis=0)
    candidate_stiffness = weight * cp.sum(
        network_gradient * network_gradient, axis=(0, 1)
    )
    sorting_rayleigh = cp.asnumpy(candidate_stiffness / candidate_mass).astype(
        np.float64
    )
    if not np.all(np.isfinite(sorting_rayleigh)):
        raise FloatingPointError("non-finite candidate Rayleigh quotient")
    if str(getattr(cfg, "causal_sorting", "fixed_order")) == "geometry_rayleigh":
        causal_order = np.argsort(sorting_rayleigh, kind="stable").tolist()
    else:
        causal_order = list(range(modes))
    for causal_rank, network_index in enumerate(causal_order):
        residual = network_value[:, network_index].copy()
        coefficients = cp.zeros((modes,), dtype=backend.F32)
        coefficients[network_index] = np.float32(1.0)
        if causal_rank:
            previous = basis[:, :causal_rank]
            previous_transform = transform[:, :causal_rank]
            for _ in range(passes):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= float(cfg.mass_floor):
            raise FloatingPointError(
                f"peeled streamfunction causal rank {causal_rank + 1} "
                f"(network {network_index + 1}) is unresolved: "
                f"M={mass_value:.8e}"
            )
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis[:, causal_rank] = residual * inverse_norm
        transform[:, causal_rank] = coefficients * inverse_norm
        masses.append(mass_value)
    gram = weight * (basis.T @ basis)
    identity = cp.eye(modes, dtype=backend.F32)
    diagnostics = {
        "peeled_rank": modes,
        "reference_points": count,
        "reorthogonalization_passes": passes,
        "peeled_orthogonality_max_abs": float(
            cp.max(cp.abs(gram - identity)).get()
        ),
        "residual_mass_min": float(min(masses)),
        "residual_mass_max": float(max(masses)),
        "causal_sorting": str(getattr(cfg, "causal_sorting", "fixed_order")),
        "causal_order_1based": [int(index) + 1 for index in causal_order],
        "sorting_rayleigh_by_network": sorting_rayleigh.tolist(),
    }
    return transform, diagnostics


def analytic_modes(points: np.ndarray, modes: int):
    candidates = sorted(
        (
            (math.pi**2 * (m * m + n * n), m, n)
            for m in range(1, 40)
            for n in range(1, 40)
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )[:modes]
    x, y = points[:, 0], points[:, 1]
    fields = np.column_stack(
        [2.0 * np.sin(m * math.pi * x) * np.sin(n * math.pi * y)
         for _, m, n in candidates]
    )
    return candidates, fields.astype(np.float64, copy=False)


def numerical_peeled_ritz_transform(
    network_value: cp.ndarray,
    network_gradient: cp.ndarray,
    peeling: cp.ndarray,
) -> tuple[cp.ndarray, np.ndarray, dict[str, Any]]:
    count, modes = network_value.shape
    weight = 1.0 / count
    peeled_value = network_value @ peeling
    peeled_gradient = cp.einsum(
        "ncm,mk->nck", network_gradient, peeling, optimize=True
    )
    mass = cp.asnumpy(
        np.float32(weight) * (peeled_value.T @ peeled_value)
    ).astype(np.float64)
    stiffness = cp.asnumpy(
        np.float32(weight)
        * cp.einsum("ncm,nck->mk", peeled_gradient, peeled_gradient, optimize=True)
    ).astype(np.float64)
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, ritz = scipy.linalg.eigh(
        stiffness, mass, check_finite=False, driver="gvd"
    )
    transform = peeling @ cp.asarray(ritz.astype(np.float32))
    diagnostics = {
        "ritz_mass_orthogonality_max_abs": float(
            np.max(np.abs(ritz.T @ mass @ ritz - np.eye(modes)))
        ),
        "ritz_postprocessing": (
            "numerical generalized Ritz rotation of the peeled basis only; "
            "no analytic eigenfunction or degeneracy-block alignment"
        ),
        "analytic_orientation_used": False,
    }
    return transform, eigenvalues, diagnostics


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_atlas(
    fields: np.ndarray,
    eigenvalues: np.ndarray,
    stem: Path,
    quantity: str,
) -> None:
    modes, resolution, _ = fields.shape
    fig, axes = plt.subplots(
        4, 16, figsize=(19.2, 5.65), constrained_layout=True
    )
    for index, axis in enumerate(axes.flat):
        field = fields[index]
        limit = max(float(np.max(np.abs(field))), 1.0e-12)
        axis.imshow(
            field,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        axis.set_title(
            f"{index + 1}\n$\\lambda$={eigenvalues[index]:.1f}",
            fontsize=8.0,
            pad=1.0,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color(CHARCOAL)
    fig.suptitle(
        f"Unit-square Dirichlet SIREN - K={modes} generic peeled-Ritz "
        f"{quantity} basis",
        fontsize=14.0,
        weight="bold",
    )
    save_figure(fig, stem)


def render_spectrum(
    eigenvalues: np.ndarray,
    analytic_values: np.ndarray,
    stem: Path,
) -> None:
    indices = np.arange(1, len(eigenvalues) + 1)
    relative = np.abs(eigenvalues - analytic_values) / analytic_values
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), constrained_layout=True)
    axes[0].plot(indices, analytic_values, color=CHARCOAL, linewidth=1.5, label="Analytic")
    axes[0].scatter(indices, eigenvalues, color=RED, s=12, label="Peeled SIREN", zorder=3)
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Dirichlet eigenvalue")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.18, linewidth=0.5)
    axes[1].semilogy(indices, np.maximum(relative, 1.0e-8), color=BLUE, linewidth=1.2)
    axes[1].axhline(0.01, color=RED, linestyle="--", linewidth=0.8, label="1%")
    axes[1].set_xlabel("Mode index")
    axes[1].set_ylabel("Relative eigenvalue error")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.18, linewidth=0.5)
    fig.suptitle(
        "Generic peeled K=64 spectrum - analytic values are benchmark only",
        fontsize=12.5,
        weight="bold",
    )
    save_figure(fig, stem)


def subspace_symmetry_metrics(fields: np.ndarray) -> dict[str, Any]:
    """Measure group invariance of the span, independent of basis orientation."""
    flattened = fields.reshape(fields.shape[0], -1).T.astype(np.float64)
    basis = np.linalg.qr(flattened, mode="reduced")[0]
    transforms = {
        "reflect_x": fields[:, :, ::-1].reshape(fields.shape[0], -1).T,
        "reflect_y": fields[:, ::-1, :].reshape(fields.shape[0], -1).T,
        "rotate_90": np.rot90(fields, 1, axes=(1, 2)).reshape(
            fields.shape[0], -1
        ).T,
    }
    result: dict[str, Any] = {}
    for name, transformed in transforms.items():
        transformed_basis = np.linalg.qr(
            transformed.astype(np.float64), mode="reduced"
        )[0]
        singular = np.linalg.svd(
            basis.T @ transformed_basis, compute_uv=False
        )
        minimum = float(np.min(singular))
        result[name] = {
            "mean_squared_cosine": float(np.mean(singular**2)),
            "minimum_cosine": minimum,
            "maximum_principal_angle_deg": float(
                np.degrees(np.arccos(np.clip(minimum, 0.0, 1.0)))
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-points", type=int, default=16384)
    parser.add_argument("--evaluation-points", type=int, default=16384)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    metadata, cfg, model = load_model(args.checkpoint)
    if str(cfg.basis_objective) != "streamfunction_dirichlet_rayleigh":
        raise ValueError("checkpoint does not contain the Dirichlet psi objective")
    reference = square_fluid_points(
        cfg, args.reference_points, 0.0, 0.0, int(cfg.seed) + 4_000_001
    )
    network_value, network_gradient, _, _ = raw_fields(cfg, model, reference)
    peeling, peeling_diagnostics = peeled_transform(
        cfg, network_value, network_gradient
    )
    transform, _, ritz_diagnostics = numerical_peeled_ritz_transform(
        network_value, network_gradient, peeling
    )
    del network_value, network_gradient, peeling
    cp.get_default_memory_pool().free_all_blocks()

    evaluation = square_fluid_points(
        cfg, args.evaluation_points, 0.0, 0.0, int(cfg.seed) + 5_000_001
    )
    network_value, network_gradient, network_omega, network_velocity = raw_fields(
        cfg, model, evaluation
    )
    value = network_value @ transform
    gradient = cp.einsum(
        "ncm,mk->nck", network_gradient, transform, optimize=True
    )
    _omega = network_omega @ transform
    _velocity = cp.einsum(
        "ncm,mk->nck", network_velocity, transform, optimize=True
    )
    weight = np.float32(1.0 / len(evaluation))
    mass_gram = weight * (value.T @ value)
    stiffness = weight * cp.einsum(
        "ncm,nck->mk", gradient, gradient, optimize=True
    )
    learned_values = cp.asnumpy(cp.diag(stiffness) / cp.diag(mass_gram)).astype(np.float64)
    mass_error = float(
        cp.max(cp.abs(mass_gram - cp.eye(cfg.modes, dtype=backend.F32))).get()
    )
    analytic_pairs, analytic_eval = analytic_modes(evaluation, int(cfg.modes))
    analytic_values = np.asarray(
        [item[0] for item in analytic_pairs], dtype=np.float64
    )
    relative_error = np.abs(learned_values - analytic_values) / analytic_values
    learned_eval = cp.asnumpy(value).astype(np.float64)
    learned_q = np.linalg.qr(learned_eval, mode="reduced")[0]
    analytic_q = np.linalg.qr(analytic_eval, mode="reduced")[0]
    singular = np.linalg.svd(learned_q.T @ analytic_q, compute_uv=False)

    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    grid_network_value, _, grid_network_omega, grid_network_velocity = raw_fields(
        cfg, model, grid_points
    )
    grid_value = cp.asnumpy(grid_network_value @ transform).T.reshape(
        cfg.modes, args.resolution, args.resolution
    )
    grid_omega = cp.asnumpy(grid_network_omega @ transform).T.reshape(
        cfg.modes, args.resolution, args.resolution
    )
    grid_velocity = cp.asnumpy(
        cp.einsum(
            "ncm,mk->nck", grid_network_velocity, transform, optimize=True
        )
    ).transpose(2, 0, 1).reshape(cfg.modes, args.resolution, args.resolution, 2)

    render_atlas(
        grid_value,
        learned_values,
        args.output_dir / "square_k64_generic_peeled_ritz_streamfunction_atlas",
        "streamfunction",
    )
    render_atlas(
        grid_omega,
        learned_values,
        args.output_dir / "square_k64_generic_peeled_ritz_vorticity_atlas",
        "vorticity",
    )
    render_spectrum(
        learned_values,
        analytic_values,
        args.output_dir / "square_k64_generic_peeled_spectrum_validation",
    )

    rows = []
    for index, (analytic, learned, error) in enumerate(
        zip(analytic_values, learned_values, relative_error), start=1
    ):
        rows.append(
            {
                "mode": index,
                "analytic_eigenvalue_benchmark_only": float(analytic),
                "learned_eigenvalue": float(learned),
                "relative_error": float(error),
            }
        )
    with (args.output_dir / "square_k64_generic_peeled_mode_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(metadata["epoch"]),
        "domain": "unit square without metaballs",
        "boundary_condition": "Dirichlet streamfunction psi=0",
        "basis_objective": str(cfg.basis_objective),
        "rank_metric_policy": "peeled basis only; raw rank excluded",
        "training_sampling": "IID uniform random; no D4 augmentation",
        "analytic_orientation_used": False,
        "peeled_rank": int(cfg.modes),
        "peeling": peeling_diagnostics,
        "ritz": ritz_diagnostics,
        "independent_evaluation_points": int(args.evaluation_points),
        "peeled_mass_orthogonality_max_abs": mass_error,
        "spectrum": {
            "analytic_mean": float(np.mean(analytic_values)),
            "learned_mean": float(np.mean(learned_values)),
            "median_relative_error": float(np.median(relative_error)),
            "p90_relative_error": float(np.percentile(relative_error, 90.0)),
            "max_relative_error": float(np.max(relative_error)),
            "within_1_percent": int(np.count_nonzero(relative_error <= 0.01)),
            "within_5_percent": int(np.count_nonzero(relative_error <= 0.05)),
        },
        "analytic_benchmark_only": {
            "used_to_transform_or_orient_basis": False,
            "mean_squared_cosine": float(np.mean(singular**2)),
            "minimum_cosine": float(np.min(singular)),
            "maximum_principal_angle_deg": float(
                np.degrees(np.arccos(np.clip(np.min(singular), 0.0, 1.0)))
            ),
        },
        "orientation_free_subspace_symmetry": subspace_symmetry_metrics(
            grid_value
        ),
        "outputs": {
            "streamfunction_atlas": (
                "square_k64_generic_peeled_ritz_streamfunction_atlas"
            ),
            "vorticity_atlas": "square_k64_generic_peeled_ritz_vorticity_atlas",
            "spectrum_validation": (
                "square_k64_generic_peeled_spectrum_validation"
            ),
            "fields": "square_k64_generic_peeled_basis_fields.npz",
        },
    }
    (args.output_dir / "square_k64_generic_peeled_basis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "square_k64_generic_peeled_basis_fields.npz",
        coordinates=coordinates,
        streamfunction=grid_value.astype(F32),
        vorticity=grid_omega.astype(F32),
        velocity=grid_velocity.astype(F32),
        learned_eigenvalues=learned_values,
        analytic_eigenvalues_benchmark_only=analytic_values,
        network_to_peeled_ritz_transform=cp.asnumpy(transform),
        causal_order_1based=np.asarray(
            peeling_diagnostics["causal_order_1based"], dtype=np.int32
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
