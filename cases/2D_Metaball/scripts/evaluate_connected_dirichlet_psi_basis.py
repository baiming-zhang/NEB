"""Evaluate connected-metaball K=64 Dirichlet streamfunction bases."""

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

from metaball_runtime import (
    backend,
    excluded_area,
    fluid_mask,
    geometry_from_token,
    install_backend,
    metaball_fluid_points,
    obstacle_level_numpy,
    outer_level_numpy,
)


BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#252525"
LIGHT_BLUE = "#D9EAF4"
LIGHT_RED = "#F2D7D5"
F32 = np.float32

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_model(checkpoint: Path):
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    install_backend(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = backend.cfg_namespace(config)
    return metadata, cfg, backend.HardcodedModel(cfg, arrays)


def raw_fields(
    cfg: SimpleNamespace,
    model: Any,
    points: np.ndarray,
    token: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    count, modes = len(points), int(cfg.modes)
    value = cp.empty((count, modes), dtype=backend.F32)
    gradient = cp.empty((count, 2, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, token, 0.0, False)
    ):
        value[:, index] = field.value[:, 0]
        gradient[:, :, index] = field.grad[:, 0]
    return value, gradient


def peeled_transform(
    cfg: SimpleNamespace,
    network_value: cp.ndarray,
    network_gradient: cp.ndarray,
    fluid_area: float,
) -> tuple[cp.ndarray, dict[str, Any]]:
    """Reproduce training-time dynamic sorting and two-pass psi peeling."""
    count, modes = network_value.shape
    weight = cp.asarray(np.float32(fluid_area / count), dtype=backend.F32)
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
        coefficients[network_index] = F32(1.0)
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
                f"unresolved peeled mode {causal_rank + 1}: M={mass_value:.8e}"
            )
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis[:, causal_rank] = residual * inverse_norm
        transform[:, causal_rank] = coefficients * inverse_norm
        masses.append(mass_value)
    gram = weight * (basis.T @ basis)
    diagnostics = {
        "peeled_rank": modes,
        "reference_points": count,
        "fluid_area": float(fluid_area),
        "reorthogonalization_passes": passes,
        "peeled_orthogonality_max_abs": float(
            cp.max(cp.abs(gram - cp.eye(modes, dtype=backend.F32))).get()
        ),
        "residual_mass_min": float(min(masses)),
        "residual_mass_max": float(max(masses)),
        "causal_sorting": str(getattr(cfg, "causal_sorting", "fixed_order")),
        "causal_order_1based": [int(index) + 1 for index in causal_order],
        "sorting_rayleigh_by_network": sorting_rayleigh.tolist(),
    }
    return transform, diagnostics


def numerical_ritz_transform(
    network_value: cp.ndarray,
    network_gradient: cp.ndarray,
    peeling: cp.ndarray,
    fluid_area: float,
) -> tuple[cp.ndarray, np.ndarray, dict[str, Any]]:
    """Rotate only inside the learned span; use no analytic orientation."""
    count, modes = network_value.shape
    weight = np.float32(fluid_area / count)
    peeled_value = network_value @ peeling
    peeled_gradient = cp.einsum(
        "ncm,mk->nck", network_gradient, peeling, optimize=True
    )
    mass = cp.asnumpy(weight * (peeled_value.T @ peeled_value)).astype(np.float64)
    stiffness = cp.asnumpy(
        weight
        * cp.einsum("ncm,nck->mk", peeled_gradient, peeled_gradient, optimize=True)
    ).astype(np.float64)
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, ritz = scipy.linalg.eigh(
        stiffness, mass, check_finite=False, driver="gvd"
    )
    transform = peeling @ cp.asarray(ritz.astype(F32))
    diagnostics = {
        "ritz_mass_orthogonality_max_abs": float(
            np.max(np.abs(ritz.T @ mass @ ritz - np.eye(modes)))
        ),
        "ritz_postprocessing": (
            "numerical generalized Ritz rotation within the dynamically "
            "peeled learned span; no analytic eigenfunction or degeneracy alignment"
        ),
        "analytic_orientation_used": False,
    }
    return transform, eigenvalues, diagnostics


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def geometry_preview(
    cfg: SimpleNamespace, tokens: list[float], stem: Path, resolution: int = 420
) -> None:
    coordinates = np.linspace(0.0, 1.0, resolution, dtype=F32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.65))
    for index, (axis, token) in enumerate(zip(axes.flat, tokens)):
        geometry = geometry_from_token(cfg, token)
        outer = outer_level_numpy(cfg, points).reshape(xx.shape)
        obstacle = obstacle_level_numpy(cfg, points, geometry).reshape(xx.shape)
        region = np.ones_like(outer, dtype=np.int8)
        region[outer <= 0.0] = 0
        region[(outer > 0.0) & (obstacle <= 0.0)] = 2
        axis.imshow(
            region,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=matplotlib.colors.ListedColormap(["#F2F2F2", LIGHT_BLUE, LIGHT_RED]),
            interpolation="nearest",
            vmin=0,
            vmax=2,
        )
        axis.contour(xx, yy, outer, levels=[0.0], colors=[CHARCOAL], linewidths=0.8)
        axis.contour(xx, yy, obstacle, levels=[0.0], colors=[RED], linewidths=1.25)
        for cx, cy, radius in geometry.reshape(3, 3):
            axis.add_patch(
                plt.Circle(
                    (cx, cy), radius, fill=False, color=RED,
                    linewidth=0.65, linestyle=(0, (2.0, 1.6)), alpha=0.8,
                )
            )
        axis.set_title(f"Unseen geometry {index + 1}", fontsize=12.0, pad=4.0)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.018, right=0.992, bottom=0.025, top=0.875, wspace=0.035, hspace=0.11)
    fig.suptitle(
        "Connected three-circle metaball family",
        fontsize=19.0,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.912,
        "IID circle parameters conditioned on one overlap-connected smooth union; "
        "red dashed: primitives, red solid: soft-min boundary",
        ha="center",
        va="center",
        fontsize=12.0,
        color=CHARCOAL,
    )
    save_figure(fig, stem)


def render_atlas(
    cfg: SimpleNamespace,
    model: Any,
    token: float,
    total_transform: cp.ndarray,
    eigenvalues: np.ndarray,
    stem: Path,
    resolution: int,
) -> None:
    coordinates = np.linspace(0.0, 1.0, resolution, dtype=F32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel()))
    valid = fluid_mask(cfg, all_points, token)
    raw_value, _ = raw_fields(cfg, model, all_points[valid].astype(F32), token)
    learned = cp.asnumpy(raw_value @ total_transform).T
    fields = np.full((int(cfg.modes), resolution * resolution), np.nan, dtype=F32)
    fields[:, valid] = learned.astype(F32, copy=False)
    fields = fields.reshape(int(cfg.modes), resolution, resolution)
    geometry = geometry_from_token(cfg, token)
    obstacle = obstacle_level_numpy(cfg, all_points, geometry).reshape(xx.shape)
    outer = outer_level_numpy(cfg, all_points).reshape(xx.shape)

    fig, axes = plt.subplots(4, 16, figsize=(21.0, 6.45))
    for index, axis in enumerate(axes.flat):
        field = fields[index]
        finite = np.abs(field[np.isfinite(field)])
        limit = max(float(np.percentile(finite, 99.5)), 1.0e-12)
        axis.imshow(
            field,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="bilinear",
        )
        axis.contour(xx, yy, outer, levels=[0.0], colors=[CHARCOAL], linewidths=0.45)
        axis.contour(xx, yy, obstacle, levels=[0.0], colors=[CHARCOAL], linewidths=0.55)
        axis.set_title(
            f"{index + 1}\n$\\lambda$={eigenvalues[index]:.1f}",
            fontsize=8.2,
            pad=1.2,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_linewidth(0.35)
            spine.set_color(CHARCOAL)
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.025, top=0.865, wspace=0.055, hspace=0.26)
    fig.suptitle(
        "Connected-metaball SIREN - K=64 dynamically peeled-Ritz streamfunction basis",
        fontsize=17.0,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.91,
        "Unseen geometry; numerical Ritz orientation only; no symmetry or analytic mode alignment",
        ha="center",
        fontsize=11.5,
        color=CHARCOAL,
    )
    save_figure(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-points", type=int, default=16384)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--token-start", type=int, default=3000001)
    parser.add_argument("--validation-shapes", type=int, default=4)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, model = load_model(args.checkpoint)
    if str(cfg.basis_objective) != "streamfunction_dirichlet_rayleigh":
        raise ValueError("checkpoint is not a Dirichlet streamfunction run")
    tokens = [float(args.token_start + index) for index in range(max(8, args.validation_shapes))]
    rows: list[dict[str, Any]] = []
    first_transform: cp.ndarray | None = None
    first_eigenvalues: np.ndarray | None = None
    first_peeling: dict[str, Any] | None = None
    first_ritz: dict[str, Any] | None = None
    for index, token in enumerate(tokens[: args.validation_shapes]):
        points = metaball_fluid_points(
            cfg,
            args.validation_points,
            token,
            0.0,
            int(cfg.seed) + 4_000_000 + index,
        )
        value, gradient = raw_fields(cfg, model, points, token)
        box_area = float((cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin))
        fluid_area = box_area - excluded_area(cfg, token)
        peeling, peeling_diagnostics = peeled_transform(
            cfg, value, gradient, fluid_area
        )
        total, eigenvalues, ritz_diagnostics = numerical_ritz_transform(
            value, gradient, peeling, fluid_area
        )
        row = {
            "geometry_token": int(token),
            "peeled_rank": int(peeling_diagnostics["peeled_rank"]),
            "peeled_orthogonality_max_abs": float(
                peeling_diagnostics["peeled_orthogonality_max_abs"]
            ),
            "ritz_mass_orthogonality_max_abs": float(
                ritz_diagnostics["ritz_mass_orthogonality_max_abs"]
            ),
            "residual_mass_min": float(peeling_diagnostics["residual_mass_min"]),
            "lambda_min": float(eigenvalues[0]),
            "lambda_max": float(eigenvalues[-1]),
            "fluid_area": float(fluid_area),
        }
        rows.append(row)
        if index == 0:
            first_transform = total
            first_eigenvalues = eigenvalues
            first_peeling = peeling_diagnostics
            first_ritz = ritz_diagnostics
        else:
            del total
        del value, gradient, peeling
        cp.get_default_memory_pool().free_all_blocks()

    geometry_preview(
        cfg,
        tokens[:8],
        args.output_dir / "connected_metaball_unseen_geometry_family",
    )
    assert first_transform is not None and first_eigenvalues is not None
    render_atlas(
        cfg,
        model,
        tokens[0],
        first_transform,
        first_eigenvalues,
        args.output_dir / "connected_metaball_k64_peeled_ritz_streamfunction_atlas",
        args.resolution,
    )

    with (args.output_dir / "independent_unseen_geometry_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": int(metadata["epoch"]),
        "modes": int(cfg.modes),
        "basis_objective": str(cfg.basis_objective),
        "validation_sampling": "independent IID PCG64 physical-fluid-area points",
        "validation_geometry_tokens": [int(row["geometry_token"]) for row in rows],
        "all_peeled_rank_64": all(row["peeled_rank"] == int(cfg.modes) for row in rows),
        "worst_peeled_orthogonality_max_abs": max(
            row["peeled_orthogonality_max_abs"] for row in rows
        ),
        "worst_ritz_mass_orthogonality_max_abs": max(
            row["ritz_mass_orthogonality_max_abs"] for row in rows
        ),
        "minimum_residual_mass": min(row["residual_mass_min"] for row in rows),
        "atlas_geometry_token": int(tokens[0]),
        "atlas_geometry_parameters": geometry_from_token(cfg, tokens[0]).tolist(),
        "peeling": first_peeling,
        "ritz": first_ritz,
        "analytic_orientation_used": False,
        "symmetry_augmentation_used": False,
        "raw_rank_reported": False,
        "artifacts": {
            "geometry_family": "connected_metaball_unseen_geometry_family",
            "basis_atlas": "connected_metaball_k64_peeled_ritz_streamfunction_atlas",
        },
    }
    (args.output_dir / "independent_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
