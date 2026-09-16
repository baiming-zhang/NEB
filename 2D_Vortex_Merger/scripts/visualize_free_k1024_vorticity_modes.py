#!/usr/bin/env python3
"""Render red/blue potential and vorticity of peeled free-boundary modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
    fluid_points,
)


FIRST_RANKS = tuple(range(1, 17))
SPECTRUM_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_model(checkpoint: Path) -> tuple[dict[str, Any], Any, HardcodedModel]:
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = metadata["config"]
    required = {
        "hidden_layers": 4,
        "geometry_mode": "free_box",
        "envelope_mode": "free_identity",
        "input_encoding": "raw_xy",
        "basis_objective": "potential_rayleigh",
        "spatial_derivative_order": 1,
        "causal_sorting": "geometry_rayleigh",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint protocol mismatch: {mismatches}")
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(config)
    model = HardcodedModel(cfg, arrays)
    del arrays
    return metadata, cfg, model


def raw_potential_jet1(
    model: HardcodedModel, cfg: Any, points: np.ndarray
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    count, modes = len(points), int(cfg.modes)
    point_device = cp.asarray(points, dtype=cp.float32)
    value = cp.empty((count, modes), dtype=cp.float32)
    gradient_x = cp.empty_like(value)
    gradient_y = cp.empty_like(value)
    for mode, field, _cache in model.forward_stream_jet1(
        cfg, point_device, 0.0, 0.0, False
    ):
        index = int(mode.mode_index)
        value[:, index] = field.value[:, 0]
        gradient_x[:, index] = field.grad[:, 0, 0]
        gradient_y[:, index] = field.grad[:, 0, 1]
    gauge_mean = cp.mean(value, axis=0, keepdims=True, dtype=cp.float32)
    value -= gauge_mean
    return value, gradient_x, gradient_y, gauge_mean


def causal_peeling_transform(
    cfg: Any,
    value: cp.ndarray,
    gradient_x: cp.ndarray,
    gradient_y: cp.ndarray,
    domain_area: float,
) -> tuple[cp.ndarray, np.ndarray, dict[str, Any]]:
    count, modes = value.shape
    weight = cp.asarray(domain_area / count, dtype=cp.float32)
    raw_mass = weight * cp.sum(value * value, axis=0, dtype=cp.float32)
    raw_stiffness = weight * cp.sum(
        gradient_x * gradient_x + gradient_y * gradient_y,
        axis=0,
        dtype=cp.float32,
    )
    order = cp.argsort(raw_stiffness / raw_mass, kind="stable")
    sorted_value = value[:, order]
    sorted_gx = gradient_x[:, order]
    sorted_gy = gradient_y[:, order]
    basis = cp.empty_like(sorted_value)
    basis_gx = cp.empty_like(sorted_gx)
    basis_gy = cp.empty_like(sorted_gy)
    transform = cp.zeros((modes, modes), dtype=cp.float32)
    rayleigh = cp.empty(modes, dtype=cp.float32)
    masses = cp.empty(modes, dtype=cp.float32)
    passes = int(cfg.reorthogonalization_passes)
    for rank in range(modes):
        residual = sorted_value[:, rank].copy()
        residual_gx = sorted_gx[:, rank].copy()
        residual_gy = sorted_gy[:, rank].copy()
        coefficients = cp.zeros(modes, dtype=cp.float32)
        coefficients[order[rank]] = 1.0
        if rank:
            previous = basis[:, :rank]
            previous_gx = basis_gx[:, :rank]
            previous_gy = basis_gy[:, :rank]
            previous_transform = transform[:, :rank]
            for _ in range(passes):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                residual_gx -= previous_gx @ increment
                residual_gy -= previous_gy @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= float(cfg.mass_floor):
            raise FloatingPointError(
                f"unresolved free-boundary peeled mode {rank + 1}: mass={mass_value:.8e}"
            )
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis[:, rank] = residual * inverse_norm
        basis_gx[:, rank] = residual_gx * inverse_norm
        basis_gy[:, rank] = residual_gy * inverse_norm
        transform[:, rank] = coefficients * inverse_norm
        masses[rank] = mass
        rayleigh[rank] = weight * cp.sum(
            basis_gx[:, rank] ** 2 + basis_gy[:, rank] ** 2,
            dtype=cp.float32,
        )
    gram = weight * (basis.T @ basis)
    diagnostics = {
        "peeled_rank": modes,
        "causal_order_1based": (cp.asnumpy(order) + 1).astype(int).tolist(),
        "reorthogonalization_passes": passes,
        "mean_zero_gauge_max_abs": float(cp.max(cp.abs(cp.mean(basis, axis=0))).get()),
        "peeled_orthogonality_max_abs": float(
            cp.max(cp.abs(gram - cp.eye(modes, dtype=cp.float32))).get()
        ),
        "minimum_residual_mass": float(cp.min(masses).get()),
    }
    return transform, cp.asnumpy(rayleigh), diagnostics


def evaluate_selected_vorticity(
    model: HardcodedModel,
    cfg: Any,
    transform: cp.ndarray,
    points: np.ndarray,
    ranks: tuple[int, ...],
    point_batch: int,
) -> np.ndarray:
    selected = cp.asarray(np.asarray(ranks, dtype=np.int64) - 1)
    result = np.empty((len(ranks), len(points)), dtype=np.float32)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        point_device = cp.asarray(points[start:stop], dtype=cp.float32)
        raw_vorticity = cp.empty(
            (stop - start, int(cfg.modes)), dtype=cp.float32
        )
        for mode, field, _cache in model.forward_stream_jet2(
            cfg, point_device, 0.0, 0.0, False
        ):
            index = int(mode.mode_index)
            raw_vorticity[:, index] = -(
                field.hess[:, 0, 0] + field.hess[:, 0, 2]
            )
        result[:, start:stop] = cp.asnumpy(
            (raw_vorticity @ transform[:, selected]).T
        )
        del point_device, raw_vorticity
    return result


def evaluate_selected_potential(
    model: HardcodedModel,
    cfg: Any,
    transform: cp.ndarray,
    gauge_mean: cp.ndarray,
    points: np.ndarray,
    ranks: tuple[int, ...],
    point_batch: int,
) -> np.ndarray:
    """Evaluate the actual trained field psi after gauge removal and peeling."""
    selected = cp.asarray(np.asarray(ranks, dtype=np.int64) - 1)
    result = np.empty((len(ranks), len(points)), dtype=np.float32)
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        point_device = cp.asarray(points[start:stop], dtype=cp.float32)
        raw_potential = cp.empty(
            (stop - start, int(cfg.modes)), dtype=cp.float32
        )
        for mode, field, _cache in model.forward_stream_jet1(
            cfg, point_device, 0.0, 0.0, False
        ):
            raw_potential[:, int(mode.mode_index)] = field.value[:, 0]
        raw_potential -= gauge_mean
        result[:, start:stop] = cp.asnumpy(
            (raw_potential @ transform[:, selected]).T
        )
        del point_device, raw_potential
    return result


def canonicalize(fields: np.ndarray) -> tuple[np.ndarray, list[int]]:
    result = fields.copy()
    signs: list[int] = []
    for field in result:
        anchor = int(np.argmax(np.abs(field)))
        sign = -1 if field[anchor] < 0.0 else 1
        field *= sign
        signs.append(sign)
    return result, signs


def plot_modes(
    fields: np.ndarray,
    ranks: tuple[int, ...],
    rayleigh: np.ndarray,
    width: int,
    extent: tuple[float, float, float, float],
    stem: Path,
    field_name: str,
    field_formula: str,
    field_description: str,
    modes: int,
) -> None:
    columns = 4
    rows = math.ceil(len(ranks) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12.8, 3.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, field, rank in zip(axes.flat, fields, ranks, strict=False):
        scale = max(float(np.percentile(np.abs(field), 99.0)), 1e-12)
        axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            interpolation="bilinear",
        )
        axis.set_title(
            rf"{field_name} mode {rank} · $R_\psi={rayleigh[rank - 1]:.3g}$"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    for axis in axes.flat[len(ranks) :]:
        axis.set_visible(False)
    figure.suptitle(
        rf"2D vortex merger free-boundary basis · {field_formula} · K={modes} · "
        rf"depth=5 · width={width}"
        "\n"
        "raw x,y · Potential-Rayleigh Jet1 · stable causal sort · two-pass peeling"
        f" · red/blue = positive/negative {field_description}",
        fontsize=13,
    )
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peeling-points", type=int, default=4096)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--point-batch", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, cfg, model = load_model(checkpoint)
    width = int(cfg.hidden)
    first_ranks = tuple(rank for rank in FIRST_RANKS if rank <= int(cfg.modes))
    spectrum_ranks = tuple(
        rank for rank in SPECTRUM_RANKS if rank <= int(cfg.modes)
    )
    integration_points = fluid_points(
        cfg, args.peeling_points, 0.0, 0.0, int(cfg.seed) + 3_100_000
    ).astype(np.float32)
    domain_area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
    value, gx, gy, gauge_mean = raw_potential_jet1(
        model, cfg, integration_points
    )
    transform, rayleigh, diagnostics = causal_peeling_transform(
        cfg, value, gx, gy, domain_area
    )
    del value, gx, gy

    axis = np.linspace(
        cfg.xmin, cfg.xmax, args.grid_size, endpoint=True, dtype=np.float32
    )
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    ranks = tuple(dict.fromkeys((*first_ranks, *spectrum_ranks)))
    compact_vorticity = evaluate_selected_vorticity(
        model, cfg, transform, points, ranks, args.point_batch
    )
    compact_potential = evaluate_selected_potential(
        model,
        cfg,
        transform,
        gauge_mean,
        points,
        ranks,
        args.point_batch,
    )
    compact_vorticity, vorticity_signs = canonicalize(compact_vorticity)
    compact_potential, potential_signs = canonicalize(compact_potential)
    vorticity_fields = compact_vorticity.reshape(
        len(ranks), args.grid_size, args.grid_size
    )
    potential_fields = compact_potential.reshape(
        len(ranks), args.grid_size, args.grid_size
    )
    vorticity_by_rank = {
        rank: vorticity_fields[index] for index, rank in enumerate(ranks)
    }
    potential_by_rank = {
        rank: potential_fields[index] for index, rank in enumerate(ranks)
    }
    extent = (float(cfg.xmin), float(cfg.xmax), float(cfg.ymin), float(cfg.ymax))
    first_stem = output_dir / f"width_{width:03d}_vorticity_modes_first16"
    spectrum_stem = output_dir / f"width_{width:03d}_vorticity_modes_spectrum"
    plot_modes(
        np.stack([vorticity_by_rank[rank] for rank in first_ranks]),
        first_ranks,
        rayleigh,
        width,
        extent,
        first_stem,
        "vorticity",
        r"$\omega=-\Delta\psi$",
        "vorticity",
        int(cfg.modes),
    )
    plot_modes(
        np.stack([vorticity_by_rank[rank] for rank in spectrum_ranks]),
        spectrum_ranks,
        rayleigh,
        width,
        extent,
        spectrum_stem,
        "vorticity",
        r"$\omega=-\Delta\psi$",
        "vorticity",
        int(cfg.modes),
    )
    data_path = output_dir / f"width_{width:03d}_vorticity_modes.npz"
    np.savez_compressed(
        data_path,
        x=axis,
        y=axis,
        ranks=np.asarray(ranks, dtype=np.int32),
        vorticity=vorticity_fields,
        potential_rayleigh=rayleigh[np.asarray(ranks) - 1],
    )
    potential_first_stem = output_dir / f"width_{width:03d}_potential_modes_first16"
    potential_spectrum_stem = (
        output_dir / f"width_{width:03d}_potential_modes_spectrum"
    )
    plot_modes(
        np.stack([potential_by_rank[rank] for rank in first_ranks]),
        first_ranks,
        rayleigh,
        width,
        extent,
        potential_first_stem,
        "potential",
        r"$\psi$",
        "streamfunction potential",
        int(cfg.modes),
    )
    plot_modes(
        np.stack([potential_by_rank[rank] for rank in spectrum_ranks]),
        spectrum_ranks,
        rayleigh,
        width,
        extent,
        potential_spectrum_stem,
        "potential",
        r"$\psi$",
        "streamfunction potential",
        int(cfg.modes),
    )
    potential_data_path = output_dir / f"width_{width:03d}_potential_modes.npz"
    np.savez_compressed(
        potential_data_path,
        x=axis,
        y=axis,
        ranks=np.asarray(ranks, dtype=np.int32),
        potential=potential_fields,
        potential_rayleigh=rayleigh[np.asarray(ranks) - 1],
    )
    summary = {
        "complete": True,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": int(metadata["epoch"]),
        "architecture": {
            "modes": int(cfg.modes),
            "hidden_width": width,
            "hidden_layers": int(cfg.hidden_layers),
            "total_affine_layers": int(cfg.hidden_layers) + 1,
        },
        "protocol": {
            "case": "2D_Vortex_Merger",
            "domain": "[0,2pi]x[0,2pi]",
            "boundary_condition": "free on all four sides",
            "boundary_realization": "raw physical x,y; identity envelope; no seam constraint",
            "mean_zero_streamfunction_gauge": True,
            "basis_objective": "Potential-Rayleigh",
            "training_spatial_jet_order": 1,
            "display_fields": [
                "potential psi (directly trained field)",
                "vorticity omega=-laplacian(psi)",
            ],
            "display_derivative": "analytic Jet2",
            "causal_sorting": cfg.causal_sorting,
            "reorthogonalization_passes": int(cfg.reorthogonalization_passes),
            "grid_size": args.grid_size,
        },
        "peeling": diagnostics,
        "display": {
            "first16_ranks": list(first_ranks),
            "spectrum_ranks": list(spectrum_ranks),
            "potential_signs_for_unique_ranks": dict(
                zip((str(rank) for rank in ranks), potential_signs)
            ),
            "vorticity_signs_for_unique_ranks": dict(
                zip((str(rank) for rank in ranks), vorticity_signs)
            ),
            "normalization": "independent symmetric 99th absolute percentile",
            "colormap": "RdBu_r",
        },
        "figures": {
            "first16_png": str(first_stem.with_suffix(".png")),
            "first16_pdf": str(first_stem.with_suffix(".pdf")),
            "spectrum_png": str(spectrum_stem.with_suffix(".png")),
            "spectrum_pdf": str(spectrum_stem.with_suffix(".pdf")),
            "data_npz": str(data_path),
            "potential_first16_png": str(
                potential_first_stem.with_suffix(".png")
            ),
            "potential_first16_pdf": str(
                potential_first_stem.with_suffix(".pdf")
            ),
            "potential_spectrum_png": str(
                potential_spectrum_stem.with_suffix(".png")
            ),
            "potential_spectrum_pdf": str(
                potential_spectrum_stem.with_suffix(".pdf")
            ),
            "potential_data_npz": str(potential_data_path),
        },
    }
    write_json(output_dir / "mode_visualization_summary.json", summary)
    print(json.dumps(summary["figures"], indent=2), flush=True)


if __name__ == "__main__":
    main()
