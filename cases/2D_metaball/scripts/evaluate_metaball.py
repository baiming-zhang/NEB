"""Evaluate a frozen metaball-family basis and render every learned mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metaball_runtime import (
    apply_basis_transform,
    backend,
    fluid_mask,
    geometry_from_token,
    install_backend,
    metaball_fluid_points,
    raw_basis_field_matrices,
    vorticity_peeling_transform,
)


BLUE = "#2166AC"
RED = "#B2182B"
plt.rcParams.update({"font.family": "Times New Roman"})


def load_checkpoint(path: Path):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def geometry_preview(cfg, output: Path) -> None:
    coordinates = np.linspace(0.0, 1.0, 280, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    fig, axes = plt.subplots(3, 4, figsize=(8.8, 6.6), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        token = 1_000_001.0 + index
        mask = fluid_mask(cfg, points, token).reshape(xx.shape)
        axis.imshow(
            mask.astype(float), origin="lower", extent=(0, 1, 0, 1),
            cmap=plt.matplotlib.colors.ListedColormap(["#F1F1F1", "#D7E8F4"]),
            interpolation="nearest", vmin=0, vmax=1,
        )
        geometry = geometry_from_token(cfg, token).reshape(3, 3)
        for cx, cy, radius in geometry:
            axis.add_patch(
                plt.Circle((cx, cy), radius, fill=False, color=RED, linewidth=0.65)
            )
        axis.set_title(f"shape {index + 1}", fontsize=7)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    fig.suptitle(
        "Victoria SA 2025 three-circle geometry family\n"
        "centres $\\sim U([0.25,0.75]^2)$, radii $\\sim U([0.1,0.3])$",
        fontsize=10,
    )
    save_figure(fig, output / "metaball_geometry_family")


def render_modes(
    cfg, model, output: Path, token: float, resolution: int, peeling_points: int
) -> tuple[list[dict], dict]:
    coordinates = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel()))
    valid = fluid_mask(cfg, all_points, token)
    points = all_points[valid].astype(np.float32)
    reference_points = metaball_fluid_points(
        cfg, peeling_points, token, 0.0, int(cfg.seed) + 2_100_001
    )
    transform, peeling = vorticity_peeling_transform(
        cfg, model, reference_points, token
    )
    raw_omega, raw_velocity = raw_basis_field_matrices(
        cfg, model, points, token
    )
    peeled_omega, _ = apply_basis_transform(
        raw_omega, raw_velocity, transform
    )
    peeled_omega_np = backend.cp.asnumpy(peeled_omega)
    del raw_omega, raw_velocity, peeled_omega, transform
    backend.cp.get_default_memory_pool().free_all_blocks()
    vorticity = np.full((cfg.modes, resolution * resolution), np.nan, dtype=np.float32)
    metrics: list[dict] = []
    for mode_index in range(int(cfg.modes)):
        omega = peeled_omega_np[:, mode_index]
        vorticity[mode_index, valid] = omega.astype(np.float32, copy=False)
        metrics.append(
            {
                "mode": mode_index + 1,
                "vorticity_rms": float(np.sqrt(np.mean(omega.astype(np.float64) ** 2))),
                "vorticity_abs_max": float(np.max(np.abs(omega))),
            }
        )

    per_page = 64 if int(cfg.modes) <= 64 else 128
    rows, columns = (8, 8) if per_page == 64 else (8, 16)
    for page, start in enumerate(range(0, cfg.modes, per_page), start=1):
        stop = min(start + per_page, cfg.modes)
        figure_size = (8.8, 9.0) if per_page == 64 else (15.8, 8.2)
        fig, axes = plt.subplots(
            rows, columns, figsize=figure_size, constrained_layout=True
        )
        for slot, axis in enumerate(axes.flat):
            mode = start + slot
            if mode >= stop:
                axis.axis("off")
                continue
            image = vorticity[mode].reshape(resolution, resolution)
            finite = np.abs(image[np.isfinite(image)])
            limit = max(float(np.percentile(finite, 99.0)), 1.0e-7)
            axis.imshow(
                image, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r",
                vmin=-limit, vmax=limit, interpolation="bilinear",
            )
            axis.set_title(f"{mode + 1}", fontsize=5, pad=0.5)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            f"Native CUDA Jet3 · peeled SIREN · "
            f"{str(getattr(cfg, 'coordinate_normalization', 'centered_unit')).replace('_', '-')} "
            f"· K={cfg.modes} | "
            f"vorticity modes {start + 1}–{stop}",
            fontsize=10,
        )
        save_figure(
            fig,
            output
            / f"metaball_peeled_modes_{start + 1:04d}_{stop:04d}_landscape",
        )
    return metrics, peeling


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--token",
        type=float,
        help="geometry token; use -1 for the explicit all-zero no-metaball mapping",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--peeling-points", type=int, default=4096)
    args = parser.parse_args()
    checkpoint = args.checkpoint or (
        args.run_dir / "checkpoints" / "hardcoded_sss_frozen.npz"
    )
    output = args.output_dir or (args.run_dir / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    metadata, arrays = load_checkpoint(checkpoint)
    config = dict(metadata["config"])
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    model = backend.HardcodedModel(cfg, arrays)
    token = float(cfg.probe_beta if args.token is None else args.token)

    geometry_preview(cfg, output)
    metrics, peeling = render_modes(
        cfg, model, output, token, args.resolution, args.peeling_points
    )
    with (output / "metaball_mode_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": int(metadata["epoch"]),
        "modes": int(cfg.modes),
        "geometry_token": token,
        "geometry_parameters": geometry_from_token(cfg, token).tolist(),
        "rendered_mode_pages": int(
            np.ceil(int(cfg.modes) / (64 if int(cfg.modes) <= 64 else 128))
        ),
        "mode_resolution": args.resolution,
        "basis_form": "twice-reorthogonalized vorticity-peeled modes",
        "peeling": peeling,
        "precision": "all float32",
        "method": "native CUDA Jet3 + SIREN + raw xy and nine circle parameters",
    }
    (output / "metaball_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
