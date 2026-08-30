"""Render representative learned K=64 WIRE Ritz vorticity basis functions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_siren_sweep import load_checkpoint, resolved_eigensystem
from square_runtime import backend, install_backend


EXPECTED_MODES = 64
DEFAULT_DISPLAY_MODES = (1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 56, 64)
RED = "#B2182B"
BLUE = "#2166AC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.2,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def raw_vorticity_and_gradient(
    model,
    cfg: SimpleNamespace,
    points: np.ndarray,
    include_gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    count = len(points)
    raw_vorticity = cp.empty((count, EXPECTED_MODES), dtype=backend.F32)
    raw_gradient = (
        cp.empty((2 * count, EXPECTED_MODES), dtype=backend.F32)
        if include_gradient else None
    )
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, 0.0, 0.0, False)
    ):
        vorticity, gradient = backend.vorticity_and_gradient(field)
        raw_vorticity[:, index] = vorticity[:, 0]
        if raw_gradient is not None:
            raw_gradient[:, index] = gradient[:, 0].reshape(-1)
    vorticity_host = cp.asnumpy(raw_vorticity)
    gradient_host = cp.asnumpy(raw_gradient) if raw_gradient is not None else None
    del raw_vorticity, raw_gradient
    cp.get_default_memory_pool().free_all_blocks()
    return vorticity_host, gradient_host


def ritz_rotation(
    model,
    cfg: SimpleNamespace,
    validation_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_vorticity, raw_gradient = raw_vorticity_and_gradient(
        model, cfg, validation_points, include_gradient=True
    )
    if raw_gradient is None:
        raise RuntimeError("missing vorticity gradient")
    weight = np.float32(1.0 / len(validation_points))
    mass = weight * (raw_vorticity.T @ raw_vorticity)
    stiffness = weight * (raw_gradient.T @ raw_gradient)
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, rotation, _, _ = resolved_eigensystem(mass, stiffness)
    if len(eigenvalues) != EXPECTED_MODES:
        raise ValueError(
            f"basis checkpoint resolves rank {len(eigenvalues)}, expected K=64"
        )
    return eigenvalues, rotation


def evaluate_grid_basis(
    model,
    cfg: SimpleNamespace,
    rotation: np.ndarray,
    mode_indices: np.ndarray,
    grid_size: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
    selected_rotation = rotation[:, mode_indices].astype(np.float32)
    learned = np.empty((len(points), len(mode_indices)), dtype=np.float32)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        raw, _ = raw_vorticity_and_gradient(
            model, cfg, points[start:stop], include_gradient=False
        )
        learned[start:stop] = raw @ selected_rotation
    learned = learned.reshape(grid_size, grid_size, len(mode_indices))
    for index in range(learned.shape[-1]):
        values = learned[..., index]
        pivot = np.unravel_index(np.argmax(np.abs(values)), values.shape)
        if values[pivot] < 0.0:
            learned[..., index] *= -1.0
        amplitude = float(np.max(np.abs(learned[..., index])))
        if amplitude > 0.0:
            learned[..., index] /= amplitude
    return grid_x, grid_y, learned


def plot_basis(
    learned: np.ndarray,
    eigenvalues: np.ndarray,
    display_modes: tuple[int, ...],
    config: dict,
    output_dir: Path,
) -> None:
    columns = 4
    rows = int(np.ceil(len(display_modes) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(9.6, 6.7),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for panel_index, axis in enumerate(axes.ravel()):
        if panel_index >= len(display_modes):
            axis.axis("off")
            continue
        mode = display_modes[panel_index]
        image = axis.imshow(
            learned[..., panel_index],
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        axis.set_title(
            f"mode {mode} | lambda={eigenvalues[mode - 1]:.2f}",
            pad=3.0,
            weight="bold",
        )
        axis.set_aspect("equal")
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_yticks((0.0, 0.5, 1.0))
        if panel_index // columns == rows - 1:
            axis.set_xlabel("x")
        else:
            axis.set_xticklabels([])
        if panel_index % columns == 0:
            axis.set_ylabel("y")
        else:
            axis.set_yticklabels([])
        for spine in axis.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.5)
    if image is None:
        raise ValueError("no display modes were requested")
    colorbar = fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        shrink=0.76,
        pad=0.015,
        aspect=28,
    )
    colorbar.set_label("vorticity shape (per-mode max normalized)")
    fig.suptitle(
        (
            "Recommended K=64 real-Gabor WIRE: representative learned Ritz basis\n"
            f"omega0={config['wire_omega0']:g}, sigma0={config['wire_sigma0']:g}, "
            f"width={config['hidden']}, depth={config['hidden_layers']}, "
            f"LR={config['base_learning_rate']:.0e}; deterministic sign orientation"
        ),
        fontsize=11.5,
        weight="bold",
        linespacing=1.25,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "wire_k64_practical_basis.png", dpi=320, bbox_inches="tight")
    fig.savefig(output_dir / "wire_k64_practical_basis.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=144)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--display-modes",
        type=int,
        nargs="+",
        default=list(DEFAULT_DISPLAY_MODES),
    )
    args = parser.parse_args()
    display_modes = tuple(args.display_modes)
    if not display_modes or any(mode < 1 or mode > EXPECTED_MODES for mode in display_modes):
        parser.error("--display-modes must contain values in [1, 64]")
    if args.grid_size < 16 or args.chunk_size < 1:
        parser.error("grid size must be >=16 and chunk size must be positive")

    metadata, arrays = load_checkpoint(args.checkpoint)
    config = dict(metadata["config"])
    if (
        int(config.get("modes", -1)) != EXPECTED_MODES
        or config.get("activation") != "wire"
        or config.get("wire_variant") != "official_real_gabor"
    ):
        raise ValueError("checkpoint is not the expected K=64 real-Gabor WIRE model")
    if config.get("early_stopping_enabled") is not False:
        raise ValueError("basis checkpoint did not explicitly disable early stopping")
    completion = args.checkpoint.parents[1] / "completion.json"
    completion_data = json.loads(completion.read_text(encoding="utf-8"))
    if completion_data.get("stopped_early") is not False:
        raise ValueError("basis checkpoint came from an early-stopped run")

    install_backend(config)
    cfg = SimpleNamespace(**config)
    model = backend.HardcodedModel(cfg, arrays)
    del arrays
    with np.load(args.validation_pack, allow_pickle=False) as archive:
        validation_points = archive["validation_points"][0].astype(np.float32)
    eigenvalues, rotation = ritz_rotation(model, cfg, validation_points)
    indices = np.asarray(display_modes, dtype=np.int64) - 1
    _, _, learned = evaluate_grid_basis(
        model, cfg, rotation, indices, args.grid_size, args.chunk_size
    )
    plot_basis(learned, eigenvalues, display_modes, config, args.output_dir)
    del model
    cp.get_default_memory_pool().free_all_blocks()
    print(json.dumps({
        "status": "complete",
        "display_modes": list(display_modes),
        "resolved_rank": int(len(eigenvalues)),
        "output_png": str((args.output_dir / "wire_k64_practical_basis.png").resolve()),
        "output_pdf": str((args.output_dir / "wire_k64_practical_basis.pdf").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
