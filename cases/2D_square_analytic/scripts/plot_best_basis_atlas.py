"""Render every resolved K=64 Ritz basis function for the best SIREN and WIRE runs."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from evaluate_siren_sweep import analytic_pairs, load_checkpoint
from plot_wire_basis import EXPECTED_MODES, evaluate_grid_basis, ritz_rotation
from square_runtime import backend, install_backend


MODES_PER_PAGE = 16
ROWS = 4
COLUMNS = 4

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.0,
    "axes.titlesize": 6.8,
    "axes.labelsize": 6.8,
    "xtick.labelsize": 5.8,
    "ytick.labelsize": 5.8,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})


def validate_checkpoint(
    checkpoint: Path,
    config: dict,
    expected_activation: str,
) -> None:
    if int(config.get("modes", -1)) != EXPECTED_MODES:
        raise ValueError(f"{checkpoint} is not a K=64 checkpoint")
    if config.get("activation") != expected_activation:
        raise ValueError(
            f"{checkpoint} activation is {config.get('activation')!r}, "
            f"expected {expected_activation!r}"
        )
    supported_encodings = (
        {"raw_xy", "raw_xy_padded"}
        if expected_activation == "siren"
        else {"raw_xy_padded"}
    )
    if config.get("input_encoding") not in supported_encodings:
        raise ValueError(f"{checkpoint} is not a supported raw-coordinate run")
    if expected_activation == "wire" and config.get("wire_variant") != "official_real_gabor":
        raise ValueError(f"{checkpoint} is not the official real-Gabor WIRE variant")
    if config.get("early_stopping_enabled") is not False:
        raise ValueError(f"{checkpoint} did not explicitly disable early stopping")
    completion_path = checkpoint.parents[1] / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("stopped_early") is not False:
        raise ValueError(f"{checkpoint} was produced by an early-stopped run")


def load_basis(
    checkpoint: Path,
    validation_pack: Path,
    expected_activation: str,
    grid_size: int,
    chunk_size: int,
) -> dict:
    metadata, arrays = load_checkpoint(checkpoint)
    config = dict(metadata["config"])
    validate_checkpoint(checkpoint, config, expected_activation)
    install_backend(config)
    cfg = SimpleNamespace(**config)
    model = backend.HardcodedModel(cfg, arrays)
    del arrays
    with np.load(validation_pack, allow_pickle=False) as archive:
        validation_points = archive["validation_points"][0].astype(np.float32)
    eigenvalues, rotation = ritz_rotation(model, cfg, validation_points)
    mode_indices = np.arange(EXPECTED_MODES, dtype=np.int64)
    _, _, learned = evaluate_grid_basis(
        model,
        cfg,
        rotation,
        mode_indices,
        grid_size,
        chunk_size,
    )
    del model
    cp.get_default_memory_pool().free_all_blocks()

    exact = np.asarray([item[2] for item in analytic_pairs(EXPECTED_MODES)])
    relative = np.abs(eigenvalues[1:] - exact[1:]) / exact[1:]
    return {
        "activation": expected_activation,
        "config": config,
        "eigenvalues": eigenvalues,
        "basis": learned,
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.percentile(relative, 90.0)),
        "checkpoint": str(checkpoint.resolve()),
        "validation_pack": str(validation_pack.resolve()),
    }


def configuration_line(result: dict) -> str:
    config = result["config"]
    if result["activation"] == "siren":
        if bool(config.get("canonical_siren", False)):
            architecture = (
                f"canonical raw fan-in=2; norm={config['coordinate_normalization']}; "
                f"omega_first={config['siren_first_omega0']:g}, "
                f"omega_hidden={config['siren_hidden_omega0']:g}"
            )
        else:
            architecture = f"legacy omega0={config['siren_first_layer_spatial_omega0']:g}"
    else:
        architecture = (
            f"omega0={config['wire_omega0']:g}, sigma0={config['wire_sigma0']:g}"
        )
    return (
        f"{architecture}; width={config['hidden']}, depth={config['hidden_layers']}, "
        f"LR={config['base_learning_rate']:.0e}, {config['lr_scheduler_type']}, "
        f"epochs={config['epochs']}; median={100.0 * result['median_relative_error']:.2f}%, "
        f"p90={100.0 * result['p90_relative_error']:.2f}%"
    )


def make_page(result: dict, page_index: int) -> tuple[plt.Figure, str]:
    first = page_index * MODES_PER_PAGE
    stop = first + MODES_PER_PAGE
    activation = result["activation"].upper()
    figure, axes = plt.subplots(
        ROWS,
        COLUMNS,
        figsize=(11.69, 8.27),
        squeeze=False,
    )
    image = None
    for local_index, axis in enumerate(axes.ravel()):
        mode_index = first + local_index
        image = axis.imshow(
            result["basis"][..., mode_index],
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        axis.set_title(
            f"mode {mode_index + 1:02d} | lambda={result['eigenvalues'][mode_index]:.2f}",
            pad=2.0,
            weight="bold",
        )
        axis.set_aspect("equal")
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_yticks((0.0, 0.5, 1.0))
        if local_index // COLUMNS == ROWS - 1:
            axis.set_xlabel("x")
        else:
            axis.set_xticklabels([])
        if local_index % COLUMNS == 0:
            axis.set_ylabel("y")
        else:
            axis.set_yticklabels([])
        for spine in axis.spines.values():
            spine.set_color("#666666")
            spine.set_linewidth(0.45)
    if image is None:
        raise RuntimeError("basis atlas page did not contain any modes")
    colorbar_axis = figure.add_axes((0.925, 0.19, 0.015, 0.60))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("vorticity shape (per-mode max normalized)")
    figure.suptitle(
        (
            f"Best {activation} K=64 Ritz basis | modes {first + 1:02d}-{stop:02d}\n"
            f"{configuration_line(result)}"
        ),
        fontsize=10.4,
        weight="bold",
        linespacing=1.25,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.885,
        bottom=0.055,
        top=0.885,
        wspace=0.10,
        hspace=0.20,
    )
    filename = (
        f"{result['activation']}_best_basis_modes_{first + 1:02d}_{stop:02d}.png"
    )
    return figure, filename


def write_atlases(results: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "siren_wire_k64_best_basis_atlas.pdf"
    model_paths = {
        result["activation"]: output_dir / f"{result['activation']}_k64_best_basis_atlas.pdf"
        for result in results
    }
    png_paths: list[str] = []
    with ExitStack() as stack:
        combined_pdf = stack.enter_context(PdfPages(combined_path))
        model_pdfs = {
            name: stack.enter_context(PdfPages(path))
            for name, path in model_paths.items()
        }
        for result in results:
            for page_index in range(EXPECTED_MODES // MODES_PER_PAGE):
                figure, filename = make_page(result, page_index)
                png_path = output_dir / filename
                figure.savefig(png_path, dpi=300, facecolor="white")
                combined_pdf.savefig(figure, facecolor="white")
                model_pdfs[result["activation"]].savefig(figure, facecolor="white")
                png_paths.append(str(png_path.resolve()))
                plt.close(figure)
    return {
        "combined_pdf": str(combined_path.resolve()),
        "model_pdfs": {name: str(path.resolve()) for name, path in model_paths.items()},
        "png_pages": png_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--siren-checkpoint", type=Path, required=True)
    parser.add_argument("--siren-validation-pack", type=Path, required=True)
    parser.add_argument("--wire-checkpoint", type=Path, required=True)
    parser.add_argument("--wire-validation-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=144)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.grid_size < 16 or args.chunk_size < 1:
        parser.error("grid size must be >=16 and chunk size must be positive")

    results = [
        load_basis(
            args.siren_checkpoint,
            args.siren_validation_pack,
            "siren",
            args.grid_size,
            args.chunk_size,
        ),
        load_basis(
            args.wire_checkpoint,
            args.wire_validation_pack,
            "wire",
            args.grid_size,
            args.chunk_size,
        ),
    ]
    artifacts = write_atlases(results, args.output_dir)
    summary = {
        "status": "complete",
        "modes_per_model": EXPECTED_MODES,
        "normalization": "per-mode maximum absolute vorticity",
        "sign_orientation": "positive at deterministic largest-magnitude grid point",
        "models": [
            {
                "activation": result["activation"],
                "config": result["config"],
                "resolved_rank": int(len(result["eigenvalues"])),
                "median_positive_spectrum_relative_error": result["median_relative_error"],
                "p90_positive_spectrum_relative_error": result["p90_relative_error"],
                "checkpoint": result["checkpoint"],
                "validation_pack": result["validation_pack"],
            }
            for result in results
        ],
        "artifacts": artifacts,
    }
    summary_path = args.output_dir / "siren_wire_k64_best_basis_atlas.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**artifacts, "summary": str(summary_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
