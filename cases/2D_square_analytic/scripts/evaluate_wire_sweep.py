"""Evaluate the K=64 real-Gabor WIRE sweep against the square spectrum."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import linear_sum_assignment

from evaluate_siren_sweep import (
    analytic_pairs,
    clean,
    load_checkpoint,
    neumann_modes,
    resolved_eigensystem,
)
from square_runtime import backend, install_backend


RED = "#B2182B"
BLUE = "#2166AC"
LIGHT_BLUE = "#67A9CF"
GRAY = "#666666"
CONTROL = "control_w10_s10"
EXPECTED_MODES = 64

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,
})


def parameter_count(config: dict) -> int:
    """Count both independent affine projections in every RealGaborLayer."""
    width = int(config["hidden"])
    depth = int(config["hidden_layers"])
    input_width = 2
    per_mode = 2 * width * input_width + 2 * width
    per_mode += max(depth - 1, 0) * (2 * width * width + 2 * width)
    per_mode += width + 1
    return EXPECTED_MODES * per_mode


def evaluate_one(campaign: Path, name: str, group: str) -> dict:
    root = campaign / name
    checkpoint = root / "run_001/checkpoints/hardcoded_sss_frozen.npz"
    validation = root / "inputs/fixed_validation_scipy_sobol_v1.npz"
    metadata, arrays = load_checkpoint(checkpoint)
    config = dict(metadata["config"])
    if int(config["modes"]) != EXPECTED_MODES:
        raise ValueError(f"{name} used K={config['modes']}, expected K=64")
    if (
        config.get("activation") != "wire"
        or config.get("wire_variant") != "official_real_gabor"
        or config.get("input_encoding") != "raw_xy_padded"
    ):
        raise ValueError(f"{name} is not the raw-coordinate real-Gabor WIRE arm")
    if config.get("early_stopping_enabled") is not False:
        raise ValueError(f"{name} did not explicitly disable early stopping")
    completion = json.loads((root / "run_001/completion.json").read_text(encoding="utf-8"))
    if completion.get("stopped_early") is not False:
        raise ValueError(f"{name} stopped early")

    install_backend(config)
    cfg = SimpleNamespace(**config)
    model = backend.HardcodedModel(cfg, arrays)
    del arrays
    with np.load(validation, allow_pickle=False) as archive:
        points = archive["validation_points"][0].astype(np.float32)
    count = len(points)
    raw_omega = cp.empty((count, EXPECTED_MODES), dtype=backend.F32)
    raw_gradient = cp.empty((2 * count, EXPECTED_MODES), dtype=backend.F32)
    for index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, 0.0, 0.0, False)
    ):
        omega, gradient = backend.vorticity_and_gradient(field)
        raw_omega[:, index] = omega[:, 0]
        raw_gradient[:, index] = gradient[:, 0].reshape(-1)
    weight = np.float32(1.0 / count)
    mass = cp.asnumpy(weight * (raw_omega.T @ raw_omega))
    stiffness = cp.asnumpy(weight * (raw_gradient.T @ raw_gradient))
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    eigenvalues, rotation, mass_values, threshold = resolved_eigensystem(mass, stiffness)

    exact = np.asarray([item[2] for item in analytic_pairs(len(eigenvalues))])
    relative = np.full_like(exact, np.nan)
    relative[1:] = np.abs(eigenvalues[1:] - exact[1:]) / exact[1:]
    compare = min(len(eigenvalues), EXPECTED_MODES)
    exact_modes = neumann_modes(points, analytic_pairs(compare))
    learned_modes = cp.asnumpy(
        raw_omega @ cp.asarray(rotation[:, :compare], dtype=backend.F32)
    )
    overlap = (1.0 / count) * exact_modes.T @ learned_modes
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    matched_overlap = np.abs(overlap[rows, columns])
    del model, raw_omega, raw_gradient
    cp.get_default_memory_pool().free_all_blocks()

    history = json.loads(
        (root / "run_001/data/hardcoded_training_history.json").read_text(encoding="utf-8")
    )
    final = history[-1]
    validation_rows = [row for row in history if "validation_loss" in row]
    reach = (
        float(eigenvalues[EXPECTED_MODES - 1] / analytic_pairs(EXPECTED_MODES)[-1][2])
        if len(eigenvalues) >= EXPECTED_MODES else float("nan")
    )
    return {
        "experiment": name,
        "group": group,
        "config": {
            "omega0": float(config["wire_omega0"]),
            "sigma0": float(config["wire_sigma0"]),
            "learning_rate": float(config["base_learning_rate"]),
            "scheduler": str(config["lr_scheduler_type"]),
            "width": int(config["hidden"]),
            "depth": int(config["hidden_layers"]),
            "epochs": int(config["epochs"]),
            "modes": int(config["modes"]),
            "parameter_count": parameter_count(config),
            "early_stopping_enabled": False,
        },
        "resolved_rank": int(len(eigenvalues)),
        "final_batch_rank": int(final["resolved_raw_rank"]),
        "final_validation_loss": float(validation_rows[-1]["validation_loss"]),
        "training_seconds": float(final["seconds"]),
        "mass_threshold": float(threshold),
        "min_resolved_mass_eigenvalue": float(mass_values[mass_values > threshold][0]),
        "median_positive_spectrum_relative_error": float(np.nanmedian(relative[1:])),
        "p90_positive_spectrum_relative_error": float(np.nanpercentile(relative[1:], 90.0)),
        "modes_within_20pct": int(np.sum(relative[1:] <= 0.2)),
        "spectral_reach_ratio_mode64": reach,
        "mean_absolute_mode_overlap": float(np.mean(matched_overlap)),
        "minimum_absolute_mode_overlap": float(np.min(matched_overlap)),
        "learned_eigenvalues": eigenvalues.tolist(),
        "analytic_eigenvalues": exact.tolist(),
        "validation_history": [
            {
                "epoch": int(row["epoch"]),
                "validation_loss": float(row["validation_loss"]),
                "learning_rate": float(row["learning_rate"]),
                "validation_rank": int(row["validation_min_rank"]),
            }
            for row in validation_rows
        ],
    }


def rank_key(item: dict) -> tuple:
    return (
        -item["resolved_rank"],
        item["median_positive_spectrum_relative_error"],
        item["p90_positive_spectrum_relative_error"],
    )


def plot_gabor_grid(items: dict[str, dict], output: Path) -> None:
    subset = [item for item in items.values() if item["group"] in {"gabor", "control"}]
    omegas = sorted({item["config"]["omega0"] for item in subset})
    sigmas = sorted({item["config"]["sigma0"] for item in subset})
    errors = np.full((len(sigmas), len(omegas)), np.nan)
    ranks = np.zeros_like(errors, dtype=int)
    for item in subset:
        i = sigmas.index(item["config"]["sigma0"])
        j = omegas.index(item["config"]["omega0"])
        errors[i, j] = 100.0 * item["median_positive_spectrum_relative_error"]
        ranks[i, j] = item["resolved_rank"]
    fig, axis = plt.subplots(figsize=(5.6, 3.7), constrained_layout=True)
    finite = errors[np.isfinite(errors) & (errors > 0.0)]
    image = axis.imshow(
        errors,
        cmap="RdBu_r",
        norm=LogNorm(vmin=float(np.min(finite)), vmax=float(np.max(finite))),
        aspect="auto",
    )
    for i in range(len(sigmas)):
        for j in range(len(omegas)):
            error_label = (
                f"{errors[i, j]:.1f}%"
                if errors[i, j] < 10_000.0 else f"{errors[i, j]:.1e}%"
            )
            normalized = float(image.norm(errors[i, j]))
            text_color = "white" if normalized < 0.2 or normalized > 0.78 else "black"
            axis.text(
                j, i, f"{error_label}\nr={ranks[i, j]}",
                ha="center", va="center", fontsize=8, color=text_color,
            )
    axis.set_xticks(range(len(omegas)), [f"{value:g}" for value in omegas])
    axis.set_yticks(range(len(sigmas)), [f"{value:g}" for value in sigmas])
    axis.set_xlabel("Gabor frequency omega0")
    axis.set_ylabel("Gaussian scale sigma0")
    axis.set_title("K=64 WIRE Gabor parameter screen", loc="left", weight="bold")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("median spectrum error (%)")
    fig.savefig(output / "wire_k64_gabor_grid.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "wire_k64_gabor_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def _comparison_names(items: dict[str, dict], groups: set[str]) -> list[str]:
    return [CONTROL] + [
        name for name, item in items.items()
        if item["group"] in groups and name != CONTROL
    ]


def plot_ablation(items: dict[str, dict], output: Path) -> None:
    panels = [
        ("Learning rate", {"learning_rate"}),
        ("LR scheduler", {"scheduler"}),
        ("MLP width", {"width"}),
        ("MLP depth", {"depth"}),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    for panel, (title, groups) in zip(axes.ravel(), panels):
        names = _comparison_names(items, groups)
        labels, errors, ranks = [], [], []
        for name in names:
            item = items[name]
            cfg = item["config"]
            if title == "Learning rate":
                label = f"{cfg['learning_rate']:.0e}"
            elif title == "LR scheduler":
                label = cfg["scheduler"].replace("validation_", "")
            elif title == "MLP width":
                label = f"w{cfg['width']}"
            else:
                label = f"d{cfg['depth']}"
            labels.append(label)
            errors.append(100.0 * item["median_positive_spectrum_relative_error"])
            ranks.append(item["resolved_rank"])
        x = np.arange(len(names))
        colors = [BLUE if name == CONTROL else LIGHT_BLUE for name in names]
        bars = panel.bar(x, errors, color=colors, width=0.68, zorder=3)
        panel.set_title(title, loc="left", weight="bold")
        panel.set_ylabel("median spectrum error (%)")
        panel.set_xticks(x, labels)
        for bar, rank in zip(bars, ranks):
            panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"r={rank}", ha="center", va="bottom", fontsize=6.7, color=GRAY)
        clean(panel)
    fig.suptitle("K=64 real-Gabor WIRE ablations", fontsize=12, weight="bold")
    fig.savefig(output / "wire_k64_ablation.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "wire_k64_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spectra(items: dict[str, dict], output: Path) -> None:
    ranked = sorted(items.values(), key=rank_key)
    best = ranked[0]
    control = items.get(CONTROL, ranked[0])
    gabor_items = [
        item for item in items.values() if item["group"] in {"gabor", "control"}
    ]
    worst_gabor = sorted(gabor_items, key=rank_key)[-1] if gabor_items else ranked[-1]
    exact = np.asarray(best["analytic_eigenvalues"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    axes[0].plot(np.arange(1, len(exact) + 1), exact, color="black", label="analytic Neumann")
    plotted: list[str] = []
    for item, color, label in (
        (worst_gabor, RED, f"Gabor extreme: {worst_gabor['experiment']}"),
        (control, GRAY, "official real-layer defaults"),
        (best, BLUE, f"best: {best['experiment']}"),
    ):
        if item["experiment"] in plotted:
            continue
        plotted.append(item["experiment"])
        learned = np.asarray(item["learned_eigenvalues"])
        axes[0].plot(np.arange(1, len(learned) + 1), learned, color=color, label=label)
        relative = np.abs(learned[1:] - exact[1:len(learned)]) / exact[1:len(learned)]
        axes[1].plot(np.arange(2, len(learned) + 1), relative, color=color, label=label)
    axes[0].set(title="Ritz spectrum", xlabel="mode index", ylabel="eigenvalue")
    axes[1].set(title="Positive-spectrum error", xlabel="mode index", ylabel="relative error", yscale="log")
    for panel in axes:
        clean(panel)
        panel.legend(frameon=False)
    fig.savefig(output / "wire_k64_spectrum.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "wire_k64_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    state = json.loads((args.campaign_root / "campaign_state.json").read_text(encoding="utf-8"))
    if not state.get("complete") or int(state.get("modes", -1)) != EXPECTED_MODES:
        raise ValueError("campaign is incomplete or is not K=64")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    items = {
        name: evaluate_one(args.campaign_root, name, record["group"])
        for name, record in state["experiments"].items()
    }
    ranked = sorted(items.values(), key=rank_key)
    summary = {
        "status": "complete",
        "domain": "[0,1]^2",
        "modes": EXPECTED_MODES,
        "activation": "wire",
        "wire_variant": "official_real_gabor",
        "input_encoding": "raw_xy_padded",
        "early_stopping_enabled": False,
        "selection_rule": "maximize resolved rank, then minimize median and p90 positive-spectrum relative error",
        "best_experiment": ranked[0]["experiment"],
        "analytic_reference": "lambda_mn=pi^2*(m^2+n^2), m,n>=0 (Neumann vorticity spectrum)",
        "experiments": items,
    }
    (args.output_dir / "wire_k64_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    scalar_fields = [
        "experiment", "group", "resolved_rank", "final_batch_rank",
        "final_validation_loss", "training_seconds",
        "median_positive_spectrum_relative_error",
        "p90_positive_spectrum_relative_error", "modes_within_20pct",
        "spectral_reach_ratio_mode64", "mean_absolute_mode_overlap",
    ]
    with (args.output_dir / "wire_k64_sweep_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=scalar_fields + ["omega0", "sigma0", "learning_rate", "scheduler", "width", "depth", "parameter_count"],
        )
        writer.writeheader()
        for item in ranked:
            row = {field: item[field] for field in scalar_fields}
            row.update({key: item["config"][key] for key in ("omega0", "sigma0", "learning_rate", "scheduler", "width", "depth", "parameter_count")})
            writer.writerow(row)
    if not args.summary_only:
        has_gabor_grid = any(
            item["group"] in {"gabor", "control"} for item in items.values()
        )
        if has_gabor_grid:
            plot_gabor_grid(items, args.output_dir)
        if CONTROL in items and any(
            item["group"] in {"learning_rate", "scheduler", "width", "depth"}
            for item in items.values()
        ):
            plot_ablation(items, args.output_dir)
        plot_spectra(items, args.output_dir)
    print(json.dumps({"status": "complete", "best": ranked[0]["experiment"]}, indent=2))


if __name__ == "__main__":
    main()
