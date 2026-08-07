"""Compare fan-basis checkpoints on an independent parameter-domain design.

The benchmark evaluates the same SSS/Rayleigh objective used by training, but
uses quadrature seeds that are disjoint from both training and the fixed
center-geometry validation rule.  It intentionally compares the pre-OneCycle
best checkpoint, the post-OneCycle best checkpoint, and the final checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from serve_interactive import InferenceEngine  # noqa: E402
from eigenfluid_sss.basisnet.parametric_fan3d.hardcoded_backend import (  # noqa: E402
    fluid_points,
    peel_and_adjoint,
    raw_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-best",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_100_run002"
            / "checkpoints"
            / "best_inference.npz"
        ),
    )
    parser.add_argument(
        "--new-best",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_onecycle_run003"
            / "checkpoints"
            / "best_inference.npz"
        ),
    )
    parser.add_argument(
        "--new-final",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_onecycle_run003"
            / "checkpoints"
            / "epoch_14880_inference.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "k1024_manual_onecycle_run003"
            / "full_evaluation"
        ),
    )
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--seed-offset", type=int, default=15_000_000)
    return parser.parse_args()


def design() -> list[tuple[str, str, np.ndarray]]:
    rows: list[tuple[str, str, np.ndarray]] = [
        ("center", "center", np.asarray((0.5, 0.5, 0.5), np.float32))
    ]
    for beta in (0.0, 1.0):
        for chord in (0.0, 1.0):
            for thickness in (0.0, 1.0):
                label = f"corner_b{int(beta)}c{int(chord)}t{int(thickness)}"
                rows.append(
                    (
                        label,
                        "corner",
                        np.asarray((beta, chord, thickness), np.float32),
                    )
                )
    for axis, name in enumerate(("beta", "chord", "thickness")):
        for value, side in ((0.0, "low"), (1.0, "high")):
            parameters = np.full(3, 0.5, dtype=np.float32)
            parameters[axis] = value
            rows.append((f"axis_{name}_{side}", "axis", parameters))
    return rows


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def physical_parameters(cfg: Any, values: np.ndarray) -> tuple[float, float, float]:
    return (
        cfg.beta_tip_min_deg
        + float(values[0]) * (cfg.beta_tip_max_deg - cfg.beta_tip_min_deg),
        cfg.chord_tip_min
        + float(values[1]) * (cfg.chord_tip_max - cfg.chord_tip_min),
        cfg.thickness_min
        + float(values[2]) * (cfg.thickness_max - cfg.thickness_min),
    )


def evaluate_checkpoint(
    label: str,
    checkpoint: Path,
    points_per_case: int,
    seed_count: int,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], int]:
    engine = InferenceEngine(checkpoint.resolve())
    cfg = engine.cfg
    rows: list[dict[str, Any]] = []
    cases = design()
    total = len(cases) * seed_count
    completed = 0
    for case_label, case_group, parameters in cases:
        beta, chord, thickness = physical_parameters(cfg, parameters)
        for seed_index in range(seed_count):
            seed = int(cfg.seed + seed_offset + seed_index)
            points = fluid_points(cfg, points_per_case, parameters, seed)
            raw = raw_fields(cfg, engine.model, points, parameters)
            result = peel_and_adjoint(
                cfg,
                raw,
                len(points),
                need_adjoint=False,
                need_coefficients=False,
            )
            cp.cuda.Stream.null.synchronize()
            rows.append(
                {
                    "checkpoint": label,
                    "checkpoint_path": str(checkpoint.resolve()),
                    "case": case_label,
                    "group": case_group,
                    "seed_index": seed_index,
                    "seed": seed,
                    "points": points_per_case,
                    "beta_normalized": float(parameters[0]),
                    "chord_normalized": float(parameters[1]),
                    "thickness_normalized": float(parameters[2]),
                    "beta_tip_deg": beta,
                    "chord_tip": chord,
                    "thickness_ratio": thickness,
                    "loss": float(result["loss"]),
                    "mean_energy": float(result["mean_energy"]),
                    "max_energy": float(result["max_energy"]),
                    "min_energy": float(result["min_energy"]),
                    "mean_projection_ratio": float(
                        result["mean_projection_ratio"]
                    ),
                    "orth_error": float(result["orth_error"]),
                    "effective_raw_rank": int(result["effective_raw_rank"]),
                }
            )
            completed += 1
            print(
                f"[{label} {completed:02d}/{total:02d}] "
                f"{case_label} seed={seed_index} loss={rows[-1]['loss']:.6f}",
                flush=True,
            )
    epoch = int(
        json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))[
            "epoch"
        ]
    )
    del engine
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return rows, epoch


def aggregate(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        selected = [row for row in rows if row["checkpoint"] == label]
        losses = np.asarray([row["loss"] for row in selected], np.float64)
        groups: dict[str, Any] = {}
        for group in ("center", "axis", "corner"):
            group_losses = np.asarray(
                [row["loss"] for row in selected if row["group"] == group],
                np.float64,
            )
            groups[group] = {
                "count": int(len(group_losses)),
                "mean_loss": float(np.mean(group_losses)),
                "std_loss": float(np.std(group_losses, ddof=1))
                if len(group_losses) > 1
                else 0.0,
                "min_loss": float(np.min(group_losses)),
                "max_loss": float(np.max(group_losses)),
            }
        result[label] = {
            "count": len(selected),
            "mean_loss": float(np.mean(losses)),
            "median_loss": float(np.median(losses)),
            "std_loss": float(np.std(losses, ddof=1)),
            "min_loss": float(np.min(losses)),
            "max_loss": float(np.max(losses)),
            "maximum_orthogonality_error": float(
                max(row["orth_error"] for row in selected)
            ),
            "minimum_effective_raw_rank": int(
                min(row["effective_raw_rank"] for row in selected)
            ),
            "groups": groups,
        }
    return result


def comparisons(rows: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    keyed = {
        (row["checkpoint"], row["case"], row["seed_index"]): row["loss"]
        for row in rows
    }
    output: dict[str, Any] = {}
    for baseline, candidate in (
        (labels[0], labels[1]),
        (labels[0], labels[2]),
        (labels[1], labels[2]),
    ):
        gains = []
        for case_label, _, _ in design():
            for seed_index in sorted(
                row["seed_index"]
                for row in rows
                if row["checkpoint"] == baseline and row["case"] == case_label
            ):
                old = keyed[(baseline, case_label, seed_index)]
                new = keyed[(candidate, case_label, seed_index)]
                gains.append(100.0 * (old - new) / old)
        values = np.asarray(gains, np.float64)
        output[f"{candidate}_vs_{baseline}"] = {
            "positive_means_candidate_is_better": True,
            "mean_improvement_percent": float(np.mean(values)),
            "median_improvement_percent": float(np.median(values)),
            "minimum_improvement_percent": float(np.min(values)),
            "maximum_improvement_percent": float(np.max(values)),
            "win_fraction": float(np.mean(values > 0.0)),
        }
    return output


def make_plot(rows: list[dict[str, Any]], labels: list[str], output: Path) -> None:
    cases = [case_label for case_label, _, _ in design()]
    x = np.arange(len(cases))
    colors = ("#7A7A7A", "#2474B5", "#6CB7D9")
    markers = ("o", "s", "^")
    fig, ax = plt.subplots(figsize=(13.2, 5.8), constrained_layout=True)
    for label, color, marker in zip(labels, colors, markers):
        means = []
        errors = []
        for case in cases:
            values = np.asarray(
                [
                    row["loss"]
                    for row in rows
                    if row["checkpoint"] == label and row["case"] == case
                ]
            )
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        ax.errorbar(
            x,
            means,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5.0,
            capsize=2.5,
        )
    ax.set_ylabel("Independent SSS validation loss")
    ax.set_xticks(x, cases, rotation=42, ha="right")
    ax.grid(axis="y", color="#D8E5EF", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.points * 3 < 1024:
        raise ValueError("points must provide at least 1024 velocity DOFs")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        ("pre_onecycle_best", args.old_best),
        ("onecycle_best", args.new_best),
        ("onecycle_final", args.new_final),
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    epochs: dict[str, int] = {}
    for label, checkpoint in checkpoints:
        checkpoint_rows, epoch = evaluate_checkpoint(
            label,
            checkpoint,
            args.points,
            args.seeds,
            args.seed_offset,
        )
        rows.extend(checkpoint_rows)
        epochs[label] = epoch

    csv_path = args.output_dir / "parameter_domain_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [label for label, _ in checkpoints]
    summary = {
        "status": "complete",
        "definition": {
            "objective": "independent SSS/Rayleigh validation",
            "design": "center + 8 parameter-box corners + 6 one-axis boundary points",
            "parameter_cases": len(design()),
            "independent_seeds_per_case": args.seeds,
            "points_per_case": args.points,
            "seed_offset": args.seed_offset,
            "training_validation_seed_offset": 9_000_000,
            "lower_loss_is_better": True,
        },
        "checkpoint_epochs": epochs,
        "aggregate": aggregate(rows, labels),
        "paired_comparisons": comparisons(rows, labels),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "rows_csv": str(csv_path.resolve()),
            "figure": str(
                (args.output_dir / "parameter_domain_benchmark.png").resolve()
            ),
        },
    }
    make_plot(
        rows,
        labels,
        args.output_dir / "parameter_domain_benchmark.png",
    )
    atomic_json(args.output_dir / "parameter_domain_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
