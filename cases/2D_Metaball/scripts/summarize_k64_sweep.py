"""Consolidate the K=64 metaball sweep and independent unseen-flow checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2166AC"
RED = "#B2182B"
LIGHT_BLUE = "#D7E8F4"
CHARCOAL = "#262626"


def validation_rows(run_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(
        run_dir.glob(
            "validation_unseen_peeled*/"
            "epoch0500_k64_peeled_siren_vs_laplacian.json"
        )
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        siren, laplacian = payload["methods"]
        rows.append(
            {
                "geometry_token": payload["geometry_token"],
                "siren": siren,
                "laplacian": laplacian,
                "peeling": payload["peeling"],
                "source": str(path.resolve()),
            }
        )
    return rows


def stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.result_root.resolve()
    manifest = json.loads((root / "sweep_manifest.json").read_text(encoding="utf-8"))
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)

    evaluated = []
    for arm in manifest["stage_2_arms"]:
        rows = validation_rows(Path(arm["run_dir"]))
        if not rows:
            continue
        siren_vorticity = [r["siren"]["test_vorticity_relative_l2"] for r in rows]
        siren_velocity = [r["siren"]["test_velocity_relative_l2"] for r in rows]
        siren_rank = [r["siren"]["fit_gram_effective_rank"] for r in rows]
        peeled_rank = [r["peeling"]["peeled_effective_rank"] for r in rows]
        raw_rank = [r["peeling"]["raw_effective_rank"] for r in rows]
        orthogonality = [
            r["peeling"]["peeled_orthogonality_max_abs"] for r in rows
        ]
        lap_vorticity = [r["laplacian"]["test_vorticity_relative_l2"] for r in rows]
        lap_velocity = [r["laplacian"]["test_velocity_relative_l2"] for r in rows]
        item = {
            **arm,
            "independent_case_count": len(rows),
            "independent_vorticity_relative_l2": stats(siren_vorticity),
            "independent_velocity_relative_l2": stats(siren_velocity),
            "independent_fit_gram_rank_min": int(min(siren_rank)),
            "independent_fit_gram_rank_max": int(max(siren_rank)),
            "peeling_effective_rank_min": int(min(peeled_rank)),
            "peeling_effective_rank_max": int(max(peeled_rank)),
            "raw_generator_rank_min": int(min(raw_rank)),
            "raw_generator_rank_max": int(max(raw_rank)),
            "peeled_orthogonality_max_abs": float(max(orthogonality)),
            "laplacian_vorticity_relative_l2": stats(lap_vorticity),
            "laplacian_velocity_relative_l2": stats(lap_velocity),
            "independent_cases": rows,
        }
        evaluated.append(item)

    full_rank = [
        item for item in evaluated
        if item["peeling_effective_rank_min"] == 64
        and item["independent_fit_gram_rank_min"] == 64
    ]
    winner = min(
        full_rank,
        key=lambda item: item["independent_vorticity_relative_l2"]["mean"],
    )
    winner_checkpoint = (
        Path(winner["run_dir"]) / "checkpoints/hardcoded_sss_frozen.npz"
    ).resolve()
    summary = {
        "status": "complete",
        "selection_rule": (
            "Require rank 64 for every peeled Gram and independent fit Gram, then minimize "
            "mean held-out vorticity relative L2 over four unseen geometries/flows; "
            "velocity error is secondary."
        ),
        "validated_winner": winner,
        "validated_winner_checkpoint": str(winner_checkpoint),
        "all_stage_2_independent_results": evaluated,
        "rank_interpretation": {
            "training_diagnostic": (
                "raw vorticity Gram eigenvalue > eps_float32 * max(points,modes) "
                "* largest eigenvalue (about 4.88e-4 relative at 4096 points)"
            ),
            "independent_fit": (
                "fit Gram eigenvalue > 1e-7 * largest eigenvalue"
            ),
            "note": (
                "Training raw rank describes the pre-peeling generator. The "
                "recomputed validation applies the same two-pass MGS transform as "
                "training and separately reports peeled rank and orthogonality."
            ),
        },
    }
    (output / "final_sweep_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    compact_rows = []
    for item in evaluated:
        compact_rows.append(
            {
                "normalization": item["normalization"],
                "learning_rate": item["learning_rate"],
                "hidden_omega0": item["siren_hidden_omega0"],
                "training_validation_rank": item["final_validation_rank"],
                "training_validation_loss": item["final_validation_loss"],
                "independent_cases": item["independent_case_count"],
                "independent_rank_min": item["independent_fit_gram_rank_min"],
                "peeled_rank_min": item["peeling_effective_rank_min"],
                "raw_generator_rank_min": item["raw_generator_rank_min"],
                "peeled_orthogonality_max_abs": item[
                    "peeled_orthogonality_max_abs"
                ],
                "test_vorticity_relative_l2_mean": item[
                    "independent_vorticity_relative_l2"
                ]["mean"],
                "test_velocity_relative_l2_mean": item[
                    "independent_velocity_relative_l2"
                ]["mean"],
            }
        )
    with (output / "final_sweep_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), constrained_layout=True)
    hidden = manifest["stage_1_arms"]
    omega = np.asarray([item["siren_hidden_omega0"] for item in hidden])
    rank = np.asarray([item["final_validation_rank"] for item in hidden])
    loss = np.asarray([item["final_validation_loss"] for item in hidden]) / 1.0e4
    axis = axes[0, 0]
    axis.plot(omega, rank, "o-", color=BLUE, label="effective rank")
    axis.set_xlabel(r"Hidden $\omega_0$")
    axis.set_ylabel("Training diagnostic rank", color=BLUE)
    axis.tick_params(axis="y", colors=BLUE)
    twin = axis.twinx()
    twin.plot(omega, loss, "s--", color=RED, label="validation loss")
    twin.set_ylabel(r"Validation loss ($\times 10^4$)", color=RED)
    twin.tick_params(axis="y", colors=RED)
    axis.set_title("a  Hidden-frequency sweep", loc="left", weight="bold")

    norms = ["unit_interval", "centered_unit", "standardized_uniform"]
    lrs = [1.0e-4, 2.0e-4, 4.0e-4]
    rank_grid = np.full((3, 3), np.nan)
    loss_grid = np.full((3, 3), np.nan)
    for item in manifest["stage_2_arms"]:
        i, j = norms.index(item["normalization"]), lrs.index(item["learning_rate"])
        rank_grid[i, j] = item["final_validation_rank"]
        loss_grid[i, j] = item["final_validation_loss"] / 1.0e4
    for axis, grid, title, cmap, fmt in (
        (axes[0, 1], rank_grid, "b  Final raw-generator rank", "Blues", ".0f"),
        (axes[1, 0], loss_grid, "c  Validation Rayleigh loss", "Reds", ".2f"),
    ):
        image = axis.imshow(grid, cmap=cmap, aspect="auto")
        for i in range(3):
            for j in range(3):
                axis.text(j, i, format(grid[i, j], fmt), ha="center", va="center")
        axis.set_xticks(range(3), ["1e-4", "2e-4", "4e-4"])
        axis.set_yticks(range(3), ["unit", "centered", "standardized"])
        axis.set_xlabel("Learning rate")
        axis.set_title(title, loc="left", weight="bold")
        fig.colorbar(image, ax=axis, fraction=0.045)

    axis = axes[1, 1]
    top = [item for item in evaluated if item["learning_rate"] == 4.0e-4]
    labels = [item["normalization"].replace("_uniform", "").replace("_", " ") for item in top]
    x = np.arange(len(top))
    width = 0.36
    vort = [item["independent_vorticity_relative_l2"]["mean"] for item in top]
    vel = [item["independent_velocity_relative_l2"]["mean"] for item in top]
    axis.bar(x - width / 2, vort, width, color=BLUE, label="vorticity")
    axis.bar(x + width / 2, vel, width, color=RED, label="velocity")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean held-out relative $L_2$")
    axis.set_title("d  Peeled basis on four unseen geometries", loc="left", weight="bold")
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "SIREN-K64 metaball normalization, learning-rate and hidden-frequency sweep",
        fontsize=14,
        weight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"metaball_k64_sweep_summary.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
