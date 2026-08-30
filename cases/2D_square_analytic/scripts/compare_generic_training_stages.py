"""Compare the 500- and 1000-epoch generic peeled-Ritz square bases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#252525"
LIGHT = "#F2F2F2"


def load_stage(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads(
        (directory / "square_k64_generic_peeled_basis_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with np.load(
        directory / "square_k64_generic_peeled_basis_fields.npz",
        allow_pickle=False,
    ) as archive:
        fields = {name: archive[name] for name in archive.files}
    if bool(summary.get("analytic_orientation_used", True)):
        raise ValueError("comparison refuses analytically oriented bases")
    if int(summary["peeled_rank"]) != 64:
        raise ValueError("comparison requires peeled rank 64")
    return summary, fields


def stage_overlap(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    first_matrix = first.reshape(first.shape[0], -1).T.astype(np.float64)
    second_matrix = second.reshape(second.shape[0], -1).T.astype(np.float64)
    first_q = np.linalg.qr(first_matrix, mode="reduced")[0]
    second_q = np.linalg.qr(second_matrix, mode="reduced")[0]
    singular = np.linalg.svd(first_q.T @ second_q, compute_uv=False)
    minimum = float(np.min(singular))
    return {
        "mean_squared_cosine": float(np.mean(singular**2)),
        "minimum_cosine": minimum,
        "maximum_principal_angle_deg": float(
            np.degrees(np.arccos(np.clip(minimum, 0.0, 1.0)))
        ),
    }


def relative_change(before: float, after: float) -> float:
    return 100.0 * (after - before) / before


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch500-dir", type=Path, required=True)
    parser.add_argument("--epoch1000-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary500, fields500 = load_stage(args.epoch500_dir)
    summary1000, fields1000 = load_stage(args.epoch1000_dir)
    learned500 = fields500["learned_eigenvalues"].astype(np.float64)
    learned1000 = fields1000["learned_eigenvalues"].astype(np.float64)
    analytic = fields500["analytic_eigenvalues_benchmark_only"].astype(np.float64)
    if not np.array_equal(
        analytic,
        fields1000["analytic_eigenvalues_benchmark_only"].astype(np.float64),
    ):
        raise ValueError("analytic benchmark arrays differ between stages")
    error500 = np.abs(learned500 - analytic) / analytic
    error1000 = np.abs(learned1000 - analytic) / analytic
    improved = error1000 < error500
    overlap = stage_overlap(
        fields500["streamfunction"], fields1000["streamfunction"]
    )

    spectrum500 = summary500["spectrum"]
    spectrum1000 = summary1000["spectrum"]
    subspace500 = summary500["analytic_benchmark_only"]
    subspace1000 = summary1000["analytic_benchmark_only"]
    comparison = {
        "comparison": "epoch 500 vs epoch 1000",
        "same_model_continuation": True,
        "sampling": "IID uniform random; no D4 augmentation",
        "causal_sorting": "dynamic geometry Rayleigh",
        "analytic_orientation_used": False,
        "evaluation_points_per_stage": int(
            summary500["independent_evaluation_points"]
        ),
        "peeled_rank": {
            "epoch500": int(summary500["peeled_rank"]),
            "epoch1000": int(summary1000["peeled_rank"]),
        },
        "spectrum": {
            "epoch500": spectrum500,
            "epoch1000": spectrum1000,
            "median_error_relative_change_percent": relative_change(
                spectrum500["median_relative_error"],
                spectrum1000["median_relative_error"],
            ),
            "p90_error_relative_change_percent": relative_change(
                spectrum500["p90_relative_error"],
                spectrum1000["p90_relative_error"],
            ),
            "max_error_relative_change_percent": relative_change(
                spectrum500["max_relative_error"],
                spectrum1000["max_relative_error"],
            ),
            "modes_improved": int(np.count_nonzero(improved)),
            "modes_worsened": int(np.count_nonzero(~improved)),
        },
        "analytic_subspace_benchmark_only": {
            "epoch500": subspace500,
            "epoch1000": subspace1000,
        },
        "epoch500_to_epoch1000_subspace_overlap": overlap,
        "independent_mass_orthogonality_max_abs": {
            "epoch500": float(summary500["peeled_mass_orthogonality_max_abs"]),
            "epoch1000": float(summary1000["peeled_mass_orthogonality_max_abs"]),
        },
    }
    (args.output_dir / "square_k64_e500_vs_e1000_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    modes = np.arange(1, len(analytic) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 8.8), constrained_layout=False)
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.90,
        bottom=0.115,
        hspace=0.32,
        wspace=0.13,
    )

    axes[0, 0].plot(modes, analytic, color=CHARCOAL, linewidth=1.6, label="Analytic benchmark")
    axes[0, 0].scatter(modes, learned500, color=BLUE, facecolors="none", s=24, label="Epoch 500")
    axes[0, 0].scatter(modes, learned1000, color=RED, s=15, label="Epoch 1000", zorder=3)
    axes[0, 0].set_xlabel("Mode index")
    axes[0, 0].set_ylabel("Dirichlet eigenvalue")
    axes[0, 0].legend(frameon=False, ncol=1)
    axes[0, 0].grid(alpha=0.18, linewidth=0.5)
    axes[0, 0].set_title("a  Spectrum", loc="left", weight="bold")

    axes[0, 1].semilogy(modes, error500, color=BLUE, linewidth=1.3, label="Epoch 500")
    axes[0, 1].semilogy(modes, error1000, color=RED, linewidth=1.3, label="Epoch 1000")
    axes[0, 1].axhline(0.01, color=CHARCOAL, linestyle="--", linewidth=0.8, label="1%")
    axes[0, 1].axhline(0.05, color="#777777", linestyle=":", linewidth=0.8, label="5%")
    axes[0, 1].set_xlabel("Mode index")
    axes[0, 1].set_ylabel("Relative eigenvalue error")
    axes[0, 1].legend(frameon=False, ncol=2)
    axes[0, 1].grid(alpha=0.18, linewidth=0.5)
    axes[0, 1].set_title("b  Modewise error", loc="left", weight="bold")

    limit = 1.05 * max(float(np.max(error500)), float(np.max(error1000)))
    axes[1, 0].scatter(error500, error1000, c=np.where(improved, RED, BLUE), s=24, alpha=0.9)
    axes[1, 0].plot([0, limit], [0, limit], color=CHARCOAL, linewidth=0.9, linestyle="--")
    axes[1, 0].set_xlim(0, limit)
    axes[1, 0].set_ylim(0, limit)
    axes[1, 0].set_xlabel("Epoch 500 relative error")
    axes[1, 0].set_ylabel("Epoch 1000 relative error")
    axes[1, 0].grid(alpha=0.18, linewidth=0.5)
    axes[1, 0].text(
        0.04,
        0.94,
        f"Improved: {np.count_nonzero(improved)}/64",
        transform=axes[1, 0].transAxes,
        va="top",
        color=RED,
        weight="bold",
    )
    axes[1, 0].set_title("c  Per-mode change", loc="left", weight="bold")

    axes[1, 1].axis("off")
    table_rows = [
        ("Median eigenvalue error", f"{100*spectrum500['median_relative_error']:.2f}%", f"{100*spectrum1000['median_relative_error']:.2f}%"),
        ("P90 eigenvalue error", f"{100*spectrum500['p90_relative_error']:.2f}%", f"{100*spectrum1000['p90_relative_error']:.2f}%"),
        ("Maximum eigenvalue error", f"{100*spectrum500['max_relative_error']:.2f}%", f"{100*spectrum1000['max_relative_error']:.2f}%"),
        ("Modes within 1%", str(spectrum500["within_1_percent"]), str(spectrum1000["within_1_percent"])),
        ("Modes within 5%", str(spectrum500["within_5_percent"]), str(spectrum1000["within_5_percent"])),
        ("Subspace minimum cosine", f"{subspace500['minimum_cosine']:.4f}", f"{subspace1000['minimum_cosine']:.4f}"),
        ("Maximum principal angle", f"{subspace500['maximum_principal_angle_deg']:.2f} deg", f"{subspace1000['maximum_principal_angle_deg']:.2f} deg"),
        ("Independent mass orth. error", f"{summary500['peeled_mass_orthogonality_max_abs']:.4f}", f"{summary1000['peeled_mass_orthogonality_max_abs']:.4f}"),
    ]
    table = axes[1, 1].table(
        cellText=table_rows,
        colLabels=["Metric", "Epoch 500", "Epoch 1000"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.52, 0.24, 0.24],
        bbox=[0.0, 0.05, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor(CHARCOAL)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor(LIGHT)
    axes[1, 1].set_title("d  Summary", loc="left", weight="bold")

    fig.suptitle(
        "Generic K=64 peeled-Ritz basis: 500 vs 1000 epochs",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "IID uniform sampling; dynamic Rayleigh causal sorting; no D4 augmentation; "
        "no analytic basis orientation. Analytic spectrum is used for benchmarking only.",
        ha="center",
        fontsize=9.5,
        color="#4D4D4D",
    )
    stem = args.output_dir / "square_k64_generic_e500_vs_e1000_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
