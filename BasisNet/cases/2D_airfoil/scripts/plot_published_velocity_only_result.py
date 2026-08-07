from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


CASE_ORDER = (
    ("alpha_top_p10_beta050", r"$\beta=0.50,\ \alpha=+10^\circ$"),
    ("alpha_middle_0_beta085", r"$\beta=0.85,\ \alpha=0^\circ$"),
    ("alpha_bottom_m10_beta050", r"$\beta=0.50,\ \alpha=-10^\circ$"),
)
BLUE = "#3B75AF"
RED = "#C44E52"


def _load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("status") != "complete":
        raise ValueError(f"result report is not complete: {path}")
    return report


def _plot_error_summary(report: dict, output_dir: Path) -> None:
    cases = {case["label"]: case for case in report["evaluation"]["cases"]}
    labels = [display for _, display in CASE_ORDER]
    total = [100.0 * cases[key]["velocity_error"] for key, _ in CASE_ORDER]
    wake = [100.0 * cases[key]["wake_velocity_error"] for key, _ in CASE_ORDER]
    fluctuation = [
        100.0 * cases[key]["fluctuation_velocity_error"] for key, _ in CASE_ORDER
    ]
    wake_fluctuation = [
        100.0 * cases[key]["wake_fluctuation_velocity_error"]
        for key, _ in CASE_ORDER
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), constrained_layout=True)
    x = list(range(len(labels)))
    width = 0.34

    panels = (
        (axes[0], total, wake, "Total", "Wake", "Velocity error (%)", 4.4),
        (
            axes[1],
            fluctuation,
            wake_fluctuation,
            "Fluctuation",
            "Wake fluctuation",
            "Fluctuation velocity error (%)",
            30.5,
        ),
    )
    for axis, first, second, first_label, second_label, ylabel, upper in panels:
        bars_a = axis.bar(
            [value - width / 2 for value in x],
            first,
            width,
            color=BLUE,
            label=first_label,
        )
        bars_b = axis.bar(
            [value + width / 2 for value in x],
            second,
            width,
            color=RED,
            label=second_label,
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, upper)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, loc="upper center", ncols=2)
        for bars in (bars_a, bars_b):
            axis.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)

    fig.suptitle(
        "K1024 velocity-only basis: independent Re=1500 reconstruction",
        fontsize=10.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "velocity_error_summary.png", dpi=300)
    fig.savefig(output_dir / "velocity_error_summary.pdf")
    plt.close(fig)


def _plot_velocity_reconstruction(
    report: dict, evaluation_root: Path, output_dir: Path
) -> None:
    cases = {case["label"]: case for case in report["evaluation"]["cases"]}
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 17)
    crops: list[Image.Image] = []
    titles: list[str] = []
    for case_name, display in CASE_ORDER:
        source = evaluation_root / case_name / "velocity_optimal_comparison.png"
        with Image.open(source) as image:
            # The evaluator writes velocity panels on the left and diagnostic
            # vorticity panels on the right. Publish only the velocity column.
            crop = image.convert("RGB").crop((235, 30, 800, 1144))
        # Replace the evaluator's historical "PINN" label with the current
        # BasisNet terminology while preserving the evaluated pixels.
        draw = ImageDraw.Draw(crop)
        draw.rectangle((0, 365, crop.width, 405), fill="white")
        label = (
            "BasisNet velocity-only reconstruction, velocity error="
            f"{100.0 * cases[case_name]['velocity_error']:.2f}%"
        )
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((crop.width - (bounds[2] - bounds[0])) / 2, 373),
            label,
            fill="black",
            font=font,
        )
        crops.append(crop)
        titles.append(display)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 5.35), constrained_layout=True)
    for axis, crop, title in zip(axes, crops, titles, strict=True):
        axis.imshow(crop)
        axis.set_title(title, fontsize=9.5)
        axis.axis("off")
    fig.suptitle(
        "Velocity-only K1024: GT, velocity-L2 reconstruction, and absolute error",
        fontsize=10.5,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "three_case_velocity_reconstruction.png", dpi=300)
    plt.close(fig)


def main() -> None:
    case_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=case_root / "reports" / "k1024_velocity_only_results.json",
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=case_root / "assets" / "k1024_velocity_only",
    )
    args = parser.parse_args()

    report = _load_report(args.report)
    _plot_error_summary(report, args.output_dir)
    _plot_velocity_reconstruction(report, args.evaluation_root, args.output_dir)


if __name__ == "__main__":
    main()
