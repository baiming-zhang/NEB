from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


BLUE = "#3B75AF"
RED = "#C44E52"


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("status") != "complete":
        raise ValueError(f"result report is not complete: {path}")
    return report


def plot_summary(report: dict, output_dir: Path) -> None:
    benchmarks = report["evaluation"]["benchmarks"]
    labels = ["Taylor--Green", "Oscillatory ABC"]
    mean = [100.0 * item["mean_velocity_error"] for item in benchmarks]
    maximum = [100.0 * item["max_velocity_error"] for item in benchmarks]
    x = range(len(labels))
    width = 0.34

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.35), constrained_layout=True)
    mean_bars = axis.bar(
        [value - width / 2 for value in x], mean, width, color=BLUE, label="Mean"
    )
    max_bars = axis.bar(
        [value + width / 2 for value in x],
        maximum,
        width,
        color=RED,
        label="Maximum",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Velocity relative error (%)")
    axis.set_ylim(0.0, 1.3)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2)
    axis.bar_label(mean_bars, fmt="%.3f", padding=2, fontsize=8)
    axis.bar_label(max_bars, fmt="%.3f", padding=2, fontsize=8)
    axis.set_title("K256 periodic-free basis, epoch 10000")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "velocity_error_summary.png", dpi=300)
    fig.savefig(output_dir / "velocity_error_summary.pdf")
    plt.close(fig)


def combine_reconstructions(taylor_green: Path, abc: Path, output_dir: Path) -> None:
    with Image.open(taylor_green) as first_image, Image.open(abc) as second_image:
        first = first_image.convert("RGB")
        second = second_image.convert("RGB")
        width = max(first.width, second.width)
        gap = 20
        canvas = Image.new("RGB", (width, first.height + second.height + gap), "white")
        canvas.paste(first, ((width - first.width) // 2, 0))
        canvas.paste(second, ((width - second.width) // 2, first.height + gap))
    output_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(output_dir / "best_checkpoint_reconstructions.png", optimize=True)


def main() -> None:
    case_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=case_root / "reports" / "k256_best_results.json",
    )
    parser.add_argument("--taylor-green-frame", type=Path, required=True)
    parser.add_argument("--abc-frame", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=case_root / "assets" / "k256_best",
    )
    args = parser.parse_args()
    report = load_report(args.report)
    plot_summary(report, args.output_dir)
    combine_reconstructions(
        args.taylor_green_frame, args.abc_frame, args.output_dir
    )


if __name__ == "__main__":
    main()
