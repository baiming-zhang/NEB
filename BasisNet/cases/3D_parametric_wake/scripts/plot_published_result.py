from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageChops


BLUE = "#3B75AF"
RED = "#C44E52"


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("status") != "complete":
        raise ValueError(f"result report is not complete: {path}")
    return report


def plot_history(report: dict, output_dir: Path) -> None:
    evaluation = report["evaluation"]
    errors = [100.0 * value for value in evaluation["frame_velocity_errors"]]
    times = [0.1 * index for index in range(len(errors))]
    mean = 100.0 * evaluation["output_mean_velocity_error"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(5.8, 3.25), constrained_layout=True)
    axis.plot(times, errors, color=BLUE, linewidth=1.7, marker="o", markersize=3)
    axis.axhline(mean, color=RED, linestyle="--", linewidth=1.2, label=f"Mean {mean:.2f}%")
    axis.set_xlabel("Time")
    axis.set_ylabel("Velocity relative error (%)")
    axis.set_title("K256 Re=500 wake reconstruction, epoch 5000")
    axis.grid(color="#D9D9D9", linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "velocity_error_history.png", dpi=300)
    fig.savefig(output_dir / "velocity_error_history.pdf")
    plt.close(fig)


def prepare_reconstruction(source: Path, output_dir: Path) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        background = Image.new("RGB", rgb.size, "white")
        bounds = ImageChops.difference(rgb, background).getbbox()
        if bounds is not None:
            left, top, right, bottom = bounds
            margin = 24
            rgb = rgb.crop(
                (
                    max(0, left - margin),
                    max(0, top - margin),
                    min(rgb.width, right + margin),
                    min(rgb.height, bottom + margin),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb.save(output_dir / "best_reconstruction.png", optimize=True)


def main() -> None:
    case_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=case_root / "reports" / "k256_re500_best_results.json",
    )
    parser.add_argument("--source-frame", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=case_root / "assets" / "k256_re500_best",
    )
    args = parser.parse_args()
    report = load_report(args.report)
    plot_history(report, args.output_dir)
    prepare_reconstruction(args.source_frame, args.output_dir)


if __name__ == "__main__":
    main()
