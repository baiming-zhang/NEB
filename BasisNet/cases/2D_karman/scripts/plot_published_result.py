from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


BLUE = "#3B75AF"
RED = "#C44E52"


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("status") != "complete":
        raise ValueError(f"result report is not complete: {path}")
    return report


def plot_summary(report: dict, output_dir: Path) -> None:
    cases = report["evaluation"]["cases"]
    labels = [rf"$\alpha={case['alpha']:.2f}$" for case in cases]
    velocity = [100.0 * case["velocity_error"] for case in cases]
    vorticity = [100.0 * case["vorticity_error"] for case in cases]
    x = range(len(cases))
    width = 0.34

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(5.6, 3.35), constrained_layout=True)
    velocity_bars = axis.bar(
        [value - width / 2 for value in x],
        velocity,
        width,
        color=BLUE,
        label="Velocity",
    )
    vorticity_bars = axis.bar(
        [value + width / 2 for value in x],
        vorticity,
        width,
        color=RED,
        label="Vorticity",
    )
    axis.set_xticks(list(x), labels)
    axis.set_ylabel("Independent GT relative error (%)")
    axis.set_ylim(0.0, 24.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=2)
    axis.bar_label(velocity_bars, fmt="%.2f", padding=2, fontsize=8)
    axis.bar_label(vorticity_bars, fmt="%.2f", padding=2, fontsize=8)
    axis.set_title("K96 translating-cylinder basis, epoch 4950")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "error_summary.png", dpi=300)
    fig.savefig(output_dir / "error_summary.pdf")
    plt.close(fig)


def crop_best_reconstruction(source: Path, output_dir: Path) -> None:
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        crop = rgb.crop((0, 0, int(rgb.width * 0.42), rgb.height))
        draw = ImageDraw.Draw(crop)
        draw.rectangle((0, 0, crop.width, 250), fill="white")
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 68)
        label = "K96 BasisNet, alpha=0.40, phase=0.50"
        box = draw.textbbox((0, 0), label, font=font)
        draw.text(
            ((crop.width - box[2]) / 2, 80),
            label,
            fill="black",
            font=font,
        )
        target_width = 1500
        target_height = round(crop.height * target_width / crop.width)
        crop = crop.resize((target_width, target_height), Image.Resampling.LANCZOS)
    output_dir.mkdir(parents=True, exist_ok=True)
    crop.save(output_dir / "best_reconstruction_alpha_0p40.png", optimize=True)


def main() -> None:
    case_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=case_root / "reports" / "k96_best_results.json",
    )
    parser.add_argument("--source-frame", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=case_root / "assets" / "k96_best",
    )
    args = parser.parse_args()
    report = load_report(args.report)
    plot_summary(report, args.output_dir)
    crop_best_reconstruction(args.source_frame, args.output_dir)


if __name__ == "__main__":
    main()
