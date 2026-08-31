"""Build the compiled K=64 raw-SIREN frequency-cap investigation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


RED = "#B2182B"
BLUE = "#2166AC"
LIGHT_BLUE = "#67A9CF"
INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
PALE_BLUE = "#EAF2F8"
PALE_RED = "#F9ECEA"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def build_confirmation_figure(screen: dict, confirm: dict, output: Path) -> None:
    exact = np.asarray(confirm["analytic_eigenvalues"])
    learned_200 = np.asarray(screen["learned_eigenvalues"])
    learned_600 = np.asarray(confirm["learned_eigenvalues"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), constrained_layout=True)
    modes = np.arange(1, len(exact) + 1)
    axes[0].plot(modes, exact, color="black", lw=1.5, label="analytic Neumann")
    axes[0].plot(modes, learned_200, color=LIGHT_BLUE, lw=1.25, label="epoch 200")
    axes[0].plot(modes, learned_600, color=BLUE, lw=1.4, label="epoch 600")
    axes[0].set(title="Selected omega0=8 spectrum", xlabel="mode index", ylabel="eigenvalue")
    axes[0].legend(frameon=False)
    labels = ["median error", "p90 error", "mode-64 reach error"]
    values_200 = [
        100.0 * screen["median_positive_spectrum_relative_error"],
        100.0 * screen["p90_positive_spectrum_relative_error"],
        100.0 * abs(screen["spectral_reach_ratio_mode64"] - 1.0),
    ]
    values_600 = [
        100.0 * confirm["median_positive_spectrum_relative_error"],
        100.0 * confirm["p90_positive_spectrum_relative_error"],
        100.0 * abs(confirm["spectral_reach_ratio_mode64"] - 1.0),
    ]
    x = np.arange(len(labels))
    width = 0.36
    axes[1].bar(x - width / 2, values_200, width, color=LIGHT_BLUE, label="epoch 200")
    axes[1].bar(x + width / 2, values_600, width, color=BLUE, label="epoch 600")
    axes[1].set(title="Longer fixed-budget confirmation", ylabel="error (%)")
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].legend(frameon=False)
    for panel in axes:
        panel.spines["top"].set_visible(False)
        panel.spines["right"].set_visible(False)
        panel.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
        panel.tick_params(width=0.6, length=2.5)
    fig.suptitle("K=64 raw-SIREN confirmation", fontsize=12, weight="bold")
    fig.savefig(output / "siren_k64_confirmation.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "siren_k64_confirmation.pdf", bbox_inches="tight")
    plt.close(fig)


class NumberedDocTemplate(BaseDocTemplate):
    pass


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor(GRID))
    canvas.setLineWidth(0.4)
    canvas.line(12 * mm, 9 * mm, landscape(A4)[0] - 12 * mm, 9 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(12 * mm, 5.5 * mm, "Eigenfluid GEN - K=64 rectangular-domain SIREN investigation")
    canvas.drawRightString(landscape(A4)[0] - 12 * mm, 5.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def table(data, widths, header=True, font_size=7.2, highlights=None) -> Table:
    result = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold" if header else "Helvetica"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE) if header else colors.white),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.HexColor(INK)),
        ("FONTNAME", (0, 1 if header else 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.0),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row in range(1 if header else 0, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F7F9FA")))
    for row, color in highlights or []:
        commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor(color)))
        commands.append(("FONTNAME", (0, row), (-1, row), "Helvetica-Bold"))
    result.setStyle(TableStyle(commands))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--confirm-summary", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen_summary = load(args.screen_summary)
    confirm_summary = load(args.confirm_summary)
    items = screen_summary["experiments"]
    best_name = screen_summary["best_experiment"]
    best_200 = items[best_name]
    best_600 = confirm_summary["experiments"][best_name]
    build_confirmation_figure(best_200, best_600, args.figure_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    document = NumberedDocTemplate(
        str(args.output), pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=11 * mm, bottomMargin=13 * mm,
        title="K=64 raw-SIREN frequency-cap investigation",
        author="OpenAI Codex",
        subject="Rectangular-domain GEN sanity check",
    )
    frame = Frame(
        document.leftMargin, document.bottomMargin,
        document.width, document.height, id="body"
    )
    document.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=25, textColor=colors.HexColor(INK),
        alignment=TA_LEFT, spaceAfter=4 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=10, leading=14,
        textColor=colors.HexColor(MUTED), spaceAfter=4 * mm,
    )
    heading = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=17, textColor=colors.HexColor(BLUE),
        spaceBefore=2 * mm, spaceAfter=2.2 * mm,
    )
    body = ParagraphStyle(
        "Body", parent=styles["BodyText"], fontSize=9, leading=13,
        textColor=colors.HexColor(INK), spaceAfter=2.2 * mm,
    )
    small = ParagraphStyle(
        "Small", parent=body, fontSize=7.8, leading=10.5,
        textColor=colors.HexColor(MUTED),
    )
    callout = ParagraphStyle(
        "Callout", parent=body, fontName="Helvetica-Bold", fontSize=11,
        leading=15, textColor=colors.HexColor(INK), alignment=TA_CENTER,
    )

    story = [
        Paragraph("K=64 raw-SIREN frequency-cap investigation", title),
        Paragraph(
            "Unit-square rectangular sanity check | native CUDA Jet3 | fixed-duration training | no early stopping",
            subtitle,
        ),
    ]
    cards = [
        [
            Paragraph("<b>Diagnosis</b><br/>omega0=1 is rank/bandwidth capped", callout),
            Paragraph("<b>Selected</b><br/>omega0=8, LR 4e-4, constant", callout),
            Paragraph("<b>Confirmed</b><br/>64/64 resolved at epoch 600", callout),
        ],
        [
            Paragraph("60/64 generalized rank at omega0=1", small),
            Paragraph("width 64, depth 4; 0.92M parameters", small),
            Paragraph("3.21% median spectrum error", small),
        ],
    ]
    cards_table = Table(cards, colWidths=[document.width / 3.0] * 3)
    cards_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(PALE_RED)),
        ("BACKGROUND", (1, 0), (-1, -1), colors.HexColor(PALE_BLUE)),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(GRID)),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
        ("TOPPADDING", (0, 1), (-1, 1), 4),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 5),
    ]))
    story.extend([
        cards_table,
        Spacer(1, 4 * mm),
        Paragraph("Result", heading),
        Paragraph(
            "The suspected SIREN frequency cap is real, but the remedy is a moderate first-layer spatial scale, not an arbitrarily large one. At the matched 200-epoch screen, omega0=1 resolves only 60 of 64 generalized modes. omega0=8 resolves all 64 and reduces median positive-spectrum error to 5.59%, versus 12.68% at omega0=16 and 65.78% at omega0=32.",
            body,
        ),
        Paragraph(
            "The fixed 600-epoch omega0=8 confirmation further improves median error to 3.21%, p90 error to 12.87%, and mode-64 reach to 1.037. Sixty of 63 positive modes lie within 20% of the analytic spectrum. The sampled training rank is 62 at epoch 600, while the independent generalized mass/stiffness evaluation resolves all 64 modes; the latter is the reported rank.",
            body,
        ),
        fit_image(args.figure_dir / "siren_k64_confirmation.png", document.width, 91 * mm),
        PageBreak(),
        Paragraph("Frequency-cap evidence", title),
        Paragraph(
            "Only raw xy inputs and SIREN activation are exercised. ReLU and positional-encoding arms were removed from this test campaign, while their runtime implementations remain in the codebase. Scaling raw xy by omega0 changes the first-layer spatial bandwidth without adding Fourier features; its Jet3 Hessian and third derivatives remain exactly zero.",
            body,
        ),
        fit_image(args.figure_dir / "siren_k64_spectrum.png", document.width, 105 * mm),
        Spacer(1, 2 * mm),
    ])
    frequency_rows = [["omega0", "rank", "median error", "p90 error", "within 20%", "mode-64 reach", "val. objective"]]
    for name in ("freq_omega01", "freq_omega08", "freq_omega16", "control_omega32"):
        item = items[name]
        frequency_rows.append([
            f"{item['config']['omega0']:g}", str(item["resolved_rank"]),
            pct(item["median_positive_spectrum_relative_error"]),
            pct(item["p90_positive_spectrum_relative_error"]),
            f"{item['modes_within_20pct']}/63",
            "n/a" if not np.isfinite(item["spectral_reach_ratio_mode64"]) else f"{item['spectral_reach_ratio_mode64']:.3f}",
            f"{item['final_validation_loss']:.2f}",
        ])
    story.extend([
        table(frequency_rows, [26 * mm, 20 * mm, 31 * mm, 28 * mm, 29 * mm, 31 * mm, 34 * mm], highlights=[(2, PALE_BLUE)]),
        Spacer(1, 2 * mm),
        Paragraph(
            "Interpretation: omega0=1 lacks sufficient independent high-frequency content; omega0=32 overpopulates high frequencies and shifts even the lowest Ritz modes upward. omega0=8 provides the best balance across the full 64-mode spectrum.",
            body,
        ),
        PageBreak(),
        Paragraph("Learning rate, scheduler, and MLP size", title),
        Paragraph(
            "These ablations hold omega0=32 fixed to isolate optimizer and topology effects. They should not be read as overturning the frequency conclusion: every omega0=32 arm is substantially worse than the selected omega0=8 configuration.",
            body,
        ),
        fit_image(args.figure_dir / "siren_k64_ablation.png", document.width, 122 * mm),
        Spacer(1, 1.5 * mm),
        Paragraph(
            "Learning rate: 8e-4 is best among the omega0=32 LR arms (34.09% median error), ahead of 4e-4 (65.78%) and 1e-4 (215.96%). Scheduler: validation-plateau exactly matches constant because the validation objective improves at both checks and no reduction is triggered; cosine is worse (116.98%) because it decays too early. Size: depth 2 is the best omega0=32 topology (54.66%) and is fastest, while extra width or depth does not repair a mismatched frequency scale.",
            body,
        ),
        PageBreak(),
        Paragraph("Complete 200-epoch sweep", title),
    ])
    ranked = sorted(
        items.values(),
        key=lambda item: (-item["resolved_rank"], item["median_positive_spectrum_relative_error"]),
    )
    full_rows = [[
        "experiment", "group", "omega0", "LR", "schedule", "w/d", "params",
        "rank", "median", "p90", "within20", "reach", "time (s)",
    ]]
    best_row = None
    for index, item in enumerate(ranked, start=1):
        cfg = item["config"]
        if item["experiment"] == best_name:
            best_row = index
        reach = item["spectral_reach_ratio_mode64"]
        full_rows.append([
            item["experiment"], item["group"], f"{cfg['omega0']:g}",
            f"{cfg['learning_rate']:.0e}", cfg["scheduler"].replace("validation_", ""),
            f"{cfg['width']}/{cfg['depth']}", f"{cfg['parameter_count']/1e6:.2f}M",
            str(item["resolved_rank"]), pct(item["median_positive_spectrum_relative_error"]),
            pct(item["p90_positive_spectrum_relative_error"]), str(item["modes_within_20pct"]),
            "n/a" if not np.isfinite(reach) else f"{reach:.3f}", f"{item['training_seconds']:.1f}",
        ])
    story.extend([
        table(
            full_rows,
            [34 * mm, 23 * mm, 15 * mm, 17 * mm, 23 * mm, 16 * mm, 18 * mm,
             14 * mm, 19 * mm, 18 * mm, 19 * mm, 17 * mm, 20 * mm],
            font_size=6.5,
            highlights=[(best_row, PALE_BLUE)] if best_row else None,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            "Selection was predeclared: maximize generalized resolved rank, then minimize median positive-spectrum relative error. This avoids selecting the low-frequency omega0=1 arm merely because its average Rayleigh objective is small after rank collapse.",
            body,
        ),
        PageBreak(),
        Paragraph("Protocol and reproducibility", title),
    ])
    protocol_rows = [
        ["item", "setting"],
        ["domain", "unit square [0,1]^2; analytic Neumann vorticity reference"],
        ["basis", "exactly K=64 raw-coordinate SIREN modes"],
        ["objective", "unscaled pure-vorticity Rayleigh quotient"],
        ["differentiation", "native compact analytic Jet3; handwritten parameter reverse"],
        ["precision", "FP32 network, Jet3, quadrature, Ritz, master weights, and Adam"],
        ["sampling", "4096 scrambled Sobol points per epoch; 4096 fixed validation points"],
        ["reorthogonalization", "two fixed passes; fixed mode order"],
        ["screen", "12 one-factor-at-a-time arms, 200 epochs each"],
        ["confirmation", "selected omega0=8 arm rerun from initialization for 600 epochs"],
        ["early stopping", "explicitly disabled; every completion marker reports false"],
        ["failure handling", "resumable checkpoints and retries; no screen arm required a retry"],
    ]
    story.extend([
        table(protocol_rows, [43 * mm, 210 * mm], font_size=8.0),
        Spacer(1, 4 * mm),
        Paragraph("Recommendation", heading),
        Paragraph(
            "For this rectangular sanity check, use omega0=8, learning rate 4e-4, constant LR, width 64, depth 4, and K=64. Do not add early stopping. Treat omega0 as domain-scale dependent: normalized raw coordinates need enough first-layer bandwidth to span the target spectrum, but excessive omega0 biases the entire Ritz spectrum upward.",
            body,
        ),
        Paragraph("Scope and limitations", heading),
        Paragraph(
            "The optimizer and topology sweeps were intentionally evaluated at omega0=32 to preserve a one-factor-at-a-time design around the original control. A second factorial sweep around omega0=8 could refine LR and topology further, but it is not needed to establish the frequency-cap diagnosis. Results are specific to the unit-square pure-vorticity Rayleigh sanity check and should not be transferred unchanged to other domain scales.",
            body,
        ),
        Paragraph("Generated artifacts", heading),
        Paragraph(
            "Machine-readable summaries: epochs_200/evaluation/siren_k64_sweep_summary.json and .csv; confirmation: epochs_600/evaluation/siren_k64_sweep_summary.json. Figures are supplied in PNG and vector PDF formats. Training inputs, checkpoints, histories, and completion markers are retained under the campaign result directories.",
            small,
        ),
    ])
    document.build(story)
    print(args.output)


if __name__ == "__main__":
    main()
