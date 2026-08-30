"""Build the compiled K=64 real-Gabor WIRE rectangular-domain report."""

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


def ranked(items: dict) -> list[dict]:
    return sorted(
        items.values(),
        key=lambda item: (
            -item["resolved_rank"],
            item["median_positive_spectrum_relative_error"],
            item["p90_positive_spectrum_relative_error"],
        ),
    )


def short_label(item: dict) -> str:
    cfg = item["config"]
    schedule = cfg["scheduler"].replace("validation_", "")
    return (
        f"{item['experiment'].replace('_', ' ')} "
        f"(LR {cfg['learning_rate']:.0e}, {schedule}, w{cfg['width']}/d{cfg['depth']})"
    )


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as source:
        width, height = source.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def build_confirmation_figure(summary: dict, output: Path) -> None:
    items = summary["experiments"]
    order = [item["experiment"] for item in ranked(items)]
    labels = [item["experiment"].replace("_", " ") for item in ranked(items)]
    palette = [BLUE, RED, "#666666"]
    exact = np.asarray(items[order[0]]["analytic_eigenvalues"])
    modes = np.arange(1, len(exact) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), constrained_layout=True)
    axes[0].plot(modes, exact, color="black", lw=1.5, label="analytic Neumann")
    for name, label, color in zip(order, labels, palette):
        learned = np.asarray(items[name]["learned_eigenvalues"])
        axes[0].plot(np.arange(1, len(learned) + 1), learned, color=color, lw=1.25, label=label)
    axes[0].set(title="Confirmed K=64 Ritz spectrum", xlabel="mode index", ylabel="eigenvalue")
    axes[0].legend(frameon=False)

    metric_labels = ["median", "p90", "mode-64 reach"]
    x = np.arange(len(metric_labels))
    width = 0.23
    for index, (name, label, color) in enumerate(zip(order, labels, palette)):
        item = items[name]
        values = [
            100.0 * item["median_positive_spectrum_relative_error"],
            100.0 * item["p90_positive_spectrum_relative_error"],
            100.0 * abs(item["spectral_reach_ratio_mode64"] - 1.0),
        ]
        axes[1].bar(x + (index - 1) * width, values, width, color=color, label=label)
    axes[1].set(title="Full-resolution confirmation", ylabel="error (%)")
    axes[1].set_xticks(x, metric_labels)
    axes[1].legend(frameon=False)
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
        panel.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
        panel.tick_params(width=0.6, length=2.5)
    fig.suptitle("K=64 real-Gabor WIRE confirmation", fontsize=12, weight="bold")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "wire_k64_confirmation.png", dpi=320, bbox_inches="tight")
    fig.savefig(output / "wire_k64_confirmation.pdf", bbox_inches="tight")
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
    canvas.drawString(12 * mm, 5.5 * mm, "Eigenfluid GEN - K=64 rectangular-domain WIRE study")
    canvas.drawRightString(landscape(A4)[0] - 12 * mm, 5.5 * mm, f"Page {document.page}")
    canvas.restoreState()


def make_table(data, widths, font_size=7.2, highlights=None) -> Table:
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BLUE)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.0),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F7F9FA")))
    for row, color in highlights or []:
        commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor(color)))
        commands.append(("FONTNAME", (0, row), (-1, row), "Helvetica-Bold"))
    result.setStyle(TableStyle(commands))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-gabor-summary", type=Path, required=True)
    parser.add_argument("--gabor-summary", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path, required=True)
    parser.add_argument("--confirm-summary", type=Path, required=True)
    parser.add_argument("--gabor-figure-dir", type=Path, required=True)
    parser.add_argument("--ablation-figure-dir", type=Path, required=True)
    parser.add_argument("--confirm-figure-dir", type=Path, required=True)
    parser.add_argument("--basis-figure-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    broad_gabor = load(args.broad_gabor_summary)
    gabor = load(args.gabor_summary)
    ablation = load(args.ablation_summary)
    confirmation = load(args.confirm_summary)
    gabor_items = gabor["experiments"]
    ablation_items = ablation["experiments"]
    confirmation_items = confirmation["experiments"]
    best_gabor = gabor_items[gabor["best_experiment"]]
    best_ablation = ablation_items[ablation["best_experiment"]]
    best_confirm = confirmation_items[confirmation["best_experiment"]]
    confirm_ranked = ranked(confirmation_items)
    max_rank = max(item["resolved_rank"] for item in confirmation_items.values())
    practical_candidates = [
        item for item in confirmation_items.values()
        if item["resolved_rank"] == max_rank
        and item["median_positive_spectrum_relative_error"]
        <= best_confirm["median_positive_spectrum_relative_error"] + 0.02
    ]
    practical_confirm = min(practical_candidates, key=lambda item: item["training_seconds"])
    broad_default = broad_gabor["experiments"]["control_w10_s10"]
    gabor_cfg = best_gabor["config"]
    final_cfg = best_confirm["config"]
    practical_cfg = practical_confirm["config"]
    confirm_epochs = final_cfg["epochs"]
    speedup = best_confirm["training_seconds"] / practical_confirm["training_seconds"]
    parameter_reduction = 1.0 - (
        practical_cfg["parameter_count"] / final_cfg["parameter_count"]
    )
    build_confirmation_figure(confirmation, args.confirm_figure_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = NumberedDocTemplate(
        str(args.output),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=11 * mm,
        bottomMargin=13 * mm,
        title="K=64 real-Gabor WIRE rectangular-domain study",
        author="OpenAI Codex",
        subject="WIRE parameter ablation and fixed-budget confirmation",
    )
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="body")
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

    cards = [
        [
            Paragraph(
                f"<b>Selected Gabor</b><br/>omega0={gabor_cfg['omega0']:g}, "
                f"sigma0={gabor_cfg['sigma0']:g}", callout,
            ),
            Paragraph(
                f"<b>Practical setup</b><br/>LR {practical_cfg['learning_rate']:.0e}, "
                f"{practical_cfg['scheduler'].replace('validation_', '')}, "
                f"w{practical_cfg['width']}/d{practical_cfg['depth']}", callout,
            ),
            Paragraph(
                f"<b>Accuracy optimum</b><br/>w{final_cfg['width']}/d{final_cfg['depth']}, "
                f"{best_confirm['resolved_rank']}/64 rank",
                callout,
            ),
        ],
        [
            Paragraph("official real-layer default 10/10 is mismatched", small),
            Paragraph(
                f"{practical_cfg['parameter_count'] / 1e6:.2f}M parameters; "
                f"{speedup:.2f}x faster than accuracy optimum", small,
            ),
            Paragraph(
                f"{pct(best_confirm['median_positive_spectrum_relative_error'])} median and "
                f"{pct(best_confirm['p90_positive_spectrum_relative_error'])} p90 error", small,
            ),
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

    story = [
        Paragraph("K=64 real-Gabor WIRE rectangular-domain study", title),
        Paragraph(
            "Unit-square sanity check | native CUDA Jet3 | staged parameter search | fixed-duration training | no early stopping",
            subtitle,
        ),
        cards_table,
        Spacer(1, 4 * mm),
        Paragraph("Result", heading),
        Paragraph(
            "The official real-Gabor WIRE architecture becomes useful for this basis problem once its domain scales are retuned. "
            f"The three-stage 50-epoch Gabor search selects omega0={gabor_cfg['omega0']:g} and "
            f"sigma0={gabor_cfg['sigma0']:g}. The 100-epoch one-factor screen selects "
            f"{short_label(best_ablation)} under the declared rank-then-error rule. A fresh "
            f"{confirm_epochs}-epoch, 4096-point confirmation selects {short_label(best_confirm)}, "
            f"resolves {best_confirm['resolved_rank']} generalized modes, and reaches "
            f"{pct(best_confirm['median_positive_spectrum_relative_error'])} median and "
            f"{pct(best_confirm['p90_positive_spectrum_relative_error'])} p90 positive-spectrum error. "
            f"Its mode-64 reach is {best_confirm['spectral_reach_ratio_mode64']:.4f}, with "
            f"{best_confirm['modes_within_20pct']} of 63 positive modes within 20% of the analytic spectrum.",
            body,
        ),
        Paragraph(
            f"For routine K=64 use, the practical Pareto choice is {short_label(practical_confirm)}. "
            f"It retains rank {practical_confirm['resolved_rank']} with "
            f"{pct(practical_confirm['median_positive_spectrum_relative_error'])} median and "
            f"{pct(practical_confirm['p90_positive_spectrum_relative_error'])} p90 error, while running "
            f"{speedup:.2f}x faster and using {100.0 * parameter_reduction:.0f}% fewer parameters than "
            "the accuracy-optimal width-96 arm.",
            body,
        ),
        Paragraph(
            f"The sampled final training batch rank is {best_confirm['final_batch_rank']}, while the independent fixed-point generalized mass/stiffness evaluation resolves rank {best_confirm['resolved_rank']}. The latter is the reported basis rank. Every run completed its configured epoch budget and every completion marker reports stopped_early=false.",
            body,
        ),
        fit_image(args.confirm_figure_dir / "wire_k64_confirmation.png", document.width, 62 * mm),
        PageBreak(),
        Paragraph("Reference implementation and Jet3 mapping", title),
        Paragraph(
            "The implementation follows vishwa91/wire modules/wire.py RealGaborLayer blob 7e05f699488dff25fc335ee2985052fdacefc8bb. Each hidden unit uses two independent affine projections u and v and computes cos(omega0*u) exp(-(sigma0*v)^2). The two projection matrices are stored as one stacked 2*hidden affine output, then split before activation. Initialization uses the same uniform bounds as PyTorch nn.Linear, driven by the case's deterministic NumPy PCG64 stream.",
            body,
        ),
        Paragraph(
            "The WIRE paper's headline results use the complex Gabor network. This real-valued GEN backend instead uses the official repository's RealGaborLayer, not an approximation with a shared projection. Compact Jet3 propagation is analytic through third spatial order, and the handwritten reverse uses exact fourth activation derivatives. No PyTorch, automatic differentiation, or finite differences are used in training.",
            body,
        ),
        fit_image(args.gabor_figure_dir / "wire_k64_gabor_grid.png", document.width, 91 * mm),
        Spacer(1, 2 * mm),
    ]

    gabor_rows = [["omega0", "sigma0", "rank", "median", "p90", "within20", "reach", "validation"]]
    local_ranked = ranked(gabor_items)
    gabor_sorted = [local_ranked[0], local_ranked[len(local_ranked) // 2], local_ranked[-1]]
    best_gabor_row = None
    for index, item in enumerate(gabor_sorted, start=1):
        if item["experiment"] == gabor["best_experiment"]:
            best_gabor_row = index
        cfg = item["config"]
        gabor_rows.append([
            f"{cfg['omega0']:g}", f"{cfg['sigma0']:g}", str(item["resolved_rank"]),
            pct(item["median_positive_spectrum_relative_error"]),
            pct(item["p90_positive_spectrum_relative_error"]),
            f"{item['modes_within_20pct']}/63", f"{item['spectral_reach_ratio_mode64']:.3f}",
            f"{item['final_validation_loss']:.1f}",
        ])
    story.extend([
        make_table(
            gabor_rows,
            [24 * mm, 24 * mm, 19 * mm, 28 * mm, 28 * mm, 28 * mm, 25 * mm, 34 * mm],
            highlights=[(best_gabor_row, PALE_BLUE)] if best_gabor_row else None,
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            f"Scale is the dominant factor. The selected {gabor_cfg['omega0']:g}/{gabor_cfg['sigma0']:g} arm has "
            f"{pct(best_gabor['median_positive_spectrum_relative_error'])} median error at epoch 50. "
            f"The official real-layer 10/10 defaults produce {pct(broad_default['median_positive_spectrum_relative_error'])} "
            "median error here. The broad, low-scale, and local grids show that the default Gaussian windows are too localized for this normalized smooth global eigenbasis and that the selected region is a broad local optimum rather than an isolated point.",
            body,
        ),
        PageBreak(),
        Paragraph("Optimizer, scheduler, width, and depth screen", title),
        Paragraph(
            f"All arms below hold omega0={gabor_cfg['omega0']:g} and sigma0={gabor_cfg['sigma0']:g} fixed and train for exactly 100 epochs with 2048 training points and 4096 fixed validation points. The common control is LR 1e-3, constant schedule, width 64, depth 4. Same-topology arms use byte-identical initial checkpoints.",
            body,
        ),
        fit_image(args.ablation_figure_dir / "wire_k64_ablation.png", document.width, 83 * mm),
        Spacer(1, 1.5 * mm),
    ])

    ablation_ranked = ranked(ablation_items)
    ablation_rows = [["experiment", "rank", "batch", "median", "p90", "within20", "LR", "schedule", "w/d", "time"]]
    best_ablation_row = None
    for index, item in enumerate(ablation_ranked, start=1):
        if item["experiment"] == ablation["best_experiment"]:
            best_ablation_row = index
        cfg = item["config"]
        ablation_rows.append([
            item["experiment"], str(item["resolved_rank"]), str(item["final_batch_rank"]),
            pct(item["median_positive_spectrum_relative_error"]),
            pct(item["p90_positive_spectrum_relative_error"]),
            f"{item['modes_within_20pct']}/63", f"{cfg['learning_rate']:.0e}",
            cfg["scheduler"].replace("validation_", ""), f"{cfg['width']}/{cfg['depth']}",
            f"{item['training_seconds']:.0f}s",
        ])
    story.extend([
        make_table(
            ablation_rows,
            [37 * mm, 16 * mm, 16 * mm, 24 * mm, 24 * mm, 25 * mm, 20 * mm, 31 * mm, 20 * mm, 22 * mm],
            font_size=6.8,
            highlights=[(best_ablation_row, PALE_BLUE)] if best_ablation_row else None,
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            f"The screen winner is {short_label(best_ablation)} at "
            f"{pct(best_ablation['median_positive_spectrum_relative_error'])} median and "
            f"{pct(best_ablation['p90_positive_spectrum_relative_error'])} p90 error. "
            f"The constant LR 1e-3 control reaches {pct(ablation_items['control_w10_s10']['median_positive_spectrum_relative_error'])} median error. "
            f"Cosine reaches {pct(ablation_items['scheduler_cosine']['median_positive_spectrum_relative_error'])}; "
            "validation-plateau exactly matches the constant control because validation continues improving and no reduction fires. Width and depth changes improve p90 modestly but carry substantial parameter and runtime costs.",
            body,
        ),
        PageBreak(),
        Paragraph("Full-resolution confirmation", title),
        Paragraph(
            f"The three leading one-factor choices were restarted from initialization and trained for {confirm_epochs} epochs with 4096 points per step. Matched topologies use identical initial checkpoints; different topologies use the same seed and initializer. This confirmation was predeclared as fixed duration; no checkpoint selection or early stopping is used.",
            body,
        ),
        fit_image(args.confirm_figure_dir / "wire_k64_confirmation.png", document.width, 106 * mm),
        Spacer(1, 2 * mm),
    ])

    confirm_rows = [[
        "experiment", "schedule", "LR", "w/d", "rank", "batch", "median", "p90",
        "within20", "mode64 reach", "overlap", "validation", "time",
    ]]
    confirm_order = [item["experiment"] for item in confirm_ranked]
    best_confirm_row = None
    for index, name in enumerate(confirm_order, start=1):
        item = confirmation_items[name]
        if name == confirmation["best_experiment"]:
            best_confirm_row = index
        cfg = item["config"]
        confirm_rows.append([
            name, cfg["scheduler"].replace("validation_", ""), f"{cfg['learning_rate']:.0e}",
            f"{cfg['width']}/{cfg['depth']}",
            str(item["resolved_rank"]), str(item["final_batch_rank"]),
            pct(item["median_positive_spectrum_relative_error"]),
            pct(item["p90_positive_spectrum_relative_error"]),
            f"{item['modes_within_20pct']}/63", f"{item['spectral_reach_ratio_mode64']:.4f}",
            f"{item['mean_absolute_mode_overlap']:.3f}", f"{item['final_validation_loss']:.2f}",
            f"{item['training_seconds']:.0f}s",
        ])
    story.extend([
        make_table(
            confirm_rows,
            [34 * mm, 22 * mm, 16 * mm, 16 * mm, 14 * mm, 14 * mm, 22 * mm,
             22 * mm, 23 * mm, 25 * mm, 22 * mm, 23 * mm, 18 * mm],
            font_size=6.4,
            highlights=[(best_confirm_row, PALE_BLUE)] if best_confirm_row else None,
        ),
        Spacer(1, 3 * mm),
        Paragraph(
            f"The confirmed winner is {short_label(best_confirm)} with "
            f"{pct(best_confirm['median_positive_spectrum_relative_error'])} median and "
            f"{pct(best_confirm['p90_positive_spectrum_relative_error'])} p90 error. "
            f"The next-ranked alternative is {short_label(confirm_ranked[1])}, at "
            f"{pct(confirm_ranked[1]['median_positive_spectrum_relative_error'])} median error. "
            "The independent fixed 4096-point mass/stiffness pencil supplies the generalized rank reported here; sampled-batch rank is retained separately as a conditioning diagnostic.",
            body,
        ),
        PageBreak(),
        Paragraph("Representative learned basis", title),
        Paragraph(
            "The panels below show 12 representative generalized Ritz vorticity basis functions from the recommended practical width-64 model, spanning mode 1 through mode 64. The raw network outputs are rotated by the fixed-validation generalized eigensystem, then each displayed field is divided by its own maximum absolute value so spatial structure remains legible. Eigenfunction signs are arbitrary; the plotting script orients each sign deterministically at its largest-magnitude grid point.",
            body,
        ),
        fit_image(
            args.basis_figure_dir / "wire_k64_practical_basis.png",
            document.width,
            139 * mm,
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            "Low modes vary smoothly across the full square, while the selected mid- and high-index modes develop progressively finer nodal structure. The high-index panels remain spatially resolved rather than collapsing into localized noise, consistent with the independent rank-64 mass/stiffness evaluation.",
            body,
        ),
        PageBreak(),
        Paragraph("Protocol, recommendation, and scope", title),
    ])

    protocol_rows = [
        ["item", "setting"],
        ["domain", "unit square [0,1]^2; analytic Neumann vorticity spectrum"],
        ["basis", "exactly K=64 raw-coordinate real-Gabor WIRE modes"],
        ["WIRE layer", "two independent affine projections; cos(omega0*u) exp(-(sigma0*v)^2)"],
        ["objective", "unscaled pure-vorticity Rayleigh quotient"],
        ["differentiation", "native compact analytic Jet3; handwritten parameter reverse"],
        ["precision", "FP32 network, Jet3, quadrature, Ritz, master weights, and Adam"],
        ["Gabor screen", "27 arms across broad, low-scale, and local grids; 50 epochs each"],
        ["secondary screen", "9 one-factor arms, 100 epochs, selected omega0/sigma0"],
        ["confirmation", f"3 leading one-factor arms, {confirm_epochs} epochs, 4096 training / validation points"],
        ["early stopping", "explicitly disabled; every completion marker reports false"],
        ["failure handling", "resumable checkpoints and automatic retries; no arm required a retry"],
    ]
    story.extend([
        make_table(protocol_rows, [43 * mm, 210 * mm], font_size=8.0),
        Spacer(1, 4 * mm),
        Paragraph("Recommended rectangular K=64 setup", heading),
        Paragraph(
            f"Use official-real-Gabor WIRE with omega0={gabor_cfg['omega0']:g}, sigma0={gabor_cfg['sigma0']:g}, "
            f"width {practical_cfg['width']}, depth {practical_cfg['depth']}, base LR {practical_cfg['learning_rate']:.0e}, "
            f"and the {practical_cfg['scheduler'].replace('validation_', '')} schedule over the full configured epoch budget. "
            f"This practical choice gives up only {100.0 * (practical_confirm['median_positive_spectrum_relative_error'] - best_confirm['median_positive_spectrum_relative_error']):.2f} percentage points of median error while cutting runtime by {100.0 * (1.0 - 1.0 / speedup):.0f}%. "
            "Keep early stopping disabled. Use the width-96 accuracy optimum when the lowest confirmed spectrum error matters more than compute cost.",
            body,
        ),
        Paragraph("Scope and limitations", heading),
        Paragraph(
            "This is an efficient staged search, not a full factorial optimization. Gabor parameters were screened first, then optimizer and topology factors were changed one at a time around the selected scale. Results are specific to normalized raw coordinates on the unit square and the pure-vorticity Rayleigh objective. The official complex WIRE network is not implemented here; the chosen layer is the official repository's real-Gabor variant because the native Jet3 backend is real-valued.",
            body,
        ),
        Paragraph("References and generated artifacts", heading),
        Paragraph(
            "Reference code: https://github.com/vishwa91/wire (modules/wire.py, RealGaborLayer). Paper: https://arxiv.org/abs/2301.05187. Machine-readable JSON/CSV summaries, PNG figures, vector PDF figures, training histories, checkpoints, and completion markers are retained under results/wire_ablation_k64. The implementation validator reports maximum forward derivative error 1.42e-14 and reverse-transpose relative error 2.03e-10.",
            small,
        ),
    ])
    document.build(story)
    print(args.output)


if __name__ == "__main__":
    main()
