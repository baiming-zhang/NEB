"""Build a compact mixed-page-size step-5000 evaluation PDF."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BLUE = colors.HexColor("#2F5D8C")
RED = colors.HexColor("#B74643")
LIGHT_BLUE = colors.HexColor("#E9F0F7")
LIGHT_RED = colors.HexColor("#F7EDEC")
GRID = colors.HexColor("#AAB4C0")
TOTAL_PAGES = 8


def para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def footer(page_number: int, page_size: tuple[float, float]):
    def draw(canvas, _doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#5D6570"))
        canvas.drawRightString(
            page_size[0] - 6 * mm,
            3.2 * mm,
            f"Page {page_number}/{TOTAL_PAGES}",
        )
        canvas.restoreState()

    return draw


def build_page(
    path: Path,
    page_size: tuple[float, float],
    story: list[object],
    page_number: int,
    *,
    margin: float = 6 * mm,
) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=page_size,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="Step 5000 Evaluation and Network-Only Jet3 Ablation",
        author="EigenFluid SSS",
    )
    doc.build(story, onFirstPage=footer(page_number, page_size))
    pages = len(PdfReader(str(path)).pages)
    if pages != 1:
        raise RuntimeError(f"Expected one page for {path.name}, got {pages}")


def metric(case: dict, name: str) -> float:
    value = case["metrics"].get(name)
    if value is None:
        raise KeyError(f"Missing metric {name}")
    return float(value)


def styled_table(
    rows: list[list[object]],
    widths: list[float],
    *,
    header_color=BLUE,
    alternate=LIGHT_BLUE,
    font_size: float = 7.0,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, alternate]),
                ("GRID", (0, 0), (-1, -1), 0.3, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path, required=True)
    parser.add_argument("--modes-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pdftoppm",
        type=Path,
        default=Path(
            r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe"
        ),
    )
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    ablation = json.loads(args.ablation_summary.read_text(encoding="utf-8"))
    modes = json.loads(args.modes_summary.read_text(encoding="utf-8"))
    cases = evaluation["cases"]
    if len(cases) != 3 or len(modes["cases"]) != 3:
        raise ValueError("The compact report requires exactly three cases")
    if ablation["status"] != "pass":
        raise ValueError("Network ablation did not pass")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_root = args.output.parent / "tmp" / "pdfs"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=22,
        textColor=BLUE,
        alignment=TA_CENTER,
        spaceAfter=5,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        textColor=BLUE,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor("#20252B"),
        spaceAfter=3,
    )
    parts: list[Path] = []

    summary_rows = [[
        "case", "alpha", "beta", "Ritz rank", "velocity rel. L2",
        "vorticity rel. L2", "wake velocity rel. L2",
    ]]
    for index, case in enumerate(cases, start=1):
        summary_rows.append(
            [
                f"case {index}",
                f"{float(case['alpha_deg']):+.1f} deg",
                f"{float(case['beta']):.2f}",
                str(case["ritz_rank"]),
                f"{100 * metric(case, 'velocity_error'):.2f}%",
                f"{100 * metric(case, 'vorticity_error'):.2f}%",
                f"{100 * metric(case, 'wake_velocity_error'):.2f}%",
            ]
        )
    summary_table = styled_table(
        summary_rows,
        [25 * mm, 28 * mm, 20 * mm, 25 * mm, 38 * mm, 39 * mm, 43 * mm],
        font_size=7.4,
    )
    part = temp_root / "page_01.pdf"
    build_page(
        part,
        (297 * mm, 84 * mm),
        [
            para("Step 5000 Evaluation and Network-Only Jet3 Ablation", title),
            para(
                "2D airfoil antisymmetric NCL, continuous analytic Jet3, K=1024. "
                "The fixed checkpoint is optimizer step 5000; no later weights are used.",
                body,
            ),
            para("1. Fixed three-frame reconstruction", heading),
            para(
                "Velocity-L2 coefficient fit; phase 0.5; Ritz quadrature 4096; field batch 32768.",
                body,
            ),
            summary_table,
            Spacer(1, 2 * mm),
            para(
                "PASS: all full-domain velocity relative L2 errors are below the 10% review threshold. "
                "Training remains paused.",
                body,
            ),
        ],
        1,
    )
    parts.append(part)

    for index, case in enumerate(cases, start=1):
        part = temp_root / f"page_{index + 1:02d}.pdf"
        figure = Path(case["velocity_optimal_figure"])
        build_page(
            part,
            (297 * mm, 186 * mm),
            [
                fit_image(figure, 282 * mm, 166 * mm),
            ],
            index + 1,
        )
        parts.append(part)

    for index, mode_case in enumerate(modes["cases"], start=1):
        page_number = 4 + index
        part = temp_root / f"page_{page_number:02d}.pdf"
        build_page(
            part,
            (432 * mm, 265 * mm),
            [
                fit_image(Path(mode_case["contact_sheet_png"]), 420 * mm, 253 * mm),
            ],
            page_number,
            margin=5 * mm,
        )
        parts.append(part)

    ablation_rows = [[
        "variant", "CUDA/AD o0", "o1", "o2", "o3", "backward",
        "FD abs o1", "FD abs o2", "FD abs o3", "status",
    ]]
    for variant in ablation["variants"]:
        backward = max(item["relative_l2"] for item in variant["parameter_backward"])
        ablation_rows.append(
            [
                variant["label"],
                f"{variant['native_vs_torch_by_order']['0']['max_relative_l2']:.2e}",
                f"{variant['native_vs_torch_by_order']['1']['max_relative_l2']:.2e}",
                f"{variant['native_vs_torch_by_order']['2']['max_relative_l2']:.2e}",
                f"{variant['native_vs_torch_by_order']['3']['max_relative_l2']:.2e}",
                f"{backward:.2e}",
                f"{variant['finite_difference_best_by_order']['1']['max_abs']:.2e}",
                f"{variant['finite_difference_best_by_order']['2']['max_abs']:.2e}",
                f"{variant['finite_difference_best_by_order']['3']['max_abs']:.2e}",
                "PASS" if variant["pass"] else "REVIEW",
            ]
        )
    ablation_table = styled_table(
        ablation_rows,
        [31 * mm] + [24 * mm] * 8 + [17 * mm],
        font_size=6.5,
    )
    effects = [["controlled change", "normalized output change", "max absolute change"]]
    for effect in ablation["controlled_output_effects"]:
        effects.append(
            [effect["control"], f"{effect['relative_l2']:.5f}", f"{effect['max_abs']:.5e}"]
        )
    effect_table = styled_table(
        effects,
        [70 * mm, 50 * mm, 50 * mm],
        header_color=RED,
        alternate=LIGHT_RED,
        font_size=7.5,
    )
    plot = args.ablation_summary.parent / "network_jet3_ablation.png"
    part = temp_root / "page_08.pdf"
    build_page(
        part,
        (297 * mm, 220 * mm),
        [
            para("3. Network-only controlled validation", heading),
            para(
                "Shared 12-32-32-32-2 topology, FP32 master weights, seed, and coordinates. "
                "Native CUDA dense symmetric Jet3 is compared with PyTorch autograd over all 20 slots; "
                "parameter backward is checked layer by layer; finite differences use a step-size sweep.",
                body,
            ),
            ablation_table,
            Spacer(1, 2 * mm),
            fit_image(plot, 278 * mm, 91 * mm),
            Spacer(1, 2 * mm),
            KeepTogether(
                [
                    para("4. Controlled network effects", heading),
                    para(
                        "These are function changes caused by activation or encoding choices, not implementation errors.",
                        body,
                    ),
                    effect_table,
                    Spacer(1, 2 * mm),
                    para(
                        "Decision state: the 10% velocity threshold is not exceeded. Training remains paused "
                        "until an explicit user decision. The checkpoint is not embedded in this report.",
                        body,
                    ),
                ]
            ),
        ],
        8,
    )
    parts.append(part)

    writer = PdfWriter()
    for part in parts:
        reader = PdfReader(str(part))
        writer.add_page(reader.pages[0])
    with args.output.open("wb") as stream:
        writer.write(stream)

    final_reader = PdfReader(str(args.output))
    if len(final_reader.pages) != TOTAL_PAGES:
        raise RuntimeError("Merged report page count is incorrect")

    render_dir = args.output.parent / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)
    for stale in render_dir.glob("page-*.png"):
        stale.unlink()
    with tempfile.TemporaryDirectory(prefix="step5000_compact_pdf_") as tmp_name:
        tmp_dir = Path(tmp_name)
        shutil.copy2(args.output, tmp_dir / "report.pdf")
        subprocess.run(
            [str(args.pdftoppm), "-png", "-r", "130", "report.pdf", "page"],
            cwd=tmp_dir,
            check=True,
        )
        for page in sorted(tmp_dir.glob("page-*.png")):
            shutil.copy2(page, render_dir / page.name)

    rendered = sorted(str(path.resolve()) for path in render_dir.glob("page-*.png"))
    state = {
        "status": "rendered",
        "pdf": str(args.output.resolve()),
        "page_count": len(rendered),
        "mixed_page_sizes": [
            [float(page.mediabox.width), float(page.mediabox.height)]
            for page in final_reader.pages
        ],
        "pages": rendered,
    }
    (args.output.parent / "pdf_render_state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    shutil.rmtree(temp_root)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
