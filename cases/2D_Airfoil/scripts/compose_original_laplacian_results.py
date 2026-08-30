from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ORIENTATIONS = ("up", "middle", "down")
K_VALUES = (64, 128, 256, 512, 1024, 2048)
METRICS = (
    ("velocity_error", "velocity", "Overall velocity"),
    ("vorticity_error", "vorticity", "Overall vorticity"),
    ("fluctuation_velocity_error", "fluctuation_velocity", "Fluctuation velocity"),
    ("fluctuation_vorticity_error", "fluctuation_vorticity", "Fluctuation vorticity"),
    ("wake_velocity_error", "wake_velocity", "Wake velocity"),
    ("wake_vorticity_error", "wake_vorticity", "Wake vorticity"),
)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return rows


def _write_tables(rows: list[dict[str, str]], output_dir: Path) -> dict[str, object]:
    lookup = {
        (row["orientation"], row["method"], int(row["modes"])): row
        for row in rows
        if int(row["frame"]) == 30
    }

    by_case_path = output_dir / "reconstruction_summary_by_case.csv"
    with by_case_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["orientation", "frame", "modes"]
        for _, short_name, _ in METRICS:
            fields.extend((f"original_{short_name}_error", f"laplacian_{short_name}_error"))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for orientation in ORIENTATIONS:
            for modes in K_VALUES:
                learned = lookup[(orientation, "learned", modes)]
                laplacian = lookup[(orientation, "laplacian", modes)]
                case_row: dict[str, object] = {
                    "orientation": orientation,
                    "frame": 30,
                    "modes": modes,
                }
                for source_name, short_name, _ in METRICS:
                    case_row[f"original_{short_name}_error"] = learned[source_name]
                    case_row[f"laplacian_{short_name}_error"] = laplacian[source_name]
                writer.writerow(case_row)

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["frame"]) == 30:
            grouped[(row["method"], int(row["modes"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    previous: dict[tuple[str, str], float] = {}
    for modes in K_VALUES:
        values: dict[str, object] = {}
        for method in ("learned", "laplacian"):
            group = grouped[(method, modes)]
            if len(group) != 3:
                raise ValueError(f"Expected three frame-30 cases for {method} K={modes}")
            prefix = "original" if method == "learned" else "laplacian"
            for source_name, short_name, _ in METRICS:
                mean_value = sum(float(row[source_name]) for row in group) / len(group)
                values[f"{prefix}_{short_name}_mean"] = mean_value
                previous_key = (method, short_name)
                if previous_key in previous:
                    previous_value = previous[previous_key]
                    values[f"{prefix}_{short_name}_marginal_reduction_pct"] = (
                        100.0 * (previous_value - mean_value) / previous_value
                    )
                else:
                    values[f"{prefix}_{short_name}_marginal_reduction_pct"] = ""
                previous[previous_key] = mean_value

        summary_row: dict[str, object] = {"modes": modes, **values}
        for _, short_name, _ in METRICS:
            original_value = float(values[f"original_{short_name}_mean"])
            laplacian_value = float(values[f"laplacian_{short_name}_mean"])
            summary_row[f"laplacian_vs_original_{short_name}_delta_pct"] = (
                100.0 * (original_value - laplacian_value) / original_value
            )
        summary_rows.append(summary_row)

    by_k_path = output_dir / "reconstruction_summary_by_k.csv"
    with by_k_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = output_dir / "reconstruction_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "sample_count": 3,
                "frame": 30,
                "orientations": list(ORIENTATIONS),
                "modes": list(K_VALUES),
                "summary_by_k": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"summary_rows": summary_rows, "by_case": by_case_path, "by_k": by_k_path, "json": json_path}


def _plot_metric_summary(summary_rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    canvas_w, canvas_h = 3000, 3320
    image = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    navy = (28, 55, 86)
    original_color = (33, 102, 172)
    laplacian_color = (178, 24, 43)
    grid_color = (214, 216, 219)
    axis_color = (70, 75, 80)
    title_font = _font(52, bold=True)
    panel_title_font = _font(34, bold=True)
    axis_font = _font(25)
    tick_font = _font(23)
    legend_font = _font(29)
    value_font = _font(22, bold=True)

    _draw_centered(
        draw,
        "2D Airfoil Reconstruction Error Across Three Unseen Cases",
        canvas_w // 2,
        30,
        title_font,
        navy,
    )
    legend_y = 112
    legend_center = canvas_w // 2
    draw.line((legend_center - 420, legend_y + 15, legend_center - 350, legend_y + 15), fill=original_color, width=8)
    draw.ellipse((legend_center - 394, legend_y + 2, legend_center - 368, legend_y + 28), fill=original_color)
    draw.text((legend_center - 330, legend_y - 2), "Original learned", font=legend_font, fill=axis_color)
    draw.line((legend_center + 55, legend_y + 15, legend_center + 125, legend_y + 15), fill=laplacian_color, width=8)
    draw.rectangle((legend_center + 77, legend_y + 2, legend_center + 103, legend_y + 28), fill=laplacian_color)
    draw.text((legend_center + 145, legend_y - 2), "Airfoil FD Laplacian", font=legend_font, fill=axis_color)

    outer_left, outer_right = 90, 70
    outer_top, outer_bottom = 190, 80
    gutter_x, gutter_y = 80, 80
    panel_w = (canvas_w - outer_left - outer_right - gutter_x) // 2
    panel_h = (canvas_h - outer_top - outer_bottom - 2 * gutter_y) // 3

    for panel_idx, (_, short_name, title) in enumerate(METRICS):
        row_idx, col_idx = divmod(panel_idx, 2)
        px = outer_left + col_idx * (panel_w + gutter_x)
        py = outer_top + row_idx * (panel_h + gutter_y)
        draw.text((px, py), title, font=panel_title_font, fill=navy)
        plot_left = px + 145
        plot_right = px + panel_w - 42
        plot_top = py + 70
        plot_bottom = py + panel_h - 120
        original = [100.0 * float(item[f"original_{short_name}_mean"]) for item in summary_rows]
        laplacian = [100.0 * float(item[f"laplacian_{short_name}_mean"]) for item in summary_rows]
        y_max = max(max(original), max(laplacian)) * 1.16
        if y_max <= 0.0:
            y_max = 1.0

        x_positions = [
            int(plot_left + i * (plot_right - plot_left) / (len(K_VALUES) - 1))
            for i in range(len(K_VALUES))
        ]

        for tick in range(5):
            value = y_max * tick / 4.0
            y = int(plot_bottom - (plot_bottom - plot_top) * tick / 4.0)
            draw.line((plot_left, y, plot_right, y), fill=grid_color, width=2)
            label = f"{value:.1f}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((plot_left - (box[2] - box[0]) - 18, y - 12), label, font=tick_font, fill=axis_color)
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=3)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=3)

        for x, modes in zip(x_positions, K_VALUES):
            draw.line((x, plot_bottom, x, plot_bottom + 8), fill=axis_color, width=2)
            label = str(modes)
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text((x - (box[2] - box[0]) / 2, plot_bottom + 13), label, font=tick_font, fill=axis_color)

        def y_position(value: float) -> int:
            return int(plot_bottom - value / y_max * (plot_bottom - plot_top))

        original_points = [(x, y_position(value)) for x, value in zip(x_positions, original)]
        laplacian_points = [(x, y_position(value)) for x, value in zip(x_positions, laplacian)]
        draw.line(original_points, fill=original_color, width=7, joint="curve")
        draw.line(laplacian_points, fill=laplacian_color, width=7, joint="curve")
        for x, y in original_points:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=original_color)
        for x, y in laplacian_points:
            draw.rectangle((x - 9, y - 9, x + 9, y + 9), fill=laplacian_color)

        draw.text((plot_right - 84, original_points[-1][1] - 34), f"{original[-1]:.2f}%", font=value_font, fill=original_color)
        draw.text((plot_right - 84, laplacian_points[-1][1] + 12), f"{laplacian[-1]:.2f}%", font=value_font, fill=laplacian_color)
        _draw_centered(draw, "Number of modes, K", (plot_left + plot_right) // 2, plot_bottom + 52, axis_font, axis_color)
        _paste_vertical_text(
            image,
            "Mean relative L2 error (%)",
            px + 6,
            (plot_top + plot_bottom) // 2,
            axis_font,
            axis_color,
        )

    png_path = output_dir / "reconstruction_error_summary_all_metrics.png"
    pdf_path = output_dir / "reconstruction_error_summary_all_metrics.pdf"
    image.save(png_path, format="PNG", optimize=True, dpi=(300, 300))
    page_w = 11.0 * 72.0
    page_h = page_w * canvas_h / canvas_w
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h), pageCompression=1)
    pdf.drawImage(ImageReader(image), 0, 0, width=page_w, height=page_h, preserveAspectRatio=True)
    pdf.showPage()
    pdf.save()
    return png_path, pdf_path


def _draw_centered(draw: ImageDraw.ImageDraw, text: str, center_x: int, y: int, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, font=font, fill=fill)


def _paste_vertical_text(image: Image.Image, text: str, x: int, center_y: int, font, fill) -> None:
    probe = ImageDraw.Draw(image)
    box = probe.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 12
    height = box[3] - box[1] + 12
    label = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((6, 6 - box[1]), text, font=font, fill=fill)
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (x, center_y - rotated.height // 2), rotated)


def _compose_figure(output_dir: Path) -> tuple[Path, Path]:
    panels: dict[tuple[str, str], Image.Image] = {}
    for orientation in ORIENTATIONS:
        for quantity in ("speed", "vorticity"):
            path = output_dir / f"{orientation}_frame30_{quantity}_2x{1 + len(K_VALUES)}.png"
            panels[(orientation, quantity)] = Image.open(path).convert("RGB")

    panel_w, panel_h = panels[("up", "speed")].size
    left = 150
    right = 60
    top = 185
    bottom = 55
    gutter_x = 44
    gutter_y = 42
    width = left + 2 * panel_w + gutter_x + right
    height = top + 3 * panel_h + 2 * gutter_y + bottom
    background = (250, 250, 248)
    navy = (28, 55, 86)
    red = (176, 47, 54)
    grey = (108, 116, 124)
    figure = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(figure)

    title_font = _font(64, bold=True)
    header_font = _font(46, bold=True)
    row_font = _font(38, bold=True)
    small_font = _font(28)
    _draw_centered(
        draw,
        "2D Airfoil Reconstruction: Original vs Geometry-aware FD Laplacian",
        width // 2,
        22,
        title_font,
        navy,
    )
    _draw_centered(draw, "Speed magnitude", left + panel_w // 2, 104, header_font, navy)
    _draw_centered(
        draw,
        "Vorticity",
        left + panel_w + gutter_x + panel_w // 2,
        104,
        header_font,
        red,
    )
    draw.line((left, 162, width - right, 162), fill=grey, width=3)

    row_labels = {"up": "Up", "middle": "Middle", "down": "Down"}
    for row_idx, orientation in enumerate(ORIENTATIONS):
        y = top + row_idx * (panel_h + gutter_y)
        label = row_labels[orientation]
        label_box = draw.textbbox((0, 0), label, font=row_font)
        label_w = label_box[2] - label_box[0]
        label_h = label_box[3] - label_box[1]
        draw.text(((left - label_w) / 2, y + panel_h / 2 - label_h / 2), label, font=row_font, fill=navy)
        draw.text((18, y + panel_h / 2 + 40), "frame 30", font=small_font, fill=grey)
        for col_idx, quantity in enumerate(("speed", "vorticity")):
            x = left + col_idx * (panel_w + gutter_x)
            figure.paste(panels[(orientation, quantity)], (x, y))
            draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1), outline=(185, 190, 195), width=2)

    png_path = output_dir / "original_vs_laplacian_three_cases_combined.png"
    figure.save(png_path, format="PNG", optimize=True, dpi=(300, 300))

    pdf_path = output_dir / "original_vs_laplacian_three_cases_combined.pdf"
    page_w = 20.0 * 72.0
    page_h = page_w * height / width
    pdf = canvas.Canvas(str(pdf_path), pagesize=(page_w, page_h), pageCompression=1)
    pdf.drawImage(ImageReader(figure), 0, 0, width=page_w, height=page_h, preserveAspectRatio=True)
    pdf.showPage()
    pdf.save()
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    rows = _read_metrics(output_dir / "reconstruction_metrics.csv")
    tables = _write_tables(rows, output_dir)
    png_path, pdf_path = _compose_figure(output_dir)
    metrics_png, metrics_pdf = _plot_metric_summary(tables["summary_rows"], output_dir)
    print(
        json.dumps(
            {
                "tables": {k: str(v) for k, v in tables.items() if k != "summary_rows"},
                "combined_png": str(png_path),
                "combined_pdf": str(pdf_path),
                "metrics_png": str(metrics_png),
                "metrics_pdf": str(metrics_pdf),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
