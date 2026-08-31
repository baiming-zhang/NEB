from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--eigenfluid-root", type=Path, required=True)
parser.add_argument("--laplacian-root", type=Path, required=True)
parser.add_argument("--nkb-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--frame-dir", type=Path, required=True)
args = parser.parse_args()

OUTPUT = args.output_dir
OUTPUT.mkdir(parents=True, exist_ok=True)
PDF = OUTPUT / "airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases.pdf"
PREVIEW = (
    OUTPUT
    / "airfoil_rollout30_gt_eigenfluid_laplacian_neural_kinematic_bases_preview.png"
)
FRAMES = args.frame_dir
FRAMES.mkdir(parents=True, exist_ok=True)

RED = "#B40426"
BLUE = "#3B4CC0"
INK = "#17212B"
GRAY = "#667085"
PALE = "#F4F7FA"
NATURE_RDBU = LinearSegmentedColormap.from_list(
    "nature_rdblu", [BLUE, "#DCE6F2", "#FFFFFF", "#F6D6D0", RED]
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
    }
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


eig_summary = load_json(args.eigenfluid_root / "rollout_summary.json")
lap_summary = load_json(args.laplacian_root / "rollout_summary.json")
paper_summary = load_json(args.nkb_root / "rollout_summary.json")

eig_npz = np.load(
    args.eigenfluid_root / "ns_vs_eigenfluid_rollout_fields.npz", mmap_mode="r"
)
lap_npz = np.load(
    args.laplacian_root / "ns_vs_laplacian_rollout_fields.npz", mmap_mode="r"
)
pap_npz = np.load(
    args.nkb_root / "ns_vs_paper_camera_ready_rollout_fields.npz", mmap_mode="r"
)
x, y, solid = eig_npz["x"], eig_npz["y"], eig_npz["solid"].astype(bool)
extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
fluid = ~solid
mask = np.where(solid, 1.0, np.nan)

gt_w = eig_npz["ns_omega"]
fields_w = {
    "GT": gt_w,
    "Eigenfluid, K1024": eig_npz["eigenfluid_omega"],
    "Laplacian, K1024": lap_npz["laplacian_omega"],
    "NKB Eq. (8), K10": pap_npz["paper_omega"],
}
summaries = {
    "Eigenfluid": eig_summary,
    "Laplacian": lap_summary,
    "NKB Eq. (8)": paper_summary,
}

gt_sample = np.abs(gt_w[:, fluid]).ravel()[::30]
omega_limit = float(np.percentile(gt_sample, 99.5))
error_values = []
for values in list(fields_w.values())[1:]:
    error_values.append(np.abs(values[:, fluid] - gt_w[:, fluid]).ravel()[::50])
error_limit = float(np.percentile(np.concatenate(error_values), 98.5))


def add_header(fig, title, subtitle=""):
    fig.text(0.04, 0.965, title, fontsize=17, color=INK, weight="bold", va="top")
    if subtitle:
        fig.text(0.04, 0.925, subtitle, fontsize=9, color=GRAY, va="top")
    fig.add_artist(
        plt.Line2D(
            [0.04, 0.96], [0.905, 0.905], transform=fig.transFigure, color=BLUE, lw=2.2
        )
    )


def aggregate_cell(summary, key):
    d = summary["aggregate"][key]
    return f"{100 * d['mean']:.2f} / {100 * d['final']:.2f}"


with PdfPages(PDF) as pdf:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    add_header(
        fig,
        "Airfoil rollout30",
        "GT, Eigenfluid K1024, Laplacian K1024, Neural Kinematic Bases K10",
    )
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.79])
    ax.axis("off")
    ax.text(
        0,
        0.95,
        "Mean / final relative L2 error (%)",
        fontsize=13,
        weight="bold",
        color=INK,
        va="top",
    )
    columns = [
        "Method",
        "velocity full",
        "velocity wake",
        "vorticity full",
        "vorticity wake",
    ]
    rows = []
    for name, summary in summaries.items():
        rows.append(
            [
                name,
                aggregate_cell(summary, "velocity_error"),
                aggregate_cell(summary, "wake_velocity_error"),
                aggregate_cell(summary, "vorticity_error"),
                aggregate_cell(summary, "wake_vorticity_error"),
            ]
        )
    table = ax.table(
        cellText=rows, colLabels=columns, bbox=[0, 0.55, 1, 0.27], cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(BLUE if r == 0 else ("#EDF3FA" if r % 2 else "#F8FAFC"))
        cell.set_text_props(
            color="white" if r == 0 else INK, weight="bold" if r == 0 else "normal"
        )
    protocol = [
        ["GT", "D2Q9 TRT-LBM Navier-Stokes"],
        ["Eigenfluid", "pure-vorticity epoch 5000; variational ROM"],
        ["Laplacian", "same variational ROM; rectangular basis"],
        ["NKB", "camera-ready Eq. (8) reproduction"],
    ]
    protocol_table = ax.table(
        cellText=protocol,
        colLabels=["Method", "Propagation"],
        bbox=[0, 0.18, 1, 0.25],
        cellLoc="left",
        colWidths=[0.18, 0.82],
    )
    protocol_table.auto_set_font_size(False)
    protocol_table.set_fontsize(9)
    for (r, c), cell in protocol_table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(RED if r == 0 else "#F8FAFC")
        cell.set_text_props(
            color="white" if r == 0 else INK, weight="bold" if r == 0 else "normal"
        )
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), facecolor="white")
    fig.subplots_adjust(
        left=0.08, right=0.97, bottom=0.08, top=0.86, wspace=0.22, hspace=0.32
    )
    add_header(
        fig, "Rollout error curves", "Relative L2 against the same GT trajectory"
    )
    keys = [
        ("velocity_error", "Velocity - full domain"),
        ("wake_velocity_error", "Velocity - wake"),
        ("vorticity_error", "Vorticity - full domain"),
        ("wake_vorticity_error", "Vorticity - wake"),
    ]
    colors = {"Eigenfluid": BLUE, "Laplacian": "#8C6D31", "NKB Eq. (8)": RED}
    for ax, (key, title) in zip(axes.flat, keys):
        for name, summary in summaries.items():
            ax.plot(
                range(1, 31),
                [100 * row[key] for row in summary["metrics"]],
                lw=2,
                label=name,
                color=colors[name],
            )
        ax.set_title(title)
        ax.set_xlabel("forecast frame")
        ax.set_ylabel("relative L² (%)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)

    for frame in range(30):
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        fig.subplots_adjust(
            left=0.035, right=0.97, bottom=0.09, top=0.84, wspace=0.05, hspace=0.22
        )
        add_header(
            fig,
            f"Vorticity rollout - frame {frame + 1:02d}/30",
            f"shared field scale +/-{omega_limit:.3g}; shared absolute-error scale 0...{error_limit:.3g}",
        )
        gs = fig.add_gridspec(2, 4)
        method_keys = list(fields_w.keys())
        images = []
        for col, name in enumerate(method_keys):
            ax = fig.add_subplot(gs[0, col])
            values = fields_w[name][frame]
            im = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=NATURE_RDBU,
                vmin=-omega_limit,
                vmax=omega_limit,
                interpolation="bilinear",
            )
            ax.imshow(
                mask,
                origin="lower",
                extent=extent,
                cmap="gray_r",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_title(name, fontsize=8.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            images.append(im)
            ax2 = fig.add_subplot(gs[1, col])
            if col == 0:
                ax2.axis("off")
                ax2.text(
                    0.5,
                    0.58,
                    "GT reference",
                    transform=ax2.transAxes,
                    ha="center",
                    va="center",
                    fontsize=12,
                    weight="bold",
                    color=INK,
                )
                ax2.text(
                    0.5,
                    0.40,
                    "Error panels use absolute vorticity error",
                    transform=ax2.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=GRAY,
                )
            else:
                error = np.abs(values - gt_w[frame])
                eim = ax2.imshow(
                    error,
                    origin="lower",
                    extent=extent,
                    cmap="magma",
                    vmin=0,
                    vmax=error_limit,
                    interpolation="bilinear",
                )
                ax2.imshow(
                    mask,
                    origin="lower",
                    extent=extent,
                    cmap="gray_r",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
                summary = list(summaries.values())[col - 1]
                row = summary["metrics"][frame]
                ax2.set_title(
                    f"|error| - full {100 * row['vorticity_error']:.1f}% - wake {100 * row['wake_vorticity_error']:.1f}%",
                    fontsize=7.4,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax2.set_aspect("equal")
        cax = fig.add_axes([0.17, 0.045, 0.30, 0.018])
        fig.colorbar(images[0], cax=cax, orientation="horizontal").set_label(
            "vorticity", fontsize=7
        )
        e_cax = fig.add_axes([0.60, 0.045, 0.30, 0.018])
        fig.colorbar(eim, cax=e_cax, orientation="horizontal").set_label(
            "absolute vorticity error", fontsize=7
        )
        png = FRAMES / f"frame_{frame:03d}.png"
        fig.savefig(png, dpi=120, facecolor="white")
        if frame == 29:
            fig.savefig(PREVIEW, dpi=160, facecolor="white")
        pdf.savefig(fig, dpi=140)
        plt.close(fig)

print(PDF)
print(PREVIEW)
