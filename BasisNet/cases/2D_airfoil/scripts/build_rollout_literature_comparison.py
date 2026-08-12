"""Build the seven-column airfoil rollout report and MP4."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))
from run_continuous_postprocess import encode_mp4  # noqa: E402


NATURE = LinearSegmentedColormap.from_list(
    "nature_redblue", ["#2166AC", "#F7F7F7", "#B2182B"]
)


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_row(label: str, payload: dict) -> list[str]:
    values = payload["aggregate"]
    return [
        label,
        *[
            f"{100.0 * values[key]['mean']:.2f}% / {100.0 * values[key]['final']:.2f}%"
            for key in (
                "velocity_error",
                "wake_velocity_error",
                "vorticity_error",
                "wake_vorticity_error",
            )
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eigenfluid-fields", type=Path, required=True)
    parser.add_argument("--eigenfluid-summary", type=Path, required=True)
    parser.add_argument("--laplacian-fields", type=Path, required=True)
    parser.add_argument("--laplacian-summary", type=Path, required=True)
    parser.add_argument("--nkb-fields", type=Path, required=True)
    parser.add_argument("--nkb-summary", type=Path, required=True)
    parser.add_argument("--pod-fields", type=Path, required=True)
    parser.add_argument("--pod-summary", type=Path, required=True)
    parser.add_argument("--dmd-fields", type=Path, required=True)
    parser.add_argument("--dmd-summary", type=Path, required=True)
    parser.add_argument("--opinf-fields", type=Path, required=True)
    parser.add_argument("--opinf-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eigen = load(args.eigenfluid_fields)
    lap = load(args.laplacian_fields)
    nkb = load(args.nkb_fields)
    pod = load(args.pod_fields)
    dmd = load(args.dmd_fields)
    opinf = load(args.opinf_fields)
    summaries = {
        "EigenFluid K1024": summary(args.eigenfluid_summary),
        "Laplacian K1024": summary(args.laplacian_summary),
        "NKB K10": summary(args.nkb_summary),
        "POD-Galerkin K40": summary(args.pod_summary),
        "Exact DMD K40": summary(args.dmd_summary),
        "Reg. OpInf K7": summary(args.opinf_summary),
    }
    methods = {
        "GT": (eigen["ns_u"], eigen["ns_v"], eigen["ns_omega"]),
        "EigenFluid K1024": (eigen["eigenfluid_u"], eigen["eigenfluid_v"], eigen["eigenfluid_omega"]),
        "Laplacian K1024": (lap["laplacian_u"], lap["laplacian_v"], lap["laplacian_omega"]),
        "NKB K10": (nkb["paper_u"], nkb["paper_v"], nkb["paper_omega"]),
        "POD-Galerkin K40": (pod["predicted_u"], pod["predicted_v"], pod["predicted_omega"]),
        "Exact DMD K40": (dmd["predicted_u"], dmd["predicted_v"], dmd["predicted_omega"]),
        "Reg. OpInf K7": (opinf["predicted_u"], opinf["predicted_v"], opinf["predicted_omega"]),
    }
    x, y, solid = eigen["x"], eigen["y"], eigen["solid"].astype(bool)
    fluid = ~solid
    omega_limit = float(np.percentile(np.abs(eigen["ns_omega"][:, fluid]), 99.5))
    velocity_limit = float(np.percentile(np.sqrt(eigen["ns_u"][:, fluid] ** 2 + eigen["ns_v"][:, fluid] ** 2), 99.5))

    pdf_path = args.output_dir / "airfoil_rollout30_literature_rom_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(14.5, 8.2))
        ax.axis("off")
        ax.text(0.01, 0.96, "2-D airfoil rollout30 - literature ROM comparison", fontsize=22, weight="bold", color="#222222")
        ax.text(0.01, 0.90, "Constructed D2Q9 TRT-LBM GT, Re=1500, beta=0.50, alpha=+10 deg, 500 lattice steps per frame", fontsize=11, color="#555555")
        headers = ["Method", "velocity full", "velocity wake", "vorticity full", "vorticity wake"]
        rows = [aggregate_row(label, payload) for label, payload in summaries.items()]
        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center", bbox=(0.01, 0.39, 0.98, 0.43), colWidths=[0.25, 0.187, 0.187, 0.187, 0.187])
        table.auto_set_font_size(False); table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D0D0D0")
            if row == 0:
                cell.set_facecolor("#2166AC"); cell.get_text().set_color("white"); cell.get_text().set_weight("bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F2F5F8")
        ax.text(0.01, 0.31, "Each entry is 30-frame mean / final relative L2. Training uses 41 GT snapshots for the three literature ROMs; the 30-frame evaluation trajectory is independent and initialized from the same reconstructed velocity for every method.", fontsize=10, wrap=True)
        ax.text(0.01, 0.22, "NKB follows camera-ready Eq. (8): backward trace, Euclidean closest-domain projection, source-velocity copy, and least-squares projection. POD and DMD use the float64-resolved snapshot rank; regularized OpInf uses the largest resolved quadratic feature rank and training-trajectory stability selection.", fontsize=10, wrap=True)
        ax.text(0.01, 0.10, "Primary references: Noack et al. (POD-Galerkin); Schmid 2010 (DMD); McQuarrie, Huang & Willcox 2021 (regularized Operator Inference); Liu et al. 2025 (Neural Kinematic Bases).", fontsize=9, color="#555555")
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for frame in range(30):
            fig, axes = plt.subplots(2, len(methods), figsize=(17, 5.7), constrained_layout=True)
            for column, (label, (u, v, omega)) in enumerate(methods.items()):
                speed = np.sqrt(u[frame] ** 2 + v[frame] ** 2)
                im0 = axes[0, column].imshow(np.ma.masked_where(solid, speed), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="viridis", vmin=0.0, vmax=velocity_limit, rasterized=True)
                im1 = axes[1, column].imshow(np.ma.masked_where(solid, omega[frame]), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap=NATURE, vmin=-omega_limit, vmax=omega_limit, rasterized=True)
                axes[0, column].set_title(label, fontsize=9, weight="bold")
                if label != "GT":
                    row = summaries[label]["metrics"][frame]
                    axes[0, column].text(0.02, 0.03, f"L2 {100*row['velocity_error']:.1f}% | wake {100*row['wake_velocity_error']:.1f}%", transform=axes[0, column].transAxes, fontsize=6.7, color="white", bbox=dict(facecolor="black", alpha=.55, pad=1.5))
                    axes[1, column].text(0.02, 0.03, f"L2 {100*row['vorticity_error']:.1f}% | wake {100*row['wake_vorticity_error']:.1f}%", transform=axes[1, column].transAxes, fontsize=6.7, color="black", bbox=dict(facecolor="white", alpha=.70, pad=1.5))
                for axis in axes[:, column]:
                    axis.set_xlim(0, 10); axis.set_ylim(-4, 4); axis.set_xticks([]); axis.set_yticks([])
            axes[0, 0].set_ylabel("speed", fontsize=9); axes[1, 0].set_ylabel("vorticity", fontsize=9)
            fig.suptitle(f"Rollout frame {frame + 1:02d}/30 · local LBM step {(frame + 1) * 500}", fontsize=13, weight="bold")
            pdf.savefig(fig, bbox_inches="tight", dpi=160); plt.close(fig)

    frames_dir = args.output_dir / "video_frames"
    frames_dir.mkdir(exist_ok=True)
    for stale_frame in frames_dir.glob("frame_*.png"):
        stale_frame.unlink()
    for frame in range(30):
        fig, axes = plt.subplots(1, len(methods), figsize=(17, 3.3), constrained_layout=True)
        for column, (label, (_, _, omega)) in enumerate(methods.items()):
            axes[column].imshow(np.ma.masked_where(solid, omega[frame]), origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap=NATURE, vmin=-omega_limit, vmax=omega_limit)
            axes[column].set_title(label, fontsize=9, weight="bold")
            axes[column].set_xticks([]); axes[column].set_yticks([])
            if label != "GT":
                row = summaries[label]["metrics"][frame]
                axes[column].text(.02,.03,f"vort L2 {100*row['vorticity_error']:.1f}%",transform=axes[column].transAxes,fontsize=7,bbox=dict(facecolor="white",alpha=.72,pad=1.4))
        fig.suptitle(f"2-D airfoil vorticity rollout · frame {frame + 1:02d}/30", fontsize=13, weight="bold")
        fig.savefig(frames_dir / f"frame_{frame:03d}.png", dpi=145, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    mp4_path = args.output_dir / "airfoil_rollout30_literature_rom_comparison_10fps.mp4"
    encode_mp4(frames_dir, mp4_path, args.fps)

    preview = args.output_dir / "airfoil_rollout30_literature_rom_comparison_preview.png"
    first = frames_dir / "frame_000.png"
    preview.write_bytes(first.read_bytes())
    csv_path = args.output_dir / "rollout30_literature_rom_aggregate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["method", "metric", "mean", "p95", "max", "final"])
        for label, payload in summaries.items():
            for key, values in payload["aggregate"].items():
                writer.writerow([label, key, values["mean"], values["p95"], values["max"], values["final"]])
    print(json.dumps({"pdf": str(pdf_path), "mp4": str(mp4_path), "preview": str(preview), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
