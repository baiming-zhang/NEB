"""Render the first nine step-5000 Ritz modes for every evaluated case.

The script reuses the Ritz rotation stored in each reconstruction sample.  It
does not recompute the generalized eigensystem and it does not retrain or
modify the checkpoint.  Raw analytic fields are evaluated in bounded point
chunks and immediately projected onto the first nine Ritz vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    inside_airfoil,
)
from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    load_hardcoded_model,
    raw_fields,
)
from run_continuous_postprocess import (  # noqa: E402
    velocity_panel,
    vorticity_panel,
    write_json,
)


N_MODES = 9


def grid_field(values: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    result = np.zeros(fluid.shape, dtype=np.float32)
    result[fluid] = values.astype(np.float32, copy=False)
    return result


def render_case(
    checkpoint: Path,
    sample_path: Path,
    output_dir: Path,
    *,
    grid_stride: int,
    point_chunk: int,
    field_batch: int,
) -> dict[str, object]:
    with np.load(sample_path, allow_pickle=False) as archive:
        x_full = archive["x"].astype(np.float64)
        y_full = archive["y"].astype(np.float64)
        beta = float(archive["beta"])
        alpha_deg = float(archive["alpha_deg"])
        eigenvalues = archive["ritz_eigenvalues"][:N_MODES].astype(np.float64)
        coefficients = archive["ritz_coefficients"][:, :N_MODES].astype(np.float64)

    def display_axis(values: np.ndarray) -> np.ndarray:
        intervals = max(1, int(np.ceil((len(values) - 1) / grid_stride)))
        return np.linspace(values[0], values[-1], intervals + 1, dtype=np.float64)

    x = display_axis(x_full)
    y = display_axis(y_full)

    cfg, model, _ = load_hardcoded_model(checkpoint)
    xgrid, ygrid = np.meshgrid(x, y)
    points_all = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    solid = inside_airfoil(cfg, points_all, beta, alpha_deg).reshape(xgrid.shape)
    fluid = ~solid
    points = points_all[fluid.ravel()].astype(np.float32)

    basis_u = np.empty((len(points), N_MODES), dtype=np.float32)
    basis_v = np.empty_like(basis_u)
    basis_omega = np.empty_like(basis_u)
    for start in range(0, len(points), point_chunk):
        stop = min(start + point_chunk, len(points))
        raw_u, raw_v, raw_omega = raw_fields(
            model,
            cfg,
            points[start:stop],
            beta,
            alpha_deg,
            batch=field_batch,
        )
        basis_u[start:stop] = (raw_u @ coefficients).astype(np.float32)
        basis_v[start:stop] = (raw_v @ coefficients).astype(np.float32)
        basis_omega[start:stop] = (raw_omega @ coefficients).astype(np.float32)

    speed_limit = max(
        float(np.percentile(np.hypot(basis_u, basis_v), 99.0)), 1.0e-8
    )
    omega_limit = max(float(np.percentile(np.abs(basis_omega), 99.0)), 1.0e-8)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        3,
        6,
        figsize=(18.0, 10.6),
        constrained_layout=True,
    )
    for index in range(N_MODES):
        row, pair = divmod(index, 3)
        velocity_axis = axes[row, 2 * pair]
        vorticity_axis = axes[row, 2 * pair + 1]
        u = grid_field(basis_u[:, index], fluid)
        v = grid_field(basis_v[:, index], fluid)
        omega = grid_field(basis_omega[:, index], fluid)
        velocity_panel(
            velocity_axis,
            x,
            y,
            u,
            v,
            solid,
            f"Mode {index + 1:02d} velocity",
            speed_limit,
        )
        vorticity_panel(
            vorticity_axis,
            x,
            y,
            omega,
            solid,
            f"Mode {index + 1:02d} vorticity | lambda={eigenvalues[index]:.4f}",
            omega_limit,
        )

    figure.suptitle(
        "First nine mass-orthogonal Ritz modes | "
        f"beta={beta:.2f}, alpha={alpha_deg:+.1f} deg",
        fontsize=16,
        fontweight="bold",
        color="#2F5D8C",
    )
    png_path = output_dir / "contact_sheet.png"
    pdf_path = output_dir / "contact_sheet.pdf"
    figure.savefig(png_path, dpi=180, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)

    summary = {
        "checkpoint": str(checkpoint.resolve()),
        "sample": str(sample_path.resolve()),
        "beta": beta,
        "alpha_deg": alpha_deg,
        "mode_count": N_MODES,
        "display_grid": [int(len(x)), int(len(y))],
        "display_grid_stride": int(grid_stride),
        "eigenvalues": eigenvalues.tolist(),
        "basis_coordinate_system": "mass_orthogonal_ritz",
        "derivatives": "continuous analytic Jet3",
        "contact_sheet_png": str(png_path.resolve()),
        "contact_sheet_pdf": str(pdf_path.resolve()),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--point-chunk", type=int, default=8192)
    parser.add_argument("--field-batch", type=int, default=32768)
    parser.add_argument("--grid-stride", type=int, default=2)
    args = parser.parse_args()

    evaluation = json.loads(args.evaluation_summary.read_text(encoding="utf-8"))
    checkpoint = Path(evaluation["checkpoint"])
    rendered: list[dict[str, object]] = []
    for case in evaluation["cases"]:
        sample = Path(case["sample"])
        output_dir = sample.parent / "basis_first9"
        existing_summary = output_dir / "summary.json"
        if existing_summary.is_file():
            existing = json.loads(existing_summary.read_text(encoding="utf-8"))
            if (
                int(existing.get("mode_count", 0)) == N_MODES
                and Path(existing.get("contact_sheet_png", "")).is_file()
                and Path(existing.get("contact_sheet_pdf", "")).is_file()
            ):
                rendered.append(existing)
                continue
        rendered.append(
            render_case(
                checkpoint,
                sample,
                output_dir,
                grid_stride=max(1, args.grid_stride),
                point_chunk=args.point_chunk,
                field_batch=args.field_batch,
            )
        )
    output = args.evaluation_summary.parent / "first9_modes_summary.json"
    write_json(output, {"status": "complete", "cases": rendered})
    print(json.dumps({"status": "complete", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
