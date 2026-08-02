from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from run_continuous_postprocess import (  # noqa: E402
    reconstruct_case,
    render_frame,
    render_lowest_six,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate velocity- and vorticity-optimal reconstructions on one GT frame"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", type=float, default=0.5)
    parser.add_argument("--ritz-quadrature", type=int, default=4096)
    parser.add_argument("--field-batch", type=int, default=32768)
    parser.add_argument(
        "--render-lowest-six",
        action="store_true",
        help="Render the six lowest-frequency mass-orthogonal Ritz modes",
    )
    parser.add_argument("--basis-beta", type=float)
    parser.add_argument("--basis-alpha-deg", type=float)
    parser.add_argument("--basis-grid-nx", type=int, default=501)
    parser.add_argument("--basis-grid-ny", type=int, default=401)
    parser.add_argument(
        "--velocity-only",
        action="store_true",
        help="Fit and render only the velocity-L2-optimal reconstruction",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "reconstruction_sample.npz"
    reconstruct_case(
        args.checkpoint,
        args.gt,
        sample_path,
        phase=args.phase,
        quadrature=args.ritz_quadrature,
        batch=args.field_batch,
        velocity_only=args.velocity_only,
    )
    with np.load(sample_path, allow_pickle=False) as archive:
        sample = {name: archive[name] for name in archive.files}

    velocity_error = np.hypot(
        sample["pinn_u"] - sample["actual_u"],
        sample["pinn_v"] - sample["actual_v"],
    )
    vorticity_error = np.abs(sample["pinn_omega"] - sample["actual_omega"])
    limits = {
        "speed": max(
            float(np.percentile(np.abs(sample["actual_u"]), 99.5)),
            float(np.percentile(np.abs(sample["actual_v"]), 99.5)),
            1.0e-8,
        ),
        "vorticity": max(
            float(np.percentile(np.abs(sample["actual_omega"]), 99.0)),
            1.0e-8,
        ),
        "velocity_error": max(
            float(np.percentile(velocity_error, 99.0)), 1.0e-8
        ),
        "vorticity_error": max(
            float(np.percentile(vorticity_error, 99.0)), 1.0e-8
        ),
    }
    velocity_figure = args.output_dir / "velocity_optimal_comparison.png"
    render_frame(sample, limits, velocity_figure, variant="velocity")
    vorticity_figure = None
    if not args.velocity_only:
        vorticity_opt_velocity_error = np.hypot(
            sample["vorticity_opt_u"] - sample["actual_u"],
            sample["vorticity_opt_v"] - sample["actual_v"],
        )
        vorticity_opt_vorticity_error = np.abs(
            sample["vorticity_opt_omega"] - sample["actual_omega"]
        )
        limits["vorticity_opt_velocity_error"] = max(
            float(np.percentile(vorticity_opt_velocity_error, 99.0)),
            1.0e-8,
        )
        limits["vorticity_opt_vorticity_error"] = max(
            float(np.percentile(vorticity_opt_vorticity_error, 99.0)),
            1.0e-8,
        )
        vorticity_figure = (
            args.output_dir / "vorticity_optimal_comparison.png"
        )
        render_frame(sample, limits, vorticity_figure, variant="vorticity")

    basis_dir = args.output_dir / "basis_lowest6"
    basis_summary = None
    if args.render_lowest_six:
        basis_settings = {
            "basis_beta": (
                args.basis_beta
                if args.basis_beta is not None
                else float(sample["beta"])
            ),
            "basis_alpha_deg": (
                args.basis_alpha_deg
                if args.basis_alpha_deg is not None
                else float(sample["alpha_deg"])
            ),
            "basis_grid_nx": args.basis_grid_nx,
            "basis_grid_ny": args.basis_grid_ny,
            "ritz_quadrature": args.ritz_quadrature,
            "field_batch": args.field_batch,
        }
        render_lowest_six(args.checkpoint, basis_dir, basis_settings)
        basis_summary = json.loads(
            (basis_dir / "summary.json").read_text(encoding="utf-8")
        )

    scalar_metrics = {}
    for name, value in sample.items():
        if name.endswith("_error") and np.asarray(value).ndim == 0:
            scalar_metrics[name] = float(value)
    with np.load(args.gt, allow_pickle=False) as archive:
        gt_metadata = json.loads(str(archive["metadata"]))
        snapshot_count = int(archive["u"].shape[0])
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "gt": str(args.gt.resolve()),
        "beta": float(sample["beta"]),
        "alpha_deg": float(sample["alpha_deg"]),
        "selected_frame": int(sample["frame"]),
        "snapshot_count": snapshot_count,
        "phase": args.phase,
        "ritz_rank": int(sample["ritz_rank"]),
        "coefficient_fit": str(sample["coefficient_fit"]),
        "projection_ritz_orth_error": float(
            sample["projection_ritz_orth_error"]
        ),
        "metrics": scalar_metrics,
        "gt_metadata": gt_metadata,
        "velocity_optimal_figure": str(velocity_figure.resolve()),
        "vorticity_optimal_figure": (
            str(vorticity_figure.resolve())
            if vorticity_figure is not None
            else None
        ),
        "sample": str(sample_path.resolve()),
        "basis_lowest6": (
            {
                "contact_sheet": str(
                    (basis_dir / "contact_sheet.png").resolve()
                ),
                "summary": basis_summary,
            }
            if basis_summary is not None
            else None
        ),
    }
    if not args.velocity_only:
        summary.update(
            vorticity_projection_rank=int(
                sample["vorticity_projection_rank"]
            ),
            vorticity_projection_relative_residual=float(
                sample["vorticity_projection_relative_residual"]
            ),
        )
    summary_path = args.output_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
