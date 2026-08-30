"""Project an independent airfoil NS cache onto a frozen one-shot Ritz basis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eigenfluid.gen.airfoil2d.postprocess import (  # noqa: E402
    basis_fields,
    compute_ritz_basis,
    load_model,
    relative_vector_error,
    weighted_projection,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ns-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quadrature", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.ns_cache)
    metadata = json.loads(str(cache["metadata"]))
    beta = float(metadata["beta"])
    alpha_deg = float(metadata["alpha_deg"])
    cfg, model, payload = load_model(args.checkpoint)

    ritz_path = args.output_dir / "ritz_basis.npz"
    if ritz_path.exists():
        stored = np.load(ritz_path)
        ritz_coefficients = stored["coefficients"]
        eigenvalues = stored["eigenvalues"]
        rank = int(stored["rank"])
    else:
        ritz = compute_ritz_basis(
            cfg, model, beta, alpha_deg, args.quadrature
        )
        ritz_coefficients = ritz["coefficients"]
        eigenvalues = ritz["eigenvalues"]
        rank = int(ritz["rank"])
        np.savez_compressed(
            ritz_path,
            beta=beta,
            alpha_deg=alpha_deg,
            coefficients=ritz_coefficients,
            eigenvalues=eigenvalues,
            rank=rank,
        )

    xgrid, ygrid = np.meshgrid(cache["x"], cache["y"])
    airfoil = cache["airfoil"].astype(bool)
    fluid = ~airfoil
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    _, basis_u, basis_v = basis_fields(
        model, points, beta, alpha_deg, ritz_coefficients
    )
    actual_u = cache["u"][:, fluid].astype(np.float64)
    actual_v = cache["v"][:, fluid].astype(np.float64)
    mean_u = actual_u.mean(axis=0)
    mean_v = actual_v.mean(axis=0)
    predicted_u = np.empty_like(actual_u)
    predicted_v = np.empty_like(actual_v)
    coefficients = []
    errors = []
    for frame in range(len(actual_u)):
        modal = weighted_projection(
            basis_u,
            basis_v,
            actual_u[frame] - mean_u,
            actual_v[frame] - mean_v,
        )
        coefficients.append(modal)
        predicted_u[frame] = mean_u + basis_u @ modal
        predicted_v[frame] = mean_v + basis_v @ modal
        errors.append(
            relative_vector_error(
                actual_u[frame],
                actual_v[frame],
                predicted_u[frame],
                predicted_v[frame],
            )
        )
    np.savez_compressed(
        args.output_dir / "reconstruction.npz",
        x=cache["x"],
        y=cache["y"],
        fluid=fluid,
        beta=beta,
        alpha_deg=alpha_deg,
        actual_u=actual_u,
        actual_v=actual_v,
        predicted_u=predicted_u,
        predicted_v=predicted_v,
        mean_u=mean_u,
        mean_v=mean_v,
        modal_coefficients=np.asarray(coefficients),
        frame_errors=np.asarray(errors),
    )
    write_summary(
        args.output_dir / "summary.json",
        {
            "beta": beta,
            "alpha_deg": alpha_deg,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(payload["epoch"]),
            "modes": cfg.modes,
            "quadrature": args.quadrature,
            "raw_mass_rank": rank,
            "mean_velocity_relative_error": float(np.mean(errors)),
            "max_velocity_relative_error": float(np.max(errors)),
            "training_uses_ns": False,
            "inference_optimization": False,
            "method": "one-shot raw generalized Ritz basis then linear projection",
        },
    )


if __name__ == "__main__":
    main()
