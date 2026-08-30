"""Cross-evaluate checkpoints and validation samplers after the backend switch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    body_numpy,
    cfg_namespace,
    fluid_points,
    naca_half_thickness_numpy,
    peel_step,
)


def load_model(path: Path) -> tuple[Any, HardcodedModel]:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    return cfg, HardcodedModel(cfg, arrays)


def sample_diagnostics(
    cfg: Any, points: np.ndarray, beta: float, alpha: float
) -> dict[str, float]:
    xb, yb = body_numpy(cfg, points, alpha)
    half = naca_half_thickness_numpy(cfg, xb / cfg.chord, beta)
    surface_gap = np.abs(yb) - half
    in_chord = (xb >= 0.0) & (xb <= cfg.chord)
    return {
        "minimum_absolute_surface_gap_in_chord": float(
            np.min(np.abs(surface_gap[in_chord]))
        ),
        "minimum_x_over_c_in_chord": float(np.min(xb[in_chord] / cfg.chord)),
        "maximum_x_over_c_in_chord": float(np.max(xb[in_chord] / cfg.chord)),
    }


def evaluate(
    label: str,
    cfg: Any,
    model: HardcodedModel,
    points_by_case: np.ndarray,
    betas: np.ndarray,
    alphas: np.ndarray,
) -> dict[str, Any]:
    cases = []
    for index, (points, beta, alpha) in enumerate(
        zip(points_by_case, betas, alphas)
    ):
        result = peel_step(
            cfg, model, points, float(beta), float(alpha), training=False
        )
        cases.append(
            {
                "index": index,
                "beta": float(beta),
                "alpha": float(alpha),
                "loss": result["loss"],
                "maximum_mode_energy_over_80": (
                    result["max_energy"] / cfg.rayleigh_scale
                ),
                "rank": result["effective_raw_rank"],
                **sample_diagnostics(cfg, points, float(beta), float(alpha)),
            }
        )
    return {
        "label": label,
        "mean_loss": float(np.mean([case["loss"] for case in cases])),
        "maximum_case_loss": float(np.max([case["loss"] for case in cases])),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch100", type=Path, required=True)
    parser.add_argument("--epoch200", type=Path, required=True)
    parser.add_argument("--sample-pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg100, model100 = load_model(args.epoch100)
    cfg200, model200 = load_model(args.epoch200)
    with np.load(args.sample_pack, allow_pickle=False) as archive:
        old_points = archive["validation_points"]
        betas = archive["validation_beta"]
        alphas = archive["validation_alpha"]
    new_points = np.stack(
        [
            fluid_points(
                cfg100,
                cfg100.validation_points,
                float(beta),
                float(alpha),
                cfg100.seed + 900_000 + index,
            )
            for index, (beta, alpha) in enumerate(zip(betas, alphas))
        ]
    )
    report = {
        "old_and_new_validation_points_identical": bool(
            np.array_equal(old_points, new_points)
        ),
        "mean_pointwise_distance": float(
            np.mean(np.linalg.norm(old_points - new_points, axis=-1))
        ),
        "evaluations": [
            evaluate(
                "epoch100_on_original_torch_sobol_points",
                cfg100,
                model100,
                old_points,
                betas,
                alphas,
            ),
            evaluate(
                "epoch100_on_scipy_sobol_points",
                cfg100,
                model100,
                new_points,
                betas,
                alphas,
            ),
            evaluate(
                "epoch200_on_original_torch_sobol_points",
                cfg200,
                model200,
                old_points,
                betas,
                alphas,
            ),
            evaluate(
                "epoch200_on_scipy_sobol_points",
                cfg200,
                model200,
                new_points,
                betas,
                alphas,
            ),
        ],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
