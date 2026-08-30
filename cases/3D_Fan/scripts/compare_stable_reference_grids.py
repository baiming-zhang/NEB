"""Compare equal-phase stable fan references on successive uniform grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import map_coordinates


CASE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            CASE_ROOT
            / "results"
            / "gt_rotating_stable_re500_nominal_n96"
            / "grid_sensitivity.json"
        ),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_reference(directory: str) -> dict[str, np.ndarray]:
    path = CASE_ROOT / "results" / directory / "stable_fan_reference.h5"
    with h5py.File(path, "r") as archive:
        return {
            "axis": archive["axis"][:].astype(np.float64),
            "velocity": archive["velocity"][:].astype(np.float64),
            "rotor_mask": archive["rotor_mask"][:].astype(np.float64),
        }


def interpolate_fine(
    fine: dict[str, np.ndarray],
    coarse_axis: np.ndarray,
) -> np.ndarray:
    xx, yy, zz = np.meshgrid(
        coarse_axis,
        coarse_axis,
        coarse_axis,
        indexing="ij",
    )
    fine_axis = fine["axis"]
    coordinates = np.vstack(
        (
            (xx.ravel() - fine_axis[0]) / (fine_axis[1] - fine_axis[0]),
            (yy.ravel() - fine_axis[0]) / (fine_axis[1] - fine_axis[0]),
            (zz.ravel() - fine_axis[0]) / (fine_axis[1] - fine_axis[0]),
        )
    )
    return np.stack(
        [
            map_coordinates(
                fine["velocity"][component],
                coordinates,
                order=1,
                mode="grid-wrap",
                prefilter=False,
            ).reshape(xx.shape)
            for component in range(3)
        ]
    )


def relative_l2(
    coarse: np.ndarray,
    fine_on_coarse: np.ndarray,
    mask: np.ndarray,
    subtract_inlet: bool,
) -> float:
    first = coarse.copy()
    second = fine_on_coarse.copy()
    if subtract_inlet:
        first[0] -= 1.0
        second[0] -= 1.0
    return float(
        np.linalg.norm((second[:, mask] - first[:, mask]).ravel())
        / max(np.linalg.norm(first[:, mask].ravel()), 1.0e-30)
    )


def compare(
    coarse_name: str,
    fine_name: str,
    coarse_resolution: int,
    fine_resolution: int,
) -> dict[str, Any]:
    coarse = load_reference(coarse_name)
    fine = load_reference(fine_name)
    axis = coarse["axis"]
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.hypot(yy, zz)
    masks = {
        "core": (
            (np.abs(xx) < np.pi - 0.72)
            & (radius < 1.75)
            & (coarse["rotor_mask"] < 0.08)
        ),
        "wake": (xx > 0.5) & (xx < np.pi - 0.72) & (radius < 1.5),
        "far_wake": (xx > 1.0) & (xx < np.pi - 0.72) & (radius < 1.5),
    }
    fine_on_coarse = interpolate_fine(fine, axis)
    regions = {}
    for name, mask in masks.items():
        regions[name] = {
            "points": int(np.count_nonzero(mask)),
            "full_velocity_relative_l2": relative_l2(
                coarse["velocity"],
                fine_on_coarse,
                mask,
                False,
            ),
            "disturbance_relative_l2": relative_l2(
                coarse["velocity"],
                fine_on_coarse,
                mask,
                True,
            ),
        }
    return {
        "coarse_resolution": coarse_resolution,
        "fine_resolution": fine_resolution,
        "regions": regions,
    }


def main() -> None:
    args = parse_args()
    comparisons = [
        compare(
            "gt_rotating_stable_re500_nominal",
            "gt_rotating_stable_re500_nominal_n80",
            64,
            80,
        ),
        compare(
            "gt_rotating_stable_re500_nominal_n80",
            "gt_rotating_stable_re500_nominal_n96",
            80,
            96,
        ),
    ]
    latest_core = comparisons[-1]["regions"]["core"][
        "full_velocity_relative_l2"
    ]
    payload = {
        "status": "completed",
        "quantity": "equal-phase, phase-averaged velocity",
        "interpolation": "trilinear fine-to-coarse",
        "comparisons": comparisons,
        "grid_independent_at_five_percent": bool(latest_core < 0.05),
        "interpretation": (
            "temporally phase-periodic but not grid-independent"
            if latest_core >= 0.05
            else "temporally and spatially converged at the tested tolerance"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
