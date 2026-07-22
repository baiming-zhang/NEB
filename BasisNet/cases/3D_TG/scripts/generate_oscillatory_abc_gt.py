"""Generate an exact, periodically forced, time-dependent k=1 ABC flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def abc_parameters(time_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothly oscillating ABC coefficients and spatial phases."""
    coefficients = np.asarray(
        (
            1.0 + 0.38 * np.sin(time_value),
            1.0 + 0.32 * np.sin(time_value + 2.0 * np.pi / 3.0),
            1.0 + 0.28 * np.sin(time_value + 4.0 * np.pi / 3.0),
        ),
        dtype=np.float64,
    )
    phases = np.asarray(
        (
            0.60 * np.sin(time_value),
            0.48 * np.sin(time_value + 2.0 * np.pi / 3.0),
            0.40 * np.sin(time_value + 4.0 * np.pi / 3.0),
        ),
        dtype=np.float64,
    )
    return coefficients, phases


def oscillatory_abc_velocity(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    time_value: float,
) -> np.ndarray:
    """Evaluate a phase-shifted k=1 ABC member with curl(u)=u."""
    (a, b, c), (phase_x, phase_y, phase_z) = abc_parameters(time_value)
    return np.stack(
        (
            a * np.sin(z + phase_z) + c * np.cos(y + phase_y),
            b * np.sin(x + phase_x) + a * np.cos(z + phase_z),
            c * np.sin(y + phase_y) + b * np.cos(x + phase_x),
        ),
        axis=-1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    n = args.resolution
    coordinates = np.linspace(-np.pi, np.pi, n, endpoint=False)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    times = np.linspace(0.0, 2.0 * np.pi, args.frames, endpoint=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, float | int]] = []

    with h5py.File(args.output, "w") as handle:
        coordinate_group = handle.create_group("coordinates")
        for name in ("x", "y", "z"):
            coordinate_group.create_dataset(name, data=coordinates)
        handle.create_dataset("time", data=times)
        fields = handle.create_group("fields")
        chunks = (1, min(n, 32), min(n, 32), min(n, 32), 3)
        velocity_data = fields.create_dataset(
            "velocity",
            shape=(args.frames, n, n, n, 3),
            dtype="f4",
            chunks=chunks,
            compression="lzf",
        )
        vorticity_data = fields.create_dataset(
            "vorticity",
            shape=(args.frames, n, n, n, 3),
            dtype="f4",
            chunks=chunks,
            compression="lzf",
        )
        coefficient_data = fields.create_dataset(
            "abc_coefficients", shape=(args.frames, 3), dtype="f8"
        )
        phase_data = fields.create_dataset(
            "phase_shifts", shape=(args.frames, 3), dtype="f8"
        )

        for index, time_value in enumerate(times):
            velocity = oscillatory_abc_velocity(x, y, z, float(time_value))
            coefficients, phases = abc_parameters(float(time_value))
            velocity_data[index] = velocity
            # The phase-shifted ABC family remains Beltrami: curl(u)=u.
            vorticity_data[index] = velocity
            coefficient_data[index] = coefficients
            phase_data[index] = phases
            speed = np.linalg.norm(velocity.astype(np.float64), axis=-1)
            diagnostics.append(
                {
                    "frame": index,
                    "time": float(time_value),
                    "speed_max": float(speed.max()),
                    "kinetic_energy": float(0.5 * np.mean(speed**2)),
                    "divergence_rms": 0.0,
                }
            )

        handle.attrs["reynolds_number"] = args.reynolds
        handle.attrs["boundary"] = "triply periodic closed domain"
        handle.attrs["initial_condition"] = "oscillatory phase-shifted ABC k=1 flow"
        handle.attrs["initial_condition_independent_of_basis"] = True
        handle.attrs["solver"] = "exact manufactured periodically forced NS solution"
        handle.attrs["forcing"] = "f=du/dt+nu*u; nonlinear term absorbed by pressure"

    summary = {
        "truth_type": "exact periodically forced oscillatory ABC flow",
        "classic_family": "Arnold-Beltrami-Childress",
        "large_scale_only": True,
        "active_wave_number": 1,
        "resolution": [n, n, n],
        "frames": args.frames,
        "reynolds": args.reynolds,
        "time_interval": [0.0, float(2.0 * np.pi)],
        "independent_of_basis": True,
        "output": str(args.output),
        "diagnostics": diagnostics,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
