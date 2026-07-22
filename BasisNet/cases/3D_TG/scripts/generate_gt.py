"""Generate an exact time-periodic forced Taylor--Green Navier--Stokes GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--reynolds", type=float, default=500.0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--frame-seconds", type=float, default=0.1)
    parser.add_argument("--period", type=float, default=3.0)
    args = parser.parse_args()

    n = args.resolution
    nu = 1.0 / args.reynolds
    omega_time = 2.0 * np.pi / args.period
    coordinates = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
    x, y, z = np.meshgrid(
        coordinates, coordinates, coordinates, indexing="ij"
    )
    del z

    base_u = np.sin(x) * np.cos(y)
    base_v = -np.cos(x) * np.sin(y)
    base_vorticity = 2.0 * np.sin(x) * np.sin(y)
    times = np.arange(args.frames, dtype=np.float64) * args.frame_seconds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, float]] = []
    with h5py.File(args.output, "w") as handle:
        coordinate_group = handle.create_group("coordinates")
        for name in ("x", "y", "z"):
            coordinate_group.create_dataset(name, data=coordinates)
        handle.create_dataset("time", data=times)
        field_group = handle.create_group("fields")
        velocity_data = field_group.create_dataset(
            "velocity",
            shape=(args.frames, n, n, n, 3),
            dtype="f4",
            chunks=(1, n, n, n, 3),
            compression="gzip",
            compression_opts=1,
        )
        vorticity_data = field_group.create_dataset(
            "vorticity",
            shape=(args.frames, n, n, n, 3),
            dtype="f4",
            chunks=(1, n, n, n, 3),
            compression="gzip",
            compression_opts=1,
        )
        amplitude_data = field_group.create_dataset(
            "amplitude", shape=(args.frames,), dtype="f8"
        )
        forcing_amplitude_data = field_group.create_dataset(
            "forcing_amplitude", shape=(args.frames,), dtype="f8"
        )

        velocity = np.zeros((n, n, n, 3), dtype=np.float32)
        vorticity = np.zeros_like(velocity)
        for frame, time_value in enumerate(times):
            amplitude = np.cos(omega_time * time_value)
            amplitude_rate = -omega_time * np.sin(omega_time * time_value)
            forcing_amplitude = amplitude_rate + 2.0 * nu * amplitude
            velocity[..., 0] = amplitude * base_u
            velocity[..., 1] = amplitude * base_v
            velocity[..., 2] = 0.0
            vorticity[..., 0] = 0.0
            vorticity[..., 1] = 0.0
            vorticity[..., 2] = amplitude * base_vorticity
            velocity_data[frame] = velocity
            vorticity_data[frame] = vorticity
            amplitude_data[frame] = amplitude
            forcing_amplitude_data[frame] = forcing_amplitude
            speed2 = np.sum(velocity.astype(np.float64) ** 2, axis=-1)
            omega2 = np.sum(vorticity.astype(np.float64) ** 2, axis=-1)
            diagnostics.append(
                {
                    "frame": frame,
                    "time": float(time_value),
                    "amplitude": float(amplitude),
                    "forcing_amplitude": float(forcing_amplitude),
                    "kinetic_energy": float(0.5 * np.mean(speed2)),
                    "enstrophy": float(0.5 * np.mean(omega2)),
                    "divergence_rms": 0.0,
                }
            )

        handle.attrs["truth_type"] = (
            "exact analytic time-periodic forced Taylor-Green flow"
        )
        handle.attrs["solver"] = "analytic forced incompressible Navier-Stokes"
        handle.attrs["boundary"] = "triply periodic closed domain"
        handle.attrs["reynolds_number"] = args.reynolds
        handle.attrs["viscosity"] = nu
        handle.attrs["period"] = args.period
        handle.attrs["initial_condition"] = (
            "u=(sin(x)cos(y),-cos(x)sin(y),0)"
        )
        handle.attrs["initial_condition_independent_of_basis"] = True
        handle.attrs["forcing"] = (
            "f=(a'(t)+2*nu*a(t))*(sin(x)cos(y),-cos(x)sin(y),0), "
            "a(t)=cos(2*pi*t/T)"
        )
        handle.attrs["mesh_used"] = False

    summary = {
        "truth_type": "exact analytic periodic forced Taylor-Green NS flow",
        "resolution": [n, n, n],
        "reynolds": args.reynolds,
        "viscosity": nu,
        "frames": args.frames,
        "frame_seconds": args.frame_seconds,
        "period": args.period,
        "duration": float(times[-1]),
        "boundary": "triply periodic closed domain",
        "spatial_mode": "u=(sin(x)cos(y),-cos(x)sin(y),0)",
        "time_amplitude": "a(t)=cos(2*pi*t/T)",
        "forcing_amplitude": "a'(t)+2*nu*a(t)",
        "exactly_divergence_free": True,
        "initial_condition_independent_of_basis": True,
        "diagnostics": diagnostics,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
