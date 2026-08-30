"""Solve a phase-periodic four-blade fan reference flow.

The numerical reference is independent of BasisNet.  It advances the
incompressible Navier--Stokes equations in the inertial frame with a genuinely
moving Brinkman rotor.  Four identical blades make one phase period equal to a
quarter revolution.  The run stops only after successive equal-phase fields
meet the configured L2 tolerance, or after the declared maximum cycle count.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np

from generate_rotating_gt import RotatingFanReference, load_config


CASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    CASE_ROOT / "configs" / "gt_rotating_stable_re500_nominal.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def core_mask(solver: RotatingFanReference, chi: np.ndarray) -> np.ndarray:
    cfg = solver.cfg
    xx = solver.axis[:, None, None]
    yy = solver.axis[None, :, None]
    zz = solver.axis[None, None, :]
    radius = np.sqrt(yy * yy + zz * zz)
    fringe_width = float(cfg["fringe_width"])
    return (
        (np.abs(xx) < float(cfg["xmax"]) - fringe_width)
        & (radius < float(cfg["farfield_radius"]) - 0.35 * fringe_width)
        & (chi < 0.08)
    )


def relative_l2(
    current: np.ndarray,
    previous: np.ndarray,
    mask: np.ndarray,
) -> float:
    difference = current[:, mask] - previous[:, mask]
    denominator = current[:, mask]
    return float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(denominator.ravel()), 1.0e-30)
    )


def kinetic_diagnostics(
    velocity: np.ndarray,
    vorticity: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    return {
        "core_velocity_rms": float(
            np.sqrt(np.mean(np.sum(velocity[:, mask] ** 2, axis=0)))
        ),
        "core_vorticity_rms": float(
            np.sqrt(np.mean(np.sum(vorticity[:, mask] ** 2, axis=0)))
        ),
        "core_kinetic_energy_density": float(
            0.5 * np.mean(np.sum(velocity[:, mask] ** 2, axis=0))
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if int(cfg["blade_count"]) != 4:
        raise ValueError("phase-periodic solve requires the configured four-blade rotor")
    output_dir = CASE_ROOT.parents[1] / str(cfg["result_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", cfg)

    solver = RotatingFanReference(cfg)
    phase_period = 2.0 * math.pi / (
        int(cfg["blade_count"]) * float(cfg["rotation_rate"])
    )
    minimum_cycles = int(cfg["minimum_cycles"])
    maximum_cycles = int(cfg["maximum_cycles"])
    tolerance = float(cfg["periodic_tolerance"])
    required = int(cfg["consecutive_converged_cycles"])
    average_count = int(cfg["phase_average_cycles"])

    history: list[dict[str, Any]] = []
    recent_velocity: deque[np.ndarray] = deque(maxlen=average_count)
    recent_vorticity: deque[np.ndarray] = deque(maxlen=average_count)
    previous: np.ndarray | None = None
    converged_streak = 0
    total_steps = 0
    maximum_divergence = 0.0
    start_time = time.perf_counter()

    for cycle in range(1, maximum_cycles + 1):
        total_steps += solver.advance_to(cycle * phase_period)
        velocity, vorticity, divergence = solver.snapshot()
        chi = solver.last_chi.detach().cpu().numpy().astype(np.float32)
        mask = core_mask(solver, chi)
        residual = (
            None if previous is None else relative_l2(velocity, previous, mask)
        )
        maximum_divergence = max(maximum_divergence, divergence)
        row: dict[str, Any] = {
            "cycle": cycle,
            "time": float(solver.time),
            "phase_radians_modulo_quarter_turn": float(
                math.remainder(
                    solver.omega * solver.time,
                    0.5 * math.pi,
                )
            ),
            "steps_total": total_steps,
            "periodic_relative_l2": residual,
            "max_divergence": divergence,
            **kinetic_diagnostics(velocity, vorticity, mask),
        }
        history.append(row)
        recent_velocity.append(velocity)
        recent_vorticity.append(vorticity)
        atomic_json(output_dir / "convergence_history.json", {"cycles": history})
        residual_text = "n/a" if residual is None else f"{residual:.6e}"
        print(
            f"[cycle {cycle:02d}/{maximum_cycles:02d}] "
            f"t={solver.time:.6f} residual={residual_text} "
            f"div={divergence:.3e}",
            flush=True,
        )
        if cycle >= minimum_cycles and residual is not None and residual < tolerance:
            converged_streak += 1
        else:
            converged_streak = 0
        previous = velocity
        if converged_streak >= required and len(recent_velocity) >= average_count:
            break

    phase_average_velocity = np.mean(
        np.stack(tuple(recent_velocity), axis=0),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    phase_average_vorticity = np.mean(
        np.stack(tuple(recent_vorticity), axis=0),
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)
    velocity_fluctuation = np.stack(tuple(recent_velocity), axis=0) - (
        phase_average_velocity[None]
    )
    fluctuation_ratio = float(
        np.linalg.norm(velocity_fluctuation[:, :, mask].ravel())
        / max(
            math.sqrt(len(recent_velocity))
            * np.linalg.norm(phase_average_velocity[:, mask].ravel()),
            1.0e-30,
        )
    )
    final_residual = history[-1]["periodic_relative_l2"]
    converged = bool(
        converged_streak >= required
        and final_residual is not None
        and final_residual < tolerance
    )

    with h5py.File(output_dir / "stable_fan_reference.h5", "w") as archive:
        archive.create_dataset("axis", data=solver.axis)
        archive.create_dataset(
            "velocity",
            data=phase_average_velocity,
            compression="gzip",
            compression_opts=4,
        )
        archive.create_dataset(
            "vorticity",
            data=phase_average_vorticity,
            compression="gzip",
            compression_opts=4,
        )
        archive.create_dataset("rotor_mask", data=chi, compression="gzip")
        archive.attrs["frame"] = "inertial"
        archive.attrs["phase"] = "theta modulo pi/2 = 0"
        archive.attrs["phase_average_cycles"] = len(recent_velocity)

    protocol = {
        "status": (
            "phase-periodic converged"
            if converged
            else "maximum cycles reached; final equal-phase average retained"
        ),
        "independence": "numerical GT does not use the trained PINN basis",
        "equations": "3D incompressible Navier-Stokes, inertial frame",
        "spatial_solver": "2/3-dealiased GPU pseudo-spectral projection",
        "time_integrator": "Heun RK2 plus exact exponential Brinkman split",
        "rotor_boundary": "moving no-slip u = Omega cross r",
        "phase_period_seconds": phase_period,
        "cycles_completed": len(history),
        "periodic_tolerance": tolerance,
        "final_periodic_relative_l2": final_residual,
        "converged": converged,
        "phase_average_cycles": len(recent_velocity),
        "phase_average_fluctuation_ratio": fluctuation_ratio,
        "maximum_divergence": maximum_divergence,
        "elapsed_seconds": time.perf_counter() - start_time,
        "total_steps": total_steps,
        "boundary_diagnostics": solver.boundary_diagnostics(
            phase_average_velocity
        ),
        "geometry": {
            "blade_count": int(cfg["blade_count"]),
            "beta_tip_deg": float(cfg["beta_tip_deg"]),
            "chord_tip": float(cfg["chord_tip"]),
            "thickness_ratio": float(cfg["thickness_ratio"]),
        },
    }
    atomic_json(output_dir / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
