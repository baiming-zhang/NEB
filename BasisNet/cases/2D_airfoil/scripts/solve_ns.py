from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.ns_reference import (  # noqa: E402
    LBMSolverConfig,
    solve_reference,
)
from eigenfluid_sss.basisnet.airfoil2d.training import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent 2D airfoil D2Q9 Navier-Stokes reference"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--solver-config", type=Path)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--alpha-deg", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    parser.add_argument("--reynolds", type=float)
    parser.add_argument("--lattice-inlet-speed", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--snapshot-every", type=int)
    parser.add_argument("--collision-model", choices=("trt", "bgk"))
    parser.add_argument("--initial-perturbation-fraction", type=float)
    parser.add_argument(
        "--warmup-inlet-perturbation-fraction", type=float
    )
    parser.add_argument(
        "--warmup-inlet-perturbation-duration", type=float
    )
    parser.add_argument(
        "--warmup-inlet-perturbation-period-chords", type=float
    )
    parser.add_argument(
        "--allow-unseparated-smoke",
        action="store_true",
        help="disable separated-GT quality rejection for plumbing smoke tests",
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    geometry_values = json.loads(args.config.read_text(encoding="utf-8"))
    geometry_fields = {field.name for field in fields(Config)}
    geometry = Config(
        **{
            name: value
            for name, value in geometry_values.items()
            if name in geometry_fields
        }
    )
    solver_values = asdict(LBMSolverConfig())
    if args.solver_config is not None:
        loaded = json.loads(args.solver_config.read_text(encoding="utf-8"))
        unknown = sorted(set(loaded) - set(solver_values))
        if unknown:
            raise ValueError(
                f"unknown solver config fields: {', '.join(unknown)}"
            )
        solver_values.update(loaded)
    for field in (
        "nx",
        "ny",
        "reynolds",
        "lattice_inlet_speed",
        "steps",
        "warmup_steps",
        "snapshot_every",
        "collision_model",
        "initial_perturbation_fraction",
        "warmup_inlet_perturbation_fraction",
        "warmup_inlet_perturbation_duration",
        "warmup_inlet_perturbation_period_chords",
        "device",
    ):
        value = getattr(args, field)
        if value is not None:
            solver_values[field] = value
    if args.allow_unseparated_smoke:
        solver_values["require_separated_gt"] = False
    solver = LBMSolverConfig(**solver_values)
    result = solve_reference(
        geometry, solver, args.beta, args.alpha_deg, args.output
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
