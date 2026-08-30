"""Audit strict one-pass K=1024 velocity-basis bandwidth margins."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from metaball_runtime import backend, install_backend, metaball_fluid_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--omegas", type=float, nargs="+", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--validation-pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    install_backend(config)
    with np.load(args.initial, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    # The audit must measure candidates below the production gate rather than
    # aborting at the first one.  Production thresholds are preserved in the
    # output and restored for every formal run.
    production_relative_floor = float(config["peeling_relative_mass_floor"])
    production_overlap_tolerance = float(config["peeling_overlap_tolerance"])
    audit_config = dict(config)
    audit_config["peeling_relative_mass_floor"] = 1.0e-12
    audit_config["peeling_overlap_tolerance"] = 0.999
    cfg = backend.cfg_namespace(audit_config)
    model = backend.HardcodedModel(cfg, arrays)
    rows: list[dict] = []
    started = time.perf_counter()
    for omega in args.omegas:
        cfg.siren_first_omega0 = float(omega)
        for token in args.tokens:
            points = metaball_fluid_points(
                cfg,
                int(cfg.points_per_step),
                float(token),
                0.0,
                int(cfg.seed) + int(token) * 1009,
            )
            item_started = time.perf_counter()
            diagnostic = backend.peel_step(
                cfg, model, points, float(token), 0.0, training=False
            )
            row = {
                "first_omega": float(omega),
                "geometry_token": int(token),
                "velocity_rayleigh": diagnostic["velocity_rayleigh"],
                "min_relative_residual_mass": diagnostic[
                    "min_relative_residual_mass"
                ],
                "max_post_peel_overlap": diagnostic["max_post_peel_overlap"],
                "passes": diagnostic["peeling_passes_used"],
                "accepted_under_production_rule": bool(
                    diagnostic["min_relative_residual_mass"]
                    >= production_relative_floor
                    and diagnostic["max_post_peel_overlap"]
                    <= production_overlap_tolerance
                ),
                "seconds": time.perf_counter() - item_started,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.validation_pack is not None:
            with np.load(args.validation_pack, allow_pickle=False) as validation:
                validation_points = validation["validation_points"]
                validation_tokens = validation["validation_beta"]
            for index, (points, token_value) in enumerate(
                zip(validation_points, validation_tokens)
            ):
                token = int(round(float(token_value)))
                item_started = time.perf_counter()
                diagnostic = backend.peel_step(
                    cfg, model, points, float(token), 0.0, training=False
                )
                row = {
                    "first_omega": float(omega),
                    "geometry_token": token,
                    "sample_source": f"fixed_validation_{index}",
                    "velocity_rayleigh": diagnostic["velocity_rayleigh"],
                    "min_relative_residual_mass": diagnostic[
                        "min_relative_residual_mass"
                    ],
                    "max_post_peel_overlap": diagnostic[
                        "max_post_peel_overlap"
                    ],
                    "passes": diagnostic["peeling_passes_used"],
                    "accepted_under_production_rule": bool(
                        diagnostic["min_relative_residual_mass"]
                        >= production_relative_floor
                        and diagnostic["max_post_peel_overlap"]
                        <= production_overlap_tolerance
                    ),
                    "seconds": time.perf_counter() - item_started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    summary = {
        "config": str(args.config.resolve()),
        "initial": str(args.initial.resolve()),
        "modes": int(cfg.modes),
        "points": int(cfg.points_per_step),
        "production_relative_mass_floor": production_relative_floor,
        "production_overlap_tolerance": production_overlap_tolerance,
        "rows": rows,
        "total_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
