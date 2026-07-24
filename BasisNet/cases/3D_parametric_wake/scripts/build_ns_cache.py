"""Generate an independent driven-wake Navier--Stokes reference cache."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_wake3d.ns_reference import (  # noqa: E402
    generate_reference,
    save_reference,
)
from eigenfluid_sss.basisnet.parametric_wake3d.training import Config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--final-time", type=float, default=6.0)
    parser.add_argument("--dt-max", type=float, default=0.01)
    parser.add_argument("--snapshots", type=int, default=31)
    parser.add_argument("--reynolds", type=float, default=500.0)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(Config)}
    cfg = Config(**{key: value for key, value in raw.items() if key in allowed})
    reference = generate_reference(
        cfg,
        alpha=args.alpha,
        resolution=args.resolution,
        final_time=args.final_time,
        dt_max=args.dt_max,
        snapshots=args.snapshots,
        reynolds=args.reynolds,
    )
    save_reference(args.output, reference)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "alpha": args.alpha,
                "resolution": args.resolution,
                "snapshots": args.snapshots,
                "reynolds": args.reynolds,
                "maximum_reference_divergence": float(
                    reference["divergence_max"].max()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
