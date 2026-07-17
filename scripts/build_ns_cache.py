from __future__ import annotations

import argparse
import json
from pathlib import Path

from k96_pinn.ns_reference import build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase-frames", type=int, default=17)
    args = parser.parse_args()
    cache = build_cache(args.source, args.output_dir, args.phase_frames)
    print(
        json.dumps({"cache": str(cache), "phase_frames": args.phase_frames}, indent=2)
    )


if __name__ == "__main__":
    main()
