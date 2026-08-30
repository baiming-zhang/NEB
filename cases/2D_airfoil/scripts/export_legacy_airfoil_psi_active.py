#!/usr/bin/env python3
"""Export value-only streamfunction columns from a legacy airfoil checkpoint.

The exporter deliberately runs the checkpoint with its native historical
network backend.  It writes only scalar ``A=psi`` values in the active-node
order of the current masked Cartesian complex.  All derivatives, Ritz steps,
and ROM dynamics remain in the current shared mimetic implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cupy as cp
import numpy as np


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-src", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--alpha-deg", type=float, required=True)
    parser.add_argument("--point-batch", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_src = args.legacy_src.resolve()
    sys.path.insert(0, str(legacy_src))
    from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (  # noqa: PLC0415
        load_hardcoded_model,
    )

    started = time.perf_counter()
    x = np.load(args.cache_dir / "x.npy").astype(np.float64)
    y = np.load(args.cache_dir / "y.npy").astype(np.float64)
    solid = np.load(args.cache_dir / "solid.npy").astype(bool)
    unknown = (~solid).copy()
    unknown[:, 0] = False
    unknown[0, :] = False
    unknown[-1, :] = False
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[unknown], ygrid[unknown])).astype(np.float64)

    load_started = time.perf_counter()
    cfg, model, metadata = load_hardcoded_model(args.checkpoint.resolve())
    load_seconds = time.perf_counter() - load_started
    modes = int(cfg.modes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f"{args.output.stem}.incomplete{args.output.suffix}"
    )
    output = np.lib.format.open_memmap(
        temporary_output,
        mode="w+",
        dtype=np.float32,
        shape=(len(points), modes),
    )

    forward_started = time.perf_counter()
    for start in range(0, len(points), args.point_batch):
        stop = min(start + args.point_batch, len(points))
        block = np.empty((stop - start, modes), dtype=np.float32)
        count = 0
        for mode, (_network, field, _cache) in enumerate(
            model.forward_stream(
                cfg,
                points[start:stop],
                args.beta,
                args.alpha_deg,
                False,
            )
        ):
            block[:, mode] = cp.asnumpy(field.value[:, 0]).astype(
                np.float32, copy=False
            )
            count += 1
        if count != modes:
            raise RuntimeError(f"legacy forward yielded {count} modes, expected {modes}")
        if not np.isfinite(block).all():
            raise FloatingPointError(f"non-finite psi in point block {start}:{stop}")
        output[start:stop] = block
        output.flush()
        print(f"[legacy psi] active nodes {stop}/{len(points)}", flush=True)
    forward_seconds = time.perf_counter() - forward_started
    del output
    cp.get_default_memory_pool().free_all_blocks()
    temporary_output.replace(args.output)

    manifest = {
        "complete": True,
        "completed_at": now(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "checkpoint_epoch": int(metadata["epoch"]),
        "basis_objective": str(cfg.basis_objective),
        "modes": modes,
        "active_nodes": int(len(points)),
        "grid_shape": [int(len(y)), int(len(x))],
        "beta": float(args.beta),
        "alpha_deg": float(args.alpha_deg),
        "observable": "value-only A=psi; no spatial derivative exported",
        "active_node_order": (
            "row-major fluid nodes excluding west, ymin, ymax; east outlet active"
        ),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "timings": {
            "checkpoint_load_seconds": load_seconds,
            "value_only_forward_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    write_json(args.output.with_suffix(".json"), manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
