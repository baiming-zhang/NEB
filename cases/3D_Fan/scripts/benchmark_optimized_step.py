"""Time full optimized training steps without writing a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.fan3d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
)
from eigenfluid.gen.fan3d.optimized_backend import (  # noqa: E402
    OptimizedFanModel,
    continuous_parameters,
    fluid_points,
    initialize_arrays,
    train_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = cfg_namespace(config)
    started = time.perf_counter()
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    cp.cuda.Stream.null.synchronize()
    print(
        json.dumps({"stage": "model_ready", "seconds": time.perf_counter() - started}),
        flush=True,
    )
    records: list[dict[str, float | int]] = []
    for index in range(args.steps):
        epoch = index + 1
        parameters = continuous_parameters(cfg, epoch)
        points = fluid_points(
            cfg, cfg.points_per_step, parameters, cfg.seed + 100_000 + epoch
        )
        step_started = time.perf_counter()
        result = train_step(cfg, model, points, parameters, cfg.learning_rate)
        cp.cuda.Stream.null.synchronize()
        free, total = cp.cuda.runtime.memGetInfo()
        record = {
            "stage": "step",
            "epoch": epoch,
            "seconds": time.perf_counter() - step_started,
            "loss": float(result["loss"]),
            "gpu_pool_gib": cp.get_default_memory_pool().used_bytes() / 2**30,
            "gpu_device_used_gib": (total - free) / 2**30,
            "gpu_free_gib": free / 2**30,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    print(json.dumps({"stage": "complete", "steps": records}), flush=True)


if __name__ == "__main__":
    main()
