"""Profile the three synchronized phases of one optimized training step."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.fan3d.hardcoded_backend import cfg_namespace  # noqa: E402
from eigenfluid.gen.fan3d.optimized_backend import (  # noqa: E402
    F32,
    OptimizedFanModel,
    _envelope_product_reverse,
    _field_adjoint_to_potential_jet,
    continuous_parameters,
    fluid_points,
    initialize_arrays,
    peel_and_adjoint,
    raw_fields,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--forward-batch", type=int)
    parser.add_argument("--backward-batch", type=int)
    parser.add_argument("--sss-block", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.forward_batch is not None:
        config["mode_forward_batch"] = args.forward_batch
    if args.backward_batch is not None:
        config["mode_backward_batch"] = args.backward_batch
    if args.sss_block is not None:
        config["sss_block_size"] = args.sss_block
    cfg = cfg_namespace(config)
    model = OptimizedFanModel(cfg, initialize_arrays(cfg))
    parameters = continuous_parameters(cfg, 1)
    points = fluid_points(cfg, cfg.points_per_step, parameters, cfg.seed + 100_001)

    started = time.perf_counter()
    raw = raw_fields(cfg, model, points, parameters)
    cp.cuda.Stream.null.synchronize()
    raw_seconds = time.perf_counter() - started
    print(json.dumps({"phase": "raw_fields", "seconds": raw_seconds}), flush=True)

    started = time.perf_counter()
    peeled = peel_and_adjoint(
        cfg, raw, len(points), need_adjoint=True, diagnostics=False
    )
    cp.cuda.Stream.null.synchronize()
    sss_seconds = time.perf_counter() - started
    print(json.dumps({"phase": "sss", "seconds": sss_seconds}), flush=True)

    input_jet = raw["input_jet"]
    envelope = raw["envelope_jet"]
    del raw["potential"], raw["velocity"], raw["gradient"]
    peeled.pop("q_velocity", None)
    block = max(1, int(cfg.mode_backward_batch))
    valid = cp.asarray(True)
    started = time.perf_counter()
    for start in range(0, cfg.modes, block):
        stop = min(start + block, cfg.modes)
        network_jet, cache = model.forward_jet_block(
            input_jet, start, stop, cache=True
        )
        assert cache is not None
        velocity_adjoint = peeled["adjoint_velocity"][:, start:stop].T.reshape(
            stop - start, 3, len(points)
        ).transpose(0, 2, 1)
        gradient_adjoint = peeled["adjoint_gradient"][:, start:stop].T.reshape(
            stop - start, 3, 3, len(points)
        ).transpose(0, 3, 1, 2)
        potential_adjoint = _field_adjoint_to_potential_jet(
            velocity_adjoint.astype(F32), gradient_adjoint.astype(F32)
        )
        network_adjoint = _envelope_product_reverse(
            network_jet, envelope, potential_adjoint
        )
        gradients = model.backward_jet_block(start, stop, cache, network_adjoint)
        valid &= model.adam_step_block(start, stop, gradients, cfg.learning_rate)
    if not bool(valid.get()):
        raise RuntimeError("invalid gradients")
    cp.cuda.Stream.null.synchronize()
    backward_seconds = time.perf_counter() - started
    free, total = cp.cuda.runtime.memGetInfo()
    print(
        json.dumps(
            {
                "phase": "backward_adam",
                "seconds": backward_seconds,
                "total_seconds": raw_seconds + sss_seconds + backward_seconds,
                "loss": peeled["loss"],
                "gpu_device_used_gib": (total - free) / 2**30,
                "gpu_free_gib": free / 2**30,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
