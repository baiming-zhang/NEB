"""Expand a trained hardcoded checkpoint while preserving its existing modes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def initialize_mode(
    rng: np.random.Generator,
    mode: int,
    input_features: int,
    hidden: int,
    hidden_layers: int,
) -> dict[str, np.ndarray]:
    prefix = f"m{mode:03d}"
    arrays: dict[str, np.ndarray] = {}
    parameter_names: list[str] = []
    fan_in = input_features
    for layer in range(hidden_layers):
        bound = 1.0 / math.sqrt(fan_in)
        weight_name = f"{prefix}.l{layer}.weight"
        bias_name = f"{prefix}.l{layer}.bias"
        arrays[weight_name] = rng.uniform(
            -bound, bound, (hidden, fan_in)
        ).astype(np.float32)
        arrays[bias_name] = rng.uniform(
            -bound, bound, hidden
        ).astype(np.float32)
        parameter_names.extend((weight_name, bias_name))
        fan_in = hidden
    bound = 1.0 / math.sqrt(fan_in)
    weight_name = f"{prefix}.output.weight"
    bias_name = f"{prefix}.output.bias"
    arrays[weight_name] = rng.uniform(
        -bound, bound, (1, fan_in)
    ).astype(np.float32)
    arrays[bias_name] = rng.uniform(-bound, bound, 1).astype(np.float32)
    parameter_names.extend((weight_name, bias_name))
    for index, name in enumerate(parameter_names):
        stem = f"{prefix}.adam.p{index:02d}"
        arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
        arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
        arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, required=True)
    args = parser.parse_args()

    metadata = json.loads(
        args.source.with_suffix(".json").read_text(encoding="utf-8")
    )
    config = dict(metadata["config"])
    source_modes = int(config["modes"])
    if args.modes <= source_modes:
        raise ValueError("expanded mode count must exceed the source mode count")
    with np.load(args.source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    rng = np.random.default_rng(int(config["seed"]))
    input_features = 4 * int(config["frequencies"]) + 4
    for mode in range(args.modes):
        initialized = initialize_mode(
            rng,
            mode,
            input_features,
            int(config["hidden"]),
            int(config["hidden_layers"]),
        )
        if mode >= source_modes:
            arrays.update(initialized)

    config["modes"] = int(args.modes)
    config["freeze_prefix_modes"] = source_modes
    config["orthonormal_peeling_fast_path"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, args.output)
    expanded_metadata = dict(metadata)
    expanded_metadata["config"] = config
    expanded_metadata["expansion"] = {
        "source_checkpoint": str(args.source.resolve()),
        "preserved_modes": source_modes,
        "new_modes": int(args.modes) - source_modes,
        "total_modes": int(args.modes),
        "new_mode_initializer": "numpy_pcg64_v1_continued_mode_sequence",
        "existing_modes_and_adam_state_preserved": True,
    }
    temporary_json = args.output.with_suffix(".json.tmp")
    temporary_json.write_text(
        json.dumps(expanded_metadata, indent=2), encoding="utf-8"
    )
    os.replace(temporary_json, args.output.with_suffix(".json"))
    print(json.dumps(expanded_metadata["expansion"], indent=2))


if __name__ == "__main__":
    main()
