"""One-time PyTorch checkpoint migration to a neutral NPZ checkpoint.

The fixed-graph trainer does not import PyTorch.  This utility is only the
format bridge for existing checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = payload["model"]
    arrays: dict[str, np.ndarray] = {}
    for mode in range(int(cfg["modes"])):
        source_prefix = f"modes.{mode}"
        target_prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        for layer in range(int(cfg["hidden_layers"])):
            for kind in ("weight", "bias"):
                source = f"{source_prefix}.layers.{layer}.{kind}"
                target = f"{target_prefix}.l{layer}.{kind}"
                arrays[target] = model[source].detach().numpy()
                parameter_names.append(source)
        for kind in ("weight", "bias"):
            source = f"{source_prefix}.output.{kind}"
            target = f"{target_prefix}.output.{kind}"
            arrays[target] = model[source].detach().numpy()
            parameter_names.append(source)

        optimizer = payload["optimizers"][mode]
        parameter_ids = optimizer["param_groups"][0]["params"]
        if len(parameter_ids) != len(parameter_names):
            raise RuntimeError("optimizer parameter order does not match model")
        for index, parameter_id in enumerate(parameter_ids):
            state = optimizer["state"][parameter_id]
            stem = f"{target_prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = state["exp_avg"].detach().numpy()
            arrays[f"{stem}.exp_avg_sq"] = state["exp_avg_sq"].detach().numpy()
            step = state["step"]
            arrays[f"{stem}.step"] = np.asarray(
                int(step.item() if hasattr(step, "item") else step),
                dtype=np.int64,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(args.output)
    metadata = {
        "format": "eigenfluid_airfoil_hardcoded_v1",
        "source_checkpoint": str(args.input.resolve()),
        "epoch": int(payload["epoch"]),
        "best_validation": float(payload.get("best_validation", float("inf"))),
        "config": cfg,
        "training_uses_pytorch": False,
        "spatial_ad": "hardcoded_xy_jet2",
        "parameter_backward": "hardcoded_reverse_jet2",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "epoch": metadata["epoch"],
                "arrays": len(arrays),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
