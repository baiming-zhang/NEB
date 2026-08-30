"""Compare the original PyTorch epoch-100 run with the hardcoded benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def log_times(path: Path) -> tuple[datetime, datetime]:
    pattern = re.compile(r"^\[(.*?)\]")
    times = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            times.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S"))
    if not times:
        raise RuntimeError(f"no timestamps in {path}")
    return times[0], times[-1]


def relative(new: float, old: float) -> float:
    return abs(new - old) / max(abs(old), 1.0e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    args = parser.parse_args()

    old_history = json.loads(
        (args.old_run / "data" / "training_history.json").read_text(encoding="utf-8")
    )
    new_history = json.loads(
        (args.new_run / "data" / "hardcoded_training_history.json").read_text(
            encoding="utf-8"
        )
    )
    old_by_epoch = {int(row["epoch"]): row for row in old_history}
    new_by_epoch = {int(row["epoch"]): row for row in new_history}
    common_epochs = sorted(set(old_by_epoch) & set(new_by_epoch))
    if common_epochs != list(range(1, 101)):
        raise RuntimeError(f"expected epochs 1..100, got {common_epochs[:3]}..{common_epochs[-3:]}")

    train_differences = np.asarray(
        [
            new_by_epoch[epoch]["train_loss"] - old_by_epoch[epoch]["train_loss"]
            for epoch in common_epochs
        ],
        dtype=np.float64,
    )
    train_relative = np.asarray(
        [
            relative(
                new_by_epoch[epoch]["train_loss"],
                old_by_epoch[epoch]["train_loss"],
            )
            for epoch in common_epochs
        ],
        dtype=np.float64,
    )
    parameter_pairs = [
        (
            np.asarray(new_by_epoch[epoch][key]),
            np.asarray(old_by_epoch[epoch][key]),
        )
        for epoch in common_epochs
        for key in ("beta", "alpha_deg")
    ]

    rows: list[dict[str, Any]] = []
    for epoch in (1, 50, 100):
        old = old_by_epoch[epoch]
        new = new_by_epoch[epoch]
        rows.append(
            {
                "epoch": epoch,
                "old_train_loss": old["train_loss"],
                "new_train_loss": new["train_loss"],
                "train_abs_difference": abs(new["train_loss"] - old["train_loss"]),
                "train_relative_difference": relative(
                    new["train_loss"], old["train_loss"]
                ),
                "old_validation_loss": old["validation_loss"],
                "new_validation_loss": new["validation_loss"],
                "validation_abs_difference": abs(
                    new["validation_loss"] - old["validation_loss"]
                ),
                "validation_relative_difference": relative(
                    new["validation_loss"], old["validation_loss"]
                ),
                "old_rank": old["validation_min_rank"],
                "new_rank": new["validation_min_rank"],
                "old_orth_error": old["validation_orth_error"],
                "new_orth_error": new["validation_orth_error"],
                "old_elapsed_before_validation_seconds": old["seconds"],
                "new_elapsed_before_validation_seconds": new["seconds"],
            }
        )

    old_payload = torch.load(
        args.old_run / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(old_payload["epoch"]) != 100:
        raise RuntimeError("old checkpoint is not epoch 100")
    with np.load(
        args.new_run / "checkpoints" / "hardcoded_latest.npz",
        allow_pickle=False,
    ) as archive:
        new_arrays = {name: archive[name] for name in archive.files}

    squared_difference = 0.0
    squared_reference = 0.0
    maximum_absolute = 0.0
    element_count = 0
    cfg = old_payload["config"]
    for mode in range(int(cfg["modes"])):
        for layer in range(int(cfg["hidden_layers"])):
            for kind in ("weight", "bias"):
                old_name = f"modes.{mode}.layers.{layer}.{kind}"
                new_name = f"m{mode:03d}.l{layer}.{kind}"
                old_value = old_payload["model"][old_name].numpy().astype(np.float64)
                new_value = new_arrays[new_name].astype(np.float64)
                difference = new_value - old_value
                squared_difference += float(np.sum(difference * difference))
                squared_reference += float(np.sum(old_value * old_value))
                maximum_absolute = max(
                    maximum_absolute, float(np.max(np.abs(difference)))
                )
                element_count += difference.size
        for kind in ("weight", "bias"):
            old_name = f"modes.{mode}.output.{kind}"
            new_name = f"m{mode:03d}.output.{kind}"
            old_value = old_payload["model"][old_name].numpy().astype(np.float64)
            new_value = new_arrays[new_name].astype(np.float64)
            difference = new_value - old_value
            squared_difference += float(np.sum(difference * difference))
            squared_reference += float(np.sum(old_value * old_value))
            maximum_absolute = max(maximum_absolute, float(np.max(np.abs(difference))))
            element_count += difference.size

    old_start, old_last_log = log_times(args.old_run / "train.log")
    new_start, new_complete = log_times(args.new_run / "hardcoded_train.log")
    old_checkpoint_time = datetime.fromtimestamp(
        (args.old_run / "checkpoints" / "latest.pt").stat().st_mtime
    )
    old_wall = (old_checkpoint_time - old_start).total_seconds()
    new_wall = (new_complete - new_start).total_seconds()
    report = {
        "comparison": "same initialization, beta/alpha sequence, training points, validation points",
        "epochs_compared": len(common_epochs),
        "parameter_samples_identical": all(
            np.array_equal(new, old) for new, old in parameter_pairs
        ),
        "train_loss_all_epochs": {
            "mean_absolute_difference": float(np.mean(np.abs(train_differences))),
            "rmse": float(math.sqrt(np.mean(train_differences**2))),
            "maximum_absolute_difference": float(np.max(np.abs(train_differences))),
            "maximum_relative_difference": float(np.max(train_relative)),
            "maximum_absolute_difference_epoch": int(
                np.argmax(np.abs(train_differences)) + 1
            ),
        },
        "selected_epochs": rows,
        "final_parameter_difference": {
            "elements": element_count,
            "relative_l2": math.sqrt(squared_difference / squared_reference),
            "rmse": math.sqrt(squared_difference / element_count),
            "maximum_absolute": maximum_absolute,
        },
        "timing": {
            "old_start": old_start.isoformat(sep=" "),
            "old_last_log": old_last_log.isoformat(sep=" "),
            "old_epoch100_checkpoint": old_checkpoint_time.isoformat(sep=" "),
            "old_wall_to_epoch100_checkpoint_seconds": old_wall,
            "new_start": new_start.isoformat(sep=" "),
            "new_complete": new_complete.isoformat(sep=" "),
            "new_wall_to_complete_seconds": new_wall,
            "speedup": old_wall / new_wall,
        },
        "old_checkpoint_epoch": int(old_payload["epoch"]),
        "new_checkpoint_epoch": 100,
        "new_stderr_bytes": (
            args.new_run / "launcher.stderr.log"
        ).stat().st_size,
    }
    target = args.new_run / "benchmark_comparison.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
