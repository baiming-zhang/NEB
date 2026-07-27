"""Canonical fixed-graph training entry point for the 2D airfoil case.

The public command mirrors ``2D_karman/scripts/train.py`` while delegating
the implementation to the explicit CuPy forward/reverse pass.  A new run
creates its deterministic epoch-0 state and immutable validation pack before
starting training.  Resumed runs reuse those artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
PREPARE = CASE_ROOT / "scripts" / "prepare_hardcoded_run.py"
TRAINER = CASE_ROOT / "scripts" / "train_hardcoded.py"


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    target_epochs = int(args.epochs or config["epochs"])
    inputs = args.run_dir / "reference_inputs"
    validation = inputs / "fixed_validation_scipy_sobol_v1.npz"
    resume = args.resume

    if resume is None:
        initial = inputs / "initial_epoch0000.npz"
        if not initial.exists() or not validation.exists():
            run(
                [
                    sys.executable,
                    str(PREPARE),
                    "--config",
                    str(args.config),
                    "--output-dir",
                    str(inputs),
                ]
            )
        resume = initial

    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--resume",
        str(resume),
        "--run-dir",
        str(args.run_dir),
        "--epochs",
        str(target_epochs),
        "--validation-pack",
        str(validation),
    ]
    if args.self_test:
        command.extend(
            [
                "--self-test-epochs",
                "1",
                "--self-test-points",
                "64",
            ]
        )
    run(command)


if __name__ == "__main__":
    main()
