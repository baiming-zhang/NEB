"""Run the GitHub production Jet3 trainer with square-domain case hooks."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from square_runtime import (
    AIRFOIL_ROOT,
    install_backend,
    square_continuous_parameters,
    square_fluid_points,
    validate_square_config,
    write_provenance,
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_config(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    return json.loads(sidecar.read_text(encoding="utf-8"))["config"]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    known, _ = parser.parse_known_args()
    config = checkpoint_config(known.resume)
    install_backend(config)

    production = load_module(
        AIRFOIL_ROOT / "scripts/train_hardcoded.py",
        "production_train_hardcoded",
    )
    original_validate = production.validate_config
    production.validate_config = lambda item: validate_square_config(
        item, original_validate
    )
    production.fluid_points = square_fluid_points
    production.continuous_parameters = square_continuous_parameters
    write_provenance(known.run_dir / "square_case_provenance.json", config)
    (known.run_dir / "square_all_fp32_protocol.json").write_text(
        json.dumps(
            {
                "all_numerical_paths": "float32",
                "mlp_compute": "float32",
                "mlp_accumulation": "float32",
                "jet3": "float32",
                "parameter_backward": "float32",
                "quadrature": "float32",
                "ritz": "float32",
                "master_weights": "float32",
                "adam": "float32",
                "early_stopping_enabled": False,
                "lr_scheduler_type": config["lr_scheduler_type"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    production.main()
    protocol_path = known.run_dir / "hardcoded_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["precision"] = {"all_numerical_paths": "float32"}
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    completion_path = known.run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["precision"] = {"all_numerical_paths": "float32"}
        completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
