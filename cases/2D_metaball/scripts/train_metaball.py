"""Run the production native-CUDA Jet3 trainer on the metaball family."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from metaball_runtime import (
    AIRFOIL_ROOT,
    FP32_KEYS,
    install_backend,
    metaball_continuous_parameters,
    metaball_fluid_points,
    validate_metaball_config,
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
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["config"]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    known, _ = parser.parse_known_args()
    config = checkpoint_config(known.resume)
    install_backend(config)

    production = load_module(
        AIRFOIL_ROOT / "scripts/train_hardcoded.py",
        "production_train_hardcoded_metaball",
    )
    original_validate = production.validate_config
    production.validate_config = lambda item: validate_metaball_config(
        item, original_validate
    )
    production.fluid_points = metaball_fluid_points
    production.continuous_parameters = metaball_continuous_parameters
    write_provenance(known.run_dir / "metaball_case_provenance.json", config)
    (known.run_dir / "metaball_all_fp32_protocol.json").write_text(
        json.dumps({key: config[key] for key in FP32_KEYS}, indent=2),
        encoding="utf-8",
    )
    production.main()

    protocol_path = known.run_dir / "hardcoded_protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["geometry_parameter_transport"] = (
            "legacy beta field carries an integer geometry token; the network "
            "receives the corresponding nine raw circle parameters"
        )
        protocol["precision"] = {"all_numerical_paths": "float32"}
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    completion_path = known.run_dir / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["precision"] = {"all_numerical_paths": "float32"}
        completion["geometry_family"] = config["case_domain"]
        completion_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
