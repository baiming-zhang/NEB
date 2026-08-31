"""Run the controlled K=64 raw-SIREN rectangular-domain sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = CASE_ROOT / "configs/k64_square_siren_raw_sweep_base.json"
RESULT_ROOT = CASE_ROOT / "results/siren_frequency_cap_k64"
MODES = 64

# Each group changes one factor from the shared omega32/lr4e-4/d4w64/constant
# control. The control appears once and is reused by all four ablations.
EXPERIMENTS: dict[str, dict[str, object]] = {
    "freq_omega01": {"group": "frequency", "siren_first_layer_spatial_omega0": 1.0},
    "freq_omega08": {"group": "frequency", "siren_first_layer_spatial_omega0": 8.0},
    "freq_omega16": {"group": "frequency", "siren_first_layer_spatial_omega0": 16.0},
    "control_omega32": {"group": "control", "siren_first_layer_spatial_omega0": 32.0},
    "lr_1e-4": {"group": "learning_rate", "learning_rate": 1.0e-4},
    "lr_8e-4": {"group": "learning_rate", "learning_rate": 8.0e-4},
    "scheduler_plateau": {"group": "scheduler", "lr_scheduler_type": "validation_plateau"},
    "scheduler_cosine": {"group": "scheduler", "lr_scheduler_type": "cosine"},
    "width_32": {"group": "width", "hidden": 32},
    "width_96": {"group": "width", "hidden": 96},
    "depth_2": {"group": "depth", "hidden_layers": 2},
    "depth_6": {"group": "depth", "hidden_layers": 6},
}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {' '.join(command)}\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        return completed.returncode


def resolved_config(name: str, epochs: int, smoke: bool) -> dict:
    config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    overrides = dict(EXPERIMENTS[name])
    overrides.pop("group")
    config.update(overrides)
    config["modes"] = MODES
    config["epochs"] = 2 if smoke else epochs
    config["base_learning_rate"] = config["learning_rate"]
    config["early_stopping_enabled"] = False
    config["early_stopping_patience_epochs"] = 2_147_483_647
    config["result_root"] = str((RESULT_ROOT / name).resolve())
    if smoke:
        config.update({
            "points_per_step": 128,
            "validation_points": 128,
        })
    return config


def checkpoint_epoch(path: Path) -> int:
    sidecar = path.with_suffix(".json")
    if not path.is_file() or not sidecar.is_file():
        return -1
    return int(json.loads(sidecar.read_text(encoding="utf-8"))["epoch"])


def choose_resume(inputs: Path, run_dir: Path) -> Path:
    latest = run_dir / "checkpoints/hardcoded_latest.npz"
    return latest if checkpoint_epoch(latest) >= 0 else inputs / "initial_epoch0000.npz"


def prepared_for_config(initial: Path, validation: Path, config: dict) -> bool:
    sidecar = initial.with_suffix(".json")
    if not initial.is_file() or not validation.is_file() or not sidecar.is_file():
        return False
    prepared = json.loads(sidecar.read_text(encoding="utf-8"))
    return prepared.get("config") == config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", nargs="+", choices=tuple(EXPERIMENTS))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    names = args.only or list(EXPERIMENTS)
    campaign = RESULT_ROOT / ("smoke" if args.smoke else f"epochs_{args.epochs}")
    campaign.mkdir(parents=True, exist_ok=True)
    state_path = campaign / "campaign_state.json"
    state = {
        "complete": False,
        "modes": MODES,
        "epochs": 2 if args.smoke else args.epochs,
        "activation": "siren",
        "input_encoding": "raw_xy_padded",
        "early_stopping_enabled": False,
        "experiments": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiments"].update(previous.get("experiments", {}))
    initial_hashes: dict[str, str] = {}

    for name in names:
        root = campaign / name
        inputs = root / "inputs"
        run_dir = root / "run_001"
        config_path = root / "resolved_config.json"
        config = resolved_config(name, args.epochs, args.smoke)
        atomic_json(config_path, config)
        item = state["experiments"].setdefault(name, {})
        for stale_key in ("return_code", "retry_count"):
            item.pop(stale_key, None)
        item.update({
            "status": "preparing",
            "group": EXPERIMENTS[name]["group"],
            "config": str(config_path.resolve()),
        })
        atomic_json(state_path, state)

        initial = inputs / "initial_epoch0000.npz"
        validation = inputs / "fixed_validation_scipy_sobol_v1.npz"
        if not prepared_for_config(initial, validation, config):
            code = run(
                [
                    sys.executable,
                    str(CASE_ROOT / "scripts/prepare_square_run.py"),
                    "--config", str(config_path),
                    "--output-dir", str(inputs),
                ],
                root / "orchestrator.log",
            )
            if code:
                item.update({"status": "prepare_failed", "return_code": code})
                atomic_json(state_path, state)
                raise SystemExit(code)
        initial_hashes[name] = digest(initial)

        completion = run_dir / "completion.json"
        if completion.is_file():
            completed = json.loads(completion.read_text(encoding="utf-8"))
            if (
                completed.get("complete")
                and completed.get("final_epoch") == config["epochs"]
                and completed.get("stopped_early") is False
            ):
                item.update({"status": "complete", "completion": completed})
                atomic_json(state_path, state)
                continue

        retry = 0
        while True:
            resume = choose_resume(inputs, run_dir)
            command = [
                sys.executable,
                str(CASE_ROOT / "scripts/train_square.py"),
                "--resume", str(resume),
                "--run-dir", str(run_dir),
                "--validation-pack", str(validation),
            ]
            if args.smoke:
                command.extend(("--self-test-epochs", "2", "--self-test-points", "128"))
            item.update({"status": "training", "resume_epoch": checkpoint_epoch(resume)})
            atomic_json(state_path, state)
            code = run(command, root / "orchestrator.log")
            if code == 0:
                break
            retry += 1
            item.update({"status": "retrying", "return_code": code, "retry_count": retry})
            atomic_json(state_path, state)
            if retry > args.max_retries:
                item["status"] = "failed"
                atomic_json(state_path, state)
                raise SystemExit(code)
            time.sleep(min(60, 5 * retry))

        if args.smoke:
            item["status"] = "smoke_pass"
        else:
            if not completion.is_file():
                raise RuntimeError(f"missing completion marker for {name}")
            completed = json.loads(completion.read_text(encoding="utf-8"))
            if completed.get("stopped_early") is not False:
                raise RuntimeError(f"early stopping unexpectedly active for {name}")
            item.update({"status": "complete", "completion": completed})
        atomic_json(state_path, state)

    state["initial_sha256_by_experiment"] = initial_hashes
    state["matched_initial_weights_by_topology"] = {}
    topology_groups: dict[tuple[int, int], list[str]] = {}
    for name in names:
        config = resolved_config(name, args.epochs, args.smoke)
        topology_groups.setdefault(
            (int(config["hidden"]), int(config["hidden_layers"])), []
        ).append(name)
    for topology, group_names in topology_groups.items():
        state["matched_initial_weights_by_topology"][f"w{topology[0]}d{topology[1]}"] = (
            len({initial_hashes[name] for name in group_names}) == 1
        )
    passing = "smoke_pass" if args.smoke else "complete"
    state["complete"] = all(
        state["experiments"].get(name, {}).get("status") == passing
        for name in names
    )
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
