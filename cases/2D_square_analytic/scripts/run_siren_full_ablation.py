"""Run the resumable, fixed-budget K=64 SIREN normalization/parameter ablation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path


CASE_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = CASE_ROOT / "configs/k64_square_siren_canonical_ablation_base.json"
RESULT_ROOT = CASE_ROOT / "results/siren_k64_full_ablation"
MODES = 64
FORMAL_EPOCHS = 500
NORMALIZATIONS = ("unit_interval", "centered_unit", "standardized_uniform")
EFFECTIVE_FIRST_OMEGAS = (4.0, 8.0, 16.0, 32.0, 64.0)
HIDDEN_OMEGAS = (1.0, 8.0, 30.0)
ARCHITECTURE_LRS = (1.0e-4, 4.0e-4)
OPTIMIZATION_LRS = (5.0e-5, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4, 1.6e-3)
WIDTHS = (32, 48, 64, 96, 128)
DEPTHS = (2, 3, 4, 6, 8)
SCHEDULERS = ("constant", "validation_plateau", "cosine")


def normalization_scale(normalization: str) -> float:
    return {
        "unit_interval": 1.0,
        "centered_unit": 2.0,
        "standardized_uniform": math.sqrt(12.0),
    }[normalization]


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
    return json.loads(sidecar.read_text(encoding="utf-8")).get("config") == config


def architecture_experiments() -> dict[str, dict]:
    experiments: dict[str, dict] = {}
    for learning_rate in ARCHITECTURE_LRS:
        lr_name = f"lr{learning_rate:.0e}".replace("-", "m")
        experiments[f"legacy_raw_padded_o1_{lr_name}"] = {
            "group": "legacy_architecture",
            "overrides": {
                "canonical_siren": False,
                "input_encoding": "raw_xy_padded",
                "coordinate_normalization": "centered_unit",
                "siren_first_layer_spatial_omega0": 1.0,
                "learning_rate": learning_rate,
            },
        }
        experiments[f"legacy_raw_padded_o8_{lr_name}"] = {
            "group": "legacy_architecture",
            "overrides": {
                "canonical_siren": False,
                "input_encoding": "raw_xy_padded",
                "coordinate_normalization": "centered_unit",
                "siren_first_layer_spatial_omega0": 8.0,
                "learning_rate": learning_rate,
            },
        }
        experiments[f"legacy_posenc_{lr_name}"] = {
            "group": "legacy_architecture",
            "overrides": {
                "canonical_siren": False,
                "input_encoding": "positional_encoding",
                "coordinate_normalization": "centered_unit",
                "siren_first_layer_spatial_omega0": 1.0,
                "learning_rate": learning_rate,
            },
        }
        for normalization, effective_omega, hidden_omega in itertools.product(
            NORMALIZATIONS, EFFECTIVE_FIRST_OMEGAS, HIDDEN_OMEGAS
        ):
            first_omega = effective_omega / normalization_scale(normalization)
            norm_name = {
                "unit_interval": "unit",
                "centered_unit": "center",
                "standardized_uniform": "zscore",
            }[normalization]
            name = (
                f"canonical_{norm_name}_eff{effective_omega:g}_"
                f"f{first_omega:.6g}_h{hidden_omega:g}_{lr_name}"
                .replace(".", "p")
            )
            experiments[name] = {
                "group": "canonical_normalization_frequency",
                "overrides": {
                    "canonical_siren": True,
                    "input_encoding": "raw_xy",
                    "coordinate_normalization": normalization,
                    "siren_first_layer_spatial_omega0": 1.0,
                    "siren_first_omega0": first_omega,
                    "siren_hidden_omega0": hidden_omega,
                    "learning_rate": learning_rate,
                },
            }
    return experiments


def learning_rate_experiments(base: dict) -> dict[str, dict]:
    return {
        f"lr_{learning_rate:.0e}".replace("-", "m"): {
            "group": "learning_rate",
            "overrides": {**base, "learning_rate": learning_rate, "lr_scheduler_type": "constant"},
        }
        for learning_rate in OPTIMIZATION_LRS
    }


def topology_experiments(base: dict) -> dict[str, dict]:
    experiments: dict[str, dict] = {}
    for scheduler in SCHEDULERS:
        experiments[f"scheduler_{scheduler}"] = {
            "group": "scheduler",
            "overrides": {**base, "lr_scheduler_type": scheduler},
        }
    for width in WIDTHS:
        experiments[f"width_{width}"] = {
            "group": "width",
            "overrides": {**base, "hidden": width, "lr_scheduler_type": "constant"},
        }
    for depth in DEPTHS:
        experiments[f"depth_{depth}"] = {
            "group": "depth",
            "overrides": {**base, "hidden_layers": depth, "lr_scheduler_type": "constant"},
        }
    return experiments


def rank_items(summary: dict, group: str | None = None) -> list[dict]:
    items = list(summary["experiments"].values())
    if group is not None:
        items = [item for item in items if item["group"] == group]
    return sorted(
        items,
        key=lambda item: (
            -int(item["resolved_rank"]),
            float(item["median_positive_spectrum_relative_error"]),
            float(item["p90_positive_spectrum_relative_error"]),
        ),
    )


def interaction_experiments(
    architecture_base: dict,
    lr_summary: dict,
    topology_summary: dict,
) -> dict[str, dict]:
    learning_rates = [item["config"]["learning_rate"] for item in rank_items(lr_summary)[:2]]
    widths = [item["config"]["width"] for item in rank_items(topology_summary, "width")[:2]]
    depths = [item["config"]["depth"] for item in rank_items(topology_summary, "depth")[:2]]
    schedulers = [item["config"]["scheduler"] for item in rank_items(topology_summary, "scheduler")[:2]]
    experiments: dict[str, dict] = {}
    for learning_rate, width, depth, scheduler in itertools.product(
        learning_rates, widths, depths, schedulers
    ):
        name = (
            f"lr{learning_rate:.0e}_w{width}_d{depth}_{scheduler}"
            .replace("-", "m")
        )
        experiments[name] = {
            "group": "interaction",
            "overrides": {
                **architecture_base,
                "learning_rate": learning_rate,
                "hidden": width,
                "hidden_layers": depth,
                "lr_scheduler_type": scheduler,
            },
        }
    return experiments


def confirmation_experiments(interaction_summary: dict) -> dict[str, dict]:
    experiments: dict[str, dict] = {}
    for index, item in enumerate(rank_items(interaction_summary)[:3], start=1):
        config = item["config"]
        experiments[f"rank{index}_{item['experiment']}"] = {
            "group": "confirmation",
            "overrides": {
                "canonical_siren": bool(config["canonical_siren"]),
                "input_encoding": (
                    "raw_xy" if config["architecture"] == "canonical_raw_fanin2"
                    else config["architecture"]
                ),
                "coordinate_normalization": config["coordinate_normalization"],
                "siren_first_layer_spatial_omega0": config["spatial_omega0"],
                "siren_first_omega0": config["first_omega0"],
                "siren_hidden_omega0": config["hidden_omega0"],
                "learning_rate": config["learning_rate"],
                "hidden": config["width"],
                "hidden_layers": config["depth"],
                "lr_scheduler_type": config["scheduler"],
                "seed": 20260821,
            },
        }
    return experiments


def resolved_config(
    base_config: dict,
    overrides: dict,
    epochs: int,
    points_per_step: int,
    validation_points: int,
    result_root: Path,
) -> dict:
    config = dict(base_config)
    config.update(overrides)
    config.update({
        "modes": MODES,
        "epochs": epochs,
        "points_per_step": points_per_step,
        "validation_points": validation_points,
        "base_learning_rate": float(config["learning_rate"]),
        "early_stopping_enabled": False,
        "early_stopping_patience_epochs": 2_147_483_647,
        "validate_every": min(50, max(1, epochs // 3)),
        # The production trainer intentionally fixes the neutral restart cadence.
        "checkpoint_every": 100,
        "result_root": str(result_root.resolve()),
    })
    return config


def run_campaign(
    label: str,
    experiments: dict[str, dict],
    epochs: int,
    points_per_step: int,
    validation_points: int,
    max_retries: int,
) -> Path:
    campaign = RESULT_ROOT / label
    campaign.mkdir(parents=True, exist_ok=True)
    state_path = campaign / "campaign_state.json"
    state = {
        "complete": False,
        "modes": MODES,
        "epochs": epochs,
        "points_per_step": points_per_step,
        "validation_points": validation_points,
        "early_stopping_enabled": False,
        "experiments": {},
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        state["experiments"].update({
            name: previous["experiments"][name]
            for name in experiments
            if name in previous.get("experiments", {})
        })
    base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    initial_hashes: dict[str, str] = {}

    for ordinal, (name, specification) in enumerate(experiments.items(), start=1):
        root = campaign / name
        inputs = root / "inputs"
        run_dir = root / "run_001"
        config_path = root / "resolved_config.json"
        config = resolved_config(
            base_config,
            specification["overrides"],
            epochs,
            points_per_step,
            validation_points,
            root,
        )
        atomic_json(config_path, config)
        item = state["experiments"].setdefault(name, {})
        item.update({
            "status": "preparing",
            "group": specification["group"],
            "ordinal": ordinal,
            "total": len(experiments),
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
                raise RuntimeError(f"preparation failed for {label}/{name}")
        initial_hashes[name] = digest(initial)

        completion_path = run_dir / "completion.json"
        if completion_path.is_file():
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("complete")
                and completion.get("final_epoch") == epochs
                and completion.get("stopped_early") is False
            ):
                item.pop("return_code", None)
                item.update({"status": "complete", "completion": completion})
                atomic_json(state_path, state)
                continue

        retries = 0
        while True:
            resume = choose_resume(inputs, run_dir)
            item.update({
                "status": "training",
                "resume_epoch": checkpoint_epoch(resume),
                "retry_count": retries,
            })
            atomic_json(state_path, state)
            code = run(
                [
                    sys.executable,
                    str(CASE_ROOT / "scripts/train_square.py"),
                    "--resume", str(resume),
                    "--run-dir", str(run_dir),
                    "--validation-pack", str(validation),
                ],
                root / "orchestrator.log",
            )
            if code == 0:
                break
            retries += 1
            item.update({"status": "retrying", "return_code": code, "retry_count": retries})
            atomic_json(state_path, state)
            if retries > max_retries:
                item["status"] = "failed"
                atomic_json(state_path, state)
                raise RuntimeError(f"training failed for {label}/{name} after {retries} retries")
            time.sleep(min(60, 5 * retries))

        if not completion_path.is_file():
            raise RuntimeError(f"missing completion marker for {label}/{name}")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("stopped_early") is not False:
            raise RuntimeError(f"early stopping unexpectedly activated for {label}/{name}")
        item.pop("return_code", None)
        item.update({"status": "complete", "completion": completion})
        atomic_json(state_path, state)

    state["initial_sha256_by_experiment"] = initial_hashes
    state["complete"] = all(
        state["experiments"].get(name, {}).get("status") == "complete"
        for name in experiments
    )
    atomic_json(state_path, state)
    if not state["complete"]:
        raise RuntimeError(f"campaign {label} did not complete")
    return campaign


def evaluate_campaign(campaign: Path) -> dict:
    output = campaign / "evaluation"
    code = run(
        [
            sys.executable,
            str(CASE_ROOT / "scripts/evaluate_siren_sweep.py"),
            "--campaign-root", str(campaign),
            "--output-dir", str(output),
            "--summary-only",
        ],
        campaign / "evaluation.log",
    )
    if code:
        raise RuntimeError(f"evaluation failed for {campaign.name}")
    return json.loads((output / "siren_k64_sweep_summary.json").read_text(encoding="utf-8"))


def architecture_overrides_from_summary(summary: dict) -> dict:
    best = rank_items(summary)[0]["config"]
    architecture = best["architecture"]
    return {
        "canonical_siren": bool(best["canonical_siren"]),
        "input_encoding": (
            "raw_xy" if architecture == "canonical_raw_fanin2" else architecture
        ),
        "coordinate_normalization": best["coordinate_normalization"],
        "siren_first_layer_spatial_omega0": best["spatial_omega0"],
        "siren_first_omega0": best["first_omega0"],
        "siren_hidden_omega0": best["hidden_omega0"],
        "learning_rate": best["learning_rate"],
        "hidden": best["width"],
        "hidden_layers": best["depth"],
        "lr_scheduler_type": best["scheduler"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument(
        "--through-stage",
        choices=("architecture", "learning_rate", "topology", "interaction", "confirmation"),
        default="confirmation",
    )
    args = parser.parse_args()
    stage_order = ("architecture", "learning_rate", "topology", "interaction", "confirmation")
    stop_index = stage_order.index(args.through_stage)

    architecture_campaign = run_campaign(
        "01_architecture_normalization_effective_frequency_e500_q2048",
        architecture_experiments(),
        epochs=FORMAL_EPOCHS,
        points_per_step=2048,
        validation_points=4096,
        max_retries=args.max_retries,
    )
    architecture_summary = evaluate_campaign(architecture_campaign)
    architecture_base = architecture_overrides_from_summary(architecture_summary)
    if stop_index == 0:
        return

    lr_campaign = run_campaign(
        "02_learning_rate_e500_q4096",
        learning_rate_experiments(architecture_base),
        epochs=FORMAL_EPOCHS,
        points_per_step=4096,
        validation_points=4096,
        max_retries=args.max_retries,
    )
    lr_summary = evaluate_campaign(lr_campaign)
    best_lr = rank_items(lr_summary)[0]["config"]["learning_rate"]
    optimized_base = {**architecture_base, "learning_rate": best_lr}
    if stop_index == 1:
        return

    topology_campaign = run_campaign(
        "03_scheduler_width_depth_e500_q4096",
        topology_experiments(optimized_base),
        epochs=FORMAL_EPOCHS,
        points_per_step=4096,
        validation_points=4096,
        max_retries=args.max_retries,
    )
    topology_summary = evaluate_campaign(topology_campaign)
    if stop_index == 2:
        return

    interaction_campaign = run_campaign(
        "04_top2_interaction_e500_q4096",
        interaction_experiments(architecture_base, lr_summary, topology_summary),
        epochs=FORMAL_EPOCHS,
        points_per_step=4096,
        validation_points=4096,
        max_retries=args.max_retries,
    )
    interaction_summary = evaluate_campaign(interaction_campaign)
    if stop_index == 3:
        return

    confirmation_campaign = run_campaign(
        "05_top3_confirmation_e500_q4096",
        confirmation_experiments(interaction_summary),
        epochs=FORMAL_EPOCHS,
        points_per_step=4096,
        validation_points=4096,
        max_retries=args.max_retries,
    )
    confirmation_summary = evaluate_campaign(confirmation_campaign)
    manifest = {
        "status": "complete",
        "modes": MODES,
        "formal_epochs_per_arm": FORMAL_EPOCHS,
        "early_stopping_enabled": False,
        "selection_rule": "maximize resolved rank, then minimize median and p90 positive-spectrum error",
        "stage_summaries": {
            "architecture": str((architecture_campaign / "evaluation/siren_k64_sweep_summary.json").resolve()),
            "learning_rate": str((lr_campaign / "evaluation/siren_k64_sweep_summary.json").resolve()),
            "topology": str((topology_campaign / "evaluation/siren_k64_sweep_summary.json").resolve()),
            "interaction": str((interaction_campaign / "evaluation/siren_k64_sweep_summary.json").resolve()),
            "confirmation": str((confirmation_campaign / "evaluation/siren_k64_sweep_summary.json").resolve()),
        },
        "best_confirmation": confirmation_summary["best_experiment"],
        "best_confirmation_result": confirmation_summary["experiments"][
            confirmation_summary["best_experiment"]
        ],
    }
    atomic_json(RESULT_ROOT / "full_ablation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
