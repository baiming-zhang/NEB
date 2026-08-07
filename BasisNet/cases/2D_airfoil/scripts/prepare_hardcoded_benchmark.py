"""Prepare a reproducible epoch-0 checkpoint and frozen PyTorch sample pack.

This is a one-time reference-data utility.  The actual hardcoded benchmark
trainer imports neither PyTorch nor this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.training import (  # noqa: E402
    Config,
    OfficialPeelingFields,
    continuous_parameters,
    fluid_points,
    set_seed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    config_values = json.loads(args.config.read_text(encoding="utf-8"))
    config_values["epochs"] = args.epochs
    config_field_names = {item.name for item in fields(Config)}
    cfg = Config(**{
        key: value
        for key, value in config_values.items()
        if key in config_field_names
    })
    set_seed(cfg.seed)
    model = OfficialPeelingFields(cfg)

    initial: dict[str, np.ndarray] = {}
    state = model.state_dict()
    for mode in range(cfg.modes):
        source_prefix = f"modes.{mode}"
        target_prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        for layer in range(cfg.hidden_layers):
            for kind in ("weight", "bias"):
                source = f"{source_prefix}.layers.{layer}.{kind}"
                target = f"{target_prefix}.l{layer}.{kind}"
                value = state[source].detach().cpu().numpy().copy()
                if config_values.get("mlp_compute_dtype") == "float64":
                    value = value.astype(np.float64)
                initial[target] = value
                parameter_names.append(target)
        for kind in ("weight", "bias"):
            source = f"{source_prefix}.output.{kind}"
            target = f"{target_prefix}.output.{kind}"
            value = state[source].detach().cpu().numpy().copy()
            if config_values.get("mlp_compute_dtype") == "float64":
                value = value.astype(np.float64)
            initial[target] = value
            parameter_names.append(target)
        for index, name in enumerate(parameter_names):
            stem = f"{target_prefix}.adam.p{index:02d}"
            initial[f"{stem}.exp_avg"] = np.zeros_like(initial[name])
            initial[f"{stem}.exp_avg_sq"] = np.zeros_like(initial[name])
            initial[f"{stem}.step"] = np.asarray(0, dtype=np.int64)

    checkpoint = args.output_dir / "initial_epoch0000.npz"
    save_npz(checkpoint, initial)
    checkpoint_metadata = {
        "format": "eigenfluid_airfoil_hardcoded_v1",
        "source": "deterministic PyTorch initialization with the original seed",
        "epoch": 0,
        "best_validation": float("inf"),
        "config": {**asdict(cfg), **config_values},
        "training_uses_pytorch": False,
        "spatial_ad": "hardcoded_xy_jet2",
        "parameter_backward": "hardcoded_reverse_jet2",
    }
    checkpoint.with_suffix(".json").write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )

    train_points = np.empty(
        (args.epochs, cfg.points_per_step, 2), dtype=np.float32
    )
    train_beta = np.empty(args.epochs, dtype=np.float64)
    train_alpha = np.empty(args.epochs, dtype=np.float64)
    for epoch in range(1, args.epochs + 1):
        beta, alpha = continuous_parameters(cfg, epoch)
        train_beta[epoch - 1] = beta
        train_alpha[epoch - 1] = alpha
        train_points[epoch - 1] = fluid_points(
            cfg, cfg.points_per_step, beta, alpha, cfg.seed + epoch * 1009
        )

    validation_pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples
        )
    ]
    validation_points = np.empty(
        (len(validation_pairs), cfg.validation_points, 2), dtype=np.float32
    )
    validation_beta = np.empty(len(validation_pairs), dtype=np.float64)
    validation_alpha = np.empty(len(validation_pairs), dtype=np.float64)
    for index, (beta, alpha) in enumerate(validation_pairs):
        validation_beta[index] = beta
        validation_alpha[index] = alpha
        validation_points[index] = fluid_points(
            cfg, cfg.validation_points, beta, alpha, cfg.seed + 900_000 + index
        )

    sample_pack = args.output_dir / f"sample_pack_epochs0001_{args.epochs:04d}.npz"
    save_npz(
        sample_pack,
        {
            "train_points": train_points,
            "train_beta": train_beta,
            "train_alpha": train_alpha,
            "validation_points": validation_points,
            "validation_beta": validation_beta,
            "validation_alpha": validation_alpha,
            "probe_points": fluid_points(
                cfg,
                cfg.validation_points,
                cfg.probe_beta,
                cfg.probe_alpha_deg,
                cfg.seed + 1_500_000,
            ),
        },
    )
    manifest = {
        "config": str(args.config.resolve()),
        "epochs": args.epochs,
        "points_per_epoch": cfg.points_per_step,
        "validation_cases": len(validation_pairs),
        "initial_checkpoint": str(checkpoint.resolve()),
        "initial_checkpoint_sha256": sha256(checkpoint),
        "sample_pack": str(sample_pack.resolve()),
        "sample_pack_sha256": sha256(sample_pack),
        "pytorch_used_only_to_freeze_reference_inputs": True,
        "benchmark_trainer_uses_pytorch": False,
    }
    (args.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
