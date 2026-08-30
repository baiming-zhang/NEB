"""Create a no-PyTorch epoch-0 model and immutable validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
    fluid_points,
    uniform_fluid_points,
)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = cfg_namespace(config)
    rng = np.random.default_rng(cfg.seed)
    arrays: dict[str, np.ndarray] = {}
    input_features = 4 * cfg.frequencies + 4
    first_layer_gain = float(getattr(cfg, "initial_first_layer_gain", 1.0))
    hidden_layer_gain = float(getattr(cfg, "initial_hidden_layer_gain", 1.0))
    output_layer_gain = float(getattr(cfg, "initial_output_layer_gain", 1.0))
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = input_features
        for layer in range(cfg.hidden_layers):
            gain = first_layer_gain if layer == 0 else hidden_layer_gain
            bound = gain / math.sqrt(fan_in)
            weight_name = f"{prefix}.l{layer}.weight"
            bias_name = f"{prefix}.l{layer}.bias"
            arrays[weight_name] = rng.uniform(
                -bound, bound, (cfg.hidden, fan_in)
            ).astype(np.float32)
            arrays[bias_name] = rng.uniform(
                -bound, bound, cfg.hidden
            ).astype(np.float32)
            parameter_names.extend((weight_name, bias_name))
            fan_in = cfg.hidden
        bound = output_layer_gain / math.sqrt(fan_in)
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

    initial = args.output_dir / "initial_epoch0000.npz"
    save_npz(initial, arrays)
    initial.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "eigenfluid_airfoil_hardcoded_v2",
                "source": "NumPy PCG64 deterministic initialization",
                "epoch": 0,
                "best_validation": float("inf"),
                "config": config,
                "training_uses_pytorch": False,
                "spatial_ad": "hardcoded_xy_jet2",
                "parameter_backward": "hardcoded_reverse_jet2",
                "training_state": {
                    "current_learning_rate": cfg.learning_rate,
                    "best_validation_epoch": 0,
                    "lr_reference_loss": float("inf"),
                    "lr_reference_epoch": 0,
                    "lr_reductions": 0,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples
        )
    ]
    point_generator = (
        uniform_fluid_points
        if str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
        == "stokes_certified"
        else fluid_points
    )
    validation_points = np.stack(
        [
            point_generator(
                cfg,
                cfg.validation_points,
                beta,
                alpha,
                cfg.seed + 900_000 + index,
            ).astype(np.float32)
            for index, (beta, alpha) in enumerate(pairs)
        ]
    )
    validation = args.output_dir / "fixed_validation_scipy_sobol_v1.npz"
    save_npz(
        validation,
        {
            "validation_points": validation_points,
            "validation_beta": np.asarray([pair[0] for pair in pairs]),
            "validation_alpha": np.asarray([pair[1] for pair in pairs]),
            "probe_points": point_generator(
                cfg,
                cfg.validation_points,
                cfg.probe_beta,
                cfg.probe_alpha_deg,
                cfg.seed + 1_500_000,
            ).astype(np.float32),
        },
    )
    manifest = {
        "config": str(args.config.resolve()),
        "initializer": "numpy_pcg64_v1",
        "initial_first_layer_gain": first_layer_gain,
        "initial_hidden_layer_gain": hidden_layer_gain,
        "initial_output_layer_gain": output_layer_gain,
        "sampler": (
            "uniform_nested_scipy_scrambled_sobol_v1"
            if point_generator is uniform_fluid_points
            else "scipy_scrambled_sobol_v1"
        ),
        "geometry": "c2_ellipse",
        "initial_checkpoint": str(initial.resolve()),
        "initial_sha256": digest(initial),
        "fixed_validation": str(validation.resolve()),
        "fixed_validation_sha256": digest(validation),
        "training_uses_pytorch": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
