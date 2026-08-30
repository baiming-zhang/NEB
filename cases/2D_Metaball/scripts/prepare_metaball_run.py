"""Prepare all-FP32 Kinematic-Basis geometry-family Jet inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from metaball_runtime import (
    FP32_KEYS,
    backend,
    geometry_from_token,
    install_backend,
    metaball_fluid_points,
    validate_metaball_config,
    write_provenance,
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


def validate_without_production(config: dict) -> None:
    def shadow_validator(item: dict) -> None:
        if item.get("basis_objective") not in {
            "pure_vorticity_rayleigh", "streamfunction_dirichlet_rayleigh",
            "velocity_rayleigh",
        }:
            raise ValueError("unsupported metaball basis objective")
        if item.get("sampling_mode") not in {
            "uniform_fluid_area_sobol", "uniform_random"
        }:
            raise ValueError("unsupported metaball sampling mode")
        derivative_order = int(item.get("spatial_derivative_order", -1))
        if item.get("basis_objective") == "pure_vorticity_rayleigh":
            if derivative_order != 3:
                raise ValueError("vorticity Jet3 requires third spatial derivatives")
        elif item.get("basis_objective") == "velocity_rayleigh":
            if derivative_order < 2:
                raise ValueError("velocity Rayleigh requires second spatial derivatives")
        elif derivative_order < 1:
            raise ValueError("streamfunction Rayleigh requires spatial gradients")

    validate_metaball_config(config, shadow_validator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_without_production(config)
    install_backend(config)
    cfg = backend.cfg_namespace(config)
    jet_order = int(cfg.spatial_derivative_order)
    if str(cfg.basis_objective) == "streamfunction_dirichlet_rayleigh":
        if jet_order != 1:
            raise ValueError(
                "streamfunction preparation must use spatial_derivative_order=1 "
                "to avoid redundant higher derivatives"
            )
    elif jet_order != 3:
        raise ValueError("vorticity/velocity preparation requires Jet3")
    spatial_ad = (
        "analytic_value_gradient_xy_jet1"
        if jet_order == 1
        else "analytic_compact_symmetric_xy_jet3"
    )
    parameter_backward = f"handwritten_reverse_jet{jet_order}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    arrays: dict[str, np.ndarray] = {}
    input_features = int(cfg.input_features)
    canonical = bool(getattr(cfg, "canonical_siren", False))
    first_omega = float(getattr(cfg, "siren_first_omega0", 30.0))
    hidden_omega = float(getattr(cfg, "siren_hidden_omega0", 30.0))
    if canonical and (first_omega <= 0.0 or hidden_omega <= 0.0):
        raise ValueError("canonical SIREN omega values must be positive")
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = input_features
        for layer in range(cfg.hidden_layers):
            if canonical:
                # Exact distributions used by the official implementation:
                # first-layer weights U(-1/fan_in, 1/fan_in), later weights
                # U(-sqrt(6/fan_in)/omega, +sqrt(6/fan_in)/omega).  PyTorch's
                # Linear reset leaves biases at U(-1/sqrt(fan_in), ...).
                bound = (
                    1.0 / fan_in
                    if layer == 0
                    else math.sqrt(6.0 / fan_in) / hidden_omega
                )
                bias_bound = 1.0 / math.sqrt(fan_in)
            else:
                gain = float(
                    cfg.initial_first_layer_gain
                    if layer == 0 else cfg.initial_hidden_layer_gain
                )
                bound = gain / math.sqrt(fan_in)
                bias_bound = bound
            weight_name = f"{prefix}.l{layer}.weight"
            bias_name = f"{prefix}.l{layer}.bias"
            arrays[weight_name] = rng.uniform(
                -bound, bound, (cfg.hidden, fan_in)
            ).astype(np.float32)
            arrays[bias_name] = rng.uniform(
                -bias_bound, bias_bound, cfg.hidden
            ).astype(np.float32)
            parameter_names.extend((weight_name, bias_name))
            fan_in = cfg.hidden
        bound = (
            math.sqrt(6.0 / fan_in) / hidden_omega
            if canonical
            else float(cfg.initial_output_layer_gain) / math.sqrt(fan_in)
        )
        output_bias_bound = 1.0 / math.sqrt(fan_in) if canonical else bound
        weight_name = f"{prefix}.output.weight"
        bias_name = f"{prefix}.output.bias"
        arrays[weight_name] = rng.uniform(
            -bound, bound, (1, fan_in)
        ).astype(np.float32)
        arrays[bias_name] = rng.uniform(
            -output_bias_bound, output_bias_bound, 1
        ).astype(np.float32)
        parameter_names.extend((weight_name, bias_name))
        for index, name in enumerate(parameter_names):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)

    initial = args.output_dir / "initial_epoch0000.npz"
    save_npz(initial, arrays)
    precision = {key: config[key] for key in FP32_KEYS}
    initial.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": f"eigenfluid_metaball_family_jet{jet_order}_fp32_v1",
                "source": "NumPy PCG64 deterministic initialization",
                "epoch": 0,
                "best_validation": float("inf"),
                "config": config,
                "training_uses_pytorch": False,
                "automatic_differentiation_used": False,
                "finite_difference_used": False,
                "spatial_ad": spatial_ad,
                "parameter_backward": parameter_backward,
                "precision": precision,
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

    validation_count = int(cfg.validation_beta_samples) * int(
        cfg.validation_alpha_samples
    )
    validation_tokens = np.arange(
        1_000_001, 1_000_001 + validation_count, dtype=np.float32
    )
    validation_points = np.stack(
        [
            metaball_fluid_points(
                cfg,
                cfg.validation_points,
                float(token),
                0.0,
                cfg.seed + 900_000 + index,
            )
            for index, token in enumerate(validation_tokens)
        ]
    )
    validation_geometry = np.stack(
        [geometry_from_token(cfg, float(token)) for token in validation_tokens]
    )
    sampler_stem = (
        "fixed_validation_iid_uniform_v1"
        if config.get("sampling_mode") == "uniform_random"
        else "fixed_validation_scipy_sobol_v1"
    )
    validation = args.output_dir / f"{sampler_stem}.npz"
    save_npz(
        validation,
        {
            "validation_points": validation_points.astype(np.float32),
            "validation_beta": validation_tokens,
            "validation_alpha": np.zeros(validation_count, dtype=np.float32),
            "validation_geometry": validation_geometry.astype(np.float32),
            "probe_points": metaball_fluid_points(
                cfg,
                cfg.validation_points,
                cfg.probe_beta,
                0.0,
                cfg.seed + 1_500_000,
            ),
            "probe_geometry": geometry_from_token(cfg, cfg.probe_beta),
        },
    )
    validation.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "eigenfluid_metaball_family_validation_fp32_v1",
                "config": config,
                "validation_shape_count": validation_count,
                "validation_points_per_shape": cfg.validation_points,
                "geometry_tokens": validation_tokens.tolist(),
                "geometry_parameters": validation_geometry.tolist(),
                "geometry_parameter_order": [
                    "c1x", "c1y", "r1", "c2x", "c2y", "r2",
                    "c3x", "c3y", "r3",
                ],
                "sampler": (
                    "IID PCG64 rejection sampling over physical fluid area"
                    if config.get("sampling_mode") == "uniform_random"
                    else "scrambled Sobol uniform physical fluid area"
                ),
                "dtype": "float32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "config": str(args.config.resolve()),
        "initializer": "numpy_pcg64_v1",
        "input_features": input_features,
        "geometry": (
            "Victoria SA 2025 three-circle family, conditioned to one "
            "smooth connected obstacle in a rounded square"
            if config.get("metaball_connectivity") == "connected_smooth_union"
            else "Victoria SA 2025 three-circle family in rounded square"
        ),
        "activation": "siren",
        "input_encoding": config["input_encoding"],
        "coordinate_normalization": config.get(
            "coordinate_normalization", "centered_unit"
        ),
        "basis_objective": config["basis_objective"],
        "spatial_ad": spatial_ad,
        "parameter_backward": parameter_backward,
        "precision": precision,
        "initial_checkpoint": initial.name,
        "initial_sha256": digest(initial),
        "fixed_validation": validation.name,
        "fixed_validation_sha256": digest(validation),
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
        "finite_difference_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_provenance(args.output_dir / "metaball_case_provenance.json", config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
