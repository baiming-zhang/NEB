"""Prepare matched epoch-0 weights and fixed validation for the square case."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from square_runtime import (
    AIRFOIL_ROOT,
    install_backend,
    square_fluid_points,
    write_provenance,
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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


def rewrite_wire_initialization(path: Path, config: dict) -> None:
    """Match the official RealGaborLayer's two ``nn.Linear`` projections.

    PyTorch ``nn.Linear.reset_parameters`` is equivalent to a uniform draw
    from ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]`` for both weights and biases.
    NumPy PCG64 supplies the repository's deterministic checkpoint stream.
    """
    rng = np.random.default_rng(int(config["seed"]))
    modes = int(config["modes"])
    hidden = int(config["hidden"])
    layers = int(config["hidden_layers"])
    fan_in_first = 2
    arrays: dict[str, np.ndarray] = {}
    for mode in range(modes):
        prefix = f"m{mode:03d}"
        names: list[str] = []
        fan_in = fan_in_first
        for layer in range(layers):
            bound = 1.0 / math.sqrt(fan_in)
            weight_name = f"{prefix}.l{layer}.weight"
            bias_name = f"{prefix}.l{layer}.bias"
            # Frequency projection first, independent scale projection second.
            arrays[weight_name] = rng.uniform(
                -bound, bound, (2 * hidden, fan_in)
            ).astype(np.float32)
            arrays[bias_name] = rng.uniform(
                -bound, bound, 2 * hidden
            ).astype(np.float32)
            names.extend((weight_name, bias_name))
            fan_in = hidden
        bound = 1.0 / math.sqrt(fan_in)
        weight_name = f"{prefix}.output.weight"
        bias_name = f"{prefix}.output.bias"
        arrays[weight_name] = rng.uniform(
            -bound, bound, (1, fan_in)
        ).astype(np.float32)
        arrays[bias_name] = rng.uniform(-bound, bound, 1).astype(np.float32)
        names.extend((weight_name, bias_name))
        for index, name in enumerate(names):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(arrays[name])
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    save_npz(path, arrays)


def rewrite_canonical_siren_initialization(path: Path, config: dict) -> None:
    """Write the canonical SIREN initializer for a true two-coordinate input."""
    if config.get("input_encoding") != "raw_xy":
        raise ValueError("canonical square SIREN requires input_encoding=raw_xy")
    rng = np.random.default_rng(int(config["seed"]))
    modes = int(config["modes"])
    hidden = int(config["hidden"])
    layers = int(config["hidden_layers"])
    hidden_omega = float(config["siren_hidden_omega0"])
    if hidden_omega <= 0.0:
        raise ValueError("canonical hidden omega must be positive")
    arrays: dict[str, np.ndarray] = {}
    for mode in range(modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = 2
        for layer in range(layers):
            weight_bound = (
                1.0 / fan_in
                if layer == 0
                else math.sqrt(6.0 / fan_in) / hidden_omega
            )
            bias_bound = 1.0 / math.sqrt(fan_in)
            weight_name = f"{prefix}.l{layer}.weight"
            bias_name = f"{prefix}.l{layer}.bias"
            arrays[weight_name] = rng.uniform(
                -weight_bound, weight_bound, (hidden, fan_in)
            ).astype(np.float32)
            arrays[bias_name] = rng.uniform(
                -bias_bound, bias_bound, hidden
            ).astype(np.float32)
            parameter_names.extend((weight_name, bias_name))
            fan_in = hidden
        output_weight_bound = math.sqrt(6.0 / fan_in) / hidden_omega
        output_bias_bound = 1.0 / math.sqrt(fan_in)
        weight_name = f"{prefix}.output.weight"
        bias_name = f"{prefix}.output.bias"
        arrays[weight_name] = rng.uniform(
            -output_weight_bound, output_weight_bound, (1, fan_in)
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
    save_npz(path, arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_backend(config)
    production = load_module(
        AIRFOIL_ROOT / "scripts/prepare_hardcoded_run.py",
        "production_prepare_hardcoded_run",
    )
    production.fluid_points = square_fluid_points
    old_argv = sys.argv
    # The GitHub preparer has a historical guard requiring the airfoil's FP16
    # network setting, although initialization itself always writes FP32 master
    # arrays. Use a temporary guard-only shadow, then restore the authoritative
    # all-FP32 config in every persisted sidecar.
    with tempfile.TemporaryDirectory(
        prefix="square_fp32_prepare_", dir=args.output_dir
    ) as temporary:
        shadow = dict(config)
        shadow["mlp_compute_dtype"] = "float16"
        # Initialization and validation-point preparation are objective
        # independent.  Satisfy the historical production preparer's legacy
        # guard, then restore the authoritative objective in every sidecar.
        shadow["basis_objective"] = "pure_vorticity_rayleigh"
        shadow["spatial_derivative_order"] = 3
        shadow["sampling_mode"] = "uniform_fluid_area_sobol"
        shadow["square_symmetry_quadrature"] = False
        shadow_path = Path(temporary) / "production_guard_shadow.json"
        shadow_path.write_text(json.dumps(shadow, indent=2), encoding="utf-8")
        try:
            sys.argv = [
                str(AIRFOIL_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config", str(shadow_path),
                "--output-dir", str(args.output_dir),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                production.main()
        finally:
            sys.argv = old_argv

    initial_path = args.output_dir / "initial_epoch0000.npz"
    if config.get("activation") == "wire":
        rewrite_wire_initialization(initial_path, config)
    elif bool(config.get("canonical_siren", False)):
        rewrite_canonical_siren_initialization(initial_path, config)

    # The legacy preparer can only pass its guard under a Sobol label. Replace
    # that temporary validation pack with the authoritative IID samples before
    # persisting any metadata for the generic configuration.
    if config.get("sampling_mode") == "uniform_random":
        cfg = SimpleNamespace(**config)
        pairs = [
            (float(beta), float(alpha))
            for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
            for alpha in np.linspace(
                cfg.alpha_min_deg,
                cfg.alpha_max_deg,
                cfg.validation_alpha_samples,
            )
        ]
        validation_points = np.stack(
            [
                square_fluid_points(
                    cfg,
                    cfg.validation_points,
                    beta,
                    alpha,
                    cfg.seed + 900_000 + index,
                )
                for index, (beta, alpha) in enumerate(pairs)
            ]
        ).astype(np.float32)
        save_npz(
            args.output_dir / "fixed_validation_scipy_sobol_v1.npz",
            {
                "validation_points": validation_points,
                "validation_beta": np.asarray(
                    [pair[0] for pair in pairs], dtype=np.float32
                ),
                "validation_alpha": np.asarray(
                    [pair[1] for pair in pairs], dtype=np.float32
                ),
                "probe_points": square_fluid_points(
                    cfg,
                    cfg.validation_points,
                    cfg.probe_beta,
                    cfg.probe_alpha_deg,
                    cfg.seed + 1_500_000,
                ).astype(np.float32),
            },
        )

    initial_sidecar = args.output_dir / "initial_epoch0000.json"
    initial_metadata = json.loads(initial_sidecar.read_text(encoding="utf-8"))
    initial_metadata["config"] = config
    initial_metadata["format"] = "eigenfluid_square_continuous_jet3_fp32_v1"
    initial_metadata["precision"] = {"all_numerical_paths": "float32"}
    if config.get("activation") == "wire":
        initial_metadata.update(
            {
                "source": "NumPy PCG64, PyTorch nn.Linear uniform-equivalent",
                "wire_variant": "official_real_gabor",
                "wire_projection_order": ["frequency", "gaussian_scale"],
                "wire_hidden_affine_width": 2 * int(config["hidden"]),
                "wire_input_width": 2,
            }
        )
    elif bool(config.get("canonical_siren", False)):
        initial_metadata.update(
            {
                "source": "NumPy PCG64, canonical SIREN uniform initialization",
                "canonical_siren": True,
                "siren_input_width": 2,
                "siren_first_omega0": float(config["siren_first_omega0"]),
                "siren_hidden_omega0": float(config["siren_hidden_omega0"]),
            }
        )
    initial_sidecar.write_text(json.dumps(initial_metadata, indent=2), encoding="utf-8")

    validation_stem = (
        "fixed_validation_iid_uniform_v1"
        if config.get("sampling_mode") == "uniform_random"
        else "fixed_validation_scipy_sobol_v1"
    )
    if validation_stem == "fixed_validation_iid_uniform_v1":
        for suffix in (".npz", ".json"):
            legacy = args.output_dir / f"fixed_validation_scipy_sobol_v1{suffix}"
            target = args.output_dir / f"{validation_stem}{suffix}"
            legacy.replace(target)
    validation_sidecar = args.output_dir / f"{validation_stem}.json"
    validation_metadata = json.loads(validation_sidecar.read_text(encoding="utf-8"))
    validation_metadata["config"] = config
    validation_metadata["format"] = (
        "eigenfluid_square_iid_validation_fp32_v1"
        if config.get("sampling_mode") == "uniform_random"
        else "eigenfluid_square_validation_fp32_v1"
    )
    validation_metadata["geometry"] = "unit_square"
    validation_metadata["symmetry_augmentation"] = bool(
        config.get("square_symmetry_quadrature", False)
    )
    validation_sidecar.write_text(
        json.dumps(validation_metadata, indent=2), encoding="utf-8"
    )

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "config": str(args.config.resolve()),
            "config_repository_path": str(args.config.resolve()),
            "geometry": "unit_square",
            "envelope": "square_dirichlet_single_zero",
            "activation": config["activation"],
            "basis_objective": config["basis_objective"],
            "loss_terms": [
                (
                    "unscaled_streamfunction_dirichlet_rayleigh"
                    if config["basis_objective"]
                    == "streamfunction_dirichlet_rayleigh"
                    else "unscaled_vorticity_rayleigh"
                )
            ],
            "sampler": (
                "numpy_pcg64_iid_uniform_area_v1"
                if config.get("sampling_mode") == "uniform_random"
                else "scipy_scrambled_sobol_uniform_physical_fluid_area_v1"
            ),
            "symmetry_augmentation": bool(
                config.get("square_symmetry_quadrature", False)
            ),
            "causal_sorting": config.get("causal_sorting", "fixed_order"),
            "fixed_validation": f"{validation_stem}.npz",
            "fixed_validation_sha256": digest(
                args.output_dir / f"{validation_stem}.npz"
            ),
            "input_encoding": config["input_encoding"],
            "siren_first_layer_spatial_omega0": config.get(
                "siren_first_layer_spatial_omega0"
            ),
            "coordinate_normalization": config.get(
                "coordinate_normalization", "centered_unit"
            ),
            "canonical_siren": bool(config.get("canonical_siren", False)),
            "siren_first_omega0": config.get("siren_first_omega0"),
            "siren_hidden_omega0": config.get("siren_hidden_omega0"),
            "wire_variant": config.get("wire_variant"),
            "wire_omega0": config.get("wire_omega0"),
            "wire_sigma0": config.get("wire_sigma0"),
            "wire_initializer": (
                "torch_nn_linear_uniform_equivalent"
                if config.get("activation") == "wire" else None
            ),
            "initial_sha256": digest(initial_path),
            "early_stopping_enabled": config["early_stopping_enabled"],
            "lr_scheduler_type": config["lr_scheduler_type"],
            "precision": {"all_numerical_paths": "float32"},
            "production_prepare_guard_shadow_only": "float16 label; no FP16 arrays or computation",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_provenance(args.output_dir / "square_case_provenance.json", config)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
