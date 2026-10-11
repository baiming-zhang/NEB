"""Create the continuous-Jet epoch-0 model and validation quadrature."""

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

from neb2d.basisnet.metaball2d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
    fluid_points,
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
    objective = str(config.get("basis_objective", ""))
    if objective not in {
        "pure_vorticity_rayleigh",
        "streamfunction_dirichlet_rayleigh",
        "velocity_rayleigh",
    }:
        raise ValueError(f"unsupported basis objective: {objective!r}")
    sampling_mode = str(config.get("sampling_mode", ""))
    if sampling_mode not in {"uniform_fluid_area_sobol", "uniform_random"}:
        raise ValueError("sampling must be uniform Sobol or IID uniform random")
    derivative_order = int(config.get("spatial_derivative_order", -1))
    required_order = {
        "streamfunction_dirichlet_rayleigh": 1,
        "velocity_rayleigh": 2,
        "pure_vorticity_rayleigh": 3,
    }[objective]
    if derivative_order != required_order:
        raise ValueError(
            f"{objective} requires the minimal analytic Jet{required_order} path"
        )
    compute_dtype = str(config.get("mlp_compute_dtype", "")).lower()
    if compute_dtype not in {"float16", "float32"}:
        raise ValueError("mlp_compute_dtype must be float16 or float32")
    for key in (
        "mlp_accumulation_dtype",
        "jet_dtype",
        "parameter_backward_dtype",
        "quadrature_dtype",
        "ritz_dtype",
        "master_weight_dtype",
        "optimizer_dtype",
    ):
        if config.get(key) != "float32":
            raise ValueError(f"{key} must be float32")
    cfg = cfg_namespace(config)
    jet_order = required_order
    rng = np.random.default_rng(cfg.seed)
    arrays: dict[str, np.ndarray] = {}
    input_features = int(
        getattr(
            cfg,
            "input_features",
            6
            if getattr(cfg, "input_encoding", "")
            == "normalized_raw_xy_plus_geometry"
            else 4 * cfg.frequencies + 4,
        )
    )
    first_layer_gain = float(getattr(cfg, "initial_first_layer_gain", 1.0))
    hidden_layer_gain = float(getattr(cfg, "initial_hidden_layer_gain", 1.0))
    output_layer_gain = float(getattr(cfg, "initial_output_layer_gain", 1.0))
    for mode in range(cfg.modes):
        prefix = f"m{mode:03d}"
        parameter_names: list[str] = []
        fan_in = input_features
        for layer in range(cfg.hidden_layers):
            if bool(getattr(cfg, "canonical_siren", False)):
                bound = (
                    1.0 / fan_in
                    if layer == 0
                    else math.sqrt(6.0 / fan_in)
                    / float(cfg.siren_hidden_omega0)
                )
                bias_bound = 1.0 / math.sqrt(fan_in)
            else:
                gain = first_layer_gain if layer == 0 else hidden_layer_gain
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
            math.sqrt(6.0 / fan_in) / float(cfg.siren_hidden_omega0)
            if bool(getattr(cfg, "canonical_siren", False))
            else output_layer_gain / math.sqrt(fan_in)
        )
        bias_bound = 1.0 / math.sqrt(fan_in)
        weight_name = f"{prefix}.output.weight"
        bias_name = f"{prefix}.output.bias"
        arrays[weight_name] = rng.uniform(
            -bound, bound, (1, fan_in)
        ).astype(np.float32)
        arrays[bias_name] = rng.uniform(-bias_bound, bias_bound, 1).astype(np.float32)
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
                "format": f"eigenfluid_metaball2d_continuous_jet{jet_order}_v1",
                "source": "NumPy PCG64 deterministic initialization",
                "epoch": 0,
                "best_validation": float("inf"),
                "config": config,
                "training_uses_pytorch": False,
                "automatic_differentiation_used": False,
                "finite_difference_used": False,
                "spatial_ad": (
                    "analytic_value_gradient_xy_jet1"
                    if jet_order == 1
                    else (
                        "analytic_value_gradient_compact_hessian_xy_jet2"
                        if jet_order == 2
                        else "analytic_compact_symmetric_xy_jet3"
                    )
                ),
                "parameter_backward": f"handwritten_reverse_jet{jet_order}",
                "loss_terms": [
                    {
                        "streamfunction_dirichlet_rayleigh": (
                            "unscaled_streamfunction_dirichlet_rayleigh"
                        ),
                        "velocity_rayleigh": "unscaled_velocity_rayleigh",
                        "pure_vorticity_rayleigh": "unscaled_vorticity_rayleigh",
                    }[objective]
                ],
                "precision": {
                    "neural_network_forward": compute_dtype,
                    "all_other_numerical_paths": "float32"
                },
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

    if getattr(cfg, "geometry_mode", "") != "fixed_metaball_topology":
        raise ValueError("this initializer is isolated to fixed Metaball topology cases")
    pairs = [(0.0, 0.0)]
    validation_points = np.stack(
        [
            fluid_points(
                cfg,
                cfg.validation_points,
                beta,
                alpha,
                cfg.seed + 900_000 + index,
            ).astype(np.float32)
            for index, (beta, alpha) in enumerate(pairs)
        ]
    )
    validation_stem = (
        "fixed_validation_iid_uniform_v1"
        if sampling_mode == "uniform_random"
        else "fixed_validation_scipy_sobol_v1"
    )
    validation = args.output_dir / f"{validation_stem}.npz"
    save_npz(
        validation,
        {
            "validation_points": validation_points,
            "validation_beta": np.asarray(
                [pair[0] for pair in pairs], dtype=np.float32
            ),
            "validation_alpha": np.asarray(
                [pair[1] for pair in pairs], dtype=np.float32
            ),
            "probe_points": fluid_points(
                cfg,
                cfg.validation_points,
                cfg.probe_beta,
                cfg.probe_alpha_deg,
                cfg.seed + 1_500_000,
            ).astype(np.float32),
        },
    )
    # The generated NPZ packs are intentionally ignored by Git.  Keep the
    # small sidecars versioned so a user can verify or regenerate the exact
    # inputs without downloading model files.
    validation.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "eigenfluid_metaball2d_continuous_validation_v1",
                "config": config,
                "validation_shape_count": len(pairs),
                "validation_points_per_shape": cfg.validation_points,
                "validation_pairs": [
                    {"beta": beta, "alpha_deg": alpha}
                    for beta, alpha in pairs
                ],
                "probe_beta": cfg.probe_beta,
                "probe_alpha_deg": cfg.probe_alpha_deg,
                "sampler": (
                    "numpy_pcg64_iid_uniform_physical_fluid_area_v1"
                    if sampling_mode == "uniform_random"
                    else "scipy_scrambled_sobol_uniform_physical_fluid_area_v1"
                ),
                "dtype": "float32",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "config": str(args.config.resolve()),
        "config_repository_path": str(args.config.resolve()),
        "initializer": "numpy_pcg64_v1",
        "initial_first_layer_gain": first_layer_gain,
        "initial_hidden_layer_gain": hidden_layer_gain,
        "initial_output_layer_gain": output_layer_gain,
        "sampler": (
            "numpy_pcg64_iid_uniform_physical_fluid_area_v1"
            if sampling_mode == "uniform_random"
            else "scipy_scrambled_sobol_uniform_physical_fluid_area_v1"
        ),
        "quadrature_measure": "physical_fluid_area",
        "geometry": "fixed_metaball_topology",
        "subcase": config["subcase"],
        "outer_tau": config["outer_tau"],
        "envelope": config["envelope_mode"],
        "boundary_condition": config.get("boundary_condition"),
        "gauge_fixing": config.get("gauge_fixing"),
        "harmonic_mode": config.get("harmonic_mode"),
        "basis_objective": objective,
        "loss_terms": [
            {
                "streamfunction_dirichlet_rayleigh": (
                    "unscaled_streamfunction_dirichlet_rayleigh"
                ),
                "velocity_rayleigh": "unscaled_velocity_rayleigh",
                "pure_vorticity_rayleigh": "unscaled_vorticity_rayleigh",
            }[objective]
        ],
        "finite_difference_used": False,
        "spatial_ad": (
            "analytic_value_gradient_xy_jet1"
            if jet_order == 1
            else (
                "analytic_value_gradient_compact_hessian_xy_jet2"
                if jet_order == 2
                else "analytic_compact_symmetric_xy_jet3"
            )
        ),
        "parameter_backward": f"handwritten_reverse_jet{jet_order}",
        "precision": {
            "neural_network_forward": compute_dtype,
            "all_other_numerical_paths": "float32"
        },
        "initial_checkpoint": initial.name,
        "initial_sha256": digest(initial),
        "fixed_validation": validation.name,
        "fixed_validation_sha256": digest(validation),
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
