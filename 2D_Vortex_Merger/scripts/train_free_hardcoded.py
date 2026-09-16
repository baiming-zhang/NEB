"""Train a continuous-Jet Rayleigh basis with causal residual peeling.

This entry point deliberately supports one mathematical objective only.  It
uses the explicit CuPy kernels and handwritten parameter reverse pass in
``hardcoded_backend``; PyTorch and automatic differentiation are forbidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    peel_step,
)


SUPPORTED_OBJECTIVES = {"potential_rayleigh"}
JET_ORDER_BY_OBJECTIVE = {
    "potential_rayleigh": 1,
}
FORBIDDEN_CONFIG_KEYS = {
    "rayleigh_scale",
    "hybrid_velocity_weight",
    "hybrid_vorticity_weight",
    "hybrid_velocity_rayleigh_scale",
    "hybrid_vorticity_rayleigh_scale",
    "hybrid_vorticity_mass_length_scale_chords",
    "hybrid_vorticity_difference_chords",
    "hybrid_vorticity_anchor_points",
    "hybrid_reorthogonalization_passes",
    "vorticity_regularization_weight",
    "small_vorticity_regularization_weight",
    "large_vorticity_preservation_weight",
    "peeled_subspace_vorticity_regularization_weight",
    "small_vorticity_scale_chords",
    "small_vorticity_anchor_points",
    "global_fraction",
    "airfoil_fraction",
    "wake_fraction",
    "airfoil_band_chords",
    "wake_length_chords",
    "wake_half_height_chords",
}
FP32_CONFIG_KEYS = (
    "mlp_accumulation_dtype",
    "jet_dtype",
    "parameter_backward_dtype",
    "quadrature_dtype",
    "ritz_dtype",
    "master_weight_dtype",
    "optimizer_dtype",
)
HISTORY_KEYS = [
    "epoch",
    "seconds",
    "training_elapsed_seconds",
    "sample_pack_load_seconds",
    "sample_pack_upload_seconds",
    "sample_pack_startup_h2d_bytes",
    "sample_pack_startup_h2d_transfers",
    "sample_pack_device",
    "wall_time_iso",
    "wall_time_unix_ns",
    "beta",
    "alpha_deg",
    "train_loss",
    "train_vorticity_rayleigh",
    "train_streamfunction_rayleigh",
    "train_velocity_rayleigh",
    "train_mean_energy",
    "train_min_energy",
    "train_max_energy",
    "train_mean_projection_ratio",
    "train_orth_error",
    "train_rank",
    "train_peeled_rank",
    "validation_loss",
    "validation_vorticity_rayleigh",
    "validation_streamfunction_rayleigh",
    "validation_velocity_rayleigh",
    "validation_loss_p95",
    "validation_loss_max",
    "validation_mean_energy",
    "validation_min_energy",
    "validation_max_energy",
    "validation_mean_projection_ratio",
    "validation_orth_error",
    "validation_min_rank",
    "validation_peeled_rank",
    "probe_loss",
    "probe_vorticity_rayleigh",
    "probe_streamfunction_rayleigh",
    "probe_velocity_rayleigh",
    "probe_orth_error",
    "probe_rank",
    "probe_peeled_rank",
    "learning_rate",
    "best_validation_epoch",
    "early_stop_counter_epochs",
    "lr_plateau_counter_epochs",
    "requested_modes",
    "spatial_jet_order_used",
    "causal_sort_device",
    "diagnostic_host_transfers_per_step",
    "optimizer_device",
    "optimizer_host_transfers_per_step",
    "input_host_to_device_transfers_per_step",
    "train_gradient_norm_min",
    "train_gradient_norm_max",
    "train_min_relative_residual_mass",
    "train_max_post_peel_overlap",
    "train_peeling_passes_used",
    "train_rejected_modes",
    "validation_min_relative_residual_mass",
    "validation_max_post_peel_overlap",
]


def append_log(run_dir: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (run_dir / "hardcoded_train.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


POINT_SAMPLE_KEYS = frozenset(
    {"train_points", "validation_points", "probe_points"}
)


def load_sample_pack(path: Path | None) -> dict[str, np.ndarray] | None:
    """Load and close an immutable PCG64 sample pack on the host."""
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def upload_sample_pack_points(
    *packs: dict[str, np.ndarray] | None,
) -> tuple[
    tuple[dict[str, np.ndarray | cp.ndarray] | None, ...],
    cp.ndarray | None,
    dict[str, int | str],
]:
    """Upload all train/validation/probe points in one contiguous H2D copy.

    Geometry parameters deliberately stay in their small NumPy arrays and are
    passed as kernel-launch scalars.  Only the bulk point tensors need device
    residency.  The returned storage object owns the single allocation; each
    pack receives a reshaped CuPy view into it.
    """
    specifications: list[tuple[int, str, tuple[int, ...], int]] = []
    total_values = 0
    for pack_index, pack in enumerate(packs):
        if pack is None:
            continue
        for name in POINT_SAMPLE_KEYS:
            if name not in pack:
                continue
            array = pack[name]
            if array.dtype != np.float32:
                raise TypeError(
                    f"{name} must be locked float32, received {array.dtype}"
                )
            specifications.append(
                (pack_index, name, tuple(array.shape), int(array.size))
            )
            total_values += int(array.size)

    copied_packs: list[dict[str, np.ndarray | cp.ndarray] | None] = [
        None if pack is None else dict(pack) for pack in packs
    ]
    if total_values == 0:
        return (
            tuple(copied_packs),
            None,
            {
                "sample_pack_device": "host_per_step",
                "sample_pack_startup_h2d_transfers": 0,
                "sample_pack_startup_h2d_bytes": 0,
            },
        )

    host_storage = np.empty(total_values, dtype=np.float32)
    offset = 0
    slices: list[tuple[int, str, tuple[int, ...], slice]] = []
    for pack_index, name, shape, size in specifications:
        pack = packs[pack_index]
        if pack is None:
            raise RuntimeError("sample-pack specification lost its owner")
        destination = slice(offset, offset + size)
        host_storage[destination] = np.asarray(
            pack[name], dtype=np.float32
        ).reshape(-1)
        slices.append((pack_index, name, shape, destination))
        offset += size

    # Exactly one bulk H2D operation.  Synchronizing here makes the startup
    # upload timer honest and keeps it outside ``training_elapsed_seconds``.
    device_storage = cp.asarray(host_storage)
    cp.cuda.get_current_stream().synchronize()
    for pack_index, name, shape, source in slices:
        copied_pack = copied_packs[pack_index]
        if copied_pack is None:
            raise RuntimeError("sample-pack device view lost its owner")
        copied_pack[name] = device_storage[source].reshape(shape)
    return (
        tuple(copied_packs),
        device_storage,
        {
            "sample_pack_device": "gpu_resident",
            "sample_pack_startup_h2d_transfers": 1,
            "sample_pack_startup_h2d_bytes": int(device_storage.nbytes),
        },
    )


def load_neutral(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint or sidecar is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, arrays


def save_neutral(
    path: Path,
    model: HardcodedModel,
    config: dict[str, Any],
    epoch: int,
    best_validation: float,
    training_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **model.arrays())
    os.replace(temporary, path)
    objective = str(config.get("basis_objective", ""))
    jet_order = JET_ORDER_BY_OBJECTIVE[objective]
    metadata = {
        "format": f"eigenfluid_vortex_merger_free_rawxy_jet{jet_order}_v1",
        "epoch": epoch,
        "best_validation": best_validation,
        "config": config,
        "training_uses_pytorch": False,
        "automatic_differentiation_used": False,
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
        "training_state": training_state,
    }
    metadata_path = path.with_suffix(".json")
    metadata_temporary = metadata_path.with_suffix(".json.tmp")
    metadata_temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(metadata_temporary, metadata_path)


def save_history(run_dir: Path, history: list[dict[str, Any]]) -> None:
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    target = data_dir / "hardcoded_training_history.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temporary, target)
    with (data_dir / "hardcoded_training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_KEYS)
        writer.writeheader()
        writer.writerows(
            {key: row.get(key, "") for key in HISTORY_KEYS} for row in history
        )


def validate_config(config: dict[str, Any]) -> None:
    objective = str(config.get("basis_objective", ""))
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(
            f"unsupported basis_objective={objective!r}; expected one of "
            f"{sorted(SUPPORTED_OBJECTIVES)}"
        )
    forbidden = sorted(FORBIDDEN_CONFIG_KEYS.intersection(config))
    if forbidden:
        raise ValueError(f"forbidden auxiliary/scaled objective fields: {forbidden}")
    if config.get("sampling_mode") not in {
        "uniform_fluid_area_sobol",
        "uniform_random",
    }:
        raise ValueError("sampling must be uniform Sobol or IID uniform random")
    if config.get("quadrature_measure") != "physical_fluid_area":
        raise ValueError("quadrature measure must be physical fluid area")
    boundary_pair = (config.get("geometry_mode"), config.get("envelope_mode"))
    if boundary_pair != ("free_box", "free_identity"):
        raise ValueError(
            "vortex-merger training requires free_box + free_identity"
        )
    if config.get("input_encoding") != "raw_xy":
        raise ValueError("free_box requires unencoded raw_xy input")
    if int(config.get("input_features", -1)) != 2:
        raise ValueError("raw_xy input_features must equal 2")
    derivative_order = int(config.get("spatial_derivative_order", -1))
    required_order = JET_ORDER_BY_OBJECTIVE[objective]
    if derivative_order != required_order:
        raise ValueError(
            f"{objective} requires the minimal analytic Jet{required_order} path"
        )
    if int(config.get("reorthogonalization_passes", -1)) != 2:
        raise ValueError("free-boundary Potential-Rayleigh requires two MGS passes")
    if str(config.get("mlp_compute_dtype", "")).lower() not in {
        "float16",
        "float32",
    }:
        raise ValueError("mlp_compute_dtype must be float16 or float32")
    wrong_precision = {
        key: config.get(key) for key in FP32_CONFIG_KEYS
        if str(config.get(key, "")).lower() != "float32"
    }
    if wrong_precision:
        raise ValueError(f"all non-network numerical paths must be float32: {wrong_precision}")
    if (
        str(config.get("mlp_compute_dtype", "")).lower() == "float16"
        and bool(config.get("fp32_boundary_layers", True))
    ):
        raise ValueError("all neural-network layers, including first/output, must use FP16 shadows")
    if int(config.get("checkpoint_every", 0)) != 100:
        raise ValueError("hardcoded_latest checkpoint cadence must be 100 epochs")
    archive_every = int(config.get("archive_checkpoint_every", 0))
    checkpoint_every = int(config["checkpoint_every"])
    if archive_every < checkpoint_every or archive_every % checkpoint_every != 0:
        raise ValueError(
            "permanent checkpoint cadence must be a positive multiple of "
            "hardcoded_latest cadence"
        )


def diagnostic_rayleigh_key(item: dict[str, Any]) -> str:
    keys = [
        key for key in (
            "vorticity_rayleigh", "streamfunction_rayleigh", "velocity_rayleigh"
        )
        if key in item
    ]
    if len(keys) != 1:
        raise KeyError(f"expected exactly one Rayleigh diagnostic, got {keys}")
    return keys[0]


def require_pure_diagnostic(item: dict[str, Any]) -> None:
    required = {
        "loss",
        "mean_energy",
        "min_energy",
        "max_energy",
        "orth_error",
        "peeled_rank",
        "mode_energies",
    }
    missing = sorted(required.difference(item))
    if missing:
        raise KeyError(f"backend pure-vorticity diagnostics missing {missing}")
    key = diagnostic_rayleigh_key(item)
    if not math.isfinite(float(item["loss"])):
        raise FloatingPointError("non-finite Rayleigh loss")
    if float(item["loss"]) != float(item[key]):
        raise RuntimeError("loss must equal the unscaled Rayleigh mean")


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    for item in items:
        require_pure_diagnostic(item)
    losses = np.asarray([item["loss"] for item in items], dtype=np.float32)
    keys = {diagnostic_rayleigh_key(item) for item in items}
    if len(keys) != 1:
        raise RuntimeError(f"mixed validation objectives: {sorted(keys)}")
    rayleigh_key = keys.pop()
    ranks = [int(item["peeled_rank"]) for item in items]
    result = {
        "loss": float(np.mean(losses)),
        "loss_p95": float(np.percentile(losses, 95.0)),
        "loss_max": float(np.max(losses)),
        "mean_energy": float(np.mean([item["mean_energy"] for item in items])),
        "min_energy": float(np.min([item["min_energy"] for item in items])),
        "max_energy": float(np.max([item["max_energy"] for item in items])),
        "mean_projection_ratio": float(np.mean([
            item.get("mean_projection_ratio", float("nan")) for item in items
        ])),
        "orth_error": float(np.max([item["orth_error"] for item in items])),
        "min_rank": min(ranks),
        "peeled_rank": min(ranks),
        "requested_modes": int(items[0].get("requested_modes", len(items[0]["mode_energies"]))),
        "min_relative_residual_mass": float(np.min([
            item.get("min_relative_residual_mass", float("nan")) for item in items
        ])),
        "max_post_peel_overlap": float(np.max([
            item.get("max_post_peel_overlap", item["orth_error"]) for item in items
        ])),
    }
    result[rayleigh_key] = float(np.mean(losses))
    return result


def validate(
    cfg: Any,
    model: HardcodedModel,
    sample_pack: dict[str, np.ndarray | cp.ndarray] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples
        )
    ]
    for index, (beta, alpha) in enumerate(pairs):
        if sample_pack is None:
            points = fluid_points(
                cfg, cfg.validation_points, beta, alpha, cfg.seed + 900_000 + index
            )
        else:
            beta = float(sample_pack["validation_beta"][index])
            alpha = float(sample_pack["validation_alpha"][index])
            points = sample_pack["validation_points"][index]
        item = peel_step(
            cfg, model, points, beta, alpha, False,
            reference_model=None, full_diagnostics=True,
        )
        require_pure_diagnostic(item)
        details.append(item)
    return aggregate(details), details


def training_state(
    learning_rate: float,
    best_epoch: int,
    reference_loss: float,
    reference_epoch: int,
    reductions: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "current_learning_rate": learning_rate,
        "best_validation_epoch": best_epoch,
        "lr_reference_loss": reference_loss,
        "lr_reference_epoch": reference_epoch,
        "lr_reductions": reductions,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--sample-pack", type=Path)
    parser.add_argument("--validation-pack", type=Path)
    parser.add_argument("--self-test-epochs", type=int, default=0)
    parser.add_argument("--self-test-points", type=int, default=256)
    parser.add_argument(
        "--invalid-candidate-max-resamples",
        type=int,
        default=0,
        help=(
            "For velocity-Rayleigh training only, deterministically resample "
            "the same epoch after a single-peel candidate is atomically "
            "rejected before any optimizer update."
        ),
    )
    parser.add_argument(
        "--training-elapsed-floor",
        type=float,
        default=0.0,
        help="Lower bound for cumulative pure-training seconds after recovery.",
    )
    parser.add_argument(
        "--extra-archive-epochs",
        type=int,
        nargs="*",
        default=[],
        help="Additional exact epochs that must receive named archive checkpoints.",
    )
    parser.add_argument(
        "--peeling-relative-mass-floor",
        type=float,
        help=(
            "Explicitly override the resumed checkpoint's single-peel relative-"
            "mass floor. The effective value is persisted in all new checkpoints."
        ),
    )
    args = parser.parse_args()
    if "torch" in sys.modules:
        raise RuntimeError("the hardcoded trainer must not import PyTorch")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    (args.run_dir / "data").mkdir(exist_ok=True)
    metadata, arrays = load_neutral(args.resume)
    config = dict(metadata["config"])
    resumed_relative_mass_floor = config.get("peeling_relative_mass_floor")
    if args.peeling_relative_mass_floor is not None:
        config["peeling_relative_mass_floor"] = float(
            args.peeling_relative_mass_floor
        )
    if args.epochs is not None:
        if args.epochs <= int(metadata["epoch"]):
            raise ValueError("--epochs must exceed the resume epoch")
        config["epochs"] = args.epochs
    validate_config(config)
    objective = str(config["basis_objective"])
    rayleigh_key = {
        "potential_rayleigh": "streamfunction_rayleigh",
    }[objective]
    rayleigh_label = {
        "streamfunction_rayleigh": "psi_R",
        "vorticity_rayleigh": "omega_R",
        "velocity_rayleigh": "velocity_R",
    }[rayleigh_key]
    spatial_ad = str(metadata.get("spatial_ad", ""))
    expected_jet = f"jet{JET_ORDER_BY_OBJECTIVE[objective]}"
    if expected_jet not in spatial_ad.lower():
        raise ValueError(
            f"refusing {spatial_ad!r} checkpoint metadata for {objective}; "
            f"expected {expected_jet}"
        )

    sample_pack_load_started = time.perf_counter()
    sample_pack_host = load_sample_pack(args.sample_pack)
    validation_pack_host = load_sample_pack(args.validation_pack)
    sample_pack_load_seconds = time.perf_counter() - sample_pack_load_started
    sample_pack_upload_started = time.perf_counter()
    (
        (sample_pack, validation_pack),
        sample_pack_device_storage_owner,
        sample_pack_audit,
    ) = upload_sample_pack_points(sample_pack_host, validation_pack_host)
    sample_pack_upload_seconds = (
        time.perf_counter() - sample_pack_upload_started
    )
    del sample_pack_host, validation_pack_host
    if sample_pack_device_storage_owner is not None:
        expected_bytes = int(
            sample_pack_audit["sample_pack_startup_h2d_bytes"]
        )
        if int(sample_pack_device_storage_owner.nbytes) != expected_bytes:
            raise RuntimeError("sample-pack device allocation audit mismatch")
    cfg = cfg_namespace(config)
    model = HardcodedModel(cfg, arrays)
    start_epoch = int(metadata["epoch"]) + 1
    if args.stop_after_epoch is not None:
        if not start_epoch <= args.stop_after_epoch <= int(cfg.epochs):
            raise ValueError(
                "--stop-after-epoch must be between the resume epoch and "
                "the configured final epoch"
            )
        if args.self_test_epochs > 0:
            raise ValueError(
                "--stop-after-epoch cannot be combined with --self-test-epochs"
            )
    stop_epoch = (
        start_epoch + args.self_test_epochs - 1
        if args.self_test_epochs > 0
        else (
            int(args.stop_after_epoch)
            if args.stop_after_epoch is not None
            else int(cfg.epochs)
        )
    )
    history_path = args.run_dir / "data" / "hardcoded_training_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.exists() else []
    )
    best_validation = float(metadata.get("best_validation", float("inf")))
    restored = metadata.get("training_state", {})
    current_lr = float(restored.get("current_learning_rate", cfg.learning_rate))
    cfg.learning_rate = current_lr
    config["learning_rate"] = current_lr
    scheduler_type = str(
        config.get("lr_scheduler_type", "validation_plateau")
    ).lower()
    if scheduler_type not in {"constant", "validation_plateau", "cosine"}:
        raise ValueError(f"unsupported lr_scheduler_type={scheduler_type!r}")
    early_stopping_enabled = bool(
        config.get("early_stopping_enabled", True)
    )
    best_epoch = int(restored.get("best_validation_epoch", 0))
    lr_reference_loss = float(restored.get("lr_reference_loss", best_validation))
    lr_reference_epoch = int(restored.get("lr_reference_epoch", 0))
    lr_reductions = int(restored.get("lr_reductions", 0))

    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    sampling_mode = str(getattr(cfg, "sampling_mode", ""))
    sampling_sequence = (
        "IID PCG64 uniform random"
        if sampling_mode == "uniform_random"
        else "scrambled Sobol"
    )
    protocol = {
        "algorithm": f"{causal_sorting}_residual_peeling_{objective}",
        "backend": "explicit_cupy_cuda_kernels",
        "basis_objective": objective,
        "loss_terms": [
            {
                "name": rayleigh_key,
                "formula": (
                    "integral(|grad(psi)|^2)/integral(psi^2)"
                    if rayleigh_key == "streamfunction_rayleigh"
                    else (
                        "integral(|grad(u)|_F^2)/integral(|u|^2), "
                        "u=(psi_y,-psi_x)"
                        if rayleigh_key == "velocity_rayleigh"
                        else "integral(|grad(omega)|^2)/integral(omega^2)"
                    )
                ),
                "coefficient": 1.0,
                "scaled": False,
            }
        ],
        "auxiliary_loss_terms": [],
        "finite_difference_used": False,
        "input_coordinates": "physical raw x,y; no positional/Fourier encoding",
        "spatial_derivatives": (
            "analytic Jet1 (value,x,y) in x,y"
            if objective == "potential_rayleigh"
            else (
                "compact analytic Jet2 (value, gradient, Hessian) in x,y"
                if objective == "velocity_rayleigh"
                else "compact symmetric analytic Jet3 in x,y"
            )
        ),
        "parameter_backward": (
            "handwritten reverse Jet1; no tensor autograd graph"
            if objective == "potential_rayleigh"
            else (
                "handwritten reverse Jet2; no tensor autograd graph"
                if objective == "velocity_rayleigh"
                else "handwritten reverse Jet3; no tensor autograd graph"
            )
        ),
        "causal_sort_device": (
            "gpu" if causal_sorting == "geometry_rayleigh" else "not_applicable"
        ),
        "optimizer_device": "gpu_batched_all_modes",
        "training_step_host_diagnostic_transfers": 1,
        "training_step_optimizer_host_transfers": 0,
        "training_step_input_host_to_device_transfers": (
            0
            if sample_pack_audit["sample_pack_device"] == "gpu_resident"
            else 1
        ),
        "diagnostic_host_transfers_per_step": 1,
        "optimizer_host_transfers_per_step": 0,
        "input_host_to_device_transfers_per_step": (
            0
            if sample_pack_audit["sample_pack_device"] == "gpu_resident"
            else 1
        ),
        "sample_pack_device": sample_pack_audit["sample_pack_device"],
        "sample_pack_startup_h2d_transfers": sample_pack_audit[
            "sample_pack_startup_h2d_transfers"
        ],
        "sample_pack_startup_h2d_bytes": sample_pack_audit[
            "sample_pack_startup_h2d_bytes"
        ],
        "sample_pack_load_seconds": sample_pack_load_seconds,
        "sample_pack_upload_seconds": sample_pack_upload_seconds,
        "sample_pack_timing_excluded_from_training_elapsed_seconds": True,
        "sample_pack_residency_scope": (
            "train/validation/probe point tensors; beta/alpha remain locked "
            "NumPy host launch scalars and incur no bulk per-step copy"
        ),
        "coordinate_jets_reused_across_mode_blocks": True,
        "pytorch_imported_by_trainer": False,
        "automatic_differentiation_used": False,
        "precision": {
            "neural_network_forward": (
                "float16 shadows with float32 accumulation"
                if str(cfg.mlp_compute_dtype).lower() == "float16"
                else "float32"
            ),
            "spatial_jet": "float32",
            "parameter_reverse": "float32",
            "quadrature": "float32",
            "ritz_and_reorthogonalization": "float32",
            "master_weights": "float32",
            "adam_state": "float32",
        },
        "sampling": {
            "sequence": sampling_sequence,
            "measure": "uniform physical fluid area",
            "points_per_epoch": cfg.points_per_step,
            "importance_mixture": False,
            "symmetry_augmentation": bool(
                getattr(cfg, "square_symmetry_quadrature", False)
            ),
        },
        "fixed_mode_order": causal_sorting == "fixed_order",
        "mode_sorting_used": causal_sorting == "geometry_rayleigh",
        "mode_sorting_criterion": (
            f"detached unprojected {rayleigh_label} quotient on the current "
            "geometry and quadrature points"
            if causal_sorting == "geometry_rayleigh"
            else None
        ),
        "previous_modes_detached": True,
        "peeling_passes": int(getattr(cfg, "reorthogonalization_passes", 1)),
        "invalid_candidate_policy": (
            (
                "reject before optimizer transaction; deterministically "
                "resample the same epoch"
                if args.invalid_candidate_max_resamples > 0
                else "reject before optimizer transaction"
            )
            if objective == "velocity_rayleigh" else None
        ),
        "invalid_candidate_max_resamples": args.invalid_candidate_max_resamples,
        "peeling_relative_mass_floor": getattr(
            cfg, "peeling_relative_mass_floor", None
        ),
        "peeling_overlap_tolerance": getattr(
            cfg, "peeling_overlap_tolerance", None
        ),
        "rank_metric": "peeled basis columns only; raw rank excluded",
        "gradient_clipping": {
            "maximum_global_norm": cfg.grad_clip,
            "role": "finite-arithmetic safeguard only; not a loss term",
        },
        "checkpoint_every": cfg.checkpoint_every,
        "archive_checkpoint_every": cfg.archive_checkpoint_every,
        "extra_archive_epochs": sorted(set(args.extra_archive_epochs)),
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopping_patience_epochs": (
            cfg.early_stopping_patience_epochs
            if early_stopping_enabled else None
        ),
        "lr_scheduler": {
            "type": scheduler_type,
            "patience_epochs": cfg.lr_scheduler_patience_epochs,
            "factor": cfg.lr_scheduler_factor,
            "minimum_learning_rate": cfg.min_learning_rate,
        },
        "reference_sample_pack": (
            str(args.sample_pack.resolve()) if args.sample_pack else None
        ),
        "fixed_validation_pack": (
            str(args.validation_pack.resolve()) if args.validation_pack else None
        ),
    }
    (args.run_dir / "hardcoded_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    append_log(
        args.run_dir,
        f"start continuous-{expected_jet.capitalize()} {objective} "
        f"no-PyTorch device=cuda "
        f"modes={cfg.modes} points={cfg.points_per_step} "
        f"resume_epoch={metadata['epoch']} lr={current_lr:.6e}",
    )
    append_log(
        args.run_dir,
        "sample pack startup "
        f"device={sample_pack_audit['sample_pack_device']} "
        "h2d_transfers="
        f"{sample_pack_audit['sample_pack_startup_h2d_transfers']} "
        f"bytes={sample_pack_audit['sample_pack_startup_h2d_bytes']} "
        f"load={sample_pack_load_seconds:.6f}s "
        f"upload={sample_pack_upload_seconds:.6f}s "
        "(excluded from training_elapsed_seconds)",
    )
    if args.peeling_relative_mass_floor is not None:
        append_log(
            args.run_dir,
            "explicit peeling relative-mass floor override "
            f"{float(resumed_relative_mass_floor):.8e} -> "
            f"{float(cfg.peeling_relative_mass_floor):.8e}",
        )

    started = time.perf_counter()
    elapsed_before_resume = 0.0
    if history:
        elapsed_before_resume = float(
            history[-1].get("training_elapsed_seconds", history[-1].get("seconds", 0.0))
        )
    elapsed_before_resume = max(
        elapsed_before_resume, float(args.training_elapsed_floor)
    )
    stopped_early = False
    final_epoch = int(metadata["epoch"])
    for epoch in range(start_epoch, stop_epoch + 1):
        if args.self_test_epochs <= 0 and scheduler_type == "cosine":
            denominator = max(int(cfg.epochs) - 1, 1)
            progress = min(max((epoch - 1) / denominator, 0.0), 1.0)
            current_lr = float(cfg.min_learning_rate) + 0.5 * (
                float(cfg.base_learning_rate) - float(cfg.min_learning_rate)
            ) * (1.0 + math.cos(math.pi * progress))
            cfg.learning_rate = current_lr
            config["learning_rate"] = current_lr
        count = args.self_test_points if args.self_test_epochs > 0 else cfg.points_per_step
        if sample_pack is not None and args.self_test_epochs <= 0:
            index = epoch - 1
            if index >= len(sample_pack["train_beta"]):
                raise IndexError(f"sample pack has no entry for epoch {epoch}")
            beta = float(sample_pack["train_beta"][index])
            alpha = float(sample_pack["train_alpha"][index])
            points = sample_pack["train_points"][index]
        else:
            beta, alpha = continuous_parameters(cfg, epoch)
            points = fluid_points(cfg, count, beta, alpha, cfg.seed + epoch * 1009)
        rejected_attempts: list[dict[str, Any]] = []
        for resample_attempt in range(args.invalid_candidate_max_resamples + 1):
            if resample_attempt > 0:
                if sample_pack is not None:
                    raise RuntimeError(
                        "invalid-candidate resampling is incompatible with a fixed sample pack"
                    )
                resample_seed = (
                    cfg.seed + epoch * 1009 + resample_attempt * 104729
                )
                points = fluid_points(
                    cfg, count, beta, alpha, resample_seed
                )
            else:
                resample_seed = cfg.seed + epoch * 1009
            try:
                diagnostic = peel_step(
                    cfg, model, points, beta, alpha, True,
                    reference_model=None, full_diagnostics=True,
                )
                break
            except FloatingPointError as error:
                message = str(error)
                recoverable = (
                    objective == "velocity_rayleigh"
                    and "no parameters updated" in message
                    and resample_attempt < args.invalid_candidate_max_resamples
                )
                if not recoverable:
                    raise
                rejected_attempts.append(
                    {
                        "attempt": resample_attempt,
                        "seed": int(resample_seed),
                        "error": message,
                    }
                )
                append_log(
                    args.run_dir,
                    f"epoch={epoch:04d} rejected IID quadrature attempt="
                    f"{resample_attempt} seed={resample_seed}: {message}; "
                    "retrying with a new deterministic IID sample",
                )
        require_pure_diagnostic(diagnostic)
        process_elapsed_seconds = time.perf_counter() - started
        epoch_wall_time_ns = time.time_ns()
        row: dict[str, Any] = {
            "epoch": epoch,
            # Keep the legacy per-process timer and additionally persist a
            # monotonic total that remains continuous across checkpoint resumes.
            "seconds": process_elapsed_seconds,
            "training_elapsed_seconds": (
                elapsed_before_resume + process_elapsed_seconds
            ),
            "sample_pack_load_seconds": sample_pack_load_seconds,
            "sample_pack_upload_seconds": sample_pack_upload_seconds,
            "sample_pack_startup_h2d_bytes": sample_pack_audit[
                "sample_pack_startup_h2d_bytes"
            ],
            "sample_pack_startup_h2d_transfers": sample_pack_audit[
                "sample_pack_startup_h2d_transfers"
            ],
            "sample_pack_device": diagnostic.get(
                "sample_pack_device", sample_pack_audit["sample_pack_device"]
            ),
            "wall_time_iso": datetime.fromtimestamp(
                epoch_wall_time_ns / 1.0e9
            ).astimezone().isoformat(),
            "wall_time_unix_ns": epoch_wall_time_ns,
            "beta": beta,
            "alpha_deg": alpha,
            "train_loss": diagnostic["loss"],
            "train_mean_energy": diagnostic["mean_energy"],
            "train_min_energy": diagnostic["min_energy"],
            "train_max_energy": diagnostic["max_energy"],
            "train_mean_projection_ratio": diagnostic.get(
                "mean_projection_ratio", float("nan")
            ),
            "train_orth_error": diagnostic["orth_error"],
            "train_rank": diagnostic["peeled_rank"],
            "train_peeled_rank": diagnostic["peeled_rank"],
            "learning_rate": current_lr,
            "train_causal_order_1based": diagnostic.get(
                "causal_order_1based", list(range(1, int(cfg.modes) + 1))
            ),
            "train_sorting_rayleigh_by_network": diagnostic.get(
                "sorting_rayleigh_by_network", []
            ),
            "spatial_jet_order_used": diagnostic.get(
                "spatial_jet_order_used", int(cfg.spatial_derivative_order)
            ),
            "causal_sort_device": diagnostic.get(
                "causal_sort_device", "host"
            ),
            "diagnostic_host_transfers_per_step": diagnostic.get(
                "diagnostic_host_transfers_per_step", ""
            ),
            "optimizer_device": diagnostic.get(
                "optimizer_device", "gpu_per_mode"
            ),
            "optimizer_host_transfers_per_step": diagnostic.get(
                "optimizer_host_transfers_per_step", ""
            ),
            "input_host_to_device_transfers_per_step": diagnostic.get(
                "input_host_to_device_transfers_per_step", ""
            ),
            "train_gradient_norm_min": diagnostic.get(
                "gradient_norm_min", float("nan")
            ),
            "train_gradient_norm_max": diagnostic.get(
                "gradient_norm_max", float("nan")
            ),
            "train_min_relative_residual_mass": diagnostic.get(
                "min_relative_residual_mass", float("nan")
            ),
            "train_max_post_peel_overlap": diagnostic.get(
                "max_post_peel_overlap", diagnostic["orth_error"]
            ),
            "train_peeling_passes_used": diagnostic.get(
                "peeling_passes_used", ""
            ),
            "train_rejected_modes": diagnostic.get("rejected_modes", ""),
            "train_rejected_quadrature_attempts": rejected_attempts,
        }
        row[f"train_{rayleigh_key}"] = diagnostic[rayleigh_key]
        if args.self_test_epochs > 0:
            append_log(
                args.run_dir,
                f"self-test epoch={epoch:04d} "
                f"{rayleigh_label}={diagnostic['loss']:.6e} "
                f"orth={diagnostic['orth_error']:.2e} "
                f"min_rel_mass={diagnostic.get('min_relative_residual_mass', float('nan')):.2e} "
                f"peeled_rank={diagnostic['peeled_rank']}",
            )
        elif epoch % cfg.validate_every == 0:
            validation_source = validation_pack or sample_pack
            validation, details = validate(cfg, model, validation_source)
            row.update({
                "validation_loss": validation["loss"],
                "validation_loss_p95": validation["loss_p95"],
                "validation_loss_max": validation["loss_max"],
                "validation_mean_energy": validation["mean_energy"],
                "validation_min_energy": validation["min_energy"],
                "validation_max_energy": validation["max_energy"],
                "validation_mean_projection_ratio": validation[
                    "mean_projection_ratio"
                ],
                "validation_orth_error": validation["orth_error"],
                "validation_min_rank": validation["min_rank"],
                "validation_peeled_rank": validation["peeled_rank"],
                "validation_min_relative_residual_mass": validation[
                    "min_relative_residual_mass"
                ],
                "validation_max_post_peel_overlap": validation[
                    "max_post_peel_overlap"
                ],
                "validation_mode_energies": [item["mode_energies"] for item in details],
                "validation_causal_order_1based": [
                    item.get(
                        "causal_order_1based",
                        list(range(1, int(cfg.modes) + 1)),
                    )
                    for item in details
                ],
            })
            row[f"validation_{rayleigh_key}"] = validation[rayleigh_key]
            threshold = cfg.lr_scheduler_relative_threshold
            improved = validation["loss"] < best_validation * (1.0 - threshold)
            if improved:
                best_validation = validation["loss"]
                best_epoch = epoch
                lr_reference_loss = best_validation
                lr_reference_epoch = epoch
                state = training_state(
                    current_lr, best_epoch, lr_reference_loss,
                    lr_reference_epoch, lr_reductions,
                )
                save_neutral(
                    args.run_dir / "checkpoints" / "hardcoded_best_model.npz",
                    model, config, epoch, best_validation, state,
                )
            elif (
                scheduler_type == "validation_plateau"
                and epoch - lr_reference_epoch >= cfg.lr_scheduler_patience_epochs
                and current_lr > cfg.min_learning_rate
            ):
                current_lr = max(
                    cfg.min_learning_rate, current_lr * cfg.lr_scheduler_factor
                )
                cfg.learning_rate = current_lr
                config["learning_rate"] = current_lr
                lr_reference_loss = validation["loss"]
                lr_reference_epoch = epoch
                lr_reductions += 1
                append_log(args.run_dir, f"lr reduced epoch={epoch:04d} lr={current_lr:.6e}")
            row.update({
                "learning_rate": current_lr,
                "best_validation_epoch": best_epoch,
                "early_stop_counter_epochs": epoch - best_epoch,
                "lr_plateau_counter_epochs": epoch - lr_reference_epoch,
            })
            append_log(
                args.run_dir,
                f"epoch={epoch:04d} beta={beta:.5f} alpha={alpha:+.3f} "
                f"train_{rayleigh_label}={diagnostic['loss']:.6e} "
                f"validation_{rayleigh_label}={validation['loss']:.6e} "
                f"val_p95={validation['loss_p95']:.6e} lr={current_lr:.3e} "
                f"orth={validation['orth_error']:.2e} "
                f"min_rel_mass={validation['min_relative_residual_mass']:.2e} "
                f"max_overlap={validation['max_post_peel_overlap']:.2e} "
                f"peeled_rank={validation['peeled_rank']}",
            )
        if (
            args.self_test_epochs <= 0 and cfg.probe_every > 0
            and epoch % cfg.probe_every == 0
        ):
            if validation_pack is not None and "probe_points" in validation_pack:
                probe_points = validation_pack["probe_points"]
            elif sample_pack is not None and "probe_points" in sample_pack:
                probe_points = sample_pack["probe_points"]
            else:
                probe_points = fluid_points(
                    cfg, cfg.validation_points, cfg.probe_beta, cfg.probe_alpha_deg,
                    cfg.seed + 1_500_000,
                )
            probe = peel_step(
                cfg, model, probe_points, cfg.probe_beta, cfg.probe_alpha_deg,
                False, reference_model=None, full_diagnostics=True,
            )
            require_pure_diagnostic(probe)
            row.update({
                "probe_loss": probe["loss"],
                "probe_orth_error": probe["orth_error"],
                "probe_rank": probe["peeled_rank"],
                "probe_peeled_rank": probe["peeled_rank"],
            })
            row[f"probe_{rayleigh_key}"] = probe[rayleigh_key]
        row["requested_modes"] = int(diagnostic.get("requested_modes", cfg.modes))
        history.append(row)
        # Persist a compact atomic heartbeat every epoch.  Full history and
        # model checkpoints retain their configured cadence; this file exists
        # only so long campaigns can be monitored without synchronizing the
        # training process or changing its numerical path.
        live_status = {
            "status": "training",
            "epoch": epoch,
            "configured_epochs": int(cfg.epochs),
            "wall_time_iso": row["wall_time_iso"],
            "training_elapsed_seconds": row["training_elapsed_seconds"],
            "train_loss": row["train_loss"],
            "train_orth_error": row["train_orth_error"],
            "train_peeled_rank": row["train_peeled_rank"],
            "train_min_relative_residual_mass": row[
                "train_min_relative_residual_mass"
            ],
            "train_max_post_peel_overlap": row[
                "train_max_post_peel_overlap"
            ],
            "learning_rate": row["learning_rate"],
            "validation_loss": row.get("validation_loss"),
            "validation_loss_p95": row.get("validation_loss_p95"),
            "validation_loss_max": row.get("validation_loss_max"),
            "validation_orth_error": row.get("validation_orth_error"),
            "validation_min_relative_residual_mass": row.get(
                "validation_min_relative_residual_mass"
            ),
            "validation_max_post_peel_overlap": row.get(
                "validation_max_post_peel_overlap"
            ),
            "best_validation_epoch": row.get(
                "best_validation_epoch", best_epoch
            ),
            "best_validation_loss": (
                best_validation if math.isfinite(best_validation) else None
            ),
        }
        live_path = args.run_dir / "live_status.json"
        live_temporary = live_path.with_suffix(".json.tmp")
        live_temporary.write_text(
            json.dumps(live_status, indent=2), encoding="utf-8"
        )
        os.replace(live_temporary, live_path)
        final_epoch = epoch
        state = training_state(
            current_lr, best_epoch, lr_reference_loss,
            lr_reference_epoch, lr_reductions,
        )
        extra_archive_due = epoch in set(args.extra_archive_epochs)
        regular_checkpoint_due = epoch % cfg.checkpoint_every == 0
        if args.self_test_epochs <= 0 and (
            regular_checkpoint_due or extra_archive_due
        ):
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config, epoch, best_validation, state,
            )
            if epoch % cfg.archive_checkpoint_every == 0 or extra_archive_due:
                archive = (
                    args.run_dir / "checkpoints" / f"hardcoded_epoch{epoch:04d}.npz"
                )
                save_neutral(archive, model, config, epoch, best_validation, state)
                append_log(args.run_dir, f"archived checkpoint epoch={epoch:04d} path={archive}")
            save_history(args.run_dir, history)
        if (
            args.self_test_epochs <= 0 and early_stopping_enabled
            and best_epoch > 0
            and epoch - best_epoch >= cfg.early_stopping_patience_epochs
        ):
            stopped_early = True
            save_neutral(
                args.run_dir / "checkpoints" / "hardcoded_latest.npz",
                model, config, epoch, best_validation, state,
            )
            save_history(args.run_dir, history)
            append_log(
                args.run_dir,
                f"early stopping epoch={epoch:04d} best_epoch={best_epoch:04d}",
            )
            break

    if args.self_test_epochs > 0:
        append_log(args.run_dir, f"hardcoded self-test complete epochs={args.self_test_epochs}")
        return
    final_state = training_state(
        current_lr, best_epoch, lr_reference_loss, lr_reference_epoch,
        lr_reductions, stopped_early=stopped_early,
    )
    planned_pause = (
        args.stop_after_epoch is not None
        and final_epoch == int(args.stop_after_epoch)
        and final_epoch < int(cfg.epochs)
    )
    if planned_pause:
        latest = args.run_dir / "checkpoints" / "hardcoded_latest.npz"
        archive = (
            args.run_dir / "checkpoints" / f"hardcoded_epoch{final_epoch:04d}.npz"
        )
        save_neutral(
            latest, model, config, final_epoch, best_validation, final_state
        )
        if not archive.exists():
            save_neutral(
                archive, model, config, final_epoch, best_validation, final_state
            )
        save_history(args.run_dir, history)
        pause = {
            "paused": True,
            "epoch": final_epoch,
            "configured_epochs": int(cfg.epochs),
            "checkpoint": str(latest.resolve()),
            "archive_checkpoint": str(archive.resolve()),
            "training_elapsed_seconds": (
                float(history[-1]["training_elapsed_seconds"])
                if history else 0.0
            ),
        }
        pause_path = args.run_dir / "planned_pause.json"
        pause_temporary = pause_path.with_suffix(".json.tmp")
        pause_temporary.write_text(json.dumps(pause, indent=2), encoding="utf-8")
        os.replace(pause_temporary, pause_path)
        live_status.update(
            status="paused_for_evaluation",
            epoch=final_epoch,
            training_elapsed_seconds=pause["training_elapsed_seconds"],
        )
        live_temporary.write_text(
            json.dumps(live_status, indent=2), encoding="utf-8"
        )
        os.replace(live_temporary, live_path)
        append_log(
            args.run_dir,
            f"planned training pause epoch={final_epoch:04d} "
            f"checkpoint={latest}",
        )
        return
    frozen = args.run_dir / "checkpoints" / "hardcoded_sss_frozen.npz"
    save_neutral(
        frozen, model, config, final_epoch, best_validation, final_state
    )
    save_history(args.run_dir, history)
    final_history_row = history[-1] if history else {}
    completion = {
        "complete": True,
        "final_epoch": final_epoch,
        "configured_epochs": int(cfg.epochs),
        "stopped_early": stopped_early,
        "best_validation": best_validation,
        "best_validation_epoch": best_epoch,
        "basis_objective": objective,
        "spatial_jet_order_used": JET_ORDER_BY_OBJECTIVE[objective],
        "causal_sort_device": (
            "gpu" if causal_sorting == "geometry_rayleigh" else "not_applicable"
        ),
        "optimizer_device": "gpu_batched_all_modes",
        "diagnostic_host_transfers_per_step": final_history_row.get(
            "diagnostic_host_transfers_per_step"
        ),
        "optimizer_host_transfers_per_step": final_history_row.get(
            "optimizer_host_transfers_per_step"
        ),
        "input_host_to_device_transfers_per_step": final_history_row.get(
            "input_host_to_device_transfers_per_step"
        ),
        "third_derivative_channels_allocated": final_history_row.get(
            "third_derivative_channels_allocated"
        ),
        "sample_pack_device": sample_pack_audit["sample_pack_device"],
        "sample_pack_startup_h2d_transfers": sample_pack_audit[
            "sample_pack_startup_h2d_transfers"
        ],
        "sample_pack_startup_h2d_bytes": sample_pack_audit[
            "sample_pack_startup_h2d_bytes"
        ],
        "sample_pack_load_seconds": sample_pack_load_seconds,
        "sample_pack_upload_seconds": sample_pack_upload_seconds,
        "sample_pack_timing_excluded_from_training_elapsed_seconds": True,
        "finite_difference_used": False,
        "training_uses_pytorch": False,
        "precision": {
            "neural_network": (
                "float16 operands/shadows; float32 accumulation"
                if str(cfg.mlp_compute_dtype).lower() == "float16"
                else "float32"
            ),
            "all_other_numerical_paths": "float32",
        },
        "frozen_checkpoint": str(frozen.resolve()),
    }
    completion_path = args.run_dir / "completion.json"
    completion_temporary = completion_path.with_suffix(".json.tmp")
    completion_temporary.write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    os.replace(completion_temporary, completion_path)
    live_status.update(
        status="complete",
        epoch=final_epoch,
        configured_epochs=int(cfg.epochs),
    )
    live_temporary.write_text(
        json.dumps(live_status, indent=2), encoding="utf-8"
    )
    os.replace(live_temporary, live_path)
    append_log(
        args.run_dir,
        f"training complete epoch={final_epoch} stopped_early={stopped_early} "
        f"frozen={frozen}",
    )


if __name__ == "__main__":
    main()
