"""Run the shared-seed K2048 Velocity/Vorticity training comparison.

The default invocation performs a read-only protocol audit.  Passing
``--execute`` verifies the completed Potential K2048 reference, materializes
one shared IID pack reproducing its seed schedule, and then trains only the
Velocity and Vorticity arms.  The two new arms therefore use the Potential
seed, samples, byte-identical epoch-0 model/Adam arrays, and otherwise differ
only in their Rayleigh field and required spatial jet order.

Recovery is allowed only from ``hardcoded_latest.npz`` inside the same arm.
All objectives for a paired seed read the exact same train and validation NPZ
files and start from byte-identical model/Adam arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy


CASE_ROOT = Path(__file__).resolve().parents[1]
WORKTREE_ROOT = CASE_ROOT.parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    inside_airfoil,
)

RUNNER_PATH = Path(__file__).resolve()
GENERATOR_PATH = (
    CASE_ROOT / "scripts/generate_airfoil_strict_paired_pcg64_pack.py"
)
TRAINER_PATH = CASE_ROOT / "scripts/train_hardcoded.py"
BACKEND_PATH = (
    CASE_ROOT
    / "src/eigenfluid_sss/basisnet/airfoil2d/hardcoded_backend.py"
)
SEQUENCE_ROOT = (
    CASE_ROOT
    / "results/airfoil_strict_velocity_vorticity_k2048_e2000_seed20260823_stable_mgs_v2"
)
CONFIG_ROOT = SEQUENCE_ROOT / "configs"
PACK_ROOT = SEQUENCE_ROOT / "paired_sample_packs"
ARM_ROOT = SEQUENCE_ROOT / "arms"
STATE_PATH = SEQUENCE_ROOT / "sequence_state.json"
MANIFEST_PATH = SEQUENCE_ROOT / "sequence_manifest.json"
PID_PATH = SEQUENCE_ROOT / "runner_pid.json"
PAIRED_SEEDS = (20260823,)
OBJECTIVE_ORDER = ("potential", "velocity", "vorticity")
TRAIN_OBJECTIVES = ("velocity", "vorticity")
OBJECTIVE_ORDER_BY_SEED = {
    20260823: TRAIN_OBJECTIVES,
}
TARGET_MODES = 2048
TARGET_EPOCH = 2000
PREFIXES = (64, 128, 256, 512, 1024, 2048)
ARCHIVE_EPOCHS = (500, 1000, 1500, 2000)
CHECKPOINT_MUTABLE_CONFIG_FIELDS = frozenset({"learning_rate"})

REFERENCE_POTENTIAL_RUN = (
    CASE_ROOT
    / "results/siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
    / "arms/k2048/run_001"
)
REFERENCE_POTENTIAL_INITIAL = (
    REFERENCE_POTENTIAL_RUN.parent / "inputs/initial_epoch0000.npz"
)
REFERENCE_POTENTIAL_FROZEN = (
    REFERENCE_POTENTIAL_RUN / "checkpoints/hardcoded_sss_frozen.npz"
)
REFERENCE_POTENTIAL_VALIDATION = (
    REFERENCE_POTENTIAL_RUN.parent
    / "inputs/fixed_validation_iid_uniform_v1.npz"
)
REFERENCE_POTENTIAL_INITIAL_SHA256 = (
    "9775e8be159a2ac2dd2240f1246f4ba74417de53bf7706a006caf21c649cb821"
)
REFERENCE_POTENTIAL_FROZEN_SHA256 = (
    "92a1cea0d4a369df11788a6e285e6e22a8ca7871f9466e5898654e60f11439e9"
)
REFERENCE_POTENTIAL_VALIDATION_SHA256 = (
    "4b723fe1f504bd6f197c2b134a25ad9696817b427e098a11df0f85896c307162"
)

OBJECTIVES: dict[str, dict[str, Any]] = {
    "potential": {
        "objective": "streamfunction_dirichlet_rayleigh",
        "jet_order": 1,
        "metric": "A-metric",
        "template": CASE_ROOT
        / "configs/k2048_airfoil_strict_potential_rayleigh_2000.json",
    },
    "velocity": {
        "objective": "velocity_rayleigh",
        "jet_order": 2,
        "metric": "u-metric",
        "template": CASE_ROOT
        / "configs/k2048_airfoil_strict_velocity_rayleigh_2000.json",
    },
    "vorticity": {
        "objective": "pure_vorticity_rayleigh",
        "jet_order": 3,
        "metric": "omega-metric",
        "template": CASE_ROOT
        / "configs/k2048_airfoil_strict_vorticity_rayleigh_2000.json",
    },
}

COMMON_REQUIRED = {
    "modes": 2048,
    "epochs": 2000,
    "hidden": 32,
    "hidden_layers": 4,
    "siren_first_omega0": 8.0,
    "siren_hidden_omega0": 30.0,
    "points_per_step": 4096,
    "sampling_mode": "uniform_random",
    "reorthogonalization_passes": 2,
    "causal_sorting": "geometry_rayleigh",
    "optimizer": "adam",
    "learning_rate": 2.0e-4,
    "base_learning_rate": 2.0e-4,
    "lr_scheduler_type": "cosine",
    "min_learning_rate": 1.0e-6,
    "grad_clip": 5.0,
    "validate_every": 25,
    "early_stopping_enabled": False,
    "quadrature_measure": "physical_fluid_area",
    # A residual mass is a squared-amplitude ratio.  The shared 1e-12 floor
    # therefore rejects only candidates below an approximately 1e-6 amplitude
    # ratio, while retaining a tenfold margin over FP32 amplitude epsilon.
    "peeling_relative_mass_floor": 1.0e-12,
}

FP32_KEYS = (
    "mlp_compute_dtype",
    "mlp_accumulation_dtype",
    "jet_dtype",
    "parameter_backward_dtype",
    "quadrature_dtype",
    "ritz_dtype",
    "master_weight_dtype",
    "optimizer_dtype",
)

SAMPLE_PACK_RUNTIME_EXPECTED = {
    "sample_pack_device": "gpu_resident",
    "sample_pack_startup_h2d_transfers": 1,
    "input_host_to_device_transfers_per_step": 0,
    "optimizer_host_transfers_per_step": 0,
    "diagnostic_host_transfers_per_step": 1,
}
SAMPLE_PACK_TIMING_EXCLUSION_KEY = (
    "sample_pack_timing_excluded_from_training_elapsed_seconds"
)
SAMPLE_PACK_TIMING_KEYS = (
    "sample_pack_load_seconds",
    "sample_pack_upload_seconds",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_source_lock() -> dict[str, Any]:
    return {
        "runner": {"path": str(RUNNER_PATH), "sha256": sha256(RUNNER_PATH)},
        "generator": {
            "path": str(GENERATOR_PATH.resolve()),
            "sha256": sha256(GENERATOR_PATH),
        },
        "trainer": {
            "path": str(TRAINER_PATH.resolve()),
            "sha256": sha256(TRAINER_PATH),
        },
        "backend": {
            "path": str(BACKEND_PATH.resolve()),
            "sha256": sha256(BACKEND_PATH),
        },
        "config_templates": {
            objective: {
                "path": str(template_path(objective).resolve()),
                "sha256": sha256(template_path(objective)),
            }
            for objective in OBJECTIVE_ORDER
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def template_path(objective: str) -> Path:
    return Path(OBJECTIVES[objective]["template"])


def generated_config_path(seed: int, objective: str) -> Path:
    return CONFIG_ROOT / f"seed_{seed}" / f"{objective}.json"


def arm_root(seed: int, objective: str) -> Path:
    return ARM_ROOT / f"seed_{seed}" / objective


def run_dir(seed: int, objective: str) -> Path:
    return arm_root(seed, objective) / "run_001"


def pack_dir(seed: int) -> Path:
    return PACK_ROOT / f"seed_{seed}"


def train_pack(seed: int) -> Path:
    return pack_dir(seed) / "paired_train_pcg64_iid_e2000_p4096.npz"


def validation_pack(seed: int) -> Path:
    return pack_dir(seed) / "paired_validation_pcg64_iid_p4096.npz"


def config_differences(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    return {
        key: (reference.get(key), candidate.get(key))
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key) != candidate.get(key)
    }


def audit_templates() -> dict[str, Any]:
    templates = {
        objective: read_json(template_path(objective))
        for objective in OBJECTIVE_ORDER
    }
    potential = templates["potential"]
    allowed_objective_differences = {
        "basis_objective",
        "spatial_derivative_order",
        "result_root",
    }
    pairwise_differences: dict[str, list[str]] = {}
    for objective, config in templates.items():
        mismatches = {
            key: (config.get(key), expected)
            for key, expected in COMMON_REQUIRED.items()
            if config.get(key) != expected
        }
        wrong_precision = {
            key: config.get(key)
            for key in FP32_KEYS
            if str(config.get(key, "")).lower() != "float32"
        }
        spec = OBJECTIVES[objective]
        if config.get("basis_objective") != spec["objective"]:
            mismatches["basis_objective"] = (
                config.get("basis_objective"),
                spec["objective"],
            )
        if int(config.get("spatial_derivative_order", -1)) != int(
            spec["jet_order"]
        ):
            mismatches["spatial_derivative_order"] = (
                config.get("spatial_derivative_order"),
                spec["jet_order"],
            )
        if mismatches or wrong_precision:
            raise RuntimeError(
                f"{objective} strict template audit failed: "
                f"mismatches={mismatches}, precision={wrong_precision}"
            )
        changed = set(config_differences(potential, config))
        if not changed.issubset(allowed_objective_differences):
            raise RuntimeError(
                f"{objective} changes non-mechanism fields: {sorted(changed)}"
            )
        pairwise_differences[f"potential_vs_{objective}"] = sorted(changed)
    if PAIRED_SEEDS != (20260823,):
        raise RuntimeError("shared Potential seed contract changed")
    if set(OBJECTIVE_ORDER_BY_SEED) != set(PAIRED_SEEDS):
        raise RuntimeError("objective-order seed set changed")
    for seed, order in OBJECTIVE_ORDER_BY_SEED.items():
        if order != TRAIN_OBJECTIVES:
            raise RuntimeError(
                f"seed {seed} objective order differs from {TRAIN_OBJECTIVES}"
            )
    return {
        "status": "strict_protocol_audited",
        "target_modes": TARGET_MODES,
        "causal_prefixes": list(PREFIXES),
        "paired_seeds": list(PAIRED_SEEDS),
        "objective_order_by_seed": {
            str(seed): list(OBJECTIVE_ORDER_BY_SEED[seed])
            for seed in PAIRED_SEEDS
        },
        "reference_objective": "potential",
        "trained_objectives": list(TRAIN_OBJECTIVES),
        "order_design": (
            "reuse the completed Potential K2048 seed-20260823 run as an "
            "immutable reference; train only Velocity then Vorticity"
        ),
        "template_sha256": {
            name: sha256(template_path(name)) for name in OBJECTIVE_ORDER
        },
        "pairwise_config_differences": pairwise_differences,
        "common": COMMON_REQUIRED,
        "precision": {key: "float32" for key in FP32_KEYS},
        "sample_pack_runtime_contract": {
            **SAMPLE_PACK_RUNTIME_EXPECTED,
            SAMPLE_PACK_TIMING_EXCLUSION_KEY: True,
            "sample_pack_startup_h2d_bytes": (
                expected_sample_pack_startup_h2d_bytes(potential)
            ),
            "sample_pack_residency_scope": (
                "train/validation/probe point tensors; beta/alpha remain "
                "locked NumPy host kernel-launch scalars"
            ),
        },
        "source_lock": protocol_source_lock(),
        "rayleigh": (
            "R_k^(F)=||grad(F_tilde_k)||_h^2/||F_tilde_k||_h^2, "
            "F in {A,u,omega}"
        ),
        "shared_transform_contract": (
            "For each objective, the causal triangular MGS transform is a "
            "mode-space linear map.  The same map acts on A, u=curl(A), "
            "and omega=curl(u); spatial differentiation commutes with this "
            "mode-space map."
        ),
    }


def make_generated_configs() -> dict[tuple[int, str], dict[str, Any]]:
    generated: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in PAIRED_SEEDS:
        common_by_objective: dict[str, dict[str, Any]] = {}
        for objective in OBJECTIVE_ORDER:
            config = dict(read_json(template_path(objective)))
            relative_root = arm_root(seed, objective).resolve().relative_to(
                WORKTREE_ROOT.resolve()
            ).as_posix()
            config.update(seed=seed, result_root=relative_root)
            path = generated_config_path(seed, objective)
            if path.exists() and read_json(path) != config:
                raise RuntimeError(f"existing generated config differs: {path}")
            if not path.exists():
                write_json(path, config)
            generated[(seed, objective)] = config
            common_by_objective[objective] = {
                key: value
                for key, value in config.items()
                if key not in {
                    "basis_objective",
                    "spatial_derivative_order",
                    "result_root",
                }
            }
        if len(
            {
                json.dumps(value, sort_keys=True)
                for value in common_by_objective.values()
            }
        ) != 1:
            raise RuntimeError(f"seed {seed} generated configs are not paired")
    return generated


def audit_reference_potential(
    canonical_initial: Path | None = None,
) -> dict[str, Any]:
    """Verify the immutable completed Potential run used by the comparison."""

    required = {
        "initial": REFERENCE_POTENTIAL_INITIAL,
        "initial_metadata": REFERENCE_POTENTIAL_INITIAL.with_suffix(".json"),
        "frozen": REFERENCE_POTENTIAL_FROZEN,
        "frozen_metadata": REFERENCE_POTENTIAL_FROZEN.with_suffix(".json"),
        "validation": REFERENCE_POTENTIAL_VALIDATION,
        "completion": REFERENCE_POTENTIAL_RUN / "completion.json",
        "protocol": REFERENCE_POTENTIAL_RUN / "hardcoded_protocol.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Potential reference is incomplete: {missing}")
    initial_hash = sha256(required["initial"])
    frozen_hash = sha256(required["frozen"])
    validation_hash = sha256(required["validation"])
    if initial_hash != REFERENCE_POTENTIAL_INITIAL_SHA256:
        raise RuntimeError(
            "Potential reference epoch-0 archive changed: "
            f"{initial_hash} != {REFERENCE_POTENTIAL_INITIAL_SHA256}"
        )
    if frozen_hash != REFERENCE_POTENTIAL_FROZEN_SHA256:
        raise RuntimeError(
            "Potential reference frozen checkpoint changed: "
            f"{frozen_hash} != {REFERENCE_POTENTIAL_FROZEN_SHA256}"
        )
    if validation_hash != REFERENCE_POTENTIAL_VALIDATION_SHA256:
        raise RuntimeError(
            "Potential reference validation pack changed: "
            f"{validation_hash} != {REFERENCE_POTENTIAL_VALIDATION_SHA256}"
        )
    if not zipfile.is_zipfile(required["initial"]) or not zipfile.is_zipfile(
        required["frozen"]
    ):
        raise RuntimeError("Potential reference archive is not a readable NPZ")

    initial_metadata = read_json(required["initial_metadata"])
    frozen_metadata = read_json(required["frozen_metadata"])
    completion = read_json(required["completion"])
    protocol = read_json(required["protocol"])
    initial_config = initial_metadata.get("config", {})
    frozen_config = frozen_metadata.get("config", {})
    if int(initial_metadata.get("epoch", -1)) != 0:
        raise RuntimeError("Potential reference initialization is not epoch 0")
    if int(frozen_metadata.get("epoch", -1)) != TARGET_EPOCH:
        raise RuntimeError("Potential reference frozen checkpoint is not epoch 2000")
    if not bool(completion.get("complete")) or int(
        completion.get("final_epoch", -1)
    ) != TARGET_EPOCH:
        raise RuntimeError("Potential reference completion is not epoch 2000")
    if completion.get("basis_objective") != OBJECTIVES["potential"]["objective"]:
        raise RuntimeError("Potential reference completion has another objective")
    if protocol.get("basis_objective") != OBJECTIVES["potential"]["objective"]:
        raise RuntimeError("Potential reference protocol has another objective")
    if int(completion.get("spatial_jet_order_used", -1)) != 1:
        raise RuntimeError("Potential reference did not use Jet1")
    for label, config in (
        ("initial", initial_config),
        ("frozen", frozen_config),
    ):
        if int(config.get("seed", -1)) != PAIRED_SEEDS[0]:
            raise RuntimeError(f"Potential {label} seed is not 20260823")
        for key, expected in COMMON_REQUIRED.items():
            # The legacy checkpoint predates the explicit optimizer field;
            # train_hardcoded.py used Adam by default.  Its mutable current LR
            # is also expected to have reached the cosine endpoint.
            if key in {
                "optimizer",
                "learning_rate",
                "peeling_relative_mass_floor",
            }:
                continue
            if config.get(key) != expected:
                raise RuntimeError(
                    f"Potential reference {label} {key}={config.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if config.get("basis_objective") != OBJECTIVES["potential"]["objective"]:
            raise RuntimeError(f"Potential reference {label} objective changed")
        if int(config.get("spatial_derivative_order", -1)) != 1:
            raise RuntimeError(f"Potential reference {label} is not Jet1")
    if float(frozen_config.get("learning_rate", float("nan"))) != float(
        COMMON_REQUIRED["min_learning_rate"]
    ):
        raise RuntimeError("Potential reference did not reach the cosine LR endpoint")
    sampling = protocol.get("sampling", {})
    if sampling.get("sequence") != "IID PCG64 uniform random" or int(
        sampling.get("points_per_epoch", -1)
    ) != COMMON_REQUIRED["points_per_step"]:
        raise RuntimeError("Potential reference sampling protocol changed")

    canonical_hash = sha256(canonical_initial) if canonical_initial else None
    if canonical_hash is not None and canonical_hash != initial_hash:
        raise RuntimeError(
            "new Velocity/Vorticity canonical initialization does not equal "
            "the completed Potential epoch-0 archive"
        )
    return {
        "status": "complete_reference_verified",
        "role": "immutable_external_reference_not_retrained",
        "seed": PAIRED_SEEDS[0],
        "objective_name": "potential",
        "basis_objective": OBJECTIVES["potential"]["objective"],
        "spatial_jet_order": 1,
        "final_epoch": TARGET_EPOCH,
        "initial_checkpoint": str(required["initial"].resolve()),
        "initial_archive_sha256": initial_hash,
        "frozen_checkpoint": str(required["frozen"].resolve()),
        "frozen_checkpoint_sha256": frozen_hash,
        "fixed_validation_pack": str(required["validation"].resolve()),
        "fixed_validation_pack_sha256": validation_hash,
        "completion": str(required["completion"].resolve()),
        "completion_sha256": sha256(required["completion"]),
        "protocol": str(required["protocol"].resolve()),
        "protocol_sha256": sha256(required["protocol"]),
        "canonical_initialization_exact_match": canonical_hash == initial_hash
        if canonical_hash is not None
        else None,
        "legacy_default_equivalence": {
            "optimizer": "Adam (train_hardcoded.py default)",
            "checkpoint_learning_rate": "cosine endpoint 1e-6",
            "peeling_relative_mass_floor": (
                "not evaluated by the completed Jet1 Potential kernel; the "
                "shared 1e-12 squared-amplitude safeguard is used by all new "
                "configs"
            ),
        },
    }


def gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = [
            field.strip()
            for field in subprocess.check_output(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            .strip()
            .splitlines()[0]
            .split(",")
        ]
    except (IndexError, OSError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": repr(error)}
    return {
        "available": True,
        "name": fields[0],
        "uuid": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": int(fields[3]),
        "memory_used_mib": int(fields[4]),
        "utilization_percent": int(fields[5]),
        "temperature_c": int(fields[6]),
    }


def update_state(**updates: Any) -> None:
    state = read_json(STATE_PATH) if STATE_PATH.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    state["updated_unix_ns"] = time.time_ns()
    write_json(STATE_PATH, state)


def update_arm(seed: int, objective: str, **updates: Any) -> None:
    manifest = read_json(MANIFEST_PATH)
    for arm in manifest["arms"]:
        if int(arm["seed"]) == seed and arm["objective_name"] == objective:
            arm.update(updates)
            arm["updated_at"] = now()
            arm["updated_unix_ns"] = time.time_ns()
            break
    else:
        raise KeyError((seed, objective))
    manifest["updated_at"] = now()
    manifest["updated_unix_ns"] = time.time_ns()
    write_json(MANIFEST_PATH, manifest)


def ensure_manifest(
    audit: dict[str, Any],
    generated: dict[tuple[int, str], dict[str, Any]],
) -> None:
    if MANIFEST_PATH.exists():
        manifest = read_json(MANIFEST_PATH)
        if manifest.get("protocol_audit") != audit:
            raise RuntimeError("existing sequence manifest protocol changed")
        return
    reference = audit_reference_potential()
    created_at = now()
    created_ns = time.time_ns()
    write_json(
        MANIFEST_PATH,
        {
            "experiment": (
                "airfoil_strict_velocity_vorticity_k2048_e2000_seed20260823"
            ),
            "status": "initialized",
            "created_at": created_at,
            "created_unix_ns": created_ns,
            "protocol_audit": audit,
            "strict_order": [
                {"seed": seed, "objective": objective}
                for seed in PAIRED_SEEDS
                for objective in OBJECTIVE_ORDER_BY_SEED[seed]
            ],
            "order_control": (
                "completed Potential is an immutable reference; train only "
                "Velocity then Vorticity with its seed"
            ),
            "shared_seed": 20260823,
            "shared_seed_provenance": "Potential training seed",
            "reference_baseline": reference,
            "fresh_epoch0_per_new_arm": True,
            "same_initialization_as_reference_potential": True,
            "cross_arm_checkpoint_reuse": False,
            "recovery_policy": "same-arm hardcoded_latest.npz only",
            "sample_policy": (
                "one train NPZ and one validation NPZ generated by NumPy "
                "with the completed Potential seed 20260823; Velocity and "
                "Vorticity read the same absolute paths and hashes; a "
                "pre-training audit reproduces the legacy Potential schedule"
            ),
            "sample_residency_contract": {
                "sample_pack_device": "gpu_resident",
                "sample_pack_startup_h2d_transfers": 1,
                "input_host_to_device_transfers_per_step": 0,
                "optimizer_host_transfers_per_step": 0,
                "diagnostic_host_transfers_per_step": 1,
                "sample_pack_timing_excluded_from_training_elapsed_seconds": (
                    True
                ),
                "timing": (
                    "pack file load and startup GPU upload are reported "
                    "separately and excluded from training_elapsed_seconds"
                ),
            },
            "evaluation_design": {
                "checkpoint": "single exact K2048 frozen checkpoint per arm",
                "prefixes": list(PREFIXES),
                "prefix_policy": (
                    "for each evaluation geometry, form the objective-metric "
                    "K2048 causal sort and two-pass MGS transform once, persist "
                    "that full transform, and evaluate its nested first "
                    "64/128/256/512/1024/2048 columns; do not train lower-K arms"
                ),
                "common_metrics": [
                    "velocity_L2_projection_error",
                    "vorticity_L2_projection_error",
                    "rollout_velocity_error",
                    "rollout_vorticity_error",
                    "rollout_spectrum_error",
                    "training_elapsed_seconds",
                    "peak_gpu_memory_mib",
                    "Gram_max_abs_defect",
                ],
            },
            "arms": [
                {
                    "seed": seed,
                    "objective_name": objective,
                    "basis_objective": OBJECTIVES[objective]["objective"],
                    "metric": OBJECTIVES[objective]["metric"],
                    "spatial_jet_order": OBJECTIVES[objective]["jet_order"],
                    "status": "pending",
                    "config": str(
                        generated_config_path(seed, objective).resolve()
                    ),
                    "config_sha256": sha256(
                        generated_config_path(seed, objective)
                    ),
                    "config_snapshot": generated[(seed, objective)],
                    "result_root": str(arm_root(seed, objective).resolve()),
                    "run_dir": str(run_dir(seed, objective).resolve()),
                    "shared_train_pack": str(train_pack(seed).resolve()),
                    "shared_validation_pack": str(
                        validation_pack(seed).resolve()
                    ),
                    "archive_epochs": list(ARCHIVE_EPOCHS),
                    "causal_prefixes": list(PREFIXES),
                }
                for seed in PAIRED_SEEDS
                for objective in TRAIN_OBJECTIVES
            ],
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "gpu": gpu_snapshot(),
            },
        },
    )
    write_json(
        STATE_PATH,
        {
            "status": "initialized",
            "created_at": created_at,
            "created_unix_ns": created_ns,
            "sequence_root": str(SEQUENCE_ROOT.resolve()),
            "current_seed": None,
            "current_objective": None,
            "completed_arms": [],
            "total_arms": len(PAIRED_SEEDS) * len(TRAIN_OBJECTIVES),
            "reference_objective": "potential",
            "reference_status": "complete_reference_verified",
        },
    )


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=WORKTREE_ROOT, check=True)


def ensure_pack(seed: int) -> tuple[Path, Path, dict[str, Any]]:
    directory = pack_dir(seed)
    manifest_path = directory / "paired_sample_manifest.json"
    run(
        [
            sys.executable,
            str(
                CASE_ROOT
                / "scripts/generate_airfoil_strict_paired_pcg64_pack.py"
            ),
            "--config",
            str(generated_config_path(seed, "potential")),
            "--output-dir",
            str(directory),
        ]
    )
    manifest = read_json(manifest_path)
    training = train_pack(seed)
    validation = validation_pack(seed)
    if manifest["train_pack"]["sha256"] != sha256(training):
        raise RuntimeError(f"seed {seed} train-pack hash mismatch")
    if manifest["validation_pack"]["sha256"] != sha256(validation):
        raise RuntimeError(f"seed {seed} validation-pack hash mismatch")
    return training, validation, manifest


def expected_pack_source_lock(config_path: Path) -> dict[str, Any]:
    return {
        "generator": {
            "path": str(GENERATOR_PATH.resolve()),
            "sha256": sha256(GENERATOR_PATH),
        },
        "trainer": {
            "path": str(TRAINER_PATH.resolve()),
            "sha256": sha256(TRAINER_PATH),
        },
        "backend": {
            "path": str(BACKEND_PATH.resolve()),
            "sha256": sha256(BACKEND_PATH),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256(config_path),
        },
        "runtime": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }


def require_finite_float32(name: str, value: np.ndarray) -> None:
    if value.dtype != np.float32:
        raise RuntimeError(f"{name} dtype {value.dtype} is not float32")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains a non-finite value")


def require_finite_float64(name: str, value: np.ndarray) -> None:
    if value.dtype != np.float64:
        raise RuntimeError(f"{name} dtype {value.dtype} is not float64")
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(f"{name} contains a non-finite value")


def audit_sample_pack_contents(
    config: dict[str, Any], training: Path, validation: Path
) -> dict[str, Any]:
    """Validate shapes, locked dtypes, bounds, and solid exclusion."""

    cfg = cfg_namespace(config)
    expected_train_names = {
        "train_points",
        "train_beta",
        "train_alpha",
        "train_epoch_1based",
    }
    with np.load(training, allow_pickle=False) as archive:
        if set(archive.files) != expected_train_names:
            raise RuntimeError(
                f"train pack arrays {sorted(archive.files)} do not match "
                f"{sorted(expected_train_names)}"
            )
        train_points_array = archive["train_points"]
        train_beta_array = archive["train_beta"]
        train_alpha_array = archive["train_alpha"]
        train_epoch_array = archive["train_epoch_1based"]
    expected_train_shape = (int(cfg.epochs), int(cfg.points_per_step), 2)
    if train_points_array.shape != expected_train_shape:
        raise RuntimeError(
            f"train_points shape {train_points_array.shape} != {expected_train_shape}"
        )
    if train_beta_array.shape != (int(cfg.epochs),) or train_alpha_array.shape != (
        int(cfg.epochs),
    ):
        raise RuntimeError("train geometry arrays have the wrong epoch count")
    if train_epoch_array.shape != (int(cfg.epochs),) or not np.array_equal(
        train_epoch_array,
        np.arange(1, int(cfg.epochs) + 1, dtype=train_epoch_array.dtype),
    ):
        raise RuntimeError("train_epoch_1based is not the exact 1..epochs sequence")
    if train_epoch_array.dtype != np.int32:
        raise RuntimeError("train_epoch_1based must be int32")
    require_finite_float32("train_points", train_points_array)
    # The completed Potential run used the full float64 Sobol geometry
    # parameters as host kernel-launch scalars.  Keeping these two vectors in
    # float64 is required for exact reference-schedule reproduction; sampled
    # point tensors and every GPU numerical path remain FP32.
    require_finite_float64("train_beta", train_beta_array)
    require_finite_float64("train_alpha", train_alpha_array)
    if np.any((train_beta_array < 0.0) | (train_beta_array > 1.0)):
        raise RuntimeError("train beta is outside [0,1]")
    if np.any(
        (train_alpha_array < np.float32(cfg.alpha_min_deg))
        | (train_alpha_array > np.float32(cfg.alpha_max_deg))
    ):
        raise RuntimeError("train alpha is outside the configured interval")
    if np.any(
        (train_points_array[..., 0] < np.float32(cfg.xmin))
        | (train_points_array[..., 0] > np.float32(cfg.xmax))
        | (train_points_array[..., 1] < np.float32(cfg.ymin))
        | (train_points_array[..., 1] > np.float32(cfg.ymax))
    ):
        raise RuntimeError("train point is outside the physical box")
    for index in range(int(cfg.epochs)):
        if np.any(
            inside_airfoil(
                cfg,
                train_points_array[index],
                float(train_beta_array[index]),
                float(train_alpha_array[index]),
            )
        ):
            raise RuntimeError(f"train epoch {index + 1} contains a solid point")

    expected_validation_names = {
        "validation_points",
        "validation_beta",
        "validation_alpha",
        "probe_points",
    }
    with np.load(validation, allow_pickle=False) as archive:
        if set(archive.files) != expected_validation_names:
            raise RuntimeError(
                f"validation pack arrays {sorted(archive.files)} do not match "
                f"{sorted(expected_validation_names)}"
            )
        validation_points_array = archive["validation_points"]
        validation_beta_array = archive["validation_beta"]
        validation_alpha_array = archive["validation_alpha"]
        probe_points_array = archive["probe_points"]
    pair_count = int(cfg.validation_beta_samples) * int(
        cfg.validation_alpha_samples
    )
    expected_validation_shape = (
        pair_count,
        int(cfg.validation_points),
        2,
    )
    if validation_points_array.shape != expected_validation_shape:
        raise RuntimeError(
            "validation_points shape "
            f"{validation_points_array.shape} != {expected_validation_shape}"
        )
    if validation_beta_array.shape != (pair_count,) or validation_alpha_array.shape != (
        pair_count,
    ):
        raise RuntimeError("validation geometry arrays have the wrong pair count")
    if probe_points_array.shape != (int(cfg.validation_points), 2):
        raise RuntimeError("probe_points shape is incorrect")
    for name, value in (
        ("validation_points", validation_points_array),
        ("validation_beta", validation_beta_array),
        ("validation_alpha", validation_alpha_array),
        ("probe_points", probe_points_array),
    ):
        require_finite_float32(name, value)
    if np.any((validation_beta_array < 0.0) | (validation_beta_array > 1.0)):
        raise RuntimeError("validation beta is outside [0,1]")
    if np.any(
        (validation_alpha_array < np.float32(cfg.alpha_min_deg))
        | (validation_alpha_array > np.float32(cfg.alpha_max_deg))
    ):
        raise RuntimeError("validation alpha is outside the configured interval")
    for name, value in (
        ("validation_points", validation_points_array),
        ("probe_points", probe_points_array),
    ):
        if np.any(
            (value[..., 0] < np.float32(cfg.xmin))
            | (value[..., 0] > np.float32(cfg.xmax))
            | (value[..., 1] < np.float32(cfg.ymin))
            | (value[..., 1] > np.float32(cfg.ymax))
        ):
            raise RuntimeError(f"{name} contains a point outside the physical box")
    for index in range(pair_count):
        if np.any(
            inside_airfoil(
                cfg,
                validation_points_array[index],
                float(validation_beta_array[index]),
                float(validation_alpha_array[index]),
            )
        ):
            raise RuntimeError(f"validation pair {index} contains a solid point")
    if np.any(
        inside_airfoil(
            cfg,
            probe_points_array,
            float(np.float32(cfg.probe_beta)),
            float(np.float32(cfg.probe_alpha_deg)),
        )
    ):
        raise RuntimeError("probe_points contains a solid point")
    return {
        "train_points_shape": list(train_points_array.shape),
        "validation_points_shape": list(validation_points_array.shape),
        "probe_points_shape": list(probe_points_array.shape),
        "train_epoch_count": int(len(train_epoch_array)),
        "all_finite": True,
        "train_points_are_float32": True,
        "train_geometry_scalars_are_float64": True,
        "validation_arrays_are_float32": True,
        "all_points_outside_solid_after_float32_storage": True,
    }


def audit_seed_packs(seed: int, training: Path, validation: Path) -> dict[str, Any]:
    manifest_path = pack_dir(seed) / "paired_sample_manifest.json"
    manifest = read_json(manifest_path)
    config_path = generated_config_path(seed, "potential")
    config = read_json(config_path)
    actual_train_hash = sha256(training)
    actual_validation_hash = sha256(validation)
    if actual_train_hash != manifest["train_pack"]["sha256"]:
        raise RuntimeError(f"seed {seed} train pack changed on disk")
    if actual_validation_hash != manifest["validation_pack"]["sha256"]:
        raise RuntimeError(f"seed {seed} validation pack changed on disk")
    if actual_validation_hash != REFERENCE_POTENTIAL_VALIDATION_SHA256:
        raise RuntimeError(
            f"seed {seed} validation pack is not byte-identical to the "
            "completed Potential reference"
        )
    expected_lock = expected_pack_source_lock(config_path)
    if manifest.get("source_lock") != expected_lock:
        raise RuntimeError(f"seed {seed} pack source lock changed")
    contents = audit_sample_pack_contents(config, training, validation)
    return {
        "audited_at": now(),
        "audited_unix_ns": time.time_ns(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256(manifest_path),
        "train_pack_sha256": actual_train_hash,
        "validation_pack_sha256": actual_validation_hash,
        "source_lock": expected_lock,
        **contents,
    }


def reference_geometry_history() -> list[tuple[int, float, float]]:
    """Read only epoch/beta/alpha from the large legacy history JSON."""

    path = REFERENCE_POTENTIAL_RUN / "data/hardcoded_training_history.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[tuple[int, float, float]] = []
    epoch: int | None = None
    beta: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip().rstrip(",")
            if line.startswith('"epoch":'):
                epoch = int(line.split(":", 1)[1])
                beta = None
            elif epoch is not None and line.startswith('"beta":'):
                beta = float(line.split(":", 1)[1])
            elif (
                epoch is not None
                and beta is not None
                and line.startswith('"alpha_deg":')
            ):
                alpha = float(line.split(":", 1)[1])
                records.append((epoch, beta, alpha))
                epoch = None
                beta = None
    if len(records) != TARGET_EPOCH or [row[0] for row in records] != list(
        range(1, TARGET_EPOCH + 1)
    ):
        raise RuntimeError("Potential reference geometry history is incomplete")
    return records


def audit_reference_sample_schedule(
    seed: int, training: Path
) -> dict[str, Any]:
    """Prove that the fixed pack reproduces the completed Potential schedule."""

    if seed != PAIRED_SEEDS[0]:
        raise RuntimeError("sample schedule must use the Potential seed")
    config = read_json(generated_config_path(seed, "potential"))
    cfg = cfg_namespace(config)
    history = reference_geometry_history()
    with np.load(training, allow_pickle=False) as archive:
        train_points_array = archive["train_points"]
        train_beta_array = archive["train_beta"]
        train_alpha_array = archive["train_alpha"]
        if train_points_array.shape != (
            TARGET_EPOCH,
            int(cfg.points_per_step),
            2,
        ):
            raise RuntimeError("paired train pack has the wrong shape")
        for index, (epoch, beta, alpha) in enumerate(history):
            deterministic_beta, deterministic_alpha = continuous_parameters(
                cfg, epoch
            )
            if beta != deterministic_beta or alpha != deterministic_alpha:
                raise RuntimeError(
                    f"Potential reference epoch {epoch} geometry does not "
                    "match its deterministic seed schedule"
                )
            if train_beta_array[index] != beta or train_alpha_array[index] != alpha:
                raise RuntimeError(
                    f"paired pack epoch {epoch} geometry differs from the "
                    "Potential reference"
                )
            expected_points = np.asarray(
                fluid_points(
                    cfg,
                    int(cfg.points_per_step),
                    beta,
                    alpha,
                    seed + epoch * 1009,
                ),
                dtype=np.float32,
            )
            if not np.array_equal(train_points_array[index], expected_points):
                raise RuntimeError(
                    f"paired pack epoch {epoch} points differ from the "
                    "Potential PCG64 seed schedule"
                )
    geometry_array = np.asarray(
        [(beta, alpha) for _, beta, alpha in history],
        dtype=np.float64,
    )
    return {
        "status": "reference_schedule_exactly_reproduced",
        "reference_seed": seed,
        "epochs_audited": TARGET_EPOCH,
        "points_per_epoch": int(cfg.points_per_step),
        "point_rows_exact_backend_fp32_inputs": TARGET_EPOCH,
        "geometry_values_exact_float64": True,
        "geometry_schedule_sha256": hashlib.sha256(
            geometry_array.tobytes(order="C")
        ).hexdigest(),
        "point_sampling_kernel": (
            "numpy.default_rng(PCG64(seed + epoch*1009)); legacy "
            "uniform_random rejection sampler"
        ),
        "geometry_sampling_kernel": (
            "scrambled Sobol(seed + 700000 + epoch), with endpoint-period "
            "corner overrides"
        ),
    }


def canonical_initial_dir(seed: int) -> Path:
    return SEQUENCE_ROOT / "paired_initialization" / f"seed_{seed}"


def ensure_canonical_initial(seed: int) -> tuple[Path, dict[str, Any]]:
    directory = canonical_initial_dir(seed)
    initial = directory / "initial_epoch0000.npz"
    if not initial.exists():
        run(
            [
                sys.executable,
                str(CASE_ROOT / "scripts/prepare_hardcoded_run.py"),
                "--config",
                str(generated_config_path(seed, "potential")),
                "--output-dir",
                str(directory),
            ]
        )
    metadata = read_json(initial.with_suffix(".json"))
    if int(metadata.get("epoch", -1)) != 0:
        raise RuntimeError(f"seed {seed} canonical initialization is not epoch 0")
    return initial, metadata


def objective_initial_metadata(
    canonical: dict[str, Any], config: dict[str, Any], objective: str
) -> dict[str, Any]:
    spec = OBJECTIVES[objective]
    jet_order = int(spec["jet_order"])
    metadata = dict(canonical)
    metadata.update(
        format=f"eigenfluid_airfoil_continuous_jet{jet_order}_v1",
        source=(
            "byte-identical paired-seed initialization arrays; objective "
            "metadata specialized without redrawing parameters"
        ),
        config=config,
        spatial_ad=(
            "analytic_value_gradient_xy_jet1"
            if jet_order == 1
            else (
                "analytic_value_gradient_compact_hessian_xy_jet2"
                if jet_order == 2
                else "analytic_compact_symmetric_xy_jet3"
            )
        ),
        parameter_backward=f"handwritten_reverse_jet{jet_order}",
        loss_terms=[
            {
                "potential": "unscaled_streamfunction_dirichlet_rayleigh",
                "velocity": "unscaled_velocity_rayleigh",
                "vorticity": "unscaled_vorticity_rayleigh",
            }[objective]
        ],
    )
    return metadata


def ensure_arm_initial(seed: int, objective: str) -> Path:
    canonical_path, canonical_metadata = ensure_canonical_initial(seed)
    inputs = arm_root(seed, objective) / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    initial = inputs / "initial_epoch0000.npz"
    if not initial.exists():
        try:
            os.link(canonical_path, initial)
        except OSError:
            shutil.copyfile(canonical_path, initial)
    config = read_json(generated_config_path(seed, objective))
    metadata = objective_initial_metadata(
        canonical_metadata, config, objective
    )
    sidecar = initial.with_suffix(".json")
    if sidecar.exists() and read_json(sidecar) != metadata:
        raise RuntimeError(f"existing epoch-0 metadata mismatch: {sidecar}")
    if not sidecar.exists():
        write_json(sidecar, metadata)
    # The arm archive is either a hard link or an exact byte copy of the
    # canonical archive.  A raw SHA-256 comparison is both stronger for this
    # contract and avoids opening 81,920 tiny NPY members one by one.
    if sha256(initial) != sha256(canonical_path):
        raise RuntimeError(f"seed {seed} {objective} initialization differs")
    return initial


def audit_checkpoint_config(
    expected: dict[str, Any], actual: dict[str, Any], epoch: int
) -> dict[str, Any]:
    differences = config_differences(expected, actual)
    illegal = set(differences).difference(CHECKPOINT_MUTABLE_CONFIG_FIELDS)
    if illegal:
        raise RuntimeError(
            "checkpoint changed immutable config fields: "
            f"{sorted(illegal)}; differences={differences}"
        )
    if epoch == 0 and differences:
        raise RuntimeError(
            f"epoch-0 checkpoint config must be exact: {differences}"
        )
    if "learning_rate" in differences:
        learning_rate = float(actual.get("learning_rate", float("nan")))
        minimum = float(expected["min_learning_rate"])
        maximum = float(expected["base_learning_rate"])
        if not math.isfinite(learning_rate) or not minimum <= learning_rate <= maximum:
            raise RuntimeError(
                f"checkpoint learning_rate {learning_rate} is outside "
                f"[{minimum}, {maximum}]"
            )
    return {
        "full_config_key_count": len(expected),
        "mutable_fields": sorted(CHECKPOINT_MUTABLE_CONFIG_FIELDS),
        "observed_mutable_differences": sorted(differences),
        "all_immutable_fields_exact": True,
    }


def validate_checkpoint(
    seed: int,
    objective: str,
    checkpoint: Path,
    *,
    exact_epoch: int | None = None,
) -> dict[str, Any]:
    if not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file():
        raise FileNotFoundError(checkpoint)
    if not zipfile.is_zipfile(checkpoint):
        raise RuntimeError(f"not a readable NPZ: {checkpoint}")
    metadata = read_json(checkpoint.with_suffix(".json"))
    epoch = int(metadata.get("epoch", -1))
    expected_config = read_json(generated_config_path(seed, objective))
    actual_config = metadata.get("config", {})
    audit_checkpoint_config(expected_config, actual_config, epoch)
    if exact_epoch is not None and epoch != exact_epoch:
        raise RuntimeError(
            f"checkpoint epoch {metadata.get('epoch')} != {exact_epoch}"
        )
    return metadata


def parse_compute_apps_output(output: str) -> list[dict[str, Any]]:
    applications: list[dict[str, Any]] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "no running processes" in line.lower():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise RuntimeError(f"cannot parse nvidia-smi compute-app line: {line!r}")
        gpu_uuid, pid_text, process_name, memory_text = fields
        try:
            pid = int(pid_text)
        except ValueError as error:
            raise RuntimeError(f"invalid compute-app pid in {line!r}") from error
        try:
            memory_mib: int | str = int(memory_text)
        except ValueError:
            memory_mib = memory_text
        applications.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": memory_mib,
            }
        )
    return applications


def parse_pmon_output(output: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 9:
            raise RuntimeError(f"cannot parse nvidia-smi pmon line: {line!r}")
        try:
            gpu_index = int(fields[0])
            pid = int(fields[1])
        except ValueError as error:
            raise RuntimeError(f"invalid nvidia-smi pmon line: {line!r}") from error

        def metric(value: str) -> int | None:
            try:
                return int(value)
            except ValueError:
                return None

        records[pid] = {
            "gpu_index": gpu_index,
            "process_type": fields[2],
            "sm_percent": metric(fields[3]),
            "memory_percent": metric(fields[4]),
            "pmon_command": " ".join(fields[9:]) if len(fields) > 9 else fields[-1],
        }
    return records


def gpu_compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
        )
        pmon_output = subprocess.check_output(
            ["nvidia-smi", "pmon", "-c", "1"],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi compute-app gate failed") from error
    pmon = parse_pmon_output(pmon_output)
    blockers = []
    for application in parse_compute_apps_output(output):
        if int(application["pid"]) == os.getpid():
            continue
        usage = pmon.get(int(application["pid"]))
        numeric_memory = isinstance(application["used_gpu_memory_mib"], int)
        is_compute = bool(
            usage
            and (
                usage["process_type"] == "C"
                or (usage["sm_percent"] or 0) > 0
                or (usage["memory_percent"] or 0) > 0
            )
        )
        # Linux compute-app queries normally report numeric memory even when a
        # process is momentarily idle.  Windows WDDM also reports GUI apps in
        # this table with [N/A]; pmon's C/C+G type disambiguates those entries.
        if not is_compute and not numeric_memory:
            continue
        application["pmon"] = usage
        try:
            process = psutil.Process(int(application["pid"]))
            application["command"] = " ".join(process.cmdline())
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            application["command"] = None
        blockers.append(application)
    return blockers


def wait_for_gpu(seed: int, objective: str) -> None:
    while True:
        blockers = gpu_compute_processes()
        if not blockers:
            return
        update_state(
            status="waiting_for_gpu",
            current_seed=seed,
            current_objective=objective,
            blockers=blockers,
            gpu=gpu_snapshot(),
        )
        time.sleep(30.0)


def completion_ok(seed: int, objective: str) -> bool:
    completion = run_dir(seed, objective) / "completion.json"
    frozen = (
        run_dir(seed, objective)
        / "checkpoints/hardcoded_sss_frozen.npz"
    )
    if not completion.exists() or not frozen.exists():
        return False
    payload = read_json(completion)
    return (
        bool(payload.get("complete"))
        and int(payload.get("final_epoch", -1)) == TARGET_EPOCH
        and payload.get("basis_objective")
        == OBJECTIVES[objective]["objective"]
    )


def expected_sample_pack_startup_h2d_bytes(
    config: dict[str, Any],
) -> int:
    """Return the exact bytes in the one-shot train/validation/probe upload."""

    train_values = (
        int(config["epochs"]) * int(config["points_per_step"]) * 2
    )
    validation_pairs = int(config["validation_beta_samples"]) * int(
        config["validation_alpha_samples"]
    )
    validation_values = (
        validation_pairs * int(config["validation_points"]) * 2
    )
    probe_values = int(config["validation_points"]) * 2
    return (train_values + validation_values + probe_values) * np.dtype(
        np.float32
    ).itemsize


def audit_sample_pack_runtime_contract(
    protocol: dict[str, Any],
    history: list[dict[str, Any]],
    completion: dict[str, Any],
    *,
    expected_startup_h2d_bytes: int,
) -> dict[str, Any]:
    """Audit actual device residency and transfers, including resumed rows."""

    if not history:
        raise RuntimeError("sample-pack runtime audit requires non-empty history")
    required = set(SAMPLE_PACK_RUNTIME_EXPECTED) | set(
        SAMPLE_PACK_TIMING_KEYS
    ) | {"sample_pack_startup_h2d_bytes"}
    records = [
        ("protocol", protocol),
        *((f"history[{index}]", row) for index, row in enumerate(history)),
        ("completion", completion),
    ]
    for label, record in records:
        missing = required.difference(record)
        if missing:
            raise RuntimeError(
                f"{label} is missing sample-pack runtime fields: "
                f"{sorted(missing)}"
            )
        for key, expected in SAMPLE_PACK_RUNTIME_EXPECTED.items():
            if record[key] != expected:
                raise RuntimeError(
                    f"{label} {key}={record[key]!r}, expected {expected!r}"
                )
        observed_bytes = int(record["sample_pack_startup_h2d_bytes"])
        if observed_bytes != int(expected_startup_h2d_bytes):
            raise RuntimeError(
                f"{label} startup H2D bytes {observed_bytes} != "
                f"{expected_startup_h2d_bytes}"
            )
        for key in SAMPLE_PACK_TIMING_KEYS:
            seconds = float(record[key])
            if not math.isfinite(seconds) or seconds < 0.0:
                raise RuntimeError(
                    f"{label} {key} must be finite and non-negative"
                )

    # The exclusion flag describes timer scope and is emitted at the two
    # run-level boundaries.  Per-epoch rows carry the separately measured
    # load/upload durations but intentionally avoid repeating this prose-level
    # contract flag.
    for label, record in (("protocol", protocol), ("completion", completion)):
        if record.get(SAMPLE_PACK_TIMING_EXCLUSION_KEY) is not True:
            raise RuntimeError(
                f"{label} does not exclude sample-pack startup from "
                "training_elapsed_seconds"
            )

    # The protocol is rewritten at each same-arm resume and completion is
    # emitted by the final invocation.  Its final history row must therefore
    # carry the exact same separately timed startup measurements.
    final_row = history[-1]
    for key in (*SAMPLE_PACK_TIMING_KEYS, "sample_pack_startup_h2d_bytes"):
        if protocol[key] != final_row[key] or completion[key] != final_row[key]:
            raise RuntimeError(
                f"final sample-pack runtime field {key} is inconsistent "
                "between protocol/history/completion"
            )
    return {
        **SAMPLE_PACK_RUNTIME_EXPECTED,
        SAMPLE_PACK_TIMING_EXCLUSION_KEY: True,
        "sample_pack_startup_h2d_bytes": int(
            final_row["sample_pack_startup_h2d_bytes"]
        ),
        "sample_pack_load_seconds": float(
            final_row["sample_pack_load_seconds"]
        ),
        "sample_pack_upload_seconds": float(
            final_row["sample_pack_upload_seconds"]
        ),
        "history_rows_audited": len(history),
    }


def audit_complete_arm(seed: int, objective: str) -> dict[str, Any]:
    directory = run_dir(seed, objective)
    protocol = read_json(directory / "hardcoded_protocol.json")
    completion = read_json(directory / "completion.json")
    history = read_json(
        directory / "data/hardcoded_training_history.json"
    )
    if len(history) != TARGET_EPOCH or int(history[-1]["epoch"]) != TARGET_EPOCH:
        raise RuntimeError(f"seed {seed} {objective} history is not 2000 epochs")
    row = history[-1]
    required_finite = (
        "validation_loss",
        "validation_loss_p95",
        "validation_loss_max",
        "validation_orth_error",
        "training_elapsed_seconds",
    )
    if not all(math.isfinite(float(row[key])) for key in required_finite):
        raise FloatingPointError(f"seed {seed} {objective} non-finite final row")
    expected_jet = int(OBJECTIVES[objective]["jet_order"])
    if int(row["spatial_jet_order_used"]) != expected_jet:
        raise RuntimeError(f"seed {seed} {objective} did not use Jet{expected_jet}")
    if int(row["train_peeling_passes_used"]) != 2:
        raise RuntimeError(f"seed {seed} {objective} did not use two-pass MGS")
    if str(row["causal_sort_device"]) != "gpu":
        raise RuntimeError(f"seed {seed} {objective} causal sort left the GPU")
    config = read_json(generated_config_path(seed, objective))
    sample_pack_runtime = audit_sample_pack_runtime_contract(
        protocol,
        history,
        completion,
        expected_startup_h2d_bytes=(
            expected_sample_pack_startup_h2d_bytes(config)
        ),
    )
    archives: dict[str, Any] = {}
    for epoch in ARCHIVE_EPOCHS:
        checkpoint = directory / f"checkpoints/hardcoded_epoch{epoch:04d}.npz"
        validate_checkpoint(
            seed, objective, checkpoint, exact_epoch=epoch
        )
        archives[str(epoch)] = {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
        }
    frozen = directory / "checkpoints/hardcoded_sss_frozen.npz"
    validate_checkpoint(seed, objective, frozen, exact_epoch=TARGET_EPOCH)
    return {
        "final_epoch": TARGET_EPOCH,
        "validation_mean": float(row["validation_loss"]),
        "validation_p95": float(row["validation_loss_p95"]),
        "validation_max": float(row["validation_loss_max"]),
        "peeled_rank": int(row["validation_peeled_rank"]),
        "Gram_max_abs_defect": float(row["validation_orth_error"]),
        "training_elapsed_seconds": float(row["training_elapsed_seconds"]),
        "spatial_jet_order_used": int(row["spatial_jet_order_used"]),
        "peeling_passes_used": int(row["train_peeling_passes_used"]),
        "causal_sort_device": str(row["causal_sort_device"]),
        "optimizer_device": str(row.get("optimizer_device", "")),
        "sample_pack_runtime": sample_pack_runtime,
        "archives": archives,
        "causal_prefixes": list(PREFIXES),
    }


def train_arm(
    seed: int,
    objective: str,
    training: Path,
    validation: Path,
) -> dict[str, Any]:
    initial_pack_audit = audit_seed_packs(seed, training, validation)
    if completion_ok(seed, objective):
        metrics = audit_complete_arm(seed, objective)
        update_arm(
            seed,
            objective,
            status="complete",
            last_sample_pack_audit=initial_pack_audit,
            final_metrics=metrics,
        )
        return metrics
    initial = ensure_arm_initial(seed, objective)
    expected_initial_digest = sha256(initial)
    update_arm(
        seed,
        objective,
        status="ready",
        initial_checkpoint=str(initial.resolve()),
        initial_archive_sha256=expected_initial_digest,
        shared_train_pack=str(training.resolve()),
        shared_train_pack_sha256=initial_pack_audit["train_pack_sha256"],
        shared_validation_pack=str(validation.resolve()),
        shared_validation_pack_sha256=initial_pack_audit[
            "validation_pack_sha256"
        ],
        last_sample_pack_audit=initial_pack_audit,
    )
    for attempt in range(1, 4):
        # Re-read both NPZ files, recompute both hashes, and redo the full
        # shape/finite/FP32/solid audit before every attempt (including retry).
        attempt_pack_audit = audit_seed_packs(seed, training, validation)
        wait_for_gpu(seed, objective)
        latest = run_dir(seed, objective) / "checkpoints/hardcoded_latest.npz"
        resume = latest if latest.exists() else initial
        metadata = validate_checkpoint(seed, objective, resume)
        resume_epoch = int(metadata["epoch"])
        if not 0 <= resume_epoch < TARGET_EPOCH:
            raise RuntimeError(f"invalid same-arm resume epoch {resume_epoch}")
        history_path = (
            run_dir(seed, objective)
            / "data/hardcoded_training_history.json"
        )
        elapsed_floor = 0.0
        if history_path.exists():
            history = read_json(history_path)
            kept = [
                row for row in history if int(row.get("epoch", -1)) <= resume_epoch
            ]
            if len(kept) != len(history):
                write_json(history_path, kept)
            if kept:
                elapsed_floor = float(kept[-1]["training_elapsed_seconds"])
        update_state(
            status="training",
            current_seed=seed,
            current_objective=objective,
            attempt=attempt,
            resume_epoch=resume_epoch,
            gpu=gpu_snapshot(),
            last_sample_pack_audit=attempt_pack_audit,
        )
        update_arm(
            seed,
            objective,
            status="training",
            attempt=attempt,
            resume_epoch=resume_epoch,
            resume_checkpoint=str(resume.resolve()),
            recovery_policy_verified="same-arm only",
            sample_pack_audit=attempt_pack_audit,
            training_started_at=now(),
            training_started_unix_ns=time.time_ns(),
        )
        try:
            run(
                [
                    sys.executable,
                    "-u",
                    str(CASE_ROOT / "scripts/train_hardcoded.py"),
                    "--resume",
                    str(resume),
                    "--run-dir",
                    str(run_dir(seed, objective)),
                    "--epochs",
                    str(TARGET_EPOCH),
                    "--sample-pack",
                    str(training),
                    "--validation-pack",
                    str(validation),
                    "--training-elapsed-floor",
                    str(elapsed_floor),
                    "--extra-archive-epochs",
                    *[str(epoch) for epoch in ARCHIVE_EPOCHS],
                ]
            )
            metrics = audit_complete_arm(seed, objective)
            update_arm(seed, objective, status="complete", final_metrics=metrics)
            return metrics
        except Exception as error:
            update_arm(
                seed,
                objective,
                status="attempt_failed",
                attempt=attempt,
                error=repr(error),
            )
            if attempt == 3:
                raise
            time.sleep(10.0)
    raise AssertionError("unreachable")


def assert_paired_initializations(seed: int) -> dict[str, Any]:
    canonical, _ = ensure_canonical_initial(seed)
    digests = {
        objective: sha256(ensure_arm_initial(seed, objective))
        for objective in TRAIN_OBJECTIVES
    }
    reference = audit_reference_potential(canonical)
    digests["potential_reference"] = reference["initial_archive_sha256"]
    if len(set(digests.values())) != 1:
        raise RuntimeError(f"seed {seed} initial model arrays differ: {digests}")
    return {
        "sha256": next(iter(digests.values())),
        "archives": digests,
        "reference": reference,
    }


def active_duplicate_runner() -> dict[str, Any] | None:
    if not PID_PATH.exists():
        return None
    record = read_json(PID_PATH)
    pid = int(record.get("pid", -1))
    if pid <= 0 or pid == os.getpid() or not psutil.pid_exists(pid):
        return None
    try:
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return None
    if Path(__file__).name.lower() in command.lower():
        return {"pid": pid, "command": command}
    return None


def execute(audit: dict[str, Any]) -> None:
    duplicate = active_duplicate_runner()
    if duplicate:
        raise RuntimeError(f"strict runner already active: {duplicate}")
    generated = make_generated_configs()
    ensure_manifest(audit, generated)
    write_json(
        PID_PATH,
        {
            "pid": os.getpid(),
            "started_at": now(),
            "command": sys.argv,
        },
    )
    all_metrics: dict[str, Any] = {}
    try:
        for seed in PAIRED_SEEDS:
            training, validation, sample_manifest = ensure_pack(seed)
            sample_schedule_audit = audit_reference_sample_schedule(
                seed, training
            )
            initialization_audit = assert_paired_initializations(seed)
            manifest = read_json(MANIFEST_PATH)
            manifest["reference_baseline"] = initialization_audit["reference"]
            manifest["reference_sample_schedule_audit"] = (
                sample_schedule_audit
            )
            manifest["updated_at"] = now()
            manifest["updated_unix_ns"] = time.time_ns()
            write_json(MANIFEST_PATH, manifest)
            all_metrics[str(seed)] = {
                "paired_initial_archive_sha256": initialization_audit[
                    "sha256"
                ],
                "paired_initial_archives": initialization_audit["archives"],
                "potential_reference": initialization_audit["reference"],
                "reference_sample_schedule_audit": sample_schedule_audit,
                "shared_train_pack_sha256": sample_manifest["train_pack"][
                    "sha256"
                ],
                "shared_validation_pack_sha256": sample_manifest[
                    "validation_pack"
                ]["sha256"],
                "objectives": {},
            }
            for objective in OBJECTIVE_ORDER_BY_SEED[seed]:
                metrics = train_arm(
                    seed,
                    objective,
                    training,
                    validation,
                )
                all_metrics[str(seed)]["objectives"][objective] = metrics
                completed = [
                    f"{s}:{name}"
                    for s in PAIRED_SEEDS
                    for name in OBJECTIVE_ORDER_BY_SEED[s]
                    if completion_ok(s, name)
                ]
                update_state(
                    status="arm_complete",
                    current_seed=seed,
                    current_objective=objective,
                    completed_arms=completed,
                )
        manifest = read_json(MANIFEST_PATH)
        manifest.update(
            status="complete",
            completed_at=now(),
            completed_unix_ns=time.time_ns(),
            final_metrics=all_metrics,
        )
        write_json(MANIFEST_PATH, manifest)
        update_state(
            status="complete",
            current_seed=None,
            current_objective=None,
            completed_arms=[
                f"{seed}:{objective}"
                for seed in PAIRED_SEEDS
                for objective in OBJECTIVE_ORDER_BY_SEED[seed]
            ],
        )
    finally:
        if PID_PATH.exists():
            record = read_json(PID_PATH)
            if int(record.get("pid", -1)) == os.getpid():
                PID_PATH.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Verify the Potential reference, materialize its paired packs, "
            "and train the Velocity/Vorticity arms."
        ),
    )
    args = parser.parse_args()
    audit = audit_templates()
    print(json.dumps(audit, indent=2), flush=True)
    if args.execute:
        execute(audit)
    else:
        print(
            "AUDIT ONLY: no sample packs, checkpoints, or training processes "
            "were created; pass --execute to start.",
            flush=True,
        )


if __name__ == "__main__":
    main()
