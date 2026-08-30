"""Materialize the completed Potential run's sampling schedule for reuse.

The shared pack reproduces the original K=2048 Potential run exactly: geometry
parameters use its deterministic scrambled-Sobol schedule, while fluid points
use its NumPy ``default_rng``/PCG64 rejection sampler.  Velocity and Vorticity
read the same immutable files, so they receive the Potential seed's identical
geometry and point sequence rather than merely sharing an integer seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import scipy


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

GENERATOR_PATH = Path(__file__).resolve()
TRAINER_PATH = CASE_ROOT / "scripts/train_hardcoded.py"
BACKEND_PATH = (
    CASE_ROOT
    / "src/eigenfluid/gen/airfoil2d/hardcoded_backend.py"
)

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    cfg_namespace,
    continuous_parameters,
    fluid_points,
    inside_airfoil,
)


TRAIN_NAME = "paired_train_pcg64_iid_e2000_p4096.npz"
VALIDATION_NAME = "paired_validation_pcg64_iid_p4096.npz"
MANIFEST_NAME = "paired_sample_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def audit_config(config: dict[str, Any]) -> None:
    required = {
        "modes": 2048,
        "epochs": 2000,
        "points_per_step": 4096,
        "validation_points": 4096,
        "validation_beta_samples": 3,
        "validation_alpha_samples": 3,
        "sampling_mode": "uniform_random",
        "quadrature_measure": "physical_fluid_area",
        "geometry_mode": "cinf_ellipse",
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"strict paired-pack config mismatch: {mismatches}")


def pcg64_fluid_points(
    cfg: SimpleNamespace,
    count: int,
    beta: float,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Return the original Potential sampler's exact FP32 training input."""

    # ``fluid_points`` draws all x coordinates and then all y coordinates from
    # NumPy default_rng (PCG64), applies the solid rejection in FP64, and is the
    # path used by the completed Potential run.  Cast only after acceptance,
    # exactly where the old trainer transferred inputs to its FP32 backend.
    points = np.asarray(
        fluid_points(cfg, int(count), float(beta), float(alpha), int(seed)),
        dtype=np.float32,
    )
    if points.shape != (count, 2) or points.dtype != np.float32:
        raise RuntimeError("failed to reproduce the Potential PCG64 sample")
    if np.any(inside_airfoil(cfg, points, float(beta), float(alpha))):
        raise RuntimeError("Potential PCG64 sample moved inside after FP32 cast")
    return points


def potential_parameters(
    cfg: SimpleNamespace, epoch: int
) -> tuple[float, float]:
    """Return the completed Potential run's exact geometry parameters."""

    beta, alpha = continuous_parameters(cfg, int(epoch))
    return float(beta), float(alpha)


def source_lock(config_path: Path) -> dict[str, Any]:
    return {
        "generator": {
            "path": str(GENERATOR_PATH),
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


def pack_contract(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "seed",
        "epochs",
        "points_per_step",
        "validation_points",
        "validation_beta_samples",
        "validation_alpha_samples",
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "chord",
        "quarter_chord_x",
        "quarter_chord_y",
        "thickness_min",
        "thickness_max",
        "geometry_mode",
        "alpha_min_deg",
        "alpha_max_deg",
        "endpoint_period",
        "sampling_mode",
        "quadrature_measure",
        "probe_beta",
        "probe_alpha_deg",
    )
    return {key: config[key] for key in keys}


def verify_existing(
    output_dir: Path,
    expected_contract: dict[str, Any],
    expected_source_lock: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sampling_contract") != expected_contract:
        raise RuntimeError("existing paired sample manifest has another contract")
    if manifest.get("source_lock") != expected_source_lock:
        raise RuntimeError("existing paired sample manifest source lock changed")
    for key in ("train_pack", "validation_pack"):
        record = manifest[key]
        path = output_dir / record["name"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"existing {key} is missing or has a hash mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    audit_config(config)
    cfg = cfg_namespace(config)
    contract = pack_contract(config)
    expected_source_lock = source_lock(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = verify_existing(args.output_dir, contract, expected_source_lock)
    if existing is not None:
        print(json.dumps(existing, indent=2))
        return
    if args.verify_only:
        raise FileNotFoundError(args.output_dir / MANIFEST_NAME)

    started = time.perf_counter()
    # Keep training geometry scalars in FP64 because the completed Potential
    # run passed the Python floats returned by ``continuous_parameters`` to the
    # backend.  Coordinates are FP32, matching the actual network input.
    train_beta = np.empty(cfg.epochs, dtype=np.float64)
    train_alpha = np.empty(cfg.epochs, dtype=np.float64)
    train_points = np.empty(
        (cfg.epochs, cfg.points_per_step, 2), dtype=np.float32
    )
    for epoch in range(1, cfg.epochs + 1):
        beta, alpha = potential_parameters(cfg, epoch)
        train_beta[epoch - 1] = beta
        train_alpha[epoch - 1] = alpha
        train_points[epoch - 1] = pcg64_fluid_points(
            cfg,
            cfg.points_per_step,
            beta,
            alpha,
            cfg.seed + epoch * 1009,
        )
        if epoch % 100 == 0:
            print(f"generated train epoch {epoch}/{cfg.epochs}", flush=True)

    pairs = [
        (float(beta), float(alpha))
        for beta in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for alpha in np.linspace(
            cfg.alpha_min_deg,
            cfg.alpha_max_deg,
            cfg.validation_alpha_samples,
        )
    ]
    validation_beta = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
    validation_alpha = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
    validation_points = np.stack(
        [
            pcg64_fluid_points(
                cfg,
                cfg.validation_points,
                float(validation_beta[index]),
                float(validation_alpha[index]),
                cfg.seed + 900_000 + index,
            )
            for index in range(len(pairs))
        ],
        axis=0,
    )
    probe_beta = float(np.float32(cfg.probe_beta))
    probe_alpha = float(np.float32(cfg.probe_alpha_deg))
    probe_points = pcg64_fluid_points(
        cfg,
        cfg.validation_points,
        probe_beta,
        probe_alpha,
        cfg.seed + 1_500_000,
    )

    train_arrays = {
        "train_points": train_points,
        "train_beta": train_beta,
        "train_alpha": train_alpha,
        "train_epoch_1based": np.arange(1, cfg.epochs + 1, dtype=np.int32),
    }
    validation_arrays = {
        "validation_points": validation_points,
        "validation_beta": validation_beta,
        "validation_alpha": validation_alpha,
        "probe_points": probe_points,
    }
    train_path = args.output_dir / TRAIN_NAME
    validation_path = args.output_dir / VALIDATION_NAME
    save_npz(train_path, train_arrays)
    save_npz(validation_path, validation_arrays)
    manifest = {
        "format": "airfoil_potential_schedule_shared_pack_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "sampling_contract": contract,
        "source_lock": expected_source_lock,
        "paired_objectives_read_identical_files": True,
        "geometry_parameter_schedule": (
            "potential_scrambled_sobol_seed_plus_700000_plus_epoch"
        ),
        "fluid_point_sampler": (
            "potential_numpy_default_rng_pcg64_iid_uniform_physical_fluid_area_v1"
        ),
        "airfoil_rejection_note": (
            "The completed Potential path rejects FP64 rectangle samples "
            "against the solid ellipse, replenishes from the same PCG64 "
            "stream, and casts the accepted coordinates to FP32 for training."
        ),
        "dtypes": {
            "train_points": "float32",
            "train_geometry_parameters": "float64",
            "validation_and_probe": "float32",
        },
        "train_pack": {
            "name": train_path.name,
            "sha256": sha256(train_path),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": array_sha256(value),
                }
                for key, value in train_arrays.items()
            },
        },
        "validation_pack": {
            "name": validation_path.name,
            "sha256": sha256(validation_path),
            "arrays": {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": array_sha256(value),
                }
                for key, value in validation_arrays.items()
            },
        },
        "generation_wall_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / MANIFEST_NAME, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
