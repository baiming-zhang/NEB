"""Static and small-data tests for the strict paired K2048 protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


CASE_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, CASE_ROOT / "scripts" / filename
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_templates_are_mechanism_only_and_share_potential_seed() -> None:
    runner = load_script(
        "strict_rayleigh_runner_audit",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    audit = runner.audit_templates()
    assert audit["status"] == "strict_protocol_audited"
    assert audit["causal_prefixes"] == [64, 128, 256, 512, 1024, 2048]
    assert audit["paired_seeds"] == [20260823]
    for changed in audit["pairwise_config_differences"].values():
        assert set(changed) <= {
            "basis_objective",
            "spatial_derivative_order",
            "result_root",
        }
    assert audit["objective_order_by_seed"] == {
        "20260823": ["velocity", "vorticity"]
    }
    assert audit["reference_objective"] == "potential"
    assert audit["trained_objectives"] == ["velocity", "vorticity"]
    assert "immutable reference" in audit["order_design"]
    assert audit["source_lock"]["runner"]["sha256"]
    assert audit["source_lock"]["generator"]["sha256"]
    assert audit["source_lock"]["trainer"]["sha256"]
    assert audit["source_lock"]["backend"]["sha256"]
    residency = audit["sample_pack_runtime_contract"]
    assert residency["sample_pack_device"] == "gpu_resident"
    assert residency["sample_pack_startup_h2d_transfers"] == 1
    assert residency["input_host_to_device_transfers_per_step"] == 0
    assert residency["optimizer_host_transfers_per_step"] == 0
    assert residency["diagnostic_host_transfers_per_step"] == 1


def test_completed_potential_is_a_hashed_external_reference() -> None:
    runner = load_script(
        "strict_rayleigh_runner_reference",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    reference = runner.audit_reference_potential()
    assert reference["status"] == "complete_reference_verified"
    assert reference["role"] == "immutable_external_reference_not_retrained"
    assert reference["seed"] == 20260823
    assert reference["final_epoch"] == 2000
    assert reference["initial_archive_sha256"] == (
        runner.REFERENCE_POTENTIAL_INITIAL_SHA256
    )
    assert reference["frozen_checkpoint_sha256"] == (
        runner.REFERENCE_POTENTIAL_FROZEN_SHA256
    )
    assert reference["canonical_initialization_exact_match"] is None


def test_checkpoint_config_allows_only_bounded_learning_rate() -> None:
    runner = load_script(
        "strict_rayleigh_runner_checkpoint",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    expected = {
        "seed": 7,
        "modes": 2048,
        "learning_rate": 2.0e-4,
        "base_learning_rate": 2.0e-4,
        "min_learning_rate": 1.0e-6,
    }
    assert runner.audit_checkpoint_config(expected, dict(expected), 0)[
        "all_immutable_fields_exact"
    ]
    scheduled = dict(expected, learning_rate=7.5e-5)
    audit = runner.audit_checkpoint_config(expected, scheduled, 500)
    assert audit["observed_mutable_differences"] == ["learning_rate"]
    with pytest.raises(RuntimeError, match="immutable"):
        runner.audit_checkpoint_config(
            expected, dict(scheduled, modes=1024), 500
        )
    with pytest.raises(RuntimeError, match="outside"):
        runner.audit_checkpoint_config(
            expected, dict(expected, learning_rate=8.0e-4), 500
        )
    with pytest.raises(RuntimeError, match="epoch-0"):
        runner.audit_checkpoint_config(expected, scheduled, 0)


def test_compute_app_parser_is_not_command_name_based() -> None:
    runner = load_script(
        "strict_rayleigh_runner_gpu",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    output = (
        "GPU-a, 1234, python.exe, 24000\n"
        "GPU-b, 5678, completely-unrecognized-worker.exe, 1024\n"
    )
    applications = runner.parse_compute_apps_output(output)
    assert [item["pid"] for item in applications] == [1234, 5678]
    assert applications[1]["process_name"] == "completely-unrecognized-worker.exe"
    assert runner.parse_compute_apps_output("") == []
    pmon = runner.parse_pmon_output(
        "# gpu pid type sm mem enc dec jpg ofa command\n"
        "0 1234 C 97 100 - - - - python.exe\n"
        "0 5678 C+G - - - - - - desktop.exe\n"
    )
    assert pmon[1234]["process_type"] == "C"
    assert pmon[1234]["sm_percent"] == 97
    assert pmon[5678]["sm_percent"] is None


def test_gpu_resident_pack_runtime_contract_and_exact_bytes() -> None:
    runner = load_script(
        "strict_rayleigh_runner_pack_runtime",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    config = json.loads(
        (
            CASE_ROOT
            / "configs/k2048_airfoil_strict_potential_rayleigh_2000.json"
        ).read_text(encoding="utf-8")
    )
    expected_bytes = runner.expected_sample_pack_startup_h2d_bytes(config)
    assert expected_bytes == (
        (2000 * 4096 * 2 + 9 * 4096 * 2 + 4096 * 2)
        * np.dtype(np.float32).itemsize
    )
    base = {
        **runner.SAMPLE_PACK_RUNTIME_EXPECTED,
        "sample_pack_startup_h2d_bytes": expected_bytes,
        "sample_pack_load_seconds": 0.25,
        "sample_pack_upload_seconds": 0.5,
    }
    run_level = {
        **base,
        runner.SAMPLE_PACK_TIMING_EXCLUSION_KEY: True,
    }
    prior_resume_row = {
        **base,
        "sample_pack_load_seconds": 0.2,
        "sample_pack_upload_seconds": 0.4,
    }
    audit = runner.audit_sample_pack_runtime_contract(
        dict(run_level),
        [prior_resume_row, dict(base)],
        dict(run_level),
        expected_startup_h2d_bytes=expected_bytes,
    )
    assert audit["sample_pack_device"] == "gpu_resident"
    assert audit["sample_pack_startup_h2d_transfers"] == 1
    assert audit["input_host_to_device_transfers_per_step"] == 0
    assert audit["history_rows_audited"] == 2
    with pytest.raises(RuntimeError, match="expected 0"):
        bad_completion = dict(
            run_level, input_host_to_device_transfers_per_step=1
        )
        runner.audit_sample_pack_runtime_contract(
            dict(run_level),
            [dict(base)],
            bad_completion,
            expected_startup_h2d_bytes=expected_bytes,
        )
    with pytest.raises(RuntimeError, match="H2D bytes"):
        runner.audit_sample_pack_runtime_contract(
            dict(run_level),
            [dict(base)],
            dict(run_level),
            expected_startup_h2d_bytes=expected_bytes + 4,
        )


def test_pcg64_sampler_masks_after_float32_storage() -> None:
    generator = load_script(
        "strict_rayleigh_pack_sampler",
        "generate_airfoil_strict_paired_pcg64_pack.py",
    )
    config = json.loads(
        (
            CASE_ROOT
            / "configs/k2048_airfoil_strict_potential_rayleigh_2000.json"
        ).read_text(encoding="utf-8")
    )
    cfg = generator.cfg_namespace(config)
    beta = float(np.float32(0.37654321))
    alpha = float(np.float32(7.1234567))
    points = generator.pcg64_fluid_points(cfg, 64, beta, alpha, 12345)
    repeated = generator.pcg64_fluid_points(cfg, 64, beta, alpha, 12345)
    assert points.shape == (64, 2)
    assert points.dtype == np.float32
    assert np.array_equal(points, repeated)
    assert not np.any(generator.inside_airfoil(cfg, points, beta, alpha))


def test_small_pack_content_audit_rejects_float32_solid_point(
    tmp_path: Path,
) -> None:
    runner = load_script(
        "strict_rayleigh_runner_pack",
        "run_airfoil_strict_rayleigh_k2048_paired_e2000.py",
    )
    generator = load_script(
        "strict_rayleigh_pack_small",
        "generate_airfoil_strict_paired_pcg64_pack.py",
    )
    config = json.loads(
        (
            CASE_ROOT
            / "configs/k2048_airfoil_strict_potential_rayleigh_2000.json"
        ).read_text(encoding="utf-8")
    )
    config.update(
        epochs=2,
        points_per_step=8,
        validation_points=8,
        validation_beta_samples=1,
        validation_alpha_samples=1,
    )
    cfg = generator.cfg_namespace(config)
    train_beta = np.asarray([0.5, 0.5], dtype=np.float64)
    train_alpha = np.asarray([0.0, 0.0], dtype=np.float64)
    train_points = np.stack(
        [
            generator.pcg64_fluid_points(
                cfg,
                8,
                float(train_beta[index]),
                float(train_alpha[index]),
                100 + index,
            )
            for index in range(2)
        ]
    )
    validation_beta = np.asarray([0.5], dtype=np.float32)
    validation_alpha = np.asarray([0.0], dtype=np.float32)
    validation_points = generator.pcg64_fluid_points(
        cfg, 8, 0.5, 0.0, 200
    )[None, ...]
    probe_points = generator.pcg64_fluid_points(cfg, 8, 0.5, 6.0, 300)
    training = tmp_path / "train.npz"
    validation = tmp_path / "validation.npz"
    np.savez(
        training,
        train_points=train_points,
        train_beta=train_beta,
        train_alpha=train_alpha,
        train_epoch_1based=np.arange(1, 3, dtype=np.int32),
    )
    np.savez(
        validation,
        validation_points=validation_points,
        validation_beta=validation_beta,
        validation_alpha=validation_alpha,
        probe_points=probe_points,
    )
    audit = runner.audit_sample_pack_contents(config, training, validation)
    assert audit["all_points_outside_solid_after_float32_storage"]
    assert audit["train_points_are_float32"]
    assert audit["train_geometry_scalars_are_float64"]

    np.savez(
        training,
        train_points=train_points,
        train_beta=train_beta.astype(np.float32),
        train_alpha=train_alpha.astype(np.float32),
        train_epoch_1based=np.arange(1, 3, dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="train_beta dtype.*not float64"):
        runner.audit_sample_pack_contents(config, training, validation)

    train_points[0, 0] = np.asarray([2.5, 0.0], dtype=np.float32)
    np.savez(
        training,
        train_points=train_points,
        train_beta=train_beta,
        train_alpha=train_alpha,
        train_epoch_1based=np.arange(1, 3, dtype=np.int32),
    )
    with pytest.raises(RuntimeError, match="solid point"):
        runner.audit_sample_pack_contents(config, training, validation)
