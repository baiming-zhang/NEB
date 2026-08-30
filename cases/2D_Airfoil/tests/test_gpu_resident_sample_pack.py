"""Transport regression tests for immutable paired PCG64 sample packs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


CASE_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = CASE_ROOT / "scripts" / "train_hardcoded.py"


def _trainer():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("sample-pack transport test requires a CUDA device")
        cp.cuda.Device().use()
    except Exception as error:  # pragma: no cover - depends on test host
        pytest.skip(f"sample-pack transport test requires working CUDA: {error}")
    specification = importlib.util.spec_from_file_location(
        "airfoil_train_hardcoded_sample_pack_test", TRAINER_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load train_hardcoded.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return cp, module


def test_pcg64_point_pack_is_bitwise_identical_and_resume_index_stable() -> None:
    cp, trainer = _trainer()
    rng = np.random.Generator(np.random.PCG64(20260830))
    train_points = rng.uniform(-4.0, 10.0, (7, 8, 2)).astype(np.float32)
    validation_points = rng.uniform(-4.0, 10.0, (3, 8, 2)).astype(np.float32)
    probe_points = rng.uniform(-4.0, 10.0, (8, 2)).astype(np.float32)
    train_beta = rng.uniform(0.0, 1.0, 7).astype(np.float32)
    train_alpha = rng.uniform(-12.0, 12.0, 7).astype(np.float32)
    training = {
        "train_points": train_points,
        "train_beta": train_beta,
        "train_alpha": train_alpha,
        "train_epoch_1based": np.arange(1, 8, dtype=np.int32),
    }
    validation = {
        "validation_points": validation_points,
        "validation_beta": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "validation_alpha": np.asarray([-12.0, 0.0, 12.0], dtype=np.float32),
        "probe_points": probe_points,
    }

    (resident, resident_validation), storage, audit = (
        trainer.upload_sample_pack_points(training, validation)
    )
    assert storage is not None
    assert resident is not None
    assert resident_validation is not None
    assert audit == {
        "sample_pack_device": "gpu_resident",
        "sample_pack_startup_h2d_transfers": 1,
        "sample_pack_startup_h2d_bytes": (
            train_points.nbytes
            + validation_points.nbytes
            + probe_points.nbytes
        ),
    }
    assert isinstance(resident["train_beta"], np.ndarray)
    assert isinstance(resident["train_alpha"], np.ndarray)

    for device, host in (
        (resident["train_points"], train_points),
        (resident_validation["validation_points"], validation_points),
        (resident_validation["probe_points"], probe_points),
    ):
        recovered = cp.asnumpy(device)
        assert recovered.dtype == host.dtype
        assert recovered.shape == host.shape
        assert recovered.tobytes(order="C") == host.tobytes(order="C")
        assert cp.shares_memory(device, storage)

    # A checkpoint at epoch 3 resumes with epoch 4, hence the unchanged
    # zero-based pack index 3.  Device residency must not alter this mapping.
    resumed_epoch = 4
    resumed_index = resumed_epoch - 1
    recovered_epoch = cp.asnumpy(resident["train_points"][resumed_index])
    assert recovered_epoch.tobytes() == train_points[resumed_index].tobytes()
    assert float(resident["train_beta"][resumed_index]) == float(
        train_beta[resumed_index]
    )
    assert float(resident["train_alpha"][resumed_index]) == float(
        train_alpha[resumed_index]
    )
