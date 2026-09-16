from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (  # noqa: E402
    encoding_jet1,
    encoding_jet2,
    envelope_jet1,
    envelope_jet2,
    fluid_points,
    xy_jets1,
    xy_jets2,
)


def free_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        xmin=0.0,
        xmax=2.0 * math.pi,
        ymin=0.0,
        ymax=2.0 * math.pi,
        geometry_mode="free_box",
        envelope_mode="free_identity",
        input_encoding="raw_xy",
        frequencies=6,
        sampling_mode="uniform_random",
    )


def test_raw_xy_encoding_has_exact_physical_values_and_identity_jets() -> None:
    cfg = free_cfg()
    points = np.asarray(
        [
            [0.0, 0.37],
            [2.0 * math.pi, 0.37],
            [1.19, 0.0],
            [1.19, 2.0 * math.pi],
        ],
        dtype=np.float32,
    )
    x1, y1 = xy_jets1(points)
    encoded1 = encoding_jet1(cfg, x1, y1, 0.0, 0.0)
    envelope1 = envelope_jet1(cfg, x1, y1, 0.0, 0.0)
    x2, y2 = xy_jets2(points)
    encoded2 = encoding_jet2(cfg, x2, y2, 0.0, 0.0)
    envelope2 = envelope_jet2(cfg, x2, y2, 0.0, 0.0)
    cp.testing.assert_allclose(encoded1.value, cp.asarray(points), atol=0.0)
    cp.testing.assert_allclose(encoded2.value, cp.asarray(points), atol=0.0)
    expected_gradient = cp.asarray(
        [[[1.0, 0.0], [0.0, 1.0]]] * len(points), dtype=cp.float32
    )
    cp.testing.assert_array_equal(encoded1.grad, expected_gradient)
    cp.testing.assert_array_equal(encoded2.grad, expected_gradient)
    cp.testing.assert_array_equal(encoded2.hess, cp.zeros_like(encoded2.hess))
    cp.testing.assert_array_equal(envelope1.value, cp.ones_like(envelope1.value))
    cp.testing.assert_array_equal(envelope1.grad, cp.zeros_like(envelope1.grad))
    cp.testing.assert_array_equal(envelope2.hess, cp.zeros_like(envelope2.hess))


def test_free_box_sampler_has_no_solid_rejection() -> None:
    cfg = free_cfg()
    points = fluid_points(cfg, 2048, 0.0, 0.0, 17)
    assert points.shape == (2048, 2)
    assert np.all(points >= 0.0)
    assert np.all(points[:, 0] < 2.0 * math.pi)
    assert np.all(points[:, 1] < 2.0 * math.pi)
