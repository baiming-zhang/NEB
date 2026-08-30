"""Validate raw-coordinate SIREN omega0 scaling in the square Jet3 encoding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np

from square_runtime import backend, square_encoding_jet


def main() -> None:
    points = np.asarray([[0.1, 0.2], [0.5, 0.75], [0.9, 0.4]], dtype=np.float32)
    x, y = backend.xy_jets(points, backend.F32)
    records = []
    for omega0 in (1.0, 8.0, 16.0, 32.0):
        cfg = SimpleNamespace(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            frequencies=6,
            input_encoding="raw_xy_padded",
            siren_first_layer_spatial_omega0=omega0,
        )
        encoded = square_encoding_jet(cfg, x, y, 0.0, 0.0)
        value = cp.asnumpy(encoded.value)
        gradient = cp.asnumpy(encoded.grad)
        hessian = cp.asnumpy(encoded.hess)
        third = cp.asnumpy(encoded.third)
        expected = omega0 * (2.0 * points - 1.0)
        np.testing.assert_allclose(value[:, :2], expected, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(gradient[:, 0, 0], 2.0 * omega0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(gradient[:, 1, 1], 2.0 * omega0, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(gradient[:, 0, 1], 0.0)
        np.testing.assert_array_equal(gradient[:, 1, 0], 0.0)
        np.testing.assert_array_equal(value[:, 2:], 0.0)
        np.testing.assert_array_equal(hessian, 0.0)
        np.testing.assert_array_equal(third, 0.0)
        records.append({
            "omega0": omega0,
            "maximum_value_error": float(np.max(np.abs(value[:, :2] - expected))),
            "active_spatial_gradient": 2.0 * omega0,
            "hessian_exactly_zero": bool(np.all(hessian == 0.0)),
            "third_exactly_zero": bool(np.all(third == 0.0)),
        })
    output = Path(__file__).resolve().parents[1] / "results/siren_frequency_cap_k64/encoding_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "pass", "records": records}, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
