#!/usr/bin/env python3
"""Build one nested K-prefix state from one strict K2048 checkpoint.

Only the raw network potential is evaluated.  A compatible masked-Cartesian
chain derives velocity and vorticity, one objective chooses the geometry-
specific causal order and two-pass MGS transform, and that exact transform is
then applied to all three physical fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    inside_airfoil,
)
from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    load_hardcoded_model,
)
from eigenfluid_sss.basisnet.airfoil2d.mimetic_fd import (  # noqa: E402
    build_masked_cartesian_complex,
)
from eigenfluid_sss.basisnet.airfoil2d.strict_prefix import (  # noqa: E402
    FORMAL_FULL_MODES,
    FORMAL_SAMPLING_MODE,
    FORMAL_SAMPLING_RNG,
    STRICT_PREFIXES,
    apply_network_transform,
    build_causal_transform,
    canonical_objective,
    derivative_commutation_audit,
    derive_mimetic_fields,
    file_sha256,
    objective_grams,
    persist_prefix_state,
)


def generated_grid(cfg: Any, beta: float, alpha: float, spacing: float):
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    nx_float = (float(cfg.xmax) - float(cfg.xmin)) / spacing
    ny_float = (float(cfg.ymax) - float(cfg.ymin)) / spacing
    if not np.isclose(nx_float, round(nx_float), rtol=0.0, atol=1.0e-8):
        raise ValueError("spacing must divide the configured x extent")
    if not np.isclose(ny_float, round(ny_float), rtol=0.0, atol=1.0e-8):
        raise ValueError("spacing must divide the configured y extent")
    x = np.linspace(cfg.xmin, cfg.xmax, int(round(nx_float)) + 1)
    y = np.linspace(cfg.ymin, cfg.ymax, int(round(ny_float)) + 1)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    solid = inside_airfoil(cfg, points, beta, alpha).reshape(len(y), len(x))
    return x, y, solid


def load_grid(
    cfg: Any,
    beta: float,
    alpha: float,
    grid_npz: Path | None,
    spacing: float,
):
    if grid_npz is None:
        return generated_grid(cfg, beta, alpha, spacing)
    with np.load(grid_npz, allow_pickle=False) as archive:
        required = {"x", "y", "solid"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"grid NPZ is missing {sorted(missing)}")
        x = archive["x"].astype(np.float64)
        y = archive["y"].astype(np.float64)
        solid = archive["solid"].astype(bool)
    xx, yy = np.meshgrid(x, y)
    expected = inside_airfoil(
        cfg,
        np.column_stack((xx.ravel(), yy.ravel())),
        beta,
        alpha,
    ).reshape(len(y), len(x))
    if solid.shape != expected.shape or not np.array_equal(solid, expected):
        raise ValueError("grid solid mask does not match the requested geometry")
    return x, y, solid


def materialize_raw_potential(
    cfg: Any,
    model: Any,
    complex_: Any,
    beta: float,
    alpha: float,
    point_chunk: int,
) -> np.ndarray:
    """Evaluate raw network values at active nodes; do not request derivatives."""

    import cupy as cp

    if point_chunk <= 0:
        raise ValueError("point_chunk must be positive")
    yy, xx = np.meshgrid(complex_.y, complex_.x, indexing="ij")
    active_points = np.column_stack((xx[complex_.unknown], yy[complex_.unknown]))
    active = np.empty((len(active_points), int(cfg.modes)), dtype=np.float32)
    for start in range(0, len(active_points), point_chunk):
        stop = min(start + point_chunk, len(active_points))
        points = active_points[start:stop].astype(np.float32, copy=False)
        for mode, field, _cache in model.forward_stream_jet1(
            cfg, points, beta, alpha, False
        ):
            active[start:stop, int(mode.mode_index)] = cp.asnumpy(
                field.value[:, 0]
            ).astype(np.float32, copy=False)
    if not np.all(np.isfinite(active)):
        raise FloatingPointError("checkpoint produced non-finite potential values")
    grid = np.zeros((*complex_.unknown.shape, int(cfg.modes)), dtype=np.float32)
    grid[complex_.unknown] = active
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=("potential", "velocity", "vorticity"), required=True
    )
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--alpha-deg", type=float, required=True)
    parser.add_argument("--grid-npz", type=Path)
    parser.add_argument("--spacing", type=float, default=0.1)
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument("--mode-batch", type=int)
    parser.add_argument(
        "--no-persist-fields", action="store_true",
        help="Persist the transform/audits but omit the four full basis arrays.",
    )
    parser.add_argument(
        "--allow-small-smoke", action="store_true",
        help="Permit a non-K2048 checkpoint for an explicit development smoke.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    cfg, model, metadata = load_hardcoded_model(checkpoint)
    checkpoint_objective = canonical_objective(str(cfg.basis_objective))
    if checkpoint_objective != args.objective:
        raise ValueError(
            f"checkpoint objective {checkpoint_objective!r} != {args.objective!r}"
        )
    modes = int(cfg.modes)
    if modes != FORMAL_FULL_MODES and not args.allow_small_smoke:
        raise ValueError(
            f"strict evaluator requires K{FORMAL_FULL_MODES}, received K{modes}"
        )
    if not args.allow_small_smoke and int(metadata.get("epoch", -1)) != 2000:
        raise ValueError("strict evaluator requires the frozen epoch-2000 checkpoint")
    if int(getattr(cfg, "reorthogonalization_passes", -1)) != 2:
        raise ValueError("strict evaluator requires two-pass MGS")
    expected_jet = {"potential": 1, "velocity": 2, "vorticity": 3}[
        args.objective
    ]
    if int(getattr(cfg, "spatial_derivative_order", -1)) != expected_jet:
        raise ValueError(
            f"strict {args.objective} checkpoint must declare Jet{expected_jet}"
        )
    if str(getattr(cfg, "causal_sorting", "")) != "geometry_rayleigh":
        raise ValueError("strict evaluator requires geometry-Rayleigh causal sorting")
    sampling_mode = str(getattr(cfg, "sampling_mode", ""))
    if not args.allow_small_smoke and sampling_mode != FORMAL_SAMPLING_MODE:
        raise ValueError(
            "formal K2048 checkpoint must use uniform_random training samples"
        )
    bit_generator = np.random.default_rng(0).bit_generator.__class__.__name__
    expected_bit_generator = FORMAL_SAMPLING_RNG.rsplit(".", 1)[-1]
    if not args.allow_small_smoke and bit_generator != expected_bit_generator:
        raise RuntimeError(
            f"formal IID contract requires numpy PCG64, received {bit_generator}"
        )
    if args.mode_batch is not None:
        if args.mode_batch <= 0:
            raise ValueError("mode_batch must be positive")
        cfg.mode_forward_batch = int(args.mode_batch)

    timings: dict[str, float] = {}
    started = time.perf_counter()
    x, y, solid = load_grid(
        cfg, args.beta, args.alpha_deg, args.grid_npz, args.spacing
    )
    complex_ = build_masked_cartesian_complex(x, y, solid)
    timings["grid_and_complex_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_potential_grid = materialize_raw_potential(
        cfg,
        model,
        complex_,
        args.beta,
        args.alpha_deg,
        args.point_chunk,
    )
    timings["raw_network_value_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    raw_fields = derive_mimetic_fields(complex_, raw_potential_grid)
    mass, stiffness = objective_grams(complex_, raw_fields, args.objective)
    state = build_causal_transform(
        mass,
        stiffness,
        args.objective,
        passes=2,
        mass_floor=float(cfg.mass_floor),
    )
    timings["mimetic_and_transform_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    basis_fields = apply_network_transform(raw_fields, state.transform_network)
    commutation = derivative_commutation_audit(
        complex_, raw_fields, basis_fields
    )
    timings["shared_transform_and_audit_seconds"] = time.perf_counter() - started

    prefixes = STRICT_PREFIXES if modes == FORMAL_FULL_MODES else (modes,)
    provenance = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_sidecar_sha256": file_sha256(checkpoint.with_suffix(".json")),
        "checkpoint_epoch": int(metadata.get("epoch", -1)),
        "checkpoint_basis_objective": str(cfg.basis_objective),
        "formal_full_modes": FORMAL_FULL_MODES,
        "training_sampling_mode": sampling_mode,
        "training_sampling_rng": FORMAL_SAMPLING_RNG,
        "geometry": {"beta": args.beta, "alpha_deg": args.alpha_deg},
        "grid_source": (
            str(args.grid_npz.resolve()) if args.grid_npz is not None else "generated"
        ),
        "timings_before_persist": dict(timings),
    }
    persist_started = time.perf_counter()
    state_path, manifest_path, manifest = persist_prefix_state(
        args.output_dir,
        state,
        basis_fields,
        complex_,
        prefixes=prefixes,
        provenance=provenance,
        commutation=commutation,
        persist_fields=not args.no_persist_fields,
    )
    timings["persist_seconds"] = time.perf_counter() - persist_started
    print(
        json.dumps(
            {
                "state": str(state_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "state_sha256": manifest["state_npz"]["sha256"],
                "objective": state.objective,
                "modes": modes,
                "prefixes": list(prefixes),
                "orthogonality_max_abs": state.orthogonality_max_abs,
                "commutation": commutation,
                "timings": timings,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
