#!/usr/bin/env python3
"""Advance the airfoil GT initial frame in two compatible K=2048 ROM spaces.

The coefficient ODE is the vorticity--Galerkin reduced model introduced by
de Witt, Lessig, and Fiume (2012).  For time integration we use the later
variational implicit trapezoidal realization with exact modal viscous decay.

The two compared spaces are

* the learned Potential--Rayleigh streamfunction space, and
* the cached geometry-aware finite-difference Laplacian streamfunction space.

Both methods provide only nodal streamfunction columns.  A shared masked
Cartesian incidence complex then constructs rotated-gradient edge fluxes,
weak curl/Laplacian, nodal nonlinear samples, quadrature, and outlet traces.
The affine dynamics use the frozen-base closure F(a)-F(0), because the GT
archive contains velocity but no pressure field.  The existing FD eigensolve
is reused; this script never recomputes FD eigenvectors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    body_numpy,
    cfg_namespace,
)
from eigenfluid.gen.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    generalized_symmetric_eig,
    load_hardcoded_model,
)
from eigenfluid.gen.airfoil2d.rom_dynamics import OpenBoundaryTerms  # noqa: E402
from eigenfluid.gen.airfoil2d.mimetic_fd import (  # noqa: E402
    MaskedCartesianComplex,
    build_masked_cartesian_complex,
    diagnostics as mimetic_diagnostics,
    evaluate_streamfunction_block,
    weak_curl_from_grid_velocity,
)
from eigenfluid.gen.airfoil2d.rom_dynamics import (  # noqa: E402
    frozen_base_rhs_offset,
    variational_advection_rhs,
)
from rollout_eigenfluid_rom import (  # noqa: E402
    ImplicitTrapezoidalROM,
    metric_rows,
)


ORIENTATIONS = ("up", "middle", "down")
LEARNED_METHODS = ("potential", "velocity", "vorticity")
METHODS = (*LEARNED_METHODS, "fd_mimetic")
COLORS = {
    "potential": "#003F88",
    "velocity": "#6A51A3",
    "vorticity": "#8B0012",
    "fd_mimetic": "#2F2F2F",
}
LABELS = {
    "potential": "Potential--Rayleigh",
    "velocity": "Velocity--Rayleigh",
    "vorticity": "Vorticity--Rayleigh",
    "fd_mimetic": "Mimetic FD--Laplacian",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat()


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, np.generic):
        return strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_json(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def interpolate_fields(
    x: np.ndarray,
    y: np.ndarray,
    fields: tuple[np.ndarray, ...],
    points: np.ndarray,
) -> tuple[np.ndarray, ...]:
    query = np.column_stack((points[:, 1], points[:, 0]))
    values: list[np.ndarray] = []
    for field in fields:
        interpolator = RegularGridInterpolator(
            (y, x), field, method="linear", bounds_error=True
        )
        values.append(np.asarray(interpolator(query), dtype=np.float32))
    return tuple(values)


def raw_fields_compatible(
    model: Any,
    cfg: Any,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
    batch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate velocity/curl while accepting both archived Jet tensor layouts."""

    raw_u = np.empty((len(points), cfg.modes), dtype=np.float32)
    raw_v = np.empty_like(raw_u)
    raw_omega = np.empty_like(raw_u)
    mode_batch = max(1, int(getattr(cfg, "mode_forward_batch", 1)))
    point_budget = int(getattr(cfg, "postprocess_mode_point_budget", 262_144))
    effective_batch = min(batch, max(256, point_budget // mode_batch))
    for start in range(0, len(points), effective_batch):
        stop = min(start + effective_batch, len(points))
        for mode, (_network, field, _cache) in enumerate(
            model.forward_stream(cfg, points[start:stop], beta, alpha_deg, False)
        ):
            grad = field.grad
            hess = field.hess
            if grad.ndim == 3:
                gx, gy = grad[:, 0, 0], grad[:, 0, 1]
            elif grad.ndim == 2:
                gx, gy = grad[:, 0], grad[:, 1]
            else:
                raise ValueError(f"unsupported Jet gradient shape {grad.shape}")
            if hess.ndim == 4:
                laplacian = hess[:, 0, 0, 0] + hess[:, 0, 1, 1]
            elif hess.ndim == 3 and hess.shape[1:] == (1, 3):
                # Jet3 stores the symmetric Hessian as (xx, xy, yy).
                laplacian = hess[:, 0, 0] + hess[:, 0, 2]
            elif hess.ndim == 3 and hess.shape[1:] == (2, 2):
                laplacian = hess[:, 0, 0] + hess[:, 1, 1]
            elif hess.ndim == 2 and hess.shape[1] == 3:
                laplacian = hess[:, 0] + hess[:, 2]
            else:
                raise ValueError(f"unsupported Jet Hessian shape {hess.shape}")
            raw_u[start:stop, mode] = cp.asnumpy(gy)
            raw_v[start:stop, mode] = cp.asnumpy(-gx)
            raw_omega[start:stop, mode] = cp.asnumpy(-laplacian)
        print(f"[Potential fields] points {stop}/{len(points)}", flush=True)
    return raw_u, raw_v, raw_omega


def active_grid_points(complex_: MaskedCartesianComplex) -> np.ndarray:
    xgrid, ygrid = np.meshgrid(complex_.x, complex_.y)
    return np.column_stack((xgrid[complex_.unknown], ygrid[complex_.unknown])).astype(
        np.float32
    )


def potential_peeled_psi_on_active_grid(
    model: Any,
    cfg: Any,
    complex_: MaskedCartesianComplex,
    beta: float,
    alpha_deg: float,
    peeled_transform: np.ndarray | None,
    point_batch: int,
    workspace: Path,
    method: str = "potential",
) -> tuple[np.memmap, dict[str, Any]]:
    """Evaluate only psi and optionally rotate it before the shared FD complex."""

    started = time.perf_counter()
    points = active_grid_points(complex_)
    modes = int(cfg.modes)
    workspace.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(workspace, mode="w+", dtype=np.float32, shape=(len(points), modes))
    transform_gpu = (
        None
        if peeled_transform is None
        else cp.asarray(peeled_transform.astype(np.float32))
    )
    for start in range(0, len(points), point_batch):
        stop = min(start + point_batch, len(points))
        raw = np.empty((stop - start, modes), dtype=np.float32)
        if method == "potential":
            stream = model.forward_stream_jet1
        elif method == "velocity" and hasattr(model, "forward_stream_jet2"):
            stream = model.forward_stream_jet2
        else:
            stream = model.forward_stream
        for mode, (_network, field, _cache) in enumerate(
            stream(cfg, points[start:stop], beta, alpha_deg, False)
        ):
            raw[:, mode] = cp.asnumpy(field.value[:, 0]).astype(np.float32, copy=False)
        if transform_gpu is None:
            output[start:stop] = raw
        else:
            output[start:stop] = cp.asnumpy(cp.asarray(raw) @ transform_gpu)
        print(f"[{LABELS[method]} psi] active nodes {stop}/{len(points)}", flush=True)
        del raw
    output.flush()
    if transform_gpu is not None:
        del transform_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return output, {
        "seconds": time.perf_counter() - started,
        "active_nodes": int(len(points)),
        "modes": modes,
        "rayleigh_observable": method,
        "pre_ritz_transform": (
            "identity" if peeled_transform is None else "archived peeling transform"
        ),
        "observable": "streamfunction value only; all derivatives use the shared mimetic complex",
        "workspace": str(workspace.resolve()),
    }


def mimetic_fields_from_psi(
    complex_: MaskedCartesianComplex,
    modes: int,
    batch_modes: int,
    label: str,
    *,
    full_grid_psi: np.ndarray | None = None,
    active_psi: np.ndarray | None = None,
    cached_eigenvalues: np.ndarray | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, dict[str, Any]]:
    """Apply one incidence/collocation/curl chain to either learned or FD psi."""

    if (full_grid_psi is None) == (active_psi is None):
        raise ValueError("provide exactly one of full_grid_psi or active_psi")
    started = time.perf_counter()
    node_count = complex_.node_count
    result_u = cp.empty((node_count, modes), dtype=cp.float32)
    result_v = cp.empty_like(result_u)
    result_omega = cp.empty_like(result_u)
    audit_max = {
        "weak_curl_max_abs_fp64": 0.0,
        "weak_curl_relative_max_abs_fp64": 0.0,
        "rotated_gradient_divergence_max_abs_fp64": 0.0,
        "rotated_gradient_divergence_scaled_max_abs_fp64": 0.0,
    }
    eigen_residuals: list[float] = []
    for start in range(0, modes, batch_modes):
        stop = min(start + batch_modes, modes)
        if full_grid_psi is not None:
            block = np.asarray(full_grid_psi[..., start:stop], dtype=np.float32)
        else:
            block = np.zeros(complex_.unknown.shape + (stop - start,), dtype=np.float32)
            block[complex_.unknown] = np.asarray(active_psi[:, start:stop], dtype=np.float32)
        host_u, host_v, host_omega, audit = evaluate_streamfunction_block(complex_, block)
        result_u[:, start:stop] = cp.asarray(host_u)
        result_v[:, start:stop] = cp.asarray(host_v)
        result_omega[:, start:stop] = cp.asarray(host_omega)
        for key in tuple(audit_max):
            source_key = key.removesuffix("_fp64")
            audit_max[key] = max(audit_max[key], float(audit[source_key]))
        if cached_eigenvalues is not None:
            values = np.asarray(block[complex_.unknown], dtype=np.float64)
            applied = complex_.laplacian @ values
            residual = applied - values * np.asarray(
                cached_eigenvalues[start:stop], dtype=np.float64
            )[None, :]
            numerator = np.linalg.norm(residual, axis=0)
            denominator = np.maximum(
                np.abs(np.asarray(cached_eigenvalues[start:stop], dtype=np.float64))
                * np.linalg.norm(values, axis=0),
                1.0e-30,
            )
            eigen_residuals.extend((numerator / denominator).tolist())
        print(f"[{label} mimetic fields] modes {start + 1}-{stop}/{modes}", flush=True)
        del block, host_u, host_v, host_omega
        gc.collect()
    diagnostics_payload: dict[str, Any] = {
        "seconds": time.perf_counter() - started,
        "modes": modes,
        "chain": "psi -> oriented edge gradient -> rotated edge velocity -> nodal collocation; omega=(Gx.T Gx+Gy.T Gy)psi",
        **audit_max,
    }
    if eigen_residuals:
        diagnostics_payload["cached_eigen_residual_max_fp64_operator_fp32_psi"] = float(
            np.max(eigen_residuals)
        )
        diagnostics_payload["cached_eigen_residual_p95_fp64_operator_fp32_psi"] = float(
            np.percentile(eigen_residuals, 95.0)
        )
    return result_u, result_v, result_omega, diagnostics_payload


def mimetic_outlet_normal_from_psi(
    complex_: MaskedCartesianComplex,
    modes: int,
    batch_modes: int,
    *,
    full_grid_psi: np.ndarray | None = None,
    active_psi: np.ndarray | None = None,
) -> tuple[cp.ndarray, dict[str, Any]]:
    """Return the normal velocity on outlet edge midpoints from the same d0."""

    if (full_grid_psi is None) == (active_psi is None):
        raise ValueError("provide exactly one psi source")
    nx = len(complex_.x)
    outlet_edge_indices = np.asarray(
        [
            edge
            for edge, (first, second) in enumerate(complex_.y_edge_flat_nodes)
            if int(first) % nx == nx - 1 and int(second) % nx == nx - 1
        ],
        dtype=np.int64,
    )
    if len(outlet_edge_indices) != len(complex_.y) - 1:
        raise RuntimeError(
            f"expected {len(complex_.y) - 1} outlet vertical edges, got {len(outlet_edge_indices)}"
        )
    result = cp.empty((len(outlet_edge_indices), modes), dtype=cp.float32)
    started = time.perf_counter()
    for start in range(0, modes, batch_modes):
        stop = min(start + batch_modes, modes)
        if full_grid_psi is not None:
            values = np.asarray(
                full_grid_psi[..., start:stop][complex_.unknown], dtype=np.float64
            )
        else:
            values = np.asarray(active_psi[:, start:stop], dtype=np.float64)
        grad_y = complex_.gradient_y @ values
        result[:, start:stop] = cp.asarray(
            np.asarray(grad_y[outlet_edge_indices], dtype=np.float32)
        )
    weights = np.full(len(outlet_edge_indices), complex_.dy, dtype=np.float32)
    flux = cp.asarray(weights) @ result
    return result, {
        "seconds": time.perf_counter() - started,
        "edge_midpoints": int(len(outlet_edge_indices)),
        "tangential_fluctuation_trace": "exactly zero from natural d_x psi outlet trace",
        "maximum_absolute_integrated_normal_mode_flux": float(cp.max(cp.abs(flux)).get()),
    }


def _realized_mass_gram(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    weight: float | np.ndarray,
) -> cp.ndarray:
    """Return the symmetric velocity-mass Gram of realized FP32 fields."""

    if np.ndim(weight) == 0:
        gram = cp.float32(weight) * (
            basis_u.T @ basis_u + basis_v.T @ basis_v
        )
    else:
        weights_gpu = cp.asarray(np.asarray(weight, dtype=np.float32))[:, None]
        gram = basis_u.T @ (weights_gpu * basis_u)
        gram += basis_v.T @ (weights_gpu * basis_v)
        del weights_gpu
    return cp.float32(0.5) * (gram + gram.T)


def _realized_stiffness_gram(
    basis_omega: cp.ndarray,
    weight: float | np.ndarray,
) -> cp.ndarray:
    """Return the symmetric vorticity-stiffness Gram of realized fields."""

    if np.ndim(weight) == 0:
        gram = cp.float32(weight) * (basis_omega.T @ basis_omega)
    else:
        weights_gpu = cp.asarray(np.asarray(weight, dtype=np.float32))[:, None]
        gram = basis_omega.T @ (weights_gpu * basis_omega)
        del weights_gpu
    return cp.float32(0.5) * (gram + gram.T)


def _refine_realized_ritz_basis_fp64(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    weight: float | np.ndarray,
    eigenvalues: np.ndarray,
    rotation: np.ndarray,
    modes: int,
    label: str,
    *,
    trigger: float = 5.0e-3,
    target: float = 5.0e-4,
    max_iterations: int = 3,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Repair loss of mass orthogonality after the FP32 field rotation.

    The dense generalized solve is already performed in FP64, but applying an
    ill-conditioned coefficient rotation to a large FP32 field matrix can
    lose several digits.  When that *realized* basis fails the existing
    orthogonality contract, measure its mass and stiffness Gramians, solve a
    second FP64 whitening/Ritz problem, and apply the resulting near-identity
    correction to all physical fields.  This preserves the span and restores
    both mass orthonormality and diagonal viscous stiffness; no tolerance is
    relaxed.
    """

    identity = np.eye(modes, dtype=np.float64)
    mass_gpu = _realized_mass_gram(basis_u, basis_v, weight)
    mass_fp64 = cp.asnumpy(mass_gpu).astype(np.float64)
    del mass_gpu
    mass_fp64 = 0.5 * (mass_fp64 + mass_fp64.T)
    pre_error = float(np.max(np.abs(mass_fp64 - identity)))
    diagnostics: dict[str, Any] = {
        "mass_orthogonality_pre_refinement_max_abs_fp32": pre_error,
        "fp64_whitening_trigger": float(trigger),
        "fp64_whitening_target": float(target),
        "fp64_whitening_iterations": 0,
        "fp64_whitening_applied": False,
    }
    if pre_error <= trigger:
        diagnostics["mass_orthogonality_post_refinement_max_abs_fp32"] = pre_error
        return (
            basis_u,
            basis_v,
            basis_omega,
            eigenvalues,
            rotation,
            diagnostics,
        )

    last_fp64_orth = math.inf
    last_fp64_stiffness_offdiag = math.inf
    condition_numbers: list[float] = []
    realized_error = pre_error
    for iteration in range(1, max_iterations + 1):
        stiffness_gpu = _realized_stiffness_gram(basis_omega, weight)
        stiffness_fp64 = cp.asnumpy(stiffness_gpu).astype(np.float64)
        del stiffness_gpu
        stiffness_fp64 = 0.5 * (stiffness_fp64 + stiffness_fp64.T)

        mass_spectrum = np.linalg.eigvalsh(mass_fp64)
        if not np.all(np.isfinite(mass_spectrum)) or mass_spectrum[0] <= 0.0:
            raise FloatingPointError(
                f"{label} realized mass Gram is not FP64 positive definite"
            )
        condition_numbers.append(
            float(mass_spectrum[-1] / mass_spectrum[0])
        )
        refined_values, correction, rank = generalized_symmetric_eig(
            stiffness_fp64,
            mass_fp64,
            modes,
            sample_rows=2 * basis_u.shape[0],
        )
        if rank != modes or correction.shape != (modes, modes):
            raise RuntimeError(
                f"{label} FP64 realized-basis whitening rank {rank}, "
                f"expected {modes}"
            )
        if not np.all(np.isfinite(refined_values)) or float(
            refined_values[0]
        ) < -1.0e-6:
            raise FloatingPointError(
                f"{label} FP64 realized-basis Ritz solve is invalid"
            )
        refined_values = np.maximum(refined_values, 0.0)
        corrected_mass_fp64 = correction.T @ mass_fp64 @ correction
        corrected_stiffness_fp64 = correction.T @ stiffness_fp64 @ correction
        last_fp64_orth = float(
            np.max(np.abs(corrected_mass_fp64 - identity))
        )
        last_fp64_stiffness_offdiag = float(
            np.max(
                np.abs(
                    corrected_stiffness_fp64
                    - np.diag(np.diag(corrected_stiffness_fp64))
                )
            )
        )

        correction_gpu = cp.asarray(correction.astype(np.float32))
        corrected_u = basis_u @ correction_gpu
        del basis_u
        basis_u = corrected_u
        corrected_v = basis_v @ correction_gpu
        del basis_v
        basis_v = corrected_v
        corrected_omega = basis_omega @ correction_gpu
        del basis_omega, correction_gpu
        basis_omega = corrected_omega
        rotation = rotation @ correction
        eigenvalues = refined_values

        mass_gpu = _realized_mass_gram(basis_u, basis_v, weight)
        mass_fp64 = cp.asnumpy(mass_gpu).astype(np.float64)
        del mass_gpu
        mass_fp64 = 0.5 * (mass_fp64 + mass_fp64.T)
        realized_error = float(np.max(np.abs(mass_fp64 - identity)))
        diagnostics["fp64_whitening_iterations"] = iteration
        if realized_error <= target:
            break

    diagnostics.update(
        {
            "fp64_whitening_applied": True,
            "fp64_whitening_mass_condition_numbers": condition_numbers,
            "fp64_whitening_coefficient_mass_max_abs": last_fp64_orth,
            "fp64_whitening_coefficient_stiffness_offdiag_max_abs": (
                last_fp64_stiffness_offdiag
            ),
            "mass_orthogonality_post_refinement_max_abs_fp32": realized_error,
        }
    )
    return (
        basis_u,
        basis_v,
        basis_omega,
        eigenvalues,
        rotation,
        diagnostics,
    )


def gpu_ritz(
    raw_u: cp.ndarray,
    raw_v: cp.ndarray,
    raw_omega: cp.ndarray,
    weight: float | np.ndarray,
    modes: int,
    label: str,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Mass-orthonormalize and diagonalize viscous stiffness in one span."""

    started = time.perf_counter()
    cp.cuda.Device().synchronize()
    # Form one weighted Gram contribution at a time.  Keeping all three
    # Q-by-K weighted fields alive would add about 4.5 GiB at K=2048 and can
    # make the later basis rotation exceed a 16-GiB card.
    if np.ndim(weight) == 0:
        scalar_weight = cp.float32(weight)
        mass_gpu = scalar_weight * (raw_u.T @ raw_u)
        mass_gpu += scalar_weight * (raw_v.T @ raw_v)
        stiffness_gpu = scalar_weight * (raw_omega.T @ raw_omega)
    else:
        weights_gpu = cp.asarray(np.asarray(weight, dtype=np.float32))[:, None]
        mass_gpu = raw_u.T @ (weights_gpu * raw_u)
        mass_gpu += raw_v.T @ (weights_gpu * raw_v)
        stiffness_gpu = raw_omega.T @ (weights_gpu * raw_omega)
        del weights_gpu
    mass = cp.asnumpy(cp.float32(0.5) * (mass_gpu + mass_gpu.T)).astype(np.float64)
    stiffness = cp.asnumpy(
        cp.float32(0.5) * (stiffness_gpu + stiffness_gpu.T)
    ).astype(np.float64)
    del mass_gpu, stiffness_gpu
    cp.get_default_memory_pool().free_all_blocks()
    print(f"[{label}] solve dense K={modes} generalized Ritz problem", flush=True)
    eigenvalues, rotation, rank = generalized_symmetric_eig(
        stiffness, mass, modes, sample_rows=2 * raw_u.shape[0]
    )
    if rank != modes or rotation.shape != (modes, modes):
        raise RuntimeError(f"{label} quadrature rank {rank}, expected {modes}")
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) < -1.0e-6:
        raise FloatingPointError(f"{label} has invalid Ritz eigenvalues")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotation_gpu = cp.asarray(rotation.astype(np.float32))
    basis_u = raw_u @ rotation_gpu
    basis_v = raw_v @ rotation_gpu
    basis_omega = raw_omega @ rotation_gpu
    del raw_u, raw_v, raw_omega
    cp.get_default_memory_pool().free_all_blocks()
    (
        basis_u,
        basis_v,
        basis_omega,
        eigenvalues,
        rotation,
        refinement_diagnostics,
    ) = _refine_realized_ritz_basis_fp64(
        basis_u,
        basis_v,
        basis_omega,
        weight,
        eigenvalues,
        rotation,
        modes,
        label,
    )
    orth_error = float(
        refinement_diagnostics[
            "mass_orthogonality_post_refinement_max_abs_fp32"
        ]
    )
    cp.cuda.Device().synchronize()
    diagnostics = {
        "rank": int(rank),
        "mass_orthogonality_max_abs_fp32": orth_error,
        "ritz_eigenvalue_min": float(eigenvalues[0]),
        "ritz_eigenvalue_max": float(eigenvalues[-1]),
        "seconds": time.perf_counter() - started,
        **refinement_diagnostics,
    }
    if orth_error > 5.0e-3:
        raise RuntimeError(f"{label} mass orthogonality {orth_error:.3e} is too large")
    return basis_u, basis_v, basis_omega, eigenvalues, rotation, diagnostics


def fd_fields_on_points(
    cache_dir: Path,
    points: np.ndarray,
    modes: int,
    batch_modes: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, dict[str, Any]]:
    """Interpolate cached FD psi, its velocity, and exact modal curl."""

    started = time.perf_counter()
    x = np.load(cache_dir / "x.npy").astype(np.float32)
    y = np.load(cache_dir / "y.npy").astype(np.float32)
    eigenvalues = np.load(cache_dir / "eigenvalues.npy").astype(np.float32)[:modes]
    psi = np.load(cache_dir / "psi.npy", mmap_mode="r")
    if psi.shape != (len(y), len(x), 2048) or modes > psi.shape[-1]:
        raise ValueError(f"invalid FD cache shape {psi.shape}")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    query = np.column_stack((points[:, 1], points[:, 0]))
    result_u = cp.empty((len(points), modes), dtype=cp.float32)
    result_v = cp.empty_like(result_u)
    result_omega = cp.empty_like(result_u)
    options = {"method": "linear", "bounds_error": True}
    for start in range(0, modes, batch_modes):
        stop = min(start + batch_modes, modes)
        block = np.asarray(psi[..., start:stop], dtype=np.float32)
        gx = np.gradient(block, dx, axis=1, edge_order=2).astype(np.float32)
        gy = np.gradient(block, dy, axis=0, edge_order=2).astype(np.float32)
        # Natural outlet condition used by the archived eigensolve.
        gx[:, -1, :] = 0.0
        host_psi = np.asarray(
            RegularGridInterpolator((y, x), block, **options)(query),
            dtype=np.float32,
        )
        host_gx = np.asarray(
            RegularGridInterpolator((y, x), gx, **options)(query),
            dtype=np.float32,
        )
        host_gy = np.asarray(
            RegularGridInterpolator((y, x), gy, **options)(query),
            dtype=np.float32,
        )
        result_u[:, start:stop] = cp.asarray(host_gy)
        result_v[:, start:stop] = cp.asarray(-host_gx)
        result_omega[:, start:stop] = cp.asarray(
            host_psi * eigenvalues[None, start:stop]
        )
        print(f"[FD fields] modes {start + 1}-{stop}/{modes}", flush=True)
        del block, gx, gy, host_psi, host_gx, host_gy
        gc.collect()
    return result_u, result_v, result_omega, {
        "seconds": time.perf_counter() - started,
        "cache": str(cache_dir.resolve()),
        "cached_eigenvalue_min": float(eigenvalues[0]),
        "cached_eigenvalue_max": float(eigenvalues[-1]),
        "vorticity": "omega_i = lambda_i psi_i for cached -Delta psi_i=lambda_i psi_i",
    }


def potential_fields_on_points(
    cfg: Any,
    model: Any,
    checkpoint_metadata: dict[str, Any],
    checkpoint: Path,
    peeled_state: Path,
    points: np.ndarray,
    beta: float,
    alpha_deg: float,
    field_batch: int,
) -> tuple[Any, Any, cp.ndarray, cp.ndarray, cp.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate raw K2048 fields and rotate into the archived peeled span."""

    started = time.perf_counter()
    modes = int(cfg.modes)
    with np.load(peeled_state, allow_pickle=False) as archive:
        peeled_transform = archive["transform"].astype(np.float32)
    if peeled_transform.shape != (modes, modes):
        raise ValueError(f"invalid peeled transform {peeled_transform.shape}")
    raw_u, raw_v, raw_omega = raw_fields_compatible(
        model, cfg, points, beta, alpha_deg, field_batch
    )
    transform_gpu = cp.asarray(peeled_transform)
    peeled_u = cp.asarray(raw_u) @ transform_gpu
    peeled_v = cp.asarray(raw_v) @ transform_gpu
    peeled_omega = cp.asarray(raw_omega) @ transform_gpu
    del raw_u, raw_v, raw_omega, transform_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return cfg, model, peeled_u, peeled_v, peeled_omega, peeled_transform, {
        "seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "peeled_state": str(peeled_state.resolve()),
        "checkpoint_epoch": int(checkpoint_metadata["epoch"]),
        "spatial_jet_order_used_for_basis_fields": 2,
        "note": "training used Jet1; rollout curl is the required Hessian-derived observable",
    }


def project_initial(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    weight: float | np.ndarray,
) -> tuple[cp.ndarray, float]:
    tu = cp.asarray(target_u)
    tv = cp.asarray(target_v)
    if np.ndim(weight) == 0:
        weighted_tu = cp.float32(weight) * tu
        weighted_tv = cp.float32(weight) * tv
        numerator_weight: Any = cp.float32(weight)
    else:
        numerator_weight = cp.asarray(np.asarray(weight, dtype=np.float32))
        weighted_tu = numerator_weight * tu
        weighted_tv = numerator_weight * tv
    coefficients = basis_u.T @ weighted_tu + basis_v.T @ weighted_tv
    residual_u = basis_u @ coefficients - tu
    residual_v = basis_v @ coefficients - tv
    numerator = cp.sum(numerator_weight * (residual_u * residual_u + residual_v * residual_v))
    denominator = cp.sum(numerator_weight * (tu * tu + tv * tv))
    error = float(cp.sqrt(numerator / cp.maximum(denominator, cp.float32(1.0e-30))).get())
    del tu, tv, residual_u, residual_v
    return coefficients, error


def rollout_coefficients(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    initial: cp.ndarray,
    mean_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    eigenvalues: np.ndarray,
    volume_weight: float | np.ndarray,
    viscosity: float,
    output_dt: float,
    frames: int,
    substeps: int,
    outlet: OpenBoundaryTerms,
    label: str,
    base_flow_closure: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    gpu_mean_u = cp.asarray(mean_fields[0])
    gpu_mean_v = cp.asarray(mean_fields[1])
    gpu_mean_omega = cp.asarray(mean_fields[2])
    gpu_weight: Any = (
        cp.float32(volume_weight)
        if np.ndim(volume_weight) == 0
        else cp.asarray(np.asarray(volume_weight, dtype=np.float32))
    )
    base_started = time.perf_counter()
    zero = cp.zeros(len(initial), dtype=cp.float32)
    raw_base = variational_advection_rhs(
        cp,
        basis_u,
        basis_v,
        basis_omega,
        zero,
        gpu_weight,
        gpu_mean_u,
        gpu_mean_v,
        gpu_mean_omega,
        outlet,
    )
    volume_base = variational_advection_rhs(
        cp,
        basis_u,
        basis_v,
        basis_omega,
        zero,
        gpu_weight,
        gpu_mean_u,
        gpu_mean_v,
        gpu_mean_omega,
        None,
    )
    outlet_base = raw_base - volume_base
    if base_flow_closure == "frozen-base-subtracted":
        constant_rhs = frozen_base_rhs_offset(
            cp,
            basis_u,
            basis_v,
            basis_omega,
            gpu_weight,
            gpu_mean_u,
            gpu_mean_v,
            gpu_mean_omega,
            outlet,
        )
    elif base_flow_closure == "none":
        constant_rhs = cp.zeros_like(raw_base)
    else:
        raise ValueError(f"unknown base-flow closure {base_flow_closure}")
    corrected_zero = raw_base + constant_rhs
    base_diagnostics = {
        "closure": base_flow_closure,
        "resolved_base_rhs_norm": float(cp.linalg.norm(raw_base).get()),
        "resolved_volume_base_rhs_norm": float(cp.linalg.norm(volume_base).get()),
        "resolved_outlet_base_rhs_norm": float(cp.linalg.norm(outlet_base).get()),
        "constant_closure_rhs_norm": float(cp.linalg.norm(constant_rhs).get()),
        "corrected_rhs_at_zero_norm": float(cp.linalg.norm(corrected_zero).get()),
        "seconds": time.perf_counter() - base_started,
        "interpretation": (
            "calibrated F(a)-F(0); GT has no pressure archive, so this is not a "
            "recovered pressure boundary condition"
        ),
    }
    integrator = ImplicitTrapezoidalROM(
        basis_u,
        basis_v,
        basis_omega,
        gpu_mean_u,
        gpu_mean_v,
        gpu_mean_omega,
        cp.asarray(eigenvalues.astype(np.float32)),
        gpu_weight,
        viscosity,
        newton_tolerance=2.0e-6,
        gmres_tolerance=2.0e-5,
        open_boundary=outlet,
        constant_rhs=constant_rhs,
    )
    values = np.empty((frames + 1, len(initial)), dtype=np.float32)
    values[0] = cp.asnumpy(initial)
    state = initial
    internal_dt = output_dt / substeps
    started = time.perf_counter()
    per_frame_seconds: list[float] = []
    energy_budget: list[dict[str, Any]] = []

    def append_budget(frame: int, value: cp.ndarray) -> None:
        budget_started = time.perf_counter()
        raw = integrator.raw_rhs(value)
        volume_total = variational_advection_rhs(
            cp,
            basis_u,
            basis_v,
            basis_omega,
            value,
            gpu_weight,
            gpu_mean_u,
            gpu_mean_v,
            gpu_mean_omega,
            None,
        )
        self_rhs = variational_advection_rhs(
            cp,
            basis_u,
            basis_v,
            basis_omega,
            value,
            gpu_weight,
        )
        mean_volume_increment = volume_total - volume_base - self_rhs
        outlet_increment = (raw - volume_total) - outlet_base
        mean_and_outlet_increment = mean_volume_increment + outlet_increment
        viscous = -cp.float32(viscosity) * integrator.eigenvalues * value
        total = raw + constant_rhs + viscous
        decomposed = (
            self_rhs
            + mean_volume_increment
            + outlet_increment
            + raw_base
            + constant_rhs
            + viscous
        )
        energy_budget.append(
            {
                "frame": frame,
                "time": frame * output_dt,
                "coefficient_norm": float(cp.linalg.norm(value).get()),
                "kinetic_energy": float((cp.float32(0.5) * (value @ value)).get()),
                "power_self_advection": float((value @ self_rhs).get()),
                "power_mean_and_outlet_increment": float(
                    (value @ mean_and_outlet_increment).get()
                ),
                "power_mean_volume_increment": float(
                    (value @ mean_volume_increment).get()
                ),
                "power_outlet_increment": float((value @ outlet_increment).get()),
                "power_resolved_base": float((value @ raw_base).get()),
                "power_constant_closure": float((value @ constant_rhs).get()),
                "power_viscous": float((value @ viscous).get()),
                "power_total": float((value @ total).get()),
                "power_sum_defect": float((value @ (total - decomposed)).get()),
                "budget_seconds": time.perf_counter() - budget_started,
            }
        )

    append_budget(0, state)
    for frame in range(1, frames + 1):
        frame_started = time.perf_counter()
        for _ in range(substeps):
            state = integrator.step(state, internal_dt)
        values[frame] = cp.asnumpy(state)
        append_budget(frame, state)
        per_frame_seconds.append(time.perf_counter() - frame_started)
        print(
            f"[{label} ROM {frame:02d}/{frames}] norm={float(cp.linalg.norm(state).get()):.6e} "
            f"frame_wall={time.perf_counter() - frame_started:.1f}s",
            flush=True,
        )
    return values, {
        "integrator": "implicit trapezoidal Newton-GMRES with Strang-split exact viscous decay",
        "frames": frames,
        "substeps_per_frame": substeps,
        "output_dt": output_dt,
        "internal_dt": internal_dt,
        "newton_iterations": integrator.newton_iterations,
        "gmres_iterations": integrator.linear_iterations,
        "adaptive_step_splits": integrator.step_splits,
        "seconds": time.perf_counter() - started,
        "per_frame_seconds": per_frame_seconds,
        "initial_coefficient_norm": float(np.linalg.norm(values[0])),
        "final_coefficient_norm": float(np.linalg.norm(values[-1])),
        "base_flow": base_diagnostics,
        "energy_budget_by_frame": energy_budget,
        "energy_budget_seconds": float(
            np.sum([float(row["budget_seconds"]) for row in energy_budget])
        ),
    }


def fd_reconstruct_stride2(
    cache_dir: Path,
    raw_coefficients: np.ndarray,
    mean_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    x = np.load(cache_dir / "x.npy").astype(np.float32)
    y = np.load(cache_dir / "y.npy").astype(np.float32)
    psi = np.load(cache_dir / "psi.npy", mmap_mode="r")
    eigenvalues = np.load(cache_dir / "eigenvalues.npy").astype(np.float32)
    frames = raw_coefficients.shape[1]
    shape = (frames, len(y), len(x))
    result_u = np.zeros(shape, dtype=np.float32)
    result_v = np.zeros(shape, dtype=np.float32)
    result_omega = np.zeros(shape, dtype=np.float32)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    coefficients_gpu = cp.asarray(raw_coefficients.astype(np.float32))
    modes = raw_coefficients.shape[0]
    for start in range(0, modes, batch_modes):
        stop = min(start + batch_modes, modes)
        block = np.asarray(psi[..., start:stop], dtype=np.float32)
        gx = np.gradient(block, dx, axis=1, edge_order=2).astype(np.float32)
        gy = np.gradient(block, dy, axis=0, edge_order=2).astype(np.float32)
        gx[:, -1, :] = 0.0
        coefficients = coefficients_gpu[start:stop]
        result_u += cp.asnumpy(cp.asarray(gy).reshape(-1, stop - start) @ coefficients).T.reshape(shape)
        result_v -= cp.asnumpy(cp.asarray(gx).reshape(-1, stop - start) @ coefficients).T.reshape(shape)
        omega = block * eigenvalues[None, None, start:stop]
        result_omega += cp.asnumpy(cp.asarray(omega).reshape(-1, stop - start) @ coefficients).T.reshape(shape)
        print(f"[FD reconstruct] modes {start + 1}-{stop}/{modes}", flush=True)
        del block, gx, gy, omega, coefficients
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    result_u += mean_fields[0][None]
    result_v += mean_fields[1][None]
    result_omega += mean_fields[2][None]
    return result_u, result_v, result_omega, {"seconds": time.perf_counter() - started}


def potential_reconstruct_stride2(
    cfg: Any,
    model: Any,
    total_transform: np.ndarray,
    coefficients: np.ndarray,
    gt: dict[str, np.ndarray],
    beta: float,
    alpha_deg: float,
    field_batch: int,
    point_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    x = gt["x"][::2].astype(np.float32)
    y = gt["y"][::2].astype(np.float32)
    solid = gt["solid"][::2, ::2].astype(bool)
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    raw_coefficients = total_transform @ coefficients.T.astype(np.float64)
    raw_coefficients_gpu = cp.asarray(raw_coefficients.astype(np.float32))
    frames = coefficients.shape[0]
    compact_u = np.empty((len(points), frames), dtype=np.float32)
    compact_v = np.empty_like(compact_u)
    compact_omega = np.empty_like(compact_u)
    for start in range(0, len(points), point_chunk):
        stop = min(start + point_chunk, len(points))
        raw_u, raw_v, raw_omega = raw_fields_compatible(
            model, cfg, points[start:stop], beta, alpha_deg, field_batch
        )
        compact_u[start:stop] = cp.asnumpy(cp.asarray(raw_u) @ raw_coefficients_gpu)
        compact_v[start:stop] = cp.asnumpy(cp.asarray(raw_v) @ raw_coefficients_gpu)
        compact_omega[start:stop] = cp.asnumpy(cp.asarray(raw_omega) @ raw_coefficients_gpu)
        print(f"[Potential reconstruct] points {stop}/{len(points)}", flush=True)
        del raw_u, raw_v, raw_omega
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
    shape = (frames,) + solid.shape
    result_u = np.zeros(shape, dtype=np.float32)
    result_v = np.zeros(shape, dtype=np.float32)
    result_omega = np.zeros(shape, dtype=np.float32)
    result_u[:, fluid] = compact_u.T
    result_v[:, fluid] = compact_v.T
    result_omega[:, fluid] = compact_omega.T
    mean_u = gt["u"].mean(axis=0, dtype=np.float64)[::2, ::2].astype(np.float32)
    mean_v = gt["v"].mean(axis=0, dtype=np.float64)[::2, ::2].astype(np.float32)
    mean_omega = gt["vorticity"].mean(axis=0, dtype=np.float64)[::2, ::2].astype(np.float32)
    result_u[:, fluid] += mean_u[fluid]
    result_v[:, fluid] += mean_v[fluid]
    result_omega[:, fluid] += mean_omega[fluid]
    return result_u, result_v, result_omega, {"seconds": time.perf_counter() - started}


def reconstruct_from_active_basis(
    basis_u: cp.ndarray,
    basis_v: cp.ndarray,
    basis_omega: cp.ndarray,
    coefficients: np.ndarray,
    mean_grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    complex_: MaskedCartesianComplex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct both methods from the exact basis used by native-grid dynamics."""

    started = time.perf_counter()
    coefficient_gpu = cp.asarray(coefficients.T.astype(np.float32))
    compact_u = cp.asnumpy(basis_u @ coefficient_gpu).T
    compact_v = cp.asnumpy(basis_v @ coefficient_gpu).T
    compact_omega = cp.asnumpy(basis_omega @ coefficient_gpu).T
    shape = (len(coefficients),) + complex_.unknown.shape
    result_u = np.broadcast_to(mean_grids[0], shape).copy().astype(np.float32)
    result_v = np.broadcast_to(mean_grids[1], shape).copy().astype(np.float32)
    result_omega = np.broadcast_to(mean_grids[2], shape).copy().astype(np.float32)
    result_u[:, complex_.unknown] += compact_u
    result_v[:, complex_.unknown] += compact_v
    result_omega[:, complex_.unknown] += compact_omega
    result_u[:, complex_.solid] = 0.0
    result_v[:, complex_.solid] = 0.0
    result_omega[:, complex_.solid] = 0.0
    return result_u, result_v, result_omega, {
        "seconds": time.perf_counter() - started,
        "representation": "the same native active-node Ritz basis used by the ROM",
    }


def make_outlet(
    basis_u: cp.ndarray,
    mean_u: np.ndarray,
    mean_v: np.ndarray,
    weights: np.ndarray,
) -> OpenBoundaryTerms:
    basis_v = cp.zeros_like(basis_u)
    return OpenBoundaryTerms(
        basis_u=basis_u,
        basis_v=basis_v,
        basis_normal=basis_u,
        weights=cp.asarray(weights),
        static_pressure=cp.float32(0.0),
        mean_u=cp.asarray(mean_u),
        mean_v=cp.asarray(mean_v),
    )


def wake_mask(cfg: Any, x: np.ndarray, y: np.ndarray, solid: np.ndarray, alpha: float) -> np.ndarray:
    xgrid, ygrid = np.meshgrid(x, y)
    xb, yb = body_numpy(cfg, np.column_stack((xgrid.ravel(), ygrid.ravel())), alpha)
    return (
        (~solid).ravel()
        & (xb >= cfg.chord)
        & (xb <= cfg.chord * (1.0 + float(getattr(cfg, "wake_length_chords", 3.0))))
        & (np.abs(yb) <= cfg.chord * float(getattr(cfg, "wake_half_height_chords", 0.75)))
    ).reshape(solid.shape)


def save_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_errors(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [str(key) for key in rows[0] if str(key).endswith("_error")]
    return {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "p95": float(np.percentile([float(row[key]) for row in rows], 95.0)),
            "max": float(np.max([float(row[key]) for row in rows])),
            "final": float(rows[-1][key]),
        }
        for key in keys
    }


def render_error_curves(
    output_dir: Path,
    metrics_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    entries = (
        ("velocity_error", "Velocity"),
        ("vorticity_error", "Vorticity"),
        ("fluctuation_velocity_error", "Fluctuation velocity"),
        ("fluctuation_vorticity_error", "Fluctuation vorticity"),
        ("wake_velocity_error", "Wake velocity"),
        ("wake_vorticity_error", "Wake vorticity"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), sharex=True)
    for axis, (key, title) in zip(axes.ravel(), entries):
        for method in metrics_by_method:
            rows = metrics_by_method[method]
            axis.plot(
                [int(row["frame"]) for row in rows],
                [100.0 * float(row[key]) for row in rows],
                color=COLORS[method],
                linewidth=1.35,
                marker="o",
                markersize=2.2,
                markevery=3,
                label=LABELS[method],
            )
        axis.set_title(title, fontweight="normal")
        axis.set_ylabel(r"Relative $L^2$ error (\%)")
        axis.set_xlim(1, 30)
        axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
        axis.grid(color="#D9D9D9", linewidth=0.45, alpha=0.65)
    for axis in axes[-1]:
        axis.set_xlabel("Predicted frame")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"temporal_error_curves.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)


def run_orientation(args: argparse.Namespace, orientation: str) -> dict[str, Any]:
    orientation_started = time.perf_counter()
    orientation_timings: dict[str, float] = {}
    output_dir = args.output_root / orientation
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_load_started = time.perf_counter()
    summary = json.loads(
        (args.evaluation_root / orientation / "summary.json").read_text(encoding="utf-8")
    )
    gt_path = Path(summary["gt"])
    gt = load_npz(gt_path)
    metadata = json.loads(str(gt["metadata"]))
    beta = float(summary["beta"])
    alpha = float(summary["alpha_deg"])
    cache_dir = args.laplacian_cache_root / orientation
    cache_x = np.load(cache_dir / "x.npy").astype(np.float64)
    cache_y = np.load(cache_dir / "y.npy").astype(np.float64)
    cache_solid = np.load(cache_dir / "solid.npy").astype(bool)
    cache_psi = np.load(cache_dir / "psi.npy", mmap_mode="r")
    cache_eigenvalues = np.load(cache_dir / "eigenvalues.npy").astype(np.float64)
    cache_diagnostics = json.loads((cache_dir / "diagnostics.json").read_text(encoding="utf-8"))
    orientation_timings["gt_and_operator_cache_load_seconds"] = time.perf_counter() - gt_load_started

    expected_x = np.asarray(gt["x"][:: args.grid_stride], dtype=np.float64)
    expected_y = np.asarray(gt["y"][:: args.grid_stride], dtype=np.float64)
    expected_solid = np.asarray(
        gt["solid"][:: args.grid_stride, :: args.grid_stride], dtype=bool
    )
    if not np.array_equal(cache_solid, expected_solid):
        raise ValueError("FD cache and GT stride mask differ")
    if not np.allclose(cache_x, expected_x) or not np.allclose(cache_y, expected_y):
        raise ValueError("FD cache and requested GT stride grid differ")
    if cache_psi.shape != cache_solid.shape + (2048,) or args.modes > cache_psi.shape[-1]:
        raise ValueError(f"invalid FD psi cache shape {cache_psi.shape}")

    complex_started = time.perf_counter()
    complex_ = build_masked_cartesian_complex(cache_x, cache_y, cache_solid)
    complex_diag = mimetic_diagnostics(complex_)
    orientation_timings["mimetic_complex_assembly_seconds"] = time.perf_counter() - complex_started
    write_json(
        output_dir / "operator_identity_audit.json",
        {
            "mimetic_complex": complex_diag,
            "archived_fd_cache": cache_diagnostics,
            "shared_by_methods": True,
            "potential_derivatives": "disabled; value-only psi passes through the same complex",
            "fd_interpolation": "disabled; cached psi stays on its native active-node grid",
        },
    )

    checkpoint = Path(args.checkpoint)
    checkpoint_started = time.perf_counter()
    if args.active_psi_cache is not None:
        checkpoint_metadata = json.loads(
            checkpoint.with_suffix(".json").read_text(encoding="utf-8")
        )
        cfg = cfg_namespace(checkpoint_metadata["config"])
        model = None
    else:
        cfg, model, checkpoint_metadata = load_hardcoded_model(checkpoint)
    orientation_timings["checkpoint_load_seconds"] = time.perf_counter() - checkpoint_started
    if any(method in LEARNED_METHODS for method in args.methods) and int(cfg.modes) != args.modes:
        raise ValueError(
            f"learned rollout requires an exact-K checkpoint; "
            f"checkpoint has K={cfg.modes}, requested K={args.modes}"
        )

    common_started = time.perf_counter()
    stride = args.grid_stride
    mean_u_grid = gt["u"].mean(axis=0, dtype=np.float64)[::stride, ::stride].astype(
        np.float32
    )
    mean_v_grid = gt["v"].mean(axis=0, dtype=np.float64)[::stride, ::stride].astype(
        np.float32
    )
    archived_mean_omega_grid = (
        gt["vorticity"].mean(axis=0, dtype=np.float64)[::stride, ::stride].astype(np.float32)
    )
    compatible_mean_omega = weak_curl_from_grid_velocity(
        complex_, mean_u_grid, mean_v_grid
    )
    mean_omega_grid = archived_mean_omega_grid.copy()
    mean_omega_grid[complex_.unknown] = compatible_mean_omega
    mean_fields = (
        mean_u_grid[complex_.unknown],
        mean_v_grid[complex_.unknown],
        compatible_mean_omega,
    )
    initial_u_grid = gt["u"][0, ::stride, ::stride].astype(np.float32)
    initial_v_grid = gt["v"][0, ::stride, ::stride].astype(np.float32)
    target_u = initial_u_grid[complex_.unknown] - mean_fields[0]
    target_v = initial_v_grid[complex_.unknown] - mean_fields[1]
    dx_fine = float(gt["x"][1] - gt["x"][0])
    output_dt = args.steps_per_frame * dx_fine
    viscosity = float(metadata["lattice_viscosity"]) * dx_fine
    # The LBM outlet is zero-gradient and the archive has no pressure.  Use
    # edge-midpoint traces; the base-only total-pressure term is removed by F-F0.
    outlet_mean_u = np.float32(0.5) * (
        mean_u_grid[:-1, -1] + mean_u_grid[1:, -1]
    )
    outlet_mean_v = np.float32(0.5) * (
        mean_v_grid[:-1, -1] + mean_v_grid[1:, -1]
    )
    outlet_weights = np.full(len(cache_y) - 1, complex_.dy, dtype=np.float32)
    orientation_timings["common_native_grid_mean_and_initial_setup_seconds"] = (
        time.perf_counter() - common_started
    )

    methods: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        method_started = time.perf_counter()
        method_timings: dict[str, Any] = {}
        print(f"[{orientation}] build {method} K={args.modes} native mimetic ROM", flush=True)
        potential_workspace: Path | None = None
        active_psi: np.ndarray | None = None
        full_grid_psi: np.ndarray | None = None
        psi_diag: dict[str, Any]
        if method in LEARNED_METHODS:
            peeled_transform: np.ndarray | None = None
            if args.peeled_state is not None:
                with np.load(args.peeled_state, allow_pickle=False) as archive:
                    peeled_transform = archive["transform"].astype(np.float32)
                if peeled_transform.shape != (args.modes, args.modes):
                    raise ValueError(f"invalid peeled transform {peeled_transform.shape}")
            if args.active_psi_cache is not None:
                cache_manifest = json.loads(
                    args.active_psi_cache.with_suffix(".json").read_text(
                        encoding="utf-8"
                    )
                )
                expected_cache = {
                    "complete": True,
                    "modes": int(args.modes),
                    "active_nodes": int(complex_.node_count),
                    "grid_shape": [int(len(complex_.y)), int(len(complex_.x))],
                    "beta": float(beta),
                    "alpha_deg": float(alpha),
                }
                for key, expected in expected_cache.items():
                    actual = cache_manifest.get(key)
                    if isinstance(expected, float):
                        valid = actual is not None and math.isclose(
                            float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12
                        )
                    else:
                        valid = actual == expected
                    if not valid:
                        raise ValueError(
                            f"active psi cache manifest {key}={actual!r}; "
                            f"expected {expected!r}"
                        )
                if Path(cache_manifest["checkpoint"]).resolve() != checkpoint.resolve():
                    raise ValueError("active psi cache checkpoint path differs")
                if int(cache_manifest["checkpoint_epoch"]) != int(
                    checkpoint_metadata["epoch"]
                ):
                    raise ValueError("active psi cache checkpoint epoch differs")
                if str(cache_manifest["basis_objective"]) != str(cfg.basis_objective):
                    raise ValueError("active psi cache objective differs")
                if cache_manifest["checkpoint_sha256"] != sha256(checkpoint):
                    raise ValueError("active psi cache checkpoint hash differs")
                if cache_manifest["output_sha256"] != sha256(args.active_psi_cache):
                    raise ValueError("active psi cache payload hash differs")
                active_psi = np.load(args.active_psi_cache, mmap_mode="r")
                if active_psi.shape != (complex_.node_count, args.modes):
                    raise ValueError(
                        f"invalid active psi cache shape {active_psi.shape}; expected "
                        f"{(complex_.node_count, args.modes)}"
                    )
                if peeled_transform is not None:
                    raise ValueError(
                        "--active-psi-cache is already a finalized raw span; "
                        "do not combine it with --peeled-state"
                    )
                psi_diag = {
                    "seconds": 0.0,
                    "active_nodes": int(complex_.node_count),
                    "modes": int(args.modes),
                    "rayleigh_observable": method,
                    "pre_ritz_transform": "identity",
                    "observable": (
                        "precomputed value-only streamfunction; all derivatives "
                        "use the shared mimetic complex"
                    ),
                    "source": str(args.active_psi_cache.resolve()),
                }
            else:
                if model is None:
                    raise RuntimeError("learned checkpoint model was not loaded")
                potential_workspace = output_dir / f"{method}_psi_active.tmp.f32"
                active_psi, psi_diag = potential_peeled_psi_on_active_grid(
                    model,
                    cfg,
                    complex_,
                    beta,
                    alpha,
                    peeled_transform,
                    args.psi_point_batch,
                    potential_workspace,
                    method,
                )
            field_label = LABELS[method].replace("--", "-")
            cached_values = None
        elif method == "fd_mimetic":
            full_grid_psi = cache_psi
            psi_diag = {
                "seconds": 0.0,
                "source": str((cache_dir / "psi.npy").resolve()),
                "observable": "cached streamfunction only",
            }
            field_label = "Mimetic FD-Laplacian"
            cached_values = cache_eigenvalues[: args.modes]
        else:
            raise ValueError(f"unknown method {method}")

        raw_u, raw_v, raw_omega, chain_diag = mimetic_fields_from_psi(
            complex_,
            args.modes,
            args.fd_batch_modes,
            field_label,
            full_grid_psi=full_grid_psi,
            active_psi=active_psi,
            cached_eigenvalues=cached_values,
        )
        raw_outlet_u, outlet_diag = mimetic_outlet_normal_from_psi(
            complex_,
            args.modes,
            args.fd_batch_modes,
            full_grid_psi=full_grid_psi,
            active_psi=active_psi,
        )
        field_diag = {
            "psi": psi_diag,
            "mimetic_chain": chain_diag,
            "outlet_trace": outlet_diag,
            "checkpoint": (
                str(checkpoint.resolve()) if method in LEARNED_METHODS else None
            ),
            "checkpoint_epoch": (
                int(checkpoint_metadata["epoch"])
                if method in LEARNED_METHODS
                else None
            ),
            "rayleigh_definition": (
                "||grad(F_k)||_h^2 / ||F_k||_h^2"
                if method in LEARNED_METHODS
                else None
            ),
            "rayleigh_observable": (
                {"potential": "A_k", "velocity": "u_k", "vorticity": "omega_k"}[method]
                if method in LEARNED_METHODS
                else None
            ),
            "seconds": float(psi_diag["seconds"]) + float(chain_diag["seconds"]),
        }
        basis_u, basis_v, basis_omega, eigenvalues, ritz_rotation, ritz_diag = gpu_ritz(
            raw_u,
            raw_v,
            raw_omega,
            complex_.weights,
            args.modes,
            field_label,
        )
        rotation_gpu = cp.asarray(ritz_rotation.astype(np.float32))
        outlet_basis_u = raw_outlet_u @ rotation_gpu
        outlet_flux = cp.asarray(outlet_weights) @ outlet_basis_u
        field_diag["outlet_trace"]["ritz_maximum_absolute_integrated_normal_mode_flux"] = float(
            cp.max(cp.abs(outlet_flux)).get()
        )
        del raw_u, raw_v, raw_omega, raw_outlet_u, rotation_gpu, outlet_flux
        cp.get_default_memory_pool().free_all_blocks()
        if isinstance(active_psi, np.memmap):
            active_psi.flush()
            active_psi._mmap.close()
            active_psi = None
        if potential_workspace is not None and potential_workspace.exists():
            potential_workspace.unlink()

        projection_started = time.perf_counter()
        initial, initial_error = project_initial(
            basis_u, basis_v, target_u, target_v, complex_.weights
        )
        method_timings["initial_projection_seconds"] = time.perf_counter() - projection_started
        outlet = make_outlet(
            outlet_basis_u,
            outlet_mean_u,
            outlet_mean_v,
            outlet_weights,
        )
        coefficient_frames, rollout_diag = rollout_coefficients(
            basis_u,
            basis_v,
            basis_omega,
            initial,
            mean_fields,
            eigenvalues,
            complex_.weights,
            viscosity,
            output_dt,
            args.frames,
            args.substeps_per_frame,
            outlet,
            LABELS[method],
            args.base_flow_closure,
        )
        energy_rows = rollout_diag.pop("energy_budget_by_frame")
        energy_path = output_dir / f"{method}_energy_budget_by_frame.csv"
        save_metrics(energy_path, energy_rows)
        reconstruction_diag: dict[str, Any]
        predicted_u, predicted_v, predicted_omega, reconstruction_diag = (
            reconstruct_from_active_basis(
                basis_u,
                basis_v,
                basis_omega,
                coefficient_frames,
                (mean_u_grid, mean_v_grid, mean_omega_grid),
                complex_,
            )
        )
        del basis_u, basis_v, basis_omega, initial, outlet, outlet_basis_u
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

        metrics_started = time.perf_counter()
        solid = cache_solid
        fluid = ~solid
        actual_u = gt["u"][: args.frames + 1, ::stride, ::stride].astype(np.float32)
        actual_v = gt["v"][: args.frames + 1, ::stride, ::stride].astype(np.float32)
        actual_omega = gt["vorticity"][: args.frames + 1, ::stride, ::stride].astype(
            np.float32
        )
        wake = wake_mask(cfg, cache_x, cache_y, solid, alpha)
        rows = metric_rows(
            actual_u[1:],
            actual_v[1:],
            actual_omega[1:],
            predicted_u[1:],
            predicted_v[1:],
            predicted_omega[1:],
            mean_u_grid,
            mean_v_grid,
            mean_omega_grid,
            fluid,
            wake,
            args.steps_per_frame,
        )
        method_timings["metric_computation_seconds"] = time.perf_counter() - metrics_started

        serialization_started = time.perf_counter()
        metrics_path = output_dir / f"{method}_temporal_metrics.csv"
        save_metrics(metrics_path, rows)
        fields_path = output_dir / f"{method}_rollout_fields_stride2.npz"
        np.savez_compressed(
            fields_path,
            x=cache_x,
            y=cache_y,
            solid=solid,
            actual_u=actual_u,
            actual_v=actual_v,
            actual_omega=actual_omega,
            predicted_u=predicted_u,
            predicted_v=predicted_v,
            predicted_omega=predicted_omega,
            compatible_mean_omega=mean_omega_grid,
            coefficient_frames=coefficient_frames,
            ritz_eigenvalues=eigenvalues,
        )
        method_timings.update(
            {
                "psi_generation_or_load_seconds": float(psi_diag["seconds"]),
                "fields_from_psi_seconds": float(chain_diag["seconds"]),
                "outlet_trace_seconds": float(outlet_diag["seconds"]),
                "ritz_seconds": float(ritz_diag["seconds"]),
                "base_rhs_and_correction_seconds": float(
                    rollout_diag["base_flow"]["seconds"]
                ),
                "coefficient_rollout_seconds": float(rollout_diag["seconds"]),
                "coefficient_rollout_per_frame_seconds": rollout_diag["per_frame_seconds"],
                "energy_budget_seconds": float(rollout_diag["energy_budget_seconds"]),
                "reconstruction_seconds": float(reconstruction_diag["seconds"]),
                "serialization_seconds": time.perf_counter() - serialization_started,
            }
        )
        method_timings["total_method_seconds"] = time.perf_counter() - method_started
        method_summary = {
            "complete": True,
            "method": method,
            "modes": args.modes,
            "initial_frame": 0,
            "predicted_frames": [1, args.frames],
            "initial_projection_relative_l2": initial_error,
            "basis_field_diagnostics": field_diag,
            "ritz_diagnostics": ritz_diag,
            "rollout_diagnostics": rollout_diag,
            "reconstruction_diagnostics": reconstruction_diag,
            "aggregate_errors": aggregate_errors(rows),
            "timings": method_timings,
            "metrics_csv": str(metrics_path.resolve()),
            "energy_budget_csv": str(energy_path.resolve()),
            "fields": str(fields_path.resolve()),
            "total_seconds": method_timings["total_method_seconds"],
        }
        write_json(output_dir / f"{method}_summary.json", method_summary)
        methods[method] = {"summary": method_summary, "metrics": rows}
        del predicted_u, predicted_v, predicted_omega, coefficient_frames
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()
        print(f"[{orientation}] {method} complete", flush=True)

    render_started = time.perf_counter()
    curve_methods = ("potential", "fd_mimetic")
    if set(curve_methods).issubset(methods):
        render_error_curves(
            output_dir,
            {method: methods[method]["metrics"] for method in curve_methods},
        )
    orientation_timings["figure_render_seconds"] = time.perf_counter() - render_started
    orientation_timings["total_orientation_seconds"] = time.perf_counter() - orientation_started
    result = {
        "complete": True,
        "orientation": orientation,
        "gt": str(gt_path.resolve()),
        "beta": beta,
        "alpha_deg": alpha,
        "modes": args.modes,
        "initial_gt_frame": 0,
        "predicted_gt_frames": [1, args.frames],
        "steps_per_frame": args.steps_per_frame,
        "grid_stride": args.grid_stride,
        "quadrature_points": complex_.node_count,
        "shared_native_mimetic_quadrature": True,
        "shared_41_frame_mean": True,
        "compatible_mean_vorticity": True,
        "base_flow_closure": args.base_flow_closure,
        "outlet": "edge-midpoint zero-gradient trace with frozen-base total-pressure subtraction",
        "coordinate_time_per_frame": output_dt,
        "coordinate_viscosity": viscosity,
        "methods": {method: methods[method]["summary"] for method in args.methods},
        "timings": orientation_timings,
        "total_seconds": orientation_timings["total_orientation_seconds"],
        "completed_at": now(),
    }
    write_json(output_dir / "comparison_summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    sequence = CASE_ROOT / "results" / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root", type=Path, default=sequence / "evaluation_three_geometries_frame30"
    )
    parser.add_argument(
        "--laplacian-cache-root",
        type=Path,
        default=CASE_ROOT
        / "results"
        / "siren_original_k64_to_k1024_e2000_sequence"
        / "shared_geometry_laplacian_cache_k2048",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=sequence / "arms" / "k2048" / "run_001" / "checkpoints" / "hardcoded_sss_frozen.npz",
    )
    parser.add_argument(
        "--peeled-state",
        type=Path,
        default=None,
        help=(
            "Optional archived KxK peeling transform for the selected learned method. "
            "Omit to use the raw learned span followed by the same native-grid Ritz step."
        ),
    )
    parser.add_argument(
        "--active-psi-cache",
        type=Path,
        default=None,
        help=(
            "Optional node_count x K .npy value-only streamfunction cache in the "
            "shared mimetic active-node order. Useful for legacy checkpoints whose "
            "network backend is intentionally kept in a separate exporter process."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=sequence / "temporal_rollout_gt30_k2048_mimetic_base_subtracted",
    )
    parser.add_argument("--orientations", nargs="+", choices=ORIENTATIONS, default=list(ORIENTATIONS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["potential", "fd_mimetic"],
    )
    parser.add_argument("--modes", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--steps-per-frame", type=int, default=500)
    parser.add_argument("--grid-stride", type=int, default=2)
    parser.add_argument(
        "--base-flow-closure",
        choices=("frozen-base-subtracted", "none"),
        default="frozen-base-subtracted",
    )
    parser.add_argument("--substeps-per-frame", type=int, default=10)
    parser.add_argument("--psi-point-batch", type=int, default=8192)
    parser.add_argument("--fd-batch-modes", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_learned = [method for method in args.methods if method in LEARNED_METHODS]
    if len(selected_learned) > 1:
        raise ValueError(
            "one invocation accepts at most one learned method because --checkpoint "
            "and --peeled-state are method-specific"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "status": "running",
        "started_at": now(),
        "orientations": args.orientations,
        "modes": args.modes,
        "frames": args.frames,
        "protocol": {
            "initial_condition": (
                f"GT frame 0 projected independently into each K={args.modes} space"
            ),
            "targets": "existing GT frames 1 through 30",
            "coefficient_ode": "de Witt 2012 vorticity-Galerkin ROM",
            "time_integrator": "implicit trapezoidal variational update with exact viscous split",
            "spatial_chain": "shared native masked-Cartesian incidence grad--curl--Laplacian chain",
            "learned_rayleigh": {
                "definition": "||grad(F_k)||_h^2 / ||F_k||_h^2",
                "observables": {
                    "potential": "F_k=A_k",
                    "velocity": "F_k=u_k",
                    "vorticity": "F_k=omega_k",
                },
            },
            "rollout_basis_observable": (
                "value-only streamfunction from each learned checkpoint; "
                "no learned spatial derivatives in rollout"
            ),
            "fd_observable": "cached native-grid streamfunction; no interpolation or np.gradient",
            "quadrature": "same native active-node trapezoidal weights for both methods",
            "base_flow_closure": args.base_flow_closure,
            "outlet": (
                "incremental total-pressure trace with frozen-base residual subtraction; "
                "not claimed to recover unavailable GT pressure"
            ),
            "same_mean_dt_viscosity_and_boundary_terms": True,
        },
    }
    write_json(args.output_root / "state.json", state)
    total_started = time.perf_counter()
    results: dict[str, Any] = {}
    for orientation in args.orientations:
        complete = args.output_root / orientation / "comparison_summary.json"
        if complete.exists():
            payload = json.loads(complete.read_text(encoding="utf-8"))
            if (
                bool(payload.get("complete"))
                and payload.get("base_flow_closure") == args.base_flow_closure
                and int(payload.get("grid_stride", -1)) == args.grid_stride
                and int(payload.get("modes", -1)) == args.modes
                and set(payload.get("methods", {})) == set(args.methods)
            ):
                results[orientation] = payload
                print(f"[{orientation}] already complete; reuse", flush=True)
                continue
        state.update({"orientation": orientation, "updated_at": now()})
        write_json(args.output_root / "state.json", state)
        results[orientation] = run_orientation(args, orientation)
    state.update(
        {
            "status": "complete",
            "completed_at": now(),
            "total_seconds": time.perf_counter() - total_started,
            "results": results,
        }
    )
    write_json(args.output_root / "state.json", state)
    print(json.dumps(strict_json(state), indent=2), flush=True)


if __name__ == "__main__":
    main()
