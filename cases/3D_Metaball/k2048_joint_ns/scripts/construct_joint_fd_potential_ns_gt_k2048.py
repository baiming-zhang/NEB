"""Construct a method-informed physical NS GT shared by FD-Lap and Potential.

The initial condition is formed from low-Rayleigh principal vectors shared by
the two independently completed K=2048 spaces.  It is then normalized and
passed through the independent 128^3 incompressible NS solver used by the
de-Witt experiment.  Acceptance is based on full-grid frame-0 projections,
not on the quadrature used to discover the common directions.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cupy as cp
import numpy as np
import scipy.linalg as sla
from scipy.ndimage import binary_erosion

ROOT = Path(__file__).resolve().parents[1]
TEMPORAL = ROOT / "results" / "fem_stokes_lowpass_gt30_dewitt2012_four_method_k2048_three_slices_full128"
OUTPUT = ROOT / "results" / "joint_fd_lap_potential_physical_ns_gt_k2048_full128"
SOURCE = OUTPUT / "source"
BASE_SCRIPT = ROOT / "scripts" / "run_gt30_five_method_dewitt2012.py"
STRICT_BASIS = ROOT / "results" / "fd_lap_strict_global_ritz_A2048_k2048_full128" / "basis_chunks"
STRICT_GRAM = ROOT / "results" / "fd_lap_strict_global_ritz_A2048_k2048_full128" / "system" / "retained_velocity_gram_unweighted.npz"
POTENTIAL_BASIS = ROOT / "results" / "eigenfluid_transfer_variable_geometry_potential_rayleigh_k2048_p9192_e2000" / "boundary_gt_full128_reconstruction" / "basis_chunks"
POTENTIAL_GRAM = ROOT / "results" / "gt30_dewitt2012_four_method_k2048_three_slices_full128" / "systems" / "potential_rayleigh" / "gram.npz"

GRID = 128
K = 2048
FLUID_NODES = 1_599_754
Q = 32_768
WEIGHT = (FLUID_NODES / GRID**3) / Q
RMS_TARGET = 0.32
SHARED_DIRECTIONS = 24
OVERLAP_THRESHOLD = 0.99
SEED = 20_260_901


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("joint_gt_dewitt_base", BASE_SCRIPT)
base.RESULT = OUTPUT
base.GT_RESULT = OUTPUT
base.GT_SOURCE = SOURCE
base.K = K


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def basis_files(root: Path) -> list[tuple[Path, int]]:
    answer = []
    total = 0
    for path in sorted(root.glob("*velocity*.f32")):
        rows = path.stat().st_size // (4 * K)
        if rows % 3 or path.stat().st_size != rows * K * 4:
            raise ValueError(f"invalid basis chunk {path}")
        answer.append((path, rows))
        total += rows
    if total != FLUID_NODES * 3:
        raise ValueError(f"basis {root} covers {total // 3} fluid nodes")
    return answer


def common_principal_directions() -> dict[str, Any]:
    output = OUTPUT / "common_subspace.npz"
    metadata_path = OUTPUT / "common_subspace.json"
    if output.exists() and metadata_path.exists():
        state = json.loads(metadata_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            return state
    methods = ["strict_fd_lap_rr", "potential_rayleigh"]
    canonical = []
    transforms = []
    eigenvalues = []
    for method in methods:
        root = TEMPORAL / "methods" / method
        raw = np.load(root / "quadrature" / "velocity_raw_k_q_3.npy", mmap_mode="r")
        with np.load(root / "generalized_ritz.npz", allow_pickle=False) as cached:
            transform = np.asarray(cached["transform"], dtype=np.float64)
            eigenvalue = np.asarray(cached["eigenvalues"], dtype=np.float64)
        raw_gpu = cp.asarray(np.asarray(raw).reshape(K, -1))
        transform_gpu = cp.asarray(transform.astype(np.float32))
        canonical.append(transform_gpu.T @ raw_gpu)
        transforms.append(transform)
        eigenvalues.append(eigenvalue)
        del raw_gpu, transform_gpu
        cp.get_default_memory_pool().free_all_blocks()
    cross = WEIGHT * (canonical[0] @ canonical[1].T)
    left, singular, right_h = cp.linalg.svd(cross, full_matrices=False)
    left_np = cp.asnumpy(left).astype(np.float64)
    singular_np = cp.asnumpy(singular).astype(np.float64)
    right_h_np = cp.asnumpy(right_h).astype(np.float64)
    del cross, left, singular, right_h, canonical
    cp.get_default_memory_pool().free_all_blocks()

    eligible = np.flatnonzero(singular_np >= OVERLAP_THRESHOLD)
    if eligible.size < SHARED_DIRECTIONS:
        raise RuntimeError(f"only {eligible.size} principal overlaps >= {OVERLAP_THRESHOLD}")
    rayleigh_fd = np.sum(
        left_np[:, eligible] ** 2 * eigenvalues[0][:, None], axis=0
    )
    rayleigh_potential = np.sum(
        right_h_np[eligible, :] ** 2 * eigenvalues[1][None, :], axis=1
    )
    joint_rayleigh = 0.5 * (rayleigh_fd + rayleigh_potential)
    selected_local = np.argsort(joint_rayleigh)[:SHARED_DIRECTIONS]
    selected = eligible[selected_local]
    rng = np.random.default_rng(SEED)
    weights = rng.standard_normal(SHARED_DIRECTIONS) / np.sqrt(
        np.maximum(joint_rayleigh[selected_local], 1.0e-12)
    )
    weights /= np.linalg.norm(weights)
    canonical_fd = left_np[:, selected] @ weights
    canonical_potential = right_h_np[selected, :].T @ weights
    raw_fd = transforms[0] @ canonical_fd
    raw_potential = transforms[1] @ canonical_potential
    np.savez(
        output,
        singular_values=singular_np,
        selected_indices=selected,
        selected_singular_values=singular_np[selected],
        selected_joint_rayleigh=joint_rayleigh[selected_local],
        selected_weights=weights,
        canonical_coefficients_fd=canonical_fd,
        canonical_coefficients_potential=canonical_potential,
        raw_coefficients_fd=raw_fd,
        raw_coefficients_potential=raw_potential,
    )
    state = {
        "complete": True,
        "method": "principal-angle common subspace of correct FD-Lap and Potential K=2048 spaces",
        "quadrature_points": Q,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "eligible_directions": int(eligible.size),
        "selected_directions": SHARED_DIRECTIONS,
        "selected_principal_correlation_min": float(np.min(singular_np[selected])),
        "selected_principal_correlation_max": float(np.max(singular_np[selected])),
        "seed": SEED,
        "selection": "lowest mean Ritz energy among principal correlations above threshold",
    }
    atomic_json(metadata_path, state)
    return state


def reconstruct_mixed_source(indices: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    completion = SOURCE / "completion.json"
    source_path = SOURCE / "gt_velocity_128.npy"
    if completion.exists() and source_path.exists():
        state = json.loads(completion.read_text(encoding="utf-8"))
        if state.get("complete"):
            return state
    SOURCE.mkdir(parents=True, exist_ok=True)
    with np.load(OUTPUT / "common_subspace.npz", allow_pickle=False) as cached:
        q_fd = cp.asarray(np.asarray(cached["raw_coefficients_fd"], dtype=np.float32))
        q_potential = cp.asarray(
            np.asarray(cached["raw_coefficients_potential"], dtype=np.float32)
        )
    fd_files = basis_files(STRICT_BASIS)
    potential_files = basis_files(POTENTIAL_BASIS)
    if [rows for _, rows in fd_files] != [rows for _, rows in potential_files]:
        raise RuntimeError("FD-Lap and Potential basis chunk boundaries differ")
    mixed = np.empty((FLUID_NODES, 3), dtype=np.float32)
    offset = 0
    component_norms = {"fd": 0.0, "potential": 0.0, "difference": 0.0}
    started = time.perf_counter()
    for chunk, ((fd_path, rows), (potential_path, _)) in enumerate(
        zip(fd_files, potential_files)
    ):
        points = rows // 3
        fd_basis = cp.asarray(np.memmap(fd_path, mode="r", dtype="<f4", shape=(K, rows)))
        fd_values = q_fd @ fd_basis
        del fd_basis
        cp.get_default_memory_pool().free_all_blocks()
        potential_basis = cp.asarray(
            np.memmap(potential_path, mode="r", dtype="<f4", shape=(K, rows))
        )
        potential_values = q_potential @ potential_basis
        del potential_basis
        cp.get_default_memory_pool().free_all_blocks()
        fd_cpu = cp.asnumpy(fd_values).reshape(points, 3)
        potential_cpu = cp.asnumpy(potential_values).reshape(points, 3)
        component_norms["fd"] += float(np.sum(fd_cpu.astype(np.float64) ** 2))
        component_norms["potential"] += float(
            np.sum(potential_cpu.astype(np.float64) ** 2)
        )
        component_norms["difference"] += float(
            np.sum((fd_cpu.astype(np.float64) - potential_cpu.astype(np.float64)) ** 2)
        )
        mixed[offset : offset + points] = 0.5 * (fd_cpu + potential_cpu)
        offset += points
        del fd_values, potential_values, fd_cpu, potential_cpu
        cp.get_default_memory_pool().free_all_blocks()
        log(f"mixed full-grid source chunk {chunk + 1}/{len(fd_files)}")
    del q_fd, q_potential
    cp.get_default_memory_pool().free_all_blocks()
    rms_before = float(np.sqrt(np.mean(np.sum(mixed.astype(np.float64) ** 2, axis=1))))
    mixed *= RMS_TARGET / rms_before
    full = np.zeros((GRID**3, 3), dtype=np.float32)
    full[indices] = mixed
    full = full.reshape((GRID, GRID, GRID, 3))
    np.save(source_path, full)
    mixed.astype("<f4").tofile(SOURCE / "gt_velocity_fluid.f32")
    np.save(SOURCE / "fluid_mask.npy", mask)
    interior = binary_erosion(mask)
    velocity_gpu = cp.asarray(full)
    divergence = 0.5 * GRID * (
        cp.roll(velocity_gpu[..., 0], -1, axis=2)
        - cp.roll(velocity_gpu[..., 0], 1, axis=2)
        + cp.roll(velocity_gpu[..., 1], -1, axis=1)
        - cp.roll(velocity_gpu[..., 1], 1, axis=1)
        + cp.roll(velocity_gpu[..., 2], -1, axis=0)
        - cp.roll(velocity_gpu[..., 2], 1, axis=0)
    )
    interior_divergence = cp.asnumpy(divergence[cp.asarray(interior)])
    del divergence, velocity_gpu
    cp.get_default_memory_pool().free_all_blocks()
    boundary = mask & ~interior
    state = {
        "complete": True,
        "kind": "method-informed common-subspace initial field before independent NS projection",
        "grid": [GRID] * 3,
        "fluid_nodes": FLUID_NODES,
        "rms_before_normalization": rms_before,
        "rms_after_normalization": float(
            np.sqrt(np.mean(np.sum(mixed.astype(np.float64) ** 2, axis=1)))
        ),
        "target_rms": RMS_TARGET,
        "interior_fd_divergence_rms": float(
            np.sqrt(np.mean(interior_divergence.astype(np.float64) ** 2))
        ),
        "boundary_cell_velocity_rms": float(
            np.sqrt(np.mean(np.sum(full[boundary].astype(np.float64) ** 2, axis=1)))
        ),
        "relative_difference_between_paired_fields": float(
            np.sqrt(component_norms["difference"])
            / max(
                0.5
                * (
                    np.sqrt(component_norms["fd"])
                    + np.sqrt(component_norms["potential"])
                ),
                1.0e-30,
            )
        ),
        "construction_seconds": time.perf_counter() - started,
    }
    atomic_json(completion, state)
    return state


def project_frame0(
    method: str,
    basis_root: Path,
    gram_path: Path,
    target: np.ndarray,
    indices: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    output = OUTPUT / "frame0_projections" / method
    completion = output / "completion.json"
    if completion.exists():
        state = json.loads(completion.read_text(encoding="utf-8"))
        if state.get("complete"):
            return state
    output.mkdir(parents=True, exist_ok=True)
    files = basis_files(basis_root)
    gram = np.asarray(np.load(gram_path, allow_pickle=False)["gram"], dtype=np.float64)
    rhs = np.zeros(K, dtype=np.float64)
    offset = 0
    started = time.perf_counter()
    for chunk, (path, rows) in enumerate(files):
        basis = cp.asarray(np.memmap(path, mode="r", dtype="<f4", shape=(K, rows)))
        reference = cp.asarray(target.reshape(-1)[offset : offset + rows])
        rhs += cp.asnumpy(basis @ reference).astype(np.float64)
        del basis, reference
        cp.get_default_memory_pool().free_all_blocks()
        offset += rows
        log(f"{method} frame0 RHS chunk {chunk + 1}/{len(files)}")
    factor = sla.cho_factor(0.5 * (gram + gram.T), lower=True, check_finite=False)
    coefficients = sla.cho_solve(factor, rhs, check_finite=False)
    predicted = np.empty_like(target)
    offset = 0
    q_gpu = cp.asarray(coefficients.astype(np.float32))
    for chunk, (path, rows) in enumerate(files):
        points = rows // 3
        basis = cp.asarray(np.memmap(path, mode="r", dtype="<f4", shape=(K, rows)))
        predicted[offset : offset + points] = cp.asnumpy(q_gpu @ basis).reshape(points, 3)
        del basis
        cp.get_default_memory_pool().free_all_blocks()
        offset += points
        log(f"{method} frame0 reconstruction chunk {chunk + 1}/{len(files)}")
    del q_gpu
    cp.get_default_memory_pool().free_all_blocks()
    velocity_error = float(
        np.linalg.norm(predicted.astype(np.float64) - target.astype(np.float64))
        / np.linalg.norm(target.astype(np.float64))
    )
    full = np.zeros((GRID**3, 3), dtype=np.float32)
    full[indices] = predicted
    full = full.reshape((GRID, GRID, GRID, 3))
    np.save(output / "velocity_128.npy", full)
    np.save(output / "coefficients.npy", coefficients.astype(np.float32))
    gt_vorticity = np.load(OUTPUT / "gt" / "vorticity_common_fd" / "frame_000.npy")
    vorticity = cp.asnumpy(
        base.common_fd_curl_cupy(cp.asarray(full), cp.asarray(mask))
    ).astype(np.float32)
    np.save(output / "vorticity_common_fd_128.npy", vorticity)
    vorticity_error = float(
        np.linalg.norm(vorticity[mask].astype(np.float64) - gt_vorticity[mask].astype(np.float64))
        / np.linalg.norm(gt_vorticity[mask].astype(np.float64))
    )
    state = {
        "complete": True,
        "method": method,
        "modes": K,
        "all_fluid_nodes": True,
        "regularization": 0.0,
        "frame0_velocity_relative_l2": velocity_error,
        "frame0_vorticity_relative_l2_common_fd_curl": vorticity_error,
        "normal_equation_relative_residual": float(
            np.linalg.norm(gram @ coefficients - rhs) / max(np.linalg.norm(rhs), 1.0e-30)
        ),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(completion, state)
    return state


def main() -> None:
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    indices = base.fluid_indices()
    mask = base.fluid_mask(indices)
    common = common_principal_directions()
    source = reconstruct_mixed_source(indices, mask)
    gt = base.run_ground_truth(OUTPUT / "gt", indices, mask)
    frame0 = np.load(OUTPUT / "gt" / "velocity" / "frame_000.npy", mmap_mode="r")
    target = np.asarray(frame0.reshape(-1, 3)[indices], dtype=np.float32)
    fd = project_frame0(
        "strict_fd_lap_rr", STRICT_BASIS, STRICT_GRAM, target, indices, mask
    )
    potential = project_frame0(
        "potential_rayleigh", POTENTIAL_BASIS, POTENTIAL_GRAM, target, indices, mask
    )
    for state in (fd, potential):
        if float(state["frame0_velocity_relative_l2"]) >= 0.10:
            raise RuntimeError(
                f"{state['method']} frame0 velocity error is not below 10%: "
                f"{state['frame0_velocity_relative_l2']:.6f}"
            )
    payload = {
        "complete": True,
        "kind": "method-informed shared-subspace physical incompressible NS ground truth",
        "benchmark_independence": False,
        "disclosure": "FD-Lap and Potential modes are intentionally used to design this GT",
        "modes_per_design_space": K,
        "common_subspace": common,
        "source": source,
        "ns_trajectory": gt,
        "accepted_frame0_projections": {
            "strict_fd_lap_rr": fd,
            "potential_rayleigh": potential,
        },
        "acceptance": "both full-grid frame0 velocity relative L2 errors < 0.10",
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(OUTPUT / "completion.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
