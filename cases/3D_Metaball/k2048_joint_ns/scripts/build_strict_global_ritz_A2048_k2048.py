"""Build the strict FD-Lap K=2048 basis by a global 6144->2048 Ritz solve.

This is the direct K=2048 extension of the validated K=1024 construction:
2048 Dirichlet scalar shapes times three antisymmetric tensor polarizations
form 6144 vector candidates.  One global full-grid generalized eigenproblem
retains the lowest 2048 complete vector modes.  No local 3x3 polarization
collapse is used.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as sla
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "laplacian_strict_complete_A2048_basis_full128"
OUTPUT = ROOT / "results" / "fd_lap_strict_global_ritz_A2048_k2048_full128"
GT_ROOT = ROOT / "results" / "fem_stokes_lowpass_gt_k1024_four_method_full128" / "gt"
COMPLETE_SCRIPT = ROOT / "scripts" / "reconstruct_fd_lap_complete_A2048_k2048_full128.py"
K1024_SCRIPT = ROOT / "scripts" / "build_strict_matched_fd_lap_vector_A1024_k1024.py"

GRID = 128
FLUID_NODES = 1_599_754
SCALAR_SHAPES = 2048
POLARIZATIONS = 3
CANDIDATES = SCALAR_SHAPES * POLARIZATIONS
K = 2048
WEIGHT = (1.0 / GRID) ** 3
CHUNK_POINTS = 262_144


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


complete = load_module("strict_a2048_complete_context", COMPLETE_SCRIPT)
vector = load_module("strict_a2048_global_vector_context", K1024_SCRIPT)
core = complete.core

# All reused helpers read these module globals at call time.
core.A_GROUPS = SCALAR_SHAPES
core.SCALAR_DOFS = CANDIDATES
vector.SCALAR_SHAPES = SCALAR_SHAPES
vector.CANDIDATES = CANDIDATES
vector.K = K


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def trim_working_set() -> None:
    """Evict clean mapped pages with pointer-safe Windows API declarations."""
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.restype = ctypes.c_int
    if not psapi.EmptyWorkingSet(kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())


def transpose_gradient_basis_fast(
    mode_major_path: Path,
    output_path: Path,
    progress_path: Path,
) -> np.memmap:
    """Transpose with large sequential HDD reads and writes.

    The legacy 8192-row implementation seeks across all 2048 mode slabs for
    every small block.  A 262144-row in-memory block converts those seeks into
    multi-megabyte sequential reads, then commits one contiguous 6.4 GB write.
    """
    source = np.load(mode_major_path, mmap_mode="r")
    if source.shape != (SCALAR_SHAPES, FLUID_NODES, 3):
        raise ValueError(f"unexpected mode-major gradient shape {source.shape}")
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != (FLUID_NODES, 3, SCALAR_SHAPES):
            raise ValueError(f"unexpected spatial gradient shape {output.shape}")
    else:
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(FLUID_NODES, 3, SCALAR_SHAPES),
        )
    next_row = 0
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            log("reuse complete fast spatial-major gradient basis")
            return np.load(output_path, mmap_mode="r")
        next_row = int(state.get("next_row", 0))
    chunks = (FLUID_NODES + CHUNK_POINTS - 1) // CHUNK_POINTS
    for begin in range(next_row, FLUID_NODES, CHUNK_POINTS):
        end = min(FLUID_NODES, begin + CHUNK_POINTS)
        points = end - begin
        block = np.empty((points, 3, SCALAR_SHAPES), dtype=np.float32)
        for mode_begin in range(0, SCALAR_SHAPES, 32):
            mode_end = min(SCALAR_SHAPES, mode_begin + 32)
            values = np.array(
                source[mode_begin:mode_end, begin:end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
            block[:, :, mode_begin:mode_end] = values.transpose(1, 2, 0)
            del values
        output[begin:end] = block
        output.flush()
        del block
        trim_working_set()
        complete_chunks = end // CHUNK_POINTS + int(end % CHUNK_POINTS != 0)
        atomic_json(
            progress_path,
            {
                "complete": end == FLUID_NODES,
                "next_row": end,
                "complete_chunks": complete_chunks,
                "chunks": chunks,
                "tensor_A_groups": SCALAR_SHAPES,
                "layout": "fluid-node x gradient-component x scalar-shape",
                "io": "262144-row in-memory aggregation with sequential commit",
            },
        )
        log(f"fast spatial-major gradient chunk {complete_chunks}/{chunks}")
    return np.load(output_path, mmap_mode="r")


def transpose_candidate_vorticity_fast(
    mode_major_path: Path,
    output_path: Path,
    progress_path: Path,
) -> np.memmap:
    """Transpose 118 GB with sequential mode reads and contiguous commits."""
    source = np.load(mode_major_path, mmap_mode="r")
    if source.shape != (CANDIDATES, FLUID_NODES, 3):
        raise ValueError(f"unexpected vorticity mode-major shape {source.shape}")
    if output_path.exists():
        output = np.load(output_path, mmap_mode="r+")
        if output.shape != (FLUID_NODES, 3, CANDIDATES):
            raise ValueError(f"unexpected vorticity spatial shape {output.shape}")
    else:
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(FLUID_NODES, 3, CANDIDATES),
        )
    row_chunk = 65_536
    next_row = 0
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            log("reuse complete fast candidate-vorticity transpose")
            return np.load(output_path, mmap_mode="r")
        next_row = int(state.get("next_row", 0))
    chunks = (FLUID_NODES + row_chunk - 1) // row_chunk
    for begin in range(next_row, FLUID_NODES, row_chunk):
        end = min(FLUID_NODES, begin + row_chunk)
        points = end - begin
        block = np.empty((points, 3, CANDIDATES), dtype=np.float32)
        for mode_begin in range(0, CANDIDATES, 32):
            mode_end = min(CANDIDATES, mode_begin + 32)
            values = np.array(
                source[mode_begin:mode_end, begin:end],
                dtype=np.float32,
                copy=True,
                order="C",
            )
            block[:, :, mode_begin:mode_end] = values.transpose(1, 2, 0)
            del values
        output[begin:end] = block
        output.flush()
        del block
        trim_working_set()
        complete_chunks = end // row_chunk + int(end % row_chunk != 0)
        atomic_json(
            progress_path,
            {
                "complete": end == FLUID_NODES,
                "next_row": end,
                "complete_chunks": complete_chunks,
                "chunks": chunks,
                "candidate_vector_dofs": CANDIDATES,
                "layout": "fluid-node x vector-component x candidate",
                "io": "65536-row in-memory aggregation with sequential commit",
            },
        )
        log(f"fast candidate-vorticity transpose chunk {complete_chunks}/{chunks}")
    return np.load(output_path, mmap_mode="r")


def write_retained_basis_chunks(
    gradient_path: Path,
    mixing: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    """Evaluate the retained modes without materializing 6144 velocities."""
    chunk_dir = output_root / "basis_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    gradient = np.load(gradient_path, mmap_mode="r")
    if gradient.shape != (FLUID_NODES, 3, SCALAR_SHAPES):
        raise ValueError(f"unexpected spatial gradient shape {gradient.shape}")
    if mixing.shape != (CANDIDATES, K):
        raise ValueError(f"unexpected Ritz mixing shape {mixing.shape}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    mix = torch.as_tensor(mixing.astype(np.float32), device="cuda")
    mix12 = mix[:SCALAR_SHAPES]
    mix13 = mix[SCALAR_SHAPES : 2 * SCALAR_SHAPES]
    mix23 = mix[2 * SCALAR_SHAPES :]
    chunks = (FLUID_NODES + CHUNK_POINTS - 1) // CHUNK_POINTS
    started = time.perf_counter()
    for chunk in range(chunks):
        begin = chunk * CHUNK_POINTS
        end = min(FLUID_NODES, begin + CHUNK_POINTS)
        points = end - begin
        target = chunk_dir / f"chunk_{chunk:02d}_velocity.f32"
        expected = K * points * 3 * 4
        if target.exists() and target.stat().st_size == expected:
            log(f"reuse retained velocity basis chunk {chunk + 1}/{chunks}")
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        output = np.memmap(temporary, mode="w+", dtype="<f4", shape=(K, points, 3))
        values = torch.as_tensor(
            np.asarray(gradient[begin:end]), device="cuda", dtype=torch.float32
        )
        dx, dy, dz = values[:, 0], values[:, 1], values[:, 2]
        for mode_begin in range(0, K, 64):
            mode_end = min(K, mode_begin + 64)
            ux = dy @ mix12[:, mode_begin:mode_end] + dz @ mix13[:, mode_begin:mode_end]
            uy = -dx @ mix12[:, mode_begin:mode_end] + dz @ mix23[:, mode_begin:mode_end]
            uz = -dx @ mix13[:, mode_begin:mode_end] - dy @ mix23[:, mode_begin:mode_end]
            block = torch.stack((ux, uy, uz), dim=2).permute(1, 0, 2).cpu().numpy()
            output[mode_begin:mode_end] = block
            del block, ux, uy, uz
        output.flush()
        del output, dx, dy, dz, values
        torch.cuda.empty_cache()
        os.replace(temporary, target)
        atomic_json(
            output_root / "basis_chunk_progress.json",
            {
                "complete": chunk + 1 == chunks,
                "complete_chunks": chunk + 1,
                "chunks": chunks,
                "modes": K,
                "fluid_nodes": FLUID_NODES,
            },
        )
        log(f"wrote retained velocity basis chunk {chunk + 1}/{chunks}")
    del mix, mix12, mix13, mix23
    torch.cuda.empty_cache()
    return {
        "complete": True,
        "chunks": chunks,
        "basis": str(chunk_dir.resolve()),
        "seconds": time.perf_counter() - started,
    }


def generalized_modes_rank_revealing(
    mass: np.ndarray,
    stiffness: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Solve the global Ritz problem on the positive numerical mass range.

    The true velocity mass is a Gram matrix and therefore positive
    semidefinite.  At 6144 candidates, the unchanged FP32 full-grid Gram
    accumulation exposes a small numerical nullspace (and roundoff-negative
    eigenvalues).  Mass eigendecomposition removes only directions below the
    same 1e-10 numerical-rank threshold already reported by the K=1024 audit;
    no diagonal shift or Tikhonov regularization is introduced.
    """
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as cached:
            mixing = np.asarray(cached["mixing"], dtype=np.float64)
            eigenvalues = np.asarray(cached["eigenvalues"], dtype=np.float64)
            audit = json.loads(str(cached["audit_json"].item()))
        if mixing.shape == (CANDIDATES, K) and eigenvalues.shape == (K,):
            log("reuse complete rank-revealing global generalized eigensolve")
            return mixing, eigenvalues, audit

    log("mass rank decomposition for 6144-candidate global Ritz")
    mass_eigenvalues, mass_vectors = sla.eigh(
        0.5 * (mass + mass.T), check_finite=False, driver="evd"
    )
    threshold = float(mass_eigenvalues[-1] * 1.0e-10)
    keep = mass_eigenvalues > threshold
    numerical_rank = int(np.count_nonzero(keep))
    if numerical_rank < K:
        raise RuntimeError(f"candidate mass numerical rank {numerical_rank} < {K}")
    whitener = mass_vectors[:, keep] / np.sqrt(mass_eigenvalues[keep])[None, :]
    del mass_vectors
    log(
        f"mass whitening retains {numerical_rank}/{CANDIDATES} directions "
        f"at threshold {threshold:.3e}"
    )
    reduced = whitener.T @ (stiffness @ whitener)
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_modes = sla.eigh(
        reduced,
        subset_by_index=(0, K - 1),
        driver="evr",
        check_finite=False,
    )
    mixing = whitener @ reduced_modes
    del reduced_modes, reduced, whitener
    final_mass = mixing.T @ mass @ mixing
    residual = stiffness @ mixing - (mass @ mixing) * eigenvalues[None, :]
    denominator = max(
        np.linalg.norm(stiffness @ mixing),
        np.linalg.norm((mass @ mixing) * eigenvalues[None, :]),
        1.0e-300,
    )
    audit = {
        "solver": "rank-revealing mass whitening followed by global standard symmetric eigensolve",
        "regularization": 0.0,
        "candidate_dimension": CANDIDATES,
        "mass_numerical_rank": numerical_rank,
        "discarded_numerical_null_directions": CANDIDATES - numerical_rank,
        "mass_rank_relative_threshold": 1.0e-10,
        "mass_rank_absolute_threshold": threshold,
        "mass_eigenvalue_min_raw": float(mass_eigenvalues[0]),
        "mass_eigenvalue_min_retained": float(mass_eigenvalues[keep][0]),
        "mass_eigenvalue_max": float(mass_eigenvalues[-1]),
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "mass_orthogonality_max_abs": float(np.max(np.abs(final_mass - np.eye(K)))),
        "generalized_eigen_relative_residual": float(np.linalg.norm(residual) / denominator),
        "trial_candidates": CANDIDATES,
        "retained_complete_vector_modes": K,
    }
    atomic_npz(
        output_path,
        mixing=mixing,
        eigenvalues=eigenvalues,
        mass_eigenvalues=mass_eigenvalues,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return mixing, eigenvalues, audit


def assemble_retained_gram_and_projection(
    basis_root: Path,
    target: np.ndarray,
    gram_path: Path,
) -> tuple[float, dict[str, Any]]:
    """Audit the evaluated 2048-mode basis, independent of candidate Gram."""
    weighted_mass = np.zeros((K, K), dtype=np.float64)
    weighted_rhs = np.zeros(K, dtype=np.float64)
    chunks = (FLUID_NODES + CHUNK_POINTS - 1) // CHUNK_POINTS
    started = time.perf_counter()
    for chunk in range(chunks):
        begin = chunk * CHUNK_POINTS
        end = min(FLUID_NODES, begin + CHUNK_POINTS)
        points = end - begin
        path = basis_root / "basis_chunks" / f"chunk_{chunk:02d}_velocity.f32"
        basis = np.memmap(path, mode="r", dtype="<f4", shape=(K, points * 3))
        values = torch.as_tensor(np.asarray(basis), device="cuda", dtype=torch.float32)
        reference = torch.as_tensor(
            np.asarray(target[0, begin:end]).reshape(-1),
            device="cuda",
            dtype=torch.float32,
        )
        weighted_mass += (WEIGHT * (values @ values.T)).double().cpu().numpy()
        weighted_rhs += (WEIGHT * (values @ reference)).double().cpu().numpy()
        del reference, values, basis
        torch.cuda.empty_cache()
        log(f"evaluated retained Gram block {chunk + 1}/{chunks}")
    weighted_mass = 0.5 * (weighted_mass + weighted_mass.T)
    eigenvalues = sla.eigvalsh(weighted_mass, check_finite=False, driver="evd")
    factor = sla.cho_factor(weighted_mass, lower=True, check_finite=False)
    coefficients = sla.cho_solve(factor, weighted_rhs, check_finite=False)
    reference_energy = float(WEIGHT * np.sum(target.astype(np.float64) ** 2))
    error_energy = float(
        reference_energy
        - 2.0 * np.dot(coefficients, weighted_rhs)
        + coefficients @ weighted_mass @ coefficients
    )
    relative_error = float(np.sqrt(max(error_energy, 0.0) / reference_energy))
    normal_residual = float(
        np.linalg.norm(weighted_mass @ coefficients - weighted_rhs)
        / max(np.linalg.norm(weighted_rhs), 1.0e-300)
    )
    audit = {
        "source": "direct full-grid accumulation from evaluated retained basis chunks",
        "weighted_mass_min_eigenvalue": float(eigenvalues[0]),
        "weighted_mass_max_eigenvalue": float(eigenvalues[-1]),
        "weighted_mass_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "mass_identity_max_abs": float(np.max(np.abs(weighted_mass - np.eye(K)))),
        "projection_normal_relative_residual": normal_residual,
        "fem_gt_velocity_relative_l2": relative_error,
        "seconds": time.perf_counter() - started,
    }
    atomic_npz(
        gram_path,
        gram=weighted_mass / WEIGHT,
        weighted_mass=weighted_mass,
        weighted_rhs=weighted_rhs,
        fem_gt_projection_coefficients=coefficients,
        audit_json=np.asarray(json.dumps(audit)),
    )
    return relative_error, audit


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / "cache"
    system = OUTPUT / "system"
    cache.mkdir(exist_ok=True)
    system.mkdir(exist_ok=True)
    started = time.perf_counter()
    atomic_json(
        OUTPUT / "protocol.json",
        {
            "status": "running",
            "method": "strict matched FD-Lap global vector Rayleigh-Ritz",
            "potential": "A_(k,alpha)=e phi_D,k E_alpha ~ d^2",
            "source_scalar_shapes": SCALAR_SHAPES,
            "tensor_polarizations_per_shape": POLARIZATIONS,
            "trial_vector_candidates": CANDIDATES,
            "global_ritz_retained_modes": K,
            "local_polarization_collapse": False,
            "grid": [GRID] * 3,
            "fluid_nodes": FLUID_NODES,
            "subsampling": False,
            "tf32": False,
        },
    )

    _cfg, _geometry, mask, _, _ = core.previous.load_context(ROOT)
    _, flat_indices = complete.helper.cell_center_points(mask)
    if flat_indices.size != FLUID_NODES:
        raise ValueError(f"fluid-node mismatch: {flat_indices.size}")
    gradient_mode = SOURCE / "gradient_e_phiD_A2048_mode_major_128.npy"
    values = np.load(gradient_mode, mmap_mode="r")
    if values.shape != (SCALAR_SHAPES, FLUID_NODES, 3) or values.dtype != np.float32:
        raise ValueError(f"invalid source gradients: {values.shape}, {values.dtype}")
    del values

    gradient_spatial = cache / "gradient_e_phiD_A2048_spatial_major_128.npy"
    transpose_gradient_basis_fast(
        gradient_mode,
        gradient_spatial,
        cache / "gradient_spatial_progress.json",
    )

    gt_velocity_path = GT_ROOT / "gt_velocity_128.npy"
    gt_velocity = np.load(gt_velocity_path, mmap_mode="r")
    target = np.asarray(gt_velocity[mask], dtype=np.float32)[None, ...]
    mass, _correlations, mass_audit = core.assemble_mass_and_correlations(
        gradient_spatial,
        target,
        system / "candidate_mass_and_fem_gt_correlation_6144.npz",
        system / "candidate_mass_progress.json",
        spatial_blocks=3,
    )
    if mass.shape != (CANDIDATES, CANDIDATES):
        raise ValueError(f"candidate mass shape {mass.shape}")

    vorticity_mode = cache / "candidate_vorticity_6144_mode_major_128.npy"
    vector.build_candidate_vorticity_mode_major(
        gradient_mode,
        mask,
        flat_indices,
        vorticity_mode,
        cache / "candidate_vorticity_mode_progress.json",
        scalar_batch=4,
    )
    vorticity_spatial = cache / "candidate_vorticity_6144_spatial_major_128.npy"
    transpose_candidate_vorticity_fast(
        vorticity_mode,
        vorticity_spatial,
        cache / "candidate_vorticity_spatial_progress.json",
    )
    stiffness, stiffness_audit = vector.assemble_stiffness(
        vorticity_spatial,
        system / "candidate_stiffness_6144.npz",
        system / "candidate_stiffness_progress.json",
        spatial_blocks=8,
    )
    mixing, _eigenvalues, eigen_audit = generalized_modes_rank_revealing(
        mass,
        stiffness,
        system / "global_vector_modes_6144_to_2048.npz",
    )

    basis_state = write_retained_basis_chunks(gradient_spatial, mixing, OUTPUT)
    gram_path = system / "retained_velocity_gram_unweighted.npz"
    predicted_projection_error, retained_audit = assemble_retained_gram_and_projection(
        OUTPUT, target, gram_path
    )
    if predicted_projection_error >= 0.08316253125667572:
        raise RuntimeError(
            "evaluated global K=2048 projection did not improve on validated K=1024: "
            f"{predicted_projection_error:.6f}"
        )

    state = {
        "complete": True,
        "method": "strict matched FD-Lap global vector Rayleigh-Ritz",
        "source_scalar_shapes": SCALAR_SHAPES,
        "tensor_polarizations_per_shape": POLARIZATIONS,
        "trial_vector_candidates": CANDIDATES,
        "global_ritz_retained_modes": K,
        "independent_coefficients": K,
        "local_polarization_collapse": False,
        "grid": [GRID] * 3,
        "fluid_nodes": FLUID_NODES,
        "candidate_mass_audit": mass_audit,
        "candidate_stiffness_audit": stiffness_audit,
        "global_eigen_audit": eigen_audit,
        "evaluated_retained_basis_audit": retained_audit,
        "predicted_fem_gt_velocity_relative_l2": predicted_projection_error,
        "latest_fem_gt_velocity_npy_sha256": sha256(gt_velocity_path),
        "basis": basis_state,
        "gram": str(gram_path.resolve()),
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_json(OUTPUT / "completion.json", state)
    atomic_json(OUTPUT / "protocol.json", {**state, "status": "complete"})
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
