"""Thirty-frame matched-GT DNS and five-method K=1024 temporal rollout.

The reduced update follows main.pdf, Sec. 3.7 and Appendix D: exact modal
viscous half steps surround an implicit-trapezoidal vorticity-Galerkin step,
whose nonlinear solve uses Newton-GMRES.  The DNS and all reported errors use
the complete 128^3 grid.  The coefficient contraction uses the paper-matched
32,768-point common volume quadrature.  Every non-overlapping stage is timed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gt30_dewitt2012_five_method_k1024_full128"
GT_RESULT = RESULT
GT_SOURCE = ROOT / "results" / "boundary_compatible_double_zero_gt_128_v2"
PREPARED = (
    ROOT
    / "results"
    / "eigenfluid_transfer_variable_geometry_potential_rayleigh_k1024_p9192_e2000"
    / "heldout_gt_prepared_canonical"
)
DNS_SCRIPT = ROOT / "scripts" / "generate_independent_dns_gt_128.py"
UNSEEN_ROOT = ROOT / "results" / "unseen10_five_method_k1024_full128"

GRID = 128
K = 1024
FLUID_NODES = 1_599_754
FRAMES = 30
FRAME_DT = 0.01
SUBSTEPS_PER_FRAME = 10
DT = FRAME_DT / SUBSTEPS_PER_FRAME
VISCOSITY = 2.5e-4
PENALTY_TIME_SCALE = 1.0e-3
PENALTY_SMOOTHING_CELLS = 0.75
QUADRATURE_POINTS = 32_768
QUADRATURE_SEED = 20_260_831
WEIGHT = (FLUID_NODES / GRID**3) / QUADRATURE_POINTS

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


METHODS: OrderedDict[str, dict[str, Any]] = OrderedDict(
    [
        (
            "strict_fd_lap_rr",
            {
                "label": "Strict FD-Lap RR",
                "color": "#B40426",
                "basis": ROOT
                / "results"
                / "fd_lap_strict_matched_e_phiD_A1024_vector_k1024_full128"
                / "unseen10_basis_chunks",
                "gram": UNSEEN_ROOT / "systems" / "strict_fd_lap_rr" / "gram.npz",
            },
        ),
        (
            "potential_rayleigh",
            {
                "label": "Potential-Rayleigh",
                "color": "#3B4CC0",
                "basis": ROOT
                / "results"
                / "eigenfluid_transfer_variable_geometry_potential_rayleigh_k1024_p9192_e2000"
                / "boundary_gt_full128_reconstruction"
                / "basis_chunks",
                "gram": UNSEEN_ROOT / "systems" / "potential_rayleigh" / "gram.npz",
            },
        ),
        (
            "velocity_rayleigh",
            {
                "label": "Velocity-Rayleigh",
                "color": "#762A83",
                "basis": ROOT
                / "results"
                / "eigenfluid_transfer_variable_geometry_velocity_rayleigh_k1024_p9192_e2000"
                / "boundary_gt_full128_reconstruction"
                / "basis_chunks",
                "gram": UNSEEN_ROOT / "systems" / "velocity_rayleigh" / "gram.npz",
            },
        ),
        (
            "vorticity_rayleigh",
            {
                "label": "Vorticity-Rayleigh",
                "color": "#168B98",
                "basis": ROOT
                / "results"
                / "eigenfluid_transfer_variable_geometry_vorticity_rayleigh_k1024_p9192_e2000"
                / "boundary_gt_full128_reconstruction"
                / "basis_chunks",
                "gram": UNSEEN_ROOT / "systems" / "vorticity_rayleigh" / "gram.npz",
            },
        ),
        (
            "vector_hodge_lap_rr",
            {
                "label": "Vector/Hodge-Lap RR",
                "color": "#D95F02",
                "basis": UNSEEN_ROOT / "basis" / "vector_lap",
                "gram": UNSEEN_ROOT / "systems" / "vector_hodge_lap_rr" / "gram.npz",
            },
        ),
    ]
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def synchronize() -> None:
    cp.cuda.Stream.null.synchronize()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    synchronize()
    started = time.perf_counter()
    value = call()
    synchronize()
    return value, time.perf_counter() - started


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dns_base = load_module("matched_gt_dns_base", DNS_SCRIPT)


def fluid_indices() -> np.ndarray:
    values = np.fromfile(PREPARED / "fluid_indices.u32", dtype="<u4")
    if values.shape != (FLUID_NODES,) or np.any(values[1:] <= values[:-1]):
        raise RuntimeError("canonical fluid indices are missing, unsorted, or incomplete")
    return values.astype(np.int64)


def fluid_mask(indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(GRID**3, dtype=bool)
    mask[indices] = True
    return mask.reshape(GRID, GRID, GRID)


def common_fd_curl_torch(velocity: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Central periodic curl of a saved z,y,x,xyz field, masked after curl."""
    factor = 0.5 * GRID
    derivative = lambda field, axis: (
        torch.roll(field, -1, dims=axis) - torch.roll(field, 1, dims=axis)
    ) * factor
    omega = torch.stack(
        (
            derivative(velocity[..., 2], 1) - derivative(velocity[..., 1], 0),
            derivative(velocity[..., 0], 0) - derivative(velocity[..., 2], 2),
            derivative(velocity[..., 1], 2) - derivative(velocity[..., 0], 1),
        ),
        dim=-1,
    )
    return omega * mask[..., None]


def common_fd_curl_cupy(velocity: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
    factor = 0.5 * GRID
    derivative = lambda field, axis: (
        cp.roll(field, -1, axis=axis) - cp.roll(field, 1, axis=axis)
    ) * factor
    omega = cp.stack(
        (
            derivative(velocity[..., 2], 1) - derivative(velocity[..., 1], 0),
            derivative(velocity[..., 0], 0) - derivative(velocity[..., 2], 2),
            derivative(velocity[..., 1], 2) - derivative(velocity[..., 0], 1),
        ),
        axis=-1,
    )
    return omega * mask[..., None]


def run_ground_truth(output: Path, indices: np.ndarray, mask_np: np.ndarray) -> dict[str, Any]:
    invocation_wall_started = time.perf_counter()
    completion_path = output / "completion.json"
    if completion_path.exists():
        state = json.loads(completion_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            log("reuse complete matched-GT 30-frame DNS")
            return state

    output.mkdir(parents=True, exist_ok=True)
    protocol = dns_base.DNSConfig(
        resolution=GRID,
        frames=FRAMES,
        fps=10,
        frame_dt=FRAME_DT,
        substeps_per_frame=SUBSTEPS_PER_FRAME,
        viscosity=VISCOSITY,
        penalty_time_scale=PENALTY_TIME_SCALE,
        penalty_smoothing_cells=PENALTY_SMOOTHING_CELLS,
        geometry_token=12_000_001,
        initial_rms_speed=0.32,
        warmup_substeps=0,
        seed=20_260_828,
        isosurface_low_quantile=0.90,
        isosurface_high_quantile=0.975,
    )
    solid_started = time.perf_counter()
    smooth_fluid = gaussian_filter(
        mask_np.astype(np.float32), sigma=PENALTY_SMOOTHING_CELLS, mode="wrap"
    )
    solid_fraction = np.clip(1.0 - smooth_fluid, 0.0, 1.0).astype(np.float32)
    solid_seconds = time.perf_counter() - solid_started
    np.save(output / "fluid_mask.npy", mask_np)
    np.save(output / "solid_fraction.npy", solid_fraction)
    np.asarray(indices, dtype="<u4").tofile(output / "fluid_indices.u32")

    if not torch.cuda.is_available():
        raise RuntimeError("formal 128^3 DNS requires CUDA")
    device = torch.device("cuda")
    solver = dns_base.PseudoSpectralDNS(protocol, solid_fraction, mask_np, device)
    mask_gpu = torch.as_tensor(mask_np, device=device)
    checkpoint_path = output / "checkpoints" / "latest.pt"
    frame_rows: list[dict[str, Any]] = []
    active_seconds = solid_seconds

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("protocol") != protocol.__dict__:
            raise ValueError("GT checkpoint protocol mismatch")
        velocity_hat = checkpoint["velocity_hat"].to(device)
        next_frame = int(checkpoint["next_frame"])
        frame_rows = list(checkpoint["frame_rows"])
        active_seconds = float(checkpoint["active_seconds"])
        initial_projection_error = float(checkpoint["initial_projection_error"])
        log(f"resume matched-GT DNS at frame {next_frame}/{FRAMES}")
    else:
        initial, load_seconds = timed(lambda: np.load(GT_SOURCE / "gt_velocity_128.npy"))
        active_seconds += load_seconds
        initial_tensor = torch.as_tensor(initial, device=device).permute(3, 0, 1, 2)

        def project_initial():
            initial_hat = solver.operators.fft(initial_tensor)
            projected_hat = solver.operators.project(initial_hat)
            projected = solver.operators.ifft_real(projected_hat)
            error = float(
                (torch.linalg.vector_norm(projected - initial_tensor)
                 / torch.linalg.vector_norm(initial_tensor).clamp_min(1.0e-30)).item()
            )
            return projected_hat, error

        (velocity_hat, initial_projection_error), project_seconds = timed(project_initial)
        active_seconds += project_seconds
        next_frame = 0

    while next_frame < FRAMES:
        row: dict[str, Any] = {
            "frame": next_frame,
            "physical_time": next_frame * FRAME_DT,
            "substep_seconds": 0.0,
        }
        if next_frame > 0:
            def advance_frame():
                nonlocal velocity_hat
                for _ in range(SUBSTEPS_PER_FRAME):
                    velocity_hat = solver.step(velocity_hat)

            _, row["substep_seconds"] = timed(advance_frame)

        def decode_fields():
            velocity = solver.operators.ifft_real(velocity_hat).permute(1, 2, 3, 0)
            saved = torch.where(mask_gpu[..., None], velocity, torch.zeros_like(velocity))
            omega = common_fd_curl_torch(saved, mask_gpu)
            return saved, omega

        (velocity_gpu, omega_gpu), row["field_and_common_curl_seconds"] = timed(decode_fields)

        def diagnostics(
            velocity_current: torch.Tensor = velocity_gpu,
            omega_current: torch.Tensor = omega_gpu,
        ):
            fluid_velocity = velocity_current[mask_gpu]
            fluid_omega = omega_current[mask_gpu]
            speed = torch.linalg.vector_norm(fluid_velocity, dim=1)
            magnitude = torch.linalg.vector_norm(fluid_omega, dim=1)
            return {
                "kinetic_energy": float((0.5 * torch.mean(speed.square())).item()),
                "enstrophy_common_fd": float((0.5 * torch.mean(magnitude.square())).item()),
                "rms_speed": float(torch.sqrt(torch.mean(speed.square())).item()),
                "max_speed": float(torch.max(speed).item()),
                "max_vorticity_common_fd": float(torch.max(magnitude).item()),
                "cfl": float(torch.max(speed).item() * DT * GRID),
                "spectral_divergence_relative_l2": solver.operators.divergence_relative_l2(
                    velocity_hat
                ),
            }

        diag, row["diagnostics_seconds"] = timed(diagnostics)
        row.update(diag)

        def save_fields(
            velocity_current: torch.Tensor = velocity_gpu,
            omega_current: torch.Tensor = omega_gpu,
            frame: int = next_frame,
        ):
            velocity_cpu = velocity_current.cpu().numpy().astype(np.float32, copy=False)
            omega_cpu = omega_current.cpu().numpy().astype(np.float32, copy=False)
            (output / "velocity").mkdir(exist_ok=True)
            (output / "vorticity_common_fd").mkdir(exist_ok=True)
            np.save(output / "velocity" / f"frame_{frame:03d}.npy", velocity_cpu)
            np.save(output / "vorticity_common_fd" / f"frame_{frame:03d}.npy", omega_cpu)

        _, row["serialization_seconds"] = timed(save_fields)
        row["active_frame_seconds"] = float(
            row["substep_seconds"]
            + row["field_and_common_curl_seconds"]
            + row["diagnostics_seconds"]
            + row["serialization_seconds"]
        )
        active_seconds += row["active_frame_seconds"]
        frame_rows.append(row)
        next_frame += 1
        atomic_torch_save(
            checkpoint_path,
            {
                "protocol": protocol.__dict__,
                "velocity_hat": velocity_hat.detach().cpu(),
                "next_frame": next_frame,
                "frame_rows": frame_rows,
                "active_seconds": active_seconds,
                "initial_projection_error": initial_projection_error,
            },
        )
        write_csv(output / "frame_timing_and_diagnostics.csv", frame_rows)
        log(
            f"matched-GT frame {next_frame:02d}/{FRAMES} "
            f"active={row['active_frame_seconds']:.3f}s CFL={row['cfl']:.4f}"
        )
        del velocity_gpu, omega_gpu

    state = {
        "complete": True,
        "kind": "matched canonical double-zero GT advanced by independent 128^3 DNS",
        "initial_source": str((GT_SOURCE / "gt_velocity_128.npy").resolve()),
        "initial_spectral_projection_relative_l2": initial_projection_error,
        "grid": [GRID, GRID, GRID],
        "fluid_nodes": FLUID_NODES,
        "frames": FRAMES,
        "saved_frame_indices": [0, FRAMES - 1],
        "frame_dt": FRAME_DT,
        "substeps_per_frame": SUBSTEPS_PER_FRAME,
        "internal_dt": DT,
        "viscosity": VISCOSITY,
        "solver": {
            "equations": "3D incompressible Navier-Stokes",
            "spatial": "Fourier pseudo-spectral, 2/3 de-aliasing",
            "time": "RK4 advection with exact viscous half steps",
            "boundary": "Strang-split smooth Brinkman penalization",
            "pressure": "Fourier Leray projection at every stage",
        },
        "maximum_cfl": float(max(row["cfl"] for row in frame_rows)),
        "maximum_spectral_divergence_relative_l2": float(
            max(row["spectral_divergence_relative_l2"] for row in frame_rows)
        ),
        "timing_seconds": {
            "solid_fraction_setup": solid_seconds,
            "active_total_including_serialization": active_seconds,
            "sum_substeps": float(sum(row["substep_seconds"] for row in frame_rows)),
            "sum_field_and_common_curl": float(
                sum(row["field_and_common_curl_seconds"] for row in frame_rows)
            ),
            "sum_diagnostics": float(sum(row["diagnostics_seconds"] for row in frame_rows)),
            "sum_serialization": float(sum(row["serialization_seconds"] for row in frame_rows)),
            "observed_wall_seconds_this_invocation_including_checkpoints": float(
                time.perf_counter() - invocation_wall_started
            ),
        },
    }
    atomic_json(completion_path, state)
    return state


def basis_files(method: str) -> list[tuple[Path, int]]:
    root = Path(METHODS[method]["basis"])
    files = [root] if root.is_file() else sorted(root.glob("*velocity*.f32"))
    answer: list[tuple[Path, int]] = []
    total_rows = 0
    for path in files:
        match = re.search(r"rows_(\d+)_velocity", path.name)
        if path.suffix.lower() == ".npy":
            shape = np.load(path, mmap_mode="r").shape
            if shape[0] != K:
                raise RuntimeError(f"bad NPY basis mode count: {path} shape={shape}")
            rows = int(np.prod(shape[1:]))
        else:
            rows = int(match.group(1)) if match else path.stat().st_size // (4 * K)
        if rows % 3 or (path.suffix.lower() != ".npy" and path.stat().st_size != rows * K * 4):
            raise RuntimeError(f"bad basis chunk layout: {path}")
        answer.append((path, rows))
        total_rows += rows
    if not answer or total_rows != FLUID_NODES * 3:
        raise RuntimeError(
            f"{method} basis covers {total_rows // 3} nodes, expected {FLUID_NODES}"
        )
    return answer


def open_basis(path: Path, rows: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, mmap_mode="r")
        if values.shape[0] != K or int(np.prod(values.shape[1:])) != rows:
            raise RuntimeError(f"unexpected NPY basis shape: {path} {values.shape}")
        return values.reshape(K, rows)
    return np.memmap(path, mode="r", dtype="<f4", shape=(K, rows))


def prepare_quadrature(output: Path, indices: np.ndarray) -> dict[str, Any]:
    completion = output / "completion.json"
    if completion.exists():
        state = json.loads(completion.read_text(encoding="utf-8"))
        if state.get("complete"):
            return state
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(QUADRATURE_SEED))
    positions = np.sort(
        rng.choice(FLUID_NODES, size=QUADRATURE_POINTS, replace=False).astype(np.int64)
    )
    globals_ = indices[positions]
    inverse = np.full(GRID**3, -1, dtype=np.int64)
    inverse[indices] = np.arange(FLUID_NODES, dtype=np.int64)
    z, remainder = np.divmod(globals_, GRID * GRID)
    y, x = np.divmod(remainder, GRID)

    def global_index(zz: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
        return ((zz % GRID) * GRID + (yy % GRID)) * GRID + (xx % GRID)

    neighbors_global = np.stack(
        (
            global_index(z, y, x + 1),
            global_index(z, y, x - 1),
            global_index(z, y + 1, x),
            global_index(z, y - 1, x),
            global_index(z + 1, y, x),
            global_index(z - 1, y, x),
        )
    )
    neighbors = inverse[neighbors_global]
    np.save(output / "fluid_positions.npy", positions)
    np.save(output / "global_cell_indices.npy", globals_)
    np.save(output / "neighbor_fluid_positions_xp_xm_yp_ym_zp_zm.npy", neighbors)
    state = {
        "complete": True,
        "kind": "common paper-matched volume quadrature",
        "sampler": "numpy PCG64 without replacement from all canonical fluid cells",
        "seed": QUADRATURE_SEED,
        "points": QUADRATURE_POINTS,
        "grid": [GRID, GRID, GRID],
        "fluid_nodes": FLUID_NODES,
        "weight_per_point": WEIGHT,
        "boundary_neighbor_values": "zero outside the fluid mask; periodic at cube lattice edges",
        "global_indices_sha256": hashlib.sha256(globals_.astype("<u4").tobytes()).hexdigest(),
        "setup_seconds": time.perf_counter() - started,
    }
    atomic_json(completion, state)
    return state


def selected_positions() -> tuple[np.ndarray, np.ndarray]:
    root = RESULT / "quadrature"
    return (
        np.load(root / "fluid_positions.npy"),
        np.load(root / "neighbor_fluid_positions_xp_xm_yp_ym_zp_zm.npy"),
    )


def extract_quadrature_basis(
    method: str,
    files: list[tuple[Path, int]],
    positions: np.ndarray,
    neighbors: np.ndarray,
) -> dict[str, Any]:
    output = RESULT / "methods" / method / "quadrature"
    completion_path = output / "completion.json"
    velocity_path = output / "velocity_raw_k_q_3.npy"
    vorticity_path = output / "vorticity_common_fd_raw_k_q_3.npy"
    if completion_path.exists():
        state = json.loads(completion_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            log(f"reuse complete {method} quadrature basis")
            return state
    output.mkdir(parents=True, exist_ok=True)
    arrays_existed = velocity_path.exists() and vorticity_path.exists()
    if arrays_existed:
        velocity = np.load(velocity_path, mmap_mode="r+")
        vorticity = np.load(vorticity_path, mmap_mode="r+")
    else:
        velocity = np.lib.format.open_memmap(
            velocity_path, mode="w+", dtype=np.float32, shape=(K, QUADRATURE_POINTS, 3)
        )
        vorticity = np.lib.format.open_memmap(
            vorticity_path, mode="w+", dtype=np.float32, shape=(K, QUADRATURE_POINTS, 3)
        )
        velocity[:] = 0.0
        vorticity[:] = 0.0
        velocity.flush()
        vorticity.flush()
    progress_path = output / "progress.json"
    # An interrupted file can leave only part of its curl contribution in the
    # output memmap.  Rebuilding this inexpensive extraction from file zero is
    # safer than trying to infer which += writes reached disk.  Completed
    # extractions are returned above and are never rebuilt.
    next_file = 0
    offset_rows = 0
    elapsed = 0.0
    if arrays_existed or progress_path.exists():
        velocity[:] = 0.0
        vorticity[:] = 0.0
        velocity.flush()
        vorticity.flush()
        progress_path.unlink(missing_ok=True)

    # Each request is (fluid-position array, source component, curl component, sign).
    xp, xm, yp, ym, zp, zm = neighbors
    factor = 0.5 * GRID
    curl_requests = [
        (yp, 2, 0, +factor),
        (ym, 2, 0, -factor),
        (zp, 1, 0, -factor),
        (zm, 1, 0, +factor),
        (zp, 0, 1, +factor),
        (zm, 0, 1, -factor),
        (xp, 2, 1, -factor),
        (xm, 2, 1, +factor),
        (xp, 1, 2, +factor),
        (xm, 1, 2, -factor),
        (yp, 0, 2, -factor),
        (ym, 0, 2, +factor),
    ]

    for file_index, (path, rows) in enumerate(files):
        if file_index < next_file:
            continue
        started = time.perf_counter()
        point_count = rows // 3
        start = offset_rows // 3
        stop = start + point_count
        source = open_basis(path, rows)
        velocity_selected = np.flatnonzero((positions >= start) & (positions < stop))
        velocity_local = positions[velocity_selected] - start
        request_specs: list[tuple[np.ndarray, np.ndarray, int, int, float]] = []
        for request_positions, source_component, curl_component, sign in curl_requests:
            request_selected = np.flatnonzero(
                (request_positions >= start) & (request_positions < stop)
            )
            if request_selected.size:
                request_local = request_positions[request_selected] - start
                request_specs.append(
                    (
                        request_selected,
                        request_local,
                        source_component,
                        curl_component,
                        sign,
                    )
                )

        # The source cache is mode-major.  Fancy indexing source[:, columns]
        # once per derivative request causes twelve random scans of a 3.2 GiB
        # file on HDD storage.  Copying a modest contiguous mode block once and
        # doing all indexing in RAM turns that into one sequential scan while
        # preserving the exact same FP32 values and FD curl operator.
        io_mode_block = 32
        for mode_start in range(0, K, io_mode_block):
            mode_stop = min(K, mode_start + io_mode_block)
            block = np.array(
                source[mode_start:mode_stop], dtype=np.float32, copy=True, order="C"
            )
            for component in range(3):
                columns = velocity_local * 3 + component
                velocity[mode_start:mode_stop, velocity_selected, component] = block[
                    :, columns
                ]
            for (
                request_selected,
                request_local,
                source_component,
                curl_component,
                sign,
            ) in request_specs:
                columns = request_local * 3 + source_component
                vorticity[
                    mode_start:mode_stop, request_selected, curl_component
                ] += sign * block[:, columns]
        velocity.flush()
        vorticity.flush()
        elapsed += time.perf_counter() - started
        offset_rows += rows
        atomic_json(
            progress_path,
            {
                "next_file": file_index + 1,
                "files": len(files),
                "offset_rows": offset_rows,
                "active_seconds": elapsed,
            },
        )
        log(f"{method} quadrature basis file {file_index + 1}/{len(files)}")

    state = {
        "complete": True,
        "method": method,
        "modes": K,
        "quadrature_points": QUADRATURE_POINTS,
        "velocity": str(velocity_path.resolve()),
        "vorticity": str(vorticity_path.resolve()),
        "vorticity_operator": "common 128^3 second-order central FD curl with zero solid values",
        "source_files": len(files),
        "source_basis_bytes": int(sum(path.stat().st_size for path, _ in files)),
        "active_extraction_seconds": elapsed,
    }
    atomic_json(completion_path, state)
    return state


def initial_projection(
    method: str,
    files: list[tuple[Path, int]],
    gt_frame0_fluid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    output = RESULT / "methods" / method / "initial_projection.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as z:
            q = np.asarray(z["q_raw"], dtype=np.float64)
            metrics = json.loads(str(z["metrics_json"].item()))
        return q, metrics
    gram_started = time.perf_counter()
    gram = np.asarray(np.load(METHODS[method]["gram"])["gram"], dtype=np.float64)
    gram_load_seconds = time.perf_counter() - gram_started
    rhs = np.zeros(K, dtype=np.float64)
    offset = 0
    rhs_started = time.perf_counter()
    for index, (path, rows) in enumerate(files):
        basis = cp.asarray(np.asarray(open_basis(path, rows)))
        target = cp.asarray(gt_frame0_fluid.reshape(-1)[offset : offset + rows])
        rhs += cp.asnumpy(basis @ target).astype(np.float64)
        del basis, target
        cp.get_default_memory_pool().free_all_blocks()
        offset += rows
        log(f"{method} initial RHS file {index + 1}/{len(files)}")
    synchronize()
    rhs_seconds = time.perf_counter() - rhs_started
    solve_started = time.perf_counter()
    factor = sla.cho_factor(0.5 * (gram + gram.T), lower=True, check_finite=False)
    q = sla.cho_solve(factor, rhs, check_finite=False)
    solve_seconds = time.perf_counter() - solve_started
    residual = float(np.linalg.norm(gram @ q - rhs) / max(np.linalg.norm(rhs), 1.0e-30))
    metrics = {
        "gram_cache_load_seconds": gram_load_seconds,
        "full128_rhs_seconds": rhs_seconds,
        "unregularized_cholesky_solve_seconds": solve_seconds,
        "relative_normal_residual": residual,
        "all_fluid_nodes": FLUID_NODES,
        "regularization": 0.0,
    }
    atomic_npz(output, q_raw=q, rhs=rhs, metrics_json=np.asarray(json.dumps(metrics)))
    return q, metrics


def generalized_ritz(method: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    root = RESULT / "methods" / method
    output = root / "generalized_ritz.npz"
    if output.exists():
        with np.load(output, allow_pickle=False) as z:
            return (
                np.asarray(z["transform"], dtype=np.float64),
                np.asarray(z["eigenvalues"], dtype=np.float64),
                json.loads(str(z["metrics_json"].item())),
            )
    b = np.load(root / "quadrature" / "velocity_raw_k_q_3.npy", mmap_mode="r")
    w = np.load(root / "quadrature" / "vorticity_common_fd_raw_k_q_3.npy", mmap_mode="r")
    b_gpu = cp.asarray(np.asarray(b).reshape(K, -1))
    w_gpu = cp.asarray(np.asarray(w).reshape(K, -1))
    synchronize()
    assembly_started = time.perf_counter()
    mass = cp.asnumpy(WEIGHT * (b_gpu @ b_gpu.T)).astype(np.float64)
    stiffness = cp.asnumpy(WEIGHT * (w_gpu @ w_gpu.T)).astype(np.float64)
    synchronize()
    assembly_seconds = time.perf_counter() - assembly_started
    del b_gpu, w_gpu
    cp.get_default_memory_pool().free_all_blocks()
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    solve_started = time.perf_counter()
    eigenvalues, transform = sla.eigh(
        stiffness, mass, check_finite=False, driver="gvd"
    )
    solve_seconds = time.perf_counter() - solve_started
    gram = transform.T @ mass @ transform
    residual = stiffness @ transform - mass @ (transform * eigenvalues[None, :])
    residual_relative = float(
        np.linalg.norm(residual) / max(np.linalg.norm(stiffness @ transform), 1.0e-30)
    )
    mass_eigs = sla.eigvalsh(mass, check_finite=False, driver="evd")
    metrics = {
        "operator_assembly_seconds": assembly_seconds,
        "generalized_eigensolve_seconds": solve_seconds,
        "mass_condition_number": float(mass_eigs[-1] / mass_eigs[0]),
        "mass_eigenvalue_min": float(mass_eigs[0]),
        "mass_eigenvalue_max": float(mass_eigs[-1]),
        "ritz_lambda_min": float(eigenvalues[0]),
        "ritz_lambda_max": float(eigenvalues[-1]),
        "mass_orthogonality_max_abs": float(np.max(np.abs(gram - np.eye(K)))),
        "generalized_eigen_relative_residual": residual_relative,
        "regularization": 0.0,
        "quadrature_points": QUADRATURE_POINTS,
    }
    atomic_npz(
        output,
        transform=transform,
        eigenvalues=eigenvalues,
        mass=mass,
        stiffness=stiffness,
        metrics_json=np.asarray(json.dumps(metrics)),
    )
    return transform, eigenvalues, metrics


def canonical_quadrature_gpu(
    method: str, transform: np.ndarray
) -> tuple[cp.ndarray, cp.ndarray, dict[str, float]]:
    root = RESULT / "methods" / method / "quadrature"
    b = np.load(root / "velocity_raw_k_q_3.npy", mmap_mode="r")
    w = np.load(root / "vorticity_common_fd_raw_k_q_3.npy", mmap_mode="r")
    raw_b = cp.asarray(np.asarray(b).reshape(K, -1))
    raw_w = cp.asarray(np.asarray(w).reshape(K, -1))
    transform_gpu = cp.asarray(transform.astype(np.float32))
    synchronize()
    started = time.perf_counter()
    canonical_b = transform_gpu.T @ raw_b
    canonical_w = transform_gpu.T @ raw_w
    synchronize()
    seconds = time.perf_counter() - started
    del raw_b, raw_w, transform_gpu
    cp.get_default_memory_pool().free_all_blocks()
    mass_check = cp.asnumpy(WEIGHT * (canonical_b @ canonical_b.T))
    orthogonality = float(np.max(np.abs(mass_check - np.eye(K, dtype=np.float32))))
    return canonical_b, canonical_w, {
        "canonical_basis_transform_seconds": seconds,
        "fp32_mass_orthogonality_max_abs": orthogonality,
    }


class GalerkinOperator:
    def __init__(self, velocity: cp.ndarray, vorticity: cp.ndarray) -> None:
        self.velocity = velocity
        self.vorticity = vorticity

    def nonlinear_with_fields(
        self, coefficients: cp.ndarray
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
        u = (self.velocity.T @ coefficients).reshape(-1, 3)
        omega = (self.vorticity.T @ coefficients).reshape(-1, 3)
        nonlinear = WEIGHT * (self.velocity @ cp.cross(u, omega).reshape(-1))
        return nonlinear, u, omega

    def nonlinear(self, coefficients: cp.ndarray) -> cp.ndarray:
        return self.nonlinear_with_fields(coefficients)[0]

    def jacobian_vector(
        self,
        vector: cp.ndarray,
        u: cp.ndarray,
        omega: cp.ndarray,
        half_dt: float,
    ) -> cp.ndarray:
        du = (self.velocity.T @ vector).reshape(-1, 3)
        domega = (self.vorticity.T @ vector).reshape(-1, 3)
        derivative = WEIGHT * (
            self.velocity
            @ (cp.cross(du, omega) + cp.cross(u, domega)).reshape(-1)
        )
        return vector - half_dt * derivative


def gmres_gpu(
    action: Callable[[cp.ndarray], cp.ndarray],
    right_hand_side: cp.ndarray,
    relative_tolerance: float = 2.0e-4,
    maximum_iterations: int = 12,
) -> tuple[cp.ndarray, int, float]:
    beta = float(cp.linalg.norm(right_hand_side).item())
    if beta == 0.0:
        return cp.zeros_like(right_hand_side), 0, 0.0
    vectors = cp.zeros((maximum_iterations + 1, K), dtype=cp.float32)
    vectors[0] = right_hand_side / beta
    hessenberg = np.zeros((maximum_iterations + 1, maximum_iterations), dtype=np.float64)
    target = np.zeros(maximum_iterations + 1, dtype=np.float64)
    target[0] = beta
    best = cp.zeros_like(right_hand_side)
    relative_residual = 1.0
    for column in range(maximum_iterations):
        candidate = action(vectors[column]).copy()
        # Two MGS passes keep the small Krylov basis stable in FP32.
        for _ in range(2):
            for row in range(column + 1):
                coefficient = float(cp.vdot(vectors[row], candidate).item())
                hessenberg[row, column] += coefficient
                candidate -= coefficient * vectors[row]
        norm = float(cp.linalg.norm(candidate).item())
        hessenberg[column + 1, column] = norm
        if norm > 1.0e-20:
            vectors[column + 1] = candidate / norm
        small = hessenberg[: column + 2, : column + 1]
        coefficients, *_ = np.linalg.lstsq(small, target[: column + 2], rcond=None)
        residual = target[: column + 2] - small @ coefficients
        relative_residual = float(np.linalg.norm(residual) / beta)
        best = cp.asarray(coefficients.astype(np.float32)) @ vectors[: column + 1]
        if relative_residual <= relative_tolerance or norm <= 1.0e-20:
            return best, column + 1, relative_residual
    return best, maximum_iterations, relative_residual


def implicit_trapezoidal_advection_step(
    operator: GalerkinOperator,
    initial: cp.ndarray,
    dt: float,
) -> tuple[cp.ndarray, dict[str, Any]]:
    half_dt = 0.5 * dt
    nonlinear_initial = operator.nonlinear(initial)
    solution = initial + dt * nonlinear_initial
    scale = max(1.0, float(cp.linalg.norm(initial).item()))
    tolerance = 2.0e-6 * scale
    total_gmres = 0
    final_residual = math.inf
    converged = False
    for newton_iteration in range(1, 7):
        nonlinear_solution, u, omega = operator.nonlinear_with_fields(solution)
        residual = solution - initial - half_dt * (
            nonlinear_initial + nonlinear_solution
        )
        residual_norm = float(cp.linalg.norm(residual).item())
        final_residual = residual_norm / scale
        if residual_norm <= tolerance:
            converged = True
            break

        def jacobian(
            vector: cp.ndarray,
            velocity_current: cp.ndarray = u,
            omega_current: cp.ndarray = omega,
        ) -> cp.ndarray:
            return operator.jacobian_vector(
                vector, velocity_current, omega_current, half_dt
            )

        delta, gmres_iterations, _linear_residual = gmres_gpu(
            jacobian, -residual, relative_tolerance=2.0e-4, maximum_iterations=12
        )
        total_gmres += gmres_iterations
        solution += delta
    if not converged:
        nonlinear_solution, _u, _omega = operator.nonlinear_with_fields(solution)
        residual = solution - initial - half_dt * (
            nonlinear_initial + nonlinear_solution
        )
        final_residual = float(cp.linalg.norm(residual).item()) / scale
        if final_residual > 1.0e-5:
            raise FloatingPointError(
                f"Newton did not converge: relative residual={final_residual:.3e}"
            )
    before = float(cp.vdot(initial, initial).real.item())
    after = float(cp.vdot(solution, solution).real.item())
    energy_drift = abs(after - before) / max(before, 1.0e-30)
    return solution, {
        "newton_iterations": newton_iteration,
        "gmres_iterations": total_gmres,
        "newton_relative_residual": final_residual,
        "inviscid_energy_relative_drift": energy_drift,
    }


def coefficient_rollout(
    method: str,
    canonical_b: cp.ndarray,
    canonical_w: cp.ndarray,
    eigenvalues: np.ndarray,
    initial_raw: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = RESULT / "methods" / method
    coefficient_path = root / "coefficients_canonical.npy"
    progress_path = root / "rollout_progress.json"
    if coefficient_path.exists():
        coefficients = np.load(coefficient_path, mmap_mode="r+")
    else:
        coefficients = np.lib.format.open_memmap(
            coefficient_path, mode="w+", dtype=np.float32, shape=(FRAMES, K)
        )
        coefficients[:] = 0.0
        coefficients.flush()
    next_frame = 1
    active_seconds = 0.0
    substep_rows: list[dict[str, Any]] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_frame = int(progress.get("next_frame", 1))
        active_seconds = float(progress.get("active_seconds", 0.0))
        substep_rows = list(progress.get("substep_rows", []))
    else:
        initial_canonical = sla.solve(
            transform, initial_raw, assume_a="gen", check_finite=False
        ).astype(np.float32)
        coefficients[0] = initial_canonical
        coefficients.flush()

    operator = GalerkinOperator(canonical_b, canonical_w)
    current = cp.asarray(np.asarray(coefficients[next_frame - 1]))
    lambdas = cp.asarray(eigenvalues.astype(np.float32))
    viscous_half = cp.exp(-0.5 * VISCOSITY * lambdas * DT)
    initial_nonlinear = operator.nonlinear(current)
    tangency = float(
        cp.abs(cp.vdot(current, initial_nonlinear)).item()
        / max(
            float(cp.linalg.norm(current).item())
            * float(cp.linalg.norm(initial_nonlinear).item()),
            1.0e-30,
        )
    )
    maximum_energy_drift = 0.0
    if substep_rows:
        maximum_energy_drift = max(
            float(row["inviscid_energy_relative_drift"]) for row in substep_rows
        )

    for frame in range(next_frame, FRAMES):
        for substep in range(SUBSTEPS_PER_FRAME):
            synchronize()
            started = time.perf_counter()
            current *= viscous_half
            current, diagnostics = implicit_trapezoidal_advection_step(
                operator, current, DT
            )
            current *= viscous_half
            synchronize()
            elapsed = time.perf_counter() - started
            active_seconds += elapsed
            maximum_energy_drift = max(
                maximum_energy_drift,
                float(diagnostics["inviscid_energy_relative_drift"]),
            )
            substep_rows.append(
                {
                    "method": method,
                    "saved_frame": frame,
                    "internal_substep": substep + 1,
                    "physical_time_end": (frame - 1) * FRAME_DT
                    + (substep + 1) * DT,
                    "seconds": elapsed,
                    **diagnostics,
                }
            )
        coefficients[frame] = cp.asnumpy(current)
        coefficients.flush()
        atomic_json(
            progress_path,
            {
                "complete": frame + 1 == FRAMES,
                "next_frame": frame + 1,
                "active_seconds": active_seconds,
                "substep_rows": substep_rows,
            },
        )
        write_csv(root / "internal_substep_timing.csv", substep_rows)
        log(f"{method} de Witt rollout frame {frame + 1:02d}/{FRAMES}")

    metrics = {
        "coefficient_rollout_seconds": active_seconds,
        "coefficient_seconds_per_saved_interval": active_seconds / (FRAMES - 1),
        "internal_substeps": (FRAMES - 1) * SUBSTEPS_PER_FRAME,
        "total_newton_iterations": int(
            sum(int(row["newton_iterations"]) for row in substep_rows)
        ),
        "total_gmres_iterations": int(
            sum(int(row["gmres_iterations"]) for row in substep_rows)
        ),
        "maximum_newton_relative_residual": float(
            max(float(row["newton_relative_residual"]) for row in substep_rows)
        ),
        "maximum_inviscid_energy_relative_drift": maximum_energy_drift,
        "initial_advection_tangency_defect": tangency,
        "integrator": "exact viscous half step + implicit trapezoidal advection + exact viscous half step",
        "nonlinear_solver": "Newton-GMRES",
        "external_force": 0.0,
    }
    return np.asarray(coefficients), metrics


def reconstruct_trajectory(
    method: str,
    files: list[tuple[Path, int]],
    canonical_coefficients: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.memmap, dict[str, Any]]:
    root = RESULT / "methods" / method
    output = root / "velocity_trajectory_fluid.npy"
    progress_path = root / "reconstruction_progress.json"
    if output.exists():
        trajectory = np.load(output, mmap_mode="r+")
    else:
        trajectory = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float32,
            shape=(FRAMES, FLUID_NODES, 3),
        )
        trajectory[:] = 0.0
        trajectory.flush()
    next_file = 0
    offset = 0
    elapsed = 0.0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_file = int(progress.get("next_file", 0))
        offset = int(progress.get("offset_points", 0))
        elapsed = float(progress.get("active_seconds", 0.0))
        if progress.get("complete"):
            return trajectory, {"full128_reconstruction_seconds": elapsed}
    raw_coefficients = np.asarray(
        canonical_coefficients.astype(np.float64) @ transform.T,
        dtype=np.float32,
    )
    coefficient_gpu = cp.asarray(raw_coefficients)
    for file_index, (path, rows) in enumerate(files):
        if file_index < next_file:
            continue
        synchronize()
        started = time.perf_counter()
        basis = cp.asarray(
            np.asarray(open_basis(path, rows))
        )
        values = coefficient_gpu @ basis
        synchronize()
        points = rows // 3
        trajectory[:, offset : offset + points] = cp.asnumpy(values).reshape(
            FRAMES, points, 3
        )
        trajectory.flush()
        del basis, values
        cp.get_default_memory_pool().free_all_blocks()
        elapsed += time.perf_counter() - started
        offset += points
        atomic_json(
            progress_path,
            {
                "complete": file_index + 1 == len(files),
                "next_file": file_index + 1,
                "offset_points": offset,
                "active_seconds": elapsed,
            },
        )
        log(f"{method} full128 reconstruction file {file_index + 1}/{len(files)}")
    return trajectory, {"full128_reconstruction_seconds": elapsed}


def trajectory_common_curl(
    method: str,
    velocity: np.memmap,
    indices: np.ndarray,
    mask_np: np.ndarray,
) -> tuple[np.memmap, dict[str, Any]]:
    root = RESULT / "methods" / method
    output = root / "vorticity_trajectory_fluid_common_fd.npy"
    progress_path = root / "curl_progress.json"
    if output.exists():
        vorticity = np.load(output, mmap_mode="r+")
    else:
        vorticity = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float32,
            shape=(FRAMES, FLUID_NODES, 3),
        )
        vorticity[:] = 0.0
        vorticity.flush()
    next_frame = 0
    elapsed = 0.0
    frame_seconds: list[float] = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        next_frame = int(progress.get("next_frame", 0))
        elapsed = float(progress.get("active_seconds", 0.0))
        frame_seconds = [float(value) for value in progress.get("frame_seconds", [])]
        if progress.get("complete"):
            return vorticity, {
                "common_fd_curl_seconds": elapsed,
                "curl_frame_seconds": frame_seconds,
            }
    mask_gpu = cp.asarray(mask_np)
    for frame in range(next_frame, FRAMES):
        synchronize()
        started = time.perf_counter()
        full = cp.zeros((GRID**3, 3), dtype=cp.float32)
        full[indices] = cp.asarray(np.asarray(velocity[frame]))
        full = full.reshape(GRID, GRID, GRID, 3)
        omega = common_fd_curl_cupy(full, mask_gpu)
        vorticity[frame] = cp.asnumpy(omega.reshape(-1, 3)[indices])
        vorticity.flush()
        del full, omega
        cp.get_default_memory_pool().free_all_blocks()
        synchronize()
        seconds = time.perf_counter() - started
        elapsed += seconds
        frame_seconds.append(seconds)
        atomic_json(
            progress_path,
            {
                "complete": frame + 1 == FRAMES,
                "next_frame": frame + 1,
                "active_seconds": elapsed,
                "frame_seconds": frame_seconds,
            },
        )
        log(f"{method} common FD curl frame {frame + 1:02d}/{FRAMES}")
    return vorticity, {
        "common_fd_curl_seconds": elapsed,
        "curl_frame_seconds": frame_seconds,
    }


def compute_method_metrics(
    method: str,
    velocity: np.memmap,
    vorticity: np.memmap,
    indices: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "methods" / method
    output = root / "trajectory_metrics.csv"
    timing_path = root / "metric_timing.json"
    if output.exists() and timing_path.exists():
        with output.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        rows = [
            {
                key: (int(value) if key == "frame" else float(value))
                for key, value in row.items()
            }
            for row in rows
        ]
        return rows, json.loads(timing_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for frame in range(FRAMES):
        target_u = np.load(
            GT_RESULT / "gt" / "velocity" / f"frame_{frame:03d}.npy", mmap_mode="r"
        ).reshape(-1, 3)[indices]
        target_w = np.load(
            GT_RESULT / "gt" / "vorticity_common_fd" / f"frame_{frame:03d}.npy",
            mmap_mode="r",
        ).reshape(-1, 3)[indices]
        predicted_u = np.asarray(velocity[frame], dtype=np.float64)
        predicted_w = np.asarray(vorticity[frame], dtype=np.float64)
        target_u64 = np.asarray(target_u, dtype=np.float64)
        target_w64 = np.asarray(target_w, dtype=np.float64)
        velocity_error = float(
            np.linalg.norm(predicted_u - target_u64)
            / max(np.linalg.norm(target_u64), 1.0e-30)
        )
        vorticity_error = float(
            np.linalg.norm(predicted_w - target_w64)
            / max(np.linalg.norm(target_w64), 1.0e-30)
        )
        rows.append(
            {
                "frame": frame,
                "physical_time": frame * FRAME_DT,
                "velocity_relative_l2": velocity_error,
                "vorticity_relative_l2_common_fd_curl": vorticity_error,
                "predicted_velocity_rms": float(
                    np.sqrt(np.mean(np.sum(predicted_u**2, axis=1)))
                ),
                "gt_velocity_rms": float(
                    np.sqrt(np.mean(np.sum(target_u64**2, axis=1)))
                ),
            }
        )
    seconds = time.perf_counter() - started
    write_csv(output, rows)
    timing = {"metric_computation_seconds": seconds}
    atomic_json(timing_path, timing)
    summary = {
        "frame0_velocity_relative_l2": rows[0]["velocity_relative_l2"],
        "frame0_vorticity_relative_l2_common_fd_curl": rows[0][
            "vorticity_relative_l2_common_fd_curl"
        ],
        "predicted_frames_mean_velocity_relative_l2": float(
            np.mean([row["velocity_relative_l2"] for row in rows[1:]])
        ),
        "predicted_frames_mean_vorticity_relative_l2_common_fd_curl": float(
            np.mean(
                [row["vorticity_relative_l2_common_fd_curl"] for row in rows[1:]]
            )
        ),
        "final_velocity_relative_l2": rows[-1]["velocity_relative_l2"],
        "final_vorticity_relative_l2_common_fd_curl": rows[-1][
            "vorticity_relative_l2_common_fd_curl"
        ],
    }
    atomic_json(root / "metric_summary.json", summary)
    return rows, timing


def full_field_from_fluid(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.zeros((GRID**3, 3), dtype=np.float32)
    result[indices] = np.asarray(values, dtype=np.float32)
    return result.reshape(GRID, GRID, GRID, 3)


def draw_cube(axis) -> None:
    for first in (0.0, 1.0):
        for second in (0.0, 1.0):
            for start, end in (
                ((0.0, first, second), (1.0, first, second)),
                ((first, 0.0, second), (first, 1.0, second)),
                ((first, second, 0.0), (first, second, 1.0)),
            ):
                axis.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color="#D7D7D7",
                    linewidth=0.5,
                    alpha=0.9,
                )


def draw_isosurface(axis, field: np.ndarray, level: float, color: str) -> None:
    magnitude = np.linalg.norm(field, axis=-1)
    if float(np.max(magnitude)) > level:
        vertices, faces, _normals, _values = marching_cubes(
            magnitude,
            level=level,
            spacing=(1.0 / (GRID - 1),) * 3,
            step_size=2,
            allow_degenerate=False,
        )
        surface = Poly3DCollection(vertices[faces], linewidths=0.0, alpha=0.97)
        surface.set_facecolor(color)
        axis.add_collection3d(surface)
    axis.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=21, azim=-56)
    draw_cube(axis)
    axis.set_axis_off()


def render_outputs(
    all_metrics: dict[str, list[dict[str, Any]]],
    indices: np.ndarray,
) -> dict[str, Any]:
    output_png = RESULT / "output" / "png"
    output_pdf = RESULT / "output" / "pdf"
    output_png.mkdir(parents=True, exist_ok=True)
    output_pdf.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.7,
        }
    )

    started = time.perf_counter()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.15), constrained_layout=True)
    frames = np.arange(FRAMES)
    for method, settings in METHODS.items():
        rows = all_metrics[method]
        axes[0].plot(
            frames,
            100.0 * np.asarray([row["velocity_relative_l2"] for row in rows]),
            color=settings["color"],
            linewidth=1.6,
            label=settings["label"],
        )
        axes[1].plot(
            frames,
            100.0
            * np.asarray(
                [row["vorticity_relative_l2_common_fd_curl"] for row in rows]
            ),
            color=settings["color"],
            linewidth=1.6,
            label=settings["label"],
        )
    for axis, title, ylabel in (
        (axes[0], "Velocity", "Relative L2 error (%)"),
        (axes[1], "Vorticity (common FD curl)", "Relative L2 error (%)"),
    ):
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Saved frame")
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#E4E4E4", linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.8, loc="best")
    figure.suptitle(
        "K=1024 temporal rollout: de Witt vorticity-Galerkin update",
        fontweight="bold",
        fontsize=10,
    )
    figure.savefig(output_png / "temporal_error_curves.png", dpi=300)
    figure.savefig(output_pdf / "temporal_error_curves.pdf")
    plt.close(figure)
    timings["temporal_error_plot_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    gt_final = np.load(
        RESULT / "gt" / "vorticity_common_fd" / f"frame_{FRAMES - 1:03d}.npy"
    )
    gt_magnitude = np.linalg.norm(gt_final.reshape(-1, 3)[indices], axis=1)
    level = float(np.quantile(gt_magnitude, 0.965))
    panels: list[tuple[str, str, np.ndarray]] = [
        ("Ground truth", "#303030", gt_final)
    ]
    for method, settings in METHODS.items():
        trajectory = np.load(
            RESULT / "methods" / method / "vorticity_trajectory_fluid_common_fd.npy",
            mmap_mode="r",
        )
        panels.append(
            (
                settings["label"],
                settings["color"],
                full_field_from_fluid(trajectory[-1], indices),
            )
        )
    figure = plt.figure(figsize=(13.0, 3.2), facecolor="white")
    figure.suptitle(
        f"Frame {FRAMES - 1}: common absolute isovorticity, K=1024",
        y=0.98,
        fontsize=11,
        fontweight="bold",
    )
    left, right, bottom, top, gap = 0.012, 0.988, 0.12, 0.88, 0.006
    width = (right - left - 5 * gap) / 6
    for index, (label, color, field) in enumerate(panels):
        x0 = left + index * (width + gap)
        axis = figure.add_axes([x0, bottom, width, top - bottom], projection="3d")
        draw_isosurface(axis, field, level, color)
        axis.set_title(label, fontsize=8.0, pad=0)
        axis.text2D(
            -0.01,
            0.98,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    figure.text(
        0.5,
        0.035,
        rf"Common $|\omega|={level:.4f}$ (GT q96.5); full $128^3$ reconstruction and common second-order FD curl.",
        ha="center",
        fontsize=7.2,
    )
    figure.savefig(
        output_png / "frame029_gt_plus_five_methods_isovorticity.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    figure.savefig(
        output_pdf / "frame029_gt_plus_five_methods_isovorticity.pdf",
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    timings["final_isosurface_plot_seconds"] = time.perf_counter() - started

    video_path = RESULT / "gt" / "gt30_isovorticity.mp4"
    frames_dir = RESULT / "gt" / "video_frames"
    started = time.perf_counter()
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(FRAMES):
        png = frames_dir / f"frame_{frame:03d}.png"
        if png.exists():
            continue
        field = np.load(
            RESULT / "gt" / "vorticity_common_fd" / f"frame_{frame:03d}.npy"
        )
        figure = plt.figure(figsize=(5.0, 5.0), facecolor="white")
        axis = figure.add_subplot(111, projection="3d")
        draw_isosurface(axis, field, level, "#3B4CC0")
        axis.set_title(
            f"Matched GT | frame {frame:02d}/{FRAMES - 1:02d} | t={frame * FRAME_DT:.3f}",
            fontsize=10,
            fontweight="bold",
        )
        figure.savefig(png, dpi=160, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
    timings["gt_video_frame_render_seconds"] = time.perf_counter() - started
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        started = time.perf_counter()
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "10",
                "-i",
                str(frames_dir / "frame_%03d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        timings["gt_video_encode_seconds"] = time.perf_counter() - started
    else:
        timings["gt_video_encode_seconds"] = 0.0
    atomic_json(RESULT / "render_timing.json", timings)
    return {
        "timing_seconds": timings,
        "common_isovorticity_level": level,
        "temporal_error_png": str((output_png / "temporal_error_curves.png").resolve()),
        "temporal_error_pdf": str((output_pdf / "temporal_error_curves.pdf").resolve()),
        "final_isosurface_png": str(
            (output_png / "frame029_gt_plus_five_methods_isovorticity.png").resolve()
        ),
        "final_isosurface_pdf": str(
            (output_pdf / "frame029_gt_plus_five_methods_isovorticity.pdf").resolve()
        ),
        "gt_video": str(video_path.resolve()) if video_path.exists() else None,
    }


def run_method(
    method: str,
    indices: np.ndarray,
    mask_np: np.ndarray,
    gt_frame0_fluid: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = RESULT / "methods" / method
    completion_path = root / "completion.json"
    if completion_path.exists():
        state = json.loads(completion_path.read_text(encoding="utf-8"))
        if state.get("complete"):
            with (root / "trajectory_metrics.csv").open(
                "r", newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            parsed = [
                {
                    key: (int(value) if key == "frame" else float(value))
                    for key, value in row.items()
                }
                for row in rows
            ]
            log(f"reuse complete {method} temporal evaluation")
            return parsed, state
    root.mkdir(parents=True, exist_ok=True)
    method_started = time.perf_counter()
    files, basis_audit_seconds = timed(lambda: basis_files(method))
    positions, neighbors = selected_positions()
    extraction = extract_quadrature_basis(method, files, positions, neighbors)
    initial_raw, projection = initial_projection(
        method, files, gt_frame0_fluid
    )
    transform, eigenvalues, ritz = generalized_ritz(method)
    canonical_b, canonical_w, canonical_timing = canonical_quadrature_gpu(
        method, transform
    )
    canonical_coefficients, rollout = coefficient_rollout(
        method,
        canonical_b,
        canonical_w,
        eigenvalues,
        initial_raw,
        transform,
    )
    del canonical_b, canonical_w
    cp.get_default_memory_pool().free_all_blocks()
    velocity, reconstruction = reconstruct_trajectory(
        method, files, canonical_coefficients, transform
    )
    vorticity, curl_timing = trajectory_common_curl(
        method, velocity, indices, mask_np
    )
    metric_rows, metric_timing = compute_method_metrics(
        method, velocity, vorticity, indices
    )
    summary = json.loads((root / "metric_summary.json").read_text(encoding="utf-8"))
    timing = {
        "basis_file_audit_seconds": basis_audit_seconds,
        "basis_cache_read_and_quadrature_extraction_seconds": extraction[
            "active_extraction_seconds"
        ],
        **projection,
        **{
            key: value
            for key, value in ritz.items()
            if key.endswith("_seconds")
        },
        **canonical_timing,
        **{
            key: value
            for key, value in rollout.items()
            if key.endswith("_seconds")
        },
        **reconstruction,
        "common_fd_curl_seconds": curl_timing["common_fd_curl_seconds"],
        **metric_timing,
    }
    nonoverlap_keys = [
        "basis_file_audit_seconds",
        "basis_cache_read_and_quadrature_extraction_seconds",
        "gram_cache_load_seconds",
        "full128_rhs_seconds",
        "unregularized_cholesky_solve_seconds",
        "operator_assembly_seconds",
        "generalized_eigensolve_seconds",
        "canonical_basis_transform_seconds",
        "coefficient_rollout_seconds",
        "full128_reconstruction_seconds",
        "common_fd_curl_seconds",
        "metric_computation_seconds",
    ]
    timing["active_total_seconds"] = float(sum(float(timing[key]) for key in nonoverlap_keys))
    state = {
        "complete": True,
        "method": method,
        "label": METHODS[method]["label"],
        "modes": K,
        "independent_coefficients": K,
        "one_q_per_complete_mode": True,
        "grid": [GRID, GRID, GRID],
        "fluid_nodes": FLUID_NODES,
        "quadrature_points": QUADRATURE_POINTS,
        "volume_quadrature_weight": WEIGHT,
        "vorticity_operator": "common 128^3 second-order central FD curl",
        "generalized_ritz": ritz,
        "rollout": rollout,
        "errors": summary,
        "timing_seconds": timing,
        "observed_wall_seconds_this_invocation": time.perf_counter() - method_started,
    }
    atomic_json(completion_path, state)
    return metric_rows, state


def assemble_summary(
    gt_state: dict[str, Any],
    method_states: dict[str, dict[str, Any]],
    all_metrics: dict[str, list[dict[str, Any]]],
    render: dict[str, Any],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    for method, state in method_states.items():
        errors = state["errors"]
        summary_rows.append(
            {
                "method": method,
                "label": state["label"],
                "modes": K,
                "dofs": K,
                "frame0_velocity_error_percent": 100.0
                * errors["frame0_velocity_relative_l2"],
                "frame0_vorticity_error_percent": 100.0
                * errors["frame0_vorticity_relative_l2_common_fd_curl"],
                "mean_frames1_29_velocity_error_percent": 100.0
                * errors["predicted_frames_mean_velocity_relative_l2"],
                "final_frame29_velocity_error_percent": 100.0
                * errors["final_velocity_relative_l2"],
                "mean_frames1_29_vorticity_error_percent": 100.0
                * errors[
                    "predicted_frames_mean_vorticity_relative_l2_common_fd_curl"
                ],
                "final_frame29_vorticity_error_percent": 100.0
                * errors["final_vorticity_relative_l2_common_fd_curl"],
                "coefficient_rollout_seconds": state["rollout"][
                    "coefficient_rollout_seconds"
                ],
                "complete_method_active_seconds": state["timing_seconds"][
                    "active_total_seconds"
                ],
            }
        )
        for stage, seconds in state["timing_seconds"].items():
            if stage.endswith("_seconds"):
                timing_rows.append(
                    {
                        "scope": method,
                        "stage": stage,
                        "seconds": float(seconds),
                    }
                )
        for row in all_metrics[method]:
            all_frame_rows.append({"method": method, **row})
    for stage, seconds in gt_state["timing_seconds"].items():
        timing_rows.append({"scope": "ground_truth", "stage": stage, "seconds": seconds})
    for stage, seconds in render["timing_seconds"].items():
        timing_rows.append({"scope": "render", "stage": stage, "seconds": seconds})
    write_csv(RESULT / "summary_metrics.csv", summary_rows)
    write_csv(RESULT / "all_frame_metrics.csv", all_frame_rows)
    write_csv(RESULT / "timing_all_stages.csv", timing_rows)
    atomic_json(RESULT / "summary_metrics.json", {"methods": summary_rows})

    report = [
        "# Matched 3D metaball: 30-frame de Witt rollout",
        "",
        "## Protocol",
        "",
        "- Ground truth: the canonical held-out double-zero initial field, advanced independently by a full 128^3 pseudo-spectral Navier-Stokes solver with smooth Brinkman walls.",
        f"- Exactly {FRAMES} saved frames (0-{FRAMES - 1}), output interval {FRAME_DT}, {SUBSTEPS_PER_FRAME} internal steps per interval, internal dt={DT}.",
        f"- Five reduced spaces: K={K} complete modes and exactly {K} external q coefficients each.",
        f"- Common coefficient quadrature: {QUADRATURE_POINTS} deterministic volume points, matching main.pdf Appendix D; full 128^3 reconstruction and error evaluation use all {FLUID_NODES:,} fluid nodes.",
        "- Vorticity-Galerkin equation: wdot_i = -nu lambda_i w_i + sum_jk C_ijk w_j w_k, C_ijk = integral omega_k dot (Phi_i cross Phi_j) dx; no external force for this closed decay case.",
        "- Integrator: exact viscous half step, Newton-GMRES implicit trapezoidal advection, exact viscous half step.",
        "",
        "## Temporal errors",
        "",
        "| Method | Frame-0 u (%) | Mean u, frames 1-29 (%) | Final u (%) | Frame-0 omega (%) | Mean omega, frames 1-29 (%) | Final omega (%) | Coefficient rollout (s) | Complete method active (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['label']} | {row['frame0_velocity_error_percent']:.3f} | "
            f"{row['mean_frames1_29_velocity_error_percent']:.3f} | "
            f"{row['final_frame29_velocity_error_percent']:.3f} | "
            f"{row['frame0_vorticity_error_percent']:.3f} | "
            f"{row['mean_frames1_29_vorticity_error_percent']:.3f} | "
            f"{row['final_frame29_vorticity_error_percent']:.3f} | "
            f"{row['coefficient_rollout_seconds']:.3f} | "
            f"{row['complete_method_active_seconds']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Timing",
            "",
            f"GT active time including 30-frame serialization: {gt_state['timing_seconds']['active_total_including_serialization']:.3f} s.",
            "Every internal coefficient step is listed in each method's `internal_substep_timing.csv`; all non-overlapping high-level stages are in `timing_all_stages.csv`.",
            "Previously completed basis training/eigensolves are provenance, not charged to this rollout. Loading, quadrature extraction, Ritz rotation, projection, rollout, full-grid reconstruction, curl, metrics, plotting, and video encoding are all timed here.",
            "",
            "## Outputs",
            "",
            "- `output/png/temporal_error_curves.png` and matching PDF.",
            "- `output/png/frame029_gt_plus_five_methods_isovorticity.png` and matching PDF.",
            "- `gt/gt30_isovorticity.mp4` (30 frames, when ffmpeg is available).",
        ]
    )
    (RESULT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    required = [
        RESULT / "summary_metrics.csv",
        RESULT / "all_frame_metrics.csv",
        RESULT / "timing_all_stages.csv",
        RESULT / "REPORT.md",
        RESULT / "output" / "png" / "temporal_error_curves.png",
        RESULT / "output" / "pdf" / "temporal_error_curves.pdf",
        RESULT / "output" / "png" / "frame029_gt_plus_five_methods_isovorticity.png",
        RESULT / "output" / "pdf" / "frame029_gt_plus_five_methods_isovorticity.pdf",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing final artifacts: {missing}")
    for method in METHODS:
        velocity = np.load(
            RESULT / "methods" / method / "velocity_trajectory_fluid.npy", mmap_mode="r"
        )
        vorticity = np.load(
            RESULT / "methods" / method / "vorticity_trajectory_fluid_common_fd.npy",
            mmap_mode="r",
        )
        if velocity.shape != (FRAMES, FLUID_NODES, 3) or vorticity.shape != velocity.shape:
            raise RuntimeError(f"bad trajectory shape for {method}")
    return {
        "summary_rows": summary_rows,
        "timing_rows": timing_rows,
        "required_artifacts": [str(path.resolve()) for path in required],
    }


def run(skip_render: bool = False, only_gt: bool = False) -> dict[str, Any]:
    RESULT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    indices = fluid_indices()
    mask_np = fluid_mask(indices)
    if int(np.count_nonzero(mask_np)) != FLUID_NODES:
        raise RuntimeError("canonical fluid mask count mismatch")
    input_manifest = {
        "gt_source_velocity": str((GT_SOURCE / "gt_velocity_128.npy").resolve()),
        "gt_source_velocity_sha256": sha256(GT_SOURCE / "gt_velocity_128.npy"),
        "canonical_geometry_sha256": sha256(PREPARED / "geometry_xyzr.f32"),
        "fluid_indices_sha256": sha256(PREPARED / "fluid_indices.u32"),
        "paper": str((ROOT / "publish" / "neural-eigenfluid-paper" / "main.pdf").resolve()),
        "paper_sha256": sha256(ROOT / "publish" / "neural-eigenfluid-paper" / "main.pdf"),
    }
    atomic_json(RESULT / "input_manifest.json", input_manifest)
    quadrature = prepare_quadrature(RESULT / "quadrature", indices)
    gt_state = run_ground_truth(RESULT / "gt", indices, mask_np)
    if only_gt:
        return {"complete": True, "ground_truth_only": True, "gt": gt_state}
    gt_frame0 = np.load(RESULT / "gt" / "velocity" / "frame_000.npy", mmap_mode="r")
    gt_frame0_fluid = np.asarray(gt_frame0.reshape(-1, 3)[indices], dtype=np.float32)
    all_metrics: dict[str, list[dict[str, Any]]] = {}
    method_states: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        rows, state = run_method(method, indices, mask_np, gt_frame0_fluid)
        all_metrics[method] = rows
        method_states[method] = state
    render = (
        render_outputs(all_metrics, indices)
        if not skip_render
        else {"timing_seconds": {}, "skipped": True}
    )
    if skip_render:
        return {
            "complete": False,
            "numerical_complete_render_pending": True,
            "ground_truth": gt_state,
            "methods": method_states,
        }
    summary = assemble_summary(gt_state, method_states, all_metrics, render)
    result = {
        "complete": True,
        "experiment": "matched GT 30-frame temporal evolution with five K=1024 methods",
        "grid": [GRID, GRID, GRID],
        "fluid_nodes": FLUID_NODES,
        "frames": FRAMES,
        "modes_per_method": K,
        "independent_coefficients_per_method": K,
        "paper_protocol": {
            "source": input_manifest["paper"],
            "equations": "main.pdf Eq. (15)-(16), following De Witt et al. (2012)",
            "volume_quadrature_points": QUADRATURE_POINTS,
            "integrator": "exact viscous half steps around Newton-GMRES implicit trapezoidal advection",
            "internal_steps_per_saved_interval": SUBSTEPS_PER_FRAME,
        },
        "ground_truth": gt_state,
        "quadrature": quadrature,
        "methods": method_states,
        "render": render,
        "summary": summary,
        "observed_wall_seconds_this_invocation": time.perf_counter() - started,
    }
    atomic_json(RESULT / "completion.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--only-gt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(skip_render=args.skip_render, only_gt=args.only_gt)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
