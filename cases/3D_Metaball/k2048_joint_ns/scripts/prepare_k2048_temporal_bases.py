"""Prepare full-grid K=2048 velocity bases and full-grid Gram caches.

The learned reconstructions are evaluated from the completed epoch-2000
checkpoints.  The strict FD-Lap cache is reused without recomputation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "gt30_dewitt2012_four_method_k2048_three_slices_full128"
K = 2048
FLUID_NODES = 1_599_754
CHUNK_POINTS = 262_144

METHODS = {
    "potential_rayleigh": {
        "run": ROOT / "results" / "eigenfluid_transfer_variable_geometry_potential_rayleigh_k2048_p9192_e2000",
        "exe": ROOT / "native_cuda_fp32" / "bin" / "Release" / "reconstruct_boundary_gt_k2048.exe",
    },
    "velocity_rayleigh": {
        "run": ROOT / "results" / "eigenfluid_transfer_variable_geometry_velocity_rayleigh_k2048_p9192_e2000",
        "exe": ROOT / "native_cuda_fp32" / "bin" / "Release" / "reconstruct_boundary_gt_velocity_rayleigh_k2048.exe",
    },
    "vorticity_rayleigh": {
        "run": ROOT / "results" / "eigenfluid_transfer_variable_geometry_vorticity_rayleigh_k2048_p9192_e2000",
        "exe": ROOT / "native_cuda_fp32" / "bin" / "Release" / "reconstruct_boundary_gt_vorticity_rayleigh_k2048.exe",
    },
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def validate_basis(root: Path) -> None:
    chunks = (FLUID_NODES + CHUNK_POINTS - 1) // CHUNK_POINTS
    for chunk in range(chunks):
        points = min(CHUNK_POINTS, FLUID_NODES - chunk * CHUNK_POINTS)
        expected = K * points * 3 * 4
        path = root / "basis_chunks" / f"chunk_{chunk:02d}_velocity.f32"
        if not path.exists() or path.stat().st_size != expected:
            raise RuntimeError(f"invalid K=2048 velocity basis chunk: {path}")


def wait_for_gpu_idle() -> None:
    """Do not launch reconstruction while another CUDA process owns the GPU."""
    while True:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        used_mib = int(query.stdout.splitlines()[0].strip())
        # This workstation's WDDM desktop baseline is about 440 MiB.  Unlike
        # query-compute-apps (which also lists desktop graphics contexts under
        # WDDM), total allocation cleanly separates idle from a training job.
        if used_mib < 2048:
            return
        print(f"GPU busy ({used_mib} MiB allocated); waiting 60 s", flush=True)
        time.sleep(60)


def prepare_learned(method: str, settings: dict) -> dict:
    run = Path(settings["run"])
    output = run / "boundary_gt_full128_reconstruction"
    prepared = run / "heldout_gt_prepared_canonical"
    checkpoint = run / "checkpoints" / "epoch_2000.bin"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prepare_variable_transfer_heldout_gt_full128.py"),
         "--run-root", str(run)],
        cwd=ROOT,
        check=True,
    )
    metrics_path = output / "metrics.json"
    metrics_ok = False
    if metrics_path.exists():
        try:
            state = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_ok = state.get("complete") and int(state.get("modes", 0)) == K
        except (OSError, ValueError, json.JSONDecodeError):
            metrics_ok = False
    if not metrics_ok or not (output / "full128_velocity_gram_f32.bin").exists():
        output.mkdir(parents=True, exist_ok=True)
        wait_for_gpu_idle()
        stdout = output / "k2048_reconstruction.stdout.log"
        stderr = output / "k2048_reconstruction.stderr.log"
        with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
            completed = subprocess.run(
                [str(settings["exe"]), str(checkpoint), str(prepared), str(output)],
                cwd=ROOT,
                stdout=out,
                stderr=err,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"{method} K=2048 basis reconstruction failed: {completed.returncode}")
    validate_basis(output)
    raw_gram = output / "full128_velocity_gram_f32.bin"
    values = np.fromfile(raw_gram, dtype="<f4")
    if values.size != K * K:
        raise RuntimeError(f"invalid Gram cache: {raw_gram}")
    gram = values.reshape(K, K, order="F").astype(np.float64)
    gram = 0.5 * (gram + gram.T)
    gram_path = RESULT / "systems" / method / "gram.npz"
    atomic_npz(gram_path, gram=gram)
    return {
        "basis": str((output / "basis_chunks").resolve()),
        "gram": str(gram_path.resolve()),
        "metrics": str(metrics_path.resolve()),
    }


def prepare_strict() -> dict:
    basis = (
        ROOT / "results" / "laplacian_strict_complete_A2048_basis_full128"
        / "complete_velocity_A2048_mode_major_128.npy"
    )
    values = np.load(basis, mmap_mode="r")
    if values.shape != (K, FLUID_NODES, 3) or values.dtype != np.float32:
        raise RuntimeError(f"invalid strict K=2048 basis: {values.shape}, {values.dtype}")
    # Split the 39 GB NPY into the same point-chunked, mode-major raw layout
    # used by the learned bases.  This keeps each GPU transfer below 6.5 GB
    # during projection and reconstruction.
    chunk_root = RESULT / "basis" / "strict_fd_lap_rr"
    chunk_dir = chunk_root / "basis_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks = (FLUID_NODES + CHUNK_POINTS - 1) // CHUNK_POINTS
    for chunk in range(chunks):
        begin = chunk * CHUNK_POINTS
        end = min(FLUID_NODES, begin + CHUNK_POINTS)
        points = end - begin
        target = chunk_dir / f"chunk_{chunk:02d}_velocity.f32"
        expected = K * points * 3 * 4
        if target.exists() and target.stat().st_size == expected:
            print(f"reuse strict basis chunk {chunk + 1}/{chunks}", flush=True)
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        output = np.memmap(temporary, dtype="<f4", mode="w+", shape=(K, points, 3))
        for mode_begin in range(0, K, 16):
            mode_end = min(K, mode_begin + 16)
            output[mode_begin:mode_end] = values[mode_begin:mode_end, begin:end]
        output.flush()
        del output
        os.replace(temporary, target)
        print(f"wrote strict basis chunk {chunk + 1}/{chunks}", flush=True)
    validate_basis(chunk_root)
    source = (
        ROOT / "results" / "laplacian_strict_complete_A2048_basis_full128"
        / "complete_A2048_mass_and_gt_correlation.npz"
    )
    with np.load(source, allow_pickle=False) as cached:
        # The cached mass includes the uniform cell volume; initial projection
        # uses unweighted B^T B and B^T u, so remove that common factor.
        gram = np.asarray(cached["mass"], dtype=np.float64) / ((1.0 / 128.0) ** 3)
    gram_path = RESULT / "systems" / "strict_fd_lap_rr" / "gram.npz"
    atomic_npz(gram_path, gram=gram)
    return {"basis": str(chunk_dir.resolve()), "gram": str(gram_path.resolve())}


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    manifest = {"strict_fd_lap_rr": prepare_strict()}
    for method, settings in METHODS.items():
        print(f"prepare {method} K=2048 full-grid basis", flush=True)
        manifest[method] = prepare_learned(method, settings)
    payload = {"complete": True, "modes": K, "fluid_nodes": FLUID_NODES, "methods": manifest}
    atomic_json(RESULT / "basis_manifest.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
