"""Evaluate a frozen 3D TG SSS basis and its projected reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.tg3d.postprocess import load_model  # noqa: E402
from eigenfluid_sss.basisnet.tg3d.training import (  # noqa: E402
    approximate_fluid_volume,
    discrete_all_mode_fields,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sobol_points(cfg, count: int, seed: int) -> np.ndarray:
    unit = torch.quasirandom.SobolEngine(3, scramble=True, seed=seed).draw(count)
    lower = torch.tensor([cfg.xmin, cfg.ymin, cfg.zmin])
    upper = torch.tensor([cfg.xmax, cfg.ymax, cfg.zmax])
    return (lower + unit * (upper - lower)).numpy().astype(np.float32)


def infer_metrics(cfg, model, points: np.ndarray, batch: int) -> dict:
    velocities = []
    gradients = []
    divergence_max = 0.0
    divergence_sum_sq = 0.0
    divergence_count = 0
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(points), batch):
            fields = discrete_all_mode_fields(
                cfg,
                model,
                points[start : start + batch],
                0.0,
            )
            velocities.append(fields["velocity"])
            gradients.append(fields["gradient"])
            divergence = fields["divergence"]
            divergence_max = max(
                divergence_max,
                float(divergence.abs().max().cpu()),
            )
            divergence_sum_sq += float(divergence.square().sum().cpu())
            divergence_count += divergence.numel()
    velocity = torch.cat(velocities, dim=0)
    gradient = torch.cat(gradients, dim=0)
    weight = approximate_fluid_volume(cfg, 0.0) / len(points)
    mass = weight * velocity.square().sum(dim=(0, 2))
    stiffness = weight * gradient.square().sum(dim=(0, 2, 3))
    rayleigh = stiffness / torch.clamp(mass, min=cfg.mass_floor)
    order = torch.argsort(rayleigh, stable=True)
    packed = velocity[:, order, :].permute(2, 0, 1).reshape(-1, cfg.modes)
    weighted = packed * np.sqrt(weight)
    q, r = torch.linalg.qr(weighted, mode="reduced")
    gram = q.T @ q
    singular = torch.linalg.svdvals(r)
    rank = int((singular > singular[0] * 1.0e-8).sum().cpu())
    synchronize()
    elapsed = time.perf_counter() - started
    identity = torch.eye(cfg.modes, device=gram.device, dtype=gram.dtype)
    return {
        "points": len(points),
        "inference_seconds": elapsed,
        "inference_points_per_second": len(points) / elapsed,
        "raw_basis_rank": rank,
        "peeled_orthogonality_max_abs": float((gram - identity).abs().max().cpu()),
        "hard_discrete_divergence_max_abs": divergence_max,
        "hard_discrete_divergence_rms": float(
            np.sqrt(divergence_sum_sq / divergence_count)
        ),
        "rayleigh_min": float(rayleigh.min().cpu()),
        "rayleigh_mean": float(rayleigh.mean().cpu()),
        "rayleigh_max": float(rayleigh.max().cpu()),
        "mode_order_head": order[:16].cpu().tolist(),
    }


def shell_spectrum(field: np.ndarray) -> np.ndarray:
    n = field.shape[0]
    wave = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(wave, wave, wave, indexing="ij")
    shell = np.rint(np.sqrt(kx * kx + ky * ky + kz * kz)).astype(np.int32)
    maximum = int(shell.max())
    spectrum = np.zeros(maximum + 1, dtype=np.float64)
    for component in range(3):
        transform = np.fft.fftn(field[..., component])
        energy = 0.5 * np.abs(transform) ** 2 / (n**6)
        spectrum += np.bincount(
            shell.ravel(), weights=energy.ravel(), minlength=maximum + 1
        )
    return spectrum


def reconstruction_metrics(source_path: Path, reconstruction_path: Path) -> dict:
    spectra_truth = []
    spectra_reconstruction = []
    central_sum_sq = 0.0
    central_count = 0
    central_max = 0.0
    spectral_sum_sq = 0.0
    spectral_count = 0
    spectral_max = 0.0
    with h5py.File(source_path, "r") as source, h5py.File(
        reconstruction_path, "r"
    ) as reconstruction:
        x = source["coordinates/x"][:]
        spacing = float(x[1] - x[0])
        n = len(x)
        wave = np.fft.fftfreq(n, d=spacing) * (2.0 * np.pi)
        kx, ky, kz = np.meshgrid(wave, wave, wave, indexing="ij")
        errors = reconstruction["metrics/relative_velocity_error"][:]
        for frame in range(len(source["time"])):
            truth = source["fields/velocity"][frame].astype(np.float64)
            predicted = reconstruction["fields/velocity"][frame].astype(np.float64)
            spectra_truth.append(shell_spectrum(truth))
            spectra_reconstruction.append(shell_spectrum(predicted))
            divergence = (
                (np.roll(predicted[..., 0], -1, 0) - np.roll(predicted[..., 0], 1, 0))
                + (np.roll(predicted[..., 1], -1, 1) - np.roll(predicted[..., 1], 1, 1))
                + (np.roll(predicted[..., 2], -1, 2) - np.roll(predicted[..., 2], 1, 2))
            ) / (2.0 * spacing)
            central_sum_sq += float(np.sum(divergence**2))
            central_count += divergence.size
            central_max = max(central_max, float(np.max(np.abs(divergence))))
            transformed = [np.fft.fftn(predicted[..., c]) for c in range(3)]
            divergence_hat = 1j * (
                kx * transformed[0] + ky * transformed[1] + kz * transformed[2]
            )
            spectral_divergence = np.fft.ifftn(divergence_hat).real
            spectral_sum_sq += float(np.sum(spectral_divergence**2))
            spectral_count += spectral_divergence.size
            spectral_max = max(
                spectral_max, float(np.max(np.abs(spectral_divergence)))
            )
    truth_spectrum = np.mean(spectra_truth, axis=0)
    predicted_spectrum = np.mean(spectra_reconstruction, axis=0)
    denominator = max(float(np.linalg.norm(truth_spectrum)), 1.0e-30)
    return {
        "relative_velocity_error_mean": float(np.mean(errors)),
        "relative_velocity_error_max": float(np.max(errors)),
        "central_grid_divergence_rms": float(
            np.sqrt(central_sum_sq / central_count)
        ),
        "central_grid_divergence_max_abs": central_max,
        "spectral_divergence_rms": float(
            np.sqrt(spectral_sum_sq / spectral_count)
        ),
        "spectral_divergence_max_abs": spectral_max,
        "energy_spectrum_relative_l2": float(
            np.linalg.norm(predicted_spectrum - truth_spectrum) / denominator
        ),
        "kinetic_energy_relative_error": float(
            abs(predicted_spectrum.sum() - truth_spectrum.sum())
            / max(float(truth_spectrum.sum()), 1.0e-30)
        ),
        "energy_spectrum_truth": truth_spectrum.tolist(),
        "energy_spectrum_reconstruction": predicted_spectrum.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    cfg, model, checkpoint = load_model(args.checkpoint)
    points = sobol_points(cfg, args.points, args.seed)
    basis = infer_metrics(cfg, model, points, args.batch)
    reconstruction = reconstruction_metrics(args.source, args.reconstruction)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "modes": cfg.modes,
        "difference_step": cfg.difference_step,
        "basis": basis,
        "reconstruction": reconstruction,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
