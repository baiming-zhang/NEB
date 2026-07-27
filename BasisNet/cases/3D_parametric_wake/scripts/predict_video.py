"""Project an independent 3D driven-wake trajectory onto a frozen basis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt


matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    }
)

CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.parametric_wake3d.postprocess import (  # noqa: E402
    load_model,
)
from eigenfluid_sss.basisnet.parametric_wake3d.training import (  # noqa: E402
    discrete_all_mode_velocity,
    obstacle_sdf_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basis-chunk", type=int, default=512)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def vorticity_z(field: np.ndarray, length: float) -> np.ndarray:
    resolution = field.shape[-1]
    frequency = np.fft.fftfreq(resolution, d=length / resolution) * 2.0 * np.pi
    kx, ky, _ = np.meshgrid(
        frequency, frequency, frequency, indexing="ij"
    )
    uhat = np.fft.fftn(field, axes=(-3, -2, -1))
    omega_hat = 1j * (kx * uhat[1] - ky * uhat[0])
    return np.fft.ifftn(omega_hat, axes=(-3, -2, -1)).real


def infer_basis(
    cfg,
    model,
    points: np.ndarray,
    alpha: float,
    chunk: int,
) -> tuple[np.ndarray, float]:
    pieces: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(points), chunk):
            stop = min(start + chunk, len(points))
            value = (
                discrete_all_mode_velocity(
                    cfg, model, points[start:stop], alpha
                )
                .detach()
                .cpu()
                .numpy()
                .transpose(0, 2, 1)
            )
            pieces.append(value)
    return np.concatenate(pieces), time.perf_counter() - started


def render_frame(
    gt: np.ndarray,
    reconstruction: np.ndarray,
    axis: np.ndarray,
    alpha: float,
    relative_l2: float,
    output: Path,
) -> None:
    length = float(axis[-1] - axis[0] + axis[1] - axis[0])
    middle = len(axis) // 2
    gt_slice = vorticity_z(gt, length)[:, :, middle].T
    reconstruction_slice = vorticity_z(reconstruction, length)[:, :, middle].T
    error_slice = np.abs(gt_slice - reconstruction_slice)
    limit = max(
        float(np.percentile(np.abs(gt_slice), 99.0)),
        float(np.percentile(np.abs(reconstruction_slice), 99.0)),
        1.0e-8,
    )
    error_limit = max(float(np.percentile(error_slice, 99.0)), 1.0e-8)

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 12.0), constrained_layout=True)
    extent = (float(axis[0]), float(axis[-1]), float(axis[0]), float(axis[-1]))
    panels = (
        (gt_slice, "Independent NS reference", "seismic", -limit, limit),
        (
            reconstruction_slice,
            "One-shot learned-basis reconstruction",
            "seismic",
            -limit,
            limit,
        ),
        (error_slice, "Absolute vorticity error", "magma", 0.0, error_limit),
    )
    for ax, (image, title, cmap, lower, upper) in zip(axes, panels):
        artist = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="bilinear",
        )
        ax.set(title=title, xlabel="$x$", ylabel="$y$", aspect="equal")
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(
        rf"3D parametric driven wake, $\alpha={alpha:.3f}$, "
        rf"velocity relative $L_2={100.0 * relative_l2:.2f}\%$"
    )
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.source_cache) as cache:
        velocity = cache["velocity"].astype(np.float32)
        times = cache["time"].astype(np.float64)
        axis = cache["axis"].astype(np.float32)
        alpha = float(cache["alpha"])
        reference_divergence = cache["divergence_max"].astype(np.float64)

    cfg, model, payload = load_model(args.checkpoint)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(
        np.float32
    )
    fluid = obstacle_sdf_numpy(cfg, points, alpha) > 0.0
    basis, inference_seconds = infer_basis(
        cfg, model, points[fluid], alpha, args.basis_chunk
    )
    flat_basis = basis.reshape(-1, cfg.modes).astype(np.float64)
    gram = flat_basis.T @ flat_basis
    inverse = np.linalg.pinv(gram, rcond=1.0e-8)

    errors: list[float] = []
    frames: list[Path] = []
    for index, field in enumerate(velocity):
        target = field.reshape(3, -1).T[fluid].reshape(-1).astype(np.float64)
        coefficients = inverse @ (flat_basis.T @ target)
        approximation = flat_basis @ coefficients
        relative_l2 = float(
            np.linalg.norm(approximation - target)
            / max(np.linalg.norm(target), 1.0e-15)
        )
        reconstruction = np.zeros((len(points), 3), dtype=np.float32)
        reconstruction[fluid] = approximation.reshape(-1, 3)
        reconstruction = reconstruction.T.reshape(field.shape)
        frame_path = frame_dir / f"frame_{index:04d}.png"
        render_frame(
            field,
            reconstruction,
            axis,
            alpha,
            relative_l2,
            frame_path,
        )
        errors.append(relative_l2)
        frames.append(frame_path)

    movie_path = args.output_dir / "driven_wake_gt_vs_basis.mp4"
    with imageio.get_writer(
        movie_path, fps=args.fps, codec="libx264", quality=8
    ) as writer:
        for frame in frames:
            writer.append_data(imageio.imread(frame))

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "source_cache": str(args.source_cache.resolve()),
        "alpha": alpha,
        "times": times.tolist(),
        "velocity_relative_l2_per_snapshot": errors,
        "velocity_relative_l2_mean": float(np.mean(errors)),
        "velocity_relative_l2_max": float(np.max(errors)),
        "reference_divergence_max": float(reference_divergence.max()),
        "basis_inference_seconds": inference_seconds,
        "movie": str(movie_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
