"""Taylor--Green GT and independent Laplacian/potential modal rollouts.

The default ``(2,2)`` square-domain Taylor--Green solution has four
counter-rotating cells.  Its quadratic Euler self-interaction vanishes, so
each modal rollout is advanced by the exact viscous factor associated with
its own eigenvalue.  The repository includes the aligned learned ``(2,2)``
potential mode used for the published result.  Supplying the original K2048
checkpoint and Ritz transform reproduces that mode export from scratch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

try:
    import cupy as cp
except ImportError:  # The published cache path is CPU-only.
    cp = None


TAYLOR_DIR = Path(__file__).resolve().parent
CASE_ROOT = TAYLOR_DIR.parent
ROOT = CASE_ROOT.parents[1]
REPO = ROOT
SQUARE_SCRIPTS = CASE_ROOT / "scripts"
sys.path.insert(0, str(SQUARE_SCRIPTS))

from vortex_merger.fem import VorticityStreamfunctionFEM
from vortex_merger.modes import SquareLaplacianBasis

NATURE_BLUE = "#2166AC"
NATURE_RED = "#B2182B"
CHARCOAL = "#252525"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_fem_l2(
    reference: np.ndarray, approximation: np.ndarray, mass
) -> float:
    difference = (reference - approximation).reshape(-1).astype(np.float64)
    target = reference.reshape(-1).astype(np.float64)
    return float(
        np.sqrt(
            difference @ (mass @ difference)
            / max(target @ (mass @ target), 1e-30)
        )
    )


def taylor_green_gt(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    amplitude: float,
    viscosity: float,
    wave_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return streamfunction, vorticity, and velocity on ``[0, pi]^2``.

    The convention matches the project: ``u=(psi_y,-psi_x)`` and
    ``omega=-Delta psi``.
    """
    xx, yy = np.meshgrid(x, y, indexing="ij")
    q2 = 2.0 * wave_number**2
    decay = np.exp(-q2 * viscosity * times)[:, None, None]
    spatial = np.sin(wave_number * xx) * np.sin(wave_number * yy)
    psi = amplitude / q2 * decay * spatial[None]
    omega = amplitude * decay * spatial[None]
    velocity_scale = amplitude * wave_number / q2
    velocity_x = velocity_scale * decay * (
        np.sin(wave_number * xx) * np.cos(wave_number * yy)
    )[None]
    velocity_y = -velocity_scale * decay * (
        np.cos(wave_number * xx) * np.sin(wave_number * yy)
    )[None]
    return psi, omega, velocity_x, velocity_y


def evaluate_aligned_potential_mode(
    checkpoint: Path,
    transform: np.ndarray,
    points: np.ndarray,
    aligned_mode: int = 0,
) -> tuple[np.ndarray, dict, object]:
    """Evaluate one aligned Ritz mode without materializing all grid modes."""
    if cp is None:
        raise RuntimeError("checkpoint evaluation requires the optional CuPy dependency")
    from evaluate_dirichlet_psi_basis import load_model

    metadata, cfg, model = load_model(checkpoint)
    if int(cfg.modes) != transform.shape[0]:
        raise ValueError("checkpoint and saved Ritz transform mode counts differ")
    weights = cp.asarray(transform[:, aligned_mode], dtype=cp.float32)
    combined = cp.zeros(len(points), dtype=cp.float32)
    for mode, field, _ in model.forward_stream_jet1(
        cfg, points.astype(np.float32, copy=False), 0.0, 0.0, False
    ):
        network = int(mode.mode_index)
        combined += weights[network] * field.value[:, 0]
    result = cp.asnumpy(combined).astype(np.float64)
    del combined, weights, model
    cp.get_default_memory_pool().free_all_blocks()
    return result, metadata, cfg


def save_field_figure(
    path_png: Path,
    path_pdf: Path,
    x: np.ndarray,
    y: np.ndarray,
    gt: np.ndarray,
    laplacian: np.ndarray,
    potential: np.ndarray,
    error_laplacian: float,
    error_potential: float,
) -> None:
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    vmax = float(np.max(np.abs(gt)))
    field_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    potential_difference = potential - gt
    difference_limit = max(float(np.max(np.abs(potential_difference))), 1e-12)
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 3.9), constrained_layout=True)
    panels = (
        (gt, "Taylor--Green GT", field_norm),
        (
            laplacian,
            f"Laplacian K2048\n$e_{{L^2_M}}$={error_laplacian:.2e}",
            field_norm,
        ),
        (
            potential,
            f"Trained potential K2048\n$e_{{L^2_M}}$={error_potential:.3e}",
            field_norm,
        ),
        (potential_difference, "Potential - GT", difference_norm),
    )
    images = []
    for axis, (field, title, norm) in zip(axes, panels, strict=True):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            norm=norm,
            interpolation="bilinear",
        )
        images.append(image)
        axis.set(title=title, xlabel="x", ylabel="y")
        axis.set_aspect("equal")
    fig.colorbar(images[2], ax=axes[:3], fraction=0.022, pad=0.015, label="vorticity")
    fig.colorbar(images[3], ax=axes[3], fraction=0.046, pad=0.03, label="error")
    fig.suptitle("Taylor--Green 2x2 first-frame reconstruction")
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def save_summary(
    output: Path,
    times: np.ndarray,
    error_laplacian: np.ndarray,
    error_potential: np.ndarray,
    amplitude_laplacian: np.ndarray,
    amplitude_potential: np.ndarray,
    amplitude_gt: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    axes[0].semilogy(
        times,
        np.maximum(error_laplacian, 1e-17),
        color=NATURE_BLUE,
        lw=2.2,
        label="Laplacian K2048",
    )
    axes[0].semilogy(
        times,
        np.maximum(error_potential, 1e-17),
        color=NATURE_RED,
        lw=2.2,
        label="Trained potential K2048",
    )
    axes[0].set(
        xlabel="time",
        ylabel="relative FEM-mass $L^2$ error",
        title="Vorticity rollout error",
    )
    axes[0].grid(alpha=0.22, which="both")
    axes[0].legend(frameon=False)
    axes[1].plot(times, amplitude_gt, color=CHARCOAL, lw=2.4, label="exact GT")
    axes[1].plot(
        times,
        amplitude_laplacian,
        color=NATURE_BLUE,
        lw=2.0,
        ls="--",
        label="Laplacian",
    )
    axes[1].plot(
        times,
        amplitude_potential,
        color=NATURE_RED,
        lw=2.0,
        ls=":",
        label="potential",
    )
    axes[1].set(
        xlabel="time", ylabel="active-mode coefficient", title="Independent modal decay"
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"taylor_green_error_summary.{suffix}",
            dpi=240 if suffix == "png" else None,
        )
    plt.close(fig)


def render_video_frames(
    output: Path,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    gt: np.ndarray,
    laplacian: np.ndarray,
    potential: np.ndarray,
    error_laplacian: np.ndarray,
    error_potential: np.ndarray,
) -> Path:
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    vmax = float(np.max(np.abs(gt)))
    field_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    potential_difference_limit = max(
        float(np.max(np.abs(potential - gt))), 1e-12
    )
    laplacian_difference_limit = max(
        float(np.max(np.abs(laplacian - gt))), 1e-12
    )
    for index, time_value in enumerate(times):
        fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.2), constrained_layout=True)
        top = (
            (gt[index], "Taylor--Green GT"),
            (
                laplacian[index],
                f"Laplacian K2048\n$e_{{L^2_M}}$={error_laplacian[index]:.2e}",
            ),
            (
                potential[index],
                f"Trained potential K2048\n$e_{{L^2_M}}$={error_potential[index]:.3e}",
            ),
        )
        top_images = []
        for axis, (field, title) in zip(axes[0], top, strict=True):
            image = axis.imshow(
                field.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                norm=field_norm,
                interpolation="bilinear",
            )
            top_images.append(image)
            axis.set(title=title, xlabel="x", ylabel="y")
            axis.set_aspect("equal")
        fig.colorbar(
            top_images[-1], ax=axes[0], fraction=0.020, pad=0.012, label="vorticity"
        )
        differences = (
            (
                laplacian[index] - gt[index],
                "Laplacian - GT",
                laplacian_difference_limit,
            ),
            (
                potential[index] - gt[index],
                "Potential - GT",
                potential_difference_limit,
            ),
        )
        for axis, (field, title, limit) in zip(axes[1, :2], differences, strict=True):
            image = axis.imshow(
                field.T,
                origin="lower",
                extent=extent,
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                interpolation="bilinear",
            )
            axis.set(title=title, xlabel="x", ylabel="y")
            axis.set_aspect("equal")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        error_axis = axes[1, 2]
        error_axis.semilogy(
            times[: index + 1],
            np.maximum(error_laplacian[: index + 1], 1e-17),
            color=NATURE_BLUE,
            lw=2.2,
            label="Laplacian",
        )
        error_axis.semilogy(
            times[: index + 1],
            np.maximum(error_potential[: index + 1], 1e-17),
            color=NATURE_RED,
            lw=2.2,
            label="potential",
        )
        error_axis.set(
            xlim=(float(times[0]), float(times[-1])),
            ylim=(1e-17, max(1e-2, float(error_potential.max()) * 1.4)),
            xlabel="time",
            ylabel="relative $L^2_M$ error",
            title="Accumulated rollout error",
        )
        error_axis.grid(alpha=0.22, which="both")
        error_axis.legend(frameon=False)
        fig.suptitle(
            f"Taylor--Green 2x2 independent coefficient rollout, t={time_value:.3f}"
        )
        fig.savefig(frames_dir / f"compare_{index:03d}.png", dpi=170)
        plt.close(fig)
        print(f"rendered frame {index + 1:02d}/{len(times)}", flush=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to create the comparison video")
    video = output / "taylor_green_2x2_gt_vs_laplacian_vs_potential.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            "10",
            "-i",
            str(frames_dir / "compare_%03d.png"),
            "-vf",
            "scale=1920:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return video


def probe_video(video: Path, expected_frames: int) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to verify the comparison video")
    probe = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=codec_name,width,height,duration,nb_read_frames",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )["streams"][0]
    if int(probe["nb_read_frames"]) != expected_frames:
        raise RuntimeError("encoded video frame count is incorrect")
    return probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO
        / "cases/2D_Square/results/native_square_siren_k2048_e2000/"
        "fp64_mgs_run_001/checkpoints/epoch_2000_neutral.npz",
    )
    parser.add_argument(
        "--transform",
        type=Path,
        default=TAYLOR_DIR / "trained_k2048_basis_transform.npz",
        help="Aligned Ritz transform required only for checkpoint re-export.",
    )
    parser.add_argument(
        "--potential-mode-cache",
        type=Path,
        default=TAYLOR_DIR / "potential_mode_2_2.npy",
        help="Published aligned learned vorticity mode on the 97x97 grid.",
    )
    parser.add_argument(
        "--learned-eigenvalue",
        type=float,
        default=7.987962172069773,
        help="Physical Ritz eigenvalue paired with the published mode cache.",
    )
    parser.add_argument(
        "--force-checkpoint-evaluation",
        action="store_true",
        help="Ignore the published cache and evaluate checkpoint plus transform.",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output/taylor_green_2x2_validation"
    )
    parser.add_argument("--modes", type=int, default=2048)
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--final-time",
        type=float,
        default=80.0,
        help="Longer horizon makes viscous decay visible at nu=1e-3.",
    )
    parser.add_argument("--viscosity", type=float, default=1.0e-3)
    parser.add_argument("--amplitude", type=float, default=12.0)
    parser.add_argument(
        "--wave-number",
        type=int,
        default=2,
        help="Equal x/y wave number; 2 gives a 2x2 vortex-cell layout.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.modes != 2048:
        raise ValueError("the saved trained potential transform contains K2048 modes")
    basis = SquareLaplacianBasis.lowest(args.modes)
    matching_modes = np.flatnonzero(
        (basis.k1 == args.wave_number) & (basis.k2 == args.wave_number)
    )
    if matching_modes.size != 1:
        raise RuntimeError("requested equal-wave-number mode is absent or ambiguous")
    active_mode = int(matching_modes[0])
    # For k=(m,m) interacting with itself, the cross-product branches have
    # zero coefficient and the dot-product branches have a zero target wave
    # number.  Hence the quadratic Galerkin self-interaction is identically 0.
    kx = int(basis.k1[active_mode])
    ky = int(basis.k2[active_mode])
    cross = kx * ky - ky * kx
    dot_branch_targets = ((kx + kx, ky - ky), (kx - kx, ky + ky))
    if cross != 0 or not all(tx == 0 or ty == 0 for tx, ty in dot_branch_targets):
        raise RuntimeError("Taylor--Green self-interaction did not vanish")

    fem = VorticityStreamfunctionFEM(args.cells, args.viscosity)
    x = fem.x
    y = fem.y
    times = np.linspace(0.0, args.final_time, args.frames)
    gt_psi, gt_omega, gt_ux, gt_uy = taylor_green_gt(
        x, y, times, args.amplitude, args.viscosity, args.wave_number
    )
    initial_coefficients = basis.project_vorticity(gt_omega[0], x, y)
    inactive = np.ones(args.modes, dtype=bool)
    inactive[active_mode] = False
    numerical_tail_max = float(np.max(np.abs(initial_coefficients[inactive])))
    initial_coefficients[np.abs(initial_coefficients) < 1e-12 * args.amplitude] = 0.0
    if (
        np.count_nonzero(initial_coefficients) != 1
        or initial_coefficients[active_mode] == 0.0
    ):
        raise RuntimeError("Taylor--Green projection did not isolate the requested mode")

    analytic_eigenvalue = float(basis.wave_number_squared[active_mode])
    use_cache = (
        args.potential_mode_cache.is_file()
        and not args.force_checkpoint_evaluation
    )
    if use_cache:
        if args.wave_number != 2 or args.cells != 96:
            raise ValueError("the published potential cache is specific to m=2 and 97x97")
        learned_vorticity_mode = np.load(
            args.potential_mode_cache, allow_pickle=False
        ).astype(np.float64)
        if learned_vorticity_mode.shape != (len(x), len(y)):
            raise ValueError("the potential mode cache has the wrong grid shape")
        learned_eigenvalue = float(args.learned_eigenvalue)
        potential_source = "published aligned (2,2) mode cache"
        print("potential: loaded published aligned trained (2,2) mode", flush=True)
    else:
        if not args.checkpoint.is_file() or not args.transform.is_file():
            raise FileNotFoundError(
                "checkpoint re-export requires both --checkpoint and --transform"
            )
        with np.load(args.transform, allow_pickle=False) as archive:
            transform = archive["network_to_aligned_ritz_transform"].astype(
                np.float32, copy=True
            )
            learned_eigenvalues = archive["learned_eigenvalues_physical"].astype(
                np.float64, copy=True
            )
            transform_k1 = archive["k1"].copy()
            transform_k2 = archive["k2"].copy()
        if not (
            np.array_equal(transform_k1, basis.k1)
            and np.array_equal(transform_k2, basis.k2)
        ):
            raise RuntimeError("saved potential transform is not aligned to this basis")
        unit_x = x / np.pi
        unit_y = y / np.pi
        xx_unit, yy_unit = np.meshgrid(unit_x, unit_y, indexing="ij")
        points = np.column_stack((xx_unit.ravel(), yy_unit.ravel())).astype(np.float32)
        print(
            f"potential: evaluating aligned trained "
            f"({args.wave_number},{args.wave_number}) mode",
            flush=True,
        )
        learned_stream_mode_flat, metadata, cfg = evaluate_aligned_potential_mode(
            args.checkpoint, transform, points, active_mode
        )
        if int(metadata["epoch"]) != 2000 or int(cfg.modes) != args.modes:
            raise ValueError("expected the completed potential K2048/e2000 checkpoint")
        # Weak Rayleigh training produces a streamfunction realization whose
        # half value matches the unit-amplitude vorticity mode convention.
        learned_vorticity_mode = 0.5 * learned_stream_mode_flat.reshape(
            len(x), len(y)
        )
        learned_eigenvalue = float(learned_eigenvalues[active_mode])
        potential_source = "K2048/e2000 checkpoint plus aligned Ritz transform"
    amplitude_laplacian = args.amplitude * np.exp(
        -args.viscosity * analytic_eigenvalue * times
    )
    amplitude_potential = args.amplitude * np.exp(
        -args.viscosity * learned_eigenvalue * times
    )
    amplitude_gt = args.amplitude * np.exp(
        -args.viscosity * analytic_eigenvalue * times
    )
    analytic_mode = np.sin(args.wave_number * x[:, None]) * np.sin(
        args.wave_number * y[None, :]
    )
    laplacian_frames = amplitude_laplacian[:, None, None] * analytic_mode[None]
    potential_frames = (
        amplitude_potential[:, None, None] * learned_vorticity_mode[None]
    )

    error_laplacian = np.asarray(
        [relative_fem_l2(gt_omega[i], laplacian_frames[i], fem.mass) for i in range(args.frames)]
    )
    error_potential = np.asarray(
        [relative_fem_l2(gt_omega[i], potential_frames[i], fem.mass) for i in range(args.frames)]
    )
    print(
        "first frame: "
        f"Laplacian={error_laplacian[0]:.6e}, "
        f"potential={error_potential[0]:.6e}",
        flush=True,
    )
    print(
        "final frame: "
        f"Laplacian={error_laplacian[-1]:.6e}, "
        f"potential={error_potential[-1]:.6e}",
        flush=True,
    )

    np.savez_compressed(
        args.output / "taylor_green_2x2_30frames.npz",
        x=x,
        y=y,
        times=times,
        gt_streamfunction=gt_psi.astype(np.float32),
        gt_vorticity=gt_omega.astype(np.float32),
        gt_velocity_x=gt_ux.astype(np.float32),
        gt_velocity_y=gt_uy.astype(np.float32),
        laplacian_vorticity=laplacian_frames.astype(np.float32),
        potential_vorticity=potential_frames.astype(np.float32),
        laplacian_coefficients=amplitude_laplacian,
        potential_coefficients=amplitude_potential,
        laplacian_relative_l2=error_laplacian,
        potential_relative_l2=error_potential,
        analytic_eigenvalue=analytic_eigenvalue,
        learned_potential_eigenvalue=learned_eigenvalue,
        active_mode_index=active_mode,
        wave_number=args.wave_number,
        k1=basis.k1,
        k2=basis.k2,
    )

    rows = [
        {
            "frame": index,
            "time": float(times[index]),
            "laplacian_relative_l2": float(error_laplacian[index]),
            "potential_relative_l2": float(error_potential[index]),
            "gt_coefficient": float(amplitude_gt[index]),
            "laplacian_coefficient": float(amplitude_laplacian[index]),
            "potential_coefficient": float(amplitude_potential[index]),
        }
        for index in range(args.frames)
    ]
    with (args.output / "frame_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_field_figure(
        args.output / "taylor_green_2x2_first_frame.png",
        args.output / "taylor_green_2x2_first_frame.pdf",
        x,
        y,
        gt_omega[0],
        laplacian_frames[0],
        potential_frames[0],
        float(error_laplacian[0]),
        float(error_potential[0]),
    )
    save_summary(
        args.output,
        times,
        error_laplacian,
        error_potential,
        amplitude_laplacian,
        amplitude_potential,
        amplitude_gt,
    )
    video = render_video_frames(
        args.output,
        x,
        y,
        times,
        gt_omega,
        laplacian_frames,
        potential_frames,
        error_laplacian,
        error_potential,
    )
    video_probe = probe_video(video, args.frames)

    metrics = {
        "status": "passed",
        "case": "Taylor--Green 2x2 vortex cells on [0, pi]^2",
        "equations": {
            "vorticity": "A exp(-8 nu t) sin(2x) sin(2y)",
            "velocity_x": "A/4 exp(-8 nu t) sin(2x) cos(2y)",
            "velocity_y": "-A/4 exp(-8 nu t) cos(2x) sin(2y)",
        },
        "configuration": {
            "modes": args.modes,
            "cells": args.cells,
            "frames": args.frames,
            "final_time": args.final_time,
            "viscosity": args.viscosity,
            "amplitude": args.amplitude,
            "wave_number": args.wave_number,
        },
        "initialization": (
            f"GT frame 0 only; exact ({args.wave_number},{args.wave_number}) "
            "coefficient projection"
        ),
        "advance": (
            f"independent exact viscous coefficient decay; the "
            f"({args.wave_number},{args.wave_number}) quadratic self-interaction "
            "is identically zero"
        ),
        "analytic_laplacian_eigenvalue": analytic_eigenvalue,
        "trained_potential_eigenvalue": learned_eigenvalue,
        "trained_potential_eigenvalue_relative_error": abs(
            learned_eigenvalue - analytic_eigenvalue
        )
        / analytic_eigenvalue,
        "exact_final_to_initial_amplitude_ratio": float(
            amplitude_gt[-1] / amplitude_gt[0]
        ),
        "projection_numerical_tail_max_before_threshold": numerical_tail_max,
        "potential_mode_source": potential_source,
        "potential_mode_cache": (
            str(args.potential_mode_cache.resolve()) if use_cache else None
        ),
        "potential_mode_cache_sha256": (
            sha256(args.potential_mode_cache) if use_cache else None
        ),
        "checkpoint": str(args.checkpoint.resolve()) if not use_cache else None,
        "checkpoint_sha256": sha256(args.checkpoint) if not use_cache else None,
        "transform": str(args.transform.resolve()) if not use_cache else None,
        "transform_sha256": sha256(args.transform) if not use_cache else None,
        "laplacian": {
            "first_relative_l2": float(error_laplacian[0]),
            "mean_relative_l2": float(error_laplacian.mean()),
            "final_relative_l2": float(error_laplacian[-1]),
        },
        "potential": {
            "first_relative_l2": float(error_potential[0]),
            "mean_relative_l2": float(error_potential.mean()),
            "final_relative_l2": float(error_potential[-1]),
        },
        "video": str(video.resolve()),
        "video_sha256": sha256(video),
        "video_probe": video_probe,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
