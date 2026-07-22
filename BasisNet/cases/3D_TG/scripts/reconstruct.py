"""Project periodic NS vortex snapshots onto a trained divergence-free basis."""

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
    discrete_all_mode_velocity,
)


def grid_points(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    mesh = np.meshgrid(x, y, z, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3).astype(np.float32)


def evaluate_basis(
    model,
    cfg,
    points: np.ndarray,
    batch: int,
) -> np.ndarray:
    pieces = []
    with torch.no_grad():
        for start in range(0, len(points), batch):
            values = discrete_all_mode_velocity(
                cfg,
                model,
                points[start : start + batch],
                0.0,
            )
            pieces.append(values.cpu().numpy().astype(np.float32))
    return np.concatenate(pieces)


def spectral_project_grid_basis(
    basis: np.ndarray,
    n: int,
    chunk: int = 16,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wave = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    kx, ky, kz = torch.meshgrid(wave, wave, wave, indexing="ij")
    k2 = kx.square() + ky.square() + kz.square()
    safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
    modes = basis.shape[1]
    grid = basis.reshape(n, n, n, modes, 3)
    projected = np.empty_like(grid)
    for start in range(0, modes, chunk):
        stop = min(start + chunk, modes)
        velocity = torch.as_tensor(
            grid[:, :, :, start:stop, :].transpose(3, 4, 0, 1, 2),
            device=device,
        )
        field_hat = torch.fft.fftn(
            velocity, dim=(-3, -2, -1)
        )
        dot = (
            kx * field_hat[:, 0]
            + ky * field_hat[:, 1]
            + kz * field_hat[:, 2]
        )
        field_hat = field_hat - torch.stack((kx, ky, kz), dim=0)[
            None
        ] * (dot[:, None] / safe)
        field_hat[:, :, 0, 0, 0] = 0
        values = torch.fft.ifftn(
            field_hat, dim=(-3, -2, -1)
        ).real
        projected[:, :, :, start:stop, :] = (
            values.cpu().numpy().transpose(2, 3, 4, 0, 1)
        )
    return projected.reshape(-1, modes, 3)


def relative_error(
    reconstructed: np.ndarray,
    actual: np.ndarray,
) -> np.ndarray:
    numerator = np.sum((reconstructed - actual) ** 2, axis=(1, 2))
    denominator = np.sum(actual**2, axis=(1, 2))
    return np.sqrt(numerator / np.maximum(denominator, 1.0e-24))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-stride", type=int, default=2)
    parser.add_argument("--basis-batch", type=int, default=512)
    parser.add_argument("--x-block", type=int, default=4)
    parser.add_argument("--spectral-project-basis", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg, model, checkpoint = load_model(args.checkpoint)
    if cfg.domain_type != "periodic_closed":
        raise ValueError(f"expected periodic_closed, got {cfg.domain_type}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source:
        x = source["coordinates/x"][:]
        y = source["coordinates/y"][:]
        z = source["coordinates/z"][:]
        times = source["time"][:]
        stride = args.projection_stride
        if args.spectral_project_basis:
            all_points = grid_points(x, y, z)
            print(
                f"[basis] evaluating {len(all_points):,} full-grid points",
                flush=True,
            )
            full_basis = evaluate_basis(
                model, cfg, all_points, args.basis_batch
            )
            full_basis = spectral_project_grid_basis(full_basis, len(x))
            projection_basis = full_basis.reshape(
                len(x), len(y), len(z), cfg.modes, 3
            )[::stride, ::stride, ::stride].reshape(-1, cfg.modes, 3)
            projection_points = grid_points(
                x[::stride], y[::stride], z[::stride]
            )
        else:
            projection_points = grid_points(
                x[::stride], y[::stride], z[::stride]
            )
            print(
                f"[projection] evaluating {len(projection_points):,} points",
                flush=True,
            )
            projection_basis = evaluate_basis(
                model, cfg, projection_points, args.basis_batch
            )
        projection_matrix = projection_basis.transpose(0, 2, 1).reshape(
            -1, cfg.modes
        )
        truth = source["fields/velocity"][
            :, ::stride, ::stride, ::stride, :
        ].reshape(len(times), -1, 3)
        targets = truth.transpose(1, 2, 0).reshape(-1, len(times))
        coefficients = np.linalg.lstsq(
            projection_matrix.astype(np.float64),
            targets.astype(np.float64),
            rcond=1.0e-8,
        )[0].astype(np.float32)
        projected = (
            projection_matrix @ coefficients
        ).reshape(len(projection_points), 3, len(times)).transpose(2, 0, 1)
        projection_error = relative_error(projected, truth)
        singular_values = np.linalg.svd(
            projection_matrix.astype(np.float64),
            compute_uv=False,
        )
        rank = int(
            np.count_nonzero(singular_values > singular_values[0] * 1.0e-8)
        )
        print(
            f"[projection] rank={rank} mean_error={projection_error.mean():.4f}",
            flush=True,
        )

        numerator = np.zeros(len(times), dtype=np.float64)
        denominator = np.zeros(len(times), dtype=np.float64)
        with h5py.File(args.output, "w") as output:
            coordinates = output.create_group("coordinates")
            coordinates.create_dataset("x", data=x)
            coordinates.create_dataset("y", data=y)
            coordinates.create_dataset("z", data=z)
            output.create_dataset("time", data=times)
            basis_group = output.create_group("basis")
            basis_group.create_dataset(
                "modal_coefficients", data=coefficients
            )
            fields = output.create_group("fields")
            reconstructed = fields.create_dataset(
                "velocity",
                shape=(len(times), len(x), len(y), len(z), 3),
                dtype="f4",
                chunks=(
                    1,
                    min(args.x_block, len(x)),
                    min(32, len(y)),
                    min(32, len(z)),
                    3,
                ),
                compression="lzf",
            )
            for start in range(0, len(x), args.x_block):
                stop = min(start + args.x_block, len(x))
                if args.spectral_project_basis:
                    basis = full_basis.reshape(
                        len(x), len(y), len(z), cfg.modes, 3
                    )[start:stop].reshape(-1, cfg.modes, 3)
                else:
                    points = grid_points(x[start:stop], y, z)
                    basis = evaluate_basis(
                        model, cfg, points, args.basis_batch
                    )
                block = (
                    basis.transpose(0, 2, 1).reshape(-1, cfg.modes)
                    @ coefficients
                ).reshape(
                    stop - start,
                    len(y),
                    len(z),
                    3,
                    len(times),
                )
                block = np.moveaxis(block, -1, 0).astype(np.float32)
                reconstructed[:, start:stop] = block
                actual = source["fields/velocity"][:, start:stop]
                difference = block.astype(np.float64) - actual
                numerator += np.sum(difference**2, axis=(1, 2, 3, 4))
                denominator += np.sum(
                    actual.astype(np.float64) ** 2, axis=(1, 2, 3, 4)
                )
                print(
                    f"[reconstruct] x={stop:02d}/{len(x)}",
                    flush=True,
                )
            output_error = np.sqrt(
                numerator / np.maximum(denominator, 1.0e-24)
            )
            metrics = output.create_group("metrics")
            metrics.create_dataset(
                "relative_velocity_error", data=output_error
            )
            output.attrs["source"] = str(args.source)
            output.attrs["checkpoint"] = str(args.checkpoint)
            output.attrs["checkpoint_epoch"] = int(checkpoint["epoch"])
            output.attrs["modes"] = cfg.modes
            output.attrs["basis_rank"] = rank
            output.attrs["boundary"] = "triply periodic closed domain"
            output.attrs["divergence_parameterization"] = (
                "row-wise discrete divergence of a hard-coded "
                "antisymmetric tensor"
            )

    summary = {
        "source": str(args.source),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "modes": cfg.modes,
        "basis_rank": rank,
        "projection_stride": stride,
        "projection_points": len(projection_points),
        "spectral_project_basis": args.spectral_project_basis,
        "projection_relative_velocity_error": projection_error.tolist(),
        "projection_mean_error": float(projection_error.mean()),
        "output_relative_velocity_error": output_error.tolist(),
        "output_mean_error": float(output_error.mean()),
        "output_max_error": float(output_error.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
