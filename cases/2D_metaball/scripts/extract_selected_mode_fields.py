from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluate_checkpoint_vs_laplacian import finite_difference_laplacian, load_model
from metaball_runtime import backend, fluid_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", type=float, default=3_000_001.0)
    parser.add_argument("--resolution", type=int, default=72)
    args = parser.parse_args()

    metadata, cfg, model = load_model(args.checkpoint)
    if int(metadata["epoch"]) != 2300 or int(cfg.modes) != 1024:
        raise ValueError("expected epoch-2300 K=1024 checkpoint")

    selected = np.asarray(list(range(16)) + list(range(1008, 1024)), dtype=np.int32)
    lookup = {int(mode): slot for slot, mode in enumerate(selected)}
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=np.float32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    valid = fluid_mask(cfg, all_points, args.token).reshape(args.resolution, args.resolution)
    points = all_points[valid.ravel()]

    neural_omega = np.empty((len(points), len(selected)), dtype=np.float32)
    neural_velocity = np.empty((len(points), 2, len(selected)), dtype=np.float32)
    for mode_index, (_, field, _) in enumerate(
        model.forward_stream(cfg, points, args.token, 0.0, False)
    ):
        slot = lookup.get(mode_index)
        if slot is None:
            continue
        neural_omega[:, slot] = backend.cp.asnumpy(
            -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        )
        neural_velocity[:, 0, slot] = backend.cp.asnumpy(field.grad[:, 0, 1])
        neural_velocity[:, 1, slot] = backend.cp.asnumpy(-field.grad[:, 0, 0])
    del model
    backend.cp.get_default_memory_pool().free_all_blocks()

    eigenvalues, _, lap_omega_all, lap_velocity_all = finite_difference_laplacian(
        valid, spacing, 1024
    )
    lap_omega = lap_omega_all[:, selected]
    lap_velocity = lap_velocity_all[:, :, selected]

    # Canonicalize display signs using the largest-magnitude vorticity sample.
    for omega, velocity in (
        (neural_omega, neural_velocity),
        (lap_omega, lap_velocity),
    ):
        pivots = np.argmax(np.abs(omega), axis=0)
        signs = np.sign(omega[pivots, np.arange(omega.shape[1])])
        signs[signs == 0.0] = 1.0
        omega *= signs[None, :]
        velocity *= signs[None, None, :]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        coordinates=coordinates,
        valid=valid,
        selected_indices=selected,
        neural_omega=neural_omega,
        neural_velocity=neural_velocity,
        laplacian_omega=lap_omega,
        laplacian_velocity=lap_velocity,
        laplacian_eigenvalues=eigenvalues[selected],
    )
    print(args.output)


if __name__ == "__main__":
    main()
