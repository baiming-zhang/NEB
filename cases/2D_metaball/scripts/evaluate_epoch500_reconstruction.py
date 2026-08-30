"""Matched full-K reconstruction on one unseen random metaball flow."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metaball_runtime import (
    backend,
    fluid_mask,
    geometry_from_token,
    install_backend,
    metaball_envelope_jet,
    metaball_fluid_points,
)


def load_model(checkpoint: Path):
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    install_backend(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = backend.cfg_namespace(config)
    return cfg, backend.HardcodedModel(cfg, arrays)


def gt_fields(cfg, points: np.ndarray, token: float, seed: int):
    x, y = backend.xy_jets(points, backend.F32)
    envelope = metaball_envelope_jet(cfg, x, y, token, 0.0)
    rng = np.random.default_rng(seed)
    raw = backend.Jet3.constant(0.0, x.value.shape, backend.F32)
    terms = []
    for _ in range(24):
        kx, ky = rng.integers(1, 13, size=2)
        phase0 = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = rng.normal() / math.sqrt(float(kx * kx + ky * ky))
        phase = 2.0 * math.pi * (float(kx) * x + float(ky) * y) + phase0
        sine, cosine = backend.cp.sin(phase.value), backend.cp.cos(phase.value)
        raw = raw + amplitude * backend.unary(phase, sine, cosine, -sine, -cosine)
        terms.append({"kx": int(kx), "ky": int(ky), "phase": phase0, "amplitude": amplitude})
    field = envelope * raw
    velocity = backend.cp.stack((field.grad[:, 0, 1], -field.grad[:, 0, 0]), axis=1)
    vorticity = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
    return velocity.astype(backend.F32), vorticity.astype(backend.F32), terms


def basis_fields(cfg, model, points: np.ndarray, token: float):
    count, modes = len(points), int(cfg.modes)
    omega = backend.cp.empty((count, modes), dtype=backend.F32)
    velocity = backend.cp.empty((2 * count, modes), dtype=backend.F32)
    for index, (_, field, _) in enumerate(model.forward_stream(cfg, points, token, 0.0, False)):
        omega[:, index] = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        velocity[0::2, index] = field.grad[:, 0, 1]
        velocity[1::2, index] = -field.grad[:, 0, 0]
    return omega, velocity


def relative_l2(observed, expected) -> float:
    return float(
        backend.cp.linalg.norm(observed - expected).get()
        / max(float(backend.cp.linalg.norm(expected).get()), 1.0e-20)
    )


def reconstruct(checkpoint: Path, train_points, test_points, grid_points, token, seed):
    cfg, model = load_model(checkpoint)
    train_omega, train_velocity = basis_fields(cfg, model, train_points, token)
    gt_train_velocity, gt_train_omega, terms = gt_fields(cfg, train_points, token, seed)
    gram = train_omega.T @ train_omega
    scale = backend.cp.trace(gram) / np.float32(cfg.modes)
    ridge = np.float32(1.0e-6) * scale
    coefficients = backend.cp.linalg.solve(
        gram + ridge * backend.cp.eye(cfg.modes, dtype=backend.F32),
        train_omega.T @ gt_train_omega,
    )
    train_omega_rec = train_omega @ coefficients
    train_velocity_rec = (train_velocity @ coefficients).reshape(-1, 2)
    del train_omega, train_velocity, gram

    test_omega, test_velocity = basis_fields(cfg, model, test_points, token)
    gt_test_velocity, gt_test_omega, _ = gt_fields(cfg, test_points, token, seed)
    test_omega_rec = test_omega @ coefficients
    test_velocity_rec = (test_velocity @ coefficients).reshape(-1, 2)
    metrics = {
        "train_vorticity_relative_l2": relative_l2(train_omega_rec, gt_train_omega),
        "train_velocity_relative_l2": relative_l2(train_velocity_rec, gt_train_velocity),
        "test_vorticity_relative_l2": relative_l2(test_omega_rec, gt_test_omega),
        "test_velocity_relative_l2": relative_l2(test_velocity_rec, gt_test_velocity),
        "coefficient_l2": float(backend.cp.linalg.norm(coefficients).get()),
        "ridge": float(ridge.get()),
    }
    del test_omega, test_velocity

    grid_omega, grid_velocity = basis_fields(cfg, model, grid_points, token)
    gt_grid_velocity, gt_grid_omega, _ = gt_fields(cfg, grid_points, token, seed)
    result = {
        "metrics": metrics,
        "terms": terms,
        "gt_omega": backend.cp.asnumpy(gt_grid_omega),
        "rec_omega": backend.cp.asnumpy(grid_omega @ coefficients),
        "gt_velocity": backend.cp.asnumpy(gt_grid_velocity),
        "rec_velocity": backend.cp.asnumpy((grid_velocity @ coefficients).reshape(-1, 2)),
    }
    del model
    backend.cp.get_default_memory_pool().free_all_blocks()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-checkpoint", type=Path, required=True)
    parser.add_argument("--siren-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", type=float, default=3_000_001.0)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=72)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plain_meta = json.loads(args.plain_checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    install_backend(plain_meta["config"])
    cfg = backend.cfg_namespace(plain_meta["config"])
    train_points = metaball_fluid_points(cfg, args.points, args.token, 0.0, args.seed + 1)
    test_points = metaball_fluid_points(cfg, args.points, args.token, 0.0, args.seed + 2)
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=np.float32)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_grid = np.column_stack((xx.ravel(), yy.ravel()))
    valid = fluid_mask(cfg, all_grid, args.token)
    grid_points = all_grid[valid].astype(np.float32)

    plain = reconstruct(args.plain_checkpoint, train_points, test_points, grid_points, args.token, args.seed)
    siren = reconstruct(args.siren_checkpoint, train_points, test_points, grid_points, args.token, args.seed)
    methods = [("plain sin", plain, "#B2182B"), ("spatial-SIREN", siren, "#2166AC")]
    rows = [{"method": name, **item["metrics"]} for name, item, _ in methods]
    with (args.output_dir / "random_unseen_full1024_reconstruction.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "geometry_token": args.token,
        "geometry_parameters": geometry_from_token(cfg, args.token).tolist(),
        "gt_seed": args.seed,
        "fit_points": args.points,
        "independent_test_points": args.points,
        "modes": 1024,
        "coefficient_fit": "vorticity least squares with matched relative ridge 1e-6",
        "methods": rows,
        "random_gt_terms": plain["terms"],
    }
    (args.output_dir / "random_unseen_full1024_reconstruction.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    shape = (args.resolution, args.resolution)
    def image(values):
        out = np.full(args.resolution * args.resolution, np.nan, dtype=np.float32)
        out[valid] = values
        return out.reshape(shape)
    gt_omega = image(plain["gt_omega"])
    gt_speed = image(np.linalg.norm(plain["gt_velocity"], axis=1))
    omega_limit = float(np.nanpercentile(np.abs(gt_omega), 99.0))
    speed_limit = float(np.nanpercentile(gt_speed, 99.0))
    fig, axes = plt.subplots(2, 5, figsize=(13.0, 5.4), constrained_layout=True)
    panels = []
    for quantity in ("omega", "speed"):
        gt = gt_omega if quantity == "omega" else gt_speed
        rec_plain = image(plain["rec_omega"] if quantity == "omega" else np.linalg.norm(plain["rec_velocity"], axis=1))
        rec_siren = image(siren["rec_omega"] if quantity == "omega" else np.linalg.norm(siren["rec_velocity"], axis=1))
        panels.append((gt, rec_plain, rec_siren, np.abs(rec_plain - gt), np.abs(rec_siren - gt)))
    for row, values in enumerate(panels):
        limit = omega_limit if row == 0 else speed_limit
        cmap = "RdBu_r" if row == 0 else "Blues"
        titles = ["random GT", "plain sin", "spatial-SIREN", "plain |error|", "SIREN |error|"]
        for column, (axis, values_i, title) in enumerate(zip(axes[row], values, titles)):
            if column < 3 and row == 0:
                axis.imshow(values_i, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
            elif column < 3:
                axis.imshow(values_i, origin="lower", cmap=cmap, vmin=0, vmax=limit)
            else:
                axis.imshow(values_i, origin="lower", cmap="magma", vmin=0, vmax=max(float(np.nanpercentile(values_i, 99)), 1e-8))
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel("vorticity" if row == 0 else "speed", fontsize=9)
    axes[0, 1].set_title(f"plain sin\n$e_\\omega$={plain['metrics']['test_vorticity_relative_l2']:.3f}", fontsize=8)
    axes[0, 2].set_title(f"spatial-SIREN\n$e_\\omega$={siren['metrics']['test_vorticity_relative_l2']:.3f}", fontsize=8)
    axes[1, 1].set_title(f"plain sin\n$e_u$={plain['metrics']['test_velocity_relative_l2']:.3f}", fontsize=8)
    axes[1, 2].set_title(f"spatial-SIREN\n$e_u$={siren['metrics']['test_velocity_relative_l2']:.3f}", fontsize=8)
    fig.suptitle("Unseen random metaball flow · full 1024-mode reconstruction", fontsize=11, weight="bold")
    fig.savefig(args.output_dir / "random_unseen_full1024_reconstruction.png", dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
