"""Compare a peeled metaball Jet3 basis with a geometry-specific Laplacian basis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from metaball_runtime import (
    apply_basis_transform,
    backend,
    fluid_mask,
    geometry_from_token,
    install_backend,
    metaball_envelope_jet,
    metaball_fluid_points,
    raw_basis_field_matrices,
    vorticity_peeling_transform,
)


BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#262626"
F32 = np.float32


def load_model(checkpoint: Path):
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    config = dict(metadata["config"])
    install_backend(config)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = backend.cfg_namespace(config)
    return metadata, cfg, backend.HardcodedModel(cfg, arrays)


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
        terms.append(
            {
                "kx": int(kx),
                "ky": int(ky),
                "phase": float(phase0),
                "amplitude": float(amplitude),
            }
        )
    field = envelope * raw
    velocity = backend.cp.stack((field.grad[:, 0, 1], -field.grad[:, 0, 0]), axis=1)
    vorticity = -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
    return velocity.astype(backend.F32), vorticity.astype(backend.F32), terms


def relative_l2_np(observed: np.ndarray, expected: np.ndarray) -> float:
    numerator = np.linalg.norm(
        observed.astype(np.float64, copy=False) - expected.astype(np.float64, copy=False)
    )
    denominator = max(np.linalg.norm(expected.astype(np.float64, copy=False)), 1.0e-30)
    return float(numerator / denominator)


def fit_numpy(
    omega: np.ndarray,
    velocity: np.ndarray,
    gt_omega: np.ndarray,
    gt_velocity: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    ridge_relative: float,
) -> tuple[np.ndarray, dict]:
    omega_train = omega[train].astype(np.float64, copy=False)
    target_train = gt_omega[train].astype(np.float64, copy=False)
    gram = omega_train.T @ omega_train
    ridge = ridge_relative * float(np.trace(gram) / gram.shape[0])
    rhs = omega_train.T @ target_train
    coefficients = scipy.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), rhs, assume_a="pos", check_finite=False
    )
    rec_omega = (omega @ coefficients).astype(F32)
    rec_velocity = np.einsum("ncm,m->nc", velocity, coefficients, optimize=True).astype(F32)
    gram_eigenvalues = np.linalg.eigvalsh(gram)
    threshold = max(float(gram_eigenvalues[-1]) * 1.0e-7, 1.0e-20)
    metrics = {
        "train_vorticity_relative_l2": relative_l2_np(rec_omega[train], gt_omega[train]),
        "train_velocity_relative_l2": relative_l2_np(
            rec_velocity[train], gt_velocity[train]
        ),
        "test_vorticity_relative_l2": relative_l2_np(rec_omega[test], gt_omega[test]),
        "test_velocity_relative_l2": relative_l2_np(
            rec_velocity[test], gt_velocity[test]
        ),
        "fit_gram_effective_rank": int(np.count_nonzero(gram_eigenvalues > threshold)),
        "coefficient_l2": float(np.linalg.norm(coefficients)),
        "ridge": float(ridge),
    }
    return coefficients, {"omega": rec_omega, "velocity": rec_velocity, "metrics": metrics}


def fit_neural(
    cfg,
    model,
    points: np.ndarray,
    token: float,
    gt_omega_cp,
    gt_velocity_cp,
    train: np.ndarray,
    test: np.ndarray,
    ridge_relative: float,
    peeling_points: np.ndarray,
) -> dict:
    transform, peeling = vorticity_peeling_transform(
        cfg, model, peeling_points, token
    )
    raw_omega, raw_velocity = raw_basis_field_matrices(
        cfg, model, points, token
    )
    omega, velocity = apply_basis_transform(
        raw_omega, raw_velocity, transform
    )
    del raw_omega, raw_velocity
    train_cp = backend.cp.asarray(train)
    test_cp = backend.cp.asarray(test)
    omega_train = omega[train_cp]
    target_train = gt_omega_cp[train_cp]
    gram = omega_train.T @ omega_train
    ridge = F32(ridge_relative) * backend.cp.trace(gram) / F32(cfg.modes)
    coefficients = backend.cp.linalg.solve(
        gram + ridge * backend.cp.eye(int(cfg.modes), dtype=backend.F32),
        omega_train.T @ target_train,
    )
    rec_omega = omega @ coefficients
    rec_velocity = backend.cp.einsum("ncm,m->nc", velocity, coefficients)
    gram_eigenvalues = backend.cp.linalg.eigvalsh(gram)
    threshold = backend.cp.maximum(gram_eigenvalues[-1] * F32(1.0e-7), F32(1.0e-20))

    def relative(observed, expected) -> float:
        return float(
            backend.cp.linalg.norm(observed - expected).get()
            / max(float(backend.cp.linalg.norm(expected).get()), 1.0e-30)
        )

    metrics = {
        "train_vorticity_relative_l2": relative(
            rec_omega[train_cp], gt_omega_cp[train_cp]
        ),
        "train_velocity_relative_l2": relative(
            rec_velocity[train_cp], gt_velocity_cp[train_cp]
        ),
        "test_vorticity_relative_l2": relative(rec_omega[test_cp], gt_omega_cp[test_cp]),
        "test_velocity_relative_l2": relative(
            rec_velocity[test_cp], gt_velocity_cp[test_cp]
        ),
        "fit_gram_effective_rank": int(backend.cp.count_nonzero(gram_eigenvalues > threshold).get()),
        "coefficient_l2": float(backend.cp.linalg.norm(coefficients).get()),
        "ridge": float(ridge.get()),
    }
    result = {
        "omega": backend.cp.asnumpy(rec_omega),
        "velocity": backend.cp.asnumpy(rec_velocity),
        "metrics": metrics,
        "peeling": peeling,
    }
    del omega, velocity, gram, coefficients, gram_eigenvalues, transform
    backend.cp.get_default_memory_pool().free_all_blocks()
    return result


def finite_difference_laplacian(
    valid: np.ndarray, spacing: float, modes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dirichlet streamfunction modes on the exact evaluation mask."""
    ny, nx = valid.shape
    node_index = np.full((ny, nx), -1, dtype=np.int32)
    node_index[valid] = np.arange(np.count_nonzero(valid), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    inverse_h2 = 1.0 / (spacing * spacing)
    for iy, ix in np.argwhere(valid):
        row = int(node_index[iy, ix])
        rows.append(row)
        columns.append(row)
        data.append(4.0 * inverse_h2)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < ny and 0 <= jx < nx and valid[jy, jx]:
                rows.append(row)
                columns.append(int(node_index[jy, jx]))
                data.append(-inverse_h2)
    operator = scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(int(np.count_nonzero(valid)),) * 2
    )
    if modes >= operator.shape[0] - 1:
        raise ValueError(f"requested K={modes} for only {operator.shape[0]} fluid nodes")
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        operator,
        k=modes,
        sigma=0.0,
        which="LM",
        tol=2.0e-7,
        maxiter=12000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    norms = np.sqrt(np.sum(eigenvectors * eigenvectors, axis=0) * spacing * spacing)
    eigenvectors /= norms[None, :]
    pivots = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[pivots, np.arange(modes)])
    signs[signs == 0.0] = 1.0
    eigenvectors *= signs[None, :]

    psi = np.zeros((ny, nx, modes), dtype=F32)
    psi[valid] = eigenvectors.astype(F32)
    dpsi_dx = np.zeros_like(psi)
    dpsi_dy = np.zeros_like(psi)
    dpsi_dx[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / F32(2.0 * spacing)
    dpsi_dy[1:-1] = (psi[2:] - psi[:-2]) / F32(2.0 * spacing)
    velocity = np.stack((dpsi_dy[valid], -dpsi_dx[valid]), axis=1)
    omega = eigenvectors.astype(F32) * eigenvalues.astype(F32)[None, :]
    return eigenvalues, eigenvectors.astype(F32), omega, velocity


def render(
    output: Path,
    case_label: str,
    modes: int,
    epoch: int,
    coordinates: np.ndarray,
    valid: np.ndarray,
    gt_omega: np.ndarray,
    gt_velocity: np.ndarray,
    neural: dict,
    laplacian: dict,
) -> None:
    resolution = len(coordinates)

    def grid(values: np.ndarray) -> np.ndarray:
        result = np.full((resolution, resolution), np.nan, dtype=F32)
        result[valid] = values
        return result

    gt_omega_grid = grid(gt_omega)
    gt_speed_grid = grid(np.linalg.norm(gt_velocity, axis=1))
    neural_omega = grid(neural["omega"])
    lap_omega = grid(laplacian["omega"])
    neural_speed = grid(np.linalg.norm(neural["velocity"], axis=1))
    lap_speed = grid(np.linalg.norm(laplacian["velocity"], axis=1))
    neural_velocity_error = grid(np.linalg.norm(neural["velocity"] - gt_velocity, axis=1))
    lap_velocity_error = grid(np.linalg.norm(laplacian["velocity"] - gt_velocity, axis=1))
    neural_omega_error = grid(np.abs(neural["omega"] - gt_omega))
    lap_omega_error = grid(np.abs(laplacian["omega"] - gt_omega))
    omega_limit = float(np.nanpercentile(np.abs(gt_omega_grid), 99.0))
    speed_limit = float(np.nanpercentile(gt_speed_grid, 99.0))
    error_omega_limit = max(
        float(np.nanpercentile(neural_omega_error, 99.0)),
        float(np.nanpercentile(lap_omega_error, 99.0)),
        1.0e-8,
    )
    error_velocity_limit = max(
        float(np.nanpercentile(neural_velocity_error, 99.0)),
        float(np.nanpercentile(lap_velocity_error, 99.0)),
        1.0e-8,
    )
    panels = [
        (gt_omega_grid, "GT", "RdBu_r", -omega_limit, omega_limit),
        (
            neural_omega,
            f"peeled SIREN\n$e_\\omega$={neural['metrics']['test_vorticity_relative_l2']:.3f}",
            "RdBu_r",
            -omega_limit,
            omega_limit,
        ),
        (
            lap_omega,
            f"Laplacian\n$e_\\omega$={laplacian['metrics']['test_vorticity_relative_l2']:.3f}",
            "RdBu_r",
            -omega_limit,
            omega_limit,
        ),
        (neural_omega_error, "SIREN |error|", "magma", 0.0, error_omega_limit),
        (lap_omega_error, "Laplacian |error|", "magma", 0.0, error_omega_limit),
        (gt_speed_grid, "GT", "Blues", 0.0, speed_limit),
        (
            neural_speed,
            f"peeled SIREN\n$e_u$={neural['metrics']['test_velocity_relative_l2']:.3f}",
            "Blues",
            0.0,
            speed_limit,
        ),
        (
            lap_speed,
            f"Laplacian\n$e_u$={laplacian['metrics']['test_velocity_relative_l2']:.3f}",
            "Blues",
            0.0,
            speed_limit,
        ),
        (neural_velocity_error, "SIREN vector |error|", "magma", 0.0, error_velocity_limit),
        (lap_velocity_error, "Laplacian vector |error|", "magma", 0.0, error_velocity_limit),
    ]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 7.2,
            "axes.linewidth": 0.6,
            "axes.titleweight": "regular",
        }
    )
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 4.55), constrained_layout=True)
    for index, (axis, (values, title, cmap, vmin, vmax)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        axis.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.contour(valid.astype(float), levels=[0.5], colors=[CHARCOAL], linewidths=0.45)
        axis.set_title(title, fontsize=7.4, pad=1.5)
        axis.set_xticks([])
        axis.set_yticks([])
        if index in (0, 5):
            axis.set_ylabel("vorticity" if index == 0 else "velocity", fontsize=8)
    stride = max(4, resolution // 13)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    for axis, velocity in zip(axes[1, :3], (gt_velocity, neural["velocity"], laplacian["velocity"])):
        ug = grid(velocity[:, 0])
        vg = grid(velocity[:, 1])
        axis.quiver(
            xx[::stride, ::stride],
            yy[::stride, ::stride],
            ug[::stride, ::stride],
            vg[::stride, ::stride],
            color="#303030",
            scale=None,
            width=0.0022,
            headwidth=2.3,
            alpha=0.55,
        )
    fig.suptitle(
        f"{case_label} - full K={modes} reconstruction at epoch {epoch}",
        fontsize=10.5,
        weight="bold",
    )
    fig.savefig(output, dpi=320, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(
        output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token", type=float, default=3_000_001.0)
    parser.add_argument("--case-label", default="Unseen metaball flow")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--resolution", type=int, default=72)
    parser.add_argument("--modes", type=int)
    parser.add_argument("--ridge-relative", type=float, default=1.0e-6)
    parser.add_argument("--peeling-points", type=int, default=4096)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata, cfg, model = load_model(args.checkpoint)
    checkpoint_epoch = int(metadata["epoch"])
    requested_modes = int(cfg.modes) if args.modes is None else args.modes
    if int(cfg.modes) != requested_modes:
        raise ValueError(f"checkpoint K={cfg.modes}, requested K={args.modes}")
    coordinates = np.linspace(0.0, 1.0, args.resolution, dtype=F32)
    spacing = float(coordinates[1] - coordinates[0])
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(F32)
    valid = fluid_mask(cfg, all_points, args.token).reshape(args.resolution, args.resolution)
    points = all_points[valid.ravel()]
    if len(points) < requested_modes + 513:
        raise ValueError(f"only {len(points)} fluid nodes at resolution {args.resolution}")
    rng = np.random.default_rng(args.seed + 31)
    permutation = rng.permutation(len(points))
    train_count = min(
        len(points) - 512, max(2 * requested_modes, int(0.70 * len(points)))
    )
    train, test = permutation[:train_count], permutation[train_count:]
    peeling_points = metaball_fluid_points(
        cfg,
        args.peeling_points,
        args.token,
        0.0,
        args.seed + 800_001,
    )

    gt_velocity_cp, gt_omega_cp, terms = gt_fields(cfg, points, args.token, args.seed)
    neural = fit_neural(
        cfg,
        model,
        points,
        args.token,
        gt_omega_cp,
        gt_velocity_cp,
        train,
        test,
        args.ridge_relative,
        peeling_points,
    )
    gt_velocity = backend.cp.asnumpy(gt_velocity_cp)
    gt_omega = backend.cp.asnumpy(gt_omega_cp)
    del model, gt_velocity_cp, gt_omega_cp
    backend.cp.get_default_memory_pool().free_all_blocks()

    eigenvalues, eigenvectors, lap_omega, lap_velocity = finite_difference_laplacian(
        valid, spacing, requested_modes
    )
    lap_coefficients, laplacian = fit_numpy(
        lap_omega,
        lap_velocity,
        gt_omega,
        gt_velocity,
        train,
        test,
        args.ridge_relative,
    )
    del eigenvectors, lap_omega, lap_velocity, lap_coefficients
    rows = [
        {"method": f"peeled SIREN epoch {checkpoint_epoch}", **neural["metrics"]},
        {"method": "geometry-aware FD Laplacian", **laplacian["metrics"]},
    ]
    prefix = (
        f"epoch{checkpoint_epoch:04d}_k{requested_modes}_"
        "peeled_siren_vs_laplacian"
    )
    with (args.output_dir / f"{prefix}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(metadata["epoch"]),
        "geometry_token": args.token,
        "geometry_parameters": geometry_from_token(cfg, args.token).tolist(),
        "case_label": args.case_label,
        "gt_seed": args.seed,
        "random_gt_terms": terms,
        "grid_resolution": args.resolution,
        "fluid_nodes": int(len(points)),
        "fit_nodes": int(len(train)),
        "held_out_test_nodes": int(len(test)),
        "modes": requested_modes,
        "basis_form": "twice-reorthogonalized vorticity-peeled modes",
        "peeling": neural["peeling"],
        "coefficient_fit": (
            "matched vorticity least squares; "
            f"relative ridge {args.ridge_relative:.6g}"
        ),
        "laplacian_baseline": {
            "name": "geometry-aware five-point finite-difference Dirichlet streamfunction Laplacian",
            "boundary_condition": "psi=0 on outer and obstacle boundaries (slip velocity)",
            "eigensolver": "scipy sparse symmetric shift-invert eigsh",
            "eigenvalue_min": float(eigenvalues[0]),
            "eigenvalue_max": float(eigenvalues[-1]),
            "recomputed_per_geometry": True,
        },
        "methods": rows,
    }
    (args.output_dir / f"{prefix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / f"{prefix}_fields.npz",
        coordinates=coordinates,
        valid=valid,
        gt_omega=gt_omega,
        gt_velocity=gt_velocity,
        siren_omega=neural["omega"],
        siren_velocity=neural["velocity"],
        laplacian_omega=laplacian["omega"],
        laplacian_velocity=laplacian["velocity"],
        train_indices=train,
        test_indices=test,
        laplacian_eigenvalues=eigenvalues,
    )
    render(
        args.output_dir / f"{prefix}.png",
        args.case_label,
        requested_modes,
        checkpoint_epoch,
        coordinates,
        valid,
        gt_omega,
        gt_velocity,
        neural,
        laplacian,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
