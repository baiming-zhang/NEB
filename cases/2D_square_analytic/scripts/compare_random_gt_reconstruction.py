"""Compare epoch-500 and epoch-1000 bases on three random GT flows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_dirichlet_psi_basis import load_model, raw_fields


BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#252525"
F32 = np.float32


def generate_parameters(seed: int, fields: int = 3, vortices: int = 7):
    rng = np.random.default_rng(seed)
    result = []
    for field_index in range(fields):
        centers = rng.uniform(0.13, 0.87, size=(vortices, 2))
        sigmas = rng.uniform(0.105, 0.215, size=vortices)
        amplitudes = rng.normal(0.0, 1.0, size=vortices)
        amplitudes -= 0.35 * np.mean(amplitudes)
        result.append(
            {
                "field": field_index + 1,
                "centers": centers,
                "sigmas": sigmas,
                "amplitudes": amplitudes,
            }
        )
    return result


def ground_truth(
    points: np.ndarray,
    parameters: list[dict[str, Any]],
    normalization: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = x * (1.0 - x)
    b = y * (1.0 - y)
    envelope = 16.0 * a * b
    envelope_x = 16.0 * (1.0 - 2.0 * x) * b
    envelope_y = 16.0 * a * (1.0 - 2.0 * y)
    envelope_xx = -32.0 * b
    envelope_yy = -32.0 * a

    streamfunctions = []
    velocities = []
    vorticities = []
    for item in parameters:
        centers = item["centers"]
        sigmas = item["sigmas"]
        amplitudes = item["amplitudes"]
        dx = x[:, None] - centers[None, :, 0]
        dy = y[:, None] - centers[None, :, 1]
        inverse_variance = 1.0 / (sigmas[None, :] ** 2)
        kernels = np.exp(
            -0.5 * (dx * dx + dy * dy) * inverse_variance
        )
        weighted = kernels * amplitudes[None, :]
        source = np.sum(weighted, axis=1)
        source_x = np.sum(-dx * inverse_variance * weighted, axis=1)
        source_y = np.sum(-dy * inverse_variance * weighted, axis=1)
        source_xx = np.sum(
            (dx * dx * inverse_variance**2 - inverse_variance) * weighted,
            axis=1,
        )
        source_yy = np.sum(
            (dy * dy * inverse_variance**2 - inverse_variance) * weighted,
            axis=1,
        )
        psi = envelope * source
        psi_x = envelope_x * source + envelope * source_x
        psi_y = envelope_y * source + envelope * source_y
        laplacian = (
            (envelope_xx + envelope_yy) * source
            + 2.0 * (envelope_x * source_x + envelope_y * source_y)
            + envelope * (source_xx + source_yy)
        )
        streamfunctions.append(psi)
        velocities.append(np.column_stack((psi_y, -psi_x)))
        vorticities.append(-laplacian)

    streamfunction = np.column_stack(streamfunctions)
    velocity = np.stack(velocities, axis=-1)
    vorticity = np.column_stack(vorticities)
    if normalization is None:
        normalization = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1), axis=0))
    streamfunction /= normalization[None, :]
    velocity /= normalization[None, None, :]
    vorticity /= normalization[None, :]
    return streamfunction, velocity, vorticity, normalization


def basis_on_points(
    cfg: Any,
    model: Any,
    transform: cp.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    network_value, _, network_omega, network_velocity = raw_fields(
        cfg, model, points
    )
    value = cp.asnumpy(network_value @ transform).astype(np.float64)
    omega = cp.asnumpy(network_omega @ transform).astype(np.float64)
    velocity = cp.asnumpy(
        cp.einsum(
            "ncm,mk->nck", network_velocity, transform, optimize=True
        )
    ).astype(np.float64)
    return value, velocity, omega


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    axes = tuple(range(target.ndim - 1))
    numerator = np.sum((prediction - target) ** 2, axis=axes)
    denominator = np.sum(target**2, axis=axes)
    return np.sqrt(numerator / denominator)


def reconstruct_stage(
    label: str,
    checkpoint: Path,
    fields_path: Path,
    fit_points: np.ndarray,
    test_points: np.ndarray,
    gt_fit: tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_test: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    metadata, cfg, model = load_model(checkpoint)
    with np.load(fields_path, allow_pickle=False) as archive:
        transform_numpy = archive["network_to_peeled_ritz_transform"]
        grid_value = archive["streamfunction"].astype(np.float64)
        grid_velocity = archive["velocity"].astype(np.float64)
        grid_omega = archive["vorticity"].astype(np.float64)
        coordinates = archive["coordinates"].astype(np.float64)
    transform = cp.asarray(transform_numpy, dtype=cp.float32)
    fit_value, fit_velocity, fit_omega = basis_on_points(
        cfg, model, transform, fit_points
    )
    design = fit_velocity.reshape(-1, fit_velocity.shape[-1])
    targets = gt_fit[1].reshape(-1, gt_fit[1].shape[-1])
    gram = (design.T @ design) / len(design)
    rhs = (design.T @ targets) / len(design)
    ridge = 1.0e-10 * float(np.trace(gram)) / len(gram)
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(len(gram), dtype=np.float64), rhs
    )
    fit_reconstruction = (
        fit_value @ coefficients,
        np.einsum("nck,kg->ncg", fit_velocity, coefficients, optimize=True),
        fit_omega @ coefficients,
    )

    test_value, test_velocity, test_omega = basis_on_points(
        cfg, model, transform, test_points
    )
    test_reconstruction = (
        test_value @ coefficients,
        np.einsum("nck,kg->ncg", test_velocity, coefficients, optimize=True),
        test_omega @ coefficients,
    )
    grid_reconstruction = (
        np.einsum("kxy,kg->gxy", grid_value, coefficients, optimize=True),
        np.einsum("kxyc,kg->gxyc", grid_velocity, coefficients, optimize=True),
        np.einsum("kxy,kg->gxy", grid_omega, coefficients, optimize=True),
    )
    result = {
        "label": label,
        "epoch": int(metadata["epoch"]),
        "coefficients": coefficients,
        "coordinates": coordinates,
        "grid": grid_reconstruction,
        "fit_errors": {
            "streamfunction": relative_l2(fit_reconstruction[0], gt_fit[0]),
            "velocity": relative_l2(fit_reconstruction[1], gt_fit[1]),
            "vorticity": relative_l2(fit_reconstruction[2], gt_fit[2]),
        },
        "test_errors": {
            "streamfunction": relative_l2(test_reconstruction[0], gt_test[0]),
            "velocity": relative_l2(test_reconstruction[1], gt_test[1]),
            "vorticity": relative_l2(test_reconstruction[2], gt_test[2]),
        },
        "velocity_gram_condition_number": float(np.linalg.cond(gram)),
    }
    del model, transform
    cp.get_default_memory_pool().free_all_blocks()
    return result


def serializable_parameters(parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "field": int(item["field"]),
            "centers": np.asarray(item["centers"]).tolist(),
            "sigmas": np.asarray(item["sigmas"]).tolist(),
            "amplitudes": np.asarray(item["amplitudes"]).tolist(),
        }
        for item in parameters
    ]


def render_comparison(
    coordinates: np.ndarray,
    gt_grid: tuple[np.ndarray, np.ndarray, np.ndarray],
    stages: list[dict[str, Any]],
    output_stem: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 5, figsize=(17.0, 9.3), constrained_layout=False)
    fig.subplots_adjust(
        left=0.035,
        right=0.995,
        top=0.895,
        bottom=0.075,
        wspace=0.10,
        hspace=0.22,
    )
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    skip = max(1, len(coordinates) // 13)
    gt_value, gt_velocity, _ = gt_grid
    stage500, stage1000 = stages
    column_titles = [
        "Ground truth",
        "Epoch 500 reconstruction",
        "Epoch 500 velocity error",
        "Epoch 1000 reconstruction",
        "Epoch 1000 velocity error",
    ]
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=11.2, weight="bold", pad=5.0)

    for field_index in range(3):
        recon500_value = stage500["grid"][0][field_index]
        recon500_velocity = stage500["grid"][1][field_index]
        recon1000_value = stage1000["grid"][0][field_index]
        recon1000_velocity = stage1000["grid"][1][field_index]
        velocity_error500 = np.linalg.norm(
            recon500_velocity - gt_velocity[field_index], axis=-1
        )
        velocity_error1000 = np.linalg.norm(
            recon1000_velocity - gt_velocity[field_index], axis=-1
        )
        psi_limit = max(
            float(np.max(np.abs(gt_value[field_index]))),
            float(np.max(np.abs(recon500_value))),
            float(np.max(np.abs(recon1000_value))),
            1.0e-12,
        )
        error_limit = max(
            float(np.percentile(velocity_error500, 99.0)),
            float(np.percentile(velocity_error1000, 99.0)),
            1.0e-12,
        )
        flow_panels = [
            (0, gt_value[field_index], gt_velocity[field_index]),
            (1, recon500_value, recon500_velocity),
            (3, recon1000_value, recon1000_velocity),
        ]
        for column, value, velocity in flow_panels:
            axis = axes[field_index, column]
            axis.imshow(
                value,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="RdBu_r",
                vmin=-psi_limit,
                vmax=psi_limit,
                interpolation="bilinear",
            )
            axis.quiver(
                xx[::skip, ::skip],
                yy[::skip, ::skip],
                velocity[::skip, ::skip, 0],
                velocity[::skip, ::skip, 1],
                color=CHARCOAL,
                alpha=0.60,
                scale=25.0,
                width=0.0027,
                headwidth=3.0,
            )
        for column, error, stage in (
            (2, velocity_error500, stage500),
            (4, velocity_error1000, stage1000),
        ):
            axis = axes[field_index, column]
            axis.imshow(
                error,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="Reds",
                vmin=0.0,
                vmax=error_limit,
                interpolation="bilinear",
            )
            errors = stage["test_errors"]
            axis.text(
                0.03,
                0.97,
                (
                    f"$e_u$={100*errors['velocity'][field_index]:.1f}%\n"
                    f"$e_\\psi$={100*errors['streamfunction'][field_index]:.1f}%  "
                    f"$e_\\omega$={100*errors['vorticity'][field_index]:.1f}%"
                ),
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=8.2,
                color=CHARCOAL,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
            )
        axes[field_index, 0].set_ylabel(
            f"Random GT {field_index + 1}", fontsize=11.0, weight="bold"
        )
        for axis in axes[field_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            for spine in axis.spines.values():
                spine.set_color("#666666")
                spine.set_linewidth(0.55)

    fig.suptitle(
        "Reconstruction of three unseen random incompressible flows with K=64 bases",
        fontsize=16.0,
        weight="bold",
    )
    fig.text(
        0.5,
        0.025,
        "GT: seven random Gaussian vortices times the square Dirichlet envelope. "
        "Velocity coefficients fit on 32,768 IID points; errors evaluated on 65,536 disjoint IID points.",
        ha="center",
        fontsize=9.5,
        color="#4D4D4D",
    )
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint500", type=Path, required=True)
    parser.add_argument("--checkpoint1000", type=Path, required=True)
    parser.add_argument("--fields500", type=Path, required=True)
    parser.add_argument("--fields1000", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-points", type=int, default=32768)
    parser.add_argument("--test-points", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameters = generate_parameters(args.seed)
    fit_points = np.random.default_rng(args.seed + 1001).random(
        (args.fit_points, 2), dtype=np.float32
    )
    test_points = np.random.default_rng(args.seed + 2003).random(
        (args.test_points, 2), dtype=np.float32
    )
    gt_fit_all = ground_truth(fit_points, parameters)
    normalization = gt_fit_all[3]
    gt_fit = gt_fit_all[:3]
    gt_test = ground_truth(test_points, parameters, normalization)[:3]

    stage500 = reconstruct_stage(
        "epoch500",
        args.checkpoint500,
        args.fields500,
        fit_points,
        test_points,
        gt_fit,
        gt_test,
    )
    stage1000 = reconstruct_stage(
        "epoch1000",
        args.checkpoint1000,
        args.fields1000,
        fit_points,
        test_points,
        gt_fit,
        gt_test,
    )
    coordinates = stage500["coordinates"]
    if not np.array_equal(coordinates, stage1000["coordinates"]):
        raise ValueError("stage grids differ")
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="xy")
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    gt_grid_flat = ground_truth(grid_points, parameters, normalization)[:3]
    resolution = len(coordinates)
    gt_grid = (
        gt_grid_flat[0].T.reshape(3, resolution, resolution),
        gt_grid_flat[1].transpose(2, 0, 1).reshape(3, resolution, resolution, 2),
        gt_grid_flat[2].T.reshape(3, resolution, resolution),
    )

    stages = [stage500, stage1000]
    rows = []
    for stage in stages:
        for field_index in range(3):
            rows.append(
                {
                    "stage": stage["label"],
                    "epoch": stage["epoch"],
                    "ground_truth": field_index + 1,
                    "fit_velocity_relative_l2": float(
                        stage["fit_errors"]["velocity"][field_index]
                    ),
                    "test_streamfunction_relative_l2": float(
                        stage["test_errors"]["streamfunction"][field_index]
                    ),
                    "test_velocity_relative_l2": float(
                        stage["test_errors"]["velocity"][field_index]
                    ),
                    "test_vorticity_relative_l2": float(
                        stage["test_errors"]["vorticity"][field_index]
                    ),
                }
            )
    csv_path = args.output_dir / "random_gt_reconstruction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
        return {
            "epoch": int(stage["epoch"]),
            "velocity_gram_condition_number": float(
                stage["velocity_gram_condition_number"]
            ),
            "mean_test_streamfunction_relative_l2": float(
                np.mean(stage["test_errors"]["streamfunction"])
            ),
            "mean_test_velocity_relative_l2": float(
                np.mean(stage["test_errors"]["velocity"])
            ),
            "mean_test_vorticity_relative_l2": float(
                np.mean(stage["test_errors"]["vorticity"])
            ),
            "per_gt_test_streamfunction_relative_l2": stage["test_errors"][
                "streamfunction"
            ].tolist(),
            "per_gt_test_velocity_relative_l2": stage["test_errors"][
                "velocity"
            ].tolist(),
            "per_gt_test_vorticity_relative_l2": stage["test_errors"][
                "vorticity"
            ].tolist(),
        }

    summary500 = stage_summary(stage500)
    summary1000 = stage_summary(stage1000)
    velocity500 = np.asarray(
        summary500["per_gt_test_velocity_relative_l2"]
    )
    velocity1000 = np.asarray(
        summary1000["per_gt_test_velocity_relative_l2"]
    )
    summary = {
        "ground_truth_generator": (
            "seven IID random Gaussian vortices multiplied by "
            "16*x*(1-x)*y*(1-y); analytic derivatives"
        ),
        "ground_truth_seed": args.seed,
        "ground_truth_parameters": serializable_parameters(parameters),
        "normalization_velocity_rms": normalization.tolist(),
        "fit": {
            "objective": "unweighted velocity L2 least squares",
            "points": args.fit_points,
            "sampling": "IID uniform random; no D4 augmentation",
        },
        "test": {
            "points": args.test_points,
            "sampling": "disjoint IID uniform random; no D4 augmentation",
        },
        "analytic_laplacian_modes_used": False,
        "epoch500": summary500,
        "epoch1000": summary1000,
        "velocity_error_relative_change_percent": (
            100.0 * (velocity1000 - velocity500) / velocity500
        ).tolist(),
        "ground_truths_with_lower_velocity_error_at_epoch1000": int(
            np.count_nonzero(velocity1000 < velocity500)
        ),
    }
    (args.output_dir / "random_gt_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "random_gt_reconstruction_fields.npz",
        coordinates=coordinates,
        gt_streamfunction=gt_grid[0].astype(F32),
        gt_velocity=gt_grid[1].astype(F32),
        gt_vorticity=gt_grid[2].astype(F32),
        epoch500_streamfunction=stage500["grid"][0].astype(F32),
        epoch500_velocity=stage500["grid"][1].astype(F32),
        epoch500_vorticity=stage500["grid"][2].astype(F32),
        epoch1000_streamfunction=stage1000["grid"][0].astype(F32),
        epoch1000_velocity=stage1000["grid"][1].astype(F32),
        epoch1000_vorticity=stage1000["grid"][2].astype(F32),
        epoch500_coefficients=stage500["coefficients"].astype(F32),
        epoch1000_coefficients=stage1000["coefficients"].astype(F32),
    )
    render_comparison(
        coordinates,
        gt_grid,
        stages,
        args.output_dir / "random_gt_reconstruction_e500_vs_e1000",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
