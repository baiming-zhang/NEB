"""Create a four-page PDF for E2000 learned bases and the 128^3 Laplacian."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


CASE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CASE_ROOT.parents[1]
sys.path.insert(0, str(CASE_ROOT / "scripts"))
import evaluate_sweep_winner_128 as comparison  # noqa: E402
import evaluate_unseen_vs_laplacian as base  # noqa: E402
import plot_unseen_vector_reconstruction_256 as plot_utils  # noqa: E402


LAPLACIAN_ROOT = CASE_ROOT / "results" / "unseen_k64_laplacian_reconstruction_128"
ORIGINAL_CHECKPOINT = (
    CASE_ROOT
    / "results"
    / "k64_siren_dirichlet_e2000_original"
    / "run_001"
    / "checkpoints"
    / "final.pt"
)
WINNER_CHECKPOINT = (
    CASE_ROOT
    / "results"
    / "k64_siren_dirichlet_e2000_sweep_winner"
    / "run_001"
    / "checkpoints"
    / "final.pt"
)
BASE_CONFIG = (
    CASE_ROOT / "configs" / "k64_three_sphere_connected_siren_dirichlet_2000.json"
)
DEFAULT_PDF = REPO_ROOT / "output" / "pdf" / "3d_metaball_e2000_modes_random_reconstruction.pdf"
TARGET_SEEDS = (20_260_831, 20_260_832, 20_260_833)
GEOMETRY_TOKEN = 12_000_001
SAMPLE_SEED = 20_260_825
NATURE_DARK = "#243447"
NATURE_GREY = "#64748B"


def assemble_reader_stable_pdf(output: Path, page_pngs: list[Path], dpi: int = 260) -> None:
    """Package verified raster pages to avoid backend-specific PDF rendering gaps."""
    document = canvas.Canvas(str(output))
    document.setTitle("3D Metaball E2000 Modes and Random Reconstruction")
    document.setAuthor("EigenFluid-SSS")
    document.setSubject("Original E2000, winner E2000, and 128^3 Dirichlet Laplacian")
    for page_png in page_pngs:
        image = ImageReader(str(page_png))
        width_px, height_px = image.getSize()
        width_pt = width_px * 72.0 / dpi
        height_pt = height_px * 72.0 / dpi
        document.setPageSize((width_pt, height_pt))
        document.drawImage(
            image,
            0,
            0,
            width=width_pt,
            height=height_pt,
            preserveAspectRatio=True,
            mask="auto",
        )
        document.showPage()
    document.save()


def learned_basis(
    label: str,
    checkpoint: Path,
    points: np.ndarray,
    geometry: np.ndarray,
    weight: float,
) -> dict[str, Any]:
    cfg, model, payload = base.load_model(checkpoint)
    raw_potential, raw_gradient = base.evaluate_raw_learned_basis(
        cfg, model, points, geometry, 4096
    )
    potential, gradient, transform, rayleigh, metrics = comparison.peel_with_transform(
        raw_potential,
        raw_gradient,
        weight,
        cfg.reorthogonalization_passes,
    )
    velocity = base.velocity_from_potential_gradient(gradient)
    del raw_potential, raw_gradient, gradient
    return {
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "config": cfg.__dict__,
        "cfg": cfg,
        "model": model,
        "potential": potential,
        "velocity": velocity,
        "transform": transform,
        "rayleigh": rayleigh,
        "metrics": metrics,
    }


def plane_data(
    grid: dict[str, Any], geometry: np.ndarray
) -> list[dict[str, Any]]:
    index = np.full(grid["shape"], -1, dtype=np.int64)
    index[grid["mask"]] = np.arange(len(grid["points"]))
    center = np.average(geometry[:, :3], axis=0, weights=geometry[:, 3] ** 3)
    plane_specs = (
        ("xy", int(np.argmin(abs(grid["coordinates"] - center[2])))),
        ("xz", int(np.argmin(abs(grid["coordinates"] - center[1])))),
        ("yz", int(np.argmin(abs(grid["coordinates"] - center[0])))),
    )
    result: list[dict[str, Any]] = []
    for plane, plane_index in plane_specs:
        points, horizontal_label, vertical_label, fixed_axis = comparison.slice_points(
            grid["coordinates"], plane, plane_index
        )
        if plane == "xy":
            mask = grid["mask"][plane_index, :, :]
            lap_indices = index[plane_index, :, :][mask]
        elif plane == "xz":
            mask = grid["mask"][:, plane_index, :]
            lap_indices = index[:, plane_index, :][mask]
        else:
            mask = grid["mask"][:, :, plane_index]
            lap_indices = index[:, :, plane_index][mask]
        result.append(
            {
                "plane": plane,
                "plane_index": plane_index,
                "fixed_axis": fixed_axis,
                "fixed_value": float(grid["coordinates"][plane_index]),
                "horizontal_label": horizontal_label,
                "vertical_label": vertical_label,
                "mask": mask,
                "valid_points": points[mask.ravel()].astype(np.float32),
                "lap_indices": lap_indices,
            }
        )
    return result


def learned_slice_velocity_bases(
    method: dict[str, Any],
    planes: list[dict[str, Any]],
    geometry: np.ndarray,
) -> list[np.ndarray]:
    values = []
    for plane in planes:
        _, raw_gradient = base.evaluate_raw_learned_basis(
            method["cfg"], method["model"], plane["valid_points"], geometry, 4096
        )
        values.append(base.velocity_from_potential_gradient(raw_gradient))
        del raw_gradient
    return values


def laplacian_first8_fields(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    mask: np.ndarray,
    spacing: float,
    z_index: int,
) -> tuple[list[np.ndarray], list[float], list[str]]:
    gradients: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for scalar_mode in range(3):
        volume = np.zeros(mask.shape, dtype=np.float64)
        volume[mask] = np.asarray(eigenvectors[:, scalar_mode], dtype=np.float64)
        derivative_z, derivative_y, derivative_x = np.gradient(
            volume, spacing, edge_order=2
        )
        gradients[scalar_mode] = (
            derivative_x[z_index],
            derivative_y[z_index],
            derivative_z[z_index],
        )
    slice_mask = mask[z_index]
    component_names = ("A12", "A13", "A23")
    fields: list[np.ndarray] = []
    rayleigh: list[float] = []
    labels: list[str] = []
    for mode in range(8):
        scalar_mode = mode // 3
        component = mode % 3
        derivative_x, derivative_y, derivative_z = gradients[scalar_mode]
        field = np.zeros((*slice_mask.shape, 3), dtype=np.float64)
        if component == 0:
            field[..., 0] = derivative_y
            field[..., 1] = -derivative_x
        elif component == 1:
            field[..., 0] = derivative_z
            field[..., 2] = -derivative_x
        else:
            field[..., 1] = derivative_z
            field[..., 2] = -derivative_y
        field[~slice_mask] = np.nan
        fields.append(field)
        rayleigh.append(float(eigenvalues[scalar_mode]))
        labels.append(component_names[component])
    return fields, rayleigh, labels


def field_from_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    field = np.full((*mask.shape, 3), np.nan, dtype=np.float64)
    field[mask] = values
    return field


def plot_mode_page(
    pdf: PdfPages,
    png: Path,
    coordinates: np.ndarray,
    slice_mask: np.ndarray,
    z_value: float,
    methods: list[dict[str, Any]],
    fields: list[list[np.ndarray]],
    values: list[list[float]],
    component_labels: list[list[str]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    figure, axes = plt.subplots(3, 8, figsize=(16.0, 6.6), constrained_layout=True)
    figure.get_layout_engine().set(rect=(0.0, 0.04, 1.0, 0.96))
    finite = []
    for row_fields in fields:
        for field in row_fields:
            speed = np.linalg.norm(field, axis=2)
            finite.append(speed[np.isfinite(speed)])
    shared_limit = max(float(np.percentile(np.concatenate(finite), 99.0)), 1.0e-12)
    norm = Normalize(vmin=0.0, vmax=shared_limit)
    image_handle = None
    for row, method in enumerate(methods):
        for column in range(8):
            axis = axes[row, column]
            field = fields[row][column]
            magnitude = np.linalg.norm(field, axis=2)
            image_handle = axis.imshow(
                np.ma.array(magnitude, mask=~slice_mask),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="Blues",
                norm=norm,
                interpolation="bilinear",
            )
            axis.contour(
                slice_mask.astype(float),
                levels=[0.5],
                colors=[NATURE_GREY],
                linewidths=0.42,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            horizontal, vertical = plot_utils.projected_components(field, "xy")
            plot_utils.draw_streamlines(
                axis, coordinates, horizontal, vertical, slice_mask
            )
            suffix = component_labels[row][column]
            axis.set_title(
                f"m{column + 1:02d}{suffix}\n$\\lambda$={values[row][column]:.2f}",
                color=NATURE_DARK,
                pad=2,
            )
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            if row == 2:
                axis.set_xlabel("x")
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(f"{method['label']}\ny")
            else:
                axis.set_yticklabels([])
            for spine in axis.spines.values():
                spine.set_color(NATURE_GREY)
                spine.set_linewidth(0.45)
    assert image_handle is not None
    colorbar = figure.colorbar(image_handle, ax=axes, location="right", shrink=0.88, pad=0.008)
    colorbar.set_label(r"$|\mathbf{u}_k|$ (shared scale)")
    figure.suptitle(
        "First 8 vector modes on the same unseen connected three-sphere geometry\n"
        f"128 x 128 xy slice at z={z_value:.3f} | learned: peeled Rayleigh quotient | "
        "Laplacian: discrete eigenvalue",
        color=NATURE_DARK,
        fontsize=12,
        fontweight="medium",
    )
    figure.text(
        0.5,
        0.006,
        "No cross-model sign, direction, or degenerate-block alignment. "
        "Laplacian modes are ordered scalar eigenfunctions expanded over A12, A13, A23.",
        ha="center",
        va="bottom",
        color=NATURE_GREY,
        fontsize=7.3,
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=260, facecolor="white", bbox_inches="tight")
    pdf.savefig(figure, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def plot_reconstruction_page(
    pdf: PdfPages,
    png: Path,
    case_index: int,
    target_seed: int,
    coordinates: np.ndarray,
    rows: list[dict[str, Any]],
    held_out_rl2: dict[str, float],
    sample_count: int,
    fit_count: int,
) -> dict[str, float]:
    titles = (
        "Target\nheld-out velocity rL2=0",
        f"Original E2000, K=64\nheld-out velocity rL2={held_out_rl2['original_e2000']:.3f}",
        f"Winner E2000, K=64\nheld-out velocity rL2={held_out_rl2['winner_e2000']:.3f}",
        f"Laplacian, K=64\nheld-out velocity rL2={held_out_rl2['laplacian']:.3f}",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    figure = plt.figure(figsize=(13.2, 8.7), constrained_layout=True, facecolor="white")
    figure.get_layout_engine().set(rect=(0.0, 0.045, 1.0, 0.955))
    grid = figure.add_gridspec(3, 5, width_ratios=(1, 1, 1, 1, 0.045))
    axes = np.empty((3, 4), dtype=object)
    squared_errors = {"original_e2000": 0.0, "winner_e2000": 0.0, "laplacian": 0.0}
    target_square = 0.0
    for row_index, row in enumerate(rows):
        magnitudes = [np.linalg.norm(field, axis=2) for field in row["fields"]]
        limit = plot_utils.robust_limit(magnitudes)
        norm = Normalize(vmin=0.0, vmax=limit)
        for column, (field, magnitude) in enumerate(zip(row["fields"], magnitudes)):
            axis = figure.add_subplot(grid[row_index, column])
            axes[row_index, column] = axis
            axis.imshow(
                np.ma.array(magnitude, mask=~row["mask"]),
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="Blues",
                norm=norm,
                interpolation="bilinear",
            )
            axis.contour(
                row["mask"].astype(float),
                levels=[0.5],
                colors=[NATURE_GREY],
                linewidths=0.5,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
            )
            horizontal, vertical = plot_utils.projected_components(field, row["plane"])
            plot_utils.draw_streamlines(
                axis, coordinates, horizontal, vertical, row["mask"]
            )
            axis.set_aspect("equal")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks((0.0, 0.5, 1.0))
            axis.set_yticks((0.0, 0.5, 1.0))
            if row_index == 0:
                axis.set_title(titles[column], color=NATURE_DARK, pad=4)
            if row_index == 2:
                axis.set_xlabel(row["horizontal_label"])
            else:
                axis.set_xticklabels([])
            if column == 0:
                axis.set_ylabel(
                    f"{row['plane']} ({row['fixed_axis']}={row['fixed_value']:.3f})\n"
                    f"{row['vertical_label']}"
                )
            else:
                axis.set_yticklabels([])
            for spine in axis.spines.values():
                spine.set_color(NATURE_GREY)
                spine.set_linewidth(0.5)
        colorbar_axis = figure.add_subplot(grid[row_index, 4])
        colorbar = figure.colorbar(axes[row_index, 3].images[0], cax=colorbar_axis)
        colorbar.set_label(r"$|\mathbf{u}|$")
        target = row["fields"][0][row["mask"]]
        target_square += float(np.sum(target**2))
        for method_index, method in enumerate(
            ("original_e2000", "winner_e2000", "laplacian"), start=1
        ):
            approximation = row["fields"][method_index][row["mask"]]
            squared_errors[method] += float(np.sum((approximation - target) ** 2))
    slice_rl2 = {
        method: float(np.sqrt(error / max(target_square, 1.0e-30)))
        for method, error in squared_errors.items()
    }
    figure.suptitle(
        f"Random target case {case_index} | seed={target_seed} | same geometry and boundary protocol\n"
        "Three orthogonal 128 x 128 vector slices; background is full 3D speed and "
        "streamlines show in-plane direction",
        color=NATURE_DARK,
        fontsize=12,
        fontweight="medium",
    )
    figure.text(
        0.5,
        0.006,
        f"Reconstruction uses all K=64 modes. Shared fit/test split: {fit_count}/{sample_count - fit_count} "
        f"of {sample_count} sampled 128^3 fluid nodes. Three-slice rL2: "
        f"original={slice_rl2['original_e2000']:.3f}, winner={slice_rl2['winner_e2000']:.3f}, "
        f"Laplacian={slice_rl2['laplacian']:.3f}.",
        ha="center",
        va="bottom",
        color=NATURE_GREY,
        fontsize=7.5,
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=260, facecolor="white", bbox_inches="tight")
    pdf.savefig(figure, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return slice_rl2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--sample-count", type=int, default=32768)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    png_root = output.with_suffix("")
    png_root.mkdir(parents=True, exist_ok=True)

    cfg = base.training.Config(**json.loads(BASE_CONFIG.read_text(encoding="utf-8")))
    geometry = base.training.geometry_from_token(cfg, GEOMETRY_TOKEN)
    grid = base.build_grid(cfg, 128, geometry)
    metadata = json.loads(
        (LAPLACIAN_ROOT / "laplacian_scalar_metadata_128.json").read_text(encoding="utf-8")
    )
    if not metadata.get("complete"):
        raise RuntimeError("128^3 Laplacian cache is incomplete")
    eigenvectors = np.load(
        LAPLACIAN_ROOT / "laplacian_scalar_eigenvectors_128.npy", mmap_mode="r"
    )
    eigenvalues = np.asarray(metadata["eigenvalues"], dtype=np.float64)

    rng = np.random.default_rng(SAMPLE_SEED)
    sample_indices = np.sort(
        rng.choice(len(grid["points"]), size=args.sample_count, replace=False)
    )
    points = grid["points"][sample_indices]
    fit_count = int(round(0.70 * args.sample_count))
    permutation = rng.permutation(args.sample_count)
    fit_indices = permutation[:fit_count]
    test_indices = permutation[fit_count:]
    weight = len(grid["points"]) * grid["spacing"] ** 3 / args.sample_count

    original = learned_basis(
        "Original E2000", ORIGINAL_CHECKPOINT, points, geometry, weight
    )
    winner = learned_basis(
        "Winner E2000", WINNER_CHECKPOINT, points, geometry, weight
    )
    methods = [original, winner]

    scalar_gradient = comparison.scalar_gradients_at_indices(
        eigenvectors, grid["mask"], grid["spacing"], sample_indices
    )
    lap_potential, lap_gradient, lap_rayleigh, lap_labels = base.expand_laplacian_basis(
        np.asarray(eigenvectors[sample_indices], dtype=np.float64),
        scalar_gradient,
        eigenvalues,
        64,
    )
    lap_velocity = base.velocity_from_potential_gradient(lap_gradient)
    del lap_gradient, scalar_gradient

    planes = plane_data(grid, geometry)
    slice_velocity_bases = {
        "original_e2000": learned_slice_velocity_bases(original, planes, geometry),
        "winner_e2000": learned_slice_velocity_bases(winner, planes, geometry),
    }
    xy_index = planes[0]["plane_index"]
    learned_mode_fields: list[list[np.ndarray]] = []
    for key, method in zip(("original_e2000", "winner_e2000"), methods):
        peeled_velocity = np.einsum(
            "nck,km->ncm",
            slice_velocity_bases[key][0],
            method["transform"][:, :8],
            optimize=True,
        )
        learned_mode_fields.append(
            [
                field_from_values(peeled_velocity[:, :, mode], planes[0]["mask"])
                for mode in range(8)
            ]
        )
    lap_mode_fields, lap_first8, lap_component_labels = laplacian_first8_fields(
        eigenvectors,
        eigenvalues,
        grid["mask"],
        grid["spacing"],
        xy_index,
    )

    pdf_methods = [
        {"label": original["label"]},
        {"label": winner["label"]},
        {"label": "Laplacian 128^3"},
    ]
    mode_fields = [*learned_mode_fields, lap_mode_fields]
    mode_values = [
        original["rayleigh"][:8].tolist(),
        winner["rayleigh"][:8].tolist(),
        lap_first8,
    ]
    component_labels = [[""] * 8, [""] * 8, [f" {name}" for name in lap_component_labels]]

    case_payloads: list[dict[str, Any]] = []
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"] = "3D Metaball E2000 Modes and Random Reconstruction"
        info["Author"] = "EigenFluid-SSS"
        info["Subject"] = "Original E2000, winner E2000, and 128^3 Dirichlet Laplacian"
        mode_png = png_root / "page_01_first8_modes.png"
        plot_mode_page(
            pdf,
            mode_png,
            grid["coordinates"],
            planes[0]["mask"],
            planes[0]["fixed_value"],
            pdf_methods,
            mode_fields,
            mode_values,
            component_labels,
        )

        for case_index, target_seed in enumerate(TARGET_SEEDS, start=1):
            target_potential, target_gradient, target_spec = base.target_potential_and_gradient(
                cfg, points, geometry, target_seed
            )
            target_velocity = base.velocity_from_potential_gradient(target_gradient)
            coefficients: dict[str, np.ndarray] = {}
            raw_coefficients: dict[str, np.ndarray] = {}
            metrics: dict[str, dict[str, Any]] = {}
            for key, method in zip(("original_e2000", "winner_e2000"), methods):
                metrics[key] = base.fit_reconstruction(
                    method["potential"],
                    method["velocity"],
                    target_potential,
                    target_velocity,
                    fit_indices,
                    test_indices,
                    64,
                    1.0e-8,
                )
                coefficients[key], _ = base.ridge_fit_coefficients(
                    method["potential"], target_potential, fit_indices, 64, 1.0e-8
                )
                raw_coefficients[key] = method["transform"] @ coefficients[key]
            metrics["laplacian"] = base.fit_reconstruction(
                lap_potential,
                lap_velocity,
                target_potential,
                target_velocity,
                fit_indices,
                test_indices,
                64,
                1.0e-8,
            )
            coefficients["laplacian"], _ = base.ridge_fit_coefficients(
                lap_potential, target_potential, fit_indices, 64, 1.0e-8
            )
            lap_full_velocity = comparison.reconstructed_laplacian_velocity(
                eigenvectors,
                coefficients["laplacian"],
                grid["mask"],
                grid["spacing"],
            )
            rows = []
            for plane_index, plane in enumerate(planes):
                _, target_slice_gradient, _ = base.target_potential_and_gradient(
                    cfg, plane["valid_points"], geometry, target_seed
                )
                target_slice = base.velocity_from_potential_gradient(target_slice_gradient)
                original_slice = np.einsum(
                    "nck,k->nc",
                    slice_velocity_bases["original_e2000"][plane_index],
                    raw_coefficients["original_e2000"],
                    optimize=True,
                )
                winner_slice = np.einsum(
                    "nck,k->nc",
                    slice_velocity_bases["winner_e2000"][plane_index],
                    raw_coefficients["winner_e2000"],
                    optimize=True,
                )
                lap_slice = lap_full_velocity[plane["lap_indices"]]
                rows.append(
                    {
                        **{key: value for key, value in plane.items() if key != "valid_points"},
                        "fields": [
                            field_from_values(target_slice, plane["mask"]),
                            field_from_values(original_slice, plane["mask"]),
                            field_from_values(winner_slice, plane["mask"]),
                            field_from_values(lap_slice, plane["mask"]),
                        ],
                    }
                )
            held_out_rl2 = {
                key: float(value["test_velocity_relative_l2"])
                for key, value in metrics.items()
            }
            case_png = png_root / f"page_{case_index + 1:02d}_random_case_{case_index}.png"
            slice_rl2 = plot_reconstruction_page(
                pdf,
                case_png,
                case_index,
                target_seed,
                grid["coordinates"],
                rows,
                held_out_rl2,
                args.sample_count,
                fit_count,
            )
            case_payloads.append(
                {
                    "case": case_index,
                    "target_seed": target_seed,
                    "target": target_spec,
                    "held_out_reconstruction": metrics,
                    "held_out_velocity_rl2": held_out_rl2,
                    "combined_three_slice_velocity_rl2": slice_rl2,
                    "png": str(case_png.resolve()),
                }
            )
            del lap_full_velocity, rows, target_potential, target_gradient, target_velocity

    assemble_reader_stable_pdf(
        output,
        [mode_png, *(Path(case["png"]) for case in case_payloads)],
    )

    payload = {
        "complete": True,
        "pdf": str(output),
        "geometry_token": GEOMETRY_TOKEN,
        "geometry": geometry.tolist(),
        "grid": {
            "resolution": 128,
            "fluid_nodes": len(grid["points"]),
            "spacing": grid["spacing"],
            "sample_nodes": args.sample_count,
            "fit_nodes": fit_count,
            "test_nodes": args.sample_count - fit_count,
        },
        "mode_page": {
            "slice_plane": "xy",
            "slice_value": planes[0]["fixed_value"],
            "original_e2000_rayleigh_first8": mode_values[0],
            "winner_e2000_rayleigh_first8": mode_values[1],
            "laplacian_eigenvalue_first8": mode_values[2],
            "laplacian_labels_first8": lap_labels[:8],
            "alignment": "none",
            "png": str(mode_png.resolve()),
        },
        "methods": {
            "original_e2000": {
                "checkpoint": original["checkpoint"],
                "checkpoint_epoch": original["checkpoint_epoch"],
                "peeled_metrics": original["metrics"],
            },
            "winner_e2000": {
                "checkpoint": winner["checkpoint"],
                "checkpoint_epoch": winner["checkpoint_epoch"],
                "peeled_metrics": winner["metrics"],
            },
            "laplacian": {
                "metadata": metadata,
                "modes": 64,
                "construction": "22 scalar eigenfunctions expanded over A12, A13, A23",
            },
        },
        "target_cases": case_payloads,
        "protocol": {
            "same_geometry": True,
            "same_fluid_mask": True,
            "same_homogeneous_dirichlet_boundary": True,
            "same_fit_test_split": True,
            "target_changes_only_by_seed": True,
            "reconstruction_modes": 64,
            "reporting_scope": "peeled learned bases; discrete Laplacian basis",
        },
    }
    metrics_path = output.with_name(output.stem + "_metrics.json")
    comparison.atomic_json(metrics_path, payload)
    csv_path = output.with_name(output.stem + "_reconstruction.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "target_seed",
                "method",
                "held_out_potential_rl2",
                "held_out_velocity_rl2",
                "three_slice_velocity_rl2",
            )
        )
        for case in case_payloads:
            for method in ("original_e2000", "winner_e2000", "laplacian"):
                reconstruction = case["held_out_reconstruction"][method]
                writer.writerow(
                    (
                        case["case"],
                        case["target_seed"],
                        method,
                        reconstruction["test_potential_relative_l2"],
                        reconstruction["test_velocity_relative_l2"],
                        case["combined_three_slice_velocity_rl2"][method],
                    )
                )
    print(json.dumps({"pdf": str(output), "cases": case_payloads}, indent=2))


if __name__ == "__main__":
    main()
