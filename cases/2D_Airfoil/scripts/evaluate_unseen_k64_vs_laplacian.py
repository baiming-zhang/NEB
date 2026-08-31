"""Evaluate the frozen K=64 peeled airfoil basis on one unseen NS case.

The learned basis is kept in its geometry-Rayleigh causal order and receives
the configured two-pass MGS peeling only.  No Ritz rotation, analytic-mode
alignment, degeneracy-block alignment, or raw-rank statistic is used.  The
K=64 comparison is a geometry-specific finite-difference Laplacian assembled
on the same airfoil fluid domain.  It uses hard Dirichlet streamfunction values
on the airfoil, inlet, and top/bottom far field, with the same unconstrained
(natural) outlet as the learned envelope.  Both reconstructions are fitted to
the same velocity samples and use the same temporal mean field.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.sparse
import scipy.sparse.linalg


CASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))
sys.path.insert(0, str(CASE_ROOT / "scripts"))

from eigenfluid.gen.airfoil2d.hardcoded_backend import (  # noqa: E402
    HardcodedModel,
    airfoil_area,
    body_numpy,
    cfg_namespace,
    continuous_parameters,
    fluid_points,
)
BLUE = "#2166AC"
RED = "#B2182B"
CHARCOAL = "#252525"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_model(checkpoint: Path) -> tuple[dict[str, Any], Any, HardcodedModel]:
    metadata = json.loads(
        checkpoint.with_suffix(".json").read_text(encoding="utf-8")
    )
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cfg = cfg_namespace(metadata["config"])
    if cfg.basis_objective != "streamfunction_dirichlet_rayleigh":
        raise ValueError("checkpoint is not a Dirichlet streamfunction basis")
    if int(cfg.modes) != 64:
        raise ValueError(f"expected K=64, got K={cfg.modes}")
    return metadata, cfg, HardcodedModel(cfg, arrays)


def raw_fields(
    model: HardcodedModel,
    cfg: SimpleNamespace,
    points: np.ndarray,
    beta: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count, modes = len(points), int(cfg.modes)
    psi = np.empty((count, modes), dtype=np.float32)
    gx = np.empty_like(psi)
    gy = np.empty_like(psi)
    u = np.empty_like(psi)
    v = np.empty_like(psi)
    omega = np.empty_like(psi)
    for network, field, _cache in model.forward_stream(
        cfg, points, beta, alpha, False
    ):
        index = int(network.mode_index)
        psi[:, index] = cp.asnumpy(field.value[:, 0])
        gx[:, index] = cp.asnumpy(field.grad[:, 0, 0])
        gy[:, index] = cp.asnumpy(field.grad[:, 0, 1])
        u[:, index] = gy[:, index]
        v[:, index] = -gx[:, index]
        omega[:, index] = cp.asnumpy(
            -(field.hess[:, 0, 0] + field.hess[:, 0, 2])
        )
    return psi, gx, gy, u, v, omega


def peeled_transform(
    cfg: SimpleNamespace,
    psi: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    domain_area: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    value = cp.asarray(psi)
    gradient_x = cp.asarray(gx)
    gradient_y = cp.asarray(gy)
    count, modes = value.shape
    weight = cp.asarray(np.float32(domain_area / count), dtype=cp.float32)
    candidate_mass = weight * cp.sum(value * value, axis=0)
    candidate_stiffness = weight * cp.sum(
        gradient_x * gradient_x + gradient_y * gradient_y, axis=0
    )
    sorting_rayleigh = cp.asnumpy(
        candidate_stiffness / candidate_mass
    ).astype(np.float64)
    order = np.argsort(sorting_rayleigh, kind="stable").tolist()
    passes = int(cfg.reorthogonalization_passes)
    basis = cp.empty_like(value)
    transform = cp.zeros((modes, modes), dtype=cp.float32)
    masses: list[float] = []
    for causal_rank, network_index in enumerate(order):
        residual = value[:, network_index].copy()
        coefficients = cp.zeros(modes, dtype=cp.float32)
        coefficients[network_index] = 1.0
        if causal_rank:
            previous = basis[:, :causal_rank]
            previous_transform = transform[:, :causal_rank]
            for _ in range(passes):
                increment = weight * (previous.T @ residual)
                residual -= previous @ increment
                coefficients -= previous_transform @ increment
        mass = weight * cp.dot(residual, residual)
        mass_value = float(mass.get())
        if not math.isfinite(mass_value) or mass_value <= float(cfg.mass_floor):
            raise FloatingPointError(
                f"unresolved peeled column {causal_rank + 1}: M={mass_value:.8e}"
            )
        inverse_norm = cp.reciprocal(cp.sqrt(mass))
        basis[:, causal_rank] = residual * inverse_norm
        transform[:, causal_rank] = coefficients * inverse_norm
        masses.append(mass_value)

    peeled_gx = gradient_x @ transform
    peeled_gy = gradient_y @ transform
    eigenvalue_diagonal = cp.asnumpy(
        weight
        * cp.sum(peeled_gx * peeled_gx + peeled_gy * peeled_gy, axis=0)
    ).astype(np.float64)
    gram = weight * (basis.T @ basis)
    diagnostics = {
        "peeled_rank": modes,
        "causal_order_1based": [int(index) + 1 for index in order],
        "sorting_rayleigh_by_network": sorting_rayleigh.tolist(),
        "peeled_rayleigh_diagonal": eigenvalue_diagonal.tolist(),
        "two_pass_mgs": passes == 2,
        "reorthogonalization_passes": passes,
        "peeled_orthogonality_max_abs": float(
            cp.max(cp.abs(gram - cp.eye(modes, dtype=cp.float32))).get()
        ),
        "minimum_pre_normalization_mass": float(min(masses)),
        "raw_rank_reported": False,
        "ritz_rotation_used": False,
        "degenerate_block_alignment_used": False,
    }
    return cp.asnumpy(transform), eigenvalue_diagonal, diagnostics


def velocity_projection(
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    u64, v64 = basis_u.astype(np.float64), basis_v.astype(np.float64)
    target_u64, target_v64 = target_u.astype(np.float64), target_v.astype(np.float64)
    gram = u64.T @ u64 + v64.T @ v64
    right = u64.T @ target_u64 + v64.T @ target_v64
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    threshold = max(float(values[-1]) * 1.0e-10, 1.0e-14)
    retained = values > threshold
    coefficients = vectors[:, retained] @ (
        (vectors[:, retained].T @ right) / values[retained]
    )
    fitted_u = basis_u @ coefficients
    fitted_v = basis_v @ coefficients
    residual = relative_velocity(target_u, target_v, fitted_u, fitted_v)
    return coefficients, int(np.count_nonzero(retained)), residual


def relative_velocity(
    target_u: np.ndarray,
    target_v: np.ndarray,
    estimate_u: np.ndarray,
    estimate_v: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum(
        (estimate_u[mask] - target_u[mask]) ** 2
        + (estimate_v[mask] - target_v[mask]) ** 2,
        dtype=np.float64,
    )
    denominator = np.sum(
        target_u[mask] ** 2 + target_v[mask] ** 2, dtype=np.float64
    )
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def relative_scalar(
    target: np.ndarray,
    estimate: np.ndarray,
    mask: np.ndarray | slice = slice(None),
) -> float:
    numerator = np.sum((estimate[mask] - target[mask]) ** 2, dtype=np.float64)
    denominator = np.sum(target[mask] ** 2, dtype=np.float64)
    return float(np.sqrt(numerator / max(denominator, 1.0e-30)))


def geometry_laplacian(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    modes: int,
    stride: int,
    seed: int,
) -> dict[str, Any]:
    """Return an airfoil-aware five-point streamfunction Laplacian basis.

    The coarse grid is an exact strided subset of the NS grid, so the airfoil
    mask is the same mask used by reconstruction.  Inlet and top/bottom nodes
    carry hard Dirichlet values.  Outlet nodes remain unknown and omission of
    the east boundary edge gives the natural zero-normal-derivative condition,
    matching the learned envelope's lack of an outlet factor.  Missing edges
    into the solid retain their diagonal penalty, imposing psi=0 on the airfoil.
    """
    total_started = time.perf_counter_ns()
    if stride < 1:
        raise ValueError("laplacian stride must be positive")
    if (len(x) - 1) % stride or (len(y) - 1) % stride:
        raise ValueError("laplacian stride must divide both grid interval counts")
    xc = np.asarray(x[::stride], dtype=np.float64)
    yc = np.asarray(y[::stride], dtype=np.float64)
    solid_coarse = np.asarray(solid[::stride, ::stride], dtype=bool)
    dx = float(xc[1] - xc[0])
    dy = float(yc[1] - yc[0])
    uniform_tolerance = 8.0 * np.finfo(np.float32).eps * max(
        float(np.max(np.abs(xc))), float(np.max(np.abs(yc))), 1.0
    )
    if not np.allclose(np.diff(xc), dx, rtol=0.0, atol=uniform_tolerance):
        raise ValueError("x grid is not uniform")
    if not np.allclose(np.diff(yc), dy, rtol=0.0, atol=uniform_tolerance):
        raise ValueError("y grid is not uniform")
    if not math.isclose(dx, dy, rel_tol=1.0e-5, abs_tol=1.0e-7):
        raise ValueError("five-point comparison requires equal x/y spacing")

    unknown = ~solid_coarse
    unknown = unknown.copy()
    unknown[:, 0] = False  # inlet Dirichlet
    unknown[0, :] = False  # lower far-field Dirichlet
    unknown[-1, :] = False  # upper far-field Dirichlet
    # x=xmax is deliberately retained: learned psi has no outlet envelope.
    ny, nx = unknown.shape
    node_index = np.full((ny, nx), -1, dtype=np.int32)
    node_index[unknown] = np.arange(np.count_nonzero(unknown), dtype=np.int32)
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    inverse_h2 = 1.0 / (dx * dx)
    for iy, ix in np.argwhere(unknown):
        row = int(node_index[iy, ix])
        diagonal = 0.0
        for jy, jx in (
            (iy - 1, ix),
            (iy + 1, ix),
            (iy, ix - 1),
            (iy, ix + 1),
        ):
            if not (0 <= jy < ny and 0 <= jx < nx):
                # The only retained outer boundary is the outlet.  Omitting
                # this edge is the variational/natural Neumann condition.
                continue
            diagonal += inverse_h2
            if unknown[jy, jx]:
                rows.append(row)
                columns.append(int(node_index[jy, jx]))
                data.append(-inverse_h2)
        rows.append(row)
        columns.append(row)
        data.append(diagonal)
    node_count = int(np.count_nonzero(unknown))
    operator = scipy.sparse.csr_matrix(
        (data, (rows, columns)), shape=(node_count, node_count)
    )
    assembly_seconds = (time.perf_counter_ns() - total_started) / 1.0e9
    if modes >= node_count - 1:
        raise ValueError(f"requested K={modes} for only {node_count} fluid nodes")
    rng = np.random.default_rng(seed)
    eigensolve_started = time.perf_counter_ns()
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        operator,
        k=modes,
        sigma=0.0,
        which="LM",
        v0=rng.standard_normal(node_count),
        tol=2.0e-8,
        maxiter=20000,
    )
    eigensolve_seconds = (time.perf_counter_ns() - eigensolve_started) / 1.0e9
    postprocess_started = time.perf_counter_ns()
    order = np.argsort(eigenvalues, kind="stable")
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    norms = np.sqrt(np.sum(eigenvectors * eigenvectors, axis=0) * dx * dy)
    eigenvectors /= norms[None, :]
    pivots = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[pivots, np.arange(modes)])
    signs[signs == 0.0] = 1.0
    eigenvectors *= signs[None, :]

    psi = np.zeros((ny, nx, modes), dtype=np.float32)
    psi[unknown] = eigenvectors.astype(np.float32)
    gx = np.gradient(psi, dx, axis=1, edge_order=2).astype(np.float32)
    gy = np.gradient(psi, dy, axis=0, edge_order=2).astype(np.float32)
    gx[:, -1] = 0.0  # exact natural outlet trace for the interpolated field
    omega = psi * eigenvalues.astype(np.float32)[None, None, :]
    mass_gram = (dx * dy) * (eigenvectors.T @ eigenvectors)
    residuals = np.linalg.norm(
        operator @ eigenvectors - eigenvectors * eigenvalues[None, :], axis=0
    ) / np.maximum(
        np.abs(eigenvalues) * np.linalg.norm(eigenvectors, axis=0), 1.0e-30
    )
    postprocess_seconds = (time.perf_counter_ns() - postprocess_started) / 1.0e9
    total_seconds = (time.perf_counter_ns() - total_started) / 1.0e9
    return {
        "x": xc,
        "y": yc,
        "solid": solid_coarse,
        "unknown": unknown,
        "psi": psi,
        "gx": gx,
        "gy": gy,
        "omega": omega,
        "eigenvalues": eigenvalues,
        "diagnostics": {
            "method": "geometry-aware five-point finite-difference Laplacian",
            "grid_stride": stride,
            "grid_shape": [ny, nx],
            "spacing": dx,
            "unknown_nodes": node_count,
            "same_airfoil_mask_as_ns_grid": True,
            "dirichlet_boundaries": ["airfoil", "inlet", "ymin", "ymax"],
            "natural_boundaries": ["outlet"],
            "mass_orthogonality_max_abs": float(
                np.max(np.abs(mass_gram - np.eye(modes)))
            ),
            "maximum_relative_eigen_residual": float(np.max(residuals)),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
            "timing_seconds": {
                "operator_assembly": assembly_seconds,
                "eigensolve": eigensolve_seconds,
                "postprocess": postprocess_seconds,
                "total_basis_generation": total_seconds,
            },
        },
    }


def geometry_laplacian_interpolators(
    basis: dict[str, Any],
) -> tuple[
    scipy.interpolate.RegularGridInterpolator,
    scipy.interpolate.RegularGridInterpolator,
    scipy.interpolate.RegularGridInterpolator,
    scipy.interpolate.RegularGridInterpolator,
]:
    axes = (basis["y"], basis["x"])
    options = {"bounds_error": True, "method": "linear"}
    return (
        scipy.interpolate.RegularGridInterpolator(axes, basis["psi"], **options),
        scipy.interpolate.RegularGridInterpolator(axes, basis["gx"], **options),
        scipy.interpolate.RegularGridInterpolator(axes, basis["gy"], **options),
        scipy.interpolate.RegularGridInterpolator(axes, basis["omega"], **options),
    )


def geometry_laplacian_fields(
    points: np.ndarray,
    interpolators: tuple[
        scipy.interpolate.RegularGridInterpolator,
        scipy.interpolate.RegularGridInterpolator,
        scipy.interpolate.RegularGridInterpolator,
        scipy.interpolate.RegularGridInterpolator,
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = np.column_stack((points[:, 1], points[:, 0]))
    psi = np.asarray(interpolators[0](query), dtype=np.float32)
    gx = np.asarray(interpolators[1](query), dtype=np.float32)
    gy = np.asarray(interpolators[2](query), dtype=np.float32)
    omega = np.asarray(interpolators[3](query), dtype=np.float32)
    return psi, gy, -gx, omega


def as_grid(values: np.ndarray, fluid: np.ndarray) -> np.ndarray:
    result = np.zeros(fluid.shape, dtype=np.float32)
    result[fluid] = values.astype(np.float32)
    return result


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def add_solid(axis: plt.Axes, solid: np.ndarray, extent: tuple[float, ...]) -> None:
    axis.imshow(
        np.ma.masked_where(~solid, solid.astype(float)),
        origin="lower",
        extent=extent,
        cmap=matplotlib.colors.ListedColormap([CHARCOAL]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        zorder=10,
    )


def plot_modes(
    learned: np.ndarray,
    laplacian: np.ndarray,
    learned_rayleigh: np.ndarray,
    laplacian_rayleigh: np.ndarray,
    solid: np.ndarray,
    extent: tuple[float, float, float, float],
    beta: float,
    alpha: float,
    output: Path,
) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(19.0, 8.0), constrained_layout=True)
    for row in range(4):
        for pair in range(4):
            rank = 4 * row + pair
            for offset, (field, label, value) in enumerate(
                (
                    (learned[rank], "Learned", learned_rayleigh[rank]),
                    (laplacian[rank], "Airfoil FD", laplacian_rayleigh[rank]),
                )
            ):
                axis = axes[row, 2 * pair + offset]
                normalized = field / max(float(np.max(np.abs(field))), 1.0e-12)
                axis.imshow(
                    np.ma.masked_where(solid, normalized),
                    origin="lower",
                    extent=extent,
                    cmap="RdBu_r",
                    vmin=-1.0,
                    vmax=1.0,
                    interpolation="nearest",
                )
                add_solid(axis, solid, extent)
                axis.set_title(
                    f"{label} {rank + 1:02d}\n$R_\\psi$={value:.3f}",
                    fontsize=8.0,
                    color=BLUE if label == "Learned" else RED,
                    pad=1.5,
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_aspect("equal")
    fig.suptitle(
        "Unseen airfoil: first 16 streamfunction modes (each normalized by max |psi|)\n"
        rf"$\beta={beta:.6f}$, $\alpha={alpha:.5f}^\circ$; same airfoil domain and open outlet",
        fontsize=12,
    )
    save_figure(fig, output)


def plot_reconstruction(
    x: np.ndarray,
    y: np.ndarray,
    solid: np.ndarray,
    actual_u: np.ndarray,
    actual_v: np.ndarray,
    actual_omega: np.ndarray,
    learned_u: np.ndarray,
    learned_v: np.ndarray,
    learned_omega: np.ndarray,
    lap_u: np.ndarray,
    lap_v: np.ndarray,
    lap_omega: np.ndarray,
    learned_metrics: dict[str, float],
    lap_metrics: dict[str, float],
    beta: float,
    alpha: float,
    frame: int,
    output: Path,
) -> None:
    extent = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    speed = np.sqrt(actual_u**2 + actual_v**2)
    learned_speed = np.sqrt(learned_u**2 + learned_v**2)
    lap_speed = np.sqrt(lap_u**2 + lap_v**2)
    learned_speed_error = np.sqrt((learned_u - actual_u) ** 2 + (learned_v - actual_v) ** 2)
    lap_speed_error = np.sqrt((lap_u - actual_u) ** 2 + (lap_v - actual_v) ** 2)
    learned_omega_error = np.abs(learned_omega - actual_omega)
    lap_omega_error = np.abs(lap_omega - actual_omega)
    fluid = ~solid
    speed_limit = max(float(np.percentile(speed[fluid], 99.5)), 1.0e-8)
    omega_limit = max(float(np.percentile(np.abs(actual_omega[fluid]), 99.5)), 1.0e-8)
    speed_error_limit = max(
        float(np.percentile(np.concatenate((learned_speed_error[fluid], lap_speed_error[fluid])), 99.5)),
        1.0e-8,
    )
    omega_error_limit = max(
        float(np.percentile(np.concatenate((learned_omega_error[fluid], lap_omega_error[fluid])), 99.5)),
        1.0e-8,
    )
    rows = (
        (
            (speed, "GT speed", "viridis", 0.0, speed_limit),
            (learned_speed, f"Learned K64\nvelocity L2={100*learned_metrics['velocity_error']:.2f}%", "viridis", 0.0, speed_limit),
            (lap_speed, f"Airfoil FD K64\nvelocity L2={100*lap_metrics['velocity_error']:.2f}%", "viridis", 0.0, speed_limit),
            (learned_speed_error, "|Learned − GT|", "magma", 0.0, speed_error_limit),
            (lap_speed_error, "|Airfoil FD - GT|", "magma", 0.0, speed_error_limit),
        ),
        (
            (actual_omega, "GT vorticity", "RdBu_r", -omega_limit, omega_limit),
            (learned_omega, f"Learned K64\nvorticity L2={100*learned_metrics['vorticity_error']:.2f}%", "RdBu_r", -omega_limit, omega_limit),
            (lap_omega, f"Airfoil FD K64\nvorticity L2={100*lap_metrics['vorticity_error']:.2f}%", "RdBu_r", -omega_limit, omega_limit),
            (learned_omega_error, "|Learned − GT|", "magma", 0.0, omega_error_limit),
            (lap_omega_error, "|Airfoil FD - GT|", "magma", 0.0, omega_error_limit),
        ),
    )
    fig, axes = plt.subplots(2, 5, figsize=(19.0, 6.1), constrained_layout=True)
    for row_index, row in enumerate(rows):
        for column, (field, title, cmap, lower, upper) in enumerate(row):
            axis = axes[row_index, column]
            image = axis.imshow(
                np.ma.masked_where(solid, field),
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=lower,
                vmax=upper,
                interpolation="bilinear",
            )
            add_solid(axis, solid, extent)
            axis.set_title(title, fontsize=9.0, pad=2.0)
            axis.set_xlim(1.5, 9.0)
            axis.set_ylim(-2.0, 2.0)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.025, pad=0.012)
    fig.suptitle(
        rf"Unseen Re=1500 airfoil reconstruction, frame {frame}: "
        rf"$\beta={beta:.6f}$, $\alpha={alpha:.5f}^\circ$",
        fontsize=12,
    )
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=-1)
    parser.add_argument("--peeling-points", type=int, default=32768)
    parser.add_argument("--projection-points", type=int, default=65536)
    parser.add_argument("--field-batch", type=int, default=2048)
    parser.add_argument(
        "--laplacian-stride",
        type=int,
        default=2,
        help="FD grid stride relative to the NS grid (2 gives 501x401 nodes)",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_metadata, cfg, model = load_model(args.checkpoint)
    with np.load(args.gt, allow_pickle=False) as archive:
        x = archive["x"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        solid = archive["solid"].astype(bool)
        gt_metadata = json.loads(str(archive["metadata"]))
        frame = args.frame if args.frame >= 0 else len(archive["steps"]) + args.frame
        actual_u = archive["u"][frame].astype(np.float32)
        actual_v = archive["v"][frame].astype(np.float32)
        actual_omega = archive["vorticity"][frame].astype(np.float32)
        mean_u = np.mean(archive["u"], axis=0, dtype=np.float64).astype(np.float32)
        mean_v = np.mean(archive["v"], axis=0, dtype=np.float64).astype(np.float32)
        mean_omega = np.mean(archive["vorticity"], axis=0, dtype=np.float64).astype(np.float32)
    beta = float(gt_metadata["beta"])
    alpha = float(gt_metadata["alpha_deg"])
    if not (0.0 <= beta <= 1.0 and cfg.alpha_min_deg <= alpha <= cfg.alpha_max_deg):
        raise ValueError("unseen case must lie inside the trained geometry range")
    validation_pairs = {
        (float(b), float(a))
        for b in np.linspace(0.0, 1.0, cfg.validation_beta_samples)
        for a in np.linspace(cfg.alpha_min_deg, cfg.alpha_max_deg, cfg.validation_alpha_samples)
    }
    if any(abs(beta - b) < 1.0e-12 and abs(alpha - a) < 1.0e-12 for b, a in validation_pairs):
        raise ValueError("selected case is part of the fixed validation grid")

    print("[1/6] geometry-specific dynamic sorting and two-pass MGS", flush=True)
    integration_points = fluid_points(
        cfg, args.peeling_points, beta, alpha, cfg.seed + 2_300_001
    )
    psi, gx, gy, _u, _v, _omega = raw_fields(
        model, cfg, integration_points, beta, alpha
    )
    area = (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin) - airfoil_area(cfg, beta)
    transform, learned_rayleigh, peeling = peeled_transform(cfg, psi, gx, gy, area)
    del psi, gx, gy, _u, _v, _omega

    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    target_u = actual_u[fluid] - mean_u[fluid]
    target_v = actual_v[fluid] - mean_v[fluid]
    target_omega = actual_omega[fluid] - mean_omega[fluid]
    subset = np.linspace(
        0, len(points) - 1, min(args.projection_points, len(points)), dtype=np.int64
    )

    print("[2/6] learned K64 velocity projection", flush=True)
    _psi, _gx, _gy, raw_u, raw_v, _raw_omega = raw_fields(
        model, cfg, points[subset], beta, alpha
    )
    learned_basis_u = raw_u @ transform
    learned_basis_v = raw_v @ transform
    learned_coefficients, learned_projection_rank, learned_projection_residual = velocity_projection(
        learned_basis_u, learned_basis_v, target_u[subset], target_v[subset]
    )
    learned_raw_coefficients = transform @ learned_coefficients
    del _psi, _gx, _gy, raw_u, raw_v, _raw_omega, learned_basis_u, learned_basis_v

    print("[3/6] same-domain airfoil finite-difference Laplacian K64", flush=True)
    lap_basis = geometry_laplacian(
        x,
        y,
        solid,
        int(cfg.modes),
        args.laplacian_stride,
        cfg.seed + 2_400_001,
    )
    lap_interpolators = geometry_laplacian_interpolators(lap_basis)
    _lap_psi, lap_basis_u, lap_basis_v, _lap_omega = geometry_laplacian_fields(
        points[subset], lap_interpolators
    )
    lap_coefficients, lap_projection_rank, lap_projection_residual = velocity_projection(
        lap_basis_u, lap_basis_v, target_u[subset], target_v[subset]
    )
    del _lap_psi, lap_basis_u, lap_basis_v, _lap_omega

    print("[4/6] full-grid reconstructions and first 16 modes", flush=True)
    learned_fluctuation_u = np.empty(len(points), dtype=np.float32)
    learned_fluctuation_v = np.empty(len(points), dtype=np.float32)
    learned_fluctuation_omega = np.empty(len(points), dtype=np.float32)
    lap_fluctuation_u = np.empty(len(points), dtype=np.float32)
    lap_fluctuation_v = np.empty(len(points), dtype=np.float32)
    lap_fluctuation_omega = np.empty(len(points), dtype=np.float32)
    learned_modes = np.empty((16, len(points)), dtype=np.float32)
    lap_modes_psi = np.empty((16, len(points)), dtype=np.float32)
    for start in range(0, len(points), args.field_batch):
        stop = min(start + args.field_batch, len(points))
        raw_psi, _gx, _gy, raw_u, raw_v, raw_omega = raw_fields(
            model, cfg, points[start:stop], beta, alpha
        )
        learned_fluctuation_u[start:stop] = raw_u @ learned_raw_coefficients
        learned_fluctuation_v[start:stop] = raw_v @ learned_raw_coefficients
        learned_fluctuation_omega[start:stop] = raw_omega @ learned_raw_coefficients
        learned_modes[:, start:stop] = (raw_psi @ transform[:, :16]).T
        lap_psi_batch, lap_u_batch, lap_v_batch, lap_omega_batch = (
            geometry_laplacian_fields(points[start:stop], lap_interpolators)
        )
        lap_fluctuation_u[start:stop] = lap_u_batch @ lap_coefficients
        lap_fluctuation_v[start:stop] = lap_v_batch @ lap_coefficients
        lap_fluctuation_omega[start:stop] = lap_omega_batch @ lap_coefficients
        lap_modes_psi[:, start:stop] = lap_psi_batch[:, :16].T

    learned_u = mean_u[fluid] + learned_fluctuation_u
    learned_v = mean_v[fluid] + learned_fluctuation_v
    learned_omega = mean_omega[fluid] + learned_fluctuation_omega
    lap_u = mean_u[fluid] + lap_fluctuation_u
    lap_v = mean_v[fluid] + lap_fluctuation_v
    lap_omega = mean_omega[fluid] + lap_fluctuation_omega
    xb, yb = body_numpy(cfg, points, alpha)
    wake = (
        (xb >= cfg.chord)
        & (xb <= 4.0 * cfg.chord)
        & (np.abs(yb) <= 0.75 * cfg.chord)
    )

    def metrics(u: np.ndarray, v: np.ndarray, omega: np.ndarray) -> dict[str, float]:
        return {
            "velocity_error": relative_velocity(actual_u[fluid], actual_v[fluid], u, v),
            "vorticity_error": relative_scalar(actual_omega[fluid], omega),
            "fluctuation_velocity_error": relative_velocity(target_u, target_v, u - mean_u[fluid], v - mean_v[fluid]),
            "fluctuation_vorticity_error": relative_scalar(target_omega, omega - mean_omega[fluid]),
            "wake_velocity_error": relative_velocity(actual_u[fluid], actual_v[fluid], u, v, wake),
            "wake_vorticity_error": relative_scalar(actual_omega[fluid], omega, wake),
        }

    learned_metrics = metrics(learned_u, learned_v, learned_omega)
    lap_metrics = metrics(lap_u, lap_v, lap_omega)

    print("[5/6] Nature-style red-blue figures", flush=True)
    learned_mode_grids = np.stack([as_grid(item, fluid) for item in learned_modes])
    lap_mode_grids = np.stack([as_grid(item, fluid) for item in lap_modes_psi])
    bounds = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    extent = bounds
    plot_modes(
        learned_mode_grids[:, ::2, ::2],
        lap_mode_grids[:, ::2, ::2],
        learned_rayleigh[:16],
        lap_basis["eigenvalues"][:16],
        solid[::2, ::2],
        extent,
        beta,
        alpha,
        args.output_dir / "unseen_first16_learned_vs_laplacian",
    )
    learned_u_grid, learned_v_grid, learned_omega_grid = (
        as_grid(learned_u, fluid),
        as_grid(learned_v, fluid),
        as_grid(learned_omega, fluid),
    )
    lap_u_grid, lap_v_grid, lap_omega_grid = (
        as_grid(lap_u, fluid),
        as_grid(lap_v, fluid),
        as_grid(lap_omega, fluid),
    )
    plot_reconstruction(
        x,
        y,
        solid,
        actual_u,
        actual_v,
        actual_omega,
        learned_u_grid,
        learned_v_grid,
        learned_omega_grid,
        lap_u_grid,
        lap_v_grid,
        lap_omega_grid,
        learned_metrics,
        lap_metrics,
        beta,
        alpha,
        frame,
        args.output_dir / "unseen_reconstruction_learned_vs_laplacian",
    )

    print("[6/6] metrics and field archive", flush=True)
    training_pairs = [continuous_parameters(cfg, epoch) for epoch in range(1, cfg.epochs + 1)]
    closest_training_distance = min(
        math.hypot(beta - b, (alpha - a) / (cfg.alpha_max_deg - cfg.alpha_min_deg))
        for b, a in training_pairs
    )
    summary = {
        "case": {
            "beta": beta,
            "alpha_deg": alpha,
            "reynolds": float(gt_metadata["solver_config"]["reynolds"]),
            "frame": frame,
            "gt": str(args.gt.resolve()),
            "not_in_fixed_validation_grid": True,
            "closest_normalized_training_geometry_distance": closest_training_distance,
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "epoch": int(checkpoint_metadata["epoch"]),
            "modes": int(cfg.modes),
        },
        "evaluation_protocol": {
            "peeling_points": args.peeling_points,
            "projection_points": len(subset),
            "same_projection_points_for_both_methods": True,
            "coefficient_fit": "velocity L2 of temporal fluctuation",
            "mean_field": "same 41-frame independent NS temporal mean",
            "learned_basis": "geometry-Rayleigh dynamic order plus two-pass MGS",
            "laplacian_basis": (
                "lowest 64 geometry-aware five-point finite-difference "
                "Dirichlet streamfunction modes"
            ),
            "same_fluid_domain": True,
            "same_airfoil_geometry": True,
            "same_dirichlet_trace_boundaries": [
                "airfoil",
                "inlet",
                "ymin",
                "ymax",
            ],
            "same_natural_outlet": True,
            "boundary_discretization_note": (
                "learned uses a smooth double-zero hard envelope; FD uses the "
                "canonical discrete Dirichlet trace on the same boundaries"
            ),
            "ritz_rotation_used": False,
            "analytic_alignment_used_for_learned_basis": False,
            "degenerate_block_alignment_used": False,
            "raw_rank_reported": False,
        },
        "peeled_basis": peeling,
        "learned": {
            "projection_rank": learned_projection_rank,
            "projection_relative_residual": learned_projection_residual,
            **learned_metrics,
        },
        "laplacian": {
            "projection_rank": lap_projection_rank,
            "projection_relative_residual": lap_projection_residual,
            "eigensolve": lap_basis["diagnostics"],
            **lap_metrics,
        },
        "error_ratio_laplacian_over_learned": {
            key: float(lap_metrics[key] / max(learned_metrics[key], 1.0e-30))
            for key in learned_metrics
        },
        "figures": {
            "first16_png": str((args.output_dir / "unseen_first16_learned_vs_laplacian.png").resolve()),
            "first16_pdf": str((args.output_dir / "unseen_first16_learned_vs_laplacian.pdf").resolve()),
            "reconstruction_png": str((args.output_dir / "unseen_reconstruction_learned_vs_laplacian.png").resolve()),
            "reconstruction_pdf": str((args.output_dir / "unseen_reconstruction_learned_vs_laplacian.pdf").resolve()),
        },
    }
    write_json(args.output_dir / "unseen_evaluation_summary.json", summary)
    np.savez_compressed(
        args.output_dir / "unseen_reconstruction_fields.npz",
        x=x,
        y=y,
        solid=solid,
        actual_u=actual_u,
        actual_v=actual_v,
        actual_omega=actual_omega,
        learned_u=learned_u_grid,
        learned_v=learned_v_grid,
        learned_omega=learned_omega_grid,
        laplacian_u=lap_u_grid,
        laplacian_v=lap_v_grid,
        laplacian_omega=lap_omega_grid,
        learned_coefficients=learned_coefficients,
        laplacian_coefficients=lap_coefficients,
        learned_causal_order_1based=np.asarray(peeling["causal_order_1based"], dtype=np.int32),
        learned_first16_rayleigh=learned_rayleigh[:16],
        laplacian_eigenvalues=lap_basis["eigenvalues"],
        laplacian_first16_rayleigh=lap_basis["eigenvalues"][:16],
        laplacian_grid_stride=np.asarray(args.laplacian_stride, dtype=np.int32),
    )
    print(json.dumps({"learned": learned_metrics, "laplacian": lap_metrics}, indent=2))


if __name__ == "__main__":
    main()
