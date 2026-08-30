"""Build continuous (beta, alpha) GT, reconstructions, movie, and six bases."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CASE_ROOT.parents[1]
sys.path.insert(0, str(CASE_ROOT / "src"))

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import (  # noqa: E402
    body_numpy,
    inside_airfoil,
)
from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (  # noqa: E402
    compute_certified_stokes_basis,
    compute_ritz_basis,
    load_hardcoded_model,
    project_ritz_scalar,
    project_ritz_velocity,
    raw_fields,
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def path_parameter(value: float, zero_tolerance: float = 1.0e-12) -> float:
    """Canonicalize analytically zero values produced by trigonometric paths."""
    return 0.0 if abs(value) < zero_tolerance else float(value)


def numeric_option(name: str, value: float) -> str:
    """Serialize a numeric CLI option without argparse ambiguity."""
    return f"--{name}={value:.17g}"


def validate_gt_cache(
    cache_path: Path,
    airfoil_settings: dict,
    solver_settings: dict,
    beta: float,
    alpha: float,
) -> dict:
    """Validate only physics-defining metadata before reusing a GT cache."""
    required_members = {"metadata", "x", "y", "solid", "u", "v", "vorticity"}
    physical_geometry_keys = (
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "chord",
        "quarter_chord_x",
        "quarter_chord_y",
        "thickness_min",
        "thickness_max",
        "geometry_mode",
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        missing = sorted(required_members.difference(archive.files))
        if missing:
            raise ValueError(f"GT cache is missing members: {missing}")
        metadata = json.loads(str(archive["metadata"]))
    if abs(float(metadata["beta"]) - beta) > 1.0e-10:
        raise ValueError("GT beta does not match the requested parameter path")
    if abs(float(metadata["alpha_deg"]) - alpha) > 1.0e-10:
        raise ValueError("GT alpha does not match the requested parameter path")
    cached_geometry = metadata["geometry"]
    for key in physical_geometry_keys:
        if cached_geometry.get(key) != airfoil_settings.get(key):
            raise ValueError(f"GT geometry mismatch for {key}")
    if metadata.get("solver_config") != solver_settings:
        raise ValueError("GT solver configuration mismatch")
    diagnostics = metadata.get("separation_diagnostics", {})
    if (
        solver_settings.get("require_separated_gt", False)
        and not diagnostics.get("separated_gt_quality_pass", False)
    ):
        raise ValueError("GT cache did not pass the separated-flow quality gate")
    return {
        "path": str(cache_path.resolve()),
        "bytes": cache_path.stat().st_size,
        "beta": beta,
        "alpha_deg": alpha,
        "physics_validation": "passed",
        "full_array_decode": "deferred to reconstruction",
    }


def encode_mp4(frames: Path, output: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame_%03d.png"),
            "-vf",
            "scale=min(1920\\,iw):min(1080\\,ih):force_original_aspect_ratio=decrease,"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def geometry(axis, x: np.ndarray, y: np.ndarray, solid: np.ndarray) -> None:
    xgrid, ygrid = np.meshgrid(x, y)
    axis.contourf(
        xgrid,
        ygrid,
        solid.astype(float),
        levels=(0.5, 1.5),
        colors=("#24272d",),
        zorder=20,
    )
    axis.set_xlim(float(x[0]), float(x[-1]))
    axis.set_ylim(float(y[0]), float(y[-1]))
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])


def velocity_panel(
    axis,
    x,
    y,
    u,
    v,
    solid,
    title,
    limit,
    cmap="viridis",
) -> None:
    speed = np.ma.array(np.hypot(u, v), mask=solid)
    axis.pcolormesh(
        x, y, speed, shading="auto", cmap=cmap, vmin=0.0, vmax=limit
    )
    stride = max(1, len(x) // 140)
    axis.streamplot(
        x[::stride],
        y[::stride],
        np.ma.array(u, mask=solid)[::stride, ::stride],
        np.ma.array(v, mask=solid)[::stride, ::stride],
        color="white",
        density=1.1,
        linewidth=0.35,
        arrowsize=0.45,
    )
    geometry(axis, x, y, solid)
    axis.set_title(title, fontsize=9, pad=2)


def vorticity_panel(
    axis, x, y, omega, solid, title, limit, absolute=False
) -> None:
    masked = np.ma.array(np.abs(omega) if absolute else omega, mask=solid)
    axis.pcolormesh(
        x,
        y,
        masked,
        shading="auto",
        cmap="magma" if absolute else "RdBu_r",
        vmin=0.0 if absolute else -limit,
        vmax=limit,
    )
    geometry(axis, x, y, solid)
    axis.set_title(title, fontsize=9, pad=2)


def render_frame(
    sample: dict[str, np.ndarray],
    limits: dict[str, float],
    output: Path,
    variant: str = "velocity",
) -> None:
    x, y, solid = sample["x"], sample["y"], sample["solid"].astype(bool)
    beta, alpha = float(sample["beta"]), float(sample["alpha_deg"])
    if variant == "vorticity":
        field_prefix = "vorticity_opt"
        metric_prefix = "vorticity_opt_"
        reconstruction_label = "PINN vorticity-optimal reconstruction"
        limit_prefix = "vorticity_opt_"
    else:
        field_prefix = "pinn"
        metric_prefix = ""
        reconstruction_label = "PINN velocity-optimal reconstruction"
        limit_prefix = ""
    predicted_u = sample[f"{field_prefix}_u"]
    predicted_v = sample[f"{field_prefix}_v"]
    predicted_omega = sample[f"{field_prefix}_omega"]
    du = predicted_u - sample["actual_u"]
    dv = predicted_v - sample["actual_v"]
    domega = predicted_omega - sample["actual_omega"]
    figure, axes = plt.subplots(
        3, 2, figsize=(15.5, 8.8), constrained_layout=True
    )
    velocity_panel(
        axes[0, 0],
        x,
        y,
        sample["actual_u"],
        sample["actual_v"],
        solid,
        "GT velocity magnitude + streamlines",
        limits["speed"],
    )
    vorticity_panel(
        axes[0, 1],
        x,
        y,
        sample["actual_omega"],
        solid,
        "GT vorticity",
        limits["vorticity"],
    )
    velocity_panel(
        axes[1, 0],
        x,
        y,
        predicted_u,
        predicted_v,
        solid,
        f"{reconstruction_label}, velocity error="
        f"{100*float(sample[metric_prefix + 'velocity_error']):.2f}%",
        limits["speed"],
    )
    vorticity_panel(
        axes[1, 1],
        x,
        y,
        predicted_omega,
        solid,
        f"PINN vorticity, error="
        f"{100*float(sample[metric_prefix + 'vorticity_error']):.2f}%",
        limits["vorticity"],
    )
    velocity_panel(
        axes[2, 0],
        x,
        y,
        du,
        dv,
        solid,
        "Absolute velocity-vector error",
        limits[limit_prefix + "velocity_error"],
        cmap="magma",
    )
    vorticity_panel(
        axes[2, 1],
        x,
        y,
        domega,
        solid,
        "Absolute vorticity error",
        limits[limit_prefix + "vorticity_error"],
        absolute=True,
    )
    figure.suptitle(
        rf"Airfoil case: $\beta={beta:.3f}$, "
        rf"$\alpha={alpha:+.2f}^\circ$",
        fontsize=13,
    )
    figure.savefig(output, dpi=130, facecolor="white")
    plt.close(figure)


def reconstruct_case(
    checkpoint: Path,
    cache_path: Path,
    sample_path: Path,
    phase: float,
    quadrature: int,
    batch: int,
    velocity_only: bool = False,
) -> None:
    cfg, model, _ = load_hardcoded_model(checkpoint)
    cache = np.load(cache_path)
    metadata = json.loads(str(cache["metadata"]))
    beta = float(metadata["beta"])
    alpha = float(metadata["alpha_deg"])
    x = cache["x"].astype(np.float64)
    y = cache["y"].astype(np.float64)
    solid = cache["solid"].astype(bool)
    fluid = ~solid
    xgrid, ygrid = np.meshgrid(x, y)
    points = np.column_stack((xgrid[fluid], ygrid[fluid])).astype(np.float32)
    certified_stokes = (
        str(getattr(cfg, "basis_objective", "velocity_rayleigh"))
        == "stokes_certified"
    )
    if certified_stokes:
        ritz = compute_certified_stokes_basis(
            model,
            cfg,
            beta,
            alpha,
            minimum_quadrature=max(
                quadrature, int(getattr(cfg, "stokes_min_quadrature", quadrature))
            ),
            maximum_quadrature=max(
                quadrature, int(getattr(cfg, "stokes_max_quadrature", quadrature))
            ),
        )
        if int(ritz["certified_count"]) == 0:
            raise RuntimeError(
                "no Stokes mode passed independent-grid certification; "
                "reconstruction is intentionally refused"
            )
    else:
        ritz = compute_ritz_basis(
            model, cfg, beta, alpha, quadrature=quadrature
        )
    raw_u, raw_v, raw_omega = raw_fields(
        model, cfg, points, beta, alpha, batch
    )
    actual_u_all = cache["u"].astype(np.float32)
    actual_v_all = cache["v"].astype(np.float32)
    actual_omega_all = cache["vorticity"].astype(np.float32)
    frame = int(round((phase % 1.0) * (len(actual_u_all) - 1)))
    mean_u = actual_u_all[:, fluid].mean(axis=0)
    mean_v = actual_v_all[:, fluid].mean(axis=0)
    mean_omega = actual_omega_all[:, fluid].mean(axis=0)
    actual_u = actual_u_all[frame, fluid]
    actual_v = actual_v_all[frame, fluid]
    actual_omega = actual_omega_all[frame, fluid]
    modal, raw_modal, projection_diagnostics = project_ritz_velocity(
        raw_u,
        raw_v,
        actual_u - mean_u,
        actual_v - mean_v,
        ritz["coefficients"],
    )
    projection_ritz_orth_error = float(
        projection_diagnostics["orthogonality_error"]
    )
    predicted_u = mean_u + raw_u @ raw_modal
    predicted_v = mean_v + raw_v @ raw_modal
    predicted_omega = mean_omega + raw_omega @ raw_modal
    if not velocity_only:
        (
            vorticity_modal,
            vorticity_raw_modal,
            vorticity_projection_rank,
            vorticity_projection_relative_residual,
        ) = project_ritz_scalar(
            raw_omega,
            actual_omega - mean_omega,
            ritz["coefficients"],
        )
        vorticity_opt_u = mean_u + raw_u @ vorticity_raw_modal
        vorticity_opt_v = mean_v + raw_v @ vorticity_raw_modal
        vorticity_opt_omega = mean_omega + raw_omega @ vorticity_raw_modal

    def velocity_relative_error(
        target_u: np.ndarray,
        target_v: np.ndarray,
        estimate_u: np.ndarray,
        estimate_v: np.ndarray,
        mask: np.ndarray | slice = slice(None),
    ) -> float:
        numerator = np.sum(
            (estimate_u[mask] - target_u[mask]) ** 2
            + (estimate_v[mask] - target_v[mask]) ** 2
        )
        denominator = np.sum(target_u[mask] ** 2 + target_v[mask] ** 2)
        return float(np.sqrt(numerator / max(denominator, 1.0e-30)))

    def scalar_relative_error(
        target: np.ndarray,
        estimate: np.ndarray,
        mask: np.ndarray | slice = slice(None),
    ) -> float:
        numerator = np.sum((estimate[mask] - target[mask]) ** 2)
        denominator = np.sum(target[mask] ** 2)
        return float(np.sqrt(numerator / max(denominator, 1.0e-30)))

    fluctuation_u = actual_u - mean_u
    fluctuation_v = actual_v - mean_v
    fluctuation_omega = actual_omega - mean_omega
    predicted_fluctuation_u = predicted_u - mean_u
    predicted_fluctuation_v = predicted_v - mean_v
    predicted_fluctuation_omega = predicted_omega - mean_omega
    velocity_error = velocity_relative_error(
        actual_u, actual_v, predicted_u, predicted_v
    )
    vorticity_error = scalar_relative_error(actual_omega, predicted_omega)
    fluctuation_velocity_error = velocity_relative_error(
        fluctuation_u,
        fluctuation_v,
        predicted_fluctuation_u,
        predicted_fluctuation_v,
    )
    fluctuation_vorticity_error = scalar_relative_error(
        fluctuation_omega, predicted_fluctuation_omega
    )
    xb, yb = body_numpy(cfg, points.astype(np.float64), alpha)
    wake = (
        (xb >= cfg.chord)
        & (xb <= cfg.chord * (1.0 + cfg.wake_length_chords))
        & (np.abs(yb) <= cfg.chord * cfg.wake_half_height_chords)
    )
    if not np.any(wake):
        raise RuntimeError("wake validation mask is empty")
    wake_velocity_error = velocity_relative_error(
        actual_u, actual_v, predicted_u, predicted_v, wake
    )
    wake_vorticity_error = scalar_relative_error(
        actual_omega, predicted_omega, wake
    )
    wake_fluctuation_velocity_error = velocity_relative_error(
        fluctuation_u,
        fluctuation_v,
        predicted_fluctuation_u,
        predicted_fluctuation_v,
        wake,
    )
    wake_fluctuation_vorticity_error = scalar_relative_error(
        fluctuation_omega, predicted_fluctuation_omega, wake
    )
    if not velocity_only:
        vorticity_opt_velocity_error = velocity_relative_error(
            actual_u, actual_v, vorticity_opt_u, vorticity_opt_v
        )
        vorticity_opt_vorticity_error = scalar_relative_error(
            actual_omega, vorticity_opt_omega
        )
        vorticity_opt_fluctuation_velocity_error = velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            vorticity_opt_u - mean_u,
            vorticity_opt_v - mean_v,
        )
        vorticity_opt_fluctuation_vorticity_error = scalar_relative_error(
            fluctuation_omega, vorticity_opt_omega - mean_omega
        )
        vorticity_opt_wake_velocity_error = velocity_relative_error(
            actual_u, actual_v, vorticity_opt_u, vorticity_opt_v, wake
        )
        vorticity_opt_wake_vorticity_error = scalar_relative_error(
            actual_omega, vorticity_opt_omega, wake
        )
        vorticity_opt_wake_fluctuation_velocity_error = velocity_relative_error(
            fluctuation_u,
            fluctuation_v,
            vorticity_opt_u - mean_u,
            vorticity_opt_v - mean_v,
            wake,
        )
        vorticity_opt_wake_fluctuation_vorticity_error = scalar_relative_error(
            fluctuation_omega, vorticity_opt_omega - mean_omega, wake
        )

    def grid(values: np.ndarray) -> np.ndarray:
        result = np.zeros_like(solid, dtype=np.float32)
        result[fluid] = values
        return result

    payload = dict(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        solid=solid,
        beta=np.float32(beta),
        alpha_deg=np.float32(alpha),
        frame=np.int32(frame),
        actual_u=grid(actual_u),
        actual_v=grid(actual_v),
        actual_omega=grid(actual_omega),
        pinn_u=grid(predicted_u),
        pinn_v=grid(predicted_v),
        pinn_omega=grid(predicted_omega),
        modal_coefficients=modal.astype(np.float32),
        raw_modal_coefficients=raw_modal.astype(np.float32),
        coefficient_fit=np.asarray("velocity_l2"),
        basis_coordinate_system=np.asarray("mass_orthogonal_ritz"),
        projection_ritz_orth_error=np.float64(projection_ritz_orth_error),
        projection_ritz_prewhiten_orth_error=np.float64(
            projection_diagnostics["prewhiten_orthogonality_error"]
        ),
        projection_effective_rank=np.int32(projection_diagnostics["rank"]),
        projection_discarded_rank=np.int32(
            projection_diagnostics["discarded_rank"]
        ),
        projection_rank_tolerance=np.float64(
            projection_diagnostics["rank_tolerance"]
        ),
        projection_gram_condition_number=np.float64(
            projection_diagnostics["condition_number"]
        ),
        projection_solver=np.asarray(
            "projection_grid_mass_whitened_moore_penrose"
        ),
        velocity_error=np.float32(velocity_error),
        vorticity_error=np.float32(vorticity_error),
        fluctuation_velocity_error=np.float32(fluctuation_velocity_error),
        fluctuation_vorticity_error=np.float32(fluctuation_vorticity_error),
        wake_velocity_error=np.float32(wake_velocity_error),
        wake_vorticity_error=np.float32(wake_vorticity_error),
        wake_fluctuation_velocity_error=np.float32(
            wake_fluctuation_velocity_error
        ),
        wake_fluctuation_vorticity_error=np.float32(
            wake_fluctuation_vorticity_error
        ),
        wake_fluid_points=np.int32(np.count_nonzero(wake)),
        ritz_eigenvalues=ritz["eigenvalues"],
        ritz_coefficients=ritz["coefficients"],
        ritz_rank=np.int32(ritz["rank"]),
    )
    if certified_stokes:
        certified = int(ritz["certified_count"])
        payload.update(
            stokes_certified_modes=np.int32(certified),
            stokes_construction_rank=np.int32(ritz["construction_rank"]),
            stokes_certification_quadrature=np.int32(ritz["quadrature"]),
            stokes_maximum_quadrature_reached=np.bool_(
                ritz["maximum_quadrature_reached"]
            ),
            stokes_mass_defect=ritz["mass_defect"][:certified, :certified],
            stokes_mass_budget=ritz["mass_budget"][:certified, :certified],
            stokes_ritz_residual_relative=ritz[
                "ritz_residual_relative"
            ][:certified],
            stokes_ritz_residual_budget=ritz[
                "ritz_residual_budget"
            ][:certified],
            stokes_eigenvalue_drift_relative=ritz[
                "eigenvalue_drift_relative"
            ][:certified],
            stokes_eigenvalue_drift_budget=ritz[
                "eigenvalue_drift_budget"
            ][:certified],
            spectral_subspace_policy=np.asarray(
                "independent_nested_stokes_certified_prefix_only"
            ),
        )
    if not velocity_only:
        payload.update(
            vorticity_opt_u=grid(vorticity_opt_u),
            vorticity_opt_v=grid(vorticity_opt_v),
            vorticity_opt_omega=grid(vorticity_opt_omega),
            vorticity_opt_modal_coefficients=vorticity_modal.astype(np.float32),
            vorticity_opt_raw_modal_coefficients=vorticity_raw_modal.astype(
                np.float32
            ),
            vorticity_projection_rank=np.int32(vorticity_projection_rank),
            vorticity_projection_relative_residual=np.float64(
                vorticity_projection_relative_residual
            ),
            vorticity_opt_velocity_error=np.float32(
                vorticity_opt_velocity_error
            ),
            vorticity_opt_vorticity_error=np.float32(
                vorticity_opt_vorticity_error
            ),
            vorticity_opt_fluctuation_velocity_error=np.float32(
                vorticity_opt_fluctuation_velocity_error
            ),
            vorticity_opt_fluctuation_vorticity_error=np.float32(
                vorticity_opt_fluctuation_vorticity_error
            ),
            vorticity_opt_wake_velocity_error=np.float32(
                vorticity_opt_wake_velocity_error
            ),
            vorticity_opt_wake_vorticity_error=np.float32(
                vorticity_opt_wake_vorticity_error
            ),
            vorticity_opt_wake_fluctuation_velocity_error=np.float32(
                vorticity_opt_wake_fluctuation_velocity_error
            ),
            vorticity_opt_wake_fluctuation_vorticity_error=np.float32(
                vorticity_opt_wake_fluctuation_vorticity_error
            ),
        )
    np.savez_compressed(sample_path, **payload)


def sampled_percentile(paths: list[Path], key: str, percentile: float) -> float:
    pieces = []
    for path in paths:
        with np.load(path) as sample:
            values = np.abs(sample[key]).ravel()
            stride = max(1, len(values) // 100_000)
            pieces.append(values[::stride])
    return max(float(np.percentile(np.concatenate(pieces), percentile)), 1.0e-8)


def render_lowest_six(
    checkpoint: Path, output_dir: Path, settings: dict
) -> None:
    cfg, model, _ = load_hardcoded_model(checkpoint)
    beta = float(settings["basis_beta"])
    alpha = float(settings["basis_alpha_deg"])
    x = np.linspace(cfg.xmin, cfg.xmax, int(settings["basis_grid_nx"]))
    y = np.linspace(cfg.ymin, cfg.ymax, int(settings["basis_grid_ny"]))
    xgrid, ygrid = np.meshgrid(x, y)
    points_all = np.column_stack((xgrid.ravel(), ygrid.ravel()))
    solid = inside_airfoil(cfg, points_all, beta, alpha).reshape(xgrid.shape)
    fluid = ~solid
    points = points_all[fluid.ravel()].astype(np.float32)
    raw_u, raw_v, raw_omega = raw_fields(
        model, cfg, points, beta, alpha, int(settings["field_batch"])
    )
    if str(getattr(cfg, "basis_objective", "velocity_rayleigh")) == "stokes_certified":
        requested_quadrature = int(settings["ritz_quadrature"])
        ritz = compute_certified_stokes_basis(
            model,
            cfg,
            beta,
            alpha,
            minimum_quadrature=max(
                requested_quadrature,
                int(getattr(cfg, "stokes_min_quadrature", requested_quadrature)),
            ),
            maximum_quadrature=max(
                requested_quadrature,
                int(getattr(cfg, "stokes_max_quadrature", requested_quadrature)),
            ),
        )
        if int(ritz["certified_count"]) < 6:
            raise RuntimeError(
                "fewer than six Stokes modes passed independent certification"
            )
    else:
        ritz = compute_ritz_basis(
            model,
            cfg,
            beta,
            alpha,
            quadrature=int(settings["ritz_quadrature"]),
        )
    coefficients = ritz["coefficients"][:, :6]
    basis_u = raw_u @ coefficients
    basis_v = raw_v @ coefficients
    basis_omega = raw_omega @ coefficients
    output_dir.mkdir(parents=True, exist_ok=True)
    speed_limit = max(float(np.percentile(np.hypot(basis_u, basis_v), 99)), 1.0e-8)
    omega_limit = max(float(np.percentile(np.abs(basis_omega), 99)), 1.0e-8)
    figure, axes = plt.subplots(6, 2, figsize=(15.0, 15.5), constrained_layout=True)
    for index in range(6):
        def grid(values):
            result = np.zeros_like(solid, dtype=np.float32)
            result[fluid] = values
            return result

        u = grid(basis_u[:, index])
        v = grid(basis_v[:, index])
        omega = grid(basis_omega[:, index])
        velocity_panel(
            axes[index, 0],
            x,
            y,
            u,
            v,
            solid,
            f"Basis {index + 1}: velocity + streamlines",
            speed_limit,
        )
        vorticity_panel(
            axes[index, 1],
            x,
            y,
            omega,
            solid,
            f"Basis {index + 1}: vorticity, λ={ritz['eigenvalues'][index]:.4e}",
            omega_limit,
        )
        single, single_axes = plt.subplots(
            1, 2, figsize=(14.0, 3.8), constrained_layout=True
        )
        velocity_panel(
            single_axes[0], x, y, u, v, solid, "Velocity + streamlines", speed_limit
        )
        vorticity_panel(
            single_axes[1], x, y, omega, solid, "Vorticity", omega_limit
        )
        single.suptitle(
            rf"Lowest-frequency Ritz basis {index + 1}, "
            rf"$\lambda={ritz['eigenvalues'][index]:.5e}$"
        )
        single.savefig(output_dir / f"basis_{index + 1:02d}.png", dpi=150)
        plt.close(single)
    figure.suptitle(
        rf"Six lowest-frequency mass-orthogonal Ritz modes, "
        rf"$\beta={beta:.2f}$, $\alpha={alpha:+.1f}^\circ$"
    )
    figure.savefig(output_dir / "contact_sheet.png", dpi=150)
    plt.close(figure)
    write_json(
        output_dir / "summary.json",
        {
            "beta": beta,
            "alpha_deg": alpha,
            "eigenvalues": ritz["eigenvalues"][:6].tolist(),
            "rank": int(ritz["rank"]),
            "definition": "six smallest generalized Ritz eigenvalues",
            "basis_coordinate_system": "mass_orthogonal_ritz",
            "ritz_mass_orthogonality_error": float(
                np.max(
                    np.abs(
                        ritz["coefficients"].T
                        @ ritz["mass"]
                        @ ritz["coefficients"]
                        - np.eye(ritz["coefficients"].shape[1])
                    )
                )
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-frame-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--render-variant",
        choices=("velocity", "vorticity"),
        default="velocity",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.run_dir / "postprocess"
    gt_dir = output / "gt"
    sample_dir = output / "samples"
    frame_dir = output / "frames"
    vorticity_frame_dir = output / "frames_vorticity_optimal"
    basis_dir = output / "basis_lowest6"
    for directory in (
        output,
        gt_dir,
        sample_dir,
        frame_dir,
        vorticity_frame_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if args.render_frame_index is not None:
        index = args.render_frame_index
        limits = json.loads(
            (output / "render_limits.json").read_text(encoding="utf-8")
        )
        sample_path = sample_dir / f"sample_{index:03d}.npz"
        with np.load(sample_path) as archive:
            sample = {name: archive[name] for name in archive.files}
        target_dir = (
            vorticity_frame_dir
            if args.render_variant == "vorticity"
            else frame_dir
        )
        render_frame(
            sample,
            limits,
            target_dir / f"frame_{index:03d}.png",
            variant=args.render_variant,
        )
        return
    solver_config = Path(settings["solver_config"])
    if not solver_config.is_absolute():
        solver_config = WORKSPACE_ROOT / solver_config
    airfoil_config = Path(
        settings.get(
            "airfoil_config",
            "cases/2D_airfoil/configs/k256_final.json",
        )
    )
    if not airfoil_config.is_absolute():
        airfoil_config = WORKSPACE_ROOT / airfoil_config
    airfoil_settings = json.loads(airfoil_config.read_text(encoding="utf-8"))
    solver_settings = json.loads(solver_config.read_text(encoding="utf-8"))
    gt_cache_source = settings.get("gt_cache_source")
    if gt_cache_source is not None:
        gt_cache_source = Path(gt_cache_source)
        if not gt_cache_source.is_absolute():
            gt_cache_source = WORKSPACE_ROOT / gt_cache_source
    count = int(settings["parameter_samples"])
    positions = np.linspace(
        0.0,
        1.0,
        count,
        endpoint=bool(settings.get("path_endpoint", True)),
    )
    sample_paths = []
    cache_reuse = []
    state = {"stage": "gt_and_reconstruction", "completed_cases": 0, "total_cases": count}
    write_json(output / "pipeline_state.json", state)
    for index, position in enumerate(positions):
        angle = 2.0 * np.pi * position
        beta = float(settings["beta_center"] + settings["beta_amplitude"] * np.sin(angle))
        alpha = path_parameter(settings["alpha_amplitude_deg"] * np.cos(angle))
        cache_path = gt_dir / f"gt_{index:03d}.npz"
        sample_path = sample_dir / f"sample_{index:03d}.npz"
        effective_cache_path = cache_path
        if not cache_path.exists():
            source_path = (
                gt_cache_source / f"gt_{index:03d}.npz"
                if gt_cache_source is not None
                else None
            )
            if source_path is not None and source_path.exists():
                record = validate_gt_cache(
                    source_path,
                    airfoil_settings,
                    solver_settings,
                    beta,
                    alpha,
                )
                record["index"] = index
                cache_reuse.append(record)
                write_json(output / "gt_cache_reuse_manifest.json", {"cases": cache_reuse})
                effective_cache_path = source_path
                print(f"[reuse GT {index + 1:03d}/{count}] {source_path}", flush=True)
            else:
                print(
                    f"[GT {index + 1:03d}/{count}] beta={beta:.4f} alpha={alpha:+.3f}",
                    flush=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(CASE_ROOT / "scripts" / "solve_ns.py"),
                        "--config",
                        str(airfoil_config),
                        "--solver-config",
                        str(solver_config),
                        numeric_option("beta", beta),
                        numeric_option("alpha-deg", alpha),
                        "--output",
                        str(cache_path),
                    ],
                    cwd=WORKSPACE_ROOT,
                    check=True,
                )
        if not sample_path.exists():
            print(f"[reconstruct {index + 1:03d}/{count}]", flush=True)
            reconstruct_case(
                args.checkpoint,
                effective_cache_path,
                sample_path,
                phase=2.0 * position,
                quadrature=int(settings["ritz_quadrature"]),
                batch=int(settings["field_batch"]),
            )
        sample_paths.append(sample_path)
        state.update(
            {
                "completed_cases": index + 1,
                "last_beta": beta,
                "last_alpha_deg": alpha,
            }
        )
        write_json(output / "pipeline_state.json", state)

    limits = {
        "speed": max(
            sampled_percentile(sample_paths, "actual_u", 99.5),
            sampled_percentile(sample_paths, "actual_v", 99.5),
        ),
        "vorticity": sampled_percentile(sample_paths, "actual_omega", 99.0),
        "velocity_error": 1.0e-8,
        "vorticity_error": 1.0e-8,
        "vorticity_opt_velocity_error": 1.0e-8,
        "vorticity_opt_vorticity_error": 1.0e-8,
    }
    error_speed_samples, error_omega_samples = [], []
    vorticity_opt_error_speed_samples, vorticity_opt_error_omega_samples = (
        [],
        [],
    )
    for path in sample_paths:
        with np.load(path) as sample:
            error_speed = np.hypot(
                sample["pinn_u"] - sample["actual_u"],
                sample["pinn_v"] - sample["actual_v"],
            ).ravel()
            error_omega = np.abs(
                sample["pinn_omega"] - sample["actual_omega"]
            ).ravel()
            vorticity_opt_error_speed = np.hypot(
                sample["vorticity_opt_u"] - sample["actual_u"],
                sample["vorticity_opt_v"] - sample["actual_v"],
            ).ravel()
            vorticity_opt_error_omega = np.abs(
                sample["vorticity_opt_omega"] - sample["actual_omega"]
            ).ravel()
            stride = max(1, len(error_speed) // 100_000)
            error_speed_samples.append(error_speed[::stride])
            error_omega_samples.append(error_omega[::stride])
            vorticity_opt_error_speed_samples.append(
                vorticity_opt_error_speed[::stride]
            )
            vorticity_opt_error_omega_samples.append(
                vorticity_opt_error_omega[::stride]
            )
    limits["velocity_error"] = max(
        float(np.percentile(np.concatenate(error_speed_samples), 99.0)), 1.0e-8
    )
    limits["vorticity_error"] = max(
        float(np.percentile(np.concatenate(error_omega_samples), 99.0)), 1.0e-8
    )
    limits["vorticity_opt_velocity_error"] = max(
        float(
            np.percentile(
                np.concatenate(vorticity_opt_error_speed_samples), 99.0
            )
        ),
        1.0e-8,
    )
    limits["vorticity_opt_vorticity_error"] = max(
        float(
            np.percentile(
                np.concatenate(vorticity_opt_error_omega_samples), 99.0
            )
        ),
        1.0e-8,
    )
    write_json(output / "render_limits.json", limits)
    state["stage"] = "rendering_video"
    write_json(output / "pipeline_state.json", state)
    metrics = []
    for index, path in enumerate(sample_paths):
        frame_path = frame_dir / f"frame_{index:03d}.png"
        if not frame_path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--run-dir",
                    str(args.run_dir),
                    "--config",
                    str(args.config),
                    "--render-frame-index",
                    str(index),
                ],
                cwd=WORKSPACE_ROOT,
                check=True,
            )
        vorticity_frame_path = (
            vorticity_frame_dir / f"frame_{index:03d}.png"
        )
        if not vorticity_frame_path.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--run-dir",
                    str(args.run_dir),
                    "--config",
                    str(args.config),
                    "--render-frame-index",
                    str(index),
                    "--render-variant",
                    "vorticity",
                ],
                cwd=WORKSPACE_ROOT,
                check=True,
            )
        with np.load(path) as archive:
            sample = {name: archive[name] for name in archive.files}
        metrics.append(
            {
                "frame": index,
                "beta": float(sample["beta"]),
                "alpha_deg": float(sample["alpha_deg"]),
                "velocity_error": float(sample["velocity_error"]),
                "vorticity_error": float(sample["vorticity_error"]),
                "fluctuation_velocity_error": float(
                    sample["fluctuation_velocity_error"]
                ),
                "fluctuation_vorticity_error": float(
                    sample["fluctuation_vorticity_error"]
                ),
                "wake_velocity_error": float(sample["wake_velocity_error"]),
                "wake_vorticity_error": float(
                    sample["wake_vorticity_error"]
                ),
                "wake_fluctuation_velocity_error": float(
                    sample["wake_fluctuation_velocity_error"]
                ),
                "wake_fluctuation_vorticity_error": float(
                    sample["wake_fluctuation_vorticity_error"]
                ),
                "ritz_rank": int(sample["ritz_rank"]),
                "vorticity_projection_rank": int(
                    sample["vorticity_projection_rank"]
                ),
            }
        )
        for key in (
            "vorticity_opt_velocity_error",
            "vorticity_opt_vorticity_error",
            "vorticity_opt_fluctuation_velocity_error",
            "vorticity_opt_fluctuation_vorticity_error",
            "vorticity_opt_wake_velocity_error",
            "vorticity_opt_wake_vorticity_error",
            "vorticity_opt_wake_fluctuation_velocity_error",
            "vorticity_opt_wake_fluctuation_vorticity_error",
        ):
            metrics[-1][key] = float(sample[key])
    movie = output / "continuous_alpha_beta_gt_pinn_error.mp4"
    encode_mp4(frame_dir, movie, int(settings["fps"]))
    vorticity_optimal_movie = (
        output
        / "continuous_alpha_beta_gt_pinn_vorticity_optimal_error.mp4"
    )
    encode_mp4(
        vorticity_frame_dir,
        vorticity_optimal_movie,
        int(settings["fps"]),
    )
    state["stage"] = "rendering_lowest_six_basis"
    write_json(output / "pipeline_state.json", state)
    render_lowest_six(args.checkpoint, basis_dir, settings)
    aggregate_keys = (
        "velocity_error",
        "vorticity_error",
        "fluctuation_velocity_error",
        "fluctuation_vorticity_error",
        "wake_velocity_error",
        "wake_vorticity_error",
        "wake_fluctuation_velocity_error",
        "wake_fluctuation_vorticity_error",
        "vorticity_opt_velocity_error",
        "vorticity_opt_vorticity_error",
        "vorticity_opt_fluctuation_velocity_error",
        "vorticity_opt_fluctuation_vorticity_error",
        "vorticity_opt_wake_velocity_error",
        "vorticity_opt_wake_vorticity_error",
        "vorticity_opt_wake_fluctuation_velocity_error",
        "vorticity_opt_wake_fluctuation_vorticity_error",
    )
    aggregate_metrics = {}
    for key in aggregate_keys:
        values = np.asarray([item[key] for item in metrics], dtype=np.float64)
        aggregate_metrics[key] = {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95.0)),
            "max": float(np.max(values)),
            "max_case": int(np.argmax(values)),
        }
    summary = {
        "status": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "parameter_path": {
            "beta": "beta_center + beta_amplitude*sin(2*pi*s)",
            "alpha_deg": "alpha_amplitude_deg*cos(2*pi*s)",
            "samples": count,
        },
        "layout": {
            "rows": ["GT", "PINN basis reconstruction", "absolute error"],
            "columns": ["velocity magnitude and streamlines", "vorticity"],
        },
        "movie": str(movie.resolve()),
        "vorticity_optimal_movie": str(vorticity_optimal_movie.resolve()),
        "fps": int(settings["fps"]),
        "metrics": metrics,
        "aggregate_metrics": aggregate_metrics,
        "basis_contact_sheet": str((basis_dir / "contact_sheet.png").resolve()),
    }
    write_json(output / "continuous_pipeline_complete.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
