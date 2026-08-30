#!/usr/bin/env python3
"""Aggregate validation evidence for the shared-mimetic airfoil rollout.

The script is deliberately read-only with respect to rollout directories.  It
compares the legacy mixed-operator rollout with the new native-grid
grad--curl--Laplacian/frozen-base rollout and writes small CSV/JSON artifacts
under a separate summary directory.  A concurrently running dt/2 audit is
reported as ``pending`` until its method summary appears; rerunning this script
then fills the same convergence tables without any code changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METHODS = {
    "potential": {"legacy": "potential", "new": "potential"},
    "fd": {"legacy": "fd_laplacian", "new": "fd_mimetic"},
}
ERROR_METRICS = (
    "velocity_error",
    "vorticity_error",
    "wake_velocity_error",
    "wake_vorticity_error",
    "fluctuation_velocity_error",
    "fluctuation_vorticity_error",
    "wake_fluctuation_velocity_error",
    "wake_fluctuation_vorticity_error",
)
ERROR_STATS = ("mean", "p95", "max", "final")


def parse_args() -> argparse.Namespace:
    case_root = Path(__file__).resolve().parents[1]
    sequence = (
        case_root
        / "results"
        / "siren_original_jet1_gpu_k128_to_k2048_e2000_sequence"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=sequence / "temporal_rollout_gt30_k2048",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=sequence / "temporal_rollout_gt30_k2048_mimetic_base_subtracted",
    )
    parser.add_argument(
        "--dt-half-root",
        type=Path,
        default=(
            sequence
            / "temporal_rollout_gt30_k2048_mimetic_base_subtracted_dt_half_fd"
        ),
    )
    parser.add_argument("--orientation", default="up")
    parser.add_argument("--expected-modes", type=int, default=2048)
    parser.add_argument("--expected-frames", type=int, default=30)
    parser.add_argument(
        "--dt-half-methods",
        nargs="+",
        choices=tuple(METHODS),
        default=["fd"],
        help="Methods intentionally included in the dt/2 audit (default: fd).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <new-root>/validation_summary.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nested(mapping: dict[str, Any] | None, *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_path(root: Path, orientation: str, method: str) -> Path:
    return root / orientation / f"{method}_summary.json"


def error_comparison_rows(
    legacy: dict[str, dict[str, Any] | None],
    new: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        old_summary = legacy[method]
        new_summary = new[method]
        for metric in ERROR_METRICS:
            row: dict[str, Any] = {"method": method, "metric": metric}
            for stat in ERROR_STATS:
                old_value = as_float(nested(old_summary, "aggregate_errors", metric, stat))
                new_value = as_float(nested(new_summary, "aggregate_errors", metric, stat))
                row[f"legacy_{stat}"] = old_value
                row[f"new_{stat}"] = new_value
                row[f"absolute_delta_{stat}"] = (
                    None if old_value is None or new_value is None else new_value - old_value
                )
                row[f"relative_delta_percent_{stat}"] = (
                    None
                    if old_value in (None, 0.0) or new_value is None
                    else 100.0 * (new_value - old_value) / old_value
                )
            row["status"] = "complete" if old_summary and new_summary else "missing"
            rows.append(row)
    return rows


def method_gap_rows(
    legacy: dict[str, dict[str, Any] | None],
    new: dict[str, dict[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for protocol, summaries in (("legacy", legacy), ("shared_mimetic", new)):
        for metric in ERROR_METRICS:
            for stat in ERROR_STATS:
                potential = as_float(
                    nested(summaries["potential"], "aggregate_errors", metric, stat)
                )
                fd = as_float(nested(summaries["fd"], "aggregate_errors", metric, stat))
                rows.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "statistic": stat,
                        "potential": potential,
                        "fd": fd,
                        "fd_minus_potential": (
                            None if potential is None or fd is None else fd - potential
                        ),
                        "fd_over_potential": (
                            None
                            if potential in (None, 0.0) or fd is None
                            else fd / potential
                        ),
                    }
                )
    return rows


def timing_rows(
    legacy: dict[str, dict[str, Any] | None],
    new: dict[str, dict[str, Any] | None],
    predicted_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        old_summary = legacy[method]
        new_summary = new[method]
        old_basis = sum(
            value or 0.0
            for value in (
                as_float(nested(old_summary, "timings", "basis_field_seconds")),
                as_float(nested(old_summary, "timings", "outlet_basis_seconds")),
            )
        )
        new_basis = sum(
            value or 0.0
            for value in (
                as_float(nested(new_summary, "timings", "psi_generation_or_load_seconds")),
                as_float(nested(new_summary, "timings", "fields_from_psi_seconds")),
                as_float(nested(new_summary, "timings", "outlet_trace_seconds")),
            )
        )
        stages = {
            "basis_preparation": (old_basis or None, new_basis or None),
            "ritz": (
                as_float(nested(old_summary, "timings", "ritz_seconds")),
                as_float(nested(new_summary, "timings", "ritz_seconds")),
            ),
            "initial_projection": (
                as_float(nested(old_summary, "timings", "initial_projection_seconds")),
                as_float(nested(new_summary, "timings", "initial_projection_seconds")),
            ),
            "base_rhs_and_correction": (
                None,
                as_float(nested(new_summary, "timings", "base_rhs_and_correction_seconds")),
            ),
            "coefficient_rollout": (
                as_float(nested(old_summary, "timings", "coefficient_rollout_seconds")),
                as_float(nested(new_summary, "timings", "coefficient_rollout_seconds")),
            ),
            "coefficient_rollout_per_predicted_frame": (
                (
                    as_float(nested(old_summary, "timings", "coefficient_rollout_seconds"))
                    / predicted_frames
                    if as_float(nested(old_summary, "timings", "coefficient_rollout_seconds"))
                    is not None
                    else None
                ),
                (
                    as_float(nested(new_summary, "timings", "coefficient_rollout_seconds"))
                    / predicted_frames
                    if as_float(nested(new_summary, "timings", "coefficient_rollout_seconds"))
                    is not None
                    else None
                ),
            ),
            "reconstruction": (
                as_float(nested(old_summary, "timings", "reconstruction_seconds")),
                as_float(nested(new_summary, "timings", "reconstruction_seconds")),
            ),
            "serialization": (
                as_float(nested(old_summary, "timings", "serialization_seconds")),
                as_float(nested(new_summary, "timings", "serialization_seconds")),
            ),
            "total_method": (
                as_float(nested(old_summary, "timings", "total_method_seconds")),
                as_float(nested(new_summary, "timings", "total_method_seconds")),
            ),
        }
        for stage, (old_value, new_value) in stages.items():
            rows.append(
                {
                    "method": method,
                    "stage": stage,
                    "legacy_seconds": old_value,
                    "new_seconds": new_value,
                    "new_over_legacy": (
                        None
                        if old_value in (None, 0.0) or new_value is None
                        else new_value / old_value
                    ),
                    "absolute_delta_seconds": (
                        None
                        if old_value is None or new_value is None
                        else new_value - old_value
                    ),
                    "note": (
                        "legacy and new spatial/quadrature protocols differ"
                        if old_value is not None and new_value is not None
                        else "new closure stage has no legacy counterpart"
                    ),
                }
            )
    return rows


def trapz(x: list[float], y: list[float]) -> float:
    return sum(
        0.5 * (y[index] + y[index - 1]) * (x[index] - x[index - 1])
        for index in range(1, len(x))
    )


def energy_summary(path: Path, method: str, expected_frames: int) -> dict[str, Any]:
    raw_rows = read_csv(path)
    if not raw_rows:
        return {"method": method, "status": "missing", "source": str(path.resolve())}
    names = (
        "time",
        "coefficient_norm",
        "kinetic_energy",
        "power_self_advection",
        "power_mean_and_outlet_increment",
        "power_resolved_base",
        "power_constant_closure",
        "power_viscous",
        "power_total",
    )
    rows: list[dict[str, float]] = []
    for raw in raw_rows:
        row = {name: float(raw[name]) for name in names}
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite energy budget value in {path}")
        rows.append(row)
    base_cancellation = [
        row["power_resolved_base"] + row["power_constant_closure"] for row in rows
    ]
    arithmetic_residual = [
        row["power_total"]
        - (
            row["power_self_advection"]
            + row["power_mean_and_outlet_increment"]
            + row["power_resolved_base"]
            + row["power_constant_closure"]
            + row["power_viscous"]
        )
        for row in rows
    ]
    times = [row["time"] for row in rows]
    power_total = [row["power_total"] for row in rows]
    measured_ke_delta = rows[-1]["kinetic_energy"] - rows[0]["kinetic_energy"]
    integrated_power = trapz(times, power_total)
    return {
        "method": method,
        "status": "complete" if len(rows) == expected_frames + 1 else "frame_count_mismatch",
        "source": str(path.resolve()),
        "rows": len(rows),
        "expected_rows": expected_frames + 1,
        "initial_coefficient_norm": rows[0]["coefficient_norm"],
        "final_coefficient_norm": rows[-1]["coefficient_norm"],
        "coefficient_norm_ratio": rows[-1]["coefficient_norm"] / rows[0]["coefficient_norm"],
        "initial_kinetic_energy": rows[0]["kinetic_energy"],
        "final_kinetic_energy": rows[-1]["kinetic_energy"],
        "kinetic_energy_ratio": rows[-1]["kinetic_energy"] / rows[0]["kinetic_energy"],
        "measured_kinetic_energy_delta": measured_ke_delta,
        "trapezoidal_integral_power_total": integrated_power,
        "coarse_output_energy_balance_residual": measured_ke_delta - integrated_power,
        "max_abs_power_self_advection": max(abs(row["power_self_advection"]) for row in rows),
        "max_abs_base_power_cancellation_error": max(abs(value) for value in base_cancellation),
        "max_abs_power_arithmetic_residual": max(abs(value) for value in arithmetic_residual),
        "max_power_viscous": max(row["power_viscous"] for row in rows),
        "min_power_viscous": min(row["power_viscous"] for row in rows),
        "mean_power_mean_and_outlet_increment": sum(
            row["power_mean_and_outlet_increment"] for row in rows
        )
        / len(rows),
        "max_power_mean_and_outlet_increment": max(
            row["power_mean_and_outlet_increment"] for row in rows
        ),
        "positive_total_power_fraction": sum(row["power_total"] > 0.0 for row in rows)
        / len(rows),
        **(
            {
                "mean_power_mean_volume_increment": sum(
                    float(raw["power_mean_volume_increment"]) for raw in raw_rows
                )
                / len(raw_rows),
                "max_power_mean_volume_increment": max(
                    float(raw["power_mean_volume_increment"]) for raw in raw_rows
                ),
                "mean_power_outlet_increment": sum(
                    float(raw["power_outlet_increment"]) for raw in raw_rows
                )
                / len(raw_rows),
                "min_power_outlet_increment": min(
                    float(raw["power_outlet_increment"]) for raw in raw_rows
                ),
                "max_power_outlet_increment": max(
                    float(raw["power_outlet_increment"]) for raw in raw_rows
                ),
                "max_abs_power_sum_defect": max(
                    abs(float(raw["power_sum_defect"])) for raw in raw_rows
                ),
            }
            if raw_rows and "power_mean_volume_increment" in raw_rows[0]
            else {}
        ),
    }


def audit_rows(
    operator: dict[str, Any] | None,
    new: dict[str, dict[str, Any] | None],
    energies: list[dict[str, Any]],
    expected_modes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        category: str,
        method: str,
        metric: str,
        value: Any,
        criterion: str,
        passed: bool | None,
    ) -> None:
        rows.append(
            {
                "category": category,
                "method": method,
                "metric": metric,
                "value": value,
                "criterion": criterion,
                "status": "pending" if passed is None else ("pass" if passed else "fail"),
            }
        )

    factorization = as_float(nested(operator, "mimetic_complex", "laplacian_incidence_factorization_max_abs"))
    add("operator", "shared", "laplacian_incidence_factorization_max_abs", factorization, "<=1e-12", factorization is not None and factorization <= 1e-12)
    same_mask = nested(operator, "archived_fd_cache", "same_airfoil_mask_as_ns_grid")
    add("operator", "shared", "same_airfoil_mask_as_ns_grid", same_mask, "true", same_mask is True)
    shared = nested(operator, "shared_by_methods")
    add("operator", "shared", "shared_by_methods", shared, "true", shared is True)
    cache_residual = as_float(nested(operator, "archived_fd_cache", "maximum_relative_eigen_residual"))
    add("operator", "fd", "archived_cache_maximum_relative_eigen_residual", cache_residual, "<=1e-8", cache_residual is not None and cache_residual <= 1e-8)
    cache_orth = as_float(nested(operator, "archived_fd_cache", "mass_orthogonality_max_abs"))
    add("operator", "fd", "archived_cache_mass_orthogonality_max_abs", cache_orth, "<=1e-10", cache_orth is not None and cache_orth <= 1e-10)

    for method, summary in new.items():
        if summary is None:
            add("ritz", method, "summary_available", False, "true", False)
            continue
        rank = nested(summary, "ritz_diagnostics", "rank")
        add("ritz", method, "rank", rank, f"={expected_modes}", rank == expected_modes)
        orth = as_float(nested(summary, "ritz_diagnostics", "mass_orthogonality_max_abs_fp32"))
        add("ritz", method, "mass_orthogonality_max_abs_fp32", orth, "<=5e-4", orth is not None and orth <= 5e-4)
        chain = nested(summary, "basis_field_diagnostics", "mimetic_chain") or {}
        weak = as_float(chain.get("weak_curl_relative_max_abs_fp64"))
        add("chain", method, "weak_curl_relative_max_abs_fp64", weak, "<=1e-10", weak is not None and weak <= 1e-10)
        divergence = as_float(chain.get("rotated_gradient_divergence_scaled_max_abs_fp64"))
        add("chain", method, "rotated_gradient_divergence_scaled_max_abs_fp64", divergence, "<=1e-10", divergence is not None and divergence <= 1e-10)
        if method == "fd":
            fp32_residual = as_float(chain.get("cached_eigen_residual_max_fp64_operator_fp32_psi"))
            add("chain", method, "cached_eigen_residual_max_fp64_operator_fp32_psi", fp32_residual, "<=2e-3", fp32_residual is not None and fp32_residual <= 2e-3)
        outlet = nested(summary, "basis_field_diagnostics", "outlet_trace") or {}
        flux = as_float(outlet.get("ritz_maximum_absolute_integrated_normal_mode_flux"))
        add("outlet", method, "ritz_maximum_absolute_integrated_normal_mode_flux", flux, "<=5e-6", flux is not None and flux <= 5e-6)
        zero_rhs = as_float(nested(summary, "rollout_diagnostics", "base_flow", "corrected_rhs_at_zero_norm"))
        add("closure", method, "corrected_rhs_at_zero_norm", zero_rhs, "<=1e-10", zero_rhs is not None and zero_rhs <= 1e-10)

    for energy in energies:
        method = str(energy.get("method"))
        if energy.get("status") == "missing":
            add("energy", method, "budget_available", False, "true", False)
            continue
        checks = (
            ("max_abs_power_self_advection", 1e-8),
            ("max_abs_base_power_cancellation_error", 1e-10),
            # The budget is accumulated from FP32 GPU reductions; 5e-8 is
            # still over five orders below the largest observed power.
            ("max_abs_power_arithmetic_residual", 5e-8),
            ("max_power_viscous", 1e-12),
        )
        for metric, threshold in checks:
            value = as_float(energy.get(metric))
            add("energy", method, metric, value, f"<={threshold:g}", value is not None and value <= threshold)
    return rows


def dt_half_rows(
    full: dict[str, dict[str, Any] | None],
    half: dict[str, dict[str, Any] | None],
    dt_state: dict[str, Any] | None,
    requested_methods: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state_status = nested(dt_state, "status") or "missing"
    for method in METHODS:
        full_summary = full[method]
        half_summary = half[method]
        if method not in requested_methods:
            rows.append(
                {
                    "method": method,
                    "metric": "all",
                    "full_internal_dt": as_float(
                        nested(full_summary, "rollout_diagnostics", "internal_dt")
                    ),
                    "half_internal_dt": None,
                    "status": "not_run",
                    "criterion": "not requested",
                    "detail": "dt/2 audit targeted the unstable FD trajectory",
                }
            )
            continue
        if not half_summary:
            rows.append(
                {
                    "method": method,
                    "metric": "all",
                    "full_internal_dt": as_float(nested(full_summary, "rollout_diagnostics", "internal_dt")),
                    "half_internal_dt": None,
                    "status": "pending",
                    "criterion": "absolute change <=0.01 or relative change <=10%",
                    "detail": f"dt-half state={state_status}; method summary not present",
                }
            )
            continue
        for metric in ERROR_METRICS:
            full_mean = as_float(nested(full_summary, "aggregate_errors", metric, "mean"))
            half_mean = as_float(nested(half_summary, "aggregate_errors", metric, "mean"))
            full_final = as_float(nested(full_summary, "aggregate_errors", metric, "final"))
            half_final = as_float(nested(half_summary, "aggregate_errors", metric, "final"))
            mean_abs = None if full_mean is None or half_mean is None else abs(half_mean - full_mean)
            final_abs = None if full_final is None or half_final is None else abs(half_final - full_final)
            mean_rel = None if full_mean in (None, 0.0) or half_mean is None else 100.0 * mean_abs / abs(full_mean)
            final_rel = None if full_final in (None, 0.0) or half_final is None else 100.0 * final_abs / abs(full_final)
            stable_mean = mean_abs is not None and mean_rel is not None and (mean_abs <= 0.01 or mean_rel <= 10.0)
            stable_final = final_abs is not None and final_rel is not None and (final_abs <= 0.01 or final_rel <= 10.0)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "full_internal_dt": as_float(nested(full_summary, "rollout_diagnostics", "internal_dt")),
                    "half_internal_dt": as_float(nested(half_summary, "rollout_diagnostics", "internal_dt")),
                    "full_mean": full_mean,
                    "half_mean": half_mean,
                    "absolute_change_mean": mean_abs,
                    "relative_change_percent_mean": mean_rel,
                    "full_final": full_final,
                    "half_final": half_final,
                    "absolute_change_final": final_abs,
                    "relative_change_percent_final": final_rel,
                    "status": "pass" if stable_mean and stable_final else "fail",
                    "criterion": "mean and final: absolute change <=0.01 or relative change <=10%",
                    "detail": f"dt-half state={state_status}",
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.new_root / "validation_summary"
    orientation = args.orientation
    legacy = {
        key: read_json(summary_path(args.legacy_root, orientation, spec["legacy"]))
        for key, spec in METHODS.items()
    }
    new = {
        key: read_json(summary_path(args.new_root, orientation, spec["new"]))
        for key, spec in METHODS.items()
    }
    half = {
        key: read_json(summary_path(args.dt_half_root, orientation, spec["new"]))
        for key, spec in METHODS.items()
    }
    operator_path = args.new_root / orientation / "operator_identity_audit.json"
    operator = read_json(operator_path)
    dt_state = read_json(args.dt_half_root / "state.json")

    errors = error_comparison_rows(legacy, new)
    method_gaps = method_gap_rows(legacy, new)
    timings = timing_rows(legacy, new, args.expected_frames)
    energies = [
        energy_summary(
            args.new_root / orientation / f"{spec['new']}_energy_budget_by_frame.csv",
            method,
            args.expected_frames,
        )
        for method, spec in METHODS.items()
    ]
    dt_half_energies = [
        energy_summary(
            args.dt_half_root
            / orientation
            / f"{spec['new']}_energy_budget_by_frame.csv",
            method,
            args.expected_frames,
        )
        for method, spec in METHODS.items()
    ]
    audits = audit_rows(operator, new, energies, args.expected_modes)
    dt_half = dt_half_rows(new, half, dt_state, set(args.dt_half_methods))

    error_fields = ["method", "metric"] + [
        f"{prefix}_{stat}"
        for stat in ERROR_STATS
        for prefix in ("legacy", "new", "absolute_delta", "relative_delta_percent")
    ] + ["status"]
    write_csv(output_root / "old_vs_new_errors.csv", errors, error_fields)
    write_csv(
        output_root / "method_gap_errors.csv",
        method_gaps,
        (
            "protocol",
            "metric",
            "statistic",
            "potential",
            "fd",
            "fd_minus_potential",
            "fd_over_potential",
        ),
    )
    write_csv(
        output_root / "timing_comparison.csv",
        timings,
        ("method", "stage", "legacy_seconds", "new_seconds", "new_over_legacy", "absolute_delta_seconds", "note"),
    )
    energy_fields = list(
        dict.fromkeys(key for row in energies for key in row.keys())
    )
    write_csv(output_root / "energy_budget_summary.csv", energies, energy_fields)
    dt_half_energy_fields = list(
        dict.fromkeys(key for row in dt_half_energies for key in row.keys())
    )
    write_csv(
        output_root / "dt_half_energy_budget_summary.csv",
        dt_half_energies,
        dt_half_energy_fields,
    )
    write_csv(
        output_root / "validation_audits.csv",
        audits,
        ("category", "method", "metric", "value", "criterion", "status"),
    )
    dt_fields = list(dict.fromkeys(key for row in dt_half for key in row.keys()))
    write_csv(output_root / "dt_half_convergence.csv", dt_half, dt_fields)

    audit_failures = [row for row in audits if row["status"] == "fail"]
    dt_complete = all(row["status"] != "pending" for row in dt_half)
    dt_failures = [row for row in dt_half if row["status"] == "fail"]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "orientation": orientation,
        "expected_modes": args.expected_modes,
        "expected_frames": args.expected_frames,
        "sources": {
            "legacy_root": str(args.legacy_root.resolve()),
            "new_root": str(args.new_root.resolve()),
            "dt_half_root": str(args.dt_half_root.resolve()),
            "operator_identity_audit": str(operator_path.resolve()),
        },
        "source_completeness": {
            "legacy": {key: value is not None for key, value in legacy.items()},
            "new": {key: value is not None for key, value in new.items()},
            "dt_half": {key: value is not None for key, value in half.items()},
            "dt_half_state": nested(dt_state, "status") or "missing",
        },
        "all_loaded_json_values_finite": all(
            finite_tree(value)
            for value in (*legacy.values(), *new.values(), *half.values(), operator)
            if value is not None
        ),
        "validation_status": {
            "structural_audits": "pass" if not audit_failures else "fail",
            "structural_audit_failure_count": len(audit_failures),
            "dt_half": (
                "pending" if not dt_complete else ("pass" if not dt_failures else "fail")
            ),
            "dt_half_failure_count": len(dt_failures),
        },
        "operator_identity_audit": operator,
        "old_vs_new_errors": errors,
        "method_gap_errors": method_gaps,
        "timing_comparison": timings,
        "energy_budget_summary": energies,
        "dt_half_energy_budget_summary": dt_half_energies,
        "validation_audits": audits,
        "dt_half_convergence": dt_half,
        "artifact_paths": {
            "old_vs_new_errors_csv": str((output_root / "old_vs_new_errors.csv").resolve()),
            "method_gap_errors_csv": str((output_root / "method_gap_errors.csv").resolve()),
            "timing_comparison_csv": str((output_root / "timing_comparison.csv").resolve()),
            "energy_budget_summary_csv": str((output_root / "energy_budget_summary.csv").resolve()),
            "dt_half_energy_budget_summary_csv": str(
                (output_root / "dt_half_energy_budget_summary.csv").resolve()
            ),
            "validation_audits_csv": str((output_root / "validation_audits.csv").resolve()),
            "dt_half_convergence_csv": str((output_root / "dt_half_convergence.csv").resolve()),
        },
    }
    write_json(output_root / "validation_summary.json", payload)
    print(json.dumps(payload["validation_status"], indent=2))
    print(f"Wrote {output_root.resolve()}")


if __name__ == "__main__":
    main()
