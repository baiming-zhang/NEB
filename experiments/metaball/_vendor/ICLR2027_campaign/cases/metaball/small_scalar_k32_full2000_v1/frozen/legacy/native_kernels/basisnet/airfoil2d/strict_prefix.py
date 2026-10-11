














from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from _neb_paths import Path
from typing import Any, Iterable

import numpy as np

from .mimetic_fd import MaskedCartesianComplex, diagnostics as complex_diagnostics


FORMAL_FULL_MODES = 2048
FORMAL_SAMPLING_MODE = "uniform_random"
FORMAL_SAMPLING_RNG = "numpy.random.PCG64"
STRICT_PREFIXES = (64, 128, 256, 512, 1024, FORMAL_FULL_MODES)
OBJECTIVE_ALIASES = {
    "potential": "potential",
    "streamfunction_dirichlet_rayleigh": "potential",
    "velocity": "velocity",
    "velocity_rayleigh": "velocity",
    "vorticity": "vorticity",
    "pure_vorticity_rayleigh": "vorticity",
}


@dataclass(frozen=True)
class MimeticFields:


    potential_grid: np.ndarray
    potential: np.ndarray
    velocity_u: np.ndarray
    velocity_v: np.ndarray
    vorticity: np.ndarray

    @property
    def modes(self) -> int:
        return int(self.potential.shape[1])


@dataclass(frozen=True)
class CausalTransform:


    objective: str
    causal_order: np.ndarray
    transform_sorted: np.ndarray
    transform_network: np.ndarray
    mass_gram_network: np.ndarray
    stiffness_gram_network: np.ndarray
    sorting_rayleigh_network: np.ndarray
    peeled_rayleigh: np.ndarray
    pre_normalization_mass: np.ndarray
    orthogonality_max_abs: float
    triangular_lower_max_abs: float
    passes: int

    @property
    def modes(self) -> int:
        return int(self.transform_sorted.shape[0])


def canonical_objective(value: str) -> str:


    try:
        return OBJECTIVE_ALIASES[str(value)]
    except KeyError as error:
        raise ValueError(f"unsupported strict-prefix objective {value!r}") from error


def array_sha256(value: np.ndarray) -> str:


    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_mimetic_fields(
    complex_: MaskedCartesianComplex,
    potential_grid: np.ndarray,
) -> MimeticFields:


    grid = np.asarray(potential_grid, dtype=np.float64)
    if grid.ndim == 2:
        grid = grid[..., None]
    if grid.ndim != 3 or grid.shape[:2] != complex_.unknown.shape:
        raise ValueError(
            f"potential grid shape {grid.shape} does not match "
            f"{complex_.unknown.shape}"
        )
    potential = np.asarray(grid[complex_.unknown], dtype=np.float64)
    gradient_x = np.asarray(complex_.gradient_x @ potential, dtype=np.float64)
    gradient_y = np.asarray(complex_.gradient_y @ potential, dtype=np.float64)
    velocity_u = np.asarray(complex_.average_y @ gradient_y, dtype=np.float64)
    velocity_v = np.asarray(-(complex_.average_x @ gradient_x), dtype=np.float64)
    vorticity = np.asarray(complex_.laplacian @ potential, dtype=np.float64)
    fields = MimeticFields(
        potential_grid=grid,
        potential=potential,
        velocity_u=velocity_u,
        velocity_v=velocity_v,
        vorticity=vorticity,
    )
    for name, value in fields.__dict__.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"non-finite mimetic field {name}")
    return fields


def _weighted_gram(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    weights64 = np.asarray(weights, dtype=np.float64)
    if values64.ndim != 2 or weights64.shape != (values64.shape[0],):
        raise ValueError("weighted Gram shapes are incompatible")
    return values64.T @ (weights64[:, None] * values64)


def _gradient_gram(
    complex_: MaskedCartesianComplex,
    *fields: np.ndarray,
) -> np.ndarray:
    gram: np.ndarray | None = None
    scale = float(complex_.dx * complex_.dy)
    for field in fields:
        value = np.asarray(field, dtype=np.float64)
        gx = np.asarray(complex_.gradient_x @ value, dtype=np.float64)
        gy = np.asarray(complex_.gradient_y @ value, dtype=np.float64)
        current = scale * (gx.T @ gx + gy.T @ gy)
        gram = current if gram is None else gram + current
    if gram is None:
        raise ValueError("at least one field is required")
    return gram


def objective_grams(
    complex_: MaskedCartesianComplex,
    fields: MimeticFields,
    objective: str,
) -> tuple[np.ndarray, np.ndarray]:


    objective = canonical_objective(objective)
    weights = np.asarray(complex_.weights, dtype=np.float64)
    if objective == "potential":
        mass = _weighted_gram(fields.potential, weights)
        stiffness = _gradient_gram(complex_, fields.potential)
    elif objective == "velocity":
        mass = _weighted_gram(fields.velocity_u, weights) + _weighted_gram(
            fields.velocity_v, weights
        )
        stiffness = _gradient_gram(
            complex_, fields.velocity_u, fields.velocity_v
        )
    else:
        mass = _weighted_gram(fields.vorticity, weights)
        stiffness = _gradient_gram(complex_, fields.vorticity)
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    if mass.shape != stiffness.shape or mass.ndim != 2:
        raise ValueError("objective Gram matrices are incompatible")
    if not np.all(np.isfinite(mass)) or not np.all(np.isfinite(stiffness)):
        raise FloatingPointError("objective Gram matrices are non-finite")
    return mass, stiffness


def build_causal_transform(
    mass_gram: np.ndarray,
    stiffness_gram: np.ndarray,
    objective: str,
    *,
    passes: int = 2,
    mass_floor: float = 1.0e-14,
) -> CausalTransform:








    objective = canonical_objective(objective)
    mass = np.asarray(mass_gram, dtype=np.float64)
    stiffness = np.asarray(stiffness_gram, dtype=np.float64)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1]:
        raise ValueError("mass Gram must be square")
    if stiffness.shape != mass.shape:
        raise ValueError("stiffness Gram must match mass Gram")
    if int(passes) != 2:
        raise ValueError("strict prefix construction requires exactly two MGS passes")
    if not math.isfinite(mass_floor) or mass_floor <= 0.0:
        raise ValueError("mass_floor must be positive and finite")
    mass = 0.5 * (mass + mass.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    raw_mass = np.diag(mass)
    raw_stiffness = np.diag(stiffness)
    if np.any(~np.isfinite(raw_mass)) or np.any(raw_mass <= mass_floor):
        raise FloatingPointError("raw objective mass is non-finite or unresolved")
    if np.any(~np.isfinite(raw_stiffness)):
        raise FloatingPointError("raw objective stiffness is non-finite")
    sorting_rayleigh = raw_stiffness / raw_mass
    order = np.argsort(sorting_rayleigh, kind="stable").astype(np.int64)
    mass_sorted = mass[np.ix_(order, order)]
    stiffness_sorted = stiffness[np.ix_(order, order)]
    modes = mass.shape[0]
    transform_sorted = np.zeros((modes, modes), dtype=np.float64)
    pre_mass = np.empty(modes, dtype=np.float64)
    for causal_rank in range(modes):
        coefficient = np.zeros(modes, dtype=np.float64)
        coefficient[causal_rank] = 1.0
        if causal_rank:
            previous = transform_sorted[:, :causal_rank]
            for _ in range(passes):
                increment = previous.T @ mass_sorted @ coefficient
                coefficient -= previous @ increment
        current_mass = float(coefficient @ mass_sorted @ coefficient)
        if not math.isfinite(current_mass) or current_mass <= mass_floor:
            raise FloatingPointError(
                f"causal column {causal_rank + 1} collapsed: "
                f"mass={current_mass:.8e}"
            )
        pre_mass[causal_rank] = current_mass
        transform_sorted[:, causal_rank] = coefficient / math.sqrt(current_mass)

    transform_network = np.zeros_like(transform_sorted)
    transform_network[order, :] = transform_sorted
    peeled_mass = transform_sorted.T @ mass_sorted @ transform_sorted
    peeled_stiffness = transform_sorted.T @ stiffness_sorted @ transform_sorted
    peeled_rayleigh = np.diag(peeled_stiffness) / np.diag(peeled_mass)
    orthogonality = float(
        np.max(np.abs(peeled_mass - np.eye(modes, dtype=np.float64)))
    )
    triangular_defect = float(
        np.max(np.abs(np.tril(transform_sorted, k=-1)))
    )
    if not np.all(np.isfinite(peeled_rayleigh)):
        raise FloatingPointError("peeled Rayleigh diagonal is non-finite")
    return CausalTransform(
        objective=objective,
        causal_order=order,
        transform_sorted=transform_sorted,
        transform_network=transform_network,
        mass_gram_network=mass,
        stiffness_gram_network=stiffness,
        sorting_rayleigh_network=sorting_rayleigh,
        peeled_rayleigh=peeled_rayleigh,
        pre_normalization_mass=pre_mass,
        orthogonality_max_abs=orthogonality,
        triangular_lower_max_abs=triangular_defect,
        passes=int(passes),
    )


def apply_network_transform(
    fields: MimeticFields,
    transform_network: np.ndarray,
) -> MimeticFields:


    transform = np.asarray(transform_network, dtype=np.float64)
    if transform.shape != (fields.modes, fields.modes):
        raise ValueError(
            f"transform shape {transform.shape} does not match {fields.modes} modes"
        )

    def apply(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=np.float64) @ transform

    return MimeticFields(
        potential_grid=np.tensordot(
            np.asarray(fields.potential_grid, dtype=np.float64),
            transform,
            axes=([-1], [0]),
        ),
        potential=apply(fields.potential),
        velocity_u=apply(fields.velocity_u),
        velocity_v=apply(fields.velocity_v),
        vorticity=apply(fields.vorticity),
    )


def derivative_commutation_audit(
    complex_: MaskedCartesianComplex,
    raw_fields: MimeticFields,
    transformed_fields: MimeticFields,
) -> dict[str, float]:


    if raw_fields.modes != transformed_fields.modes:
        raise ValueError("raw and transformed field mode counts differ")
    derived = derive_mimetic_fields(complex_, transformed_fields.potential_grid)
    result: dict[str, float] = {}
    for name in ("potential", "velocity_u", "velocity_v", "vorticity"):
        expected = np.asarray(getattr(transformed_fields, name), dtype=np.float64)
        actual = np.asarray(getattr(derived, name), dtype=np.float64)
        difference = float(np.max(np.abs(actual - expected)))
        scale = max(float(np.max(np.abs(expected))), 1.0e-30)
        result[f"{name}_max_abs"] = difference
        result[f"{name}_relative_max_abs"] = difference / scale
    return result


def validated_prefixes(prefixes: Iterable[int], modes: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in prefixes)
    if not values or values != tuple(sorted(set(values))):
        raise ValueError("prefixes must be unique and strictly increasing")
    if values[0] <= 0 or values[-1] != int(modes):
        raise ValueError("prefixes must be positive and end at the full mode count")
    return values


def prefix_diagnostics(
    state: CausalTransform,
    basis_fields: MimeticFields,
    prefixes: Iterable[int],
) -> list[dict[str, Any]]:


    values = validated_prefixes(prefixes, state.modes)
    order = state.causal_order
    mass_sorted = state.mass_gram_network[np.ix_(order, order)]
    rows: list[dict[str, Any]] = []
    for modes in values:
        transform = state.transform_sorted[:, :modes]
        gram = transform.T @ mass_sorted @ transform
        fields = {

            "potential": basis_fields.potential[:, :modes].astype(np.float32),
            "velocity_u": basis_fields.velocity_u[:, :modes].astype(np.float32),
            "velocity_v": basis_fields.velocity_v[:, :modes].astype(np.float32),
            "vorticity": basis_fields.vorticity[:, :modes].astype(np.float32),
        }
        rows.append(
            {
                "modes": modes,
                "column_slice": [0, modes],
                "objective_gram_max_abs_defect": float(
                    np.max(np.abs(gram - np.eye(modes, dtype=np.float64)))
                ),
                "transform_sha256": array_sha256(
                    state.transform_network[:, :modes]
                ),
                "field_sha256": {
                    name: array_sha256(value) for name, value in fields.items()
                },
            }
        )
    return rows


def state_arrays(
    state: CausalTransform,
    basis_fields: MimeticFields,
    complex_: MaskedCartesianComplex,
    *,
    persist_fields: bool = True,
) -> dict[str, np.ndarray]:


    arrays = {
        "causal_order_0based": state.causal_order.astype(np.int64),
        "transform_sorted_triangular": state.transform_sorted.astype(np.float64),
        "transform_network_to_causal": state.transform_network.astype(np.float64),
        "sorting_rayleigh_network": state.sorting_rayleigh_network.astype(np.float64),
        "peeled_rayleigh": state.peeled_rayleigh.astype(np.float64),
        "pre_normalization_mass": state.pre_normalization_mass.astype(np.float64),
        "x": np.asarray(complex_.x, dtype=np.float64),
        "y": np.asarray(complex_.y, dtype=np.float64),
        "solid": np.asarray(complex_.solid, dtype=np.bool_),
        "unknown": np.asarray(complex_.unknown, dtype=np.bool_),
    }
    if persist_fields:
        arrays.update(
            basis_potential_active=basis_fields.potential.astype(np.float32),
            basis_velocity_u_active=basis_fields.velocity_u.astype(np.float32),
            basis_velocity_v_active=basis_fields.velocity_v.astype(np.float32),
            basis_vorticity_active=basis_fields.vorticity.astype(np.float32),
        )
    return arrays


def persist_prefix_state(
    output_dir: Path,
    state: CausalTransform,
    basis_fields: MimeticFields,
    complex_: MaskedCartesianComplex,
    *,
    prefixes: Iterable[int],
    provenance: dict[str, Any],
    commutation: dict[str, float],
    persist_fields: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:


    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = state_arrays(
        state, basis_fields, complex_, persist_fields=persist_fields
    )
    state_path = output_dir / "strict_prefix_state.npz"
    temporary = state_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, state_path)
    prefix_rows = prefix_diagnostics(state, basis_fields, prefixes)
    manifest = {
        "format": "airfoil_strict_nested_prefix_v1",
        "objective": state.objective,
        "modes": state.modes,
        "mgs_passes": state.passes,
        "transform_convention": (
            "basis_field = raw_field[:, causal_order] @ "
            "transform_sorted_triangular = raw_field @ "
            "transform_network_to_causal"
        ),
        "single_transform_for_all_fields": True,
        "independent_field_peeling": False,
        "triangular_lower_max_abs": state.triangular_lower_max_abs,
        "objective_orthogonality_max_abs": state.orthogonality_max_abs,
        "derivative_commutation": commutation,
        "prefixes": prefix_rows,
        "mimetic_complex": complex_diagnostics(complex_),
        "provenance": provenance,
        "state_npz": {
            "name": state_path.name,
            "sha256": file_sha256(state_path),
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": array_sha256(value),
                }
                for name, value in arrays.items()
            },
        },
    }
    manifest_path = output_dir / "strict_prefix_manifest.json"
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    os.replace(manifest_temporary, manifest_path)
    return state_path, manifest_path, manifest
