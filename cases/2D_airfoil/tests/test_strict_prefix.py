"""CPU tests for one-transform nested airfoil prefixes."""

from __future__ import annotations

import json

import numpy as np

from eigenfluid_sss.basisnet.airfoil2d.mimetic_fd import (
    build_masked_cartesian_complex,
)
from eigenfluid_sss.basisnet.airfoil2d.strict_prefix import (
    FORMAL_FULL_MODES,
    FORMAL_SAMPLING_MODE,
    FORMAL_SAMPLING_RNG,
    STRICT_PREFIXES,
    apply_network_transform,
    array_sha256,
    build_causal_transform,
    derivative_commutation_audit,
    derive_mimetic_fields,
    file_sha256,
    objective_grams,
    persist_prefix_state,
    prefix_diagnostics,
)


def test_formal_prefix_contract_is_k2048():
    assert FORMAL_FULL_MODES == 2048
    assert FORMAL_SAMPLING_MODE == "uniform_random"
    assert FORMAL_SAMPLING_RNG == "numpy.random.PCG64"
    assert STRICT_PREFIXES == (64, 128, 256, 512, 1024, 2048)


def _synthetic_state(modes: int = 6):
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(-0.5, 0.5, 11)
    solid = np.zeros((len(y), len(x)), dtype=bool)
    complex_ = build_masked_cartesian_complex(x, y, solid)
    xx, yy = np.meshgrid(x, y)
    potential = np.empty((*xx.shape, modes), dtype=np.float64)
    envelope = x[None, :] * np.sin(np.pi * (yy + 0.5))
    for mode in range(modes):
        frequency = mode + 1
        potential[..., mode] = envelope * (
            np.cos(frequency * np.pi * xx)
            + 0.17 * np.sin((frequency + 1) * np.pi * xx)
            + 0.03 * frequency * yy
        )
    mixing = np.eye(modes)
    mixing[:, -1] += 0.35 * mixing[:, 0]
    potential = potential @ mixing
    return complex_, derive_mimetic_fields(complex_, potential)


def test_causal_transform_matches_explicit_objective_gram_and_is_triangular():
    complex_, raw = _synthetic_state()
    for objective in ("potential", "velocity", "vorticity"):
        mass, stiffness = objective_grams(complex_, raw, objective)
        state = build_causal_transform(mass, stiffness, objective)
        order = state.causal_order
        sorted_mass = mass[np.ix_(order, order)]
        gram = state.transform_sorted.T @ sorted_mass @ state.transform_sorted
        np.testing.assert_allclose(gram, np.eye(raw.modes), rtol=0.0, atol=2.0e-11)
        assert state.passes == 2
        assert state.triangular_lower_max_abs == 0.0
        assert np.allclose(
            raw.potential[:, order] @ state.transform_sorted,
            raw.potential @ state.transform_network,
            rtol=0.0,
            atol=2.0e-12,
        )


def test_one_transform_commutes_with_mimetic_derivatives_and_prefixes_are_nested():
    complex_, raw = _synthetic_state()
    mass, stiffness = objective_grams(complex_, raw, "velocity")
    state = build_causal_transform(mass, stiffness, "velocity")
    basis = apply_network_transform(raw, state.transform_network)
    for name in ("potential", "velocity_u", "velocity_v", "vorticity"):
        np.testing.assert_allclose(
            getattr(basis, name),
            getattr(raw, name) @ state.transform_network,
            rtol=0.0,
            atol=2.0e-12,
        )
    audit = derivative_commutation_audit(complex_, raw, basis)
    assert max(audit.values()) < 5.0e-12

    rows = prefix_diagnostics(state, basis, (2, 4, raw.modes))
    assert [row["column_slice"] for row in rows] == [[0, 2], [0, 4], [0, 6]]
    for small, large in ((2, 4), (4, 6)):
        np.testing.assert_array_equal(
            basis.potential[:, :small], basis.potential[:, :large][:, :small]
        )
        np.testing.assert_array_equal(
            basis.vorticity[:, :small], basis.vorticity[:, :large][:, :small]
        )
        np.testing.assert_allclose(
            basis.velocity_u[:, :small],
            raw.velocity_u @ state.transform_network[:, :small],
            rtol=0.0,
            atol=2.0e-12,
        )


def test_prefix_npz_and_manifest_are_hash_anchored(tmp_path):
    complex_, raw = _synthetic_state(modes=4)
    mass, stiffness = objective_grams(complex_, raw, "vorticity")
    state = build_causal_transform(mass, stiffness, "vorticity")
    basis = apply_network_transform(raw, state.transform_network)
    commutation = derivative_commutation_audit(complex_, raw, basis)
    state_path, manifest_path, manifest = persist_prefix_state(
        tmp_path,
        state,
        basis,
        complex_,
        prefixes=(2, 4),
        provenance={"synthetic": True},
        commutation=commutation,
    )
    loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest == manifest
    assert loaded_manifest["state_npz"]["sha256"] == file_sha256(state_path)
    assert loaded_manifest["single_transform_for_all_fields"] is True
    assert loaded_manifest["independent_field_peeling"] is False
    with np.load(state_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["transform_network_to_causal"], state.transform_network
        )
        assert archive["basis_potential_active"].shape[1] == 4
        assert archive["basis_vorticity_active"].shape[1] == 4
        assert manifest["prefixes"][0]["field_sha256"]["potential"] == (
            array_sha256(archive["basis_potential_active"][:, :2])
        )
