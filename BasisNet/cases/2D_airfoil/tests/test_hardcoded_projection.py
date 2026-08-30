import numpy as np

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_postprocess import (
    certify_stokes_operators,
    generalized_symmetric_eig,
    project_ritz_velocity,
)


def test_projection_grid_whitening_matches_direct_least_squares():
    rng = np.random.default_rng(20260810)
    rows, modes = 257, 9
    raw_u = rng.normal(size=(rows, modes))
    raw_v = rng.normal(size=(rows, modes))
    target_u = rng.normal(size=rows)
    target_v = rng.normal(size=rows)
    # Deliberately destroy the construction-measure orthogonality before
    # projecting on this independent grid.
    rotation = rng.normal(size=(modes, modes))

    _, raw_modal, diagnostics = project_ritz_velocity(
        raw_u,
        raw_v,
        target_u,
        target_v,
        rotation,
    )
    stacked_basis = np.vstack((raw_u, raw_v))
    stacked_target = np.concatenate((target_u, target_v))
    expected, *_ = np.linalg.lstsq(stacked_basis, stacked_target, rcond=None)

    assert diagnostics["rank"] == modes
    assert diagnostics["prewhiten_orthogonality_error"] > 1.0e-2
    assert diagnostics["orthogonality_error"] < 1.0e-12
    assert np.allclose(raw_modal, expected, rtol=1.0e-10, atol=1.0e-10)


def test_projection_discards_only_backward_unresolved_direction():
    rng = np.random.default_rng(7)
    rows = 128
    independent = rng.normal(size=(rows, 3))
    raw_u = np.column_stack((independent, independent[:, 1]))
    raw_v = np.column_stack((rng.normal(size=(rows, 3)), np.zeros(rows)))
    # Make the fourth full velocity column exactly equal to the second.
    raw_v[:, 3] = raw_v[:, 1]
    target_u = rng.normal(size=rows)
    target_v = rng.normal(size=rows)

    _, raw_modal, diagnostics = project_ritz_velocity(
        raw_u,
        raw_v,
        target_u,
        target_v,
        np.eye(4),
    )
    stacked_basis = np.vstack((raw_u, raw_v))
    stacked_target = np.concatenate((target_u, target_v))
    expected = np.linalg.pinv(stacked_basis) @ stacked_target

    assert diagnostics["rank"] == 3
    assert diagnostics["discarded_rank"] == 1
    assert diagnostics["orthogonality_error"] < 1.0e-12
    assert np.allclose(raw_modal, expected, rtol=1.0e-10, atol=1.0e-10)


def test_generalized_stokes_rank_uses_only_backward_error():
    mass = np.diag([1.0, 0.5, 0.0])
    stiffness = np.diag([2.0, 3.0, 7.0])
    values, vectors, rank = generalized_symmetric_eig(
        stiffness, mass, 3, sample_rows=128
    )
    assert rank == 2
    assert len(values) == 2
    assert np.allclose(vectors.T @ mass @ vectors, np.eye(2))
    assert np.allclose(
        stiffness @ vectors,
        mass @ vectors * values[None, :],
    )


def test_independent_stokes_certificate_rejects_first_unstable_prefix():
    mass = np.eye(3)
    stiffness = np.diag([1.0, 2.0, 3.0])
    stable = certify_stokes_operators(
        mass,
        stiffness,
        mass,
        stiffness,
        mass,
        stiffness,
        3,
        construction_rows=256,
        independent_rows=256,
    )
    assert stable["certified_count"] == 3

    independent_stiffness = stiffness.copy()
    independent_stiffness[1, 2] = independent_stiffness[2, 1] = 1.0e-4
    unstable = certify_stokes_operators(
        mass,
        stiffness,
        mass,
        independent_stiffness,
        mass,
        independent_stiffness,
        3,
        construction_rows=256,
        independent_rows=256,
    )
    assert unstable["certified_count"] == 1
    assert unstable["ritz_residual_relative"][1] > unstable[
        "ritz_residual_budget"
    ][1]
