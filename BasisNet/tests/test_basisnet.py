import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basisnet import (  # noqa: E402
    AdaptiveController,
    BasisNet,
    BasisNetConfig,
    algorithm_manifest,
    antisymmetric_tensor,
    discrete_fields,
    infer_basis,
    least_squares_peel,
    make_optimizers,
    project_velocity,
    sss_step,
)


def small_config(dimension: int, ordering: str = "dynamic") -> BasisNetConfig:
    return BasisNetConfig(
        spatial_dim=dimension,
        modes=3,
        hidden=8,
        hidden_layers=2,
        frequencies=2,
        bounds=tuple((-math.pi, math.pi) for _ in range(dimension)),
        difference_step=0.05,
        ordering=ordering,
    )


def test_hard_divergence_identity_in_2d_and_3d():
    torch.manual_seed(1)
    for dimension in (2, 3):
        config = small_config(dimension)
        model = BasisNet(config).double()
        points = (2 * math.pi * torch.rand(12, dimension) - math.pi).double()
        fields = discrete_fields(
            model, points, torch.tensor([0.2], dtype=torch.float64)
        )
        assert torch.count_nonzero(fields["divergence"]).item() == 0


def test_antisymmetric_tensor_is_exact():
    entries = torch.randn(5, 3, dtype=torch.float64)
    tensor = antisymmetric_tensor(entries)
    assert torch.equal(tensor, -tensor.transpose(-1, -2))
    assert torch.count_nonzero(torch.diagonal(tensor, dim1=-2, dim2=-1)) == 0


def test_dynamic_sss_is_orthonormal_and_reports_order():
    torch.manual_seed(2)
    config = small_config(3)
    model = BasisNet(config).double()
    points = (2 * math.pi * torch.rand(24, 3) - math.pi).double()
    loss, diagnostics, output = sss_step(
        model,
        points,
        torch.tensor([0.4], dtype=torch.float64),
        (2 * math.pi) ** 3 / len(points),
    )
    assert torch.isfinite(loss)
    assert diagnostics["hard_discrete_divergence_max_abs"] == 0.0
    assert diagnostics["orthogonality_max_abs"] < 1.0e-8
    assert sorted(diagnostics["mode_order"]) == [0, 1, 2]
    assert output["velocity"].shape == (24, 3, 3)


def test_peeling_detaches_causal_predecessor():
    torch.manual_seed(3)
    previous_leaf = torch.randn(10, 2, dtype=torch.float64, requires_grad=True)
    previous = previous_leaf / torch.linalg.vector_norm(previous_leaf)
    raw = torch.randn(10, 2, dtype=torch.float64, requires_grad=True)
    previous_gradient = torch.randn(10, 2, 2, dtype=torch.float64)
    raw_gradient = torch.randn(10, 2, 2, dtype=torch.float64, requires_grad=True)
    previous_potential = torch.randn(10, 1, dtype=torch.float64)
    raw_potential = torch.randn(10, 1, dtype=torch.float64, requires_grad=True)
    result = least_squares_peel(
        raw,
        raw_gradient,
        raw_potential,
        [previous.detach()],
        [previous_gradient.detach()],
        [previous_potential.detach()],
        0.1,
        1.0e-12,
    )
    (result["stiffness"] / result["mass"]).backward()
    assert raw.grad is not None
    assert raw_gradient.grad is not None
    assert previous_leaf.grad is None


def test_adaptive_controller_reduces_lr_and_stops():
    model = BasisNet(small_config(2))
    optimizers = make_optimizers(model, 1.0e-3)
    controller = AdaptiveController(
        lr_patience=2,
        early_stop_patience=3,
        factor=0.5,
        min_lr=1.0e-4,
        min_delta=1.0e-4,
    )
    controller.update(1.0, optimizers)
    controller.update(1.0, optimizers)
    reduced = controller.update(1.0, optimizers)
    stopped = controller.update(1.0, optimizers)
    assert reduced["lr_reduced"]
    assert reduced["learning_rate"] == 5.0e-4
    assert stopped["should_stop"]


def test_manifest_excludes_gt_and_mesh():
    manifest = algorithm_manifest(small_config(3))
    assert manifest["dynamic_causal_sorting"]
    assert manifest["hard_discrete_divergence"]
    assert not manifest["gt_or_supervision_used_in_basis_training"]
    assert not manifest["mesh_required"]


def test_one_shot_inference_and_projection_close_the_workflow():
    torch.manual_seed(4)
    config = small_config(2)
    model = BasisNet(config).double()
    points = (2 * math.pi * torch.rand(20, 2) - math.pi).double()
    _, basis = infer_basis(
        model,
        points,
        torch.tensor([0.6], dtype=torch.float64),
        (2 * math.pi) ** 2 / len(points),
    )
    coefficients = torch.tensor([0.7, -0.2, 0.4], dtype=torch.float64)
    target = torch.einsum("k,nkd->nd", coefficients, basis["velocity"])
    projection = project_velocity(
        target, basis["velocity"], (2 * math.pi) ** 2 / len(points)
    )
    assert projection["relative_error"] < 1.0e-10
    assert torch.allclose(
        projection["coefficients"], coefficients, atol=1.0e-10, rtol=1.0e-10
    )
