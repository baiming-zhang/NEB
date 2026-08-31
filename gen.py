"""Generic Geometry-conditioned Eigenfluid Net (GEN) core.

This module contains the case-independent algorithm used by Eigenfluid:

* independent Fourier/SIREN neural fields for every mode;
* hard divergence-free 2D streamfunction and 3D antisymmetric-tensor fields;
* one commuting centered-difference operator for velocity and its gradient;
* fixed or dynamic pre-projection Rayleigh-energy ordering;
* gradient-causal least-squares residual peeling;
* shared peeling coefficients for potential, velocity, and velocity gradient;
* normalized Dirichlet/Rayleigh loss, orthogonality diagnostics, and rank;
* adaptive learning-rate reduction and validation early stopping.

Cases supply geometry, boundary envelopes, collocation sampling, physical
parameters, and quadrature weights.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import nn

Tensor = torch.Tensor
Envelope = Callable[[Tensor, Tensor], Tensor]
Ordering = Literal["fixed", "dynamic"]
Activation = Literal["sine", "elu"]


@dataclass(frozen=True)
class GENConfig:
    """Architecture and SSS settings independent of any flow case."""

    spatial_dim: Literal[2, 3] = 3
    parameter_dim: int = 1
    modes: int = 16
    hidden: int = 64
    hidden_layers: int = 4
    frequencies: int = 6
    bounds: tuple[tuple[float, float], ...] = (
        (-math.pi, math.pi),
        (-math.pi, math.pi),
        (-math.pi, math.pi),
    )
    periodic: bool = True
    difference_step: float = 0.02
    ordering: Ordering = "dynamic"
    mass_floor: float = 1.0e-12
    rayleigh_scale: float = 80.0
    grad_clip: float = 1.0
    activation: Activation = "sine"

    def __post_init__(self) -> None:
        if self.spatial_dim not in (2, 3):
            raise ValueError("spatial_dim must be 2 or 3")
        if len(self.bounds) != self.spatial_dim:
            raise ValueError("bounds must contain one interval per spatial axis")
        if self.parameter_dim < 0 or self.modes < 1:
            raise ValueError("parameter_dim must be nonnegative and modes positive")
        if self.hidden < 1 or self.hidden_layers < 1 or self.frequencies < 1:
            raise ValueError("network sizes and frequencies must be positive")
        if self.difference_step <= 0.0 or self.mass_floor <= 0.0:
            raise ValueError("difference_step and mass_floor must be positive")
        if self.ordering not in ("fixed", "dynamic"):
            raise ValueError("ordering must be 'fixed' or 'dynamic'")
        if self.activation not in ("sine", "elu"):
            raise ValueError("activation must be 'sine' or 'elu'")

    @property
    def potential_dim(self) -> int:
        """One streamfunction in 2D or three independent A_ij entries in 3D."""
        return 1 if self.spatial_dim == 2 else 3


def _parameter_rows(parameters: Tensor, count: int, width: int) -> Tensor:
    if width == 0:
        return parameters.new_zeros((count, 0))
    if parameters.ndim == 1:
        parameters = parameters.unsqueeze(0)
    if parameters.shape == (1, width):
        return parameters.expand(count, -1)
    if parameters.shape != (count, width):
        raise ValueError(
            f"parameters must have shape ({width},), (1,{width}), or "
            f"({count},{width}); got {tuple(parameters.shape)}"
        )
    return parameters


class FourierSineMode(nn.Module):
    """One independent Fourier-feature potential field.

    The historical class name is retained for checkpoint/API compatibility;
    ``config.activation`` selects either the original sine hidden layers or
    the ELU ablation.
    """

    def __init__(self, config: GENConfig) -> None:
        super().__init__()
        self.config = config
        coordinate_features = 2 * config.spatial_dim * config.frequencies
        linear_coordinates = 0 if config.periodic else config.spatial_dim
        in_features = coordinate_features + linear_coordinates + config.parameter_dim
        layers = [nn.Linear(in_features, config.hidden)]
        layers.extend(
            nn.Linear(config.hidden, config.hidden)
            for _ in range(config.hidden_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(config.hidden, config.potential_dim)

    def encode(self, points: Tensor, parameters: Tensor) -> Tensor:
        bounds = points.new_tensor(self.config.bounds)
        normal = (points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        frequencies = 2.0 ** torch.arange(
            self.config.frequencies, device=points.device, dtype=points.dtype
        )
        phase = 2.0 * math.pi * normal.unsqueeze(-1) * frequencies
        features = [torch.sin(phase).flatten(1), torch.cos(phase).flatten(1)]
        if not self.config.periodic:
            features.append(2.0 * normal - 1.0)
        features.append(
            _parameter_rows(parameters, len(points), self.config.parameter_dim)
        )
        return torch.cat(features, dim=1)

    def forward(self, points: Tensor, parameters: Tensor) -> Tensor:
        value = self.encode(points, parameters)
        for layer in self.layers:
            value = layer(value)
            if self.config.activation == "sine":
                value = torch.sin(value)
            else:
                value = torch.nn.functional.elu(value)
        return self.output(value)


class GEN(nn.Module):
    """Independent neural fields with permanent mode identities."""

    def __init__(self, config: GENConfig) -> None:
        super().__init__()
        self.config = config
        self.modes = nn.ModuleList(FourierSineMode(config) for _ in range(config.modes))

    def forward(
        self,
        points: Tensor,
        parameters: Tensor,
        envelope: Envelope | None = None,
    ) -> Tensor:
        """Return potentials with shape ``[points, modes, potential_dim]``."""
        value = torch.stack([mode(points, parameters) for mode in self.modes], dim=1)
        if envelope is None:
            return value
        factor = envelope(points, parameters)
        if factor.ndim == 1:
            factor = factor[:, None]
        if factor.shape != (len(points), 1):
            raise ValueError("envelope must return shape [points] or [points,1]")
        return value * factor[:, None, :]


def antisymmetric_tensor(entries: Tensor) -> Tensor:
    """Construct a hard-coded 3D antisymmetric tensor from (A12,A13,A23)."""
    if entries.shape[-1] != 3:
        raise ValueError("3D antisymmetric tensors require three entries")
    a12, a13, a23 = entries.unbind(dim=-1)
    zero = torch.zeros_like(a12)
    return torch.stack(
        (
            torch.stack((zero, a12, a13), dim=-1),
            torch.stack((-a12, zero, a23), dim=-1),
            torch.stack((-a13, -a23, zero), dim=-1),
        ),
        dim=-2,
    )


def _second_derivative_offsets(dimension: int) -> list[tuple[int, ...]]:
    offsets: set[tuple[int, ...]] = {tuple([0] * dimension)}
    for axis in range(dimension):
        for sign in (-1, 1):
            item = [0] * dimension
            item[axis] = sign
            offsets.add(tuple(item))
    for first in range(dimension):
        for second in range(first + 1, dimension):
            for sign_first in (-1, 1):
                for sign_second in (-1, 1):
                    item = [0] * dimension
                    item[first], item[second] = sign_first, sign_second
                    offsets.add(tuple(item))
    return sorted(offsets)


def discrete_fields(
    model: GEN,
    points: Tensor,
    parameters: Tensor,
    *,
    envelope: Envelope | None = None,
) -> dict[str, Tensor]:
    """Evaluate potentials, velocity, gradients, and hard discrete divergence.

    The same centered-difference operator is used in every direction. Mixed
    derivatives are computed once and mirrored, so ``div(curl)=0`` cancels
    algebraically in both supported dimensions.
    """
    config = model.config
    dimension = config.spatial_dim
    h = config.difference_step
    offsets = _second_derivative_offsets(dimension)
    shifted = torch.cat(
        [points + points.new_tensor(offset).mul(h) for offset in offsets], dim=0
    )
    parameter_rows = _parameter_rows(parameters, len(points), config.parameter_dim)
    shifted_parameters = parameter_rows.repeat(len(offsets), 1)
    values_all = model(shifted, shifted_parameters, envelope).reshape(
        len(offsets), len(points), config.modes, config.potential_dim
    )
    values = {offset: values_all[index] for index, offset in enumerate(offsets)}
    origin_key = tuple([0] * dimension)
    origin = values[origin_key]

    first = origin.new_empty(
        (len(points), config.modes, config.potential_dim, dimension)
    )
    for axis in range(dimension):
        plus, minus = [0] * dimension, [0] * dimension
        plus[axis], minus[axis] = 1, -1
        first[..., axis] = (values[tuple(plus)] - values[tuple(minus)]) / (2 * h)

    if dimension == 2:
        velocity = torch.stack((first[:, :, 0, 1], -first[:, :, 0, 0]), dim=2)
    else:
        a, b, c = 0, 1, 2
        velocity = torch.stack(
            (
                first[:, :, a, 1] + first[:, :, b, 2],
                -first[:, :, a, 0] + first[:, :, c, 2],
                -first[:, :, b, 0] - first[:, :, c, 1],
            ),
            dim=2,
        )

    hessian = origin.new_empty(
        (len(points), config.modes, config.potential_dim, dimension, dimension)
    )
    for first_axis in range(dimension):
        plus, minus = [0] * dimension, [0] * dimension
        plus[first_axis], minus[first_axis] = 1, -1
        hessian[..., first_axis, first_axis] = (
            values[tuple(plus)] - 2.0 * origin + values[tuple(minus)]
        ) / (h * h)
        for second_axis in range(first_axis + 1, dimension):
            pp, pm = [0] * dimension, [0] * dimension
            mp, mm = [0] * dimension, [0] * dimension
            pp[first_axis], pp[second_axis] = 1, 1
            pm[first_axis], pm[second_axis] = 1, -1
            mp[first_axis], mp[second_axis] = -1, 1
            mm[first_axis], mm[second_axis] = -1, -1
            mixed = (
                values[tuple(pp)]
                - values[tuple(pm)]
                - values[tuple(mp)]
                + values[tuple(mm)]
            ) / (4.0 * h * h)
            hessian[..., first_axis, second_axis] = mixed
            hessian[..., second_axis, first_axis] = mixed

    gradient = origin.new_empty((len(points), config.modes, dimension, dimension))
    if dimension == 2:
        for derivative in range(2):
            gradient[:, :, 0, derivative] = hessian[:, :, 0, 1, derivative]
            gradient[:, :, 1, derivative] = -hessian[:, :, 0, 0, derivative]
    else:
        for derivative in range(3):
            gradient[:, :, 0, derivative] = (
                hessian[:, :, 0, 1, derivative] + hessian[:, :, 1, 2, derivative]
            )
            gradient[:, :, 1, derivative] = (
                -hessian[:, :, 0, 0, derivative] + hessian[:, :, 2, 2, derivative]
            )
            gradient[:, :, 2, derivative] = (
                -hessian[:, :, 1, 0, derivative] - hessian[:, :, 2, 1, derivative]
            )
    if dimension == 2:
        divergence = hessian[:, :, 0, 1, 0] - hessian[:, :, 0, 0, 1]
    else:
        # Pair the commuting mixed derivatives before summation. Each pair is
        # the same stored tensor entry minus itself, so the result is exactly
        # zero rather than a roundoff-sized reassociation residual.
        divergence = (
            hessian[:, :, 0, 1, 0]
            - hessian[:, :, 0, 0, 1]
            + hessian[:, :, 1, 2, 0]
            - hessian[:, :, 1, 0, 2]
            + hessian[:, :, 2, 2, 1]
            - hessian[:, :, 2, 1, 2]
        )
    return {
        "potential": origin,
        "velocity": velocity,
        "gradient": gradient,
        "divergence": divergence,
    }


def _weights(weights: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(weights, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(len(reference))
    if value.shape != (len(reference),):
        raise ValueError("quadrature weights must be scalar or shape [points]")
    return value


def raw_rayleigh_energies(
    fields: dict[str, Tensor], weights: Tensor | float, mass_floor: float
) -> Tensor:
    """Pre-projection normalized Dirichlet energies for causal sorting."""
    weight = _weights(weights, fields["velocity"])
    mass = torch.einsum("n,nkd,nkd->k", weight, fields["velocity"], fields["velocity"])
    stiffness = torch.einsum(
        "n,nkij,nkij->k", weight, fields["gradient"], fields["gradient"]
    )
    return stiffness / torch.clamp(mass, min=mass_floor)


def causal_mode_order(
    fields: dict[str, Tensor],
    weights: Tensor | float,
    config: GENConfig,
) -> tuple[Tensor, list[int]]:
    energies = raw_rayleigh_energies(fields, weights, config.mass_floor)
    if config.ordering == "fixed":
        return energies, list(range(config.modes))
    return energies, torch.argsort(energies.detach(), stable=True).cpu().tolist()


def least_squares_peel(
    raw_velocity: Tensor,
    raw_gradient: Tensor,
    raw_potential: Tensor,
    previous_velocity: Sequence[Tensor],
    previous_gradient: Sequence[Tensor],
    previous_potential: Sequence[Tensor],
    weights: Tensor | float,
    mass_floor: float,
) -> dict[str, Tensor]:
    """Peel one mode and apply identical coefficients to all field levels."""
    weight = _weights(weights, raw_velocity)
    if previous_velocity:
        velocity_basis = torch.stack(tuple(previous_velocity), dim=1)
        gradient_basis = torch.stack(tuple(previous_gradient), dim=1)
        potential_basis = torch.stack(tuple(previous_potential), dim=1)
        # Sequential peeling has already made the causal predecessors
        # orthonormal under this quadrature. The least-squares Gram system is
        # therefore the identity and its coefficients reduce exactly to
        # weighted inner products, avoiding a redundant cubic solve.
        coefficients = torch.einsum("n,njd,nd->j", weight, velocity_basis, raw_velocity)
        reconstructed_velocity = torch.einsum("njd,j->nd", velocity_basis, coefficients)
        reconstructed_gradient = torch.einsum(
            "njab,j->nab", gradient_basis, coefficients
        )
        reconstructed_potential = torch.einsum(
            "njp,j->np", potential_basis, coefficients
        )
    else:
        coefficients = raw_velocity.new_empty(0)
        reconstructed_velocity = torch.zeros_like(raw_velocity)
        reconstructed_gradient = torch.zeros_like(raw_gradient)
        reconstructed_potential = torch.zeros_like(raw_potential)
    velocity = raw_velocity - reconstructed_velocity
    gradient = raw_gradient - reconstructed_gradient
    potential = raw_potential - reconstructed_potential
    mass = torch.einsum("n,nd,nd->", weight, velocity, velocity)
    stiffness = torch.einsum("n,nij,nij->", weight, gradient, gradient)
    projection_mass = torch.einsum(
        "n,nd,nd->", weight, reconstructed_velocity, reconstructed_velocity
    )
    scale = torch.rsqrt(torch.clamp(mass, min=mass_floor))
    return {
        "coefficients": coefficients,
        "velocity": velocity,
        "gradient": gradient,
        "potential": potential,
        "normalized_velocity": velocity * scale,
        "normalized_gradient": gradient * scale,
        "normalized_potential": potential * scale,
        "mass": mass,
        "stiffness": stiffness,
        "projection_mass": projection_mass,
    }


def sss_step(
    model: GEN,
    points: Tensor,
    parameters: Tensor,
    weights: Tensor | float,
    *,
    optimizers: Sequence[torch.optim.Optimizer] | None = None,
    envelope: Envelope | None = None,
) -> tuple[Tensor, dict[str, Any], dict[str, Tensor]]:
    """Run one complete fixed- or dynamic-order causal SSS step."""
    config = model.config
    fields = discrete_fields(model, points, parameters, envelope=envelope)
    raw_energies, order = causal_mode_order(fields, weights, config)
    weight = _weights(weights, fields["velocity"])
    previous_velocity: list[Tensor] = []
    previous_gradient: list[Tensor] = []
    previous_potential: list[Tensor] = []
    raw_velocities: list[Tensor] = []
    losses: list[Tensor] = []
    rayleigh_values: list[float] = []
    projection_values: list[float] = []

    for mode in order:
        raw_velocity = fields["velocity"][:, mode]
        raw_gradient = fields["gradient"][:, mode]
        raw_potential = fields["potential"][:, mode]
        raw_velocities.append(raw_velocity.detach())
        peeled = least_squares_peel(
            raw_velocity,
            raw_gradient,
            raw_potential,
            previous_velocity,
            previous_gradient,
            previous_potential,
            weight,
            config.mass_floor,
        )
        mass = torch.clamp(peeled["mass"], min=config.mass_floor)
        rayleigh = peeled["stiffness"] / mass
        losses.append(rayleigh / config.rayleigh_scale)
        rayleigh_values.append(float(rayleigh.detach().cpu()))
        projection_values.append(
            float((peeled["projection_mass"] / mass).detach().cpu())
        )
        # This detach is the gradient-causal filter: later losses cannot update
        # any mode that already dominates them in the current causal order.
        previous_velocity.append(peeled["normalized_velocity"].detach())
        previous_gradient.append(peeled["normalized_gradient"].detach())
        previous_potential.append(peeled["normalized_potential"].detach())

    loss = torch.stack(losses).mean()
    if optimizers is not None:
        if len(optimizers) != config.modes:
            raise ValueError("one optimizer is required per mode")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for mode, optimizer in enumerate(optimizers):
            torch.nn.utils.clip_grad_norm_(
                model.modes[mode].parameters(), config.grad_clip
            )
            optimizer.step()

    velocity_basis = torch.stack(previous_velocity, dim=1)
    potential_basis = torch.stack(previous_potential, dim=1)
    gram = torch.einsum("n,nid,njd->ij", weight, velocity_basis, velocity_basis)
    raw_basis = torch.stack(raw_velocities, dim=1)
    raw_gram = torch.einsum("n,nid,njd->ij", weight, raw_basis, raw_basis)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (raw_gram + raw_gram.T))
    rank_threshold = torch.clamp(eigenvalues[-1], min=config.mass_floor) * 1.0e-8
    identity = torch.eye(config.modes, device=gram.device, dtype=gram.dtype)
    diagnostics = {
        "loss": float(loss.detach().cpu()),
        "mean_rayleigh": sum(rayleigh_values) / len(rayleigh_values),
        "max_rayleigh": max(rayleigh_values),
        "mean_projection_ratio": sum(projection_values) / len(projection_values),
        "orthogonality_max_abs": float((gram - identity).abs().max().cpu()),
        "hard_discrete_divergence_max_abs": float(
            fields["divergence"].detach().abs().max().cpu()
        ),
        "effective_raw_rank": int((eigenvalues > rank_threshold).sum().cpu()),
        "raw_rayleigh_by_network": raw_energies.detach().cpu().tolist(),
        "peeled_rayleigh_in_causal_order": rayleigh_values,
        "mode_order": order,
        "ordering": config.ordering,
    }
    return (
        loss,
        diagnostics,
        {
            "velocity": velocity_basis,
            "potential": potential_basis,
            "gram": gram,
        },
    )


@torch.no_grad()
def infer_basis(
    model: GEN,
    points: Tensor,
    parameters: Tensor,
    weights: Tensor | float,
    *,
    envelope: Envelope | None = None,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Generate and causally peel a frozen basis in one model evaluation."""
    model.eval()
    _, diagnostics, basis = sss_step(
        model,
        points,
        parameters,
        weights,
        envelope=envelope,
    )
    return diagnostics, basis


@torch.no_grad()
def project_velocity(
    target: Tensor,
    basis: Tensor,
    weights: Tensor | float,
    *,
    mass_floor: float = 1.0e-12,
) -> dict[str, Tensor]:
    """Project independent velocity snapshots onto a frozen spatial basis.

    ``target`` may be ``[points,dimension]`` or
    ``[snapshots,points,dimension]``. ``basis`` is
    ``[points,modes,dimension]``. No basis interpolation, retraining, or
    per-parameter nonlinear optimization is performed.
    """
    squeeze = target.ndim == 2
    if squeeze:
        target = target.unsqueeze(0)
    if target.ndim != 3 or basis.ndim != 3:
        raise ValueError("target and basis must be velocity arrays")
    if target.shape[1] != basis.shape[0] or target.shape[2] != basis.shape[2]:
        raise ValueError("target and basis point/component dimensions must agree")
    weight = _weights(weights, basis)
    gram = torch.einsum("n,nkd,nld->kl", weight, basis, basis)
    rhs = torch.einsum("n,tnd,nkd->tk", weight, target, basis)
    identity = torch.eye(len(gram), device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(gram + mass_floor * identity, rhs.T).T
    reconstruction = torch.einsum("tk,nkd->tnd", coefficients, basis)
    residual = target - reconstruction
    numerator = torch.einsum("n,tnd,tnd->t", weight, residual, residual)
    denominator = torch.einsum("n,tnd,tnd->t", weight, target, target)
    relative_error = torch.sqrt(numerator / torch.clamp(denominator, min=mass_floor))
    if squeeze:
        coefficients = coefficients[0]
        reconstruction = reconstruction[0]
        relative_error = relative_error[0]
    return {
        "coefficients": coefficients,
        "reconstruction": reconstruction,
        "relative_error": relative_error,
        "gram": gram,
    }


def make_optimizers(
    model: GEN, learning_rate: float
) -> list[torch.optim.Optimizer]:
    """Create the algorithm's independent Adam optimizer for every mode."""
    return [
        torch.optim.Adam(mode.parameters(), lr=learning_rate) for mode in model.modes
    ]


@dataclass
class AdaptiveController:
    """Validation-driven LR reduction and material-improvement early stop."""

    lr_patience: int = 10
    early_stop_patience: int = 24
    factor: float = 0.5
    min_lr: float = 2.5e-5
    min_delta: float = 1.0e-4
    best: float = math.inf
    lr_bad_count: int = 0
    stop_bad_count: int = 0

    def update(
        self, validation_loss: float, optimizers: Sequence[torch.optim.Optimizer]
    ) -> dict[str, float | int | bool]:
        improved = self.best - validation_loss > self.min_delta
        if improved:
            self.best = validation_loss
            self.lr_bad_count = 0
            self.stop_bad_count = 0
        else:
            self.lr_bad_count += 1
            self.stop_bad_count += 1
        reduced = False
        if self.lr_bad_count >= self.lr_patience:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    new_lr = max(self.min_lr, float(group["lr"]) * self.factor)
                    reduced = reduced or new_lr < float(group["lr"])
                    group["lr"] = new_lr
            self.lr_bad_count = 0
        current_lr = min(
            float(group["lr"])
            for optimizer in optimizers
            for group in optimizer.param_groups
        )
        return {
            "improved": improved,
            "lr_reduced": reduced,
            "learning_rate": current_lr,
            "bad_validations": self.stop_bad_count,
            "should_stop": self.stop_bad_count >= self.early_stop_patience,
            "best_validation_loss": self.best,
        }


def algorithm_manifest(config: GENConfig) -> dict[str, Any]:
    """Serializable description stored beside checkpoints by a case trainer."""
    return {
        "algorithm": "sequential_spectral_subspace",
        "config": asdict(config),
        "independent_mode_networks": True,
        "dynamic_causal_sorting": config.ordering == "dynamic",
        "gradient_causal_filtering": True,
        "least_squares_residual_peeling": True,
        "shared_coefficients_for_potential_velocity_gradient": True,
        "hard_discrete_divergence": True,
        "antisymmetric_tensor_3d": config.spatial_dim == 3,
        "streamfunction_2d": config.spatial_dim == 2,
        "gt_or_supervision_used_in_basis_training": False,
        "mesh_required": False,
        "loss": "normalized Dirichlet/Rayleigh energy only",
        "activation": config.activation,
    }


def self_test() -> dict[str, Any]:
    """Small CPU test of the generic 2D/3D hard constraints and SSS core."""
    torch.manual_seed(7)
    results: dict[str, Any] = {}
    for dimension in (2, 3):
        bounds = tuple((-math.pi, math.pi) for _ in range(dimension))
        config = GENConfig(
            spatial_dim=dimension,
            modes=3,
            hidden=8,
            hidden_layers=2,
            frequencies=2,
            bounds=bounds,
            difference_step=0.05,
        )
        model = GEN(config).double()
        points = (2.0 * math.pi * torch.rand(20, dimension) - math.pi).double()
        parameters = torch.tensor([0.37], dtype=torch.float64)
        fields = discrete_fields(model, points, parameters)
        loss, diagnostics, output = sss_step(
            model, points, parameters, (2.0 * math.pi) ** dimension / len(points)
        )
        if torch.count_nonzero(fields["divergence"]).item() != 0:
            raise RuntimeError(f"{dimension}D hard divergence identity failed")
        if not torch.isfinite(loss) or diagnostics["orthogonality_max_abs"] > 1.0e-8:
            raise RuntimeError(f"{dimension}D SSS peeling failed")
        results[f"{dimension}d"] = {
            "hard_discrete_divergence": diagnostics["hard_discrete_divergence_max_abs"],
            "orthogonality_max_abs": diagnostics["orthogonality_max_abs"],
            "mode_order": diagnostics["mode_order"],
            "basis_shape": list(output["velocity"].shape),
        }
    results["self_test"] = "passed"
    return results


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))
