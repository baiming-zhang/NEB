"""Network-only CUDA Jet3 validation and controlled MLP ablations.

This program deliberately does not load a flow checkpoint or execute any
fluid, quadrature, Ritz, peeling, or training code.  It tests a small MLP with
one shared topology and one shared set of weights under four controlled
network variants:

* ReLU with raw xyz input;
* ReLU with positional encoding;
* SIREN with raw xyz input;
* SIREN with positional encoding.

The native CuPy implementation is the reference.  PyTorch autograd receives
the identical inputs and weights and is compared slot by slot through the
complete symmetric third-order spatial jet.  Hand-written parameter reverse
is compared layer by layer.  Finite differences are compared by derivative
order over a step-size sweep.  ReLU samples are selected away from kinks so
the derivative comparison has an unambiguous local meaning.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cupy as cp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


SEED = 20260801
DIMENSION = 3
FEATURES = 12
WIDTH = 32
DEPTH = 3
OUTPUTS = 2
POINTS = 8
FREQUENCIES = (1.0, 2.0)
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)

FIRST = ((0,), (1,), (2,))
SECOND = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
THIRD = (
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2),
    (0, 2, 2), (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2),
)
MULTI_INDICES = ((),) + FIRST + SECOND + THIRD
AXIS_NAMES = "xyz"
SLOT_NAMES = [
    "value",
    *[AXIS_NAMES[i[0]] for i in FIRST],
    *["".join(AXIS_NAMES[j] for j in i) for i in SECOND],
    *["".join(AXIS_NAMES[j] for j in i) for i in THIRD],
]


@dataclass
class Jet3:
    value: cp.ndarray
    grad: cp.ndarray
    hess: cp.ndarray
    third: cp.ndarray


def input_jets(points: np.ndarray, encoding: str, dtype: np.dtype) -> Jet3:
    points_gpu = cp.asarray(points, dtype=dtype)
    count = len(points)
    raw_value = cp.zeros((count, FEATURES), dtype=dtype)
    raw_value[:, :DIMENSION] = points_gpu
    raw_grad = cp.zeros((count, FEATURES, DIMENSION), dtype=dtype)
    for axis in range(DIMENSION):
        raw_grad[:, axis, axis] = 1.0
    raw = Jet3(
        raw_value,
        raw_grad,
        cp.zeros((count, FEATURES, DIMENSION, DIMENSION), dtype=dtype),
        cp.zeros(
            (count, FEATURES, DIMENSION, DIMENSION, DIMENSION), dtype=dtype
        ),
    )
    if encoding == "raw_xyz":
        return raw

    features: list[Jet3] = []
    for axis in range(DIMENSION):
        source = Jet3(
            raw.value[:, axis : axis + 1],
            raw.grad[:, axis : axis + 1],
            raw.hess[:, axis : axis + 1],
            raw.third[:, axis : axis + 1],
        )
        for frequency in FREQUENCIES:
            scaled = scale_jet(source, frequency)
            features.append(unary_jet(scaled, "sine"))
            features.append(unary_jet(scaled, "cosine"))
    return concatenate_jets(features)


def concatenate_jets(items: list[Jet3]) -> Jet3:
    return Jet3(
        cp.concatenate([item.value for item in items], axis=1),
        cp.concatenate([item.grad for item in items], axis=1),
        cp.concatenate([item.hess for item in items], axis=1),
        cp.concatenate([item.third for item in items], axis=1),
    )


def scale_jet(source: Jet3, factor: float) -> Jet3:
    return Jet3(
        source.value * factor,
        source.grad * factor,
        source.hess * factor,
        source.third * factor,
    )


def unary_jet(source: Jet3, kind: str) -> Jet3:
    value = source.value
    if kind == "sine":
        f0, f1 = cp.sin(value), cp.cos(value)
        f2, f3 = -f0, -f1
    elif kind == "cosine":
        f0, f1 = cp.cos(value), -cp.sin(value)
        f2, f3 = -f0, -f1
    elif kind == "relu":
        f0 = cp.maximum(value, 0.0)
        f1 = (value > 0.0).astype(value.dtype)
        f2 = cp.zeros_like(value)
        f3 = cp.zeros_like(value)
    else:
        raise ValueError(kind)

    grad = f1[..., None] * source.grad
    hess = (
        f1[..., None, None] * source.hess
        + f2[..., None, None]
        * cp.einsum("nfi,nfj->nfij", source.grad, source.grad)
    )
    third = f1[..., None, None, None] * source.third
    third += f2[..., None, None, None] * (
        cp.einsum("nfij,nfk->nfijk", source.hess, source.grad)
        + cp.einsum("nfik,nfj->nfijk", source.hess, source.grad)
        + cp.einsum("nfjk,nfi->nfijk", source.hess, source.grad)
    )
    third += f3[..., None, None, None] * cp.einsum(
        "nfi,nfj,nfk->nfijk", source.grad, source.grad, source.grad
    )
    return Jet3(f0, grad, hess, third)


def linear_jet(source: Jet3, weight: cp.ndarray, bias: cp.ndarray) -> Jet3:
    return Jet3(
        source.value @ weight.T + bias,
        cp.einsum("nfi,of->noi", source.grad, weight),
        cp.einsum("nfij,of->noij", source.hess, weight),
        cp.einsum("nfijk,of->noijk", source.third, weight),
    )


def native_jet_forward(
    points: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    encoding: str,
    dtype: np.dtype,
) -> tuple[Jet3, float]:
    jet = input_jets(points, encoding, dtype)
    minimum_preactivation = math.inf
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        jet = linear_jet(
            jet, cp.asarray(weight, dtype=dtype), cp.asarray(bias, dtype=dtype)
        )
        if layer + 1 < len(weights):
            minimum_preactivation = min(
                minimum_preactivation,
                float(cp.min(cp.abs(jet.value)).get()),
            )
            jet = unary_jet(jet, "relu" if activation == "relu" else "sine")
    return jet, minimum_preactivation


def numpy_features(points: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "raw_xyz":
        value = np.zeros((len(points), FEATURES), dtype=np.float64)
        value[:, :DIMENSION] = points
        return value
    values = []
    for axis in range(DIMENSION):
        for frequency in FREQUENCIES:
            phase = frequency * points[:, axis : axis + 1]
            values.extend((np.sin(phase), np.cos(phase)))
    return np.concatenate(values, axis=1)


def numpy_value_forward(
    points: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    encoding: str,
) -> tuple[np.ndarray, float]:
    value = numpy_features(np.asarray(points, dtype=np.float64), encoding)
    minimum = math.inf
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ weight.astype(np.float64).T + bias.astype(np.float64)
        if layer + 1 < len(weights):
            minimum = min(minimum, float(np.min(np.abs(value))))
            value = np.maximum(value, 0.0) if activation == "relu" else np.sin(value)
    return value, minimum


def torch_features(points: torch.Tensor, encoding: str) -> torch.Tensor:
    if encoding == "raw_xyz":
        zeros = torch.zeros(
            (points.shape[0], FEATURES - DIMENSION),
            dtype=points.dtype,
            device=points.device,
        )
        return torch.cat((points, zeros), dim=1)
    features = []
    for axis in range(DIMENSION):
        for frequency in FREQUENCIES:
            phase = frequency * points[:, axis : axis + 1]
            features.extend((torch.sin(phase), torch.cos(phase)))
    return torch.cat(features, dim=1)


def torch_value_forward(
    points: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    activation: str,
    encoding: str,
) -> torch.Tensor:
    value = torch_features(points, encoding)
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ weight.T + bias
        if layer + 1 < len(weights):
            value = torch.relu(value) if activation == "relu" else torch.sin(value)
    return value


def torch_jet_forward(
    points: np.ndarray,
    weights_np: list[np.ndarray],
    biases_np: list[np.ndarray],
    activation: str,
    encoding: str,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    dtype = torch.float32
    weights = [torch.tensor(item, dtype=dtype, device=device) for item in weights_np]
    biases = [torch.tensor(item, dtype=dtype, device=device) for item in biases_np]
    values, grads, hessians, thirds = [], [], [], []

    def safe_grad(scalar: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if not scalar.requires_grad:
            return source * 0.0
        result = torch.autograd.grad(
            scalar,
            source,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return source * 0.0 if result is None else result

    for point in points:
        x = torch.tensor(point[None], dtype=dtype, device=device, requires_grad=True)
        output = torch_value_forward(x, weights, biases, activation, encoding)[0]
        point_values, point_grads, point_hessians, point_thirds = [], [], [], []
        for output_index in range(OUTPUTS):
            scalar = output[output_index]
            gradient = safe_grad(scalar, x)[0]
            hessian_rows = []
            third_planes = []
            for i in range(DIMENSION):
                row = safe_grad(gradient[i], x)[0]
                hessian_rows.append(row)
                plane_rows = []
                for j in range(DIMENSION):
                    plane_rows.append(safe_grad(row[j], x)[0])
                third_planes.append(torch.stack(plane_rows))
            point_values.append(scalar)
            point_grads.append(gradient)
            point_hessians.append(torch.stack(hessian_rows))
            point_thirds.append(torch.stack(third_planes))
        values.append(torch.stack(point_values))
        grads.append(torch.stack(point_grads))
        hessians.append(torch.stack(point_hessians))
        thirds.append(torch.stack(point_thirds))
    return {
        "value": torch.stack(values).detach().cpu().numpy(),
        "grad": torch.stack(grads).detach().cpu().numpy(),
        "hess": torch.stack(hessians).detach().cpu().numpy(),
        "third": torch.stack(thirds).detach().cpu().numpy(),
    }


def native_parameter_backward(
    points: np.ndarray,
    target: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    encoding: str,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    value = cp.asarray(numpy_features(points, encoding), dtype=cp.float32)
    weights_gpu = [cp.asarray(item, dtype=cp.float32) for item in weights]
    biases_gpu = [cp.asarray(item, dtype=cp.float32) for item in biases]
    layer_inputs: list[cp.ndarray] = []
    preactivations: list[cp.ndarray] = []
    for layer, (weight, bias) in enumerate(zip(weights_gpu, biases_gpu)):
        layer_inputs.append(value)
        preactivation = value @ weight.T + bias
        preactivations.append(preactivation)
        if layer + 1 < len(weights_gpu):
            value = cp.maximum(preactivation, 0.0) if activation == "relu" else cp.sin(preactivation)
        else:
            value = preactivation
    target_gpu = cp.asarray(target, dtype=cp.float32)
    residual = value - target_gpu
    loss = cp.mean(residual * residual)
    adjoint = 2.0 * residual / residual.size
    grad_weights: list[np.ndarray] = [np.empty(0)] * len(weights_gpu)
    grad_biases: list[np.ndarray] = [np.empty(0)] * len(weights_gpu)
    for layer in reversed(range(len(weights_gpu))):
        grad_weights[layer] = cp.asnumpy(adjoint.T @ layer_inputs[layer])
        grad_biases[layer] = cp.asnumpy(cp.sum(adjoint, axis=0))
        if layer:
            adjoint = adjoint @ weights_gpu[layer]
            previous = preactivations[layer - 1]
            if activation == "relu":
                adjoint = adjoint * (previous > 0.0)
            else:
                adjoint = adjoint * cp.cos(previous)
    return float(loss.get()), grad_weights, grad_biases


def torch_parameter_backward(
    points: np.ndarray,
    target: np.ndarray,
    weights_np: list[np.ndarray],
    biases_np: list[np.ndarray],
    activation: str,
    encoding: str,
) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
    device = torch.device("cuda")
    points_t = torch.tensor(points, dtype=torch.float32, device=device)
    target_t = torch.tensor(target, dtype=torch.float32, device=device)
    weights = [
        torch.tensor(item, dtype=torch.float32, device=device, requires_grad=True)
        for item in weights_np
    ]
    biases = [
        torch.tensor(item, dtype=torch.float32, device=device, requires_grad=True)
        for item in biases_np
    ]
    output = torch_value_forward(points_t, weights, biases, activation, encoding)
    loss = torch.mean((output - target_t) ** 2)
    loss.backward()
    return (
        float(loss.detach().cpu()),
        [item.grad.detach().cpu().numpy() for item in weights],
        [item.grad.detach().cpu().numpy() for item in biases],
    )


def compact_slots(jet: Jet3) -> dict[str, np.ndarray]:
    arrays = {
        "value": cp.asnumpy(jet.value),
        "grad": cp.asnumpy(jet.grad),
        "hess": cp.asnumpy(jet.hess),
        "third": cp.asnumpy(jet.third),
    }
    slots = {"value": arrays["value"]}
    for name, index in zip(SLOT_NAMES[1:4], FIRST):
        slots[name] = arrays["grad"][(slice(None), slice(None), *index)]
    for name, index in zip(SLOT_NAMES[4:10], SECOND):
        slots[name] = arrays["hess"][(slice(None), slice(None), *index)]
    for name, index in zip(SLOT_NAMES[10:], THIRD):
        slots[name] = arrays["third"][(slice(None), slice(None), *index)]
    return slots


def compact_torch(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    slots = {"value": arrays["value"]}
    for name, index in zip(SLOT_NAMES[1:4], FIRST):
        slots[name] = arrays["grad"][(slice(None), slice(None), *index)]
    for name, index in zip(SLOT_NAMES[4:10], SECOND):
        slots[name] = arrays["hess"][(slice(None), slice(None), *index)]
    for name, index in zip(SLOT_NAMES[10:], THIRD):
        slots[name] = arrays["third"][(slice(None), slice(None), *index)]
    return slots


def error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    denominator = max(float(np.linalg.norm(reference)), 1.0e-12)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / denominator),
    }


def derivative_order(slot_index: int) -> int:
    if slot_index == 0:
        return 0
    if slot_index < 4:
        return 1
    if slot_index < 10:
        return 2
    return 3


def fd_weights(derivative: int, radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    if derivative == 0:
        return np.array([0], dtype=int), np.array([1.0])
    nodes = np.arange(-radius, radius + 1, dtype=np.float64)
    matrix = np.vstack([nodes**power for power in range(len(nodes))])
    right = np.zeros(len(nodes), dtype=np.float64)
    right[derivative] = math.factorial(derivative)
    return nodes.astype(int), np.linalg.solve(matrix, right)


def multi_index_counts(index: tuple[int, ...]) -> tuple[int, int, int]:
    return tuple(index.count(axis) for axis in range(DIMENSION))  # type: ignore[return-value]


def finite_difference_slots(
    points: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activation: str,
    encoding: str,
    step: float,
) -> dict[str, np.ndarray]:
    slots: dict[str, np.ndarray] = {}
    for slot_name, index in zip(SLOT_NAMES, MULTI_INDICES):
        counts = multi_index_counts(index)
        node_weights = [fd_weights(order) for order in counts]
        accumulator = np.zeros((len(points), OUTPUTS), dtype=np.float64)
        for selections in itertools.product(
            *[range(len(nodes)) for nodes, _ in node_weights]
        ):
            displacement = np.zeros(DIMENSION, dtype=np.float64)
            coefficient = 1.0
            for axis, selection in enumerate(selections):
                nodes, weights_axis = node_weights[axis]
                displacement[axis] = step * nodes[selection]
                coefficient *= weights_axis[selection]
            shifted = points + displacement[None]
            value, _ = numpy_value_forward(
                shifted, weights, biases, activation, encoding
            )
            accumulator += coefficient * value
        order = len(index)
        slots[slot_name] = accumulator / (step**order)
    return slots


def make_weights() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(SEED)
    dimensions = [FEATURES] + [WIDTH] * DEPTH + [OUTPUTS]
    weights, biases = [], []
    for input_width, output_width in zip(dimensions[:-1], dimensions[1:]):
        scale = 0.22 / math.sqrt(input_width)
        weights.append(
            rng.normal(0.0, scale, (output_width, input_width)).astype(np.float32)
        )
        if output_width == OUTPUTS:
            bias = rng.uniform(-0.35, 0.35, output_width)
        else:
            # A derivative implementation check must not accidentally sample
            # the nonsmooth ReLU kink.  Fixed, well-separated hidden biases
            # keep every selected stencil in one ReLU region while remaining
            # identical across all four controlled variants.
            signs = rng.choice(np.asarray((-1.0, 1.0)), output_width)
            bias = signs * rng.uniform(0.48, 0.72, output_width)
        biases.append(bias.astype(np.float32))
    return weights, biases


def select_points(weights: list[np.ndarray], biases: list[np.ndarray]) -> np.ndarray:
    rng = np.random.default_rng(SEED + 17)
    candidates = rng.uniform(-0.72, 0.72, (8192, DIMENSION))
    accepted = []
    for point in candidates:
        safe = True
        for encoding in ("raw_xyz", "posenc"):
            _, minimum = numpy_value_forward(
                point[None], weights, biases, "relu", encoding
            )
            if minimum < 0.035:
                safe = False
                break
        if safe:
            accepted.append(point)
            if len(accepted) == POINTS:
                return np.asarray(accepted, dtype=np.float64)
    raise RuntimeError("could not find enough ReLU points away from all kinks")


def summarize_order(slot_errors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    summary = {}
    for order in range(4):
        names = [name for index, name in enumerate(SLOT_NAMES) if derivative_order(index) == order]
        summary[str(order)] = {
            "max_abs": max(slot_errors[name]["max_abs"] for name in names),
            "max_relative_l2": max(slot_errors[name]["relative_l2"] for name in names),
        }
    return summary


def render_plots(summary: dict, output_dir: Path) -> None:
    colors = {"relu": "#3b6fb6", "siren": "#c74440"}
    markers = {"raw_xyz": "o", "posenc": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for variant in summary["variants"]:
        label = variant["label"]
        color = colors[variant["activation"]]
        marker = markers[variant["encoding"]]
        torch_errors = [
            variant["native_vs_torch_by_order"][str(order)]["max_relative_l2"]
            for order in range(4)
        ]
        axes[0].plot(range(4), torch_errors, marker=marker, color=color, label=label)
        fd_errors = [
            variant["finite_difference_best_by_order"][str(order)]["max_abs"]
            for order in range(1, 4)
        ]
        axes[1].plot(range(1, 4), fd_errors, marker=marker, color=color, label=label)
        backward = max(item["relative_l2"] for item in variant["parameter_backward"])
        axes[2].bar(label, backward, color=color, alpha=0.85)
    axes[0].set_title("Native CUDA vs PyTorch")
    axes[0].set_xlabel("Jet derivative order")
    axes[0].set_ylabel("max relative L2")
    axes[1].set_title("Best finite-difference agreement")
    axes[1].set_xlabel("Derivative order")
    axes[1].set_ylabel("max absolute error")
    axes[2].set_title("Parameter backward")
    axes[2].set_ylabel("max relative L2")
    axes[2].tick_params(axis="x", rotation=35, labelsize=8)
    for axis in axes:
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "network_jet3_ablation.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "network_jet3_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required for the controlled comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights, biases = make_weights()
    points = select_points(weights, biases)
    rng = np.random.default_rng(SEED + 23)
    target = rng.normal(0.0, 0.4, (len(points), OUTPUTS)).astype(np.float32)

    variants = []
    detailed_rows = []
    output_values: dict[str, np.ndarray] = {}
    for activation in ("relu", "siren"):
        for encoding in ("raw_xyz", "posenc"):
            label = f"{activation}_{encoding}"
            native32, minimum = native_jet_forward(
                points, weights, biases, activation, encoding, cp.float32
            )
            native64, _ = native_jet_forward(
                points, weights, biases, activation, encoding, cp.float64
            )
            native_slots = compact_slots(native32)
            native64_slots = compact_slots(native64)
            torch_slots = compact_torch(
                torch_jet_forward(points, weights, biases, activation, encoding)
            )
            slot_errors = {
                name: error_metrics(native_slots[name], torch_slots[name])
                for name in SLOT_NAMES
            }
            for name in SLOT_NAMES:
                detailed_rows.append(
                    {
                        "variant": label,
                        "comparison": "native_cuda_vs_torch_autograd",
                        "slot": name,
                        "order": derivative_order(SLOT_NAMES.index(name)),
                        **slot_errors[name],
                    }
                )

            native_loss, native_gw, native_gb = native_parameter_backward(
                points, target, weights, biases, activation, encoding
            )
            torch_loss, torch_gw, torch_gb = torch_parameter_backward(
                points, target, weights, biases, activation, encoding
            )
            backward = []
            for layer in range(len(weights)):
                weight_error = error_metrics(native_gw[layer], torch_gw[layer])
                bias_error = error_metrics(native_gb[layer], torch_gb[layer])
                backward.extend(
                    (
                        {"parameter": f"weight_{layer}", **weight_error},
                        {"parameter": f"bias_{layer}", **bias_error},
                    )
                )

            fd_sweep: dict[str, dict[str, dict[str, float]]] = {}
            fd_values: dict[str, dict[str, np.ndarray]] = {}
            for step in FD_STEPS:
                finite_slots = finite_difference_slots(
                    points, weights, biases, activation, encoding, step
                )
                step_key = f"{step:.1e}"
                fd_values[step_key] = finite_slots
                fd_sweep[step_key] = {
                    name: error_metrics(native64_slots[name], finite_slots[name])
                    for name in SLOT_NAMES
                }
            fd_best = {}
            for order in range(1, 4):
                names = [
                    name for index, name in enumerate(SLOT_NAMES)
                    if derivative_order(index) == order
                ]
                choices = []
                for step_key in fd_sweep:
                    combined_reference = np.concatenate(
                        [native64_slots[name].ravel() for name in names]
                    )
                    finite_slots = fd_values[step_key]
                    combined_candidate = np.concatenate(
                        [finite_slots[name].ravel() for name in names]
                    )
                    choices.append(
                        {
                            "step": float(step_key),
                            **error_metrics(combined_reference, combined_candidate),
                        }
                    )
                fd_best[str(order)] = min(choices, key=lambda item: item["relative_l2"])

            native_torch_by_order = summarize_order(slot_errors)
            pass_native_torch = max(
                native_torch_by_order[str(order)]["max_relative_l2"]
                for order in range(4)
            ) <= 5.0e-4
            pass_backward = max(item["relative_l2"] for item in backward) <= 5.0e-4
            pass_fd = all(
                fd_best[str(order)]["relative_l2"] <= relative_threshold
                or fd_best[str(order)]["max_abs"] <= absolute_threshold
                for order, relative_threshold, absolute_threshold in (
                    (1, 2.0e-4, 1.0e-5),
                    (2, 2.0e-3, 1.0e-4),
                    (3, 5.0e-2, 1.0e-2),
                )
            )
            output_values[label] = native_slots["value"]
            variants.append(
                {
                    "label": label,
                    "activation": activation,
                    "encoding": encoding,
                    "minimum_absolute_relu_preactivation": minimum if activation == "relu" else None,
                    "native_vs_torch_slots": slot_errors,
                    "native_vs_torch_by_order": native_torch_by_order,
                    "native_loss": native_loss,
                    "torch_loss": torch_loss,
                    "loss_error": error_metrics(np.array([native_loss]), np.array([torch_loss])),
                    "parameter_backward": backward,
                    "finite_difference_sweep": fd_sweep,
                    "finite_difference_best_by_order": fd_best,
                    "pass": bool(pass_native_torch and pass_backward and pass_fd),
                }
            )

    controlled_effects = []
    for first, second, control in (
        ("relu_raw_xyz", "siren_raw_xyz", "activation_at_raw_xyz"),
        ("relu_posenc", "siren_posenc", "activation_at_posenc"),
        ("relu_raw_xyz", "relu_posenc", "encoding_at_relu"),
        ("siren_raw_xyz", "siren_posenc", "encoding_at_siren"),
    ):
        controlled_effects.append(
            {
                "control": control,
                "first": first,
                "second": second,
                **error_metrics(output_values[first], output_values[second]),
            }
        )

    summary = {
        "status": "pass" if all(item["pass"] for item in variants) else "review",
        "scope": "network_only_no_fluid_system",
        "reference": "native_cuda_dense_symmetric_jet3",
        "seed": SEED,
        "network": {
            "input_feature_width": FEATURES,
            "hidden_width": WIDTH,
            "hidden_layers": DEPTH,
            "outputs": OUTPUTS,
            "shared_weights_across_all_variants": True,
            "shared_topology_across_all_variants": True,
            "raw_xyz_mapping": "xyz in first three channels; remaining channels fixed zero",
            "positional_encoding": "sin/cos for xyz at frequencies 1 and 2",
        },
        "points": points.tolist(),
        "finite_difference_steps": list(FD_STEPS),
        "thresholds": {
            "native_vs_torch_max_relative_l2": 5.0e-4,
            "parameter_backward_max_relative_l2": 5.0e-4,
            "finite_difference_relative_l2_by_order": {"1": 2.0e-4, "2": 2.0e-3, "3": 5.0e-2},
            "finite_difference_max_abs_by_order": {"1": 1.0e-5, "2": 1.0e-4, "3": 1.0e-2},
        },
        "variants": variants,
        "controlled_output_effects": controlled_effects,
    }
    (args.output_dir / "network_jet3_ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "network_jet3_slot_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=detailed_rows[0].keys())
        writer.writeheader()
        writer.writerows(detailed_rows)
    render_plots(summary, args.output_dir)
    print(json.dumps({"status": summary["status"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
