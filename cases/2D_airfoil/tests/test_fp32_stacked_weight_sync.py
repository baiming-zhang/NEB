from types import SimpleNamespace

import cupy as cp
import numpy as np

from eigenfluid_sss.basisnet.airfoil2d.hardcoded_backend import HardcodedModel


def _arrays(modes: int = 2) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for mode in range(modes):
        prefix = f"m{mode:03d}"
        parameters = {
            f"{prefix}.l0.weight": np.full((2, 6), 0.1, np.float32),
            f"{prefix}.l0.bias": np.zeros(2, np.float32),
            f"{prefix}.output.weight": np.full((1, 2), 0.1, np.float32),
            f"{prefix}.output.bias": np.zeros(1, np.float32),
        }
        arrays.update(parameters)
        for index, value in enumerate(parameters.values()):
            stem = f"{prefix}.adam.p{index:02d}"
            arrays[f"{stem}.exp_avg"] = np.zeros_like(value)
            arrays[f"{stem}.exp_avg_sq"] = np.zeros_like(value)
            arrays[f"{stem}.step"] = np.asarray(0, dtype=np.int64)
    return arrays


def test_fp32_adam_update_synchronizes_batched_forward_stack():
    cfg = SimpleNamespace(
        modes=2,
        hidden_layers=1,
        mlp_compute_dtype="float32",
        fp32_boundary_layers=False,
        learning_rate=2.0e-4,
        grad_clip=5.0,
    )
    model = HardcodedModel(cfg, _arrays())
    mode = model.modes[0]
    before = model.shadow_weight_stacks[0][0].copy()
    gradients: list[cp.ndarray] = []
    for weight, bias in zip(mode.weights, mode.biases):
        gradients.extend((cp.ones_like(weight), cp.ones_like(bias)))

    mode.adam_step(gradients, cfg)

    assert not bool(cp.array_equal(before, mode.weights[0]).get())
    assert bool(
        cp.array_equal(model.shadow_weight_stacks[0][0], mode.weights[0]).get()
    )
    assert bool(
        cp.array_equal(model.shadow_bias_stacks[0][0], mode.biases[0]).get()
    )
