"""Frozen Jet1 ablation: only forward projection and its adjoint are toggled.

Derived verbatim from the local backend function; source hashes and the two
changed conditions are recorded in outputs/gauge_ablation/protocol.json.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import cupy as cp
import numpy as np
from eigenfluid.gen.vortex_merger2d.hardcoded_backend import (
    HardcodedModel, F32, Jet1, _device_points, airfoil_area,
)

def _streamfunction_dirichlet_rayleigh_peel_step(
    cfg: SimpleNamespace,
    model: HardcodedModel,
    points: np.ndarray,
    beta: float,
    alpha: float,
    training: bool,
    full_diagnostics: bool,
) -> dict[str, Any]:
    """Train Lq with genuine Jet1 fields and a device-resident causal pass.

    ``M_q = integral psi**2`` and ``K_q = integral |grad psi|**2`` require no
    Hessian or third derivative.  Raw Rayleigh quotients, stable geometry sort,
    two-pass MGS, adjoints, gradient checks, clipping, and Adam stay on the
    GPU.  The optimizer is a device-side all-mode transaction, followed by one
    synchronization that returns the complete scalar diagnostic table.
    """
    del full_diagnostics
    geometry_mode = str(getattr(cfg, "geometry_mode", ""))
    envelope_mode = str(getattr(cfg, "envelope_mode", ""))
    periodic = (
        geometry_mode == "periodic_box"
        and envelope_mode == "periodic_identity"
        and str(getattr(cfg, "input_encoding", "")) == "periodic_sincos_xy"
    )
    free_boundary = (
        geometry_mode == "free_box"
        and envelope_mode == "free_identity"
        and str(getattr(cfg, "input_encoding", "")) == "raw_xy"
    )
    airfoil = geometry_mode == "cinf_ellipse" and envelope_mode.startswith(
        "cinf_"
    )
    if not (periodic or free_boundary or airfoil):
        raise ValueError(
            "Potential-Rayleigh Jet1 requires a periodic, free, or smooth "
            "hard-boundary trial space"
        )
    if int(getattr(cfg, "spatial_derivative_order", 0)) < 1:
        raise ValueError(
            "Dirichlet streamfunction Rayleigh requires spatial gradients"
        )
    causal_sorting = str(getattr(cfg, "causal_sorting", "fixed_order"))
    if causal_sorting not in {"fixed_order", "geometry_rayleigh"}:
        raise ValueError(f"unsupported causal_sorting {causal_sorting!r}")

    n = len(points)
    if n == 0:
        raise ValueError("Dirichlet streamfunction Rayleigh received no points")
    modes = int(cfg.modes)
    if modes != len(model.modes):
        raise ValueError("configuration/model mode count mismatch")
    area = np.float32(
        (cfg.xmax - cfg.xmin) * (cfg.ymax - cfg.ymin)
        - airfoil_area(cfg, beta)
    )
    weight = cp.asarray(area / np.float32(n), dtype=F32)
    mass_floor = float(cfg.mass_floor)
    passes = max(1, int(getattr(cfg, "reorthogonalization_passes", 2)))
    # One H2D upload per step.  Both the sorting forward pass and the cached
    # backward recomputation receive this device array; ``xy_jets1`` therefore
    # creates a no-copy CuPy view instead of uploading the same points once per
    # mode block.
    points_device = _device_points(points, F32)
    input_host_to_device_transfers = int(not isinstance(points, cp.ndarray))

    # Phase 1: evaluate each network exactly once with three Jet1 channels
    # (value, x, y), then perform the stable geometry-Rayleigh sort on-device.
    raw_value = cp.empty((n, modes), dtype=F32)
    raw_gradient = cp.empty((2 * n, modes), dtype=F32)
    for mode, field, _cache in model.forward_stream_jet1(
        cfg, points_device, beta, alpha, False
    ):
        network = int(mode.mode_index)
        raw_value[:, network] = field.value[:, 0].astype(F32, copy=False)
        raw_gradient[:, network] = field.grad[:, 0].reshape(-1).astype(
            F32, copy=False
        )
    if (periodic or free_boundary) and bool(getattr(cfg, "gauge_fixing_enabled", True)):
        # Velocity potential has an arbitrary additive streamfunction constant.
        # Project it out before sorting and peeling so the zero-energy gauge
        # cannot consume one learned mode or attract all networks.
        raw_value -= cp.mean(raw_value, axis=0, keepdims=True, dtype=F32)
    raw_mass = weight * cp.sum(raw_value * raw_value, axis=0, dtype=F32)
    raw_stiffness = weight * cp.sum(
        raw_gradient * raw_gradient, axis=0, dtype=F32
    )
    sorting_rayleigh = raw_stiffness / raw_mass
    causal_order_device = (
        cp.argsort(sorting_rayleigh, kind="stable")
        if causal_sorting == "geometry_rayleigh"
        else cp.arange(modes, dtype=cp.int64)
    )
    sorted_value = raw_value[:, causal_order_device]
    sorted_gradient = raw_gradient[:, causal_order_device]

    # Phase 2: complete causal MGS and its physical adjoints on-device.  The
    # previous basis is detached exactly as in the original SSS formulation.
    basis_value = cp.empty((n, modes), dtype=F32)
    basis_gradient = cp.empty((2 * n, modes), dtype=F32)
    mass_values = cp.empty(modes, dtype=F32)
    stiffness_values = cp.empty(modes, dtype=F32)
    rayleigh_values = cp.empty(modes, dtype=F32)
    projection_values = cp.empty(modes, dtype=F32)
    orth_values = cp.empty(modes, dtype=F32)
    adjoint_raw_value = cp.zeros_like(raw_value) if training else None
    adjoint_raw_gradient = cp.zeros_like(raw_gradient) if training else None
    freeze_prefix = int(getattr(cfg, "freeze_prefix_modes", 0))
    for causal_rank in range(modes):
        residual_value = sorted_value[:, causal_rank].copy()
        residual_gradient = sorted_gradient[:, causal_rank].copy()
        coefficients = cp.zeros((causal_rank,), dtype=F32)
        if causal_rank:
            previous_value = basis_value[:, :causal_rank]
            previous_gradient = basis_gradient[:, :causal_rank]
            for _ in range(passes):
                increment = weight * (previous_value.T @ residual_value)
                coefficients += increment
                residual_value -= previous_value @ increment
                residual_gradient -= previous_gradient @ increment
        else:
            previous_value = previous_gradient = None

        mass = weight * cp.dot(residual_value, residual_value)
        stiffness = weight * cp.dot(residual_gradient, residual_gradient)
        rayleigh = stiffness / mass
        mass_values[causal_rank] = mass
        stiffness_values[causal_rank] = stiffness
        rayleigh_values[causal_rank] = rayleigh

        requires_grad = training and causal_rank >= freeze_prefix
        if requires_grad:
            adjoint_value = (
                -2.0 * stiffness * weight / (mass * mass)
            ) * residual_value
            adjoint_gradient = (
                2.0 * weight / mass
            ) * residual_gradient
            if causal_rank:
                for _ in range(passes):
                    coefficient_adjoint = -(
                        previous_value.T @ adjoint_value
                        + previous_gradient.T @ adjoint_gradient
                    )
                    adjoint_value += weight * (
                        previous_value @ coefficient_adjoint
                    )
            if adjoint_raw_value is None or adjoint_raw_gradient is None:
                raise RuntimeError("Jet1 adjoint buffers were not allocated")
            network = causal_order_device[causal_rank]
            adjoint_raw_value[:, network] = adjoint_value
            adjoint_raw_gradient[:, network] = adjoint_gradient

        normalizer = cp.reciprocal(cp.sqrt(mass))
        normalized_value = residual_value * normalizer
        normalized_gradient = residual_gradient * normalizer
        basis_value[:, causal_rank] = normalized_value
        basis_gradient[:, causal_rank] = normalized_gradient
        normalized_mass = weight * cp.dot(normalized_value, normalized_value)
        orth_value = cp.abs(normalized_mass - F32(1.0))
        if causal_rank:
            overlap = weight * (previous_value.T @ normalized_value)
            orth_value = cp.maximum(orth_value, cp.max(cp.abs(overlap)))
        orth_values[causal_rank] = orth_value
        projection_values[causal_rank] = (
            cp.dot(coefficients, coefficients) / mass
            if causal_rank
            else F32(0.0)
        )

    peeled_rayleigh_by_network_device = cp.empty(modes, dtype=F32)
    peeled_rayleigh_by_network_device[causal_order_device] = rayleigh_values
    physical_valid_device = cp.all(
        cp.isfinite(raw_mass)
        & cp.isfinite(raw_stiffness)
        & cp.isfinite(sorting_rayleigh)
        & (raw_mass > F32(mass_floor))
    ) & cp.all(
        cp.isfinite(mass_values)
        & cp.isfinite(stiffness_values)
        & cp.isfinite(rayleigh_values)
        & cp.isfinite(projection_values)
        & cp.isfinite(orth_values)
        & (mass_values > F32(mass_floor))
    )

    # Phase 3: recompute contiguous network blocks with Jet1 caches, accumulate
    # every network's gradients, then apply one all-mode device transaction.
    # No GPU sort result is copied back to drive the forward schedule.
    gradient_norms_device = cp.zeros(modes, dtype=F32)
    optimizer_committed_device = physical_valid_device
    active_modes_device = cp.zeros(modes, dtype=cp.bool_)
    active_modes_device[causal_order_device[freeze_prefix:]] = True
    if training:
        if adjoint_raw_value is None or adjoint_raw_gradient is None:
            raise RuntimeError("Jet1 training adjoints are unavailable")
        if (periodic or free_boundary) and bool(getattr(cfg, "gauge_fixing_enabled", True)):
            # Adjoint of the sample-wise mean-zero gauge projection.
            adjoint_raw_value -= cp.mean(
                adjoint_raw_value, axis=0, keepdims=True, dtype=F32
            )
        backward_batch = max(1, int(getattr(cfg, "mode_backward_batch", 1)))
        all_weight_grads = [
            cp.zeros_like(stack) for stack in model.master_weight_stacks
        ]
        all_bias_grads = [
            cp.zeros_like(stack) for stack in model.master_bias_stacks
        ]
        backward_modes: list[HardcodedMode] = []
        backward_caches: list[Any] = []
        backward_adjoints: list[Jet1] = []

        def flush_backward_block() -> None:
            if backward_modes:
                block_weight_grads, block_bias_grads = (
                    model.backward_gradients_block_jet1(
                    cfg, backward_modes, backward_caches, backward_adjoints
                    )
                )
                start = backward_modes[0].mode_index
                stop = start + len(backward_modes)
                for target, source in zip(
                    all_weight_grads, block_weight_grads
                ):
                    target[start:stop] = source
                for target, source in zip(all_bias_grads, block_bias_grads):
                    target[start:stop] = source
                backward_modes.clear()
                backward_caches.clear()
                backward_adjoints.clear()

        for mode, field, cache in model.forward_stream_jet1(
            cfg, points_device, beta, alpha, True
        ):
            network = int(mode.mode_index)
            adjoint = Jet1(
                cp.zeros_like(field.value, dtype=F32),
                cp.zeros_like(field.grad, dtype=F32),
            )
            adjoint.value[:, 0] = adjoint_raw_value[:, network]
            adjoint.grad[:, 0] = adjoint_raw_gradient[:, network].reshape(n, 2)
            backward_modes.append(mode)
            backward_caches.append(cache)
            backward_adjoints.append(adjoint)
            if len(backward_modes) == backward_batch:
                flush_backward_block()
        flush_backward_block()
        gradient_norms_device, optimizer_committed_device = (
            model.adam_step_jet1_device(
                cfg,
                all_weight_grads,
                all_bias_grads,
                active_modes_device,
                physical_valid_device,
            )
        )

    # The sole training-step host transfer returns physical diagnostics, the
    # stable causal permutation, all gradient norms, and the transaction flag.
    diagnostic_rows = cp.asnumpy(
        cp.stack(
            (
                raw_mass,
                raw_stiffness,
                sorting_rayleigh,
                mass_values,
                stiffness_values,
                rayleigh_values,
                projection_values,
                orth_values,
                peeled_rayleigh_by_network_device,
                causal_order_device.astype(F32),
                gradient_norms_device,
                cp.broadcast_to(
                    optimizer_committed_device.astype(F32), (modes,)
                ),
            )
        )
    )
    (
        raw_mass_host,
        raw_stiffness_host,
        sorting_rayleigh_host,
        mass_host,
        stiffness_host,
        rayleigh_host,
        projection_host,
        orth_host,
        peeled_by_network_host,
        causal_order_float,
        gradient_norms_host,
        optimizer_committed_host,
    ) = diagnostic_rows
    causal_order = causal_order_float.astype(np.int64)
    invalid_raw = ~(
        np.isfinite(raw_mass_host)
        & np.isfinite(raw_stiffness_host)
        & np.isfinite(sorting_rayleigh_host)
    )
    unresolved_raw = raw_mass_host <= mass_floor
    if np.any(invalid_raw | unresolved_raw):
        network = int(np.flatnonzero(invalid_raw | unresolved_raw)[0])
        raise FloatingPointError(
            "causal sorting received an invalid streamfunction candidate "
            f"{network + 1}: M={raw_mass_host[network]:.8e}, "
            f"K={raw_stiffness_host[network]:.8e}"
        )
    invalid_peeled = ~(
        np.isfinite(mass_host)
        & np.isfinite(stiffness_host)
        & np.isfinite(rayleigh_host)
        & np.isfinite(projection_host)
        & np.isfinite(orth_host)
    )
    unresolved_peeled = mass_host <= mass_floor
    if np.any(invalid_peeled | unresolved_peeled):
        causal_rank = int(np.flatnonzero(invalid_peeled | unresolved_peeled)[0])
        network = int(causal_order[causal_rank])
        raise FloatingPointError(
            f"streamfunction candidate {network + 1} "
            f"(causal rank {causal_rank + 1}) is invalid: "
            f"M={mass_host[causal_rank]:.8e}, "
            f"K={stiffness_host[causal_rank]:.8e}"
        )
    if training and not bool(optimizer_committed_host[0]):
        active_host = np.zeros(modes, dtype=bool)
        active_host[causal_order[freeze_prefix:]] = True
        invalid_gradient = active_host & (
            ~np.isfinite(gradient_norms_host) | (gradient_norms_host <= 0.0)
        )
        network = (
            int(np.flatnonzero(invalid_gradient)[0])
            if np.any(invalid_gradient)
            else -1
        )
        raise FloatingPointError(
            "zero or non-finite total hardcoded gradient"
            + (f" for network mode {network + 1}" if network >= 0 else "")
        )
    if training:
        active_networks = set(causal_order[freeze_prefix:].astype(int).tolist())
        for network, mode in enumerate(model.modes):
            if network in active_networks:
                mode.steps = [step + 1 for step in mode.steps]

    rayleighs = rayleigh_host.astype(np.float32, copy=False).tolist()
    sorting_rayleigh_by_network = sorting_rayleigh_host.astype(
        np.float32, copy=False
    ).tolist()
    peeled_rayleigh_by_network = peeled_by_network_host.astype(
        np.float32, copy=False
    ).tolist()
    mean_rayleigh = float(np.mean(rayleigh_host, dtype=np.float32))
    return {
        "loss": mean_rayleigh,
        "base_loss": mean_rayleigh,
        "streamfunction_rayleigh": mean_rayleigh,
        "mean_energy": mean_rayleigh,
        "max_energy": float(np.max(rayleigh_host)),
        "min_energy": float(np.min(rayleigh_host)),
        "mean_projection_ratio": float(np.mean(projection_host, dtype=np.float32)),
        "orth_error": float(np.max(orth_host)),
        "peeled_rank": modes,
        "requested_modes": modes,
        "mode_energies": rayleighs,
        "mode_energies_by_network": peeled_rayleigh_by_network,
        "causal_sorting": causal_sorting,
        "causal_order": causal_order.astype(int).tolist(),
        "causal_order_1based": (causal_order + 1).astype(int).tolist(),
        "sorting_rayleigh_by_network": sorting_rayleigh_by_network,
        "spatial_jet_order_used": 1,
        "causal_sort_device": "gpu",
        "diagnostic_host_transfers_per_step": 1,
        "optimizer_device": "gpu_batched_all_modes",
        "optimizer_host_transfers_per_step": 0,
        "input_host_to_device_transfers_per_step": (
            input_host_to_device_transfers
        ),
        "sample_pack_device": (
            "gpu_resident" if input_host_to_device_transfers == 0
            else "host_per_step"
        ),
        "gradient_norm_min": (
            float(np.min(gradient_norms_host)) if training else float("nan")
        ),
        "gradient_norm_max": (
            float(np.max(gradient_norms_host)) if training else float("nan")
        ),
    }
