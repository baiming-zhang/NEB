"""Analytic second derivatives of the fixed-trace harmonic extension."""
from __future__ import annotations
import cupy as cp
import numpy as np
from neb2d.basisnet.metaball2d.geometry import get_obstacles
from neb2d.basisnet.metaball2d.hardcoded_backend import HardcodedModel,Jet2,xy_jets2,simple_zero_envelope2,_metaball_obstacle_level_jet2
def component_factors(cfg, points: np.ndarray | cp.ndarray) -> list[Jet2]:
    """Return bounded simple-zero coordinates for outer and inner boundaries."""
    if float(cfg.outer_tau) != 0.0:
        raise ValueError("the fixed comparison uses the analytic power-12 outer boundary")
    x, y = xy_jets2(points, cp.float32)
    outer_level = 1.0 - ((x - 5.0) / 4.0).pow(12) - (y / 4.0).pow(12)
    lift_coordinate_mode = str(getattr(cfg, "lift_coordinate_mode", "raw_implicit"))
    if lift_coordinate_mode == "boundary_gradient_scaled":
        # Convert each dimensionless implicit level to a signed-distance-like
        # coordinate using its representative boundary gradient.  This changes
        # only the fixed lift away from the boundary: every zero set, cardinal
        # trace, and the learned correction envelope remain exactly unchanged.
        outer_level = outer_level / (12.0 / 4.0)
    elif lift_coordinate_mode != "raw_implicit":
        raise ValueError(f"unknown lift_coordinate_mode {lift_coordinate_mode!r}")
    factors = [simple_zero_envelope2(outer_level)]
    for spec in get_obstacles(cfg.subcase):
        level = _metaball_obstacle_level_jet2(x, y, spec)
        if lift_coordinate_mode == "boundary_gradient_scaled":
            if spec.kind == "circle":
                scale = 2.0 / spec.size
            elif spec.kind == "square":
                scale = float(spec.power) / spec.size
            elif spec.kind == "triangle":
                scale = 1.0 / spec.size
            else:
                raise ValueError(f"unsupported obstacle kind {spec.kind!r}")
            level = level / scale
        factors.append(simple_zero_envelope2(level))
    return factors


def cardinal_lifts(cfg, points: np.ndarray | cp.ndarray) -> list[Jet2]:
    """Smooth cardinal traces: lift a is one on obstacle a and zero elsewhere."""
    factors = component_factors(cfg, points)
    squared = [factor.square() for factor in factors]
    products: list[Jet2] = []
    for omitted in range(len(squared)):
        product = None
        for index, factor in enumerate(squared):
            if index == omitted:
                continue
            product = factor if product is None else product * factor
        if product is None:
            raise RuntimeError("a multiply connected domain needs at least two boundaries")
        products.append(product)
    denominator = products[0]
    for product in products[1:]:
        denominator = denominator + product
    if bool(cp.any(denominator.value <= 0.0).item()):
        raise FloatingPointError("cardinal lift denominator is non-positive")
    # products[0] is the outer-boundary cardinal.  Harmonic coordinates use
    # only the inner components, in the geometry registry's stable order.
    return [product / denominator for product in products[1:]]


def learned_harmonic_fields(
    model: HardcodedModel,
    cfg,
    points: np.ndarray | cp.ndarray,
    cache: bool,
):
    lifts = cardinal_lifts(cfg, points)
    streams = list(model.forward_stream_jet2(cfg, points, 0.0, 0.0, cache))
    if len(streams) != len(lifts) or len(lifts) != int(cfg.harmonic_dimension):
        raise RuntimeError("harmonic output count does not match the topology")
    return [
        (mode, lift + correction, item_cache)
        for lift, (mode, correction, item_cache) in zip(lifts, streams)
    ]


def velocity_matrix(fields: list[tuple[object, Jet2, object]]) -> cp.ndarray:
    columns = []
    for _, field, _ in fields:
        gradient = field.grad[:, 0]
        columns.append(cp.stack((gradient[:, 1], -gradient[:, 0]), axis=1).reshape(-1))
    return cp.stack(columns, axis=1)
