"""Framework-free three-parameter cylindrical fan EigenFluid case."""

from .hardcoded_backend import (
    HardcodedFanModel,
    boundary_envelope,
    cfg_namespace,
    fluid_points,
    initialize_arrays,
    rotor_distance,
    train_step,
)

__all__ = [
    "HardcodedFanModel",
    "boundary_envelope",
    "cfg_namespace",
    "fluid_points",
    "initialize_arrays",
    "rotor_distance",
    "train_step",
]
