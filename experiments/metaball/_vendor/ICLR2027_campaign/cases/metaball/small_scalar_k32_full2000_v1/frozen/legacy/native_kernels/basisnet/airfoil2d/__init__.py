"""Airfoil geometry and native CUDA networks."""
__all__ = ["Config", "HardcodedModel"]

def __getattr__(name):
    if name == "Config":
        from .training import Config
        return Config
    if name == "HardcodedModel":
        from .hardcoded_backend import HardcodedModel
        return HardcodedModel
    raise AttributeError(name)
