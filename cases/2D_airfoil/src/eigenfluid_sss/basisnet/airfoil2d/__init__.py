"""Jointly parameterized two-dimensional airfoil BasisNet case."""

__all__ = ["Config", "OfficialPeelingFields"]


def __getattr__(name: str):
    """Keep the no-PyTorch hardcoded backend free of eager torch imports."""
    if name in __all__:
        from .training import Config, OfficialPeelingFields

        return {"Config": Config, "OfficialPeelingFields": OfficialPeelingFields}[name]
    raise AttributeError(name)
