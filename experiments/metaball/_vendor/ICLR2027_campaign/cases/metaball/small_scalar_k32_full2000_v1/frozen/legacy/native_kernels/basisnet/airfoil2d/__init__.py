

__all__ = ["Config", "OfficialPeelingFields"]


def __getattr__(name: str):

    if name in __all__:
        from .training import Config, OfficialPeelingFields

        return {"Config": Config, "OfficialPeelingFields": OfficialPeelingFields}[name]
    raise AttributeError(name)
