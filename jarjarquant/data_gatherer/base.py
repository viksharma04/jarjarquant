from abc import ABC, abstractmethod

_DATA_SOURCE_REGISTRY: dict[str, type["DataSource"]] = {}


def register_data_source(name: str):
    def decorator(cls: type["DataSource"]):
        _DATA_SOURCE_REGISTRY[name] = cls
        return cls

    return decorator


def get_data_source(name: str) -> type["DataSource"]:
    try:
        return _DATA_SOURCE_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown data source '{name}'. Available: {list(_DATA_SOURCE_REGISTRY)}"
        )


def list_data_sources() -> list[str]:
    """Return all registered names."""
    return list(_DATA_SOURCE_REGISTRY.keys())


def get_all_data_sources() -> dict[str, type["DataSource"]]:
    """Return name→class mapping."""
    return _DATA_SOURCE_REGISTRY.copy()


class DataSource(ABC):
    @abstractmethod
    async def fetch(self, ticker: str, **kwargs): ...
