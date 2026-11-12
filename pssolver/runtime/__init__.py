"""Runtime preparation, observation and common Runner modules."""

from importlib import import_module as _import_module


_IMPLICIT_SUBMODULE_NAMES = frozenset(
    {"channel_observations", "plane_observations", "state_validation"}
)


def __getattr__(name: str) -> object:
    if name not in _IMPLICIT_SUBMODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | _IMPLICIT_SUBMODULE_NAMES)


__all__: list[str] = []
