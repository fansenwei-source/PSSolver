"""Private helpers for source-compatible package facade deprecations."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Mapping, MutableMapping
import warnings


FacadeExport = tuple[str, str]


def resolve_deprecated_facade_name(
    name: str,
    *,
    facade: str,
    replacement: str,
    exports: Mapping[str, FacadeExport],
    namespace: MutableMapping[str, object],
) -> object:
    """Resolve one compatibility export and warn only on name access.

    The resolved value is cached in ``namespace``.  A caller therefore gets
    one warning for the first access in a process, while imports of child
    modules remain silent.
    """

    try:
        module_name, attribute_name = exports[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {facade!r} has no attribute {name!r}"
        ) from exc
    warnings.warn(
        f"{facade}.{name} is a compatibility facade export and is deprecated; "
        f"import it from {replacement} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    module: ModuleType = import_module(module_name)
    value = getattr(module, attribute_name)
    namespace[name] = value
    return value


def facade_dir(
    namespace: Mapping[str, object],
    exports: Mapping[str, FacadeExport],
) -> list[str]:
    """Return an introspection view that includes unresolved facade names."""

    return sorted(set(namespace) | set(exports))


__all__ = ["FacadeExport", "facade_dir", "resolve_deprecated_facade_name"]
