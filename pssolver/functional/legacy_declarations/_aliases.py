"""Resolve frozen registry compiler identities without changing their bytes."""

_COMPILER_MODULES = frozenset({"public_simulation_runner", "legacy_periodic", "legacy_channel"})


def resolve_legacy_path(path: str) -> str:
    """Resolve a registered compiler reference into the compatibility package."""
    package, layer, module, name = path.split(".")
    if (package, layer) != ("pssolver", "configuration") or module not in _COMPILER_MODULES:
        raise ValueError(f"unknown legacy compiler: {path!r}")
    return f"{__package__}.{module}.{name}"
