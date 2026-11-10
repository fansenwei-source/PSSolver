"""Resolve an explicitly allowed implementation only during construction."""

from importlib import import_module


_REFERENCES = frozenset(
    {
        "pssolver.models.active_nematics:beris_edwards_linear_operator",
        "pssolver.models.active_nematics.engine_terms:periodic_explicit_term",
        "pssolver.models.active_nematics.engine_terms:periodic_stokes_term",
    }
)


def resolve(reference):
    if reference not in _REFERENCES:
        raise ValueError(f"engine implementation reference is not allowed: {reference!r}")
    module, name = reference.split(":")
    return getattr(import_module(module), name)
