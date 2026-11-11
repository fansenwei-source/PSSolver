"""Tensor-free implementation catalogue for executable engine-v2 terms."""

from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True, slots=True)
class Capability:
    """One implementation's operator, basis and adjoint contract."""

    kind: str
    key: tuple[str, str]
    implementation: str
    adjoint_mode: str | None


CAPABILITIES = (
    Capability(
        "constraint", ("stokes", "periodic×no_slip^2"),
        "pssolver.engine.constraints:channel_stokes_term", "custom_transpose",
    ),
    Capability(
        "explicit", ("beris_edwards_q", "periodic^1/neumann^2"),
        "pssolver.engine.constraints:channel_explicit_term", None,
    ),
    Capability(
        "constraint", ("stokes", "periodic^2×free_slip"),
        "pssolver.models.active_nematics.engine_terms:plane_stokes_term", "autograd",
    ),
    Capability(
        "explicit", ("beris_edwards_q", "periodic^2×neumann"),
        "pssolver.models.active_nematics.engine_terms:plane_explicit_term", None,
    ),
    Capability(
        "constraint",
        ("stokes", "periodic^3"),
        "pssolver.models.active_nematics.engine_terms:periodic_stokes_term",
        "autograd",
    ),
    Capability(
        "explicit",
        ("beris_edwards_q", "periodic^3"),
        "pssolver.models.active_nematics.engine_terms:periodic_explicit_term",
        None,
    ),
    Capability(
        "explicit",
        ("pointwise", "periodic^3"),
        "pssolver.engine.terms:pointwise_explicit_term",
        None,
    ),
    Capability(
        "linear_recipe",
        ("beris_edwards", "any"),
        "pssolver.models.active_nematics:beris_edwards_linear_operator",
        None,
    ),
    Capability(
        "linear_recipe",
        ("expression", "periodic^3"),
        "pssolver.engine.terms:expression_symbol",
        None,
    ),
)


def lookup(kind, key):
    """Look up a capability without importing its implementation."""
    return next((item for item in CAPABILITIES if item.kind == kind and item.key == key), None)


def basis_signature(axis_kinds):
    """Run-length encode consecutive basis kinds, preserving axis order."""
    if tuple(axis_kinds) == ("periodic", "periodic", "neumann"):
        return "periodic^2×neumann"
    return "/".join(f"{name}^{sum(1 for _ in values)}" for name, values in groupby(axis_kinds))
