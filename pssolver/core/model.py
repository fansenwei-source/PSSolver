"""Declarative physical-model protocol.

This core protocol remains tensor- and runtime-free.  The opt-in execution
contract introduced after spectral planning lives in :mod:`pssolver.execution`
and extends these declarations without coupling core specifications to the
current solver implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .fields import FieldSpec


@runtime_checkable
class ModelProtocol(Protocol):
    """Minimum stable contract for a physical model specification."""

    @property
    def name(self) -> str:
        """Stable model identifier used in metadata and registries."""

        ...

    def field_specs(self) -> tuple[FieldSpec, ...]:
        """Declare evolved, algebraic, and diagnostic fields."""

        ...

    def parameter_metadata(self) -> Mapping[str, object]:
        """Return JSON-compatible physical-parameter metadata."""

        ...
