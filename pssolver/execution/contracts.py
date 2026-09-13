"""Minimal tensor execution contracts for physical models.

The contracts deliberately expose mathematical data, not transform choices or
legacy runtime containers.  A backend adapter owns the conversion between a
physical-space right-hand side and the runtime's native spectral storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch

from pssolver.core.model import ModelProtocol


@runtime_checkable
class MathematicalOperatorContext(Protocol):
    """Physical-space differential operators with component-space checks.

    Models and model-specific algebraic evaluators name mathematical source
    and destination components.  Transform families and FFT/DCT/DST details
    remain private to the numerical adapter.
    """

    def gradient(
        self,
        source_component: str,
        output_component: str,
        value: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """Return one physical derivative in the declared output space."""

        ...

    def laplacian(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Return the physical Laplacian in the component's own space."""

        ...

    def divergence(
        self,
        source_components: tuple[str, ...],
        output_component: str,
        values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return a physical divergence in the declared output space."""

        ...


@runtime_checkable
class ModelExecutionContext(MathematicalOperatorContext, Protocol):
    """Read-only mathematical data available to an executable model."""

    @property
    def physical_shape(self) -> tuple[int, ...]:
        """Physical collocation-grid shape."""

        ...

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        """Native spectral-storage shape."""

        ...

    @property
    def lengths(self) -> tuple[float, ...]:
        """Physical domain lengths."""

        ...

    @property
    def axis_coordinates(self) -> tuple[torch.Tensor, ...]:
        """Cell-centred one-dimensional coordinates, one tensor per axis."""

        ...

    @property
    def real_dtype(self) -> torch.dtype:
        """Real tensor dtype selected by the numerical plan."""

        ...

    @property
    def device(self) -> torch.device:
        """Tensor execution device."""

        ...

    @property
    def batch_size(self) -> int:
        """Number of independent physical states evolved together."""

        ...

    def laplacian_eigenvalues(self, component_name: str) -> torch.Tensor:
        """Return Laplacian eigenvalues for one declared component."""

        ...


@runtime_checkable
class ExecutableModelProtocol(ModelProtocol, Protocol):
    """Smallest contract needed by the opt-in model execution path.

    Initial values and explicit right-hand sides are physical-space tensors.
    Linear operators use the backend-native spectral layout.  Mapping keys are
    scalar component names from :meth:`ModelProtocol.field_specs`.
    """

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        """Construct initial physical-space values for evolved components."""

        ...

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        """Construct diagonal spectral linear operators."""

        ...

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        """Evaluate the explicit physical-space right-hand side.

        Implementations must treat ``state`` and ``context`` as read-only.
        """

        ...
