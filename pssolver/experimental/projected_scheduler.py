"""Boundary-safe scheduling for projected transform batches.

This module is an adapter service: it knows how to batch compatible projected
transforms, but it does not own algebraic field lifetime or model equations.
One scheduler instance is bound to one geometry/runtime/dtype/device context;
one method call is bounded to one algebraic generation or explicit-RHS island.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol

import torch

from .representations import (
    AlgebraicRepresentationCache,
    BatchedPhysicalMaterialization,
)


class ProjectedTransformDirection(str, Enum):
    FORWARD = "forward"
    INVERSE = "inverse"


class ProjectedTransformContext(Protocol):
    """Adapter operations required by the batching scheduler."""

    batch_size: int
    physical_shape: tuple[int, ...]
    spectral_shape: tuple[int, ...]
    real_dtype: torch.dtype
    device: torch.device

    @property
    def spectral_dtype(self) -> torch.dtype: ...

    def boundary_conditions(self, component_name: str) -> tuple[str, ...]: ...

    def forward_projected(
        self, component_name: str, value: torch.Tensor
    ) -> torch.Tensor: ...

    def inverse_projected(
        self, component_name: str, value: torch.Tensor
    ) -> torch.Tensor: ...

    def transform_projected_packed(
        self,
        boundary_conditions: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ProjectedTransformBatchKey:
    """Complete compatibility identity for one scheduled transform batch."""

    boundary_signature: tuple[str, ...]
    direction: ProjectedTransformDirection
    physical_shape: tuple[int, ...]
    spectral_shape: tuple[int, ...]
    batch_size: int
    real_dtype: str
    spectral_dtype: str
    device: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "boundary_signature": list(self.boundary_signature),
            "direction": self.direction.value,
            "physical_shape": list(self.physical_shape),
            "spectral_shape": list(self.spectral_shape),
            "batch_size": self.batch_size,
            "real_dtype": self.real_dtype,
            "spectral_dtype": self.spectral_dtype,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class ScheduledProjectedTransforms:
    """Ordered transform values plus an auditable batch partition."""

    values: tuple[torch.Tensor, ...]
    batch_sizes: tuple[int, ...]
    batch_keys: tuple[ProjectedTransformBatchKey, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        sizes = tuple(self.batch_sizes)
        keys = tuple(self.batch_keys)
        if not values or not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError("scheduled transforms must contain tensors")
        if len(sizes) != len(keys) or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in sizes
        ):
            raise ValueError("scheduled batch sizes and keys must align")
        if sum(sizes) != len(values):
            raise ValueError("scheduled batch sizes must cover every value")
        if not all(isinstance(key, ProjectedTransformBatchKey) for key in keys):
            raise TypeError("scheduled batch keys have an invalid type")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "batch_sizes", sizes)
        object.__setattr__(self, "batch_keys", keys)


class BoundarySignatureTransformScheduler:
    """Group only transforms with an identical resolved execution key."""

    def __init__(self, context: ProjectedTransformContext) -> None:
        self._context = context

    def _key(
        self,
        component_name: str,
        direction: ProjectedTransformDirection,
    ) -> ProjectedTransformBatchKey:
        context = self._context
        return ProjectedTransformBatchKey(
            boundary_signature=tuple(
                context.boundary_conditions(component_name)
            ),
            direction=direction,
            physical_shape=tuple(context.physical_shape),
            spectral_shape=tuple(context.spectral_shape),
            batch_size=context.batch_size,
            real_dtype=str(context.real_dtype),
            spectral_dtype=str(context.spectral_dtype),
            device=str(context.device),
        )

    def transform_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        direction: ProjectedTransformDirection,
    ) -> ScheduledProjectedTransforms:
        if isinstance(component_names, str):
            raise TypeError("component_names must be an iterable")
        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError("scheduled projected inputs must be nonempty and align")
        if len(set(component_names)) != len(component_names):
            raise ValueError("scheduled projected component names must be unique")
        if not isinstance(direction, ProjectedTransformDirection):
            raise TypeError("direction must be a ProjectedTransformDirection")

        groups: dict[ProjectedTransformBatchKey, list[int]] = {}
        for index, name in enumerate(component_names):
            groups.setdefault(self._key(name, direction), []).append(index)

        result: list[torch.Tensor | None] = [None] * len(values)
        sizes: list[int] = []
        keys: list[ProjectedTransformBatchKey] = []
        for key, indices in groups.items():
            sizes.append(len(indices))
            keys.append(key)
            if len(indices) == 1:
                index = indices[0]
                result[index] = (
                    self._context.forward_projected(
                        component_names[index], values[index]
                    )
                    if direction is ProjectedTransformDirection.FORWARD
                    else self._context.inverse_projected(
                        component_names[index], values[index]
                    )
                )
                continue
            packed = torch.cat(
                tuple(values[index] for index in indices), dim=0
            )
            transformed = self._context.transform_projected_packed(
                key.boundary_signature,
                packed,
                direction=direction,
            )
            pieces = transformed.split(self._context.batch_size, dim=0)
            if len(pieces) != len(indices):
                raise RuntimeError("scheduled projected transform split is invalid")
            for index, piece in zip(indices, pieces, strict=True):
                result[index] = piece
        if any(value is None for value in result):
            raise RuntimeError("scheduled projected transform is incomplete")
        return ScheduledProjectedTransforms(
            tuple(value for value in result if value is not None),
            tuple(sizes),
            tuple(keys),
        )

    def forward_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.FORWARD,
        )

    def inverse_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.INVERSE,
        )


class AlgebraicPhysicalIslandScheduler:
    """Schedule physical islands without owning their generation lifetime."""

    def __init__(
        self,
        transforms: BoundarySignatureTransformScheduler,
        representation_cache: AlgebraicRepresentationCache,
    ) -> None:
        if not isinstance(transforms, BoundarySignatureTransformScheduler):
            raise TypeError("transforms must be a transform scheduler")
        if not isinstance(representation_cache, AlgebraicRepresentationCache):
            raise TypeError("representation_cache must be an algebraic cache")
        self._transforms = transforms
        self._cache = representation_cache

    def materialize_dependencies(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> BatchedPhysicalMaterialization:
        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError("materialization inputs must be nonempty and align")
        if len(set(component_names)) != len(component_names):
            raise ValueError("materialization component names must be unique")
        physical: dict[str, torch.Tensor] = {}
        misses: list[tuple[str, torch.Tensor]] = []
        for name, value in zip(component_names, values, strict=True):
            cached = self._cache.physical_for(name, value)
            if cached is None:
                misses.append((name, value))
            else:
                physical[name] = cached
        batch_sizes: tuple[int, ...] = ()
        if misses:
            miss_names = tuple(name for name, _ in misses)
            miss_values = tuple(value for _, value in misses)
            scheduled = self._transforms.inverse_many(miss_names, miss_values)
            batch_sizes = scheduled.batch_sizes
            for name, spectral, value in zip(
                miss_names,
                miss_values,
                scheduled.values,
                strict=True,
            ):
                physical[name] = value
                if self._cache.active:
                    self._cache.register(name, value, spectral)
        return BatchedPhysicalMaterialization(
            MappingProxyType({name: physical[name] for name in component_names}),
            batch_sizes,
        )

    def project_outputs(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> Mapping[str, torch.Tensor]:
        scheduled = self._transforms.forward_many(component_names, values)
        return MappingProxyType(
            dict(zip(component_names, scheduled.values, strict=True))
        )


__all__ = [
    "AlgebraicPhysicalIslandScheduler",
    "BoundarySignatureTransformScheduler",
    "ProjectedTransformBatchKey",
    "ProjectedTransformDirection",
    "ScheduledProjectedTransforms",
]
