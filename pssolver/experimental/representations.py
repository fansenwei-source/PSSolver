"""Generation-local physical/spectral state for the experimental runtime.

Stage N.2 adds a lazy physical view for the same synchronized pre-RHS state.
It survives only long enough for the explicit RHS to consume declared
transients and is invalidated before the next algebraic generation.  Stage
N.3 may prefetch an explicitly declared physical computation island in
boundary-compatible transform batches; the lifetime remains unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch


@dataclass(frozen=True, slots=True)
class BatchedPhysicalMaterialization:
    """Physical values and transform-batch sizes from one bounded prefetch."""

    values: Mapping[str, torch.Tensor]
    transform_batch_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("materialized values must be a mapping")
        values = dict(self.values)
        if not values or any(
            not isinstance(name, str)
            or not name.isidentifier()
            or not isinstance(value, torch.Tensor)
            for name, value in values.items()
        ):
            raise ValueError(
                "materialized values must map identifiers to tensors"
            )
        sizes = tuple(self.transform_batch_sizes)
        if any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in sizes
        ):
            raise ValueError("transform batch sizes must be positive integers")
        if sum(sizes) > len(values):
            raise ValueError(
                "transform batches cannot contain more values than returned"
            )
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "transform_batch_sizes", sizes)


class AlgebraicGenerationState:
    """Own one synchronized algebraic generation with lazy physical values.

    Every algebraic solver still receives a read-only mapping of physical
    tensors.  The difference is that ``__getitem__`` materializes a physical
    tensor only when a solver actually asks for it.  Representation-aware
    solvers may instead request the already available native spectrum.

    The state is deliberately short lived.  It is invalidated before the next
    algebraic generation and is never serialized or used as evolved state.
    """

    def __init__(
        self,
        generation: int,
        physical: Mapping[str, torch.Tensor],
        spectral: Mapping[str, torch.Tensor],
        *,
        materialize: Callable[[str, torch.Tensor], torch.Tensor],
        materialize_many: Callable[
            [tuple[str, ...], tuple[torch.Tensor, ...]],
            BatchedPhysicalMaterialization,
        ]
        | None = None,
    ) -> None:
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
        ):
            raise ValueError("generation must be a positive integer")
        if not isinstance(physical, Mapping) or not isinstance(spectral, Mapping):
            raise TypeError("generation representations must be mappings")
        if set(physical) != set(spectral):
            raise ValueError("initial physical and spectral names must match")
        if not callable(materialize):
            raise TypeError("materialize must be callable")
        if materialize_many is not None and not callable(materialize_many):
            raise TypeError("materialize_many must be callable or None")
        self._generation = generation
        self._physical = dict(physical)
        self._spectral = dict(spectral)
        self._published: set[str] = set()
        self._materialize = materialize
        self._materialize_many = materialize_many
        self._active = True
        self._counters = {
            "physical_cache_hits": 0,
            "physical_materializations": 0,
            "on_demand_physical_materializations": 0,
            "physical_materialization_batches": 0,
            "batched_physical_components": 0,
            "singleton_materialization_batches": 0,
            "maximum_materialization_batch_size": 0,
            "spectral_dependency_hits": 0,
            "published_components": 0,
            "physical_island_prefetches": 0,
            "physical_island_requested_components": 0,
        }

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active(self) -> bool:
        return self._active

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("algebraic generation state is no longer current")

    def publish_spectral(self, name: str, value: torch.Tensor) -> None:
        """Publish one algebraic output without forcing physical storage."""

        self._require_active()
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("representation name must be a Python identifier")
        if not isinstance(value, torch.Tensor):
            raise TypeError("spectral representation must be a tensor")
        self._spectral[name] = value
        self._physical.pop(name, None)
        if name not in self._published:
            self._published.add(name)
            self._counters["published_components"] += 1

    def replace_spectral(self, name: str, value: torch.Tensor) -> None:
        """Rebind one spectrum to an equivalent packed output view."""

        self._require_active()
        if name not in self._spectral:
            raise KeyError(f"unknown spectral component {name!r}")
        if not isinstance(value, torch.Tensor):
            raise TypeError("spectral representation must be a tensor")
        self._spectral[name] = value

    def physical(self, name: str) -> torch.Tensor:
        self._require_active()
        try:
            value = self._physical[name]
        except KeyError:
            try:
                spectral = self._spectral[name]
            except KeyError as exc:
                raise KeyError(f"unknown algebraic component {name!r}") from exc
            value = self._materialize(name, spectral)
            if not isinstance(value, torch.Tensor):
                raise TypeError("materialization must return a tensor")
            self._physical[name] = value
            self._counters["physical_materializations"] += 1
            self._counters["on_demand_physical_materializations"] += 1
            self._counters["physical_materialization_batches"] += 1
            self._counters["singleton_materialization_batches"] += 1
            self._counters["maximum_materialization_batch_size"] = max(
                self._counters["maximum_materialization_batch_size"],
                1,
            )
        else:
            self._counters["physical_cache_hits"] += 1
        return value

    def prefetch_physical(self, names: tuple[str, ...]) -> None:
        """Materialize a declared physical island in compatible batches."""

        self._require_active()
        if isinstance(names, str):
            raise TypeError("physical-island names must be an iterable")
        names = tuple(names)
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in names
        ):
            raise ValueError(
                "physical-island names must be unique identifiers"
            )
        unknown = tuple(name for name in names if name not in self._spectral)
        if unknown:
            raise KeyError(f"unknown physical-island components: {unknown!r}")
        self._counters["physical_island_prefetches"] += 1
        self._counters["physical_island_requested_components"] += len(names)
        missing = tuple(name for name in names if name not in self._physical)
        if not missing:
            return
        if self._materialize_many is None:
            for name in missing:
                self.physical(name)
            return
        result = self._materialize_many(
            missing,
            tuple(self._spectral[name] for name in missing),
        )
        if not isinstance(result, BatchedPhysicalMaterialization):
            raise TypeError(
                "materialize_many must return BatchedPhysicalMaterialization"
            )
        values = dict(result.values)
        if set(values) != set(missing):
            raise ValueError(
                "batched materialization returned the wrong components"
            )
        for name in missing:
            value = values[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("batched materialization must return tensors")
            self._physical[name] = value
        batch_sizes = result.transform_batch_sizes
        transformed_components = sum(batch_sizes)
        cache_hits = len(missing) - transformed_components
        self._counters["physical_cache_hits"] += cache_hits
        self._counters["physical_materializations"] += transformed_components
        self._counters["physical_materialization_batches"] += len(batch_sizes)
        self._counters["batched_physical_components"] += sum(
            size for size in batch_sizes if size > 1
        )
        self._counters["singleton_materialization_batches"] += sum(
            size == 1 for size in batch_sizes
        )
        if batch_sizes:
            self._counters["maximum_materialization_batch_size"] = max(
                self._counters["maximum_materialization_batch_size"],
                max(batch_sizes),
            )

    def spectral(self, name: str) -> torch.Tensor:
        self._require_active()
        try:
            value = self._spectral[name]
        except KeyError as exc:
            raise KeyError(f"unknown algebraic component {name!r}") from exc
        self._counters["spectral_dependency_hits"] += 1
        return value

    def view(self, names: tuple[str, ...]) -> "AlgebraicPhysicalStateView":
        self._require_active()
        return AlgebraicPhysicalStateView(self, names)

    def snapshot(self) -> MappingProxyType[str, object]:
        unmaterialized = self._published.difference(self._physical)
        return MappingProxyType(
            {
                "schema_version": 1,
                "generation": self._generation,
                **self._counters,
                "unmaterialized_published_components": len(unmaterialized),
                "retained_physical_components": len(self._physical),
                "retained_spectral_components": len(self._spectral),
                "active": self._active,
            }
        )

    def invalidate(self) -> MappingProxyType[str, object]:
        """Release all tensor references at the next lifecycle boundary."""

        snapshot = dict(self.snapshot())
        self._active = False
        self._physical.clear()
        self._spectral.clear()
        self._published.clear()
        snapshot.update(
            {
                "active": False,
                "retained_physical_components": 0,
                "retained_spectral_components": 0,
            }
        )
        return MappingProxyType(snapshot)


class AlgebraicPhysicalStateView(Mapping[str, torch.Tensor]):
    """Read-only component subset backed by an algebraic generation state."""

    def __init__(
        self,
        generation_state: AlgebraicGenerationState,
        names: tuple[str, ...],
    ) -> None:
        if not isinstance(generation_state, AlgebraicGenerationState):
            raise TypeError("generation_state must be AlgebraicGenerationState")
        if isinstance(names, str):
            raise TypeError("component names must be an iterable, not a string")
        names = tuple(names)
        if len(set(names)) != len(names) or any(
            not isinstance(name, str) or not name.isidentifier()
            for name in names
        ):
            raise ValueError("component names must be unique identifiers")
        self._generation_state = generation_state
        self._names = names
        self._name_set = frozenset(names)

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._name_set:
            raise KeyError(name)
        return self._generation_state.physical(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def spectral(self, name: str) -> torch.Tensor:
        """Return a native spectrum without materializing physical space."""

        if name not in self._name_set:
            raise KeyError(name)
        return self._generation_state.spectral(name)

    def prefetch_physical(self, names: tuple[str, ...] | None = None) -> None:
        """Materialize a declared subset before entering a physical island."""

        selected = self._names if names is None else tuple(names)
        if any(name not in self._name_set for name in selected):
            raise KeyError("physical-island prefetch exceeds the state view")
        self._generation_state.prefetch_physical(selected)


__all__ = [
    "AlgebraicGenerationState",
    "AlgebraicPhysicalStateView",
    "BatchedPhysicalMaterialization",
]
