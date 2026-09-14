"""Generation-local physical/spectral state for the experimental runtime.

The Stage N.1 cache is deliberately narrower than a field store.  It retains
aliases only while one frozen algebraic DAG is being evaluated, requires exact
tensor object and in-place-version identity, and drops every tensor reference
when that generation ends.  It therefore cannot make data current across a
timestep or turn an algebraic value into checkpointed state.

Stage N.2 adds a lazy physical view for the same synchronized pre-RHS state.
It survives only long enough for the explicit RHS to consume declared
transients and is invalidated before the next algebraic generation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch


@dataclass(frozen=True, slots=True)
class _RepresentationPair:
    physical: torch.Tensor
    physical_version: int
    spectral: torch.Tensor
    spectral_version: int


def _tensor_version(value: torch.Tensor) -> int:
    """Return PyTorch's in-place mutation counter as a plain integer."""

    return int(value._version)


class AlgebraicRepresentationCache:
    """Reuse exact representation pairs within one algebraic generation."""

    def __init__(self) -> None:
        self._active_generation: int | None = None
        self._pairs: dict[str, _RepresentationPair] = {}
        self._counters: dict[str, int] = {}
        self._last_snapshot: MappingProxyType[str, object] | None = None

    @property
    def active(self) -> bool:
        return self._active_generation is not None

    def begin(
        self,
        generation: int,
        physical: dict[str, torch.Tensor],
        spectral: dict[str, torch.Tensor],
    ) -> None:
        """Start a fresh generation seeded by synchronized evolved fields."""

        if self.active:
            raise RuntimeError("an algebraic representation generation is active")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation <= 0
        ):
            raise ValueError("generation must be a positive integer")
        if set(physical) != set(spectral):
            raise ValueError("physical and spectral cache seeds must match")
        self._active_generation = generation
        self._last_snapshot = None
        self._pairs = {}
        self._counters = {
            "seeded_pairs": 0,
            "registered_pairs": 0,
            "forward_hits": 0,
            "forward_misses": 0,
            "inverse_hits": 0,
            "inverse_misses": 0,
        }
        try:
            for name in physical:
                self.register(name, physical[name], spectral[name], seeded=True)
        except BaseException:
            self.abort()
            raise

    @staticmethod
    def _validate_pair(
        name: str,
        physical: torch.Tensor,
        spectral: torch.Tensor,
    ) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("representation name must be a Python identifier")
        if not isinstance(physical, torch.Tensor):
            raise TypeError("physical representation must be a tensor")
        if not isinstance(spectral, torch.Tensor):
            raise TypeError("spectral representation must be a tensor")
        if physical.device != spectral.device:
            raise ValueError("representation devices must match")

    def register(
        self,
        name: str,
        physical: torch.Tensor,
        spectral: torch.Tensor,
        *,
        seeded: bool = False,
    ) -> None:
        """Register one exact pair, replacing any older value of that name."""

        if not self.active:
            raise RuntimeError("no algebraic representation generation is active")
        if not isinstance(seeded, bool):
            raise TypeError("seeded must be a bool")
        self._validate_pair(name, physical, spectral)
        self._pairs[name] = _RepresentationPair(
            physical=physical,
            physical_version=_tensor_version(physical),
            spectral=spectral,
            spectral_version=_tensor_version(spectral),
        )
        key = "seeded_pairs" if seeded else "registered_pairs"
        self._counters[key] += 1

    @staticmethod
    def _unchanged(value: torch.Tensor, version: int) -> bool:
        return _tensor_version(value) == version

    def spectral_for(
        self,
        name: str,
        physical: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return a spectrum only for the exact unchanged physical alias."""

        if not self.active:
            return None
        pair = self._pairs.get(name)
        hit = bool(
            pair is not None
            and pair.physical is physical
            and self._unchanged(physical, pair.physical_version)
            and self._unchanged(pair.spectral, pair.spectral_version)
        )
        self._counters["forward_hits" if hit else "forward_misses"] += 1
        return pair.spectral if hit else None

    def physical_for(
        self,
        name: str,
        spectral: torch.Tensor,
    ) -> torch.Tensor | None:
        """Return a physical value only for the exact unchanged spectrum."""

        if not self.active:
            return None
        pair = self._pairs.get(name)
        hit = bool(
            pair is not None
            and pair.spectral is spectral
            and self._unchanged(spectral, pair.spectral_version)
            and self._unchanged(pair.physical, pair.physical_version)
        )
        self._counters["inverse_hits" if hit else "inverse_misses"] += 1
        return pair.physical if hit else None

    def end(self) -> MappingProxyType[str, object]:
        """Close the active generation and release every retained tensor."""

        if not self.active:
            raise RuntimeError("no algebraic representation generation is active")
        snapshot: dict[str, object] = {
            "schema_version": 1,
            "generation": self._active_generation,
            **self._counters,
            "retained_pairs_after_generation": 0,
        }
        self._active_generation = None
        self._pairs.clear()
        self._counters = {}
        self._last_snapshot = MappingProxyType(snapshot)
        return self._last_snapshot

    def abort(self) -> None:
        """Release all references after an exceptional algebraic evaluation."""

        self._active_generation = None
        self._pairs.clear()
        self._counters = {}
        self._last_snapshot = None

    def last_snapshot(self) -> MappingProxyType[str, object] | None:
        return self._last_snapshot


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
        self._generation = generation
        self._physical = dict(physical)
        self._spectral = dict(spectral)
        self._published: set[str] = set()
        self._materialize = materialize
        self._active = True
        self._counters = {
            "physical_cache_hits": 0,
            "physical_materializations": 0,
            "spectral_dependency_hits": 0,
            "published_components": 0,
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
        else:
            self._counters["physical_cache_hits"] += 1
        return value

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


__all__ = [
    "AlgebraicGenerationState",
    "AlgebraicPhysicalStateView",
    "AlgebraicRepresentationCache",
]
