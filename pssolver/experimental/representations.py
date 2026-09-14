"""Generation-local physical/spectral reuse for the experimental runtime.

The cache is deliberately narrower than a field store.  It retains aliases
only while one frozen algebraic DAG is being evaluated, requires exact tensor
object and in-place-version identity, and drops every tensor reference when
that generation ends.  It therefore cannot make data current across a
timestep or turn an algebraic value into checkpointed state.
"""

from __future__ import annotations

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


__all__ = ["AlgebraicRepresentationCache"]
