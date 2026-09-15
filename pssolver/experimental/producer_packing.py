"""Producer-owned component storage for bounded architecture experiments.

The producer, not the transform scheduler, creates the packed tensor.  The
mapping exposes component views without copying and never reallocates or
reorders producer output.  Production runtimes do not import this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType

import torch


class ProducerPackedComponentValues(Mapping[str, torch.Tensor]):
    """Read-only named views over one producer-owned packed tensor.

    ``packed`` has shape ``(component, batch, *grid)``.  Every mapping value
    therefore has shape ``(batch, *grid)`` and adjacent component values are
    adjacent views of the same storage.  Construction never copies tensor
    data; callers that need a different order must make that choice inside the
    producing operator.
    """

    __slots__ = ("_component_names", "_indices", "_packed")

    def __init__(
        self,
        component_names: Sequence[str],
        packed: torch.Tensor,
    ) -> None:
        if isinstance(component_names, str):
            raise TypeError("component_names must be an iterable of names")
        names = tuple(component_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in names
            )
        ):
            raise ValueError("component_names must be unique identifiers")
        if not isinstance(packed, torch.Tensor):
            raise TypeError("packed must be a torch.Tensor")
        if packed.layout is not torch.strided:
            raise ValueError("packed must use strided layout")
        if packed.ndim < 2 or packed.shape[0] != len(names):
            raise ValueError(
                "packed must have shape (component, batch, *grid)"
            )
        if packed.shape[1] <= 0 or packed.numel() == 0:
            raise ValueError("packed batch and grid dimensions must be nonempty")
        if not packed.is_contiguous():
            raise ValueError("packed must be contiguous")
        if packed.is_conj() or packed.is_neg():
            raise ValueError("packed must not be a conjugate or negative view")
        self._component_names = names
        self._indices = MappingProxyType(
            {name: index for index, name in enumerate(names)}
        )
        self._packed = packed

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._component_names

    @property
    def packed(self) -> torch.Tensor:
        """Return the exact producer-owned tensor without copying it."""

        return self._packed

    @property
    def batch_size(self) -> int:
        return int(self._packed.shape[1])

    @property
    def component_shape(self) -> tuple[int, ...]:
        return tuple(self._packed.shape[1:])

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            index = self._indices[name]
        except KeyError as exc:
            raise KeyError(name) from exc
        return self._packed[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._component_names)

    def __len__(self) -> int:
        return len(self._component_names)

    def contiguous_transform_view(
        self,
        component_names: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return one flattened leading-batch view of a contiguous name run."""

        if component_names is None:
            names = self._component_names
        else:
            if isinstance(component_names, str):
                raise TypeError("component_names must be an iterable of names")
            names = tuple(component_names)
        if not names:
            raise ValueError("component_names must not be empty")
        try:
            indices = tuple(self._indices[name] for name in names)
        except KeyError as exc:
            raise KeyError(exc.args[0]) from exc
        start = indices[0]
        if indices != tuple(range(start, start + len(indices))):
            raise ValueError(
                "requested components are not adjacent in producer storage"
            )
        selected = self._packed.narrow(0, start, len(indices))
        return selected.flatten(0, 1)

    def to_metadata(self) -> dict[str, object]:
        """Return tensor-free storage provenance for diagnostics."""

        return {
            "schema_version": 1,
            "ownership": "producer",
            "component_names": list(self._component_names),
            "component_count": len(self),
            "batch_size": self.batch_size,
            "component_shape": list(self.component_shape),
            "dtype": str(self._packed.dtype),
            "device": str(self._packed.device),
            "layout": "component_batch_grid_contiguous",
            "copies_on_construction": 0,
            "cross_generation_reuse": False,
            "owns_packed_tensor": True,
            "retained_packed_tensor_references": 1,
            "retained_component_view_references": 0,
        }


__all__ = ["ProducerPackedComponentValues"]
