"""GC-visible live storage set difference specified by R-0079 section 2.3.

The initial StorageImpl identities remain alive through strong storage references.
Endpoint sizes come from the live storages, never from allocation-time addresses.
This is CPU qualification support, not a process RSS or allocator-peak measure.
"""

from __future__ import annotations

import gc

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor


def _live_cpu_storages():
    gc.collect()
    storages = {}
    owners = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue
        if isinstance(obj, (FakeTensor, FunctionalTensor)) or obj.is_meta:
            continue
        if obj.device.type != "cpu":
            continue
        storage = obj.untyped_storage()
        identity = storage._cdata
        storages[identity] = storage
        owners.setdefault(identity, []).append(
            {
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
                "type_name": f"{type(obj).__module__}.{type(obj).__qualname__}",
            }
        )
    return storages, owners


def _records(storages, owners, initial):
    return [
        {
            "_cdata": identity,
            "data_ptr": storage.data_ptr(),
            "nbytes": storage.nbytes(),
            "initial": identity in initial,
            "owners": owners[identity],
        }
        for identity, storage in sorted(storages.items())
    ]


class ResidentStorageCounter:
    """Hold start storages S; measure current sizes of endpoint storages L minus S."""

    def __init__(self) -> None:
        self._initial, owners = _live_cpu_storages()
        self.initial_records = _records(self._initial, owners, self._initial)

    def snapshot(self) -> dict:
        """Return integer/string metadata only, without retaining endpoint tensors."""
        live, owners = _live_cpu_storages()
        records = _records(live, owners, self._initial)
        return {
            "resident_bytes": sum(row["nbytes"] for row in records if not row["initial"]),
            "live_storages": records,
        }
