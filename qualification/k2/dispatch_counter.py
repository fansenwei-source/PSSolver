"""R-0078 dispatch counts, shared by baseline and candidate.

These counts describe dispatcher-visible tensors, not allocator scratch space.
Inductor's internal buffers are not visible here. The same frozen pointwise
kernels are used on both sides; allocator profiling is supplementary only.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


def _storage(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


class DispatchCounter(TorchDispatchMode):
    """Count aten transforms, newly stored outputs, and scalar extractions.

    No tensors or storages are retained by this counter. Resident storage is
    measured separately by identity, as specified by R-0079 section 2.3.
    """

    def __init__(self) -> None:
        super().__init__()
        self.transforms = 0
        self.allocations = 0
        self.scalar_extractions = 0
        self.operators: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        inputs = [leaf for leaf in tree_leaves((args, kwargs)) if isinstance(leaf, torch.Tensor)]
        if any(tensor.device.type != "cpu" for tensor in inputs):
            raise ValueError("S-25 dispatch counts require CPU tensors")
        pointers = {_storage(tensor)[0] for tensor in inputs}
        result = func(*args, **kwargs)
        self.operators[str(func)] += 1
        packet = func.overloadpacket
        self.transforms += int(
            packet
            in (
                torch.ops.aten._fft_r2c,
                torch.ops.aten._fft_c2r,
                torch.ops.aten._fft_c2c,
            )
        )
        self.scalar_extractions += int(packet is torch.ops.aten._local_scalar_dense)
        for tensor in tree_leaves(result):
            if isinstance(tensor, torch.Tensor):
                if tensor.device.type != "cpu":
                    raise ValueError("S-25 dispatch counts require CPU tensors")
                pointer, size = _storage(tensor)
                if size > 0 and pointer not in pointers:
                    self.allocations += 1
        return result

    def counts(self) -> dict[str, int]:
        """Return counts without addresses, suitable for exact comparisons."""
        return {
            "transforms": self.transforms,
            "allocations": self.allocations,
            "scalar_extractions": self.scalar_extractions,
        }
