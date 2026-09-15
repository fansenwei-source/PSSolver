"""Read-only tensor/storage inventory for bounded runtime diagnostics.

The traversal deliberately starts from explicitly supplied runtime roots.  It
does not inspect Python's global garbage collector and it never retains tensor
objects in the returned report.  Tensor views are recorded separately while
their backing storage is counted once.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import types
from typing import Any

import torch

_LEAF_TYPES = (
    str,
    bytes,
    bytearray,
    int,
    float,
    complex,
    bool,
    type(None),
    torch.dtype,
    torch.device,
)


def _owner_category(path: str) -> str:
    lowered = path.lower()
    if ".fields.spatial" in lowered or ".fields.spectral" in lowered:
        return "solver_fields"
    if any(token in lowered for token in (".qx", ".qy", ".q2", "laplacian")):
        return "spectral_operators"
    if any(
        token in lowered
        for token in ("representation", "generation_state", "transient_cache")
    ):
        return "algebraic_representations"
    if "scheduler" in lowered or "execution_plan" in lowered:
        return "scheduler_and_plans"
    if any(token in lowered for token in ("projector", "transform_backend")):
        return "transform_and_projection"
    if "integrator" in lowered:
        return "integrator"
    if ".model" in lowered:
        return "model"
    return "other_runtime"


def _object_attributes(value: object) -> tuple[tuple[str, Any], ...]:
    result: dict[str, Any] = {}
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            try:
                result[field.name] = object.__getattribute__(value, field.name)
            except AttributeError:
                continue
    try:
        namespace = object.__getattribute__(value, "__dict__")
    except AttributeError:
        namespace = None
    if isinstance(namespace, Mapping):
        result.update(namespace)
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or name in result:
                continue
            try:
                result[name] = object.__getattribute__(value, name)
            except AttributeError:
                continue
    return tuple(sorted(result.items()))


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def build_tensor_inventory(
    roots: Mapping[str, object],
    *,
    device: object | None = None,
    max_depth: int = 12,
    max_objects: int = 100_000,
    maximum_reported_storages: int = 1024,
) -> dict[str, object]:
    """Describe tensors reachable from explicitly named runtime roots.

    Only containers and objects implemented by ``pssolver`` or ``benchmarks``
    are expanded.  This keeps PyTorch internals, compiled functions, modules,
    and unrelated process state outside the audit.
    """

    if not isinstance(roots, Mapping) or not roots:
        raise ValueError("roots must be a nonempty mapping")
    if max_depth < 0 or max_objects <= 0 or maximum_reported_storages <= 0:
        raise ValueError("inventory bounds must be positive")
    selected_device = None if device is None else torch.device(device)
    if (
        selected_device is not None
        and selected_device.type == "cuda"
        and selected_device.index is None
    ):
        selected_device = torch.device("cuda", torch.cuda.current_device())
    visited: set[int] = set()
    tensor_records: list[dict[str, object]] = []
    storage_records: dict[tuple[str, int, int], dict[str, object]] = {}
    object_count = 0
    truncated = False

    def visit(value: object, path: str, depth: int) -> None:
        nonlocal object_count, truncated
        if isinstance(value, torch.Tensor):
            if selected_device is not None and value.device != selected_device:
                return
            logical_bytes = int(value.numel() * value.element_size())
            storage_key = _storage_identity(value)
            record = {
                "path": path,
                "category": _owner_category(path),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "logical_bytes": logical_bytes,
                "storage_offset": int(value.storage_offset()),
                "is_view": value._base is not None,
                "requires_grad": bool(value.requires_grad),
            }
            tensor_records.append(record)
            storage = storage_records.setdefault(
                storage_key,
                {
                    "device": storage_key[0],
                    "data_ptr": storage_key[1],
                    "storage_bytes": storage_key[2],
                    "paths": [],
                    "categories": set(),
                    "dtypes": set(),
                    "tensor_references": 0,
                },
            )
            storage["paths"].append(path)
            storage["categories"].add(record["category"])
            storage["dtypes"].add(record["dtype"])
            storage["tensor_references"] += 1
            return
        if isinstance(value, _LEAF_TYPES) or isinstance(
            value,
            (types.FunctionType, types.MethodType, types.ModuleType, type),
        ):
            return
        if depth > max_depth:
            truncated = True
            return
        identity = id(value)
        if identity in visited:
            return
        if object_count >= max_objects:
            truncated = True
            return
        visited.add(identity)
        object_count += 1
        if isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{path}[{key!r}]", depth + 1)
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]", depth + 1)
            return
        if isinstance(value, (set, frozenset)):
            for index, item in enumerate(sorted(value, key=repr)):
                visit(item, f"{path}[{index}]", depth + 1)
            return
        module = type(value).__module__
        if not module.startswith(("pssolver", "benchmarks")):
            return
        for name, item in _object_attributes(value):
            if name.startswith("__"):
                continue
            visit(item, f"{path}.{name}", depth + 1)

    for name in sorted(roots):
        if not isinstance(name, str) or not name:
            raise ValueError("inventory root names must be nonempty strings")
        visit(roots[name], name, 0)

    tensor_records.sort(key=lambda item: str(item["path"]))
    normalized_storages = []
    category_unique = defaultdict(int)
    category_shared = defaultdict(int)
    dtype_unique = defaultdict(int)
    for storage in storage_records.values():
        paths = sorted(set(storage["paths"]))
        categories = sorted(storage["categories"])
        dtypes = sorted(storage["dtypes"])
        size = int(storage["storage_bytes"])
        if len(categories) == 1:
            category_unique[categories[0]] += size
        else:
            for category in categories:
                category_shared[category] += size
        if len(dtypes) == 1:
            dtype_unique[dtypes[0]] += size
        normalized_storages.append(
            {
                "device": storage["device"],
                "data_ptr": storage["data_ptr"],
                "storage_bytes": size,
                "paths": paths,
                "categories": categories,
                "dtypes": dtypes,
                "tensor_references": int(storage["tensor_references"]),
            }
        )
    normalized_storages.sort(
        key=lambda item: (-int(item["storage_bytes"]), item["paths"][0])
    )
    unique_storage_bytes = sum(
        int(item["storage_bytes"]) for item in normalized_storages
    )
    return {
        "schema_version": 1,
        "scope": "explicit_runtime_object_graph",
        "device_filter": None if selected_device is None else str(selected_device),
        "truncated": truncated,
        "visited_object_count": object_count,
        "tensor_reference_count": len(tensor_records),
        "unique_storage_count": len(normalized_storages),
        "logical_tensor_bytes": sum(
            int(item["logical_bytes"]) for item in tensor_records
        ),
        "unique_storage_bytes": unique_storage_bytes,
        "aliased_tensor_reference_count": sum(
            max(0, int(item["tensor_references"]) - 1) for item in normalized_storages
        ),
        "exclusive_storage_bytes_by_category": dict(sorted(category_unique.items())),
        "shared_storage_bytes_by_category": dict(sorted(category_shared.items())),
        "unique_storage_bytes_by_dtype": dict(sorted(dtype_unique.items())),
        "tensor_references": tensor_records,
        "storages": normalized_storages[:maximum_reported_storages],
        "reported_storage_count": min(
            len(normalized_storages), maximum_reported_storages
        ),
        "all_storages_reported": (
            len(normalized_storages) <= maximum_reported_storages
        ),
    }


__all__ = ["build_tensor_inventory"]
