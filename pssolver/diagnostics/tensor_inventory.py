"""Read-only tensor/storage inventory for bounded runtime diagnostics.

The traversal deliberately starts from explicitly supplied runtime roots.  It
does not inspect Python's global garbage collector and it never retains tensor
objects in the returned report.  Tensor views are recorded separately while
their backing storage is counted once.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
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


def _is_passive_mapping(value: object) -> bool:
    """Return whether key lookup is guaranteed not to run application code."""

    return type(value) in (dict, types.MappingProxyType)


def _coalesce_storage_ranges(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Merge overlapping storage address ranges on the same device."""

    merged: list[dict[str, object]] = []
    for source in sorted(
        records,
        key=lambda item: (
            str(item["device"]),
            int(item["data_ptr"]),
            int(item["storage_bytes"]),
        ),
    ):
        record = {
            **source,
            "paths": set(source["paths"]),
            "categories": set(source["categories"]),
            "dtypes": set(source["dtypes"]),
            "source_storage_ranges": {
                (int(source["data_ptr"]), int(storage_bytes))
                for storage_bytes in source.get(
                    "reported_storage_sizes",
                    {source["storage_bytes"]},
                )
            },
        }
        start = int(record["data_ptr"])
        size = int(record["storage_bytes"])
        if merged:
            previous = merged[-1]
            previous_start = int(previous["data_ptr"])
            previous_end = previous_start + int(previous["storage_bytes"])
            if (
                size > 0
                and previous_end > previous_start
                and previous["device"] == record["device"]
                and start < previous_end
            ):
                previous["storage_bytes"] = max(previous_end, start + size) - (
                    previous_start
                )
                previous["paths"].update(record["paths"])
                previous["categories"].update(record["categories"])
                previous["dtypes"].update(record["dtypes"])
                previous["tensor_references"] += int(record["tensor_references"])
                previous["source_storage_ranges"].update(
                    record["source_storage_ranges"]
                )
                continue
        merged.append(record)
    return merged


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
    storage_records: dict[tuple[str, int], dict[str, object]] = {}
    object_count = 0
    truncated = False

    def visit(value: object, path: str, depth: int) -> None:
        nonlocal object_count, truncated
        if isinstance(value, torch.Tensor):
            if selected_device is not None and value.device != selected_device:
                return
            logical_bytes = int(value.numel() * value.element_size())
            storage_device, storage_pointer, storage_bytes = _storage_identity(value)
            storage_key = (storage_device, storage_pointer)
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
                    "device": storage_device,
                    "data_ptr": storage_pointer,
                    "storage_bytes": storage_bytes,
                    "reported_storage_sizes": set(),
                    "paths": [],
                    "categories": set(),
                    "dtypes": set(),
                    "tensor_references": 0,
                },
            )
            storage["storage_bytes"] = max(int(storage["storage_bytes"]), storage_bytes)
            storage["reported_storage_sizes"].add(storage_bytes)
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
        if _is_passive_mapping(value):
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
    raw_storage_bytes = sum(
        int(storage["storage_bytes"]) for storage in storage_records.values()
    )
    coalesced = _coalesce_storage_ranges(list(storage_records.values()))
    for storage in coalesced:
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
                "source_storage_ranges": [
                    {"data_ptr": pointer, "storage_bytes": storage_bytes}
                    for pointer, storage_bytes in sorted(
                        storage["source_storage_ranges"]
                    )
                ],
                "source_storage_range_count": len(storage["source_storage_ranges"]),
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
        "raw_storage_range_count": len(storage_records),
        "raw_storage_bytes_before_overlap_coalescing": raw_storage_bytes,
        "overlap_collapsed_bytes": raw_storage_bytes - unique_storage_bytes,
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


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _storage_identity_records(
    inventory: Mapping[str, object],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "device": storage["device"],
                "data_ptr": storage["data_ptr"],
                "storage_bytes": storage["storage_bytes"],
                "source_storage_ranges": storage["source_storage_ranges"],
            }
            for storage in inventory["storages"]
        ),
        key=lambda item: (
            str(item["device"]),
            int(item["data_ptr"]),
            int(item["storage_bytes"]),
        ),
    )


def _storage_owner_records(
    inventory: Mapping[str, object],
) -> list[dict[str, object]]:
    records = []
    for storage in inventory["storages"]:
        identity = {
            "device": storage["device"],
            "data_ptr": storage["data_ptr"],
            "storage_bytes": storage["storage_bytes"],
        }
        records.extend(
            {**identity, "path": path}
            for path in storage["paths"]
        )
    return sorted(
        records,
        key=lambda item: (
            str(item["device"]),
            int(item["data_ptr"]),
            int(item["storage_bytes"]),
            str(item["path"]),
        ),
    )


def _storage_multiplicity_records(
    inventory: Mapping[str, object],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "device": storage["device"],
                "data_ptr": storage["data_ptr"],
                "storage_bytes": storage["storage_bytes"],
                "tensor_references": storage["tensor_references"],
            }
            for storage in inventory["storages"]
        ),
        key=lambda item: (
            str(item["device"]),
            int(item["data_ptr"]),
            int(item["storage_bytes"]),
        ),
    )


def _counter_diff(
    before: list[dict[str, object]],
    after: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    before_by_json = Counter(
        json.dumps(item, allow_nan=False, separators=(",", ":"), sort_keys=True)
        for item in before
    )
    after_by_json = Counter(
        json.dumps(item, allow_nan=False, separators=(",", ":"), sort_keys=True)
        for item in after
    )

    def expand(counter: Counter[str]) -> list[dict[str, object]]:
        return [
            json.loads(serialized)
            for serialized in sorted(counter)
            for _ in range(counter[serialized])
        ]

    return expand(after_by_json - before_by_json), expand(
        before_by_json - after_by_json
    )


def compare_tensor_inventories(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    maximum_reported_differences: int = 1024,
) -> dict[str, object]:
    """Compare two bounded inventories without conflating storage and aliases.

    CUDA storage addresses are meaningful only inside the process that produced
    the two inventories.  The returned hashes are therefore evidence for a
    repeated observation in one diagnostic phase, not portable object IDs.
    """

    if maximum_reported_differences <= 0:
        raise ValueError("maximum_reported_differences must be positive")
    for label, inventory in (("before", before), ("after", after)):
        if inventory.get("schema_version") != 1:
            raise ValueError(f"{label} inventory schema is unsupported")
        if inventory.get("truncated") is not False:
            raise ValueError(f"{label} inventory is truncated")
        if inventory.get("all_storages_reported") is not True:
            raise ValueError(f"{label} inventory omits storage records")

    before_storage = _storage_identity_records(before)
    after_storage = _storage_identity_records(after)
    before_owners = _storage_owner_records(before)
    after_owners = _storage_owner_records(after)
    before_multiplicity = _storage_multiplicity_records(before)
    after_multiplicity = _storage_multiplicity_records(after)
    before_references = list(before["tensor_references"])
    after_references = list(after["tensor_references"])

    added_storage, removed_storage = _counter_diff(before_storage, after_storage)
    added_owners, removed_owners = _counter_diff(before_owners, after_owners)
    changed_multiplicity_after, changed_multiplicity_before = _counter_diff(
        before_multiplicity,
        after_multiplicity,
    )
    added_references, removed_references = _counter_diff(
        before_references,
        after_references,
    )
    full_identical = before == after
    storage_identity_equal = not added_storage and not removed_storage
    owner_paths_equal = not added_owners and not removed_owners
    tensor_references_equal = not added_references and not removed_references
    transient_cache_only = (
        not full_identical
        and storage_identity_equal
        and not removed_owners
        and not removed_references
        and bool(added_references)
        and all(
            "transient_cache" in str(item["path"]).lower()
            for item in added_references
        )
        and all(
            "transient_cache" in str(item["path"]).lower()
            for item in added_owners
        )
    )
    if full_identical:
        classification = "identical"
    elif not storage_identity_equal:
        classification = "storage_identity_changed"
    elif transient_cache_only:
        classification = "transient_cache_reference_expansion"
    else:
        classification = "reference_graph_changed_without_storage_change"

    aggregate_fields = (
        "scope",
        "device_filter",
        "visited_object_count",
        "tensor_reference_count",
        "unique_storage_count",
        "logical_tensor_bytes",
        "unique_storage_bytes",
        "raw_storage_range_count",
        "raw_storage_bytes_before_overlap_coalescing",
        "overlap_collapsed_bytes",
        "aliased_tensor_reference_count",
        "exclusive_storage_bytes_by_category",
        "shared_storage_bytes_by_category",
        "unique_storage_bytes_by_dtype",
        "reported_storage_count",
    )
    field_differences = {
        field: {"before": before.get(field), "after": after.get(field)}
        for field in aggregate_fields
        if before.get(field) != after.get(field)
    }

    def bounded(values: list[dict[str, object]]) -> list[dict[str, object]]:
        return values[:maximum_reported_differences]

    difference_groups = (
        added_storage,
        removed_storage,
        added_owners,
        removed_owners,
        changed_multiplicity_after,
        changed_multiplicity_before,
        added_references,
        removed_references,
    )
    return {
        "schema_version": 1,
        "classification": classification,
        "full_inventory_identical": full_identical,
        "unique_storage_identity_equal": storage_identity_equal,
        "storage_owner_paths_equal": owner_paths_equal,
        "tensor_references_equal": tensor_references_equal,
        "transient_cache_reference_expansion_only": transient_cache_only,
        "before_hashes": {
            "full_inventory_sha256": _canonical_sha256(before),
            "unique_storage_identity_sha256": _canonical_sha256(before_storage),
            "storage_owner_paths_sha256": _canonical_sha256(before_owners),
            "storage_reference_multiplicity_sha256": _canonical_sha256(
                before_multiplicity
            ),
            "tensor_references_sha256": _canonical_sha256(before_references),
        },
        "after_hashes": {
            "full_inventory_sha256": _canonical_sha256(after),
            "unique_storage_identity_sha256": _canonical_sha256(after_storage),
            "storage_owner_paths_sha256": _canonical_sha256(after_owners),
            "storage_reference_multiplicity_sha256": _canonical_sha256(
                after_multiplicity
            ),
            "tensor_references_sha256": _canonical_sha256(after_references),
        },
        "field_differences": field_differences,
        "added_storage_identity_count": len(added_storage),
        "removed_storage_identity_count": len(removed_storage),
        "added_storage_owner_path_count": len(added_owners),
        "removed_storage_owner_path_count": len(removed_owners),
        "changed_storage_multiplicity_count": len(changed_multiplicity_after),
        "added_tensor_reference_count": len(added_references),
        "removed_tensor_reference_count": len(removed_references),
        "added_storage_identities": bounded(added_storage),
        "removed_storage_identities": bounded(removed_storage),
        "added_storage_owner_paths": bounded(added_owners),
        "removed_storage_owner_paths": bounded(removed_owners),
        "storage_multiplicity_before": bounded(changed_multiplicity_before),
        "storage_multiplicity_after": bounded(changed_multiplicity_after),
        "added_tensor_references": bounded(added_references),
        "removed_tensor_references": bounded(removed_references),
        "all_differences_reported": all(
            len(values) <= maximum_reported_differences
            for values in difference_groups
        ),
        "maximum_reported_differences": maximum_reported_differences,
    }


__all__ = ["build_tensor_inventory", "compare_tensor_inventories"]
