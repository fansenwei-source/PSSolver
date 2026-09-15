"""Reconcile bounded tensor inventories with CUDA allocator blocks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import numbers


_ACTIVE_ALLOCATED = "active_allocated"


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _segment_matches_device(
    segment: Mapping[str, object],
    *,
    device_index: int,
    visible_device_count: int,
) -> bool:
    raw = segment.get("device")
    if raw is None:
        if visible_device_count != 1:
            raise ValueError(
                "allocator segments without device identity require exactly "
                "one visible CUDA device"
            )
        return True
    if isinstance(raw, str):
        normalized = raw.removeprefix("cuda:")
        try:
            resolved = int(normalized)
        except ValueError as exc:
            raise ValueError("allocator segment device is invalid") from exc
    else:
        resolved = _nonnegative_integer(raw, "allocator segment device")
    return resolved == device_index


def _allocator_blocks(
    segments: Sequence[Mapping[str, object]],
    *,
    device_index: int,
    visible_device_count: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    blocks: list[dict[str, object]] = []
    selected_segments = 0
    selected_segment_bytes = 0
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise TypeError("allocator segments must contain mappings")
        if not _segment_matches_device(
            segment,
            device_index=device_index,
            visible_device_count=visible_device_count,
        ):
            continue
        selected_segments += 1
        segment_start = _nonnegative_integer(
            segment.get("address"), "allocator segment address"
        )
        segment_size = _nonnegative_integer(
            segment.get("total_size"), "allocator segment size"
        )
        selected_segment_bytes += segment_size
        cursor = segment_start
        raw_blocks = segment.get("blocks")
        if not isinstance(raw_blocks, Sequence) or isinstance(
            raw_blocks, (str, bytes, bytearray)
        ):
            raise TypeError("allocator segment blocks must be a sequence")
        for block_index, block in enumerate(raw_blocks):
            if not isinstance(block, Mapping):
                raise TypeError("allocator blocks must be mappings")
            start = _nonnegative_integer(
                block.get("address", cursor), "allocator block address"
            )
            size = _nonnegative_integer(
                block.get("size"), "allocator block size"
            )
            requested = _nonnegative_integer(
                block.get("requested_size", size),
                "allocator block requested size",
            )
            state = block.get("state")
            if not isinstance(state, str) or not state:
                raise ValueError("allocator block state must be a nonempty string")
            if start < segment_start or start + size > segment_start + segment_size:
                raise ValueError("allocator block lies outside its segment")
            blocks.append(
                {
                    "block_id": len(blocks),
                    "segment_index": segment_index,
                    "block_index": block_index,
                    "address": start,
                    "size": size,
                    "requested_size": requested,
                    "state": state,
                }
            )
            cursor = start + size
        if cursor > segment_start + segment_size:
            raise ValueError("allocator blocks exceed their segment")
    return blocks, {
        "selected_segment_count": selected_segments,
        "selected_segment_bytes": selected_segment_bytes,
    }


def _covered_bytes(ranges: Sequence[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if end < start:
            raise ValueError("storage range end precedes its start")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def reconcile_tensor_inventory_with_cuda_allocator(
    inventory: Mapping[str, object],
    allocator_segments: Sequence[Mapping[str, object]],
    *,
    allocator_allocated_bytes: int,
    device_index: int,
    visible_device_count: int,
    maximum_reported_unmatched_storages: int = 32,
) -> dict[str, object]:
    """Map every reported storage range to a CUDA allocator block.

    The result contains only integer addresses, sizes, owner paths, and
    categories.  It does not retain tensors or the allocator snapshot.
    """

    if not isinstance(inventory, Mapping):
        raise TypeError("inventory must be a mapping")
    if inventory.get("all_storages_reported") is not True:
        raise ValueError("allocator reconciliation requires all storages")
    allocated = _nonnegative_integer(
        allocator_allocated_bytes, "allocator allocated bytes"
    )
    device_index = _nonnegative_integer(device_index, "device index")
    visible_device_count = _nonnegative_integer(
        visible_device_count, "visible device count"
    )
    if visible_device_count <= 0 or device_index >= visible_device_count:
        raise ValueError("CUDA device identity is inconsistent")
    maximum_reported_unmatched_storages = _nonnegative_integer(
        maximum_reported_unmatched_storages,
        "maximum reported unmatched storages",
    )
    if maximum_reported_unmatched_storages <= 0:
        raise ValueError("maximum reported unmatched storages must be positive")
    if not isinstance(allocator_segments, Sequence) or isinstance(
        allocator_segments, (str, bytes, bytearray)
    ):
        raise TypeError("allocator segments must be a sequence")

    blocks, segment_summary = _allocator_blocks(
        allocator_segments,
        device_index=device_index,
        visible_device_count=visible_device_count,
    )
    active_blocks = [
        block for block in blocks if block["state"] == _ACTIVE_ALLOCATED
    ]
    active_block_bytes = sum(int(block["size"]) for block in active_blocks)
    active_requested_bytes = sum(
        int(block["requested_size"]) for block in active_blocks
    )

    raw_storages = inventory.get("storages")
    if not isinstance(raw_storages, Sequence) or isinstance(
        raw_storages, (str, bytes, bytearray)
    ):
        raise TypeError("inventory storages must be a sequence")
    expected_count = _nonnegative_integer(
        inventory.get("unique_storage_count"), "inventory storage count"
    )
    if len(raw_storages) != expected_count:
        raise ValueError("inventory storage list is incomplete")

    matched_by_state_count: defaultdict[str, int] = defaultdict(int)
    matched_by_state_bytes: defaultdict[str, int] = defaultdict(int)
    active_coverage: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    unmatched: list[dict[str, object]] = []
    ambiguous_count = 0
    ambiguous_bytes = 0
    zero_sized_count = 0
    positive_storage_bytes = 0

    for storage in raw_storages:
        if not isinstance(storage, Mapping):
            raise TypeError("inventory storage records must be mappings")
        start = _nonnegative_integer(storage.get("data_ptr"), "storage data pointer")
        size = _nonnegative_integer(storage.get("storage_bytes"), "storage bytes")
        if size == 0:
            zero_sized_count += 1
            continue
        positive_storage_bytes += size
        end = start + size
        containing = [
            block
            for block in blocks
            if start >= int(block["address"])
            and end <= int(block["address"]) + int(block["size"])
        ]
        if len(containing) == 1:
            block = containing[0]
            state = str(block["state"])
            matched_by_state_count[state] += 1
            matched_by_state_bytes[state] += size
            if state == _ACTIVE_ALLOCATED:
                active_coverage[int(block["block_id"])].append((start, end))
                continue
        if len(containing) > 1:
            ambiguous_count += 1
            ambiguous_bytes += size
        intersecting = [
            block
            for block in blocks
            if start < int(block["address"]) + int(block["size"])
            and end > int(block["address"])
        ]
        unmatched.append(
            {
                "data_ptr": start,
                "storage_bytes": size,
                "paths": list(storage.get("paths", ())),
                "categories": list(storage.get("categories", ())),
                "containing_block_count": len(containing),
                "intersecting_allocator_blocks": [
                    {
                        "address": block["address"],
                        "size": block["size"],
                        "state": block["state"],
                    }
                    for block in intersecting
                ],
            }
        )

    active_inventory_bytes = int(
        matched_by_state_bytes.get(_ACTIVE_ALLOCATED, 0)
    )
    bytes_outside_active = positive_storage_bytes - active_inventory_bytes
    referenced_active_block_ids = set(active_coverage)
    referenced_active_block_bytes = sum(
        int(block["size"])
        for block in active_blocks
        if int(block["block_id"]) in referenced_active_block_ids
    )
    covered_active_block_bytes = sum(
        _covered_bytes(ranges) for ranges in active_coverage.values()
    )
    counter_matches_blocks = active_block_bytes == allocated
    all_inside_active = bytes_outside_active == 0 and ambiguous_count == 0
    reconciled = counter_matches_blocks and all_inside_active
    unmatched.sort(key=lambda item: (-int(item["storage_bytes"]), item["data_ptr"]))

    if reconciled:
        classification = "fully_reconciled_with_active_allocator_blocks"
    elif not counter_matches_blocks:
        classification = "allocator_counter_and_block_snapshot_differ"
    elif bytes_outside_active > 0:
        classification = "runtime_storages_outside_active_allocator_blocks"
    else:
        classification = "ambiguous_allocator_block_mapping"

    return {
        "schema_version": 1,
        "classification": classification,
        "device_index": device_index,
        "visible_device_count": visible_device_count,
        **segment_summary,
        "allocator_block_count": len(blocks),
        "active_allocated_block_count": len(active_blocks),
        "active_allocated_block_bytes": active_block_bytes,
        "active_allocated_requested_bytes": active_requested_bytes,
        "allocator_allocated_bytes": allocated,
        "allocator_counter_matches_active_block_bytes": counter_matches_blocks,
        "inventory_positive_storage_bytes": positive_storage_bytes,
        "inventory_storage_count": expected_count,
        "zero_sized_storage_count": zero_sized_count,
        "matched_storage_count_by_allocator_state": dict(
            sorted(matched_by_state_count.items())
        ),
        "matched_storage_bytes_by_allocator_state": dict(
            sorted(matched_by_state_bytes.items())
        ),
        "inventory_bytes_inside_active_allocator_blocks": active_inventory_bytes,
        "inventory_bytes_outside_active_allocator_blocks": bytes_outside_active,
        "ambiguous_storage_count": ambiguous_count,
        "ambiguous_storage_bytes": ambiguous_bytes,
        "referenced_active_allocator_block_count": len(
            referenced_active_block_ids
        ),
        "referenced_active_allocator_block_bytes": referenced_active_block_bytes,
        "covered_bytes_within_referenced_active_blocks": (
            covered_active_block_bytes
        ),
        "uncovered_bytes_within_referenced_active_blocks": (
            referenced_active_block_bytes - covered_active_block_bytes
        ),
        "unreferenced_active_allocator_block_bytes": (
            active_block_bytes - referenced_active_block_bytes
        ),
        "all_inventory_storages_within_active_allocator_blocks": (
            all_inside_active
        ),
        "accounting_reconciled": reconciled,
        "unmatched_storage_count": len(unmatched),
        "unmatched_storage_bytes": sum(
            int(item["storage_bytes"]) for item in unmatched
        ),
        "unmatched_storages": unmatched[:maximum_reported_unmatched_storages],
        "all_unmatched_storages_reported": (
            len(unmatched) <= maximum_reported_unmatched_storages
        ),
    }


__all__ = ["reconcile_tensor_inventory_with_cuda_allocator"]
