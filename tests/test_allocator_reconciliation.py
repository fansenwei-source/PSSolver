from __future__ import annotations

import pytest

from pssolver.diagnostics import (
    reconcile_tensor_inventory_with_cuda_allocator,
)


def _storage(address: int, size: int, path: str) -> dict[str, object]:
    return {
        "data_ptr": address,
        "storage_bytes": size,
        "paths": [path],
        "categories": ["model"],
    }


def _inventory(*storages: dict[str, object]) -> dict[str, object]:
    return {
        "all_storages_reported": True,
        "unique_storage_count": len(storages),
        "storages": list(storages),
    }


def _segments() -> list[dict[str, object]]:
    return [
        {
            "device": 0,
            "address": 1000,
            "total_size": 500,
            "blocks": [
                {
                    "size": 100,
                    "requested_size": 80,
                    "state": "active_allocated",
                },
                {
                    "size": 100,
                    "requested_size": 0,
                    "state": "inactive",
                },
                {
                    "size": 300,
                    "requested_size": 280,
                    "state": "active_allocated",
                },
            ],
        }
    ]


def test_reconciliation_maps_runtime_storages_to_active_allocator_blocks():
    report = reconcile_tensor_inventory_with_cuda_allocator(
        _inventory(
            _storage(1000, 80, "runtime.first"),
            _storage(1200, 280, "runtime.second"),
        ),
        _segments(),
        allocator_allocated_bytes=400,
        device_index=0,
        visible_device_count=1,
    )

    assert report["classification"] == (
        "fully_reconciled_with_active_allocator_blocks"
    )
    assert report["accounting_reconciled"] is True
    assert report["active_allocated_block_bytes"] == 400
    assert report["active_allocated_requested_bytes"] == 360
    assert report["inventory_bytes_inside_active_allocator_blocks"] == 360
    assert report["inventory_bytes_outside_active_allocator_blocks"] == 0
    assert report["uncovered_bytes_within_referenced_active_blocks"] == 40
    assert report["unmatched_storages"] == []


def test_reconciliation_reports_exact_unmatched_owner_evidence():
    report = reconcile_tensor_inventory_with_cuda_allocator(
        _inventory(
            _storage(1000, 80, "runtime.matched"),
            _storage(3000, 256, "runtime.external"),
        ),
        _segments(),
        allocator_allocated_bytes=400,
        device_index=0,
        visible_device_count=1,
    )

    assert report["classification"] == (
        "runtime_storages_outside_active_allocator_blocks"
    )
    assert report["accounting_reconciled"] is False
    assert report["inventory_bytes_outside_active_allocator_blocks"] == 256
    assert report["unmatched_storage_count"] == 1
    assert report["unmatched_storages"][0]["paths"] == ["runtime.external"]


def test_reconciliation_reports_storage_inside_nonactive_allocator_block():
    report = reconcile_tensor_inventory_with_cuda_allocator(
        _inventory(_storage(1100, 80, "runtime.inactive")),
        _segments(),
        allocator_allocated_bytes=400,
        device_index=0,
        visible_device_count=1,
    )

    assert report["classification"] == (
        "runtime_storages_outside_active_allocator_blocks"
    )
    assert report["matched_storage_bytes_by_allocator_state"] == {
        "inactive": 80
    }
    assert report["unmatched_storage_count"] == 1
    assert report["unmatched_storages"][0]["paths"] == ["runtime.inactive"]
    assert report["unmatched_storages"][0]["intersecting_allocator_blocks"] == [
        {"address": 1100, "size": 100, "state": "inactive"}
    ]


def test_reconciliation_detects_allocator_counter_snapshot_mismatch():
    report = reconcile_tensor_inventory_with_cuda_allocator(
        _inventory(_storage(1000, 80, "runtime.value")),
        _segments(),
        allocator_allocated_bytes=399,
        device_index=0,
        visible_device_count=1,
    )

    assert report["classification"] == (
        "allocator_counter_and_block_snapshot_differ"
    )
    assert report["allocator_counter_matches_active_block_bytes"] is False
    assert report["accounting_reconciled"] is False


def test_reconciliation_requires_device_identity_for_multigpu_snapshot():
    segments = _segments()
    segments[0].pop("device")

    with pytest.raises(ValueError, match="exactly one visible CUDA device"):
        reconcile_tensor_inventory_with_cuda_allocator(
            _inventory(_storage(1000, 80, "runtime.value")),
            segments,
            allocator_allocated_bytes=400,
            device_index=0,
            visible_device_count=2,
        )
