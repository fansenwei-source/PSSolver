from __future__ import annotations

from dataclasses import dataclass

import torch

from pssolver.diagnostics import build_tensor_inventory, compare_tensor_inventories
from pssolver.diagnostics.tensor_inventory import _coalesce_storage_ranges


@dataclass
class _Runtime:
    fields: object
    unrelated: object


@dataclass
class _Fields:
    spatial: torch.Tensor
    view: torch.Tensor


_Runtime.__module__ = "pssolver.testing"
_Fields.__module__ = "pssolver.testing"


def test_inventory_deduplicates_views_and_keeps_explicit_scope():
    storage = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    runtime = _Runtime(
        fields=_Fields(spatial=storage, view=storage[:, :, :2]),
        unrelated=object(),
    )

    report = build_tensor_inventory(
        {"runtime": runtime},
        device="cpu",
    )

    assert report["truncated"] is False
    assert report["tensor_reference_count"] == 2
    assert report["unique_storage_count"] == 1
    assert report["unique_storage_bytes"] == storage.untyped_storage().nbytes()
    assert report["aliased_tensor_reference_count"] == 1
    assert report["all_storages_reported"] is True
    categories = {
        item["path"]: item["category"] for item in report["tensor_references"]
    }
    assert categories["runtime.fields.spatial"] == "solver_fields"


def test_inventory_reports_a_bounded_traversal_instead_of_silently_omitting():
    report = build_tensor_inventory(
        {"root": [[torch.ones(1)]]},
        device="cpu",
        max_depth=0,
    )

    assert report["truncated"] is True
    assert report["tensor_reference_count"] == 0


def test_inventory_filters_tensors_by_device():
    report = build_tensor_inventory(
        {"root": {"value": torch.ones(3)}},
        device="meta",
    )

    assert report["tensor_reference_count"] == 0
    assert report["unique_storage_bytes"] == 0


def test_overlapping_device_address_ranges_are_counted_once():
    records = [
        {
            "device": "cuda:0",
            "data_ptr": 1000,
            "storage_bytes": 100,
            "reported_storage_sizes": {100},
            "paths": ["runtime.parent"],
            "categories": {"model"},
            "dtypes": {"torch.float64"},
            "tensor_references": 1,
        },
        {
            "device": "cuda:0",
            "data_ptr": 1050,
            "storage_bytes": 100,
            "reported_storage_sizes": {100},
            "paths": ["runtime.alias"],
            "categories": {"algebraic_representations"},
            "dtypes": {"torch.float64"},
            "tensor_references": 1,
        },
    ]

    merged = _coalesce_storage_ranges(records)

    assert len(merged) == 1
    assert merged[0]["data_ptr"] == 1000
    assert merged[0]["storage_bytes"] == 150
    assert merged[0]["paths"] == {"runtime.parent", "runtime.alias"}


def test_inventory_comparison_separates_storage_identity_from_owner_aliases():
    storage = torch.arange(8, dtype=torch.float64)
    before = build_tensor_inventory(
        {"runtime": {"value": storage}},
        device="cpu",
    )
    after = build_tensor_inventory(
        {
            "runtime": {
                "value": storage,
                "transient_cache": {"alias": storage},
            }
        },
        device="cpu",
    )

    comparison = compare_tensor_inventories(before, after)

    assert comparison["classification"] == "transient_cache_reference_expansion"
    assert comparison["full_inventory_identical"] is False
    assert comparison["unique_storage_identity_equal"] is True
    assert comparison["storage_owner_paths_equal"] is False
    assert comparison["tensor_references_equal"] is False
    assert comparison["transient_cache_reference_expansion_only"] is True
    assert comparison["added_storage_identity_count"] == 0
    assert comparison["removed_storage_identity_count"] == 0
    assert comparison["added_storage_owner_path_count"] == 1
    assert comparison["added_tensor_reference_count"] == 1
    assert comparison["changed_storage_multiplicity_count"] == 1
    assert comparison["all_differences_reported"] is True
    assert comparison["before_hashes"]["unique_storage_identity_sha256"] == (
        comparison["after_hashes"]["unique_storage_identity_sha256"]
    )
    assert comparison["before_hashes"]["storage_owner_paths_sha256"] != (
        comparison["after_hashes"]["storage_owner_paths_sha256"]
    )


def test_inventory_comparison_reports_new_storage_identity():
    before = build_tensor_inventory(
        {"runtime": {"value": torch.ones(2)}},
        device="cpu",
    )
    after = build_tensor_inventory(
        {"runtime": {"value": torch.ones(3)}},
        device="cpu",
    )

    comparison = compare_tensor_inventories(before, after)

    assert comparison["classification"] == "storage_identity_changed"
    assert comparison["unique_storage_identity_equal"] is False
    assert comparison["added_storage_identity_count"] == 1
    assert comparison["removed_storage_identity_count"] == 1


def test_inventory_comparison_rejects_incomplete_evidence():
    report = build_tensor_inventory(
        {"runtime": {"value": torch.ones(2)}},
        device="cpu",
    )
    incomplete = dict(report)
    incomplete["all_storages_reported"] = False

    try:
        compare_tensor_inventories(report, incomplete)
    except ValueError as error:
        assert "omits storage records" in str(error)
    else:
        raise AssertionError("incomplete inventory should be rejected")
