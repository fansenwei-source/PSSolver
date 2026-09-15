from __future__ import annotations

from dataclasses import dataclass

import torch

from pssolver.diagnostics import build_tensor_inventory


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
