"""Tests for the conservative R2R-C bounded-transform policy contract."""

from __future__ import annotations

import json

import pytest
import torch

from pssolver import (
    BoundedTransformQualificationCell,
    BoundedTransformSelectionContext,
    QualifiedBoundedTransformPolicy,
    load_bounded_transform_policy,
)
from pssolver.transforms import TensorProductTransformBackend


def _context(**overrides):
    values = {
        "geometry": "test_slab",
        "local_axis": 0,
        "kind": "dct",
        "direction": "forward",
        "execution_mode": "full",
        "physical_size": 8,
        "retained_count": 8,
        "line_count": 4,
        "device_type": "cpu",
        "device_name": "CPU",
        "real_dtype": "float64",
        "value_type": "real",
    }
    values.update(overrides)
    return BoundedTransformSelectionContext(**values)


def _cell(cell_id="qualified", **overrides):
    context = _context()
    values = {
        "cell_id": cell_id,
        "geometry": context.geometry,
        "local_axis": context.local_axis,
        "kind": context.kind,
        "direction": context.direction,
        "execution_mode": context.execution_mode,
        "physical_size": context.physical_size,
        "retained_count": context.retained_count,
        "minimum_line_count": context.line_count,
        "device_type": context.device_type,
        "device_name": context.device_name,
        "real_dtype": context.real_dtype,
        "value_type": context.value_type,
        "selected_algorithm": "fft",
        "evidence_id": f"evidence:{cell_id}",
    }
    values.update(overrides)
    return BoundedTransformQualificationCell(**values)


def _policy(*cells):
    return QualifiedBoundedTransformPolicy(
        name="test_policy",
        cells=tuple(cells),
    )


def test_policy_matches_only_the_complete_qualified_context():
    policy = _policy(_cell())

    selected = policy.select(_context())
    assert selected.algorithm == "fft"
    assert selected.reason == "matched_qualification_cell"
    assert selected.matched_cell_id == "qualified"

    for changed in (
        {"geometry": "channel"},
        {"local_axis": 1},
        {"kind": "dst"},
        {"direction": "inverse"},
        {"physical_size": 9, "retained_count": 9},
        {"line_count": 3},
        {"device_name": "Different CPU"},
        {"real_dtype": "float32"},
        {"value_type": "complex"},
    ):
        fallback = policy.select(_context(**changed))
        assert fallback.algorithm == "dense"
        assert fallback.reason == "no_matching_qualification_cell"
        assert fallback.matched_cell_id is None


def test_policy_metadata_roundtrip_and_artifact_loader(tmp_path):
    policy = _policy(_cell(local_axis=None))
    restored = QualifiedBoundedTransformPolicy.from_metadata(
        policy.to_metadata()
    )
    assert restored == policy

    path = tmp_path / "policy.json"
    path.write_text(
        json.dumps({"artifact": "qualification", "policy": policy.to_metadata()})
    )
    assert load_bounded_transform_policy(path) == policy


def test_policy_rejects_ambiguous_cells():
    policy = _policy(_cell("first"), _cell("second"))

    with pytest.raises(RuntimeError, match="ambiguous"):
        policy.select(_context())


def test_auto_requires_policy_and_forced_modes_reject_one():
    with pytest.raises(ValueError, match="require a qualified policy"):
        TensorProductTransformBackend(
            (8,),
            (1.0,),
            device="cpu",
            bounded_transform_algorithm="auto",
        )
    with pytest.raises(ValueError, match="only valid with the auto"):
        TensorProductTransformBackend(
            (8,),
            (1.0,),
            device="cpu",
            bounded_transform_algorithm="dense",
            bounded_transform_policy=_policy(),
        )


def test_auto_can_select_fft_forward_and_dense_inverse_independently():
    policy = _policy(_cell(direction="forward"))
    automatic = TensorProductTransformBackend(
        (8,),
        (1.0,),
        device="cpu",
        dtype=torch.float64,
        bounded_transform_algorithm="auto",
        bounded_transform_geometry="test_slab",
        bounded_transform_policy=policy,
    )
    dense = TensorProductTransformBackend(
        (8,),
        (1.0,),
        device="cpu",
        dtype=torch.float64,
    )
    values = torch.randn(4, 8, dtype=torch.float64)

    coefficients = automatic._apply_axis_transform(values, "dct", -1)
    restored = automatic._apply_axis_transform(
        coefficients,
        "dct",
        -1,
        inverse=True,
    )
    torch.testing.assert_close(
        coefficients,
        dense._apply_axis_transform(values, "dct", -1),
        rtol=6.0e-12,
        atol=6.0e-12,
    )
    torch.testing.assert_close(restored, values, rtol=6.0e-12, atol=6.0e-12)

    metadata = automatic.bounded_transform_selection_metadata()
    assert metadata["requested_algorithm"] == "auto"
    assert metadata["decision_mode"] == "qualified_allow_list"
    assert metadata["effective_algorithms"] == ["dense", "fft"]
    assert len(metadata["observed_decisions"]) == 2
    assert automatic._fft_r2r_cache
    assert automatic._matrix_cache


def test_auto_records_truncated_context_and_uses_dense_fallback():
    policy = _policy(
        _cell(
            kind="dst",
            execution_mode="truncated",
            physical_size=8,
            retained_count=4,
        )
    )
    backend = TensorProductTransformBackend(
        (8,),
        (1.0,),
        device="cpu",
        dtype=torch.float64,
        bounded_transform_algorithm="auto",
        bounded_transform_geometry="test_slab",
        bounded_transform_policy=policy,
    )
    values = torch.randn(4, 8, dtype=torch.float64)

    coefficients = backend._apply_retained_real_axis_transform(
        values,
        "dst",
        0,
        4,
        inverse=False,
    )
    assert coefficients.shape == (4, 4)
    entry = backend.bounded_transform_selection_metadata()[
        "observed_decisions"
    ][0]
    assert entry["context"]["execution_mode"] == "truncated"
    assert entry["context"]["retained_fraction"] == 0.5
    assert entry["selection"]["algorithm"] == "fft"


@pytest.mark.parametrize(
    "overrides,match",
    (
        ({"retained_count": 0}, "retained_count"),
        ({"execution_mode": "truncated"}, "disagrees"),
        ({"line_count": 0}, "line_count"),
        ({"value_type": "integer"}, "value_type"),
    ),
)
def test_context_rejects_invalid_contracts(overrides, match):
    with pytest.raises(ValueError, match=match):
        _context(**overrides)
