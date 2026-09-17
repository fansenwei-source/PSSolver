"""Tests for converting R2R-B evidence into an exact R2R-C policy."""

from __future__ import annotations

import pytest

from benchmarks.build_bounded_transform_policy import build_policy_artifact
from pssolver import QualifiedBoundedTransformPolicy


def _case(algorithm, *, forward_seconds, inverse_seconds):
    return {
        "case_id": f"{algorithm}_dct_truncated_real_n320_r160",
        "algorithm": algorithm,
        "size": 320,
        "kind": "dct",
        "execution_mode": "truncated",
        "value_type": "real",
        "retained_count": 160,
        "correctness": {
            "tolerance_relative_l2": 5.0e-13,
            "all_finite": True,
            "dense_reference_forward_relative_l2": (
                0.0 if algorithm == "dense" else 2.0e-14
            ),
            "dense_reference_inverse_relative_l2": (
                0.0 if algorithm == "dense" else 3.0e-14
            ),
        },
        "timing": {
            "forward": {
                "samples": 12,
                "median_seconds": forward_seconds,
            },
            "inverse": {
                "samples": 12,
                "median_seconds": inverse_seconds,
            },
        },
        "cuda_memory": {
            "peak_allocated_bytes": 100 if algorithm == "dense" else 200,
        },
    }


def _benchmark():
    return {
        "benchmark": "r2r_b_dense_vs_fft_bounded_axis",
        "config": {
            "algorithms": ["dense", "fft"],
            "device": "cuda",
            "dtype": "float64",
            "line_count": 4096,
        },
        "environment": {
            "cuda_available": True,
            "device_name": "Test GPU",
        },
        "cases": [
            _case("dense", forward_seconds=2.0, inverse_seconds=1.0),
            _case("fft", forward_seconds=1.0, inverse_seconds=2.0),
        ],
    }


def test_builder_accepts_only_directions_that_pass_every_gate():
    artifact = build_policy_artifact(
        _benchmark(),
        source_path="/tmp/input.json",
        source_sha256="a" * 64,
        policy_name="test_gpu_policy",
        geometry="plane",
        local_axis=2,
        minimum_speedup=1.10,
        maximum_peak_allocated_ratio=3.0,
        minimum_samples=10,
    )

    assert artifact["accepted_cell_count"] == 1
    reviews = artifact["reviewed_directions"]
    assert [review["accepted"] for review in reviews] == [True, False]
    policy = QualifiedBoundedTransformPolicy.from_metadata(
        artifact["policy"]
    )
    cell = policy.cells[0]
    assert cell.geometry == "plane"
    assert cell.local_axis == 2
    assert cell.direction == "forward"
    assert cell.physical_size == 320
    assert cell.retained_count == 160
    assert cell.minimum_line_count == 4096
    assert cell.device_name == "Test GPU"
    assert cell.evidence_id.startswith("sha256:" + "a" * 64)


def test_builder_rejects_memory_regression_and_insufficient_samples():
    artifact = build_policy_artifact(
        _benchmark(),
        source_path="/tmp/input.json",
        source_sha256="b" * 64,
        policy_name="rejected",
        geometry="plane",
        local_axis=None,
        minimum_speedup=1.10,
        maximum_peak_allocated_ratio=1.5,
        minimum_samples=13,
    )

    assert artifact["accepted_cell_count"] == 0
    assert all(
        review["accepted"] is False
        for review in artifact["reviewed_directions"]
    )


@pytest.mark.parametrize(
    "override,match",
    (
        ({"minimum_speedup": 1.0}, "greater than one"),
        ({"maximum_peak_allocated_ratio": 0.9}, "at least one"),
        ({"minimum_samples": 0}, "positive"),
        ({"geometry": ""}, "non-empty"),
    ),
)
def test_builder_rejects_invalid_contract(override, match):
    arguments = {
        "source_path": "/tmp/input.json",
        "source_sha256": "c" * 64,
        "policy_name": "policy",
        "geometry": "plane",
        "local_axis": None,
        "minimum_speedup": 1.10,
        "maximum_peak_allocated_ratio": 3.0,
        "minimum_samples": 10,
    }
    arguments.update(override)
    with pytest.raises(ValueError, match=match):
        build_policy_artifact(_benchmark(), **arguments)
