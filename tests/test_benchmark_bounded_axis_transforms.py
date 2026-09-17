"""Tests for the R2R-A baseline and R2R-B FFT candidate benchmark."""

from __future__ import annotations

import json

import pytest

from pssolver import QualifiedBoundedTransformPolicy
from benchmarks.benchmark_bounded_axis_transforms import (
    BoundedAxisBenchmarkConfig,
    run_benchmark,
)


def _small_config(**overrides):
    values = {
        "sizes": (3, 4),
        "kinds": ("dct", "dst"),
        "execution_modes": ("full", "truncated"),
        "value_types": ("real", "complex"),
        "algorithms": ("dense",),
        "retained_fraction": 0.5,
        "line_count": 3,
        "device": "cpu",
        "dtype": "float64",
        "warmup": 0,
        "repeats": 2,
        "seed": 19,
    }
    values.update(overrides)
    return BoundedAxisBenchmarkConfig(**values)


def test_dense_bounded_axis_sweep_is_finite_consistent_and_json_safe():
    result = run_benchmark(_small_config())

    assert result["schema_version"] == 1
    assert result["benchmark"] == "r2r_a_dense_bounded_axis_baseline"
    assert result["scope"] == {
        "candidate_algorithm_present": False,
        "qualified_policy_present": False,
        "production_default_changed": False,
        "basis_or_normalization_changed": False,
    }
    assert len(result["cases"]) == 16

    for case in result["cases"]:
        size = case["size"]
        retained_count = case["retained_count"]
        expected_matrix_bytes = size * size * 8
        assert case["correctness"]["all_finite"] is True
        assert (
            case["correctness"]["coefficient_recovery_relative_l2"]
            <= case["correctness"]["tolerance_relative_l2"]
        )
        if case["execution_mode"] == "full":
            assert (
                case["correctness"]["full_roundtrip_relative_l2"]
                <= case["correctness"]["tolerance_relative_l2"]
            )
            assert retained_count == size
        else:
            assert case["correctness"]["full_roundtrip_relative_l2"] is None
            assert retained_count == max(1, size // 2)
        assert case["algorithm"] == "dense"
        assert case["dense_work_model"]["cached_matrix_bytes_actual"] == (
            expected_matrix_bytes
        )
        assert case["dense_work_model"]["cached_fft_r2r_bytes_actual"] == 0
        assert case["dense_work_model"]["one_full_real_matrix_bytes"] == (
            expected_matrix_bytes
        )
        assert (
            case["dense_work_model"]["plane_dct_plus_dst_full_cache_bytes"]
            == 2 * expected_matrix_bytes
        )
        for direction in ("forward", "inverse"):
            timing = case["timing"][direction]
            assert timing["samples"] == 2
            assert timing["mean_seconds"] >= 0.0
            assert timing["minimum_seconds"] >= 0.0

    json.dumps(result)


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ({"sizes": ()}, "sizes"),
        ({"kinds": ("fft",)}, "kinds"),
        ({"execution_modes": ("unknown",)}, "execution_modes"),
        ({"value_types": ("integer",)}, "value_types"),
        ({"algorithms": ("unknown",)}, "algorithms"),
        ({"algorithms": ("auto",)}, "requires a bounded-transform policy"),
        ({"retained_fraction": 0.0}, "retained_fraction"),
        ({"line_count": 0}, "line_count"),
        ({"warmup": -1}, "warmup"),
        ({"repeats": 0}, "repeats"),
    ),
)
def test_dense_bounded_axis_sweep_rejects_invalid_contracts(override, match):
    with pytest.raises(ValueError, match=match):
        run_benchmark(_small_config(**override))



def test_fft_candidate_sweep_compares_against_dense_reference():
    result = run_benchmark(
        _small_config(
            sizes=(5,),
            algorithms=("dense", "fft"),
        )
    )

    assert result["benchmark"] == "r2r_b_dense_vs_fft_bounded_axis"
    assert result["scope"]["candidate_algorithm_present"] is True
    assert result["scope"]["qualified_policy_present"] is False
    assert len(result["cases"]) == 16
    fft_cases = [case for case in result["cases"] if case["algorithm"] == "fft"]
    assert len(fft_cases) == 8
    for case in fft_cases:
        assert (
            case["correctness"]["dense_reference_forward_relative_l2"]
            <= case["correctness"]["tolerance_relative_l2"]
        )
        assert (
            case["correctness"]["dense_reference_inverse_relative_l2"]
            <= case["correctness"]["tolerance_relative_l2"]
        )
        assert case["dense_work_model"]["cached_matrix_bytes_actual"] == 0
        assert case["dense_work_model"]["cached_fft_r2r_bytes_actual"] > 0


def test_auto_policy_sweep_uses_dense_for_unqualified_contexts(tmp_path):
    policy = QualifiedBoundedTransformPolicy(
        name="empty_safe_fallback",
        cells=(),
    )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"policy": policy.to_metadata()}))

    result = run_benchmark(
        _small_config(
            sizes=(5,),
            kinds=("dct",),
            execution_modes=("full",),
            value_types=("real",),
            algorithms=("dense", "auto"),
            policy_path=str(path),
        )
    )

    assert result["benchmark"] == "r2r_c_qualified_policy_bounded_axis"
    assert result["scope"]["qualified_policy_present"] is True
    automatic = next(
        case for case in result["cases"] if case["algorithm"] == "auto"
    )
    selection = automatic["bounded_transform_selection"]
    assert selection["effective_algorithms"] == ["dense"]
    assert selection["observed_decisions"]
    assert all(
        entry["selection"]["reason"]
        == "no_matching_qualification_cell"
        for entry in selection["observed_decisions"]
    )
