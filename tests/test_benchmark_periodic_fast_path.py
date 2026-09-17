"""Tests for the periodic fast-path A/B/C/D benchmark."""

import pytest

from benchmarks.benchmark_periodic_fast_path import (
    BenchmarkConfig,
    run_benchmark,
)


def test_small_cpu_benchmark_runs_all_four_variants():
    result = run_benchmark(
        BenchmarkConfig(
            shape=(8, 6),
            lengths=(4.0, 3.0),
            field_count=3,
            batch_size=1,
            device="cpu",
            dtype="float64",
            warmup_steps=1,
            profile_steps=2,
            trials=1,
            seed=17,
        )
    )

    assert set(result["aggregate"]) == {
        "A_advanced_axiswise",
        "B_contiguous_axiswise",
        "C_advanced_multidim",
        "D_contiguous_multidim",
    }
    assert result["correctness"]["maximum_relative_l2"] <= 1.0e-12
    assert all(len(records) == 1 for records in result["records"].values())


def test_benchmark_rejects_invalid_counts():
    with pytest.raises(ValueError, match="field_count"):
        run_benchmark(
            BenchmarkConfig(
                shape=(8, 6),
                lengths=(4.0, 3.0),
                field_count=0,
                batch_size=1,
                device="cpu",
                dtype="float64",
                warmup_steps=0,
                profile_steps=1,
                trials=1,
                seed=17,
            )
        )
