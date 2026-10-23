from __future__ import annotations

import pytest

from benchmarks.benchmark_v0_3_s09_channel_cpu_frozen import (
    FIXED_BLOCK_COUNT,
    FIXED_BLOCK_SIZE,
    FAMILIES,
    MEASURED_STEPS,
    MEMORY_LIMIT,
    MODES,
    TIME_CENTER_LIMIT,
    TIME_PAIRED_LIMIT,
    TRIALS,
    analyze,
    decorate_cell,
    fixed_block_means,
)


def _cell(family, mode, role, trial, *, time_value, memory_value):
    return {
        "config": {"family": family, "mode": mode, "role": role, "trial": trial},
        "timing": {
            "profile_center_step_seconds": time_value,
            "samples_seconds": [time_value] * MEASURED_STEPS,
        },
        "memory": {"peak_rss_during_window_bytes": memory_value},
    }


def _matrix(*, time_ratio=1.0, memory_ratio=1.0):
    cells = []
    for family in FAMILIES:
        for mode in MODES:
            for trial in range(1, TRIALS + 1):
                cells.extend(
                    (
                        _cell(
                            family,
                            mode,
                            "baseline",
                            trial,
                            time_value=1.0,
                            memory_value=1000,
                        ),
                        _cell(
                            family,
                            mode,
                            "candidate",
                            trial,
                            time_value=time_ratio,
                            memory_value=1000 * memory_ratio,
                        ),
                    )
                )
    return cells


def test_fixed_estimator_retains_sixty_samples_in_six_blocks():
    samples = [float(value) for value in range(MEASURED_STEPS)]
    means = fixed_block_means(samples)
    assert len(means) == FIXED_BLOCK_COUNT
    assert means == [4.5, 14.5, 24.5, 34.5, 44.5, 54.5]
    report = decorate_cell(
        {
            "schema": "old",
            "timing": {"samples_seconds": samples},
        }
    )
    assert report["timing"]["fixed_block_size"] == FIXED_BLOCK_SIZE
    assert report["timing"]["profile_center_step_seconds"] == 29.5
    assert report["timing"]["all_samples_retained"] is True


def test_fixed_estimator_rejects_missing_or_extra_samples():
    with pytest.raises(ValueError, match="exactly 60"):
        fixed_block_means([1.0] * 59)
    with pytest.raises(ValueError, match="exactly 60"):
        fixed_block_means([1.0] * 61)


def test_frozen_matrix_accepts_only_the_original_thresholds():
    report = analyze(_matrix(time_ratio=TIME_CENTER_LIMIT, memory_ratio=MEMORY_LIMIT))
    assert report["passed"] is True
    assert report["eligible_for_h100_preparation"] is True
    assert report["estimator"] == {
        "trials_per_cell": 6,
        "samples_per_profile": 60,
        "fixed_block_count": 6,
        "fixed_block_size": 10,
        "profile_center": "median_of_six_fixed_block_means",
        "alternating_pair_order": True,
        "all_samples_retained": True,
        "trial_replacement_allowed": False,
    }


@pytest.mark.parametrize(
    ("time_ratio", "memory_ratio"),
    (
        (TIME_PAIRED_LIMIT + 1.0e-6, 1.0),
        (TIME_CENTER_LIMIT + 1.0e-6, 1.0),
        (1.0, MEMORY_LIMIT + 1.0e-6),
    ),
)
def test_frozen_matrix_fails_closed_without_relaxation(time_ratio, memory_ratio):
    assert analyze(_matrix(time_ratio=time_ratio, memory_ratio=memory_ratio))["passed"] is False


def test_frozen_matrix_requires_all_six_trials():
    cells = _matrix()
    cells = [cell for cell in cells if cell["config"]["trial"] != TRIALS]
    with pytest.raises(ValueError, match="expected trials 1..6"):
        analyze(cells)
