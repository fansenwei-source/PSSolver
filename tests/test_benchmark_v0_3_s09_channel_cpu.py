from __future__ import annotations

import pytest

from benchmarks.benchmark_v0_3_s09_channel_cpu import (
    FAMILIES,
    MEMORY_LIMIT,
    MODES,
    TIME_CENTER_LIMIT,
    TIME_PAIRED_LIMIT,
    analyze,
)


def _cell(family, mode, role, trial, *, time_value, memory_value):
    return {
        "config": {"family": family, "mode": mode, "role": role, "trial": trial},
        "timing": {"mean_step_seconds": time_value},
        "memory": {"peak_rss_during_window_bytes": memory_value},
    }


def _matrix(*, time_ratio=1.0, memory_ratio=1.0):
    cells = []
    for family in FAMILIES:
        for mode in MODES:
            for trial in range(1, 4):
                cells.extend(
                    (
                        _cell(family, mode, "baseline", trial, time_value=1.0, memory_value=1000),
                        _cell(family, mode, "candidate", trial, time_value=time_ratio, memory_value=1000 * memory_ratio),
                    )
                )
    return cells


def test_s09_cpu_preflight_accepts_only_the_frozen_limits():
    report = analyze(_matrix(time_ratio=TIME_CENTER_LIMIT, memory_ratio=MEMORY_LIMIT))
    assert report["passed"] is True
    assert report["eligible_for_h100_preparation"] is True
    assert report["thresholds"] == {
        "median_time_ratio_max": 1.03,
        "paired_time_ratio_max": 1.05,
        "median_peak_rss_ratio_max": 1.03,
        "paired_peak_rss_ratio_max": 1.03,
        "relaxation_allowed": False,
    }


@pytest.mark.parametrize(
    ("time_ratio", "memory_ratio"),
    (
        (TIME_PAIRED_LIMIT + 1.0e-6, 1.0),
        (TIME_CENTER_LIMIT + 1.0e-6, 1.0),
        (1.0, MEMORY_LIMIT + 1.0e-6),
    ),
)
def test_s09_cpu_preflight_fails_closed_without_threshold_relaxation(
    time_ratio,
    memory_ratio,
):
    report = analyze(_matrix(time_ratio=time_ratio, memory_ratio=memory_ratio))
    assert report["passed"] is False
    assert report["eligible_for_h100_preparation"] is False


def test_s09_cpu_preflight_requires_every_paired_role():
    cells = _matrix()
    cells.pop()
    with pytest.raises(ValueError, match="missing paired role"):
        analyze(cells)
