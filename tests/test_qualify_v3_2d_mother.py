import numpy as np

from scripts_plane.qualify_v3_2d_mother import (
    count_nematic_defects,
    effective_sample_size,
    time_series_statistics,
)


def test_v3_defect_counter_reports_zero_for_uniform_planar_nematic():
    q = np.zeros((16, 12, 5), dtype=np.float64)
    q[..., 0] = 1.0 / 3.0
    q[..., 3] = -1.0 / 6.0
    assert count_nematic_defects(q) == (0, 0, 0)


def test_v3_effective_sample_size_distinguishes_constant_and_correlated_data():
    assert effective_sample_size(np.ones(20)) == 20.0
    independent_like = np.array([1.0, -1.0] * 10)
    correlated = np.repeat(np.array([-1.0, 1.0]), 10)
    assert effective_sample_size(independent_like) == 20.0
    assert effective_sample_size(correlated) < 4.0


def test_v3_time_series_statistics_exposes_dimensionless_drift_gates():
    times = np.arange(20, dtype=float)
    stationary = np.array([-1.0, 1.0] * 10)
    drifting = np.arange(20, dtype=float)
    stationary_stats = time_series_statistics(times, stationary)
    drifting_stats = time_series_statistics(times, drifting)
    assert stationary_stats["full_window_drift_over_std"] < 0.3
    assert drifting_stats["full_window_drift_over_std"] > 2.0
