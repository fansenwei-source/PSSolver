"""Analysis-only contracts for the rc5 G-1 timing variability diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest

from benchmarks.analyze_rc5_g1_timing_variability import (
    CLASSIFICATION,
    _profiler_source_audit,
    analyze,
    main,
)
from benchmarks.plan_rc5_g1_focused_performance_h100 import build_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILER = ROOT / "benchmarks/profile_rc458_runtime_family.py"


def _historical_profiler(tmp_path: Path) -> Path:
    path = tmp_path / "historical_profiler.py"
    path.write_text(
        """def run_profile(config):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    timer.reset()
    samples = _timed_steps(case.step, config.profile_steps, device)
    return samples
""",
        encoding="utf-8",
    )
    return path


def _timing(samples: list[float]) -> dict[str, object]:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    return {
        "samples_seconds": samples,
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": statistics.median(samples),
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": stdev / mean,
    }


def _profiles(root: Path) -> None:
    root.mkdir()
    for entry in build_plan()["profile_matrix"]:
        samples = [1.0] * 60
        if entry["key"] == "baseline_plane_compiled_v2_plane_R128_trial1":
            samples[0] = 1.5
        if entry["key"] == "candidate_plane_compiled_v2_plane_R128_trial1":
            samples[0] = 1.5
            samples[17] = 2.0
        payload = {
            "config": {
                "family": entry["family"],
                "provider_role": entry["provider_role"],
                "trial": entry["trial"],
            },
            "timing": _timing(samples),
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(payload, allow_nan=False), encoding="utf-8"
        )


def test_source_audit_proves_empty_cache_precedes_first_timed_step():
    audit = _profiler_source_audit(PROFILER)
    calls = audit["calls"]
    assert calls["torch.cuda.empty_cache"] < calls["_timed_steps"]
    assert calls["torch.cuda.reset_peak_memory_stats"] < calls["_timed_steps"]
    assert audit["empty_cache_after_warmup_before_timing"] is True
    assert audit["untimed_rewarm_between_empty_cache_and_timing"] is True
    assert audit["first_post_empty_cache_step_is_timed"] is False
    assert calls["torch.cuda.empty_cache"] < audit["post_flush_rewarm_loop"]
    assert audit["post_flush_rewarm_loop"] < calls["timer.reset"]


def test_analysis_separates_first_sample_contamination_from_residual_variability(
    tmp_path,
):
    profile_root = tmp_path / "profiles"
    _profiles(profile_root)
    result = analyze(profile_root, _historical_profiler(tmp_path))
    assert result["classification"] == CLASSIFICATION
    assert result["analysis_only"] is True
    assert result["profile_count"] == 48
    assert result["full_window_cv_failure_count"] == 2
    assert result["diagnostic_excluding_first_sample_cv_failure_count"] == 1
    assert result["profiles_with_first_sample_overhead"] == 2
    assert result["first_sample_contamination_supported"] is True
    assert result["first_sample_contamination_is_complete_explanation"] is False
    assert result["residual_in_window_variability_supported"] is True
    assert result["residual_in_window_variability_cause_resolved"] is False
    assert result["candidate_first_ratio_mean_higher_family_count"] >= 0
    assert isinstance(
        result["candidate_first_ratio_mean_higher_in_all_families"], bool
    )
    assert result["execution_order_effect_causal"] is False
    plane = next(
        family
        for family in result["families"]
        if family["family"] == "plane_compiled_v2"
    )
    assert plane["baseline_first_full_mean_ratio_mean"] > 0.0
    assert plane["candidate_first_full_mean_ratio_mean"] > 0.0
    assert "diagnostic_excluding_first_sample_correlation_median" in plane
    assert result["raw_samples_modified"] is False
    assert result["outlier_removal_performed"] is False
    assert result["trial_replacement_performed"] is False
    assert result["thresholds_changed"] is False
    assert result["qualification_gate_change_authorized"] is False
    assert result["new_h100_submission_count"] == 0
    assert result["eligible_for_cumulative_reassembly"] is False
    assert result["g1_complete"] is False


def test_analysis_rejects_reported_statistic_mismatch(tmp_path):
    profile_root = tmp_path / "profiles"
    _profiles(profile_root)
    path = profile_root / "baseline_plane_compiled_v2_plane_R128_trial1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["timing"]["mean_timestep_seconds"] = 99.0
    path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match raw samples"):
        analyze(profile_root, _historical_profiler(tmp_path))


def test_analysis_rejects_incomplete_or_extra_file_set(tmp_path):
    profile_root = tmp_path / "profiles"
    _profiles(profile_root)
    (profile_root / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile file set differs"):
        analyze(profile_root, _historical_profiler(tmp_path))


def test_cli_writes_strict_atomic_json_without_reclassifying_v7(tmp_path):
    profile_root = tmp_path / "profiles"
    _profiles(profile_root)
    output = tmp_path / "analysis.json"
    arguments = [
        "--profile-root",
        str(profile_root),
        "--profiler-source",
        str(_historical_profiler(tmp_path)),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["classification"] == CLASSIFICATION
    assert payload["formal_v7_classification_preserved"] == (
        "FAIL_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_GATE"
    )
    with pytest.raises(FileExistsError):
        main(arguments)
    assert not list(tmp_path.rglob("*.tmp-*"))
