"""Fail-closed tests for the frozen RC6 single-H100 plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.plan_rc6_single_h100_qualification import (
    CLASSIFICATION,
    CONSUMER_SOURCE,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
    SCHEMA,
    build_plan,
    main,
)


def test_rc6_plan_binds_provider_and_consumer_sources():
    plan = build_plan()
    assert plan["schema"] == SCHEMA
    assert plan["classification"] == CLASSIFICATION
    assert plan["provider"]["baseline_commit"] == PROVIDER_BASELINE
    assert plan["provider"]["candidate_commit"] == PROVIDER_CANDIDATE
    assert plan["consumer"]["qualified_source_commit"] == CONSUMER_SOURCE
    assert plan["consumer"]["provider_identity_migration_required"] is False


def test_rc6_plan_has_exactly_seven_cuda_only_tests():
    tests = build_plan()["cuda_only_tests"]
    assert len(tests) == len(set(tests)) == 7
    assert all("::test_" in node for node in tests)


def test_rc6_profile_matrix_is_complete_unique_and_order_balanced():
    plan = build_plan()
    profiles = plan["profile_matrix"]
    assert len(profiles) == plan["profile_contract"]["profile_count"] == 192
    assert [row["sequence"] for row in profiles] == list(range(1, 193))
    assert len({row["key"] for row in profiles}) == 192
    cells = {(row["family"], row["grid"]) for row in profiles}
    assert len(cells) == plan["profile_contract"]["cell_count"] == 16
    for family, grid in cells:
        rows = [
            row
            for row in profiles
            if row["family"] == family and row["grid"] == grid
        ]
        assert len(rows) == 12
        for trial in range(1, 7):
            pair = [row for row in rows if row["trial"] == trial]
            assert [row["order_position"] for row in pair] == [1, 2]
            expected = ["baseline", "candidate"] if trial % 2 else ["candidate", "baseline"]
            assert [row["provider_role"] for row in pair] == expected


def test_rc6_profile_matrix_keeps_measurement_and_transform_contracts_exact():
    plan = build_plan()
    for row in plan["profile_matrix"]:
        assert row["warmup_steps"] == 10
        assert row["timed_steps"] == row["samples_per_profile"] == 60
        assert row["fixed_block_size"] * row["fixed_block_count"] == 60
        assert row["expected_forward_raw_calls"] == (
            row["steady_forward_transforms_per_step"] * 60
            + row["scheduled_refreshes_in_timed_window"]
        )
        assert row["expected_inverse_raw_calls"] == (
            row["steady_inverse_transforms_per_step"] * 60
        )
        assert row["device"] == "cuda"
        assert row["isolated_empty_triton_cache"] is True
        assert row["isolated_empty_torchinductor_cache"] is True


def test_rc6_plan_preserves_adopted_balanced_crossover_thresholds():
    plan = build_plan()
    contract = plan["profile_contract"]
    assert contract["baseline_then_candidate_trials"] == [1, 3, 5]
    assert contract["candidate_then_baseline_trials"] == [2, 4, 6]
    assert contract["all_samples_retained"] is True
    assert contract["outlier_removal_allowed"] is False
    assert contract["sample_deletion_allowed"] is False
    assert contract["trial_replacement_allowed"] is False
    thresholds = plan["thresholds"]
    assert thresholds["aggregate_all_sample_mean_ratio_max"] == 1.03
    assert thresholds["median_of_six_trial_ratios_max"] == 1.03
    assert thresholds["baseline_then_candidate_stratum_median_ratio_max"] == 1.05
    assert thresholds["candidate_then_baseline_stratum_median_ratio_max"] == 1.05
    assert thresholds["minimum_trials_at_or_below_individual_reference"] == 4


def test_rc6_plan_is_fail_closed_and_does_not_claim_release_authority():
    plan = build_plan()
    assert plan["submission"] == {
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_without_new_authorization": False,
        "fail_fast": True,
        "complete_written_only_after_all_gates_and_checksums": True,
    }
    success = plan["success"]
    assert success["independent_post_release_verification_required"] is True
    assert success["eligible_for_automatic_merge"] is False
    assert success["eligible_for_default_promotion"] is False
    assert success["production_default_changed"] is False


def test_rc6_plan_atomic_writer_refuses_overwrite_and_round_trips(tmp_path: Path):
    output = tmp_path / "plan.json"
    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--output", str(output)])
