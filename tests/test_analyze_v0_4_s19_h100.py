"""S-19 synthetic records use fmean; frozen statistics are not weakened."""
import copy
import inspect
import json
import statistics
import types

import pytest

from benchmarks import analyze_v0_3_s09_h100 as frozen
from benchmarks import analyze_v0_4_s19_h100 as current
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import build_plan
from tests import test_analyze_v0_3_s09_h100 as fixture


def _profiles(path, scale=1.01):
    namespace = dict(fixture._write_profiles.__globals__, build_plan=build_plan)
    function = types.FunctionType(fixture._write_profiles.__code__, namespace)
    function(path, scale)


def _auxiliary():
    result = fixture._auxiliary()
    plan = build_plan()
    for row in result["stages"]:
        row.update(provider_candidate_commit=current.CANDIDATE_COMMIT,
                   candidate_wheel_sha256=current.CANDIDATE_WHEEL_SHA256,
                   execution_device="cuda", finite=True, fallback_used=False, classification="PASS")
    result["stages"][0].update(baseline_wheel_sha256=current.BASELINE_WHEEL_SHA256,
                               consumer_wheel_sha256=current.CONSUMER_WHEEL_SHA256)
    result["stages"][1].update(node_ids=plan["cuda_only_tests"], passed_count=7,
                               nonpassing_count=0, junit_written=True)
    result["stages"][2]["formal_h100_contract"] = True
    result["stages"][4]["cross_device_cell_count"] = 8
    result["stages"][5].update(same_absolute_work_path=True, independent_empty_caches=True,
                               baseline_self_comparison=True, public_dispatch_completed=True,
                               four_pcg_paths=True, all_fields_and_steps_recorded=True,
                               trajectory_cases=40, trajectory_steps=20)
    result["stages"][6].update(consumer_source_commit=current.CONSUMER_RUNTIME_COMMIT,
                               consumer_wheel_sha256=current.CONSUMER_WHEEL_SHA256,
                               passed_count=303, skip_node_ids=[plan["consumer_h100_gate"]["allowed_skip"]])
    return result


def test_same_frozen_performance_function_same_groups_on_identical_input(tmp_path):
    profiles = tmp_path / "profiles"
    _profiles(profiles)
    auxiliary = _auxiliary()
    namespace = dict(frozen.analyze.__globals__, build_plan=build_plan,
                     _validate_auxiliary=current.validate_auxiliary)
    reference = types.FunctionType(frozen.analyze.__code__, namespace)(profiles, auxiliary)
    actual = current.analyze(profiles, auxiliary)
    assert actual["groups"] == reference["groups"]
    assert actual["qualification_complete"] is True
    assert actual["profile_count"] == 192
    assert actual["recorded_performance_failures"] == []
    assert "statistics.fmean" in inspect.getsource(fixture._write_profiles)


def test_n_stays_failed_after_passing_performance_and_control(tmp_path):
    profiles = tmp_path / "profiles"
    _profiles(profiles)
    auxiliary = _auxiliary()
    auxiliary["stages"][5]["classification"] = "N"
    result = current.analyze(profiles, auxiliary)
    assert result["numerical_failed"] is True
    assert not result["qualification_complete"]
    assert result["classification"] == "FAIL_S19_H100"
    assert not result["eligible_for_release_integration_review"]


def test_d0013_is_not_an_s19_exception_and_every_failure_is_retained(tmp_path):
    profiles = tmp_path / "profiles"
    _profiles(profiles)
    for family in ("channel_legacy", "channel_compiled_v2"):
        for trial in range(1, 7):
            path = profiles / f"candidate_{family}_channel_R512x40x40_trial{trial}.json"
            value = json.loads(path.read_text())
            value["memory"]["peak_reserved_bytes"] = 2061539
            path.write_text(json.dumps(value))
    result = current.analyze(profiles, _auxiliary())
    failed = [g for g in result["groups"] if not g["passed"]]
    assert len(failed) == len(result["recorded_performance_failures"]) == 2
    assert {g["family"] for g in failed} == {"channel_legacy", "channel_compiled_v2"}
    assert all({key for key, passed in g["checks"].items() if not passed} == {"peak_reserved"} for g in failed)
    assert not result["qualification_complete"]
    assert result["d0013_exception_applies"] is False


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "cpu", "nonfinite", "fallback", "identity", "N_control"])
def test_auxiliary_bad_execution_cannot_be_relabelled_n(mutation):
    value = _auxiliary()
    if mutation == "missing":
        value["stages"].pop()
    elif mutation == "duplicate":
        value["stages"].append(copy.deepcopy(value["stages"][0]))
    elif mutation == "reordered":
        value["stages"].reverse()
    else:
        key, changed = {"cpu": ("execution_device", "cpu"), "nonfinite": ("finite", False),
                        "fallback": ("fallback_used", True), "identity": ("candidate_wheel_sha256", "bad"),
                        "N_control": ("classification", "N")}[mutation]
        value["stages"][-1][key] = changed
    with pytest.raises(HardQualificationError):
        current.validate_auxiliary(value, build_plan())


def test_missing_profile_and_false_mean_fail_closed_not_recorded_performance(tmp_path):
    profiles = tmp_path / "profiles"
    _profiles(profiles)
    path = profiles / "baseline_plane_legacy_production_plane_R128_trial1.json"
    value = json.loads(path.read_text())
    value["timing"]["mean_timestep_seconds"] = statistics.fmean(value["timing"]["samples_seconds"]) + 1e-9
    path.write_text(json.dumps(value))
    with pytest.raises(HardQualificationError, match="raw-sample mean"):
        current.analyze(profiles, _auxiliary())
    path.unlink()
    with pytest.raises(FileNotFoundError):
        current.analyze(profiles, _auxiliary())


def test_auxiliary_accepts_only_exact_d0022_nonfinite_records_and_keeps_n():
    value = _auxiliary()
    row = value['stages'][5]
    row.update(finite=False, classification='N', nonfinite_records=[
        ['fixed_float32_roundoff_floor', 'oracle', 'gradient'],
        ['fixed_float32_floor_crossing', 'oracle', 'gradient']])
    current.validate_auxiliary(value, build_plan())
    for bad in (['fixed_float32_roundoff_floor', 'oracle', 'result'],
                ['fixed_float32_floor_crossing', 'implicit', 'gradient']):
        changed = copy.deepcopy(value)
        changed['stages'][5]['nonfinite_records'].append(bad)
        with pytest.raises(HardQualificationError, match='outside D-0022'):
            current.validate_auxiliary(changed, build_plan())
