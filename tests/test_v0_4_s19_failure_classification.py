"""D-0019: H stops, N records every field and cannot turn into PASS."""

import copy
import json

import numpy as np
import pytest

from benchmarks._s19_compare import (
    HardQualificationError,
    compare_field,
    compare_trace,
    final_verdict,
    require_execution,
)


def _trace():
    return [
        {"step": step, "runtime_identity": {"key": "same"},
         "configuration_identity": {"absolute_output": "/same/work"},
         "progress": {"step_count": step},
         "fields": {"Q": np.array([1.0, 2.0]), "u": np.array([0.0, 1.0])}}
        for step in range(3)
    ]


def test_n_records_each_step_field_and_remains_failed_after_control_and_profiles():
    baseline, candidate = _trace(), _trace()
    candidate[1]["fields"]["Q"][0] += np.finfo(np.float64).eps
    candidate[2]["fields"]["u"][0] = 1.0e-30
    report = compare_trace(baseline, candidate)
    assert report["classification"] == "N"
    assert report["continue_control_and_profiles"] is True
    assert len(report["fields"]) == 6
    assert report["fields"][-1]["max_relative_difference"] == "Infinity"
    assert report["fields"][2]["max_absolute_difference"] == np.finfo(np.float64).eps
    verdict = final_verdict([report, {"classification": "PASS", "name": "control"},
                             {"classification": "PASS", "name": "profiles"}], formal_h100=True)
    assert verdict["classification"] == "FAIL_N"
    assert verdict["qualification_passed"] is False
    json.dumps(verdict, allow_nan=False)


@pytest.mark.parametrize("key", ["runtime_identity", "configuration_identity", "progress"])
def test_trace_identity_differences_are_h_not_n(key):
    a, b = _trace(), _trace()
    b[1][key] = {"wrong": True}
    with pytest.raises(HardQualificationError, match="identity"):
        compare_trace(a, b)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_is_h(value):
    with pytest.raises(HardQualificationError, match="non-finite"):
        compare_field(np.array([1.0]), np.array([value]), field="p", step=1)


def test_signed_zero_is_not_misreported_as_bitwise_equal():
    result = compare_field(np.array([0.0]), np.array([-0.0]), field="p", step=0)
    assert result["classification"] == "N"
    assert result["max_absolute_difference"] == 0


@pytest.mark.parametrize("key,value", [
    ("finite", False), ("installed_wheel_only", False), ("cache_initially_empty", False),
    ("fallback_used", True), ("returncode", 1), ("execution_device", "cpu"),
    ("provider_commit", "wrong"), ("torch_version", "wrong"),
])
def test_execution_identity_environment_device_fallback_and_crash_stop(key, value):
    expected = {"execution_device": "cuda:0", "provider_commit": "bound", "torch_version": "2.5.1+cu121"}
    observed = dict(expected, finite=True, installed_wheel_only=True,
                    cache_initially_empty=True, fallback_used=False, returncode=0)
    require_execution(observed, expected)
    observed[key] = value
    with pytest.raises(HardQualificationError):
        require_execution(observed, expected)


def test_missing_fields_missing_steps_and_array_schema_are_h():
    a, b = _trace(), _trace()
    with pytest.raises(HardQualificationError):
        compare_trace(a, b[:-1])
    broken = copy.deepcopy(b)
    del broken[0]["fields"]["Q"]
    with pytest.raises(HardQualificationError):
        compare_trace(a, broken)
    with pytest.raises(HardQualificationError):
        compare_field(np.ones(1), np.ones(2), field="Q", step=0)


def test_cpu_success_never_claims_h100_qualification():
    report = final_verdict([compare_trace(_trace(), _trace())], formal_h100=False)
    assert report["classification"] == "PASS_CPU_REHEARSAL_ONLY"
    assert report["qualification_passed"] is False
    assert report["cpu_rehearsal_passed"] is True
