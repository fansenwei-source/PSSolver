"""Tests for the versioned Periodic Hermitian-repair profiler."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from benchmarks.analyze_periodic_hermitian_profiles import (
    FROZEN_INPUT_SHA256,
    HermitianProfileEvidenceError,
    PASS_CLASSIFICATION,
    analyze,
    validate_capability_audit,
)
from benchmarks.profile_periodic_functional import (
    PROFILE_ROLES,
    PeriodicFunctionalProfileConfig,
)
from benchmarks.profile_periodic_hermitian_qualification import (
    EAGER_INAPPLICABLE_REASON,
    run_qualification_profile,
)


BASELINE = "a" * 40
CANDIDATE = "b" * 40


def _initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    values = np.random.default_rng(20260930).normal(
        scale=0.01,
        size=(6, 6, 4, 5),
    ).astype(np.float64)
    np.save(path, values, allow_pickle=False)
    return path


def _config(path, role):
    return PeriodicFunctionalProfileConfig(
        role=role,
        trial=1,
        shape=(6, 6, 4),
        lengths=(6.0, 6.0, 4.0),
        initial_q_path=str(path),
        device="cpu",
        warmup_steps=0,
        profile_steps=1,
    )


@pytest.fixture(scope="module")
def role_reports(tmp_path_factory):
    root = tmp_path_factory.mktemp("hermitian-profiler")
    path = _initial_q(root)
    return {
        role: run_qualification_profile(
            _config(path, role),
            variant="candidate",
        )
        for role in PROFILE_ROLES
    }


def test_profiler_records_actual_fallback_transform_and_compile_capabilities(
    role_reports,
):
    for role, report in role_reports.items():
        audit = report["capability_audit"]
        assert audit["role"] == role
        assert audit["timing_contaminated"] is False
        assert audit["runtime_selection"]["fallback_used"] is False
        assert audit["transform_dispatch"]["forward_calls"] > 0
        assert audit["transform_dispatch"]["inverse_calls"] > 0
        assert audit["result"]["finite"] is True
        assert len(audit["result"]["sha256"]) == 64
        assert audit["compilation"]["pointwise_execution"] == {
            "requested": "eager",
            "effective": "eager",
        }
        for name in ("graph_breaks", "compile_fallback"):
            assert audit["compilation"][name] == {
                "applicable": False,
                "value": None,
                "reason": EAGER_INAPPLICABLE_REASON,
            }


def test_capability_validator_rejects_invented_eager_values(role_reports):
    audit = copy.deepcopy(role_reports["functional_forward"]["capability_audit"])
    audit["compilation"]["graph_breaks"]["value"] = 0
    with pytest.raises(HermitianProfileEvidenceError, match="must not invent"):
        validate_capability_audit(
            audit,
            role="functional_forward",
            candidate=True,
        )


def test_capability_validator_rejects_fallback_and_missing_projection(role_reports):
    audit = copy.deepcopy(role_reports["production_forward"]["capability_audit"])
    audit["runtime_selection"]["fallback_used"] = True
    with pytest.raises(HermitianProfileEvidenceError, match="fallback"):
        validate_capability_audit(
            audit,
            role="production_forward",
            candidate=True,
        )

    audit = copy.deepcopy(role_reports["production_forward"]["capability_audit"])
    del audit["runtime_selection"]["metadata"]["hermitian_state_projection"]
    with pytest.raises(HermitianProfileEvidenceError, match="projection identity"):
        validate_capability_audit(
            audit,
            role="production_forward",
            candidate=True,
        )


def _matrix(role_reports):
    reports = []
    for variant, commit, factor in (
        ("baseline", BASELINE, 1.0),
        ("candidate", CANDIDATE, 1.01),
    ):
        for shape in ((128, 128, 32), (320, 320, 80)):
            for role in PROFILE_ROLES:
                for trial in range(1, 4):
                    report = copy.deepcopy(role_reports[role])
                    report["variant"] = variant
                    base = report["base_profile"]
                    base["config"].update(
                        {
                            "shape": list(shape),
                            "lengths": [100.0, 100.0, 20.0],
                            "trial": trial,
                            "device": "cuda:0",
                            "initial_q_sha256": FROZEN_INPUT_SHA256[
                                "R128" if shape[0] == 128 else "R320"
                            ],
                            "warmup_steps": 5,
                            "profile_steps": 20,
                        }
                    )
                    base["environment"].update(
                        {
                            "cuda_available": True,
                            "device": "cuda:0",
                            "device_name": "NVIDIA H100 PCIe",
                            "tf32_matmul": False,
                            "tf32_cudnn": False,
                        }
                    )
                    base["environment"]["git"] = {
                        "head": commit,
                        "status_porcelain": "",
                    }
                    samples = [factor] * 20
                    base["timing"].update(
                        {
                            "samples_seconds": samples,
                            "mean_seconds": factor,
                            "median_seconds": factor,
                        }
                    )
                    base["memory"] = {
                        "peak_allocated_bytes": int(1000 * factor),
                        "peak_active_bytes": int(1000 * factor),
                        "peak_reserved_bytes": int(2000 * factor),
                        "device_total_bytes": 10000,
                    }
                    reports.append(report)
    return reports


def test_analyzer_accepts_non_regression_matrix(role_reports):
    result = analyze(
        _matrix(role_reports),
        baseline_commit=BASELINE,
        candidate_commit=CANDIDATE,
    )
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["passed"] is True
    assert result["profile_count"] == 36


def test_analyzer_rejects_performance_and_transform_regressions(role_reports):
    reports = _matrix(role_reports)
    target = next(
        report
        for report in reports
        if report["variant"] == "candidate"
        and report["base_profile"]["config"]["shape"] == [128, 128, 32]
        and report["base_profile"]["config"]["role"] == "functional_forward"
    )
    target["base_profile"]["timing"].update(
        {
            "samples_seconds": [1.2] * 20,
            "mean_seconds": 1.2,
            "median_seconds": 1.2,
        }
    )
    with pytest.raises(HermitianProfileEvidenceError, match="mean regression"):
        analyze(
            reports,
            baseline_commit=BASELINE,
            candidate_commit=CANDIDATE,
        )

    reports = _matrix(role_reports)
    reports[-1]["capability_audit"]["transform_dispatch"]["inverse_calls"] += 1
    with pytest.raises(HermitianProfileEvidenceError, match="transform calls changed"):
        analyze(
            reports,
            baseline_commit=BASELINE,
            candidate_commit=CANDIDATE,
        )
