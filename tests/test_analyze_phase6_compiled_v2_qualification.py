"""Tests for the frozen Phase 6 H100 evidence aggregator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts_plane.analyze_phase6_compiled_v2_qualification import (
    EXPECTED_COMPARISON_ROLES,
    REQUIRED_AUXILIARY_GATES,
    analyze_phase6_performance,
    analyze_phase6_qualification,
    main,
)


COMMIT = "a" * 40
QUALIFICATION_COMMIT = "b" * 40
SOURCE_MANIFEST_SHA256 = "c" * 64


def _write_json(path: Path, value: object) -> Path:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _profile_value(
    runtime: str,
    shape: tuple[int, int, int],
    trial: int,
    *,
    timestep: float,
) -> dict[str, object]:
    grid = shape[0]
    return {
        "schema_version": 1,
        "config": {
            "runtime_path": runtime,
            "trial": trial,
            "shape": list(shape),
            "lengths": [100.0, 100.0, 20.0],
            "device": "cuda",
            "dtype": "float64",
            "dt": 0.005,
            "activity_number": 18.0,
            "dealias_rule": "cubic_half",
            "projected_transform_execution": "truncated",
            "transform_execution_order": "real_first",
            "spectral_storage": "hermitian_half",
            "molecular_field_linear_space": "spectral",
            "stress_divergence_sum_space": "spectral",
            "pointwise_execution": "compile",
            "reuse_q_gradients": True,
            "spectral_refresh_interval": None,
            "warmup_steps": 10,
            "profile_steps": 50,
            "seed": 24,
            "initial_q_path": None,
        },
        "runtime_identity": {
            "requested": runtime,
            "effective": runtime,
            "fallback_used": False,
        },
        "environment": {
            "git": {"head": COMMIT, "dirty": False},
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "transform_calls": {
            "forward_per_step": 7,
            "inverse_per_step": 32,
        },
        "throughput": {"mean_timestep_seconds": timestep},
        "memory": {
            "peak_allocated_bytes": grid * 1000,
            "peak_reserved_bytes": grid * 2000,
        },
        "pointwise_compile": {
            "dynamo_during_profile": {"graph_breaks": 0},
        },
        "initial_q_sha256": f"initial-{grid}",
        "final_state_sha256": f"final-{grid}-{trial}",
        "completed_steps": 60,
        "finite": True,
    }


def _comparison_value(role: str) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "comparison_role": role,
        "classification": "PASS",
        "require_byte_identity": True,
        "byte_identity_gate": True,
        "array_count": 3,
        "maximum_gate_relative_l2": 0.0,
    }
    if role.endswith("cross_100"):
        value.update(
            {
                "require_initial_q_identity": True,
                "initial_q_identity_gate": True,
            }
        )
    return value


def _evidence(tmp_path: Path):
    profiles = []
    for shape in ((128, 128, 32), (320, 320, 80)):
        for runtime in ("legacy_production", "compiled_v2"):
            for trial in (1, 2, 3):
                base = 0.01 if shape[0] == 128 else 0.05
                factor = 0.98 if runtime == "compiled_v2" else 1.0
                value = _profile_value(
                    runtime,
                    shape,
                    trial,
                    timestep=base * factor * (1.0 + (trial - 2) * 0.002),
                )
                profiles.append(
                    _write_json(
                        tmp_path / f"profile_{shape[0]}_{runtime}_{trial}.json",
                        value,
                    )
                )
    comparisons = [
        _write_json(tmp_path / f"{role}.json", _comparison_value(role))
        for role in sorted(EXPECTED_COMPARISON_ROLES)
    ]
    auxiliary = _write_json(
        tmp_path / "auxiliary.json",
        {
            "schema_version": 1,
            "classification": "PASS_PHASE6_AUXILIARY_GATES",
            "expected_commit": COMMIT,
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "slurm": {"state": "COMPLETED", "exit_code": "0:0"},
            "environment": {"gpu_name": "NVIDIA H100 PCIe", "tf32": False},
            "gates": {name: True for name in sorted(REQUIRED_AUXILIARY_GATES)},
            "production_default_changed": False,
        },
    )
    return profiles, comparisons, auxiliary


def test_accelerated_evidence_passes_and_only_authorizes_long_run(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)

    report = analyze_phase6_qualification(
        profiles,
        comparisons,
        auxiliary,
        expected_commit=COMMIT,
    )

    assert report["classification"] == "PASS_PHASE6_H100_CANDIDATE_ACCELERATED"
    assert report["performance_class"] == "accelerated"
    assert report["all_gates_passed"] is True
    assert report["eligible_for_long_run_stage"] is True
    assert report["eligible_for_default_promotion"] is False
    assert report["production_default_changed"] is False
    assert report["profile_count"] == 12
    assert report["comparison_count"] == 6


def test_hpcc_noise_scale_results_are_performance_equivalent(tmp_path):
    profiles, _, _ = _evidence(tmp_path)
    legacy = {
        128: (0.007015893, 0.007020438, 0.007045301),
        320: (0.045313743, 0.045340887, 0.045239500),
    }
    compiled = {
        128: (0.007021055, 0.007072679, 0.006948143),
        320: (0.045323951, 0.045331491, 0.045252297),
    }
    for path in profiles:
        value = json.loads(path.read_text(encoding="utf-8"))
        config = value["config"]
        source = legacy if config["runtime_path"] == "legacy_production" else compiled
        value["throughput"]["mean_timestep_seconds"] = source[
            config["shape"][0]
        ][config["trial"] - 1]
        _write_json(path, value)

    report = analyze_phase6_performance(profiles, expected_commit=COMMIT)

    assert report["classification"] == (
        "PASS_PHASE6_H100_PERFORMANCE_EQUIVALENT_ONLY"
    )
    assert report["performance_class"] == "performance_equivalent"
    assert report["performance_gates_passed"] is True
    assert report["eligible_for_scientific_gate_recovery"] is True
    assert report["eligible_for_long_run_stage"] is False
    assert report["grids"]["R128"]["candidate_faster_count"] == 1
    assert report["grids"]["R320"]["candidate_faster_count"] == 1
    assert report["grids"]["R128"][
        "candidate_faster_count_is_informational"
    ] is True
    assert report["grids"]["R320"]["gates"][
        "every_paired_trial_non_regression"
    ] is True


def test_performance_regression_fails_closed(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(profiles[3].read_text(encoding="utf-8"))
    value["throughput"]["mean_timestep_seconds"] = 0.02
    _write_json(profiles[3], value)

    report = analyze_phase6_qualification(
        profiles,
        comparisons,
        auxiliary,
        expected_commit=COMMIT,
    )

    assert report["classification"] == (
        "FAIL_PHASE6_H100_CANDIDATE_PERFORMANCE_REGRESSION"
    )
    assert report["performance_class"] == "regressed"
    assert report["eligible_for_long_run_stage"] is False


def test_authorized_recovery_binds_separate_profile_and_qualification_commits(
    tmp_path,
):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(auxiliary.read_text(encoding="utf-8"))
    value.update(
        {
            "expected_commit": QUALIFICATION_COMMIT,
            "formal_h100_submission_count": 2,
            "authorized_recovery": True,
            "source_profile_evidence_reused": True,
            "source_profile_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_profile_job": {
                "job_id": "10837343",
                "state": "CANCELLED",
                "profilers_complete": True,
            },
        }
    )
    _write_json(auxiliary, value)

    report = analyze_phase6_qualification(
        profiles,
        comparisons,
        auxiliary,
        expected_commit=COMMIT,
        expected_qualification_commit=QUALIFICATION_COMMIT,
        expected_h100_submission_count=2,
        expected_source_manifest_sha256=SOURCE_MANIFEST_SHA256,
        expected_source_profile_job_id="10837343",
    )

    assert report["expected_profile_commit"] == COMMIT
    assert report["expected_qualification_commit"] == QUALIFICATION_COMMIT
    assert report["expected_h100_submission_count"] == 2
    assert report["auxiliary_gates"]["source_profile_job"]["job_id"] == (
        "10837343"
    )
    assert report["eligible_for_long_run_stage"] is True


def test_recovery_requires_explicit_provenance(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(auxiliary.read_text(encoding="utf-8"))
    value["expected_commit"] = QUALIFICATION_COMMIT
    value["formal_h100_submission_count"] = 2
    _write_json(auxiliary, value)

    with pytest.raises(ValueError, match="authorized-recovery provenance"):
        analyze_phase6_qualification(
            profiles,
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
            expected_qualification_commit=QUALIFICATION_COMMIT,
            expected_h100_submission_count=2,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            expected_source_profile_job_id="10837343",
        )


def test_recovery_rejects_wrong_source_profiler_job(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(auxiliary.read_text(encoding="utf-8"))
    value.update(
        {
            "expected_commit": QUALIFICATION_COMMIT,
            "formal_h100_submission_count": 2,
            "authorized_recovery": True,
            "source_profile_evidence_reused": True,
            "source_profile_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_profile_job": {
                "job_id": "wrong-job",
                "state": "CANCELLED",
                "profilers_complete": True,
            },
        }
    )
    _write_json(auxiliary, value)

    with pytest.raises(ValueError, match="source profiler Job"):
        analyze_phase6_qualification(
            profiles,
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
            expected_qualification_commit=QUALIFICATION_COMMIT,
            expected_h100_submission_count=2,
            expected_source_manifest_sha256=SOURCE_MANIFEST_SHA256,
            expected_source_profile_job_id="10837343",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("fallback", "fallback"),
        ("commit", "Git identity"),
        ("final_state", "final-state identities"),
        ("transform_calls", "transform-call contract"),
    ),
)
def test_profile_identity_failures_are_rejected(tmp_path, mutation, message):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(profiles[0].read_text(encoding="utf-8"))
    if mutation == "fallback":
        value["runtime_identity"]["fallback_used"] = True
    elif mutation == "commit":
        value["environment"]["git"]["head"] = "b" * 40
    elif mutation == "final_state":
        value["final_state_sha256"] = "different"
    else:
        value["transform_calls"]["inverse_per_step"] = 31
    _write_json(profiles[0], value)

    with pytest.raises(ValueError, match=message):
        analyze_phase6_qualification(
            profiles,
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
        )


def test_incomplete_or_duplicate_matrices_are_rejected(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    with pytest.raises(ValueError, match="exactly 12"):
        analyze_phase6_qualification(
            profiles[:-1],
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
        )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        analyze_phase6_qualification(
            [*profiles[:-1], profiles[0]],
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
        )
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        analyze_phase6_qualification(
            profiles,
            [*comparisons[:-1], comparisons[0]],
            auxiliary,
            expected_commit=COMMIT,
        )


def test_comparison_must_require_and_pass_byte_identity(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(comparisons[0].read_text(encoding="utf-8"))
    value["byte_identity_gate"] = False
    _write_json(comparisons[0], value)

    with pytest.raises(ValueError, match="byte identity failed"):
        analyze_phase6_qualification(
            profiles,
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
        )


def test_auxiliary_gate_failure_is_rejected(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    value = json.loads(auxiliary.read_text(encoding="utf-8"))
    value["gates"]["diagnostics_byte_identity_R320"] = False
    _write_json(auxiliary, value)

    with pytest.raises(ValueError, match="auxiliary gates failed"):
        analyze_phase6_qualification(
            profiles,
            comparisons,
            auxiliary,
            expected_commit=COMMIT,
        )


def test_cli_writes_once_and_refuses_overwrite(tmp_path):
    profiles, comparisons, auxiliary = _evidence(tmp_path)
    output = tmp_path / "phase6.json"
    arguments = [
        *(item for path in profiles for item in ("--profile", str(path))),
        *(item for path in comparisons for item in ("--comparison", str(path))),
        "--auxiliary-gates",
        str(auxiliary),
        "--expected-commit",
        COMMIT,
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["all_gates_passed"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(arguments)


def test_cli_performance_only_does_not_claim_long_run_eligibility(tmp_path):
    profiles, _, _ = _evidence(tmp_path)
    output = tmp_path / "performance.json"
    arguments = [
        *(item for path in profiles for item in ("--profile", str(path))),
        "--expected-commit",
        COMMIT,
        "--performance-only",
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["performance_gates_passed"] is True
    assert report["eligible_for_scientific_gate_recovery"] is True
    assert report["eligible_for_long_run_stage"] is False
