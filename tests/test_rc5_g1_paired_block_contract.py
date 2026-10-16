"""CPU-only contracts for the proposed rc5 G-1 paired-block estimator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import pytest

from benchmarks.analyze_rc5_g1_paired_block_replay import (
    FAIL_CLASSIFICATION,
    FORMAL_PASS_CLASSIFICATION,
    PASS_CLASSIFICATION,
    analyze,
    main as analyze_main,
)
from benchmarks.plan_rc5_g1_paired_block_contract import (
    BLOCK_COUNT,
    BLOCK_SIZE,
    CLASSIFICATION,
    build_plan,
    main as plan_main,
)
from benchmarks.run_rc5_g1_paired_block_profile_matrix import (
    AUTHORIZATION_SCHEMA,
    PROGRESS_SCHEMA,
    SUPPORT_FILES,
    build_command_plan,
    preflight,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _samples(scale: float) -> list[float]:
    values = [scale] * 60
    for index in (5, 25, 45):
        values[index] = 2.0 * scale
    return values


def _write_matrix(root: Path, *, candidate_factor: float = 1.02) -> tuple[Path, Path]:
    plan = build_plan()
    profiles = root / "profiles"
    progress = root / "progress"
    profiles.mkdir(parents=True)
    progress.mkdir(parents=True)
    completed = []
    for sequence, entry in enumerate(plan["profile_matrix"], start=1):
        factor = candidate_factor if entry["provider_role"] == "candidate" else 1.0
        samples = _samples(factor)
        mean = statistics.fmean(samples)
        median = statistics.median(samples)
        stdev = statistics.stdev(samples)
        grid = plan["grid_contracts"][entry["grid"]]
        profile = {
            "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "config": {
                "family": entry["family"],
                "provider_role": entry["provider_role"],
                "provider_commit": entry["provider_commit"],
                "provider_wheel_sha256": entry["provider_wheel_sha256"],
                "trial": entry["trial"],
                "shape": grid["shape"],
                "lengths": grid["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
                "warmup_steps": 10,
                "profile_steps": 60,
                "post_flush_rewarm_steps": 3,
            },
            "provider_identity": {
                "role": entry["provider_role"],
                "commit": entry["provider_commit"],
                "wheel_sha256": entry["provider_wheel_sha256"],
            },
            "runtime_identity": {
                "requested": entry["runtime_path"],
                "effective": entry["runtime_path"],
                "fallback_used": False,
            },
            "environment": {
                "allocated_device": "cuda:0",
                "device_name": "NVIDIA H100 PCIe",
                "tf32_caller_before": {
                    "float32_matmul_precision": "highest",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "tf32_caller_after": {
                    "float32_matmul_precision": "highest",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
            },
            "finite": True,
            "completed_profile_steps": 60,
            "runtime_completed_steps": 73,
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "transform_calls": {
                "forward": entry["expected_forward_raw_calls"],
                "inverse": entry["expected_inverse_raw_calls"],
            },
            "timing": {
                "samples_seconds": samples,
                "mean_timestep_seconds": mean,
                "median_timestep_seconds": median,
                "sample_standard_deviation_seconds": stdev,
                "coefficient_of_variation": stdev / mean,
            },
            "memory": {
                "peak_allocated_bytes": 102 if entry["provider_role"] == "candidate" else 100,
                "peak_active_bytes": 102 if entry["provider_role"] == "candidate" else 100,
                "peak_reserved_bytes": 102 if entry["provider_role"] == "candidate" else 100,
            },
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": "b" * 64,
        }
        memory_value = 102 if entry["provider_role"] == "candidate" else 100
        profile["measurement_window"] = {
            "warmup_steps": 10,
            "post_flush_rewarm_steps": 3,
            "timed_steps": 60,
            "steps_after_warmup": 10,
            "measurement_start_completed_steps": 13,
            "rewarm_timed": False,
            "rewarm_in_formal_transform_counters": False,
            "rewarm_in_formal_compiler_counters": False,
            "rewarm_in_formal_memory_peaks": False,
            "formal_memory_semantics": "steady_state_after_allocation_recovery",
            "memory_snapshots": {
                name: {
                    "allocated_bytes": memory_value,
                    "reserved_bytes": memory_value,
                    "active_bytes": memory_value,
                    "peak_allocated_bytes": memory_value,
                    "peak_reserved_bytes": memory_value,
                    "peak_active_bytes": memory_value,
                }
                for name in (
                    "after_warmup",
                    "after_empty_cache",
                    "after_post_flush_rewarm",
                    "after_formal_timed_window",
                )
            },
            "cold_recovery_memory": {
                "allocated_delta_bytes": 0,
                "reserved_delta_bytes": 0,
                "active_delta_bytes": 0,
            },
            "device_telemetry_after_formal_window": {
                "applicable": False,
                "reason": "synthetic_cpu_contract_test",
            },
        }
        profile["diagnostic_replay"] = {
            "separate_fresh_runtime": True,
            "used_for_performance_or_memory_adjudication": False,
            "finite": True,
            "completed_steps": 73,
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": "b" * 64,
            "step_records": [
                {
                    "step_index": index,
                    "runtime_completed_steps": 13 + index,
                    "diagnostics": {
                        "applicable": False,
                        "reason": "synthetic_cpu_contract_test",
                    },
                }
                for index in range(1, 61)
            ],
        }
        path = profiles / f"{entry['key']}.json"
        path.write_text(json.dumps(profile, allow_nan=False), encoding="utf-8")
        completed.append(
            {"key": entry["key"], "sequence": sequence, "sha256": _sha256(path)}
        )
        (progress / f"{sequence:03d}.json").write_text(
            json.dumps(
                {
                    "schema": "pssolver.rc5.g1.measurement_window_progress.v1",
                    "completed": completed,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
    return profiles, progress


def test_plan_retains_all_samples_and_all_numeric_thresholds():
    plan = build_plan()
    assert plan["classification"] == CLASSIFICATION
    assert plan["profile_count"] == 48
    assert BLOCK_SIZE == 10
    assert BLOCK_COUNT == 6
    raw = plan["raw_sample_contract"]
    assert raw["samples_per_profile"] == 60
    assert raw["block_boundaries_zero_based_half_open"] == [
        [0, 10],
        [10, 20],
        [20, 30],
        [30, 40],
        [40, 50],
        [50, 60],
    ]
    assert raw["outlier_removal_allowed"] is False
    assert raw["sample_deletion_allowed"] is False
    assert raw["trial_replacement_allowed"] is False
    assert plan["thresholds"] == {
        "aggregate_all_sample_mean_ratio_max": 1.03,
        "median_of_profile_means_ratio_max": 1.03,
        "median_of_all_fixed_block_ratios_max": 1.03,
        "every_trial_median_fixed_block_ratio_max": 1.05,
        "peak_allocated_group_ratio_max": 1.03,
        "peak_reserved_group_ratio_max": 1.03,
    }
    assert plan["timing_diagnostics"]["coefficient_of_variation_reference"] == 0.05
    assert plan["timing_diagnostics"]["standalone_cv_rejection_authorized"] is False
    assert plan["authorization"]["h100_submission_authorized"] is True
    assert plan["authorization"]["formal_contract_adoption_authorized"] is True
    assert plan["authorization"]["maximum_formal_h100_submissions"] == 1
    assert plan["authorization"]["source_bound_execution_authorization_required"] is True
    assert plan["authorization"]["automatic_retry_authorized"] is False


def test_common_mode_spikes_are_retained_and_do_not_create_a_false_regression(tmp_path):
    profiles, progress = _write_matrix(tmp_path)
    result = analyze(profiles, progress)
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["proposed_contract_passed"] is True
    assert result["formal_requalification_performed"] is False
    assert result["qualification_complete"] is False
    assert all(group["passed"] for group in result["groups"])
    assert sum(
        group["cv_reference_exceedance_profile_count"] for group in result["groups"]
    ) == 48
    assert all(
        len(pair["fixed_block_ratios"]) == 6
        for group in result["groups"]
        for pair in group["pairs"]
    )


def test_new_evidence_mode_requires_full_window_and_emits_formal_gate_result(tmp_path):
    profiles, progress = _write_matrix(tmp_path)
    result = analyze(
        profiles,
        progress,
        evidence_mode="new_pre_registered_evidence",
        progress_schema="pssolver.rc5.g1.measurement_window_progress.v1",
    )
    assert result["classification"] == FORMAL_PASS_CLASSIFICATION
    assert result["analysis_only"] is False
    assert result["post_hoc_v10_replay"] is False
    assert result["formal_requalification_performed"] is True

    target = profiles / f"{build_plan()['profile_matrix'][-1]['key']}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.pop("diagnostic_replay")
    target.write_text(json.dumps(payload), encoding="utf-8")
    final_progress = progress / "048.json"
    progress_payload = json.loads(final_progress.read_text(encoding="utf-8"))
    progress_payload["completed"][-1]["sha256"] = _sha256(target)
    final_progress.write_text(json.dumps(progress_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic replay is missing"):
        analyze(
            profiles,
            progress,
            evidence_mode="new_pre_registered_evidence",
            progress_schema="pssolver.rc5.g1.measurement_window_progress.v1",
        )


@pytest.mark.parametrize("candidate_factor", [1.031, 1.06])
def test_real_regressions_are_rejected_without_deleting_samples(tmp_path, candidate_factor):
    profiles, progress = _write_matrix(tmp_path, candidate_factor=candidate_factor)
    result = analyze(profiles, progress)
    assert result["classification"] == FAIL_CLASSIFICATION
    assert result["proposed_contract_passed"] is False
    assert not all(group["passed"] for group in result["groups"])


def test_missing_nonfinite_or_sequence_tamper_fails_closed(tmp_path):
    profiles, progress = _write_matrix(tmp_path)
    target = profiles / f"{build_plan()['profile_matrix'][-1]['key']}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["timing"]["samples_seconds"][0] = float("nan")
    target.write_text(json.dumps(payload), encoding="utf-8")
    final_progress = progress / "048.json"
    final_payload = json.loads(final_progress.read_text(encoding="utf-8"))
    final_payload["completed"][-1]["sha256"] = _sha256(target)
    final_progress.write_text(json.dumps(final_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must be finite"):
        analyze(profiles, progress)

    second_root = tmp_path / "second"
    profiles, progress = _write_matrix(second_root)
    final = progress / "048.json"
    payload = json.loads(final.read_text(encoding="utf-8"))
    payload["completed"][-1]["sequence"] = 47
    final.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="progress 48 differs"):
        analyze(profiles, progress)


def test_clis_are_atomic_and_refuse_overwrite(tmp_path):
    plan_output = tmp_path / "plan.json"
    assert plan_main(["--output", str(plan_output)]) == 0
    assert json.loads(plan_output.read_text(encoding="utf-8")) == build_plan()
    with pytest.raises(FileExistsError):
        plan_main(["--output", str(plan_output)])

    profiles, progress = _write_matrix(tmp_path / "matrix")
    analysis_output = tmp_path / "analysis.json"
    assert analyze_main(
        [
            "--profile-root",
            str(profiles),
            "--progress-root",
            str(progress),
            "--output",
            str(analysis_output),
        ]
    ) == 0
    assert json.loads(analysis_output.read_text(encoding="utf-8"))[
        "classification"
    ] == PASS_CLASSIFICATION
    with pytest.raises(FileExistsError):
        analyze_main(
            [
                "--profile-root",
                str(profiles),
                "--progress-root",
                str(progress),
                "--output",
                str(analysis_output),
            ]
        )


def _runner_inputs(tmp_path: Path) -> dict[str, Path]:
    baseline = tmp_path / "baseline" / "bin" / "python"
    candidate = tmp_path / "candidate" / "bin" / "python"
    baseline.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    baseline.write_bytes(b"python\n")
    candidate.write_bytes(b"python\n")
    support = tmp_path / "support"
    for relative in SUPPORT_FILES:
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    return {"baseline": baseline, "candidate": candidate, "support": support}


def test_runner_planning_is_ready_but_execution_requires_bound_authority(tmp_path):
    inputs = _runner_inputs(tmp_path)
    arguments = {
        "baseline_python": inputs["baseline"],
        "candidate_python": inputs["candidate"],
        "support_root": inputs["support"],
        "output_root": tmp_path / "output",
    }
    command_plan = build_command_plan(**arguments)
    preflight(command_plan, for_execution=False)
    assert command_plan["profile_count"] == 48
    assert command_plan["progress_schema"] == PROGRESS_SCHEMA
    assert command_plan["evidence_mode"] == "new_pre_registered_evidence"
    assert command_plan["execution_authorization"] is None
    assert not Path(command_plan["output_root"]).exists()
    with pytest.raises(PermissionError, match="requires an authorization record"):
        preflight(command_plan, for_execution=True)

    source_sha256 = {
        relative: _sha256(inputs["support"] / relative)
        for relative in (
            "benchmarks/plan_rc5_g1_paired_block_contract.py",
            "benchmarks/analyze_rc5_g1_paired_block_replay.py",
        )
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema": AUTHORIZATION_SCHEMA,
                "formal_contract_adoption_authorized": True,
                "single_h100_submission_authorized": True,
                "automatic_retry_authorized": False,
                "maximum_formal_h100_submissions": 1,
                "source_sha256": source_sha256,
            }
        ),
        encoding="utf-8",
    )
    authorized = build_command_plan(
        **arguments,
        authorization_record=authorization,
    )
    preflight(authorized, for_execution=True)
    assert authorized["execution_authorization"]["sha256"] == _sha256(authorization)

    (inputs["support"] / "benchmarks/analyze_rc5_g1_paired_block_replay.py").write_text(
        "# tampered\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source identity differs"):
        preflight(authorized, for_execution=True)


def test_versioned_contract_record_binds_replay_and_requires_new_authority():
    record = json.loads(
        (
            ROOT
            / "notes"
            / "PSSolver_v0_2_0rc5_g1_paired_block_contract_v13.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == CLASSIFICATION
    replay = record["v10_post_hoc_replay"]
    assert replay["classification"] == FAIL_CLASSIFICATION
    assert replay["analysis_only"] is True
    assert replay["profile_count"] == replay["progress_record_count"] == 48
    assert replay["execution_sequence_validated"] is True
    assert replay["formal_v10_reclassified"] is False
    assert replay["qualification_granted"] is False
    failed = [group for group in replay["groups"] if not group["passed"]]
    assert [(group["family"], group["grid"]) for group in failed] == [
        ("channel_compiled_v2", "channel_R128x20x20")
    ]
    assert failed[0]["maximum_trial_median_fixed_block_ratio"] > 1.05
    assert record["thresholds"]["numeric_thresholds_relaxed"] is False
    assert record["thresholds"]["estimator_semantics_changed"] is True
    assert record["scope"]["qualification_complete"] is False
    assert record["scope"]["g1_complete"] is False
    assert record["authorization"]["push_authorized"] is True
    assert record["authorization"]["h100_submission_authorized"] is True
    assert record["authorization"]["formal_contract_adoption_authorized"] is True
    assert record["authorization"]["maximum_formal_h100_submissions"] == 1
    assert record["authorization"]["automatic_retry_authorized"] is False
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected

    job = (ROOT / "jobs" / "rc5_g1_paired_block_h100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:H100:1" in job
    assert "#SBATCH --no-requeue" in job
    assert "run_rc5_g1_checkpoint_tmp_preflight.py" in job
    assert "run_rc5_g1_environment.py" in job
    assert "run_rc5_g1_paired_block_profile_matrix.py" in job
    assert "--authorization-record" in job
    assert "PAIRED_BLOCK_COMPLETE" in job
    assert "touch \"${RC5_G1_OUTPUT_ROOT}/COMPLETE\"" not in job
