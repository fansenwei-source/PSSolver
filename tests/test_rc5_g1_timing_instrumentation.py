"""CPU-only contracts for the rc5 G-1 timing instrumentation harness."""

from __future__ import annotations

import hashlib
import gc
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc5_g1_timing_instrumentation import (
    CLASSIFICATION,
    analyze,
)
from benchmarks.diagnose_rc5_g1_timing_instrumentation import (
    GCStepRecorder,
    SCHEMA as PROFILE_SCHEMA,
    parse_telemetry_line,
    resolve_visible_device_binding,
    scheduled_refresh_due,
)
from benchmarks.plan_rc5_g1_timing_instrumentation import (
    FAMILIES,
    build_plan,
)
from benchmarks.run_rc5_g1_timing_instrumentation import (
    _atomic_json,
    build_command_plan,
    preflight,
)


def test_plan_freezes_analysis_only_scope_and_formal_boundaries():
    plan = build_plan()

    assert plan["classification"] == "READY_FOR_ANALYSIS_ONLY_INSTRUMENTATION"
    assert plan["families"] == list(FAMILIES)
    assert plan["profile_count"] == 12
    assert [row["sequence"] for row in plan["profile_matrix"]] == list(
        range(1, 13)
    )
    assert {row["provider_role"] for row in plan["profile_matrix"]} == {
        "candidate"
    }
    assert all(row["timed_steps"] == 60 for row in plan["profile_matrix"])
    assert all(row["device"] == "cuda" for row in plan["profile_matrix"])
    assert {
        family: sum(row["family"] == family for row in plan["profile_matrix"])
        for family in FAMILIES
    } == {family: 3 for family in FAMILIES}

    boundaries = plan["frozen_boundaries"]
    assert boundaries["formal_raw_sample_count"] == 60
    assert boundaries["formal_cv_threshold"] == 0.05
    assert boundaries["formal_threshold_changed"] is False
    assert boundaries["outlier_removal_allowed"] is False
    assert boundaries["trial_replacement_allowed"] is False
    assert boundaries["formal_v10_reclassification_allowed"] is False

    authorization = plan["authorization"]
    assert authorization == {
        "implementation_and_cpu_tests_authorized": True,
        "push_authorized": False,
        "h100_submission_authorized": False,
        "automatic_retry_authorized": False,
        "production_default_changed": False,
    }


def test_refresh_markers_use_absolute_completed_step_phase():
    due_indices = []
    for index in range(60):
        completed_before = 13 + index
        marker = scheduled_refresh_due(
            "channel_compiled_v2", completed_before, completed_before + 1
        )
        if marker["due"]:
            due_indices.append(index)
    assert due_indices == [6, 26, 46]
    assert scheduled_refresh_due("plane_compiled_v2", 19, 20) == {
        "applicable": False,
        "due": False,
        "interval": None,
        "source": "frozen_rc5_g1_transform_call_contract",
    }
    with pytest.raises(RuntimeError, match="did not advance once"):
        scheduled_refresh_due("channel_compiled_v2", 19, 21)


def test_gc_recorder_associates_events_and_unregisters_exact_callback():
    recorder = GCStepRecorder()
    callbacks_before = tuple(gc.callbacks)
    recorder.start("synthetic")
    callback = recorder.registered_callback
    assert callback is not None
    assert callback in gc.callbacks
    recorder.begin_step(7)
    callback("start", {"generation": 1, "collected": 0, "uncollectable": 0})
    callback("stop", {"generation": 1, "collected": 3, "uncollectable": 0})
    recorder.end_step()
    recorder.stop()

    assert tuple(gc.callbacks) == callbacks_before
    assert recorder.registered_callback is None
    metadata = recorder.to_metadata()
    matching = [event for event in metadata["events"] if event["step_index"] == 7]
    assert [event["callback_phase"] for event in matching] == ["start", "stop"]
    assert all(event["location"] == "within_step" for event in matching)


def test_telemetry_parser_and_visible_device_binding_are_explicit():
    line = (
        "2026/10/16 12:00:00.000, 3, GPU-abc, 00000000:41:00.0, P0, "
        "1410, 1593, 71.25, 35, 82, 14"
    )
    row = parse_telemetry_line(line)
    assert row == {
        "timestamp": "2026/10/16 12:00:00.000",
        "index": 3,
        "uuid": "GPU-abc",
        "pci_bus_id": "00000000:41:00.0",
        "pstate": "P0",
        "sm_clock_mhz": 1410.0,
        "memory_clock_mhz": 1593.0,
        "power_draw_watts": 71.25,
        "temperature_celsius": 35.0,
        "gpu_utilization_percent": 82.0,
        "memory_utilization_percent": 14.0,
    }
    assert resolve_visible_device_binding("3", [row]) == {
        "resolved": True,
        "reason": None,
        "token": "3",
        "physical_index": 3,
        "uuid": "GPU-abc",
    }
    assert resolve_visible_device_binding("GPU-abc", [row])["resolved"] is True
    assert resolve_visible_device_binding(None, [row]) == {
        "resolved": False,
        "reason": "CUDA_VISIBLE_DEVICES_is_missing",
        "token": None,
    }
    assert resolve_visible_device_binding("0,1", [row])["resolved"] is False
    with pytest.raises(ValueError, match="wrong field count"):
        parse_telemetry_line("too,few,fields")


def test_runner_builds_twelve_isolated_installed_wheel_commands(tmp_path):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    support = tmp_path / "support"
    script = support / "benchmarks" / "diagnose_rc5_g1_timing_instrumentation.py"
    script.parent.mkdir(parents=True)
    script.write_text("# support\n", encoding="utf-8")
    output = tmp_path / "not-created-output"

    command_plan = build_command_plan(
        candidate_python=python,
        support_root=support,
        output_root=output,
    )
    preflight(command_plan)
    assert command_plan["analysis_only"] is True
    assert command_plan["profile_count"] == 12
    assert len({row["compiler_cache"] for row in command_plan["rows"]}) == 12
    for row in command_plan["rows"]:
        command = row["command"]
        assert command[0] == str(python)
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--profile-steps") + 1] == "60"

    (support / "pssolver").mkdir()
    with pytest.raises(RuntimeError, match="must not contain"):
        preflight(command_plan)


def _synthetic_window(family: str, *, synchronized: bool) -> dict[str, object]:
    records = []
    for index in range(60):
        completed_before = 13 + index
        records.append(
            {
                "step_index_zero_based": index,
                "step_index_one_based": index + 1,
                "runtime_completed_steps_before": completed_before,
                "runtime_completed_steps_after": completed_before + 1,
                "cuda_event_seconds": 0.01 + index * 1.0e-6,
                "host_duration_seconds": 0.011 + index * 1.0e-6,
                "scheduled_refresh": scheduled_refresh_due(
                    family, completed_before, completed_before + 1
                ),
                "public_diagnostics": {
                    "applicable": synchronized,
                    "reason": None if synchronized else "not_sampled",
                },
            }
        )
    return {
        "records": records,
        "gc": {
            "events": [
                {
                    "step_index": 9,
                    "location": "within_step",
                    "callback_phase": "start",
                }
            ]
        },
        "background_gpu_telemetry": {
            "row_count": 1,
            "parse_errors": [],
            "binding": {
                "resolved": True,
                "physical_index": 0,
                "uuid": "GPU-test",
            },
        },
    }


def test_analyzer_accepts_complete_synthetic_matrix_without_qualification_claim(
    tmp_path,
):
    plan = build_plan()
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    for entry in plan["profile_matrix"]:
        profile = {
            "schema": PROFILE_SCHEMA,
            "analysis_only": True,
            "used_for_formal_performance_adjudication": False,
            "config": {
                "family": entry["family"],
                "provider_commit": entry["provider_commit"],
            },
            "windows": {
                "asynchronous_compatibility": _synthetic_window(
                    entry["family"], synchronized=False
                ),
                "synchronized_attribution_replay": _synthetic_window(
                    entry["family"], synchronized=True
                ),
            },
            "identity": {
                "initial_state_sha256_equal": True,
                "final_state_sha256_equal": True,
            },
        }
        (profile_root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )

    result = analyze(profile_root)
    assert result["classification"] == CLASSIFICATION
    assert result["instrumentation_evidence_complete"] is True
    assert result["profile_count"] == 12
    assert result["formal_requalification_performed"] is False
    assert result["formal_v10_classification_preserved"] == (
        "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
    )
    assert result["root_cause_automatically_adjudicated"] is False
    assert result["correlation_is_causation"] is False


def test_atomic_json_round_trips_and_refuses_overwrite(tmp_path):
    output = tmp_path / "result.json"
    _atomic_json(output, {"finite": True, "analysis_only": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "analysis_only": True,
        "finite": True,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_json(output, {"analysis_only": False})


def test_implementation_record_binds_sources_without_authorizing_h100():
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (
            root
            / "notes"
            / "PSSolver_v0_2_0rc5_g1_timing_instrumentation_v12.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == (
        "READY_FOR_ANALYSIS_ONLY_TIMING_INSTRUMENTATION"
    )
    assert record["base_commit"] == (
        "2dabdf3439c2d4d602efc5f2a16ba0c7f1491803"
    )
    assert record["scope"]["formal_requalification_performed"] is False
    assert record["scope"]["h100_execution_performed"] is False
    assert record["authorization"]["push_authorized"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["automatic_retry_authorized"] is False
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
