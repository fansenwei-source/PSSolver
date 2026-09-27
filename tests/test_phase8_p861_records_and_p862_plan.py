"""Freeze the P8.6.1 record and pre-bind P8.6.2 H100 plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _json(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_p861_record_binds_the_audited_implementation_and_verification():
    record = _json("phase_8_p861_local_cumulative_audit.json")
    assert record["classification"] == "PASS_P8_6_1_LOCAL_CUMULATIVE_AUDIT"
    assert record["identity"]["implementation_commit"] == (
        "aeee095a2191fe80b311e488144566f55da3c19f"
    )
    assert record["identity"]["analyzer_sha256"] == _sha256(
        "benchmarks/analyze_phase8_cumulative_closure.py"
    )
    assert record["identity"]["test_sha256"] == _sha256(
        "tests/test_phase8_p861_local_cumulative_audit.py"
    )
    assert record["verification"]["full"] == (
        "2410 passed, 8 subtests passed"
    )
    assert record["scope"] == {
        "runtime_or_timestep_changed": False,
        "numerical_operator_changed": False,
        "compiler_registration_changed": False,
        "public_api_changed": False,
        "checkpoint_or_output_schema_changed": False,
        "production_default_changed": False,
    }


def test_p861_record_closes_only_local_audit():
    authorization = _json("phase_8_p861_local_cumulative_audit.json")[
        "authorization"
    ]
    assert authorization == {
        "p8_6_1_complete": True,
        "eligible_for_p8_6_2_h100_planning": True,
        "p8_6_2_h100_executed": False,
        "p8_6_3_authorized": False,
        "phase_9_authorized": False,
    }


def test_p862_plan_requires_source_binding_and_one_installed_wheel_job():
    plan = _json("phase_8_p862_h100_qualification_plan.json")
    assert plan["classification"] == (
        "P8_6_2_INTEGRATED_H100_CLOSURE_PLANNED_NOT_EXECUTED"
    )
    assert plan["qualification_source"] == {
        "binding_record_required": True,
        "binding_record": (
            "notes/architecture_v0_2/phase_8_p862_source_binding.json"
        ),
        "exact_commit_bound": False,
    }
    assert plan["qualification_gates"]["single_job"] is True
    assert plan["qualification_gates"]["installed_wheel_only"] is True
    assert plan["qualification_gates"]["automatic_retry"] is False
    assert plan["authorization"]["formal_h100_submission_limit"] == 1


def test_p862_plan_covers_exact_public_runtime_paths_and_two_auxiliary_paths():
    cases = _json("phase_8_p862_h100_qualification_plan.json")["h100_cases"]
    assert [case["runtime_path"] for case in cases[:6]] == [
        "legacy_production",
        "compiled_v2",
        "legacy_channel",
        "compiled_channel_v2",
        "periodic_spectral",
        "channel_complete_stress",
    ]
    assert [case["case"] for case in cases[6:]] == [
        "plane_static_q_lifting",
        "finite_q_robin_relaxation_pilot",
    ]
    assert all(case["public"] for case in cases[:7])
    assert cases[7]["public"] is False


def test_p862_plan_moves_all_six_cuda_only_tests_to_the_h100_job():
    plan = _json("phase_8_p862_h100_qualification_plan.json")
    tests = plan["cuda_only_tests"]
    assert len(tests) == 6
    assert len(set(tests)) == 6
    assert plan["cpu_gate"]["cuda_only_deselect_count"] == 6
    assert all("::test_" in node_id for node_id in tests)


def test_p861_and_p862_sources_are_registered_without_regenerating_pdf():
    archive = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "phase_8_p861_local_cumulative_audit.md",
        "phase_8_p861_local_cumulative_audit.json",
        "phase_8_p862_h100_qualification_plan.md",
        "phase_8_p862_h100_qualification_plan.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert archive.count(token) == 1
        positions.append(archive.index(token))
    assert positions == sorted(positions)
