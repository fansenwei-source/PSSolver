"""Regression checks for the frozen RC4.3.6 installed-wheel record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc436_installed_cpu_closure.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc436_records_complete_source_and_installed_wheel_closure():
    record = _record()
    assert record["classification"] == (
        "PASS_RC4_3_6_INSTALLED_WHEEL_LIFTING_CPU_CLOSURE"
    )
    assert record["installed_wheel_gate"]["passed"] == 11
    assert record["installed_wheel_gate"]["source_shadow_import"] is False
    assert record["source_verification"]["dedicated_source_tests_passed"] == 11
    assert record["source_verification"]["full_CPU_suite"] == {
        "passed": 3025,
        "skipped": 0,
        "deselected_cuda_only": 7,
        "subtests_passed": 8,
        "failed": 0,
        "xfailed": 0,
        "elapsed_seconds": 200.99,
    }
    assert record["artifact_identity"]["manifest_verification"] == "23/23 PASS"
    assert record["artifact_identity"]["complete_present"] is True


def test_rc436_freezes_independent_second_order_observations():
    convergence = _record()["independent_convergence"]
    assert convergence["wall_points"] == [16, 32, 64]
    assert all(
        left > right
        for left, right in zip(convergence["errors"], convergence["errors"][1:])
    )
    assert min(convergence["adjacent_orders"]) > (
        convergence["minimum_required_adjacent_order"]
    )
    assert convergence["public_lifted_Beris_Edwards_qualified"] is False


def test_rc436_preserves_the_fail_closed_public_boundary():
    record = _record()
    closure = record["behavioral_closure"]
    assert closure["public_lifted_Plane_fail_closed"].startswith("PASS_4_OF_4")
    assert closure["public_homogeneous_Neumann_requested_effective_identity"].startswith(
        "PASS_4_OF_4"
    )
    assert closure["same_version_internal_lifted_file_restart"] == (
        "PASS_BYTE_IDENTICAL_4_STEP_VS_2_PLUS_2"
    )
    assert closure["public_lifting_capability_reopened"] is False
    assert closure["localized_policy_publicly_qualified"] is False


def test_rc436_record_binds_qualification_sources_and_scope():
    record = _record()
    for relative, expected in record["qualification_source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, relative
    assert _sha256(
        ROOT / record["git_identity"]["predecessor_record"]
    ) == record["git_identity"]["predecessor_record_sha256"]
    assert _sha256(
        ROOT / record["git_identity"]["qualification_plan"]
    ) == record["git_identity"]["qualification_plan_sha256"]
    assert record["scope"] == {
        "runtime_hot_path_changed": False,
        "checkpoint_acceptance_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
        "push_executed": False,
    }
    assert record["eligibility"]["rc4_3_6_complete"] is True
    assert record["eligibility"]["eligible_for_rc4_3_7_planning"] is True
    assert record["authorization"]["rc4_3_7_H100_authorized"] is False
    assert record["authorization"]["push_authorized"] is False


def test_rc436_bootstrap_history_does_not_reclassify_solver_evidence():
    history = _record()["bootstrap_history"]
    assert history["pre_collection_failures"] == 2
    assert history["formal_result_written_before_dependency_closure"] is False
    assert history["solver_failure"] is False
    assert history["scientific_failure"] is False
