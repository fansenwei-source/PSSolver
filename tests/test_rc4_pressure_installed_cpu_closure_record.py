"""Regression checks for the frozen RC4.4.5 pressure CPU closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc445_installed_cpu_closure.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc445_binds_runtime_support_plan_and_predecessor():
    record = _record()
    identity = record["git_identity"]
    assert record["classification"] == (
        "PASS_RC4_4_5_PROVIDER_SOURCE_AND_INSTALLED_WHEEL_PRESSURE_CPU_CLOSURE"
    )
    assert identity["runtime_source_commit"] == (
        "dc8c87abc7b216231d09ca12fcc7e76b2249a7ab"
    )
    assert identity["qualification_support_commit"] == (
        "115067817244858eb77a164aae4ef90759af9a2a"
    )
    assert _sha256(ROOT / identity["predecessor_record"]) == identity[
        "predecessor_record_sha256"
    ]
    assert _sha256(ROOT / identity["qualification_plan"]) == identity[
        "qualification_plan_sha256"
    ]


def test_rc445_records_exact_source_and_installed_results():
    record = _record()
    installed = record["installed_wheel_gate"]
    assert installed["classification"] == (
        "PASS_RC4_4_5_INSTALLED_WHEEL_PRESSURE_CPU_CLOSURE"
    )
    assert installed["passed"] == 85
    assert installed["source_shadow_import"] is False
    for field in ("failed", "skipped", "xfailed", "collection_errors"):
        assert installed[field] == 0
    source = record["source_verification"]
    assert source["qualification_runner_unit_tests_passed"] == 5
    assert source["dedicated_source_tests_passed"] == 85
    assert source["full_CPU_suite"]["passed"] == 3138
    assert source["full_CPU_suite"]["deselected_cuda_only"] == 7
    assert source["full_CPU_suite"]["subtests_passed"] == 8
    assert source["full_CPU_suite"]["failed"] == 0


def test_rc445_closes_provider_cpu_pressure_contract_without_overclaiming():
    closure = _record()["pressure_closure"]
    assert closure["B25_scale_robust_primal_and_transpose"].startswith("PASS_")
    assert closure["B28_capability_metadata_truth"].startswith("PASS_")
    assert closure["B29_fixed_and_tolerance_policy"].startswith("PASS_")
    assert closure["pressure_convergence_schema_version"] == 2
    assert closure["pressure_transpose_protocol_version"] == "2"
    assert closure["implicit_pressure_adjoint_version"] == "2"
    assert closure["production_forward_identity"].endswith(".v4")
    assert closure["functional_forward_identity"].endswith(".v4")
    assert closure["functional_derivative_identity"].endswith(".v4")
    assert closure["same_version_checkpoint_round_trip"] == "PASS"
    assert closure["stale_or_tampered_identity_rejection"].startswith("PASS_")
    assert closure["consumer_requalification_executed"] is False
    assert closure["h100_requalification_executed"] is False


def test_rc445_binds_qualification_sources_and_archive_integrity():
    record = _record()
    for relative, expected in record["qualification_source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, relative
    artifacts = record["artifact_identity"]
    assert artifacts["manifest_verification"] == "23/23 PASS"
    assert artifacts["complete_present"] is True
    assert len(artifacts["checksums_sha256"]) == 64


def test_rc445_scope_authorizes_only_consumer_planning_next():
    record = _record()
    assert record["scope"] == {
        "runtime_hot_path_changed_by_RC4_4_5": False,
        "checkpoint_acceptance_changed_by_RC4_4_5": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
        "push_executed": False,
    }
    assert record["eligibility"]["rc4_4_5_complete"] is True
    assert record["eligibility"]["eligible_for_rc4_4_6_planning"] is True
    assert record["authorization"]["rc4_4_6_consumer_change_authorized"] is False
    assert record["authorization"]["rc4_4_7_H100_authorized"] is False
    assert record["authorization"]["push_authorized"] is False
