"""Audit the imported P7.6/P7.7 evidence and final Phase 7 closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _json(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p76_closure_imports_negative_and_long_run_evidence():
    record = _json("phase_7_p76_h100_closure.json")

    assert record["classification"] == "PASS_P7_6_CHANNEL_H100_CLOSURE"
    assert record["imported_report"]["sha256"] == (
        "801c86ddbf0f7ca0312624ca5fa724394e7cf466f8d84a92869382e9de008fa9"
    )
    assert record["negative_gates"]["subgates_passed"] == 8
    assert record["negative_gates"]["subgates_total"] == 8
    assert record["negative_gates"]["all_rejected_before_target_mutation"] is True
    assert record["long_run"]["q_u_p_byte_identical"] == "30/30"
    assert record["long_run"]["diagnostics_byte_identical"] is True
    assert record["long_run"]["pressure_zero_mean_gauge"] == "PASS"
    assert [item["entries"] for item in record["archive_evidence"]] == [
        243,
        39,
        34,
        38,
    ]
    assert record["eligibility"] == {
        "p7_6_complete": True,
        "phase_7_channel_migration_complete": True,
        "eligible_for_phase_8_planning": True,
        "phase_8_authorized": False,
        "phase_9_authorized": False,
        "production_default_changed": False,
        "compiled_channel_v2_promoted": False,
    }


def test_p7712_closure_imports_attribution_correct_recovery():
    record = _json("phase_7_p7712_h100_closure.json")

    assert record["classification"] == (
        "PASS_P7_7_12_H100_EVIDENCE_WITH_ATTRIBUTION_CORRECT_RECOVERY"
    )
    assert record["imported_report"]["sha256"] == (
        "ddc0af086b0fd4ecd2ab458f5cd36bd47d757748cc413a929f1cd6b29350968e"
    )
    assert record["execution"]["new_h100_submissions"] == 0
    assert record["execution"]["simulation_commands_started"] == 0
    assert record["input_evidence"]["report_count"] == 36
    assert record["input_evidence"]["old_manifest_verification_before"] == (
        "568/568 PASS"
    )
    assert record["input_evidence"]["old_manifest_verification_after"] == (
        "568/568 PASS"
    )
    attribution = record["public_timestep_attribution"]
    assert attribution["public_timestep_loop_count"] == 0
    assert attribution["application_elapsed_timer_passed_through"] is True
    assert attribution["public_wrapper_inside_application_timer"] is False
    assert record["numerical_equivalence"]["continuous_resume_q_u_p_complete"] == (
        "byte-for-byte"
    )
    assert record["numerical_equivalence"]["common_final_diagnostics_byte_identical"] is True
    assert record["recovery_archive"]["manifest_verification"] == "55/55 PASS"
    assert record["recovery_archive"]["complete_marker"] is True
    assert record["qualification"]["p7_7_complete"] is True


def test_phase7_final_closure_binds_exact_authoritative_records():
    record = _json("phase_7_final_closure.json")
    chain = record["closure_chain"]

    assert record["classification"] == (
        "PASS_PHASE7_CHANNEL_AND_PUBLIC_SIMULATION_ARCHITECTURE_CLOSURE"
    )
    assert record["status"] == "complete"
    assert len(chain) == 3
    for item in chain:
        path = ROOT / item["record"]
        assert item["complete"] is True
        assert _sha256(path) == item["record_sha256"]
    assert record["prerequisite"]["phase_6_complete"] is True
    assert _sha256(NOTES / "phase_6_final_closure.json") == record["prerequisite"][
        "phase_6_closure_sha256"
    ]


def test_phase7_closure_freezes_architecture_and_nonclaims():
    record = _json("phase_7_final_closure.json")

    assert record["architecture_outcome"] == {
        "tensor_free_public_simulation_declaration": True,
        "model_geometry_boundary_composition_seam": True,
        "compile_before_allocation": True,
        "application_dispatch_before_timestep": True,
        "application_dispatch_in_timestep": False,
        "package_owned_runtime_construction": True,
        "stable_public_result_protocol": True,
        "runtime_fallback_allowed": False,
        "checkpoint_identity_fail_closed": True,
        "plane_and_channel_independently_qualified": True,
    }
    assert [item["application"] for item in record["qualified_applications"]] == [
        "plane_complete_stress_beris_edwards",
        "channel_legacy_active_force_active_nematics",
    ]
    assert record["public_api"]["execution_of_unqualified_pair_allowed"] is False
    assert record["compatibility"]["plane_default"] == "legacy_production"
    assert record["compatibility"]["channel_default"] == "legacy_channel"
    assert all(value is False for value in record["non_claims"].values())
    assert record["local_closure_verification"] == {
        "targeted": "15 passed",
        "full": "2199 passed, 8 subtests passed",
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "pdf_regenerated": False,
    }
    assert record["eligibility"] == {
        "qualification_complete": True,
        "phase_7_complete": True,
        "p7_7_complete": True,
        "eligible_for_phase_8_planning": True,
        "phase_8_authorized": False,
        "phase_9_authorized": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def test_future_verbatim_archive_includes_complete_phase7_chain_without_rebuild():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    names = (
        "phase_7_p76_h100_closure.md",
        "phase_7_p76_h100_closure.json",
        "phase_7_p7710_public_runner_result.md",
        "phase_7_p7711_cpu_equivalence.md",
        "phase_7_p7712_cpu_oracle_recovery.md",
        "phase_7_p7712_h100_qualification_plan.md",
        "phase_7_p7712_h100_evidence_recovery.md",
        "phase_7_p7712_h100_closure.md",
        "phase_7_p7712_h100_closure.json",
        "phase_7_final_closure.md",
        "phase_7_final_closure.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
