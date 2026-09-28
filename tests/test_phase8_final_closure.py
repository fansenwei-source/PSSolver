"""Audit the complete Phase 8 evidence chain and final scope freeze."""

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


def test_phase8_final_closure_binds_the_exact_authoritative_chain():
    record = _json("phase_8_final_closure.json")
    assert record["classification"] == (
        "PASS_PHASE8_CAPABILITY_EXPANSION_AND_INTEGRATED_H100_CLOSURE"
    )
    assert record["status"] == "complete"
    assert len(record["closure_chain"]) == 7
    for item in record["closure_chain"]:
        assert item["complete"] is True
        assert _sha256(ROOT / item["record"]) == item["record_sha256"]
    assert _sha256(NOTES / "phase_7_final_closure.json") == record[
        "prerequisite"
    ]["phase_7_closure_sha256"]


def test_phase8_closure_freezes_architecture_invariants():
    outcome = _json("phase_8_final_closure.json")["architecture_outcome"]
    assert outcome == {
        "tensor_free_public_simulation_declaration": True,
        "model_geometry_boundary_declarations_remain_orthogonal": True,
        "capability_resolution_before_allocation": True,
        "application_dispatch_before_timestep": True,
        "registry_lookup_in_timestep": False,
        "unregistered_combinations_fail_closed": True,
        "runtime_fallback_allowed": False,
        "installed_wheel_catalog_matches_source": True,
        "checkpoint_identity_and_payload_fail_closed": True,
        "plane_periodic_and_channel_independently_qualified": True,
    }


def test_phase8_closure_freezes_exact_public_catalog_and_applications():
    record = _json("phase_8_final_closure.json")
    catalog = record["public_catalog"]
    assert (
        catalog["model_count"],
        catalog["geometry_count"],
        catalog["boundary_policy_count"],
        catalog["qualified_combination_count"],
        catalog["qualified_runtime_path_count"],
    ) == (2, 3, 8, 4, 6)
    assert [item["application"] for item in record["qualified_applications"]] == [
        "plane_complete_stress_beris_edwards",
        "channel_legacy_active_force_active_nematics",
        "periodic_complete_stress_beris_edwards",
        "channel_complete_stress_beris_edwards",
    ]


def test_phase8_closure_preserves_defaults_and_boundary_limits():
    record = _json("phase_8_final_closure.json")
    compatibility = record["compatibility"]
    assert compatibility["plane_default"] == "legacy_production"
    assert compatibility["channel_default"] == "legacy_channel"
    assert compatibility["production_default_changed"] is False
    assert compatibility["compiled_plane_promoted"] is False
    assert compatibility["compiled_channel_promoted"] is False
    boundaries = record["boundary_capabilities"]
    assert boundaries["plane_static_prescribed_q"] is True
    assert boundaries["plane_strong_homeotropic_q"] is True
    assert boundaries["plane_strong_planar_q"] is True
    assert boundaries["finite_q_robin_relaxation_pilot_qualified"] is True
    assert boundaries["finite_q_robin_publicly_executable"] is False
    assert boundaries["nonhomogeneous_neumann_supported"] is False
    assert all(value is False for value in record["non_claims"].values())


def test_phase8_closure_authorizes_only_phase9_planning():
    record = _json("phase_8_final_closure.json")
    assert record["local_closure_verification"] == {
        "targeted": "10 passed",
        "full": "2427 passed, 8 subtests passed",
        "failed": 0,
        "skipped": 0,
        "deselected": 0,
        "xfailed": 0,
        "git_diff_check": "pass",
        "pdf_regenerated": False,
    }
    assert record["eligibility"] == {
        "qualification_complete": True,
        "phase_8_complete": True,
        "p8_6_complete": True,
        "eligible_for_phase_9_planning": True,
        "phase_9_authorized": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def test_future_verbatim_archive_includes_phase8_closure_without_rebuild():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "phase_8_p861_local_cumulative_audit.md",
        "phase_8_p861_local_cumulative_audit.json",
        "phase_8_p862_h100_qualification_plan.md",
        "phase_8_p862_h100_qualification_plan.json",
        "phase_8_p862_source_binding.json",
        "phase_8_p862_h100_closure.md",
        "phase_8_p862_h100_closure.json",
        "phase_8_final_closure.md",
        "phase_8_final_closure.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
