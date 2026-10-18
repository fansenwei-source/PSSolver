"""Content-addressed RC4.3.3 implementation record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc433_lifting_capability_restriction.json"
)
RC444 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc444_pressure_metadata_identity_connection.json"
)
RC454 = ROOT / "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
RC455 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
)
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
RC6_R6 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_successor_sha256(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        RC444,
        RC454,
        RC455,
        RC456,
        RC5_R56,
        RC6_R2,
        RC6_R3,
        RC6_R6,
    ):
        successor = json.loads(path.read_text(encoding="utf-8")).get(
            "historical_successor_sources", {}
        ).get(relative)
        if successor is None:
            continue
        if successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
    return resolved


def test_rc433_records_the_frozen_restriction_and_parent_contract():
    record = _record()

    assert record["phase"] == "RC4.3.3"
    assert record["classification"] == (
        "PASS_RC4_3_3_PUBLIC_LIFTING_CAPABILITY_RESTRICTION"
    )
    baseline = record["baseline"]
    assert baseline["implementation_parent_commit"] == (
        "66e92dfb10ec63b7deadd0d37f8037e048224789"
    )
    assert _sha256(ROOT / baseline["contract_record"]) == baseline[
        "contract_record_sha256"
    ]
    assert _sha256(ROOT / baseline["analysis_record"]) == baseline[
        "analysis_record_sha256"
    ]


def test_rc433_records_fail_closed_order_and_catalog_state():
    record = _record()
    implementation = record["implementation"]
    catalog = record["catalog"]

    assert implementation["mode"] == "public_fail_closed_restriction"
    assert implementation["rejection_code"] == (
        "unsupported_lifted_wall_molecular_field"
    )
    assert all(
        implementation[key] is True
        for key in (
            "rejection_before_run_spec",
            "rejection_before_lowering",
            "rejection_before_construction",
            "rejection_before_runtime",
            "rejection_before_output",
        )
    )
    assert catalog["declarations_remain_importable"] is True
    assert catalog["prescribed_q_executable"] is False
    assert catalog["strong_homeotropic_q_executable"] is False
    assert catalog["strong_planar_q_executable"] is False
    assert catalog["homogeneous_neumann_plane_remains_qualified"] is True


def test_rc433_preserves_internal_layers_and_reports_successful_spaces():
    record = _record()
    spaces = record["molecular_field_space_metadata"]
    preserved = record["preserved_internal_capabilities"]

    assert spaces["requested_equals_effective"] is True
    assert spaces["fallback_used"] is False
    assert set(spaces["successful_public_compilers"]) == {
        "plane_complete_stress_beris_edwards",
        "periodic_complete_stress_beris_edwards",
        "channel_complete_stress_beris_edwards",
    }
    assert all(value is False for value in preserved.values())


def test_rc433_sources_are_content_addressed_and_scope_is_narrow():
    record = _record()

    superseded_by_rc434 = {
        "tests/test_phase8_p844_lifting_workflow_restart.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_rc434:
            continue
        actual = _sha256(ROOT / relative)
        assert _resolve_successor_sha256(relative, expected) == actual
    scope = record["scope"]
    assert all(value is False for value in scope.values())
    assert record["verification"]["H100_executed"] is False
    assert record["authorization"][
        "rc4_3_4_implementation_authorized"
    ] is False
    assert record["eligibility"][
        "eligible_for_rc4_3_4_implementation"
    ] is True
