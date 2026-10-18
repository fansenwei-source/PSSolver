from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_r513_record_binds_new_and_successor_sources():
    record = _record()
    g1_successors = json.loads(G1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    glibc_successors = json.loads(
        G1_GLIBC_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    rc6_successors = json.loads(RC6_R1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r2_successors = json.loads(RC6_R2_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r3_successors = json.loads(RC6_R3_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r4_successors = json.loads(RC6_R4_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r5_successors = json.loads(RC6_R5_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    for relative, expected in record["new_sources"].items():
        assert _sha256(relative) == expected
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        resolved = registration["current_sha256"]
        successor = g1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = glibc_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_successors.get(relative)
        accepted = (
            []
            if successor is None
            else successor.get(
                "accepted_predecessor_sha256",
                [successor["historical_sha256"]],
            )
        )
        if successor is not None and resolved in accepted:
            resolved = successor["current_sha256"]
        successor = rc6_r2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r4_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r5_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(relative) == resolved
        assert registration["introduced_by_phase"] == "R5-13"
        assert registration["reason"]


def test_r513_dispositions_are_explicit_and_compatibility_preserving():
    dispositions = _record()["dispositions"]
    assert dispositions["B27"]["status"] == "closed_hardened"
    assert dispositions["B14"]["status"] == "closed_hardened"
    assert dispositions["H100_test_portability"]["status"] == "closed_hardened"
    assert dispositions["NEW_15_pdf_text"]["status"] == "closed_hardened"
    assert dispositions["A3"]["status"] == "retained_for_compatibility"
    assert dispositions["B28"]["status"] == "retained_for_compatibility"
    assert dispositions["N4"]["status"] == "deferred_by_scope"


def test_r513_scope_remains_cpu_only_and_fail_closed():
    record = _record()
    assert record["status"] == "installed_wheel_cpu_verified"
    assert record["classification"] == (
        "PASS_R5_13_OPTIONAL_HYGIENE_INSTALLED_WHEEL_CPU_CLOSURE"
    )
    assert record["authorization"] == {
        "r5_13_complete": True,
        "eligible_for_next_work_item": True,
        "eligible_for_push": False,
        "eligible_for_h100": False,
    }
    scope = record["scope"]
    for key in (
        "scientific_equations_changed",
        "production_default_changed",
        "checkpoint_schema_changed",
        "legacy_checkpoint_identity_changed",
        "lifting_reopened",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False


def test_r513_installed_wheel_evidence_is_exact_and_cpu_only():
    evidence = _record()["installed_wheel_evidence"]
    assert evidence["source_archive_sha256"] == (
        "6d71f6e1ba2170b2bccd5e939b1583044ef26f0f51733b33aa3f543e18769edb"
    )
    assert evidence["wheel_sha256"] == (
        "05b3549c817747365a9c2a2ff614dbaa8648a4ac07435fc3713b209673261fdf"
    )
    assert evidence["source_shadow_import"] is False
    assert evidence["pip_check"] == "PASS"
    assert evidence["targeted"] == {"passed": 49, "skipped": 0, "failed": 0}
    assert evidence["applicable_full"] == {
        "passed": 3601,
        "skipped": 7,
        "failed": 0,
        "subtests_passed": 8,
        "excluded_source_layout_tests": 7,
    }
    assert len(evidence["skip_identity"]) == 7
    assert evidence["checksums"] == {"passed": 9, "failed": 0}
    assert evidence["nematics3d_modified"] is False
