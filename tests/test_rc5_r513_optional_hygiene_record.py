from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_r513_record_binds_new_and_successor_sources():
    record = _record()
    for relative, expected in record["new_sources"].items():
        assert _sha256(relative) == expected
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == registration["current_sha256"]
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
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_R5_13_OPTIONAL_HYGIENE_CPU_CLOSURE"
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
