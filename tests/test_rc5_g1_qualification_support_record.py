"""Evidence contract for the rc5 G-1 qualification-support batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_qualification_support.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_support_record_binds_classification_and_exact_identities():
    record = _record()
    assert record["phase"] == "G-1.5"
    assert record["classification"] == "READY_RC5_G1_VERSIONED_QUALIFICATION_SUPPORT"
    assert record["qualification_complete"] is False
    assert record["provider"]["baseline_commit"] == "658869456d762c917ca4409b091539c3733df2fb"
    assert record["provider"]["candidate_qualified_source_commit"] == "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
    assert record["consumer"]["qualified_source_commit"] == "c14e6b56c88081e7f9a5642e65ffd722eec7445b"


def test_support_record_registers_every_immutable_source_hash():
    for relative, expected in _record()["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected


def test_support_record_keeps_periodic_and_unchanged_path_oracles_separate():
    numerical = _record()["numerical_semantics"]
    assert numerical["global_final_state_sha_equality"] is False
    assert numerical["repaired_periodic_families"] == [
        "periodic_complete_stress",
        "periodic_functional",
    ]
    assert numerical["physical_relative_l2_max"] == 1.0e-12
    assert numerical["physical_linf_max"] == 1.0e-12
    assert numerical["candidate_hermitian_violation_max"] == 0.0
    assert len(numerical["byte_identity_families"]) == 6


def test_support_record_preserves_the_authorization_boundary():
    record = _record()
    assert record["authorization"] == {
        "push_executed": False,
        "h100_submission_authorized": False,
        "h100_submission_count": 0,
        "automatic_retry": False,
        "tag_created": False,
        "release_created": False,
        "pypi_published": False,
        "production_default_changed": False,
        "nematics3d_modified": False,
    }
    assert record["readiness"]["versioned_support_complete"] is True
    assert record["readiness"]["eligible_for_h100_submission"] is False
    assert record["next_gate"].startswith("freeze_the_installed_wheel_stage_command_plan")
