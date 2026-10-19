"""Bind the RC6 qualification-support record to the implemented files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_qualification_support.json"
SUCCESSOR_RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_nvcc_bootstrap_recovery.json"
PROFILE_TF32_RECORD = (
    ROOT / "notes/PSSolver_v0_2_0rc6_profile_child_tf32_recovery.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc6_qualification_support_record_binds_exact_files_and_scope():
    value = json.loads(RECORD.read_text(encoding="utf-8"))
    successor = json.loads(SUCCESSOR_RECORD.read_text(encoding="utf-8"))
    profile_tf32 = json.loads(PROFILE_TF32_RECORD.read_text(encoding="utf-8"))
    assert value["classification"] == "READY_RC6_QUALIFICATION_ONLY_ORCHESTRATION_AND_ANALYSIS"
    assert value["implemented"]["profile_count"] == 192
    assert value["validation"]["pytest_passed"] == 22
    assert value["validation"]["pytest_failed"] == 0
    for relative, expected in value["support_files"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            candidates = (
                successor["historical_successor_sources"],
                profile_tf32["historical_successor_sources"],
            )
            registrations = [item[relative] for item in candidates if relative in item]
            assert len(registrations) == 1
            registration = registrations[0]
            assert registration["predecessor_sha256"] == expected
            assert registration["successor_sha256"] == actual
    remaining = value["remaining_before_submission"]
    assert remaining["stage_producers_implemented"] is False
    assert remaining["stage_producers_frozen"] is False
    assert remaining["explicit_single_h100_authorization_received"] is False
    assert set(value["authorization"].values()) == {False}
    status = value["status"]
    assert status["qualification_support_complete"] is True
    assert status["qualification_complete"] is False
    assert status["eligible_for_h100_submission"] is False
    assert status["eligible_for_release"] is False
    assert status["production_default_changed"] is False
