"""Bind the RC6 pressure component diagnostic support and scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_pressure_component_diagnostic.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_binds_the_failed_job_without_inventing_a_root_cause():
    failure = _record()["failed_qualification"]
    assert failure["job_id"] == "10900775"
    assert failure["failed_gate"] == "pressure_floor_fixed_and_tolerance_modes_on_cuda"
    assert failure["contract_pytests_passed"] == 36
    assert failure["formal_profiles_generated"] == 0
    assert failure["root_cause_attribution"] == (
        "UNRESOLVED_REQUIRES_COMPONENT_RESOLVED_PRESSURE_CONTRACT_DIAGNOSTIC"
    )


def test_record_binds_exact_diagnostic_sources():
    for relative, expected in _record()["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_diagnostic_reuses_the_failed_contract_without_relaxation():
    contract = _record()["diagnostic_contract"]
    assert contract["analysis_only"] is True
    assert contract["shape"] == [8, 6, 4]
    assert contract["lengths"] == [8.0, 6.0, 4.0]
    assert contract["seed"] == 9
    assert contract["friction"] == 0.2
    assert contract["viscosity"] == 0.73
    assert contract["dtypes"] == ["float32", "float64"]
    assert contract["tolerance_mode"]["max_iterations"] == 2000
    assert contract["fixed_mode"]["fixed_iterations"] == 4
    assert contract["same_as_failed_smoke"] is True
    assert contract["threshold_relaxed"] is False
    assert contract["solver_path_changed"] is False


def test_record_is_support_only_and_has_no_push_or_h100_authority():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_PRESSURE_COMPONENT_DIAGNOSTIC_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "qualification_support_only"
    )
    authorization = record["authorization"]
    assert all(value is False for value in authorization.values())
