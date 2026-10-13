"""Static record checks for the RC4.4.3 pressure-policy implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc443_pressure_policy_implementation.json"
)
RC444 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc444_pressure_metadata_identity_connection.json"
)
RC447_RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_python310_compatibility_recovery.json"
)
RC447_PERFORMANCE_REPAIR = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_repair.json"
)
RC447_PERFORMANCE_RECOVERY_V10 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_recovery_v10.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc443_record_binds_parent_contract_and_current_sources():
    record = _record()
    assert record["phase"] == "RC4.4.3"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_4_3_PROVIDER_PRESSURE_POLICY_IMPLEMENTATION"
    )
    assert record["implementation_parent"] == (
        "069c4d84f26763699e4c92d9482eb492a87ae7fc"
    )
    contract = record["frozen_contract"]
    assert _sha256(contract["path"]) == contract["sha256"]
    latest = json.loads(RC444.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    recovery = json.loads(RC447_RECOVERY.read_text(encoding="utf-8"))[
        "successor_sources"
    ]
    performance_repair = json.loads(
        RC447_PERFORMANCE_REPAIR.read_text(encoding="utf-8")
    )["successor_sources"]
    performance_recovery = json.loads(
        RC447_PERFORMANCE_RECOVERY_V10.read_text(encoding="utf-8")
    )["successor_sources"]
    for relative, expected in record["source_sha256"].items():
        actual = _sha256(relative)
        if actual == expected:
            continue
        successor = performance_recovery.get(
            relative,
            performance_repair.get(
                relative,
                latest.get(relative, recovery.get(relative)),
            ),
        )
        assert successor is not None
        predecessor = expected
        if relative in performance_recovery:
            predecessor = performance_repair[relative]["current_sha256"]
        elif relative in performance_repair:
            intermediate = latest.get(relative, recovery.get(relative))
            if intermediate is not None:
                predecessor = intermediate["current_sha256"]
        assert successor["historical_sha256"] == predecessor
        assert successor["current_sha256"] == actual
        expected_phase = (
            "RC4.4.7-performance-recovery-v10-provider"
            if relative in performance_recovery
            else
            "RC4.4.7-performance-repair-v9-provider"
            if relative in performance_repair
            else "RC4.4.4"
            if relative in latest
            else "RC4.4.7-recovery"
        )
        assert successor["introduced_by_phase"] == expected_phase


def test_rc443_registers_exact_historical_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/linear_solvers/stokes/channel_no_slip.py",
        "pssolver/functional/pressure_metadata.py",
        "pssolver/functional/channel_pressure.py",
        "pssolver/functional/channel_pressure_adjoint.py",
        "tests/test_phase9_p982_stable_functional_api.py",
    }
    latest = json.loads(RC444.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    recovery = json.loads(RC447_RECOVERY.read_text(encoding="utf-8"))[
        "successor_sources"
    ]
    performance_repair = json.loads(
        RC447_PERFORMANCE_REPAIR.read_text(encoding="utf-8")
    )["successor_sources"]
    performance_recovery = json.loads(
        RC447_PERFORMANCE_RECOVERY_V10.read_text(encoding="utf-8")
    )["successor_sources"]
    for relative, registration in successors.items():
        assert registration["introduced_by_phase"] == "RC4.4.3"
        assert registration["historical_sha256"] != registration["current_sha256"]
        actual = _sha256(relative)
        if actual != registration["current_sha256"]:
            newest = performance_recovery.get(
                relative,
                performance_repair.get(
                    relative,
                    latest.get(relative, recovery.get(relative)),
                ),
            )
            assert newest is not None
            predecessor = registration["current_sha256"]
            if relative in performance_recovery:
                predecessor = performance_repair[relative]["current_sha256"]
            elif relative in performance_repair:
                intermediate = latest.get(relative, recovery.get(relative))
                if intermediate is not None:
                    predecessor = intermediate["current_sha256"]
            assert newest["historical_sha256"] == predecessor
            assert newest["current_sha256"] == actual
            expected_phase = (
                "RC4.4.7-performance-recovery-v10-provider"
                if relative in performance_recovery
                else
                "RC4.4.7-performance-repair-v9-provider"
                if relative in performance_repair
                else "RC4.4.4"
                if relative in latest
                else "RC4.4.7-recovery"
            )
            assert newest["introduced_by_phase"] == expected_phase
        else:
            assert actual == registration["current_sha256"]
        assert registration["reason"]


def test_rc443_freezes_policy_scale_and_failure_semantics():
    record = _record()
    fixed = record["policy"]["fixed_work"]
    tolerance = record["policy"]["relative_tolerance"]
    assert fixed["exact_budget_required"] is True
    assert fixed["relative_tolerance_role"] == "diagnostic_only_not_early_exit"
    assert tolerance["relative_tolerance_role"] == "required_postcondition"
    assert tolerance["unmet_cap_fails_closed"] is True
    assert record["scale_contract"]["absolute_breakdown_threshold_present"] is False
    assert record["failure_contract"]["partial_result_published"] is False
    assert record["failure_contract"][
        "production_warm_start_mutated_on_failure"
    ] is False


def test_rc443_versions_diagnostics_without_claiming_public_connection():
    diagnostics = _record()["diagnostics_core"]
    assert diagnostics["schema_version"] == 2
    assert diagnostics["pressure_transpose_protocol_version"] == "2"
    assert diagnostics["implicit_pressure_adjoint_version"] == "2"
    assert diagnostics["new_primal_invalidates_published_transpose_snapshot"] is True
    assert diagnostics["older_primal_transpose_does_not_overwrite_current_snapshot"] is True
    assert _record()["deferred_to_rc4_4_4"] == [
        "tensor-free declaration derived-policy metadata connection",
        "compiled public Simulation normalization connection",
        "public runtime diagnostics exposure",
        "capability truth metadata repair",
        "production and functional forward/derivative identity v4",
        "checkpoint restart compatibility and negative gates",
    ]


def test_rc443_stops_before_consumer_h100_push_or_default_changes():
    record = _record()
    assert record["authorization"] == {
        "rc4_4_3_complete": True,
        "rc4_4_4_planning_eligible": True,
        "rc4_4_4_implementation_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "h100_submission_authorized": False,
        "push_authorized": False,
        "production_default_changed": False,
    }
    nonclaims = record["nonclaims"]
    assert nonclaims["B25_closed"] is False
    assert nonclaims["B28_closed"] is False
    assert nonclaims["B29_closed"] is False
    assert nonclaims["rc4_4_complete"] is False
    assert nonclaims["H100_qualified"] is False
    assert nonclaims["production_default_changed"] is False
    assert nonclaims["PSSolver_Control_modified"] is False
    assert nonclaims["nematics3d_modified"] is False
