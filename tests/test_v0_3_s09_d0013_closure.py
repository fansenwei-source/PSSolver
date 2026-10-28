"""Bindings for the D-0013 S-09 cumulative closure candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_51_s09_d0013_closure.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
SYNC = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync")
DECISION = SYNC / "decisions/D-0013.json"
R0031 = SYNC / "results/R-0031.result.json"
R0032 = SYNC / "results/R-0032.result.json"
R0033 = SYNC / "results/R-0033.result.json"
CLOSURE = Path("/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v1")


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d0013_and_independent_reviews_are_exactly_bound() -> None:
    record = _read(RECORD)
    assert _sha256(DECISION) == record["decision"]["sha256"] == (
        "6ecd1b92a19fae5c0e66895b9b3216996a2a58d1c6d37fe8358331c8a6fd6f1f"
    )
    assert _read(DECISION)["status"] == "confirmed"
    expected = {
        "R-0031": (R0031, "INTERIM_PASS_AWAITING_USER_DECISION"),
        "R-0032": (R0032, "REWORK_REQUIRED"),
        "R-0033": (R0033, "PASS"),
    }
    for review, (path, classification) in expected.items():
        assert _sha256(path) == record["reviews"][review]["sha256"]
        assert _read(path)["classification"] == classification


def test_noninterrupting_recompute_proves_the_exception_is_exhaustive() -> None:
    record = _read(RECORD)
    recompute_path = CLOSURE / "analysis/noninterrupting_recompute.json"
    recompute = _read(recompute_path)
    assert _sha256(recompute_path) == record["closure_evidence"][
        "noninterrupting_recompute_sha256"
    ]
    assert recompute["profile_count"] == 192
    assert recompute["group_count"] == 16
    assert recompute["failed_group_count"] == 2
    assert recompute["all_non_exempt_checks_passed"] is True
    assert recompute["analyzer_modified"] is False
    assert recompute["thresholds_modified"] is False
    failures = {
        (group["family"], group["grid"]): group
        for group in recompute["groups"]
        if group["failed_checks"]
    }
    assert set(failures) == {
        ("channel_legacy", "channel_R512x40x40"),
        ("channel_compiled_v2", "channel_R512x40x40"),
    }
    for group in failures.values():
        assert group["failed_checks"] == ["peak_reserved"]
        assert group["peak_reserved_group_ratio"] == 1.0307692307692307
        assert group["peak_allocated_group_ratio"] == 1.0150638559607594
        assert group["trials_at_or_below_individual_reference"] == 6


def test_closure_adjudication_changes_no_analyzer_threshold_or_scope() -> None:
    record = _read(RECORD)
    path = CLOSURE / "analysis/closure_adjudication.json"
    adjudication = _read(path)
    assert _sha256(path) == record["closure_evidence"][
        "closure_adjudication_sha256"
    ]
    assert adjudication["classification"] == record["classification"]
    assert adjudication["qualification_complete"] is True
    assert adjudication["s09_complete"] is True
    assert adjudication["cumulative_job"]["job_id"] == "10911538"
    assert adjudication["diagnostic_job"]["job_id"] == "10912583"
    assert adjudication["frozen_analyzer_recompute"]["analyzer_code_modified"] is False
    assert adjudication["frozen_analyzer_recompute"]["thresholds_modified"] is False
    exception = adjudication["bounded_exception"]
    assert exception["decision"] == "D-0013"
    assert exception["scope_is_exhaustive"] is True
    assert exception["runtime_family"] == [
        "channel_legacy",
        "channel_compiled_v2",
    ]
    assert exception["grid"] == "channel_R512x40x40"
    assert exception["dtype"] == "float32"
    assert exception["check"] == "peak_reserved_group_ratio"
    assert exception["applies_to_other_groups"] is False
    assert exception["applies_to_future_versions"] is False


def test_v41_v42_and_v44_match_the_d0013_boundary() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-41"]["status"] == "accepted"
    assert issues["V-41"]["decision"] == "D-0013"
    assert issues["V-42"]["status"] == "verified"
    assert issues["V-42"]["fixed_in"] == (
        "f3620b87581de5f29356ed1da89673cf08ffc8a7"
    )
    assert issues["V-42"]["verified_by"] == "R-0031"
    assert issues["V-44"]["status"] == "registered"
    assert "616bc453a38b8f1459b1c278b76596f9e83e8687" in issues["V-44"][
        "notes"
    ]
    correct = "616bc453d50040b20d1ef448cf2647358aeee1cc"
    assert correct in issues["V-44"]["notes"]
    subprocess.run(
        ["git", "cat-file", "-e", f"{correct}^{{commit}}"],
        cwd=ROOT,
        check=True,
    )


def test_s10_carry_forward_is_bound_without_a_new_h100_claim() -> None:
    record = _read(RECORD)
    carry = record["s10_carry_forward"]
    assert carry == {
        "decision": "D-0013",
        "reviews": ["R-0032", "R-0033"],
        "h100_rerun_required": False,
        "basis": (
            "R-0033 passed S-10 in the HPCC-mirror Python 3.10, NumPy 1.22.3, "
            "torch 2.5.1 environment after R-0032 identified the fingerprint rework."
        ),
    }


def test_closure_successors_are_forward_only_and_authorize_no_external_action() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        actual = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == actual
        assert resolve_s03_successor(relative, successor["current_sha256"]) == actual
    assert record["record_scope"] == {
        "evidence_only": True,
        "runtime_changed": False,
        "analyzer_changed": False,
        "thresholds_changed": False,
        "profiles_changed": False,
        "production_default_changed": False,
    }
    assert record["authorization"] == {
        "local_evidence_import": True,
        "push": False,
        "hpcc_write": False,
        "slurm_or_h100": False,
        "tag_or_release": False,
    }
