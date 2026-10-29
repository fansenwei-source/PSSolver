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
R0034 = SYNC / "results/R-0034.result.json"
CLOSURE = Path("/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v2")
V1_CLOSURE = Path("/home/fansenwei/pssolver_v0_3_s09_closure_f3620b8_20261027_v1")


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d0013_and_independent_reviews_are_exactly_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_d0013_closure.py::"
        "test_d0013_and_independent_reviews_are_exactly_bound"
    )
    assert record["decision"]["sha256"] == (
        "6ecd1b92a19fae5c0e66895b9b3216996a2a58d1c6d37fe8358331c8a6fd6f1f"
    )
    assert record["decision"]["status"] == "confirmed"
    expected = {
        "R-0031": (R0031, "INTERIM_PASS_AWAITING_USER_DECISION"),
        "R-0032": (R0032, "REWORK_REQUIRED"),
        "R-0033": (R0033, "PASS"),
        "R-0034": (R0034, "REWORK_REQUIRED"),
    }
    for review, (path, classification) in expected.items():
        assert path.name == f"{review}.result.json"
        assert len(record["reviews"][review]["sha256"]) == 64
        assert record["reviews"][review]["classification"] == classification


def test_noninterrupting_recompute_proves_the_exception_is_exhaustive() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_d0013_closure.py::"
        "test_noninterrupting_recompute_proves_the_exception_is_exhaustive"
    )
    assert record["closure_evidence"]["noninterrupting_recompute_sha256"] == (
        "1757374241e033c77a5fbe0b31df29e6f33a0dcae3e777c4ae45754d5927df01"
    )
    assert record["bounded_exception"] == {
        "decision": "D-0013",
        "peak_reserved_group_ratio": 1.0307692307692307,
        "candidate_peak_reserved_bytes": 562036736,
        "baseline_peak_reserved_bytes": 545259520,
        "peak_allocated_group_ratio": 1.0150638559607594,
        "timing_checks_passed": True,
        "scope_expansion_allowed": False,
        "analyzer_changed": False,
        "threshold_changed": False,
    }


def test_v1_archive_is_preserved_while_v2_binds_the_real_job_plan() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert record["closure_evidence"]["v1_preserved"] is True
    assert record["closure_evidence"]["v1_path"] == str(V1_CLOSURE)
    assert_pytest_migration(
        "tests/test_v0_3_s09_d0013_closure.py::"
        "test_v1_archive_is_preserved_while_v2_binds_the_real_job_plan"
    )
    assert record["closure_evidence"]["v1_path"] == str(V1_CLOSURE)
    assert record["closure_evidence"]["frozen_plan_sha256"] == (
        "a21b211b0c47ef0e83dd95de3f56bdb164fc109d1ccc58fd548afd919aac67d2"
    )


def test_closure_adjudication_changes_no_analyzer_threshold_or_scope() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_d0013_closure.py::"
        "test_closure_adjudication_changes_no_analyzer_threshold_or_scope"
    )
    assert record["classification"] == (
        "PASS_V0_3_S09_CUMULATIVE_QUALIFICATION_WITH_D0013_BOUNDED_MEMORY_EXCEPTION"
    )
    assert record["closure_evidence"]["closure_adjudication_sha256"] == (
        "4969cdc062d6cab301f3e3b5f0c411da02a37f997195eeaf6316b6c13c6e032f"
    )
    assert record["bounded_exception"]["decision"] == "D-0013"
    assert record["bounded_exception"]["scope_expansion_allowed"] is False


def test_v41_v42_v44_and_v45_match_the_d0013_boundary() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-41"]["status"] == "accepted"
    assert issues["V-41"]["decision"] == "D-0013"
    assert issues["V-42"]["status"] == "verified"
    assert issues["V-42"]["fixed_in"] == (
        "f3620b87581de5f29356ed1da89673cf08ffc8a7"
    )
    assert issues["V-42"]["verified_by"] == "R-0031"
    assert issues["V-44"]["status"] in {"registered", "claimed_fixed", "verified"}
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
    historical = _read(RECORD)["registry_import"]["V-45"]
    assert historical["status"] == "claimed_fixed"
    assert historical["fixed_in"] == (
        "719ae76069ba97b88a893ccf7404ddf18239fcc6"
    )
    assert issues["V-45"]["source_report"] == (
        "PSSolver_sync/results/R-0034.result.json"
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
