"""Evidence bindings for the S-09 V-35/V-36 rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_34_s09_checkpoint_compatibility_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s09_checkpoint_rework_binds_exact_runtime_and_fixture() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_S09_CHECKPOINT_COMPATIBILITY_REWORK_CANDIDATE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["claims"]["V-35"]["rejected_literals"] == [
        "1.1", "0.1-provisional", "9.9"
    ]
    assert record["claims"]["V-35"]["payload_read_before_rejection"] is False
    assert record["claims"]["V-36"]["shared_disposition"] == (
        "state_only_migration"
    )
    assert record["fixture"]["source_commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )


def test_v35_v36_advance_only_after_r0023_reviews_the_runtime() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ("V-35", "V-36"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
        )
        assert issues[issue_id]["verified_by"] == "R-0024"
    assert issues["A3-18"]["status"] == "verified"
    assert issues["A3-18"]["verified_by"] == "R-0025"
    assert issues["NEW-14"]["status"] == "open"
    assert issues["V-33"]["status"] == "registered"
    assert issues["V-37"]["status"] == "verified"
    assert issues["V-37"]["fixed_in"] == (
        "87cf34cf4e17096a701f5cc5bc81076d14a9d237"
    )
    assert issues["V-37"]["verified_by"] == "R-0024"


def test_s09_checkpoint_rework_successors_are_forward_only() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s09_checkpoint_rework_does_not_authorize_external_action() -> None:
    record = _read(RECORD)
    scope = record["scope"]
    assert scope["runtime_paths_changed"] == [
        "pssolver/functional/channel_checkpoint.py",
        "pssolver/functional/periodic_checkpoint.py",
    ]
    assert all(
        value is False
        for key, value in scope.items()
        if key != "runtime_paths_changed"
    )
    assert record["next_request"] == "R-0023"
