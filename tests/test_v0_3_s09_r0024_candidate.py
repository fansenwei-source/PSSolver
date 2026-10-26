"""Bindings for the final local R-0024 pre-H100 review candidate."""

from __future__ import annotations

import json
from pathlib import Path

from v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_37_s09_r0024_candidate.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r0024_candidate_binds_exact_claimed_fixes() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_boundary"]["claimed_fixed"] == {
        "V-35": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "V-36": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "V-37": "87cf34cf4e17096a701f5cc5bc81076d14a9d237",
    }
    for issue_id, fixed_in in record["registry_boundary"]["claimed_fixed"].items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0024"
    assert issues["A3-18"]["status"] == "verified"
    assert issues["A3-18"]["verified_by"] == "R-0025"
    assert issues["NEW-14"]["status"] == "open"


def test_r0024_candidate_successors_are_forward_only() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert successor["current_sha256"]
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == __import__("hashlib").sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()


def test_r0024_candidate_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert not any(record["scope"].values())
    assert record["next_request"] == "R-0024"
