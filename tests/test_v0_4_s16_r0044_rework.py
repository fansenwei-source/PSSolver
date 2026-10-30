"""Forward-only contract for the S-16 R-0044 rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_74_s16_r0044_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
R0043_TEST = ROOT / "tests/test_v0_4_s16_r0043_rework.py"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_74_binds_r0044_and_every_forward_source():
    record = _read(RECORD)
    assert record["successor"] == 74
    assert record["classification"] == (
        "S16_R0044_REWORK_CLAIMED_FIXED_PENDING_R0045"
    )
    assert record["r0044"]["result_sha256"] == (
        "b97c27ac0a133cb85167a6e680623c613438a0da7c1ac287f8274b3778f50d26"
    )
    assert record["r0044"]["archive"]["checksums_sha256"] == (
        "dc52d303673ecb21a1209bbb0a9697dc4cb504fd4fe0b18b8f80900062b584ee"
    )
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
        assert identity["historical_sha256"] != identity["current_sha256"]
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0044_registry_updates_and_v53_are_exact():
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        "V-50": "80e96ba6cdcd4a16c26c79ac4eea4aa1d35cb992",
        "V-51": "d5c604a780f9e60c52f00135f163e207a0a0140a",
        "V-52": "80e96ba6cdcd4a16c26c79ac4eea4aa1d35cb992",
    }
    for issue_id, fixed_in in expected.items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0044"
    expected = {
        "id": "V-53",
        "title": "S-16 test still branches on whether a PSSolver_sync file exists",
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0044.result.json",
        "first_found": "S-16 final review R-0044",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "claimed_fixed",
        "fixed_in": "05513a4132b1eb38e9bdff825f98a31b276c2932",
        "target_release": "0.4 S-16",
        "notes": issues["V-53"]["notes"],
    }
    for key, value in expected.items():
        if key != "status":
            assert issues["V-53"][key] == value
    assert issues["V-53"]["status"] in {"claimed_fixed", "verified"}
    if issues["V-53"]["status"] == "verified":
        assert issues["V-53"]["verified_by"] == "R-0045"


def test_v53_removes_the_external_file_existence_branch():
    source = R0043_TEST.read_text(encoding="utf-8")
    assert "/home/fansenwei/Desktop/Develop/PSSolver_sync" not in source
    assert "R0043.is_file()" not in source
    record = _read(RECORD)
    assert record["external_evidence_branch_scan"] == {
        "scope": "tests/**/*.py",
        "external_root": "/home/fansenwei",
        "remaining_external_evidence_existence_branches": 0,
        "repository_fixture_and_temporary_path_branches_excluded": True,
    }


def test_r0044_rework_authorizes_no_external_or_gpu_action():
    record = _read(RECORD)
    assert record["scope"] == {
        "runtime_changed": False,
        "public_interface_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
