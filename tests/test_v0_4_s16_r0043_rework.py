"""Forward-only contract for the S-16 R-0043 rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_72_s16_r0043_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_72_binds_the_exact_r0043_rework_sources():
    record = _read(RECORD)
    assert record["successor"] == 72
    assert record["classification"] == (
        "S16_R0043_REWORK_CLAIMED_FIXED_R8_IMPLEMENTATION_PENDING"
    )
    assert record["r0043"]["result_sha256"] == (
        "c2524ea963cb4c6d6a91d80e197e16dc8899613e42c01bb66e1b3468b90671ef"
    )
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["current_sha256"]
        )
        assert identity["historical_sha256"] != identity["current_sha256"]
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0043_findings_are_claimed_at_the_exact_fix_sources():
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        "V-50": "80e96ba6cdcd4a16c26c79ac4eea4aa1d35cb992",
        "V-51": "d5c604a780f9e60c52f00135f163e207a0a0140a",
        "V-52": "80e96ba6cdcd4a16c26c79ac4eea4aa1d35cb992",
    }
    for issue_id, fixed_in in expected.items():
        assert issues[issue_id]["status"] in {"claimed_fixed", "verified"}
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id].get("verified_by") in {None, "R-0044"}


def test_successor_72_corrects_completion_forward_without_overclaiming_r8():
    record = _read(RECORD)
    assert record["forward_corrections"]["successor_70_overclaim"][
        "immutable"
    ] is True
    assert record["r7"]["converted_to_shared_helper"] == 55
    assert set(record["r7"]["justified_exact_local_encoders"]) == {
        "pssolver/io/checkpoint_identity.py"
    }
    assert record["r9"]["v030_top_level_relations_not_widened"] is True
    assert record["r8_inventory"] == {
        "v1_rewritten": False,
        "v2_path": "notes/PSSolver_v0_4_0_s16_r8_inventory_v2.json",
        "test_files": 27,
        "failing_tests": 35,
        "external_roots": 13,
        "execution_observed": True,
        "implementation_started": False,
    }


def test_r0043_rework_authorizes_no_external_or_gpu_action():
    record = _read(RECORD)
    assert record["scope"] == {
        "public_interface_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
        "r8_implementation_started": False,
    }
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
