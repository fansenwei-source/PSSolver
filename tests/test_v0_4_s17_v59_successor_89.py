"""Forward-only binding for the S-17 V-59 rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_89_s17_v59_rework.json"
REWORK = ROOT / "notes/PSSolver_v0_4_0_s17_v59_rework.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_89_advances_every_changed_source() -> None:
    record = _read(RECORD)
    assert record["successor"] == 89
    assert record["classification"] == "V59_CLAIMED_FIXED_AWAITING_R0051_REVIEW"
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_successor_89_binds_the_exact_v59_fix_and_scope() -> None:
    record = _read(RECORD)
    rework = _read(REWORK)
    assert record["git_identity"] == {
        "implementation": "223b86dde0eb22ff1816e5a91ee93195ff5718a7",
        "implementation_parent": "8bcc540a6834246b500d3df01551680c7154e2a7",
        "record_commit_is_qualified_source": False,
    }
    assert record["rework_record_sha256"] == _sha256(REWORK)
    assert rework["behavior"]["submodule_attribute_access_warning_count"] == 0
    assert rework["behavior"]["functional_export_names_changed"] is False
    assert all(value is False for value in record["scope"].values())


def test_successor_89_authorizes_no_external_action() -> None:
    assert _read(RECORD)["authorization"] == {
        "local_fix_and_records": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
