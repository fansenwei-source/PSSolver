"""Record-level contract for the S-17 V-59 rework."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s17_v59_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v59_record_freezes_the_exact_v030_submodule_sets() -> None:
    record = _read(RECORD)
    assert record["classification"] == "V59_CLAIMED_FIXED_AWAITING_R0051_REVIEW"
    assert record["git_identity"] == {
        "implementation": "223b86dde0eb22ff1816e5a91ee93195ff5718a7",
        "parent": "8bcc540a6834246b500d3df01551680c7154e2a7",
        "record_commit_is_qualified_source": False,
    }
    contract = record["implicit_submodule_contract"]
    assert len(contract["pssolver.functional"]) == 14
    assert len(contract["pssolver.operators"]) == 4
    assert len(contract["pssolver.runtime"]) == 12
    assert len(contract["pssolver.workflows"]) == 13
    assert contract["pssolver.operators"] == [
        "lifting",
        "projection",
        "robin",
        "tensor_divergence",
    ]


def test_v59_is_claimed_fixed_without_expanding_scope() -> None:
    record = _read(RECORD)
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-59")
    assert issue["status"] in {"claimed_fixed", "verified"}
    assert issue["fixed_in"] == record["git_identity"]["implementation"]
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_fix": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
