"""Forward correction for the R4 equivalence helper work path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_98_s18_r4_equivalence_helper.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_98_binds_the_forward_helper_correction() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 98
    assert record["git_identity"] == {
        "qualification_helper_fix": "aebab348f8a570a5bbf1314efc1867e399047c4b",
        "previous_candidate": "8f12ca31b0b9b587e7d8469b09997934d4b92ba7",
        "runtime_implementation": "9cf54553b9f1bb2b94c2807c2e47adedeb415846",
        "record_commit_is_qualified_source": False,
    }
    relative, binding = next(iter(record["historical_successor_sources"].items()))
    assert binding["historical_sha256"] == (
        "9ea5c8c3fabca9d6033a57aea3fdcbb4efe18edeeb0e3eab0ddef2cdd264beb2"
    )
    assert binding["current_sha256"] == (
        "3570adfea175a5485ef66dced5cc3f48565906b84cc55ea8cf37463f3cbc258f"
    )
    assert _sha256(relative) == resolve_s03_successor(
        relative,
        binding["historical_sha256"],
    )


def test_successor_98_is_qualification_only_and_has_no_remote_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_qualification_helper_fix": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
