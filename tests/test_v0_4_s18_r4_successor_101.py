"""Forward bindings for the common R4 child-process root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_101_s18_r4_equivalence_cwd.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_101_binds_both_forward_corrections() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 101
    assert len(record["historical_successor_sources"]) == 3
    for relative, binding in record["historical_successor_sources"].items():
        assert _sha256(relative) == resolve_s03_successor(
            relative,
            binding["historical_sha256"],
        )


def test_successor_101_is_qualification_only_and_has_no_remote_authority() -> None:
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
