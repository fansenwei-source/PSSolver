"""Forward-only source bindings for the local S-18 R4 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_97_s18_r4_step_progress.json"
)


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_97_binds_every_r4_source_forward() -> None:
    record = _read()
    assert record["successor"] == 97
    assert record["git_identity"] == {
        "implementation": "9cf54553b9f1bb2b94c2807c2e47adedeb415846",
        "implementation_parent": (
            "9ffe2fe9d148e10bc5ec6c23cdac373e5ebdb280"
        ),
        "record_commit_is_qualified_source": False,
    }
    successors = record["historical_successor_sources"]
    assert len(successors) == 10
    for relative, registration in successors.items():
        assert _sha256(relative) == resolve_s03_successor(
            relative,
            registration["historical_sha256"],
        )
        assert _sha256(relative) == registration["current_sha256"]


def test_successor_97_binds_introduced_sources_and_scope() -> None:
    record = _read()
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(relative) == resolve_s03_successor(relative, expected)
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_implementation": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
