"""Forward correction for the R-0056 import-record source hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_103_r0056_import_hash.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_103_corrects_only_the_stale_r0056_import_hash() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 103
    assert record["superseded_import_attempt"]["commit"] == (
        "3d406a1fef055e32162a3aa9b43814c85b76df3a"
    )
    corrections = record["historical_successor_sources"]
    assert set(corrections) == {
        "notes/PSSolver_v0_3_0_s03_successor_102_r0056_import_zh.md"
    }
    for relative, binding in corrections.items():
        assert _sha256(relative) == binding["current_sha256"]
        assert _sha256(relative) == resolve_s03_successor(
            relative,
            binding["historical_sha256"],
        )


def test_successor_103_binds_sources_and_grants_no_remote_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(relative) == resolve_s03_successor(relative, expected)
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_record_correction": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
