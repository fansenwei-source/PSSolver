"""Forward-only correction for successor 76's manually expanded commit ID."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_77_s16_commit_identity_correction.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_77_maps_the_nonexistent_identity_to_git_output() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    wrong, actual = next(iter(record["commit_identity_corrections"].items()))
    assert record["successor"] == 77
    assert subprocess.run(
        ["git", "cat-file", "-e", f"{wrong}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    ).returncode != 0
    assert subprocess.run(
        ["git", "rev-parse", "--verify", f"{actual}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == actual
    assert subprocess.run(
        ["git", "rev-parse", "6267333"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == actual


def test_successor_77_binds_changed_and_introduced_sources() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_successor_77_changes_only_record_validation() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
