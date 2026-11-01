"""Forward contract for importing the R-0055 S-18 R2 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_96_r0055_import.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_96_binds_r0055_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 96
    assert record["classification"] == "R0055_PASS_IMPORTED_S18_R2_COMPLETE_PENDING_PUSH"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_v0_4_claude_s18_r0055_review_20261031_v1",
        "archive_checksums_entries": 89,
        "archive_checksums_sha256": (
            "bd62261f89ff7e12020a268ba665e812cadae35c4ccd240f59c34111810095e0"
        ),
        "classification": "PASS",
        "request": "R-0055",
        "result_sha256": (
            "fcf6b72831db7b293126464fc8b3a151b1689888d80a512a2537bc123a026a5a"
        ),
    }
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0055_closes_only_r2_and_preserves_future_batch_obligations() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == []
    assert record["new_findings"] == []
    assert record["s18_r2_closure"]["complete"] is True
    assert record["s18_r2_closure"]["aggregate_r1_r9_closed"] is False
    assert record["s18_r2_closure"]["remaining_batches"] == ["R4", "R5", "R1"]
    assert record["follow_up_constraints"] == {
        "future_writer_tree_cases": [
            "Functional Channel checkpoint export",
            "Functional Periodic checkpoint export",
        ],
        "reference_read_tensor_check_order": (
            "informational; the changed ordering is not externally reachable"
        ),
    }


def test_r0055_import_changes_no_runtime_or_authorization() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
