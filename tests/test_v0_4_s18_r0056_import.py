"""Forward contract for importing the R-0056 S-18 R4 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_102_r0056_import.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_102_binds_r0056_and_forward_sources() -> None:
    record = _read()
    assert record["successor"] == 102
    assert record["classification"] == "R0056_PASS_IMPORTED_S18_R4_COMPLETE_PENDING_PUSH"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_v0_4_claude_s18_r0056_review_20261101_v1",
        "archive_checksums_entries": 64,
        "archive_checksums_sha256": (
            "711d8f8fec012acc9856657e6a728bb32f80630840724da56fe3365f1dc7ab3a"
        ),
        "classification": "PASS",
        "request": "R-0056",
        "result_sha256": (
            "19532dbb81fa1da63337995edce3c9621c04a9b381e49f25e418b93308b27a99"
        ),
    }
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(relative) == resolve_s03_successor(relative, expected)


def test_r0056_closes_only_r4_and_preserves_future_batch_obligations() -> None:
    record = _read()
    assert record["registry_updates"] == []
    assert record["new_findings"] == []
    closure = record["s18_r4_closure"]
    assert closure["complete"] is True
    assert closure["aggregate_r1_r9_closed"] is False
    assert closure["remaining_batches"] == ["R5", "R1"]
    assert closure["bitwise_equivalence"] is True
    assert closure["real_nfs_retest_triggered"] is False
    assert closure["gpu_or_h100_requalification_required"] is False


def test_r0056_records_every_intentionally_unchanged_r4_file() -> None:
    record = _read()
    dispositions = record["unchanged_r4_batch_change_dispositions"]
    assert set(dispositions) == {
        "pssolver/integrator.py",
        "pssolver/integrators/scalar_reference.py",
        "pssolver/workflows/channel_beris_edwards.py",
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/workflows/plane_checkpoint.py",
    }
    assert all(item["disposition"] == "intentionally_unchanged" for item in dispositions.values())
    assert all(item["deferred_to"] is None for item in dispositions.values())
    assert dispositions["pssolver/integrator.py"]["retained_behavior"] == (
        "independent legacy step_count/refresh_count advancement"
    )


def test_r0056_import_changes_no_runtime_or_authorization() -> None:
    record = _read()
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
