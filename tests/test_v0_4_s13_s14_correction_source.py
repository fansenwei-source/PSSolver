"""Bindings for the local S-13 cleanup and the first dev/0.4 correction source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s13_s14_correction_source.json"
ARCHIVE = Path("/home/fansenwei/pssolver_v0_4_s13_workspace_cleanup_20261029_v1")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s13_archive_is_complete_and_content_addressed() -> None:
    cleanup = _read(RECORD)["s13_workspace_cleanup"]
    assert cleanup["classification"] == "PASS_S13_WORKSPACE_CLEANUP"
    assert cleanup["checksums_entries"] == 13
    assert (ARCHIVE / "COMPLETE").is_file()
    for key, relative in {
        "checksums_sha256": "checksums.sha256",
        "pruned_sha256": "PRUNED.txt",
        "stopped_pre_deletion_sha256": "STOPPED_PRE_DELETION.md",
        "pre_deletion_inventory_sha256": "pre_deletion_inventory.json",
        "worktree_inventory_sha256": "worktree_inventory.json",
        "post_cleanup_sha256": "post_cleanup.json",
        "complete_sha256": "COMPLETE",
    }.items():
        assert _sha256(ARCHIVE / relative) == cleanup[key]
    assert cleanup["deleted"] == {
        "named_non_trash_targets": 11,
        "trash_pairs": 81,
        "trash_name_set_sha256": (
            "cca10934378013af23c1bb7d40bff7ab295e350740951c0e00bf94d2c67682c0"
        ),
    }
    assert cleanup["retained"]["worktrees_inventoried"] == 60
    assert cleanup["retained"]["worktrees_deleted"] == 0
    assert cleanup["retained"]["baseline_mismatch_or_missing"] == 0


def test_s14_source_hashes_and_historical_records_are_exact() -> None:
    record = _read(RECORD)
    for relative, expected in record["source_hashes"].items():
        historical = subprocess.run(
            ["git", "show", f"eee4487658eada196ce3916cb08bd646da4993c6:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected
    for item in record["superseded_record_corrections"].values():
        assert item["rewritten"] is False
        assert _sha256(ROOT / item["path"]) == item["sha256"]
    assert set(record["s14_corrections"]) == {"V-47", "V-48", "V-49"}
    assert "D-0013" in record["s14_corrections"]["V-47"]
    assert "D-0014" in record["s14_corrections"]["V-48"]
    assert "G-01 through G-08" in record["s14_corrections"]["V-49"]


def test_s13_s14_source_changes_no_runtime_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"] == {
        "records_and_documentation_only": True,
        "runtime_changed": False,
        "packaging_changed": False,
        "version_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
    }
    assert record["authorization"] == {
        "local_commit": True,
        "push": False,
        "tag": False,
        "github_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
