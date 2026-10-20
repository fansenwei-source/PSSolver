"""Fail-closed planning gate for the v0.2.1 N-1 maintenance fix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_plan.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_n1_plan_is_the_only_v021_scope():
    record = _record()
    assert record["classification"] == (
        "PLAN_V0_2_1_N1_NFS_CHECKPOINT_PUBLICATION_FROZEN"
    )
    assert record["status"] == "planning_only"
    assert record["source"]["branch"] == "maint/v0.2"
    assert record["scope"]["only_issue"] == (
        "NFS-compatible no-clobber checkpoint publication"
    )
    for key, value in record["scope"].items():
        if key != "only_issue":
            assert value is False


def test_n1_plan_freezes_all_contract_items():
    record = _record()
    assert len(record["base_contract"]) == 18
    assert len(record["supplemental_contract"]) == 8
    text = "\n".join(
        [*record["base_contract"], *record["supplemental_contract"]]
    )
    for required in (
        "RENAME_NOREPLACE",
        "atomic mkdir",
        "read the published checkpoint back",
        "directory-fsync",
        "byte-for-byte identical",
        "errno and statfs",
        "syscall(316)",
        "cross-client concurrency",
        "provider identity",
    ):
        assert required in text


def test_n1_plan_does_not_authorize_implementation_or_external_actions():
    authorization = _record()["authorization"]
    assert authorization["plan_freeze_authorized"] is True
    for key in (
        "implementation_started",
        "push_authorized",
        "h100_authorized",
        "tag_authorized",
        "release_authorized",
    ):
        assert authorization[key] is False


def test_n1_h100_is_small_and_separately_authorized():
    qualification = _record()["qualification"]
    assert qualification["h100"] == (
        "one small checkpoint/restart job only after separate user authorization"
    )
