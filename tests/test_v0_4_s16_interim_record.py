"""Bindings for the S-16 R7/R9 implementation and R8 inventory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_70_s16_interim.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_s16_r7_r9_implementation_identity_and_sources_are_exact():
    record = _read()
    identity = record["git_identity"]
    assert identity["implementation"] == (
        "38262ecd066003821ef954a46d12fe7c23bfe4f5"
    )
    assert subprocess.check_output(
        ["git", "rev-parse", f"{identity['implementation']}^"],
        cwd=ROOT,
        text=True,
    ).strip() == identity["implementation_parent"]
    for relative, (historical, current) in record[
        "historical_successor_sources"
    ].items():
        assert _sha256(
            subprocess.check_output(
                ["git", "show", f"{identity['baseline']}:{relative}"],
                cwd=ROOT,
            )
        ) == historical
        assert _sha256(
            subprocess.check_output(
                ["git", "show", f"{identity['implementation']}:{relative}"],
                cwd=ROOT,
            )
        ) == current
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(
            subprocess.check_output(
                ["git", "show", f"{identity['implementation']}:{relative}"],
                cwd=ROOT,
            )
        ) == expected


def test_s16_r8_inventory_and_proposed_decision_are_bound_without_implementation():
    record = _read()
    inventory = ROOT / record["r8"]["inventory"]["path"]
    assert _sha256(inventory.read_bytes()) == record["r8"]["inventory"]["sha256"]
    assert record["r8"]["decision"]["status"] == "proposed"
    assert record["r8"]["status"] == (
        "inventory_complete_implementation_blocked_on_user_decision"
    )
    for relative, expected in record["new_inventory_sources"].items():
        assert _sha256((ROOT / relative).read_bytes()) == expected


def test_s16_interim_scope_changes_no_public_or_numerical_contract():
    record = _read()
    assert record["r9"]["gpu_specific_path_changed"] is False
    assert record["r9"]["legacy_exception_count_after"] == 6
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
