"""Bindings for the local S-09/S-10 integration candidate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_58_s09_s10_integration.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
BASE = "ff70b4c3be4c41830ced0316d5d28ef9ef6ce4df"
REPLAYED_PREFIX = [
    "c6d4b5e989435472e983bce30938f2003751aa3b",
    "da8356ae0c9d35821b44ed2c9eb44ef70ba13449",
    "55fa5b0e829d477acea31efc78b29478813a9dcd",
    "26a51777651eb70cb09bef65164fb18c5a18a4e4",
    "a54074013e5f0fb9bfe278baf944f805bf86eef4",
    "98fb4a150f24cf8a1e40eedce15c1251452d7f78",
    "f08eb63e9827d24087c9603f0788d6fec491a035",
    "ce1874c4470fe575e6e8d6acd5930a3eb5d14006",
    "79b4e84945f12f155614ac2094aba6128055f8b8",
]
DUPLICATE_R0031_PATHS = [
    "notes/PSSolver_v0_3_0_s03_successor_51_r0031_import.json",
    "notes/PSSolver_v0_3_0_s03_successor_51_r0031_import_zh.md",
    "tests/test_v0_3_s09_r0031_import.py",
]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def test_integration_is_linear_on_the_reviewed_s09_tip() -> None:
    record = _read(RECORD)
    identity = record["git_identity"]
    assert record["plan_step"] == "S-10"
    assert identity["integration_base"] == (
        BASE
    )
    assert _git("merge-base", identity["integration_base"], "HEAD") == (
        identity["integration_base"]
    )
    expected_tree = record["runtime_boundary"]["qualified_source_pssolver_tree"]
    assert _git("rev-parse", identity["qualified_s10_runtime_source"] + ":pssolver") == (
        expected_tree
    )
    assert _git("rev-parse", "HEAD:pssolver") == expected_tree


def test_duplicate_r0031_import_is_not_replayed() -> None:
    replay = _read(RECORD)["replay"]
    commits = _git("rev-list", "--reverse", f"{BASE}..HEAD").splitlines()
    assert commits == [*REPLAYED_PREFIX, _git("rev-parse", "HEAD")]
    assert len(commits) == 10
    assert set(replay["original_to_replayed"]) == {
        "e75ef77c2dc03237db1ddae8b275f9c8055fe06b",
        "dc955c5db5f0b67949115062577a46d22776083e",
        "1cd484f56816ea0a2532b8ebfa8010a0a60b74e3",
        "28ba3bc3c9c930b0a100bf2d926c11dca8b100d3",
        "61838af7910d955b27268abd4a79e4d32f235f58",
        "c37991ba37819d88525dd774fb063ae6234cdce2",
        "c70ea9790af4532241f8e7342fcc17436d1cb6b0",
        "18588b1be7faaa09c1e1fdb0828c1861e4e69757",
        "e915111b2f640739ca9f04601dd0147a923c451c",
        "b0a203430b1b5a6bc03bfcb377e6e459a28dbbc9",
    }
    assert all(not (ROOT / relative).exists() for relative in DUPLICATE_R0031_PATHS)
    assert replay["successor_renumbering"] == {
        "46": 53,
        "47": 54,
        "48": 55,
        "49": 56,
        "50": 57,
    }


def test_integration_registry_preserves_both_reviewed_boundaries() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-33"]["status"] == "verified"
    assert issues["V-33"]["verified_by"] == "R-0033"
    assert issues["V-33"]["fixed_in"] == (
        "a54074013e5f0fb9bfe278baf944f805bf86eef4"
    )
    assert issues["V-41"]["status"] == "accepted"
    assert issues["V-41"]["decision"] == "D-0013"
    assert issues["V-42"]["status"] == "verified"
    assert issues["V-42"]["verified_by"] == "R-0031"
    assert issues["V-43"]["status"] == "verified"
    assert issues["V-43"]["verified_by"] == "R-0033"
    assert issues["V-43"]["fixed_in"] == (
        "a54074013e5f0fb9bfe278baf944f805bf86eef4"
    )
    assert issues["V-44"]["status"] == "registered"
    assert issues["V-45"]["status"] == "verified"
    assert issues["V-45"]["verified_by"] == "R-0035"
    assert issues["V-46"]["status"] == "claimed_fixed"
    assert record["registry_boundary"] == {
        key: {field: issues[key][field] for field in expected}
        for key, expected in record["registry_boundary"].items()
    }


def test_integration_successor_is_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        for predecessor in successor.get(
            "accepted_predecessor_sha256", [successor["historical_sha256"]]
        ):
            assert resolve_s03_successor(relative, predecessor) == live


def test_integration_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["local_only"] is True
    assert record["scope"]["runtime_changed_after_reviewed_s10"] is False
    assert record["scope"]["duplicate_r0031_imported"] is False
    assert all(value is False for value in record["authorization"].values())
    assert record["next_request"] == "R-0037"
