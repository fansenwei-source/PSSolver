"""Static gates for the pre-decision R8 dependency inventory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "notes/PSSolver_v0_4_0_s16_r8_inventory.json"


def _inventory() -> dict[str, object]:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_r8_inventory_covers_the_exact_eighteen_live_absolute_tests():
    record = _inventory()
    expected = {
        "tests/test_analyze_v0_3_s09_cpu_6x60_stop.py",
        "tests/test_v0_3_s08_r0021_import.py",
        "tests/test_v0_3_s09_d0013_closure.py",
        "tests/test_v0_3_s09_r0022_import.py",
        "tests/test_v0_3_s09_r0023_import.py",
        "tests/test_v0_3_s09_r0024_import.py",
        "tests/test_v0_3_s09_r0025_import.py",
        "tests/test_v0_3_s09_r0026_candidate.py",
        "tests/test_v0_3_s09_r0026_import.py",
        "tests/test_v0_3_s09_r0027_import.py",
        "tests/test_v0_3_s09_r0028_rework.py",
        "tests/test_v0_3_s09_r0029_rework.py",
        "tests/test_v0_3_s09_r0030_import.py",
        "tests/test_v0_3_s09_r0035_import.py",
        "tests/test_v0_3_s10_r0033_import.py",
        "tests/test_v0_3_s10_r0036_rework.py",
        "tests/test_v0_3_s10_r0037_import.py",
        "tests/test_v0_4_s13_s14_correction_source.py",
    }
    assert set(record["tests"]) == expected
    assert record["summary"]["tests_with_live_absolute_dependencies"] == 18


def test_r8_inventory_is_analysis_only_and_requires_a_user_decision():
    record = _inventory()
    assert record["classification"] == "R8_INVENTORY_COMPLETE_AWAITING_D0015"
    assert record["summary"]["fresh_github_clone_is_self_contained"] is False
    assert record["scope"] == {
        "implementation_changed": False,
        "historical_record_rewritten": False,
        "skip_or_xfail_added": False,
        "external_action_authorized": False,
    }
    assert len(record["sync_files"]["files"]) == 19
    assert len(record["non_sync_resources"]) == 17


def test_r8_plan_preserves_history_but_removes_live_machine_paths():
    effect = _inventory()["r8_effect"]
    assert "byte-for-byte unchanged" in effect["historical_records"]
    assert "repository-relative" in effect["tests"]
    assert "fresh clone" in effect["large_archives"]
    assert "without rewriting" in effect["successor_records"]
