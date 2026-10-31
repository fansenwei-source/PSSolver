"""Bindings for the local R-0048 S-17 rework sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s17_r0048_rework_sources.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_r0048_rework_sources_are_exact_and_cross_repository_explicit() -> None:
    record = _read()
    assert record["classification"] == (
        "R0048_REWORK_SOURCES_COMPLETED_AWAITING_R0049"
    )
    sources = record["fix_sources"]
    assert sources["V-56"]["repository"] == "PSSolver-Control"
    assert sources["V-56"]["commit"] == (
        "215e35c5504a305d20a28ad3987fd5ed2f94683a"
    )
    assert sources["V-57"]["commit"] == (
        "140bd273693c341545ceb023bfd2de2e36458e07"
    )
    assert sources["V-58"]["repository"] == "PSSolver"
    assert sources["V-58"]["commit"] == (
        "d98472aa975776fcd78f95b133d2abb747bce83b"
    )
    assert set(sources["V-56"]["files"]) == {
        "docs/pssolver_rc4_pressure_source_mapping.json",
        "tests/test_pssolver_rc4_pressure_migration.py",
    }
    assert set(sources["V-57"]["files"]) == {
        "docs/pssolver_v0_3_0_main_migration.json",
        "docs/pssolver_v0_3_0_main_migration.md",
        "tests/test_pssolver_v0_3_0_main_migration.py",
    }


def test_r0048_rework_sources_do_not_authorize_implementation_or_actions() -> None:
    scope = _read()["scope"]
    assert all(value is False for value in scope.values())
