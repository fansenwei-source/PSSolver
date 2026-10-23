"""Bindings for the S-07 qualification-support correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_20_s07_qualification_support.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s07_support_correction_is_exact() -> None:
    record = _read(RECORD)
    assert record["git_identity"]["runtime_source"] == (
        "8aceaab446e57a13866090ec06ecdf30e33ef490"
    )
    correction = record["corrections"]
    assert correction["source_suite_collects_installed_gate"] is False
    assert correction["historical_channel_bindings_use_successor_chain"] is True
    assert correction["frozen_phase9_record_rewritten"] is False
    helper = ROOT / correction["installed_gate_current_path"]
    assert helper.is_file()
    assert _sha256(helper) == correction["installed_gate_sha256"]
    assert not (ROOT / correction["installed_gate_old_path"]).exists()


def test_s07_support_successors_are_forward_only() -> None:
    for relative, registration in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, registration["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s07_support_changes_no_runtime_or_external_state() -> None:
    record = _read(RECORD)
    assert record["corrections"]["runtime_failure_supported"] is False
    scope = record["scope"]
    assert scope["evidence_and_qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "evidence_and_qualification_support_only"
    )
