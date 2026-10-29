from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_v0_3_s09_cpu_6x60_stop import (
    EXPECTED_SELF_SHA256,
    _read_bound,
)
from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration


def test_s09_stop_analysis_preserves_failure_and_uses_no_new_profiles():
    entry = assert_pytest_migration(
        "tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::"
        "test_s09_stop_analysis_preserves_failure_and_uses_no_new_profiles"
    )
    assert entry["semantic_check_ids"] == ["S09_STOP_ANALYSIS"]


def test_s09_stop_analysis_separates_transient_from_iteration_count():
    entry = assert_pytest_migration(
        "tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::"
        "test_s09_stop_analysis_separates_transient_from_iteration_count"
    )
    assert entry["semantic_check_ids"] == ["S09_STOP_ANALYSIS"]


def test_s09_stop_analysis_does_not_overclaim_unique_cause():
    entry = assert_pytest_migration(
        "tests/test_analyze_v0_3_s09_cpu_6x60_stop.py::"
        "test_s09_stop_analysis_does_not_overclaim_unique_cause"
    )
    assert entry["semantic_check_ids"] == ["S09_STOP_ANALYSIS"]


def test_s09_stop_analysis_rejects_unbound_input(tmp_path):
    copied = tmp_path / "matrix.json"
    copied.write_text(json.dumps({"not": "the matrix"}), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash mismatch"):
        _read_bound(copied, EXPECTED_SELF_SHA256)
