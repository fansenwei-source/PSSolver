"""Current S-24 source bindings and historical audit disposition stay explicit."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_141_s24_retirement.json"


def test_s24_successor_binds_current_sources_without_rewriting_history():
    record = json.loads(RECORD.read_text())
    assert record["successor"] == 141
    assert record["historical_records_rewritten"] is False
    for relative, binding in record["historical_successor_sources"].items():
        if relative == "tests/test_architecture_stage_c.py":
            continue
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(
            relative, binding["historical_sha256"]
        ), relative
    for relative, expected in record["introduced_sources"].items():
        if relative == "tests/test_architecture_stage_c.py":
            continue
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(
            relative, expected
        ), relative


def test_s24_disposition_covers_all_b_consumers_and_original_test_functions():
    consumers = json.loads((ROOT / "qualification/s24_consumers.json").read_text())
    dispositions = {
        row["path"]: row
        for row in json.loads((ROOT / "qualification/s24_test_disposition.json").read_text())
    }
    historical = json.loads((ROOT / "qualification/historical_stage_tests.json").read_text())
    assert len(historical["nodeids"]) == len(set(historical["nodeids"]))
    b_tests = [row for row in consumers["group_b_consumers"] if row["path"].startswith("tests/")]
    assert len(b_tests) == 21
    for row in b_tests:
        assert row["path"] in dispositions
        assert row["coverage"] == dispositions[row["path"]]
    for row in dispositions.values():
        for name in row["historical_only_test_functions"]:
            prefix = row["path"] + "::" + name
            assert any(
                node == prefix or node.startswith(prefix + "[") for node in historical["nodeids"]
            ), prefix


def test_s24_o6_retained_modules_have_current_consumers():
    report = json.loads((ROOT / "qualification/s24_consumers.json").read_text())
    for row in report["o6_current_consumer_evidence"]:
        if row["retained"]:
            assert row["current_direct_consumers"], row["module"]
            assert all(
                path.startswith(("pssolver/", "tests/")) for path in row["current_direct_consumers"]
            )
        else:
            assert not (ROOT / (row["module"].replace(".", "/") + ".py")).exists()
