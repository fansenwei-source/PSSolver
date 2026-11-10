"""S-25's live-tree guard, separate from S-24's frozen endpoint audit."""

import hashlib
import json
from pathlib import Path

from qualification.s24_source_audit import retained_changes
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
BASE = "aee82899d091b0d3bbfa62f0fe1cf83817574285"


def test_every_retained_test_edit_has_an_explicit_s25_disposition():
    record = json.loads((ROOT / "qualification/s25_test_disposition.json").read_text())
    actual = retained_changes(ROOT, ("tests",), base=BASE)
    expected = {(row["path"], row["unit"]) for row in record["changed_retained_units"]}
    assert {(row["path"], row["unit"]) for row in actual["semantic"]} == expected
    assert actual["format_only"] == record["format_only_units"]
    for row in record["changed_retained_units"]:
        assert row["rule"] in {"R2", "R3", "R4", "R5", "R8", "R9"}
        assert row["reason"]


def test_forward_record_binds_changed_sources_without_historical_rewrite():
    record = json.loads(
        (ROOT / "notes/PSSolver_v0_3_0_s03_successor_143_s25_engine.json").read_text()
    )
    assert record["historical_records_rewritten"] is False
    for path, expected in record["introduced_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, expected
        )
