"""S-25's live-tree guard, separate from S-24's frozen endpoint audit."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
BASE = "aee82899d091b0d3bbfa62f0fe1cf83817574285"


def test_forward_record_binds_changed_sources_without_historical_rewrite():
    record = json.loads(
        (ROOT / "notes/PSSolver_v0_3_0_s03_successor_143_s25_engine.json").read_text()
    )
    assert record["historical_records_rewritten"] is False
    retired = {"pssolver/planning/periodic.py", "pssolver/engine/resolve.py"}
    for path, expected in record["introduced_sources"].items():
        if path in retired:
            continue
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, expected
        )
