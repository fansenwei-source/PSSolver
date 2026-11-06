"""Forward-bound S-23 support, without pinning any later registry state."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]


def test_s23_sources_and_previous_engineering_policy_resolve_forward():
    record = json.loads(
        (ROOT / "notes/PSSolver_v0_3_0_s03_successor_137_s23_baselines.json").read_text()
    )
    for path, historical in record["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, historical
        )
    for path, entry in record["historical_successor_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, entry["historical_sha256"]
        )
    assert record["review_order"] == ["R-0071 / S-22", "S-23 independently"]
    assert not any(record["scope"].values())


def test_current_baseline_table_preserves_all_three_runs_and_full_matrix():
    table = json.loads((ROOT / "tests/baselines/v0_4/trajectories.json").read_text())
    assert table["three_way_raw_equality_including_identities"] is True
    assert len(table["cases"]) == 123
    assert sum(row["disposition"] == "accepted" for row in table["cases"].values()) == 104
    assert all(
        row["pre_rhs_field_bytes_equal"] and row["steps"] == 64 for row in table["p5"].values()
    )
