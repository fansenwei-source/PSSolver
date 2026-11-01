"""Forward binding for the successor-98 test correction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_100_s18_r4_forward_test.json"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_100_binds_the_forward_test_correction() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 100
    relative, binding = next(iter(record["historical_successor_sources"].items()))
    assert binding == {
        "historical_sha256": "a4ba86f65c7af3bd8301eb80e8d1782e292b9ee2bbbea6c5bb21e2db96318bc9",
        "current_sha256": "9eaeeae324eb7bc3c18abace648b5145f409a894a026a53e6e51029512194dc0",
    }
    assert _sha256(relative) == resolve_s03_successor(
        relative,
        binding["historical_sha256"],
    )


def test_successor_100_is_records_only_and_has_no_remote_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_test_fix": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
