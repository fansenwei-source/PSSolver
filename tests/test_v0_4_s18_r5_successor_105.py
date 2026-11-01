from __future__ import annotations

import hashlib
import json
from pathlib import Path

from v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_105_s18_r5_frozen_proxy_test.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_successor_105_forwards_only_the_historical_proxy_test() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 105
    assert set(record["historical_successor_sources"]) == {
        "tests/test_v0_3_s09_ap_cache_diagnostic.py"
    }
    for relative, binding in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative,
            binding["historical_sha256"],
        ) == binding["current_sha256"]
        assert _sha256(relative) == binding["current_sha256"]
    assert record["reason"]["frozen_benchmark_changed"] is False
    assert record["reason"]["skip_or_xfail_added"] is False
    assert record["reason"]["runtime_changed"] is False


def test_successor_105_grants_no_remote_or_gpu_authority() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert not any(record["scope"].values())
    assert record["authorization"] == {
        "local_test_compatibility": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
