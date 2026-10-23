from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_32_s09_api11_compatibility.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_s09_api11_record_binds_runtime_and_frozen_legacy_contract() -> None:
    record = _record()
    assert record["classification"] == (
        "PASS_S09_API_1_1_LEGACY_CHECKPOINT_COMPATIBILITY_REPAIR"
    )
    assert record["git_identity"]["runtime_commit"] == (
        "dc12dc6fbf984fb1a98e1de672903e51750742f3"
    )
    assert record["contract"] == {
        "current_functional_api_version": "1.1",
        "released_checkpoint_api_version": "1.0",
        "released_identity_schema_frozen": True,
        "released_migration_provenance_frozen": True,
        "api_1_0_checkpoint_reader_retained": True,
        "historical_records_rewritten": False,
    }


def test_s09_api11_record_registers_exact_forward_successors() -> None:
    successors = _record()["historical_successor_sources"]
    assert len(successors) == 15
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["reason"]
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == resolve_s03_successor(
            relative, registration["current_sha256"]
        )


def test_s09_api11_record_does_not_authorize_external_actions() -> None:
    authorization = _record()["authorization"]
    assert not any(authorization.values())
