"""Static closure record for P9.3 replay and durable-state conversion."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p93_replay_checkpoint_bridge.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p93_record_closes_cpu_replay_and_one_bidirectional_bridge():
    record = _record()

    assert record["phase"] == "P9.3"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_3_CPU_REPLAY_AND_PERIODIC_STATE_BRIDGE"
    )
    assert record["baseline"]["commit"] == (
        "b61e14c9e7e1e67c0afbfc6dfae882a740794005"
    )
    assert record["replay"]["same_fixed_execution_identity"] == "bitwise"
    assert record["replay"]["cuda_replay_qualified"] is False
    assert record["checkpoint_bridge"]["one_shared_periodic_codec"] is True
    assert record["checkpoint_bridge"][
        "functional_export_loadable_by_production"
    ] is True
    assert record["checkpoint_bridge"][
        "production_export_loadable_by_functional"
    ] is True


def test_p93_record_freezes_fail_closed_import_and_nonclaims():
    record = _record()

    assert "functional_runtime_identity" in record["preflight"][
        "before_payload_open"
    ]
    assert "functional_state_layout" in record["preflight"][
        "before_payload_open"
    ]
    assert "sha256" in record["preflight"]["payload_validation"]
    assert record["preflight"]["implicit_dtype_conversion"] is False
    assert record["capabilities"]["deterministic_replay_cpu"] == "bitwise"
    assert record["capabilities"]["durable_checkpoint_bridge"] is True
    assert record["authorization"]["p9_4_implementation_authorized"] is False
    assert record["authorization"]["h100_authorized"] is False


def test_future_archive_lists_p93_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    p92 = '"phase_9_p92_periodic_functional_runtime.json"'
    p93_md = '"phase_9_p93_replay_checkpoint_bridge.md"'
    p93_json = '"phase_9_p93_replay_checkpoint_bridge.json"'

    assert source.count(p93_md) == 1
    assert source.count(p93_json) == 1
    assert source.index(p92) < source.index(p93_md) < source.index(p93_json)
    assert _record()["qualification"]["verbatim_pdf_regenerated"] is False
