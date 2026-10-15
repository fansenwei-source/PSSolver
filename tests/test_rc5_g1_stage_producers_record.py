"""Evidence contract for the rc5 G-1 versioned stage producers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_stage_producers.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_stage_producer_record_binds_all_five_missing_stages_and_hashes():
    record = _record()
    assert record["classification"] == (
        "READY_RC5_G1_NINE_STAGE_PRODUCER_SET_FOR_HPCC_SUPPORT_REGENERATION"
    )
    assert record["qualification_complete"] is False
    producers = record["producer_contract"]["new_producers"]
    assert set(producers) == {
        "environment", "cuda_only", "checkpoints", "numerical", "consumer"
    }
    for value in producers.values():
        path = ROOT / value["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"]


def test_checkpoint_and_numerical_contracts_remain_fail_closed():
    record = _record()
    checkpoint = record["checkpoint_contract"]
    assert len(checkpoint["families"]) == 4
    assert checkpoint["formal_directions"] == ["cpu_to_cuda", "cuda_to_cpu"]
    assert checkpoint["local_cpu_to_cpu_smoke_passed"] == 4
    assert checkpoint["payload_tamper_rejects_before_mutation"] is True
    assert checkpoint["identity_mismatch_rejects_before_mutation"] is True
    numerical = record["numerical_contract"]
    assert numerical["runtime_family_count"] == 8
    assert numerical["local_cuda_two_step_smoke_passed"] == 8
    assert set(numerical["candidate_periodic_hermitian_violation"].values()) == {0.0}


def test_stage_producer_record_preserves_authorization_boundary():
    record = _record()
    assert not any(record["authorization"].values())
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert record["next_gate"].startswith("regenerate_source_free_hpcc_support")
