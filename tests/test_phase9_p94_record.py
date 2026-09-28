"""Static closure record for P9.4 CPU gradient and R12 validation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p94_gradient_consistency.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p94_record_closes_only_the_cpu_gradient_and_r12_slice():
    record = _record()

    assert record["phase"] == "P9.4"
    assert record["status"] == "complete_for_cpu_identity"
    assert record["classification"] == (
        "PASS_P9_4_CPU_GRADIENT_AND_R12_VALIDATION"
    )
    assert record["baseline"]["commit"] == "3bc3841"
    assert record["scope"]["qualified_device_type"] == "cpu"
    assert record["qualification"]["cuda_qualified"] is False
    assert record["authorization"]["p9_5_implementation_authorized"] is False


def test_p94_record_freezes_detach_stride_and_r12_contracts():
    record = _record()

    assert record["gradient_validation"]["state_detach_negative_gate_passed"]
    assert record["gradient_validation"]["control_detach_negative_gate_passed"]
    assert record["checkpoint_stride"]["tested_strides"] == [1, 2, 3]
    assert record["checkpoint_stride"]["activity_gradients"] == (
        "bitwise_identical"
    )
    assert record["r12"]["tolerance_override_available"] is False
    assert record["r12"]["deliberate_mismatch_rejected"] is True
    assert record["r12"]["formal_contract"] == "strict_predeclared_tolerance"


def test_future_archive_lists_p94_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    p93 = '"phase_9_p93_replay_checkpoint_bridge.json"'
    p94_md = '"phase_9_p94_gradient_consistency.md"'
    p94_json = '"phase_9_p94_gradient_consistency.json"'

    assert source.count(p94_md) == 1
    assert source.count(p94_json) == 1
    assert source.index(p93) < source.index(p94_md) < source.index(p94_json)
    assert _record()["qualification"]["verbatim_pdf_regenerated"] is False
