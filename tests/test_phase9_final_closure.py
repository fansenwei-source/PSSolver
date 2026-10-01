"""Static contract for the P9.8.5 evidence and P9.8.6 final record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _record(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def test_p985_binds_the_qualified_provider_consumer_and_evidence_chain():
    record = _record("phase_9_p985_h100_closure.json")
    assert record["classification"] == (
        "PASS_P9_8_5_STABLE_FUNCTIONAL_API_CUMULATIVE_CLOSURE_"
        "WITH_CANONICAL_ZERO_START_CHANNEL_ORACLE"
    )
    assert record["provider"]["qualification_commit"] == (
        "0838ecd1cd314a5e8879ce6d1a1ced921d0cf69f"
    )
    assert record["consumer"]["qualification_commit"] == (
        "28857a59610df355469dca55372e0d0cd1311f3c"
    )
    assert [(item["manifest_entries"], item["manifest_passed"]) for item in record["evidence"]] == [
        (36, 36),
        (230, 230),
        (29, 29),
        (72, 72),
        (30, 30),
    ]
    assert all(len(item["manifest_sha256"]) == 64 for item in record["evidence"])
    assert record["evidence"][-1]["complete_marker"] is True


def test_p985_freezes_periodic_and_channel_numerical_semantics():
    record = _record("phase_9_p985_h100_closure.json")
    assert record["periodic_component"]["maximum_hermitian_violation"] == 0.0
    channel = record["channel_component"]
    assert set(channel["gates"].values()) == {"pass"}
    assert channel["canonical_zero_start_oracle"][
        "one_step_q_u_p_byte_identical"
    ] is True
    assert channel["canonical_zero_start_oracle"][
        "four_step_q_u_p_byte_identical"
    ] is True
    assert channel["production_warm_start"]["preserved"] is True
    assert channel["production_warm_start"][
        "functional_hidden_state_extended"
    ] is False
    assert channel["production_warm_start"][
        "all_frozen_numerical_bounds_passed"
    ] is True
    checkpoint = record["checkpoint_integrity"]
    assert checkpoint["schema_helper_dynamic"] is True
    assert checkpoint["raw_byte_tamper_rejected"] is True
    assert checkpoint["target_mutated"] is False
    assert checkpoint["timestep_started"] is False


def test_phase9_final_record_freezes_public_scope_without_promoting_defaults():
    record = _record("phase_9_final_closure.json")
    assert record["classification"] == (
        "PASS_PHASE_9_STABLE_FUNCTIONAL_API_FINAL_CLOSURE"
    )
    contract = record["public_contract"]
    assert contract["module"] == "pssolver.functional.api"
    assert contract["functional_api_version"] == "1.0"
    assert contract["qualified_runtime_kinds"] == [
        "periodic_activity_batch_one",
        "channel_activity_batch_one",
    ]
    assert contract["qualified_batch_sizes"] == [1]
    assert contract["silent_fallback"] is False
    assert contract["heuristic_checkpoint_repair"] is False
    assert contract["private_provider_imports"] is False
    assert record["numerical_semantics"]["production_default_changed"] is False
    assert record["release_decision"]["package_version_changed"] is False
    assert record["release_decision"]["tag_created"] is False
    assert record["release_decision"]["wheel_published"] is False


def test_phase9_is_complete_without_overclaiming_control_science():
    record = _record("phase_9_final_closure.json")
    authorization = record["authorization"]
    assert authorization["p9_8_6_complete"] is True
    assert authorization["phase_9_complete"] is True
    assert authorization["eligible_for_post_phase9_planning"] is True
    assert authorization["release_candidate_preparation_eligible"] is True
    assert authorization["release_performed"] is False
    assert authorization["larger_batch_qualified"] is False
    assert authorization["optimizer_campaign_authorized"] is False
    assert authorization["scientific_control_result"] is False
    assert authorization["production_default_changed"] is False


def test_future_archive_tracks_final_records_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "phase_9_p985_h100_closure.md",
        "phase_9_p985_h100_closure.json",
        "phase_9_final_closure.md",
        "phase_9_final_closure.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
    assert _record("phase_9_final_closure.json")["boundaries"][
        "verbatim_pdf_regenerated"
    ] is False
