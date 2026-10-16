"""Evidence contract for the rc5 G-1.8 HPCC plan freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_hpcc_plan_freeze.json"
SUCCESSOR = ROOT / "notes/PSSolver_v0_2_0rc5_g1_consumer_recovery_v5.json"
WINDOW_RECOVERY = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_transform_window_recovery_v6.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _assert_historical_or_successor(relative: str, historical: str) -> None:
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    if actual == historical:
        return
    recovered = json.loads(SUCCESSOR.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    predecessor = historical
    if recovered is not None and recovered["historical_sha256"] == historical:
        predecessor = recovered["sha256"]
        if actual == predecessor:
            return
    successor = json.loads(WINDOW_RECOVERY.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ][relative]
    assert predecessor in successor["accepted_predecessor_sha256"]
    assert successor["current_sha256"] == actual


def test_g1_hpcc_plan_freeze_binds_the_exact_nine_stage_order():
    record = _record()
    assert record["phase"] == "G-1.8"
    assert record["classification"] == (
        "READY_RC5_G1_FROZEN_NINE_STAGE_HPCC_PLAN_AWAITING_PUSH_AND_H100_AUTHORIZATION"
    )
    assert record["qualification_complete"] is False
    plan = record["frozen_stage_plan"]
    assert plan["stage_count"] == 9
    assert plan["stage_order"] == [
        "environment",
        "cuda_only",
        "tf32_scope",
        "checkpoints",
        "numerical",
        "profiles",
        "consumer",
        "assemble",
        "analyze",
    ]
    assert plan["each_stage_has_one_hash_bound_python_and_script"] is True
    assert plan["inline_python_commands"] is False
    assert plan["shell_command_strings"] is False
    assert plan["preflight_replay"] == "PASS"
    assert plan["direct_help_import_smoke"] == "9/9 PASS"


def test_g1_hpcc_plan_freeze_hashes_versioned_command_files():
    record = _record()
    for relative, expected in record["command_file_sha256"].items():
        _assert_historical_or_successor(relative, expected)


def test_g1_hpcc_plan_freeze_binds_the_96_profile_matrix():
    profile = _record()["profile_command_plan"]
    assert profile["profile_count"] == 96
    assert profile["unique_keys"] == 96
    assert profile["sequence"] == "1-96"
    assert profile["device_cuda"] == 96
    assert profile["device_cuda_0"] == 0
    assert profile["preexisting_profile_outputs"] == 0


def test_g1_hpcc_plan_freeze_preserves_the_authorization_boundary():
    record = _record()
    authorization = record["authorization"]
    assert not any(authorization.values())
    assert record["hpcc"]["planned_h100_output_root_absent_after_final_preflight"] is True
    assert record["support_archive"]["source_free"] is True
    assert record["support_archive"]["contains_pssolver_package_tree"] is False
    assert record["next_gate"].startswith("obtain_explicit_push_authorization")


def test_g1_hpcc_plan_freeze_does_not_rewrite_failed_predecessor_record():
    predecessor = _record()["predecessor_records"]
    for value in predecessor.values():
        actual = hashlib.sha256((ROOT / value["path"]).read_bytes()).hexdigest()
        assert actual == value["sha256"]
    assert predecessor["hpcc_preflight"]["historical_failure_classification_preserved"] is True
