"""Validate the rc3 symbol-level PSSolver-Control compatibility record."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = (
    ROOT / "notes/PSSolver_v0_2_0rc3_constitutive_compatibility.json"
)

EXPECTED_SYMBOLS = {
    "beris_edwards_active_stress_components",
    "beris_edwards_algebraic_stress_components",
    "beris_edwards_bulk_molecular_field_components",
    "beris_edwards_distortion_stress_components",
    "beris_edwards_molecular_field_components",
    "beris_edwards_q_nonlinear_components",
}


def _record() -> dict:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _top_level_function_hashes(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    return {
        node.name: hashlib.sha256(
            "".join(lines[node.lineno - 1 : node.end_lineno]).encode("utf-8")
        ).hexdigest()
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_record_is_symbol_scoped_and_fail_closed():
    record = _record()

    assert record["status"] == "compatible_for_listed_consumer_symbols"
    assert record["provider"]["baseline_commit"] == (
        "d878abd3caecffb99f2cbf9255963a42f401af11"
    )
    assert record["provider"]["target_commit"] == (
        "bb4489075301176c78772b07d14021432d4ad2b1"
    )
    assert record["files"]["beris_edwards"][
        "whole_file_byte_identical"
    ] is False
    assert record["files"]["q_tensor"]["whole_file_byte_identical"] is True
    assert record["authorization"][
        "consumer_planar_pin_update_may_proceed"
    ] is True
    assert (
        record["authorization"]["consumer_push_authorized_by_this_record"]
        is False
    )
    assert record["authorization"]["production_default_changed"] is False
    assert record["authorization"]["nematics3d_modified"] is False


def test_recorded_target_symbol_hashes_match_the_current_provider_source():
    record = _record()
    source_path = ROOT / record["files"]["beris_edwards"]["path"]
    observed = _top_level_function_hashes(source_path)
    symbols = record["symbol_hash_contract"]["symbols"]

    assert {item["provider_name"] for item in symbols} == EXPECTED_SYMBOLS
    for item in symbols:
        assert item["baseline_sha256"] == item["target_sha256"]
        assert observed[item["provider_name"]] == item["target_sha256"]


def test_recorded_target_file_hashes_match_the_current_provider_source():
    record = _record()

    for key in ("beris_edwards", "q_tensor"):
        entry = record["files"][key]
        payload = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == entry["target_file_sha256"]

    assert record["files"]["q_tensor"]["baseline_git_blob"] == (
        record["files"]["q_tensor"]["target_git_blob"]
    )
    assert record["files"]["q_tensor"]["baseline_file_sha256"] == (
        record["files"]["q_tensor"]["target_file_sha256"]
    )


def test_only_the_provider_runtime_adapter_is_classified_as_changed():
    record = _record()
    change = record["classified_provider_change"]

    assert change["commit"] == "5e1ef3ed6cf5954e723f10ae923de5873d010253"
    assert change["changed_top_level_definition"] == (
        "BerisEdwardsQNonlinearModel"
    )
    assert change["listed_consumer_symbols_changed"] is False
    assert change["baseline_definition_sha256"] != (
        change["target_definition_sha256"]
    )
