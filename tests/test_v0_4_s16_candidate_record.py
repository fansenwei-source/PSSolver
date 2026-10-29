"""Bindings for the corrected local S-16 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_71_s16_candidate.json"


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_successor_71_binds_the_reachable_candidate_and_tree_identities():
    record = _read()
    identity = record["git_identity"]
    assert identity["candidate_correction"] == (
        "7574d41f28f50e45f36b63215ba85f1d98f1f9fb"
    )
    assert subprocess.check_output(
        ["git", "rev-parse", f"{identity['candidate_correction']}^"],
        cwd=ROOT,
        text=True,
    ).strip() == identity["candidate_correction_parent"]
    for name, commit in (
        ("v0_3_0_pssolver_tree", identity["v0_3_0_release_source"]),
        ("candidate_pssolver_tree", identity["candidate_correction"]),
    ):
        assert subprocess.check_output(
            ["git", "rev-parse", f"{commit}:pssolver"], cwd=ROOT, text=True
        ).strip() == record["tree_identity"][name]


def test_successor_71_advances_every_corrected_source_without_rewriting_history():
    record = _read()
    parent = record["git_identity"]["candidate_correction_parent"]
    current = record["git_identity"]["candidate_correction"]
    for relative, successor in record["historical_successor_sources"].items():
        historical = subprocess.check_output(
            ["git", "show", f"{parent}:{relative}"], cwd=ROOT
        )
        final = subprocess.check_output(
            ["git", "show", f"{current}:{relative}"], cwd=ROOT
        )
        assert _sha256(historical) == successor["historical_sha256"]
        assert _sha256(final) == successor["current_sha256"]


def test_successor_71_preserves_scope_and_requires_independent_review():
    record = _read()
    assert record["classification"].endswith("PENDING_R0043")
    assert record["candidate_correction"]["tensor_free_declaration_restored"] is True
    assert record["candidate_correction"]["historical_records_rewritten"] is False
    assert record["numerical_equivalence"]["status"] == (
        "passed_locally_pending_independent_review"
    )
    assert record["r8"]["implementation_started"] is False
    assert record["r8"]["decision_status"] == "proposed"
    assert all(value is False for value in record["scope"].values())
    assert all(value is False for value in record["authorization"].values())


def test_successor_71_numerical_support_is_content_addressed():
    record = _read()["numerical_equivalence"]
    assert _sha256((ROOT / record["support"]).read_bytes()) == record["support_sha256"]
    assert _sha256((ROOT / record["support_test"]).read_bytes()) == record[
        "support_test_sha256"
    ]
