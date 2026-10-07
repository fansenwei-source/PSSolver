from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc3_known_issues_matrix.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_matrix_binds_the_released_rc3_baseline_without_claiming_runtime_changes():
    record = _record()
    assert record["schema_version"] == 1
    assert record["baseline"] == {
        "release": "v0.2.0rc3",
        "commit": "5071d73a00e0d19be62ebd39edf9918818d1267a",
        "branch": "fix/v0.2.0rc4-audit",
        "matrix_changes_runtime": False,
        "released_tag_is_immutable": True,
    }
    assert record["release_boundary"] == {
        "rc3_declared_scope_qualified": True,
        "all_audit_items_closed": False,
        "eligible_for_final_v0_2_0": False,
        "new_features_authorized": False,
        "production_default_changed": False,
        "pssolver_control_change_required_for_this_matrix": False,
        "nematics3d_modified": False,
    }


def test_matrix_binds_both_external_audit_documents_by_digest():
    assert _record()["source_audits"] == [
        {
            "name": "PSSolver_v0_2_0rc1_bug_redundancy_audit_zh.pdf",
            "sha256": "a3a6edbc7c09ab09760230a9aaa22d0ffa552bae272fd48e1787123f54d9d4d1",
        },
        {
            "name": "PSSolver_v0_2_0rc2_fix_verification_zh.pdf",
            "sha256": "140986b3f6f20b0c0c874b21df02b037cf8ee1bb32e43dd11bc2d638cd9b8f76",
        },
    ]


def test_matrix_has_each_audit_identifier_exactly_once():
    items = _record()["items"]
    identifiers = [item["id"] for item in items]
    expected = (
        [f"B{index}" for index in range(1, 31)]
        + [f"A{index}" for index in range(1, 6)]
        + [f"N{index}" for index in range(1, 7)]
        + [f"R{index}" for index in range(1, 10)]
    )
    assert identifiers == expected
    assert len(set(identifiers)) == 50


def test_state_counts_match_the_machine_readable_summary():
    record = _record()
    counts: dict[str, int] = {}
    for item in record["items"]:
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    assert counts == {
        "closed_verified": 18,
        "partial_residual": 20,
        "accepted_limitation": 3,
        "deferred_architecture": 9,
    }
    assert record["summary"] == {
        "item_count": 50,
        **counts,
        "first_implementation_batch": "rc4.1_plane_nyquist",
        "next_implementation_batch": "rc4.2_checkpoint_identity",
    }


def test_release_blockers_and_closed_rc3_regressions_are_not_conflated():
    items = {item["id"]: item for item in _record()["items"]}
    for identifier in ("B1", "B2", "B3", "B4", "N1", "N2"):
        assert items[identifier]["state"] == "closed_verified"
    assert items["B5"]["state"] == "closed_verified"
    assert items["B5"]["disposition"] == "closed"
    assert items["B5"]["closure_record"] == (
        "notes/PSSolver_v0_2_0rc4_rc412_h100_closure.json"
    )
    for identifier in ("B8", "B29", "N3", "N4", "N5", "N6"):
        assert items[identifier]["state"] == "partial_residual"
        assert items[identifier]["disposition"] != "closed"


def test_first_batch_is_only_plane_nyquist_and_has_non_regression_gates():
    record = _record()
    first = record["work_queue"][0]
    assert first["batch"] == "rc4.1_plane_nyquist"
    assert first["items"] == ["B5"]
    assert first["consumer_requalification"] is False
    assert set(first["required_gates"]) == {
        "manufactured_stokes_cpu",
        "odd_grid_byte_identity",
        "even_grid_storage_equivalence",
        "single_h100_non_regression",
    }


def test_architecture_work_is_deferred_until_correctness_batches_close():
    record = _record()
    architecture = record["work_queue"][-1]
    assert architecture["batch"] == "post_0_2_architecture"
    assert architecture["items"] == [f"R{index}" for index in range(1, 10)]
    assert all(
        item["state"] == "deferred_architecture"
        for item in record["items"]
        if item["id"].startswith("R")
    )


def test_rc41_progress_closes_only_b5_and_authorizes_only_rc42_planning():
    assert _record()["progress"] == {
        "completed_batches": ["rc4.1_plane_nyquist"],
        "next_batch": "rc4.2_checkpoint_identity",
        "b5_complete": True,
        "b5_closure_record": (
            "notes/PSSolver_v0_2_0rc4_rc412_h100_closure.json"
        ),
        "eligible_for_rc4_2_planning": True,
        "rc4_2_implementation_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
