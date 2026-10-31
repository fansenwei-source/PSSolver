"""Exact cross-repository identity boundary for S-17 records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_85_s17_external_identity_correction.json"
)


def test_s17_external_identities_are_exact_and_field_scoped() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["successor"] == 85
    assert record["external_git_identity_fields"] == {
        "PSSolver_v0_3_0_s03_successor_83_r0048_import.json:reviewed_control_candidate": {
            "repository": "PSSolver-Control",
            "commit": "39e7bdcacbb7a33733c17da9fc2809cfc2feba2f",
            "reason": "R-0048 reviewed the Control migration candidate in the separate Consumer repository.",
        },
        "PSSolver_v0_3_0_s03_successor_84_s17_r0048_rework.json:control_v56_fix": {
            "repository": "PSSolver-Control",
            "commit": "215e35c5504a305d20a28ad3987fd5ed2f94683a",
            "reason": "V-56 is fixed by the reachable-history mapping in the Consumer repository.",
        },
        "PSSolver_v0_3_0_s03_successor_84_s17_r0048_rework.json:control_v57_record_fix": {
            "repository": "PSSolver-Control",
            "commit": "140bd273693c341545ceb023bfd2de2e36458e07",
            "reason": "V-57 is fixed by the rebuilt-history record in the Consumer repository.",
        },
    }
    assert record["guard_policy"] == {
        "match_record_path_field_and_commit": True,
        "generic_missing_object_exception": False,
        "other_pssolver_identities_must_resolve": True,
    }


def test_s17_external_identity_correction_changes_no_runtime_or_action() -> None:
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert all(value is False for value in record["scope"].values())
