"""Fail-closed record checks for the G-1 split-host historical audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_historical_audit_portability_v17.json"
)
RC6_R1 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
RC6_R4 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
RC6_R5 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
RC6_R6 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
RC6_FREEZE = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1 = ROOT / "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
V021_RELEASE = ROOT / "notes/pssolver_v0_2_1.json"
V022_V17 = ROOT / "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
V022_RELEASE = ROOT / "notes/pssolver_v0_2_2.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_historical_audit_record_binds_only_forward_test_successors():
    record = _record()
    assert record["classification"] == (
        "PASS_RC5_G1_HISTORICAL_AUDIT_PORTABILITY_RECOVERY"
    )
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
    }
    rc6_successors = json.loads(RC6_R1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r2_successors = json.loads(RC6_R2.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r3_successors = json.loads(RC6_R3.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r4_successors = json.loads(RC6_R4.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r5_successors = json.loads(RC6_R5.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r6_successors = json.loads(RC6_R6.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_freeze_successors = json.loads(RC6_FREEZE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    final_release_successors = json.loads(
        FINAL_RELEASE.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    n1_successors = json.loads(V021_N1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    release_successors = json.loads(V021_RELEASE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    v17_successors = json.loads(V022_V17.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    v022_release_successors = json.loads(
        V022_RELEASE.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        resolved = registration["current_sha256"]
        successor = rc6_successors.get(relative)
        accepted = (
            []
            if successor is None
            else successor.get(
                "accepted_predecessor_sha256",
                [successor["historical_sha256"]],
            )
        )
        if successor is not None and resolved in accepted:
            resolved = successor["current_sha256"]
        successor = rc6_r2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r4_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r5_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r6_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_freeze_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = final_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = n1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = v17_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = v022_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(relative) == resolved
        assert registration["production_runtime_changed"] is False
        assert registration["scientific_threshold_changed"] is False


def test_historical_audit_record_preserves_split_host_and_scope_boundaries():
    record = _record()
    split = record["split_host_artifacts"]
    assert split["r5_6"]["verified_on_archive_host"] is True
    assert split["r5_7"]["verified_on_archive_host"] is True
    assert split["copied_to_hpcc"] is False
    assert split["synthetic_hpcc_paths_created"] is False
    assert record["validation"]["corrected_local_record_tests"] == (
        "19 passed, 0 failed"
    )
    assert record["validation"]["local_split_host_artifact_tests"] == (
        "2 passed, 0 failed"
    )
    assert record["scope"] == {
        "production_source_modified": False,
        "runtime_modified": False,
        "scientific_equations_modified": False,
        "thresholds_modified": False,
        "production_default_changed": False,
        "new_h100_submission_count": 0,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
    }
