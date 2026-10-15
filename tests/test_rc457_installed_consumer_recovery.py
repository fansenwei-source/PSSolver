"""RC4.5.7 installed-wheel Consumer identity-schema recovery record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
)
INITIAL = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
)
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RC458 = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
RELEASE = ROOT / "notes/pssolver_v0_2_0_rc4.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc457_recovery_freezes_commits_artifacts_and_installed_results():
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_RC4_5_7_INSTALLED_WHEEL_CPU_CONSUMER_CLOSURE_"
        "WITH_TF32_IDENTITY_MIGRATION"
    )
    assert record["qualification_complete"] is True
    assert record["baseline"]["provider_commit"] == (
        "a195bd05992bc6553c72c400587d2b07afe722d0"
    )
    assert record["consumer_migration"]["commits"] == [
        "de669fa97c523405ea8ba86cf4b27859c80fd258",
        "b8539e0ec4c137a6c13746f6832e26e74be24515",
    ]
    assert record["consumer_migration"]["base_commit"] == (
        "c7e7f8631d94c98bd6fa92192e0b806ac25c912c"
    )
    gate = record["installed_wheel_gate"]
    assert gate["provider"] == {"passed": 6, "failed": 0, "skipped": 0}
    assert gate["consumer"] == {
        "passed": 310,
        "failed": 0,
        "skipped": 1,
        "xfailed": 0,
    }
    assert gate["pip_check"] == "PASS"
    assert gate["source_shadow_import"] is False


def test_rc457_recovery_binds_the_exact_tf32_identity_without_api_bumps():
    record = _read(RECORD)
    assert record["identity"]["tf32_policy"] == {
        "tf32_requested": "off",
        "tf32_effective": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    assert record["identity"]["provider_identity_sha256"] == (
        "f83d7b368bda23aacdf1c70e77de974df1c05a3e8b0c7886191dd82c14ade416"
    )
    scope = record["scope"]
    assert scope["functional_api_version_changed"] is False
    assert scope["adapter_version_changed"] is False
    assert scope["control_algorithm_changed"] is False
    assert scope["production_default_changed"] is False
    assert scope["H100_executed"] is False


def test_rc457_recovery_preserves_initial_failure_and_binds_successors():
    record = _read(RECORD)
    closure_successors = _read(RC458)["historical_successor_sources"]
    release_successors = _read(RELEASE)["historical_successor_sources"]
    post_release_successors = _read(POST_RELEASE)["historical_successor_sources"]
    rc5_w1_successors = _read(RC5_W1)["historical_successor_sources"]
    rc5_r51_successors = _read(RC5_R51)["historical_successor_sources"]
    rc5_r52_successors = _read(RC5_R52)["historical_successor_sources"]
    rc5_r53_successors = _read(RC5_R53)["historical_successor_sources"]
    rc5_r55_successors = _read(RC5_R55)["historical_successor_sources"]
    rc5_r56_successors = _read(RC5_R56)["historical_successor_sources"]
    rc5_r57_successors = _read(RC5_R57)["historical_successor_sources"]
    rc5_r58_successors = _read(RC5_R58)["historical_successor_sources"]
    assert _sha256(INITIAL) == record["baseline"]["initial_record_sha256"]
    assert _read(INITIAL)["qualification_complete"] is False
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        resolved = registration["current_sha256"]
        successor = closure_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = post_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_w1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r51_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r52_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r53_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r55_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r56_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r57_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r58_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolved
    for relative, expected in record["source_sha256"].items():
        resolved = expected
        successor = closure_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = post_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_w1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r51_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r52_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r53_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r55_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r56_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r57_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r58_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolved


def test_rc457_recovery_authorizes_only_the_rc458_h100_closure():
    record = _read(RECORD)
    authorization = record["authorization"]
    assert authorization["rc4_5_7_complete"] is True
    assert authorization["eligible_for_rc4_5_8_h100_closure"] is True
    assert authorization["rc4_5_8_h100_authorized"] is False
    assert authorization["push_authorized"] is False
    assert authorization["eligible_for_automatic_merge"] is False
    assert authorization["eligible_for_default_promotion"] is False
    assert authorization["production_default_changed"] is False

    progress = _read(MATRIX)["progress"]
    assert progress["rc4_5_7_record_sha256"] == _sha256(RECORD)
    assert progress["rc4_5_7_initial_record_sha256"] == _sha256(INITIAL)
    assert progress["eligible_for_rc4_5_8_h100_closure"] is True
