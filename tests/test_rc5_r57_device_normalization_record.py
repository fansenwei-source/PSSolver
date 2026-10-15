"""Content-addressed R5-7 execution-device normalization closure record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r57_record_declares_cpu_closure_without_scope_expansion():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_R5_7_EXECUTION_DEVICE_NORMALIZATION_CPU_CLOSURE"
    )
    assert record["git_identity"] == {
        "branch": "fix/v0.2.0rc5",
        "implementation_commit": "2d67ba27de7e55c39382cbd3eadcfbf0541e3ba5",
        "qualified_source_commit": "2d67ba27de7e55c39382cbd3eadcfbf0541e3ba5",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }


def test_r57_record_binds_implementation_and_regression_sources():
    record = _record()
    r5_8_successors = json.loads(R5_8_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_9_successors = json.loads(R5_9_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_10_successors = json.loads(R5_10_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_11_successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            resolved = expected
            r5_8_registration = r5_8_successors.get(relative)
            if (
                r5_8_registration is not None
                and r5_8_registration["historical_sha256"] == resolved
            ):
                resolved = r5_8_registration["current_sha256"]
            r5_9_registration = r5_9_successors.get(relative)
            if (
                r5_9_registration is not None
                and r5_9_registration["historical_sha256"] == resolved
            ):
                resolved = r5_9_registration["current_sha256"]
            r5_10_registration = r5_10_successors.get(relative)
            if (
                r5_10_registration is not None
                and r5_10_registration["historical_sha256"] == resolved
            ):
                resolved = r5_10_registration["current_sha256"]
            r5_11_registration = r5_11_successors.get(relative)
            if (
                r5_11_registration is not None
                and r5_11_registration["historical_sha256"] == resolved
            ):
                resolved = r5_11_registration["current_sha256"]
            assert _sha256(ROOT / relative) == resolved


def test_r57_record_binds_every_historical_successor():
    successors = _record()["historical_successor_sources"]
    r5_8_successors = json.loads(R5_8_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_9_successors = json.loads(R5_9_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_10_successors = json.loads(R5_10_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_11_successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert len(successors) == 35
    for relative, registration in successors.items():
        resolved = registration["current_sha256"]
        r5_8_registration = r5_8_successors.get(relative)
        if (
            r5_8_registration is not None
            and r5_8_registration["historical_sha256"] == resolved
        ):
            resolved = r5_8_registration["current_sha256"]
        r5_9_registration = r5_9_successors.get(relative)
        if (
            r5_9_registration is not None
            and r5_9_registration["historical_sha256"] == resolved
        ):
            resolved = r5_9_registration["current_sha256"]
        r5_10_registration = r5_10_successors.get(relative)
        if (
            r5_10_registration is not None
            and r5_10_registration["historical_sha256"] == resolved
        ):
            resolved = r5_10_registration["current_sha256"]
        r5_11_registration = r5_11_successors.get(relative)
        if (
            r5_11_registration is not None
            and r5_11_registration["historical_sha256"] == resolved
        ):
            resolved = r5_11_registration["current_sha256"]
        assert _sha256(ROOT / relative) == resolved
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["reason"]


def test_r57_record_binds_the_device_contract():
    contract = _record()["contract"]
    assert contract["canonical_request_tokens"] == [
        "auto",
        "cpu",
        "cuda",
        "cuda:N",
    ]
    assert contract["shared_tensor_free_grammar"] is True
    assert contract["functional_auto_matches_public_compiler"] is True
    assert contract["functional_tensor_identity_requires_concrete_device"] is True
    assert contract["runtime_construction_precedes_output_publication"] is True
    assert contract["cuda_unavailable_leaves_output_directory_absent"] is True
    assert contract["output_metadata_records_requested_and_resolved_device"] is True
    assert contract["checkpoint_identity_changed"] is False
    assert contract["runtime_scientific_identity_changed"] is False


def test_r57_record_binds_installed_wheel_artifacts_and_identity():
    evidence = _record()["installed_wheel_evidence"]
    directory = Path(evidence["directory"])
    assert _sha256(directory / evidence["source_archive"]) == (
        evidence["source_archive_sha256"]
    )
    assert _sha256(directory / "package/dist" / evidence["wheel"]) == (
        evidence["wheel_sha256"]
    )
    imported = Path(evidence["pssolver_import"])
    assert imported.is_file()
    assert imported.is_relative_to(directory / "venv")
    assert evidence["pip_check"] == "PASS"
    assert evidence["source_shadow_import"] is False


def test_r57_record_binds_source_and_installed_cpu_results():
    validation = _record()["validation"]
    assert validation["r5_7_regression_cpu"] == "23 passed, 0 failed"
    assert validation["historical_successor_cpu"] == "205 passed, 0 failed"
    assert validation["source_full_cpu"] == (
        "3546 passed, 7 skipped, 0 failed"
    )
    assert validation["installed_wheel_targeted_cpu"] == (
        "23 passed, 0 failed"
    )
    assert validation["installed_wheel_applicable_full_cpu"] == (
        "3539 passed, 7 skipped, 0 failed"
    )
    exclusion = validation["installed_wheel_source_only_exclusion"]
    assert exclusion["path"] == "tests/test_phase8_p861_local_cumulative_audit.py"
    assert exclusion["test_count"] == 7


def test_r57_record_authorizes_only_the_next_work_item():
    record = _record()
    assert record["scope"] == {
        "scientific_equations_changed": False,
        "numerical_algorithms_changed": False,
        "checkpoint_schema_changed": False,
        "runtime_scientific_identity_changed": False,
        "production_default_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "H100_required": False,
        "H100_executed": False,
        "push_executed": False,
    }
    assert record["authorization"] == {
        "r5_7_cpu_closure_complete": True,
        "eligible_for_next_work_item": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
