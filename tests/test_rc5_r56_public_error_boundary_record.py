"""Content-addressed R5-6 public-error-boundary closure record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
import pssolver.api
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
G1_HISTORICAL_AUDIT_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_historical_audit_portability_v17.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_R6_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
)
RC6_FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)
V021_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_1.json"
V022_V17_RECORD = ROOT / (
    "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
)
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _r57_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        R5_7_RECORD,
        R5_8_RECORD,
        R5_9_RECORD,
        R5_10_RECORD,
        R5_11_RECORD,
        R5_12_RECORD,
        R5_13_RECORD,
        G1_RECORD,
        G1_GLIBC_RECORD,
        G1_HISTORICAL_AUDIT_RECORD,
        RC6_R1_RECORD,
        RC6_R2_RECORD,
        RC6_R3_RECORD,
        RC6_R4_RECORD,
        RC6_R5_RECORD,
        RC6_R6_RECORD,
        RC6_FREEZE_RECORD,
        FINAL_RELEASE_RECORD,
        V021_N1_RECORD,
        V021_RELEASE_RECORD,
        V022_V17_RECORD,
        V022_RELEASE_RECORD,
    ):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        accepted = (
            []
            if registration is None
            else registration.get(
                "accepted_predecessor_sha256",
                [registration["historical_sha256"]],
            )
        )
        if registration is not None and resolved in accepted:
            resolved = registration["current_sha256"]
    return resolved


def test_r56_record_declares_cpu_closure_without_scope_expansion():
    record = _record()
    assert record["classification"] == (
        "PASS_R5_6_STRUCTURED_PUBLIC_ERROR_BOUNDARY_CPU_CLOSURE"
    )
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["git_identity"]["implementation_commit"] == (
        "cf572bcfee0cb397961944eace123fb8fbe25683"
    )
    assert record["git_identity"]["pushed"] is False
    assert record["scope"] == {
        "scientific_equations_changed": False,
        "numerical_algorithms_changed": False,
        "checkpoint_schema_changed": False,
        "production_default_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "H100_required": False,
        "H100_executed": False,
        "push_executed": False,
    }


def test_r56_record_binds_every_changed_implementation_and_regression_source():
    record = _record()
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _r57_successor(relative, expected)


def test_r56_successor_registrations_bind_historical_and_current_sources():
    successors = _record()["historical_successor_sources"]
    assert len(successors) == 26
    for relative, registration in successors.items():
        assert _sha256(relative) == _r57_successor(
            relative,
            registration["current_sha256"],
        )
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["reason"]


def test_r56_exports_public_error_types_from_both_public_surfaces():
    assert pssolver.PublicSimulationCompilationError is (
        PublicSimulationCompilationError
    )
    assert pssolver.api.PublicSimulationCompilationError is (
        PublicSimulationCompilationError
    )
    assert pssolver.PublicCompilationRejectionCode is (
        PublicCompilationRejectionCode
    )
    assert pssolver.api.PublicCompilationRejectionCode is (
        PublicCompilationRejectionCode
    )


def test_r56_record_binds_the_structured_rejection_matrix():
    contract = _record()["contract"]
    assert contract["reproducer_case_count"] == 58
    assert contract["compile_rejection_count"] == 48
    assert contract["structured_compile_rejection_count"] == 48
    assert contract["unstructured_compile_rejection_count"] == 0
    assert contract["pre_fix_application_contract_count"] == 45
    assert contract["post_fix_application_contract_count"] == 0
    assert contract["missing_stokes_code"] == "malformed_application_spec"
    assert contract["robin_nonfinite_context"] == "positive_infinity"
    assert contract["tensor_allocation_before_rejection"] is False
    assert contract["output_created_before_rejection"] is False


def test_r56_record_binds_installed_wheel_identity_and_cpu_results():
    evidence = _record()["installed_wheel_evidence"]
    directory = Path(evidence["directory"])
    assert hashlib.sha256(
        (directory / "package/provider_cf572bc.tar").read_bytes()
    ).hexdigest() == evidence["source_archive_sha256"]
    assert hashlib.sha256(
        (directory / "package/dist" / evidence["wheel"]).read_bytes()
    ).hexdigest() == evidence["wheel_sha256"]
    assert evidence["pip_check"] == "PASS"
    assert evidence["source_shadow_import"] is False
    assert _record()["validation"]["installed_wheel_targeted_cpu"] == (
        "41 passed, 0 failed"
    )


def test_r56_record_authorizes_only_the_next_cpu_work_item():
    assert _record()["authorization"] == {
        "r5_6_cpu_closure_complete": True,
        "eligible_for_r5_7_implementation": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
