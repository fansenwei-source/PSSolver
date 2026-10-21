"""Immutable PSSolver 0.2.0rc4 release-document contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

from pssolver import capability_catalog


PROJECT_ROOT = Path(__file__).parents[1]
POST_RELEASE = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
RC5_R55 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R59 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC5_R510 = PROJECT_ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
RC5_R511 = PROJECT_ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R512 = PROJECT_ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
RC5_R513 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
RC5_G1 = PROJECT_ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
RC5_G1_GLIBC = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_R6 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
)
V021_N1 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)
V022_V17 = PROJECT_ROOT / (
    "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
)


def test_rc4_release_record_preserves_historical_package_version():
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc4.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["release"] == "0.2.0rc4"
    assert record["public_contract"]["package_version"] == "0.2.0rc4"


def test_rc4_release_documents_and_machine_record_agree():
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    scope = (PROJECT_ROOT / "notes" / "pssolver_v0_2_scope.md").read_text(
        encoding="utf-8"
    )
    release_notes = (
        PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc4.md"
    ).read_text(encoding="utf-8")
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc4.json").read_text(
            encoding="utf-8"
        )
    )

    assert "## 0.2.0rc4" in changelog
    assert "PSSolver v0.2 support scope" in scope
    assert "PSSolver 0.2.0rc4 release candidate" in release_notes
    assert record["release"] == "0.2.0rc4"
    assert record["status"] == "release_preparation_source_candidate"
    assert record["source"]["rc4_closure_registration_commit"] == (
        "bc5eb408f0d2c9e9dfcf5fb559809bd5fcdaacfd"
    )
    assert record["local_verification"]["complete_cpu_suite"] == "pending"
    assert record["local_verification"]["release_metadata_tests"] == (
        "22 passed"
    )
    assert record["local_verification"]["h100_release_smoke"] == "pending"
    assert record["artifacts"]["tag_created"] is False
    assert record["artifacts"]["pypi_published"] is False


def test_release_candidate_capability_counts_match_live_catalog():
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc4.json").read_text(
            encoding="utf-8"
        )
    )["public_contract"]
    catalog = capability_catalog()

    assert len(catalog.models) == record["models"]
    assert len(catalog.geometries) == record["geometries"]
    assert len(catalog.boundary_policies) == record["boundary_policies"]
    assert len(catalog.qualified_combinations) == record["qualified_combinations"]
    assert sum(
        len(combination.runtime_paths)
        for combination in catalog.qualified_combinations
    ) == record["runtime_paths"]


def test_release_candidate_registers_exact_portability_successors():
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc4.json").read_text(
            encoding="utf-8"
        )
    )
    successors = record["historical_successor_sources"]
    post_release_successors = json.loads(POST_RELEASE.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_w1_successors = json.loads(RC5_W1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r51_successors = json.loads(RC5_R51.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r52_successors = json.loads(RC5_R52.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r53_successors = json.loads(RC5_R53.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r55_successors = json.loads(RC5_R55.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r56_successors = json.loads(RC5_R56.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r57_successors = json.loads(RC5_R57.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r58_successors = json.loads(RC5_R58.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r59_successors = json.loads(RC5_R59.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r510_successors = json.loads(RC5_R510.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r511_successors = json.loads(RC5_R511.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r512_successors = json.loads(RC5_R512.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r513_successors = json.loads(RC5_R513.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_g1_successors = json.loads(RC5_G1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_g1_glibc_successors = json.loads(
        RC5_G1_GLIBC.read_text(encoding="utf-8")
    )["historical_successor_sources"]
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
    n1_successors = json.loads(V021_N1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    v17_successors = json.loads(V022_V17.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert set(successors) == {
        "tests/test_rc447_pressure_bounded_operator_recovery.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_cross_runtime_robustness_plan.py",
        "tests/test_rc4_pressure_adjudication.py",
        "tests/test_rc4_pressure_installed_cpu_closure_record.py",
        "tests/test_rc4_pressure_policy_implementation.py",
        "tests/test_rc4_pressure_policy_implementation_record.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        resolved = registration["current_sha256"]
        post_release = post_release_successors.get(relative)
        if (
            post_release is not None
            and post_release["historical_sha256"] == resolved
        ):
            resolved = post_release["current_sha256"]
        rc5_w1 = rc5_w1_successors.get(relative)
        if rc5_w1 is not None and rc5_w1["historical_sha256"] == resolved:
            resolved = rc5_w1["current_sha256"]
        rc5_r51 = rc5_r51_successors.get(relative)
        if rc5_r51 is not None and rc5_r51["historical_sha256"] == resolved:
            resolved = rc5_r51["current_sha256"]
        rc5_r52 = rc5_r52_successors.get(relative)
        if rc5_r52 is not None and rc5_r52["historical_sha256"] == resolved:
            resolved = rc5_r52["current_sha256"]
        rc5_r53 = rc5_r53_successors.get(relative)
        if rc5_r53 is not None and rc5_r53["historical_sha256"] == resolved:
            resolved = rc5_r53["current_sha256"]
        rc5_r55 = rc5_r55_successors.get(relative)
        if rc5_r55 is not None and rc5_r55["historical_sha256"] == resolved:
            resolved = rc5_r55["current_sha256"]
        rc5_r56 = rc5_r56_successors.get(relative)
        if rc5_r56 is not None and rc5_r56["historical_sha256"] == resolved:
            resolved = rc5_r56["current_sha256"]
        rc5_r57 = rc5_r57_successors.get(relative)
        if rc5_r57 is not None and rc5_r57["historical_sha256"] == resolved:
            resolved = rc5_r57["current_sha256"]
        rc5_r58 = rc5_r58_successors.get(relative)
        if rc5_r58 is not None and rc5_r58["historical_sha256"] == resolved:
            resolved = rc5_r58["current_sha256"]
        rc5_r59 = rc5_r59_successors.get(relative)
        if rc5_r59 is not None and rc5_r59["historical_sha256"] == resolved:
            resolved = rc5_r59["current_sha256"]
        rc5_r510 = rc5_r510_successors.get(relative)
        if rc5_r510 is not None and rc5_r510["historical_sha256"] == resolved:
            resolved = rc5_r510["current_sha256"]
        rc5_r511 = rc5_r511_successors.get(relative)
        if rc5_r511 is not None and rc5_r511["historical_sha256"] == resolved:
            resolved = rc5_r511["current_sha256"]
        rc5_r512 = rc5_r512_successors.get(relative)
        if rc5_r512 is not None and rc5_r512["historical_sha256"] == resolved:
            resolved = rc5_r512["current_sha256"]
        rc5_r513 = rc5_r513_successors.get(relative)
        if rc5_r513 is not None and rc5_r513["historical_sha256"] == resolved:
            resolved = rc5_r513["current_sha256"]
        rc5_g1 = rc5_g1_successors.get(relative)
        if rc5_g1 is not None and rc5_g1["historical_sha256"] == resolved:
            resolved = rc5_g1["current_sha256"]
        rc5_g1_glibc = rc5_g1_glibc_successors.get(relative)
        if (
            rc5_g1_glibc is not None
            and rc5_g1_glibc["historical_sha256"] == resolved
        ):
            resolved = rc5_g1_glibc["current_sha256"]
        rc6 = rc6_successors.get(relative)
        accepted = (
            []
            if rc6 is None
            else rc6.get(
                "accepted_predecessor_sha256",
                [rc6["historical_sha256"]],
            )
        )
        if rc6 is not None and resolved in accepted:
            resolved = rc6["current_sha256"]
        rc6 = rc6_r2_successors.get(relative)
        if rc6 is not None and rc6["historical_sha256"] == resolved:
            resolved = rc6["current_sha256"]
        rc6 = rc6_r3_successors.get(relative)
        if rc6 is not None and rc6["historical_sha256"] == resolved:
            resolved = rc6["current_sha256"]
        rc6 = rc6_r4_successors.get(relative)
        if rc6 is not None and rc6["historical_sha256"] == resolved:
            resolved = rc6["current_sha256"]
        rc6 = rc6_r5_successors.get(relative)
        if rc6 is not None and rc6["historical_sha256"] == resolved:
            resolved = rc6["current_sha256"]
            rc6 = rc6_r6_successors.get(relative)
            if rc6 is not None and rc6["historical_sha256"] == resolved:
                resolved = rc6["current_sha256"]
            n1 = n1_successors.get(relative)
            if n1 is not None and n1["historical_sha256"] == resolved:
                resolved = n1["current_sha256"]
            v17 = v17_successors.get(relative)
            if v17 is not None and v17["historical_sha256"] == resolved:
                resolved = v17["current_sha256"]
            assert resolve_s03_successor(relative, resolved) == hashlib.sha256(
                (PROJECT_ROOT / relative).read_bytes()
            ).hexdigest()
        assert registration["introduced_by_phase"] == (
            "v0.2.0rc4-release-preparation"
        )
        assert registration["runtime_changed"] is False
        assert registration["scientific_threshold_changed"] is False
        assert registration["historical_record_rewritten"] is False
        assert registration["reason"]


def test_release_candidate_documents_are_in_source_manifest():
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required = (
        "notes/pssolver_v0_2_scope.md",
        "notes/pssolver_v0_2_0_rc1.md",
        "notes/pssolver_v0_2_0_rc1.json",
        "notes/pssolver_v0_2_0_rc2.md",
        "notes/pssolver_v0_2_0_rc2.json",
        "notes/pssolver_v0_2_0_rc3.md",
        "notes/pssolver_v0_2_0_rc3.json",
        "notes/pssolver_v0_2_0_rc4.md",
        "notes/pssolver_v0_2_0_rc4.json",
        "notes/architecture_v0_2/phase_8_final_closure.json",
        "notes/architecture_v0_2/phase_9_p985_h100_closure.json",
        "notes/architecture_v0_2/phase_9_final_closure.json",
    )
    for relative in required:
        assert (PROJECT_ROOT / relative).is_file()
        assert f"include {relative}" in manifest
