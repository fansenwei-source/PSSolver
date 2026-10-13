"""Release-candidate contracts for PSSolver 0.2.0rc4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver import capability_catalog
from pssolver.functional.api import functional_protocol_provenance


PROJECT_ROOT = Path(__file__).parents[1]


def test_release_candidate_version_and_public_metadata_agree():
    assert pssolver.__version__ == "0.2.0rc4"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.2.0rc4",
    }


def test_release_candidate_documents_and_machine_record_agree():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
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

    assert "PSSolver 0.2.0rc4" in readme
    assert "## 0.2.0rc4" in changelog
    assert "PSSolver v0.2 support scope" in scope
    assert "PSSolver 0.2.0rc4 release candidate" in release_notes
    assert record["release"] == pssolver.__version__
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
        assert registration["current_sha256"] == hashlib.sha256(
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
