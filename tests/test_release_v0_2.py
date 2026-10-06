"""Release-candidate contracts for PSSolver 0.2.0rc2."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver
from pssolver import capability_catalog
from pssolver.functional.api import functional_protocol_provenance


PROJECT_ROOT = Path(__file__).parents[1]


def test_release_candidate_version_and_public_metadata_agree():
    assert pssolver.__version__ == "0.2.0rc2"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.2.0rc2",
    }


def test_release_candidate_documents_and_machine_record_agree():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    scope = (PROJECT_ROOT / "notes" / "pssolver_v0_2_scope.md").read_text(
        encoding="utf-8"
    )
    release_notes = (
        PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc2.md"
    ).read_text(encoding="utf-8")
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc2.json").read_text(
            encoding="utf-8"
        )
    )

    assert "PSSolver 0.2.0rc2" in readme
    assert "## 0.2.0rc2" in changelog
    assert "PSSolver v0.2 support scope" in scope
    assert "PSSolver 0.2.0rc2 release candidate" in release_notes
    assert record["release"] == pssolver.__version__
    assert record["status"] == "local_release_gates_passed"
    assert record["local_verification"]["complete_cpu_suite"] == (
        "2733 passed, 6 deselected, 8 subtests passed"
    )
    assert record["local_verification"]["h100_release_smoke"] == "pending"
    assert record["artifacts"]["tag_created"] is False
    assert record["artifacts"]["pypi_published"] is False


def test_release_candidate_capability_counts_match_live_catalog():
    record = json.loads(
        (PROJECT_ROOT / "notes" / "pssolver_v0_2_0_rc2.json").read_text(
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


def test_release_candidate_documents_are_in_source_manifest():
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    required = (
        "notes/pssolver_v0_2_scope.md",
        "notes/pssolver_v0_2_0_rc1.md",
        "notes/pssolver_v0_2_0_rc1.json",
        "notes/pssolver_v0_2_0_rc2.md",
        "notes/pssolver_v0_2_0_rc2.json",
        "notes/architecture_v0_2/phase_8_final_closure.json",
        "notes/architecture_v0_2/phase_9_p985_h100_closure.json",
        "notes/architecture_v0_2/phase_9_final_closure.json",
    )
    for relative in required:
        assert (PROJECT_ROOT / relative).is_file()
        assert f"include {relative}" in manifest
