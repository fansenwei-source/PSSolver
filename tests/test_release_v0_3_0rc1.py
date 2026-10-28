"""Local source-freeze contract for PSSolver v0.3.0rc1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import subprocess

import pssolver
from pssolver.functional.api import FUNCTIONAL_API_VERSION, functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_3_0rc1.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rc1_version_and_public_provenance_are_frozen() -> None:
    assert pssolver.__version__ == "0.3.0rc1"
    assert FUNCTIONAL_API_VERSION == "1.1"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.3.0rc1",
    }
    assert "## 0.3.0rc1 — 2026-10-28" in (ROOT / "CHANGELOG.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PSSolver 0.3.0rc1 is" in readme
    assert "qualified executable combinations in 0.3.0rc1 are" in readme


def test_rc1_source_freeze_is_a_linear_successor_of_dev03() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_V0_3_0RC1_LOCAL_SOURCE_FREEZE_CANDIDATE"
    )
    source = record["source"]
    assert source["development_tip"] == (
        "0e96ee5d05193b0ddae189f6171ba387a8a96b6b"
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source["development_tip"], "HEAD"],
        cwd=ROOT,
        check=True,
    )
    assert source["runtime_delta_from_development_tip"] == [
        "pssolver/_version.py: 0.3.0.dev0 -> 0.3.0rc1"
    ]
    for key in (
        "solver_equations_changed_by_freeze",
        "numerical_algorithms_changed_by_freeze",
        "checkpoint_schema_changed_by_freeze",
        "functional_api_version_changed_by_freeze",
        "provider_identity_changed_by_freeze",
        "production_default_changed_by_freeze",
        "threshold_changed_by_freeze",
        "skip_or_xfail_added_by_freeze",
    ):
        assert source[key] is False


def test_rc1_registry_has_no_untriaged_or_registered_finding() -> None:
    record = _read(RECORD)
    issues = _read(REGISTRY)["issues"]
    counts = Counter(item["status"] for item in issues)
    for status in ("open", "registered", "claimed_fixed", "coverage_gap"):
        counts.setdefault(status, 0)
    assert len(issues) == 51
    assert dict(counts) == record["registry"]["status_counts_at_freeze"]
    assert counts["open"] == counts["registered"] == counts["coverage_gap"] == 0
    assert {item["id"] for item in issues if item["status"] == "claimed_fixed"} == {
        "V-44"
    }
    v44 = next(item for item in issues if item["id"] == "V-44")
    assert v44["fixed_in"] == "4baa26c8c67d5e01c7615fa1dae29679f450645b"
    assert "R-0038" in v44["notes"]


def test_rc1_cumulative_gates_stop_at_local_artifacts_and_independent_acceptance() -> None:
    gates = _read(RECORD)["completion_gates"]
    assert gates["G-01"]["status"] == "candidate_pending_R-0038_only_for_V-44"
    for gate in ("G-02", "G-03", "G-04", "G-07", "G-08"):
        assert gates[gate]["status"] == "PASS"
    assert gates["G-06"]["status"] == "PASS_WITH_D-0013_BOUNDED_EXCEPTION"
    assert gates["G-05"]["status"] == "pending_clean_artifact_and_installed_wheel_gate"
    assert gates["G-09"]["status"] == "pending_R-0038_independent_candidate_acceptance"


def test_rc1_source_freeze_does_not_authorize_external_actions() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_source_freeze_authorized"] is True
    assert authorization["local_artifact_build_authorized"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_authorized",
        "hpcc_test_directory_authorized",
        "slurm_or_h100_authorized",
    ):
        assert authorization[key] is False
