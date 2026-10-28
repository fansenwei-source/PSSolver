"""Local final source-freeze contract for PSSolver v0.3.0."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pssolver
from pssolver.functional.api import FUNCTIONAL_API_VERSION, functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_3_0.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_version_and_public_provenance_are_frozen() -> None:
    assert pssolver.__version__ == "0.3.0"
    assert FUNCTIONAL_API_VERSION == "1.1"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.3.0",
    }
    assert "## 0.3.0 — 2026-10-28" in (ROOT / "CHANGELOG.md").read_text(
        encoding="utf-8"
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PSSolver 0.3.0 is" in readme
    assert "qualified executable combinations in 0.3.0 are" in readme


def test_final_source_freeze_is_a_linear_successor_of_the_rc1_closeout() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_V0_3_0_LOCAL_FINAL_SOURCE_FREEZE_CANDIDATE"
    source = record["source"]
    assert source["source_parent"] == "72813d753dc9ca52e39240a643781c22605fd187"
    assert source["rc1_release_source"] == (
        "c031cdae646edc5b202b05238db1eb513b255805"
    )
    for commit in (source["source_parent"], source["rc1_release_source"]):
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=ROOT,
            check=True,
        )
    assert source["runtime_delta_from_rc1_release_source"] == [
        "pssolver/_version.py: 0.3.0rc1 -> 0.3.0"
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


def test_final_registry_snapshot_has_no_open_or_claimed_item() -> None:
    registry = _read(RECORD)["registry"]
    assert registry["issue_count"] == 51
    assert registry["status_counts_at_freeze"] == {
        "open": 0,
        "registered": 0,
        "claimed_fixed": 0,
        "verified": 47,
        "deferred": 2,
        "accepted": 2,
        "coverage_gap": 0,
    }
    assert registry["untriaged"] == []


def test_final_source_freeze_waits_for_r0040_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["completion_gates"]["G-09_final_candidate_acceptance"] == (
        "pending_R0040"
    )
    assert record["completion_gates"]["G-09_final_post_release_acceptance"] == (
        "not_started"
    )
    authorization = record["authorization"]
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
