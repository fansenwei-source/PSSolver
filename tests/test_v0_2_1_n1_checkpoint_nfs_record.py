"""Evidence gate for the v0.2.1 N-1 local CPU candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_1.json"
V022_V17 = ROOT / "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"
DEV_INTEGRATION_RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _v17_successor(relative: str, historical: str) -> str:
    registration = json.loads(V022_V17.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(DEV_INTEGRATION_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    return historical


def test_n1_record_binds_the_implementation_commit_and_sources():
    record = _record()
    identity = record["git_identity"]
    assert identity["implementation_commit"] == (
        "927dc89aa63bcee9b03248f52662389bc947c47a"
    )
    assert identity["implementation_parent"] == (
        "e3d04f773723234b609bd51d5d0b6efb154bdd8b"
    )
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            identity["implementation_parent"],
            identity["implementation_commit"],
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert changed == [
        "pssolver/io/checkpoint.py",
        "tests/test_v0_2_1_n1_checkpoint_nfs.py",
    ]
    for relative, expected in record["source_sha256"].items():
        assert _sha256(relative) == _v17_successor(relative, expected)


def test_n1_record_is_a_forward_only_historical_successor():
    record = _record()
    release_successors = json.loads(
        RELEASE_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    parent = record["git_identity"]["implementation_parent"]
    for relative, successor in record["historical_successor_sources"].items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == successor[
            "historical_sha256"
        ]
        expected = successor["current_sha256"]
        registration = release_successors.get(relative)
        if registration is not None and registration["historical_sha256"] == expected:
            expected = registration["current_sha256"]
        assert _sha256(relative) == _v17_successor(relative, expected)
        assert successor["reason"]


def test_n1_record_preserves_schema_api_identity_and_defaults():
    record = _record()
    assert record["scope"] == {
        "solver_equations_changed": False,
        "numerical_algorithms_changed": False,
        "checkpoint_schema_changed": False,
        "functional_api_version_changed": False,
        "provider_identity_changed": False,
        "production_default_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
    }
    identity = record["provider_identity"]
    assert identity["identity_source_files_changed"] is False
    assert identity["provider_identity_changed"] is False
    assert {
        identity["periodic_canonical_sha256"],
        identity["channel_fixed_canonical_sha256"],
        identity["channel_tolerance_canonical_sha256"],
    } == {
        "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d",
        "b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905",
        "b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908",
    }


def test_n1_record_does_not_overclaim_real_nfs_or_external_authority():
    record = _record()
    assert record["classification"] == (
        "PASS_V0_2_1_N1_NFS_CHECKPOINT_PUBLICATION_LOCAL_CPU_CANDIDATE"
    )
    results = record["contract_results"]
    assert results["real_nfs_directory_fsync_characterized"] is False
    assert results["real_nfs_cross_client_concurrency_complete"] is False
    evidence = record["test_evidence"]
    assert evidence["installed_wheel"] == "pending"
    assert evidence["real_home_nfs"] == "pending"
    assert evidence["real_scratch1_nfs"] == "pending"
    assert evidence["cross_client_nfs"] == "pending"
    assert evidence["h100"] == "not_authorized"
    authorization = record["authorization"]
    assert authorization["implementation_complete"] is True
    assert authorization["eligible_for_real_nfs_cpu_qualification"] is True
    for key in (
        "push_authorized",
        "h100_authorized",
        "tag_authorized",
        "release_authorized",
        "automatic_retry",
    ):
        assert authorization[key] is False
